"""
db.py — SQLite persistent state store.

Every order, position, and P&L event is written here so the bot
survives crashes and restarts without losing track of what it owns.

All writes happen immediately (no batching) to keep the on-disk state
as fresh as possible. SQLite's WAL mode ensures concurrent reads are safe.
"""

import sqlite3
import os
from datetime import datetime, timezone
from config import DB_PATH


# ── Schema ────────────────────────────────────────────────────────────────────

_SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS orders (
    client_order_id   TEXT PRIMARY KEY,
    kalshi_order_id   TEXT,
    ticker            TEXT    NOT NULL,
    side              TEXT    NOT NULL,
    count             INTEGER NOT NULL,
    yes_price         INTEGER NOT NULL,
    strategy          TEXT,
    -- status: pending | resting | executed | canceled | failed
    status            TEXT    DEFAULT 'pending',
    filled_count      INTEGER DEFAULT 0,
    avg_fill_price    REAL,
    placed_at         TEXT,
    updated_at        TEXT
);

CREATE INDEX IF NOT EXISTS idx_orders_ticker  ON orders(ticker);
CREATE INDEX IF NOT EXISTS idx_orders_status  ON orders(status);
CREATE INDEX IF NOT EXISTS idx_orders_kalshi  ON orders(kalshi_order_id);

CREATE TABLE IF NOT EXISTS positions (
    ticker         TEXT PRIMARY KEY,
    side           TEXT,
    contracts      INTEGER DEFAULT 0,
    avg_entry_cents REAL,
    opened_at      TEXT,
    updated_at     TEXT
);

CREATE TABLE IF NOT EXISTS pnl_history (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker       TEXT,
    side         TEXT,
    contracts    INTEGER,
    entry_cents  REAL,
    exit_cents   REAL,
    pnl_cents    REAL,
    closed_at    TEXT
);

CREATE TABLE IF NOT EXISTS reconciliation_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at      TEXT,
    orders_synced INTEGER,
    positions_synced INTEGER,
    notes       TEXT
);

CREATE TABLE IF NOT EXISTS trade_analytics (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker             TEXT    NOT NULL,
    side               TEXT    NOT NULL,
    sport              TEXT,
    matched_event_id   TEXT,
    regime             TEXT,
    exec_mode          TEXT,
    confidence_score   REAL,
    fair_probability   REAL,
    market_probability REAL,
    raw_edge           REAL,
    edge_after_slip    REAL,
    spread_at_entry    REAL,
    entry_price        INTEGER,
    fill_price         REAL,
    slippage_cents     REAL,
    contracts          INTEGER,
    partial            INTEGER DEFAULT 0,
    market_type        TEXT,
    entry_at           TEXT,
    exit_price         REAL,
    exit_at            TEXT,
    exit_reason        TEXT,
    hold_seconds       REAL,
    pnl_cents          REAL,
    outcome            TEXT    DEFAULT 'open'
);

CREATE INDEX IF NOT EXISTS idx_ta_ticker  ON trade_analytics(ticker);
CREATE INDEX IF NOT EXISTS idx_ta_sport   ON trade_analytics(sport);
CREATE INDEX IF NOT EXISTS idx_ta_outcome ON trade_analytics(outcome);
"""


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    """Create tables if they don't exist and run any pending migrations."""
    with _conn() as c:
        c.executescript(_SCHEMA)
    _migrate_db()


def _migrate_db():
    """
    Apply incremental schema migrations to existing databases.
    Safe to call repeatedly — each ALTER TABLE is guarded by a column existence
    check, so it is a no-op when the column already exists.
    """
    with _conn() as c:
        existing_cols = {
            row[1]
            for row in c.execute("PRAGMA table_info(trade_analytics)").fetchall()
        }
        existing_idx = {
            row[0]
            for row in c.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }

        # v2: market_type column + index
        if "market_type" not in existing_cols:
            try:
                c.execute("ALTER TABLE trade_analytics ADD COLUMN market_type TEXT")
            except Exception:
                pass  # concurrent startup race — column already added

        if "idx_ta_market_type" not in existing_idx:
            try:
                c.execute(
                    "CREATE INDEX IF NOT EXISTS idx_ta_market_type "
                    "ON trade_analytics(market_type)"
                )
            except Exception:
                pass

        # v3: model_name, parsed_line, signal_reason columns
        for col_def in [
            ("model_name",   "ALTER TABLE trade_analytics ADD COLUMN model_name TEXT"),
            ("parsed_line",  "ALTER TABLE trade_analytics ADD COLUMN parsed_line REAL"),
            ("signal_reason","ALTER TABLE trade_analytics ADD COLUMN signal_reason TEXT"),
        ]:
            col, stmt = col_def
            if col not in existing_cols:
                try:
                    c.execute(stmt)
                except Exception:
                    pass  # already added

        if "idx_ta_model_name" not in existing_idx:
            try:
                c.execute(
                    "CREATE INDEX IF NOT EXISTS idx_ta_model_name "
                    "ON trade_analytics(model_name)"
                )
            except Exception:
                pass

        # v4: portfolio construction columns
        for col, stmt in [
            ("overlap_level",
             "ALTER TABLE trade_analytics ADD COLUMN overlap_level TEXT"),
            ("concentration_score",
             "ALTER TABLE trade_analytics ADD COLUMN concentration_score REAL"),
            ("allocation_rank",
             "ALTER TABLE trade_analytics ADD COLUMN allocation_rank REAL"),
            ("portfolio_event_exposure_cents",
             "ALTER TABLE trade_analytics ADD COLUMN portfolio_event_exposure_cents REAL"),
            ("portfolio_sport_exposure_cents",
             "ALTER TABLE trade_analytics ADD COLUMN portfolio_sport_exposure_cents REAL"),
        ]:
            if col not in existing_cols:
                try:
                    c.execute(stmt)
                except Exception:
                    pass  # concurrent startup race

        if "idx_ta_overlap" not in existing_idx:
            try:
                c.execute(
                    "CREATE INDEX IF NOT EXISTS idx_ta_overlap "
                    "ON trade_analytics(overlap_level)"
                )
            except Exception:
                pass

        # v5: entry timing layer columns
        for col, stmt in [
            ("entry_timing_classification",
             "ALTER TABLE trade_analytics ADD COLUMN entry_timing_classification TEXT"),
            ("urgency_score",
             "ALTER TABLE trade_analytics ADD COLUMN urgency_score REAL"),
            ("staged_entry_flag",
             "ALTER TABLE trade_analytics ADD COLUMN staged_entry_flag INTEGER DEFAULT 0"),
            ("is_add_entry",
             "ALTER TABLE trade_analytics ADD COLUMN is_add_entry INTEGER DEFAULT 0"),
            ("missed_edge_cents",
             "ALTER TABLE trade_analytics ADD COLUMN missed_edge_cents REAL"),
        ]:
            if col not in existing_cols:
                try:
                    c.execute(stmt)
                except Exception:
                    pass  # concurrent startup race

        if "idx_ta_timing" not in existing_idx:
            try:
                c.execute(
                    "CREATE INDEX IF NOT EXISTS idx_ta_timing "
                    "ON trade_analytics(entry_timing_classification)"
                )
            except Exception:
                pass


# ── Orders ────────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_order(client_order_id: str, ticker: str, side: str, count: int,
               yes_price: int, strategy: str, status: str = "pending",
               kalshi_order_id: str = None):
    """Insert or update an order record."""
    with _conn() as c:
        c.execute("""
            INSERT INTO orders
                (client_order_id, kalshi_order_id, ticker, side, count,
                 yes_price, strategy, status, placed_at, updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(client_order_id) DO UPDATE SET
                kalshi_order_id = excluded.kalshi_order_id,
                status          = excluded.status,
                updated_at      = excluded.updated_at
        """, (client_order_id, kalshi_order_id, ticker, side, count,
              yes_price, strategy, status, _now(), _now()))


def update_order(client_order_id: str, **fields):
    """Update arbitrary fields on an order. Automatically sets updated_at."""
    fields["updated_at"] = _now()
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [client_order_id]
    with _conn() as c:
        c.execute(f"UPDATE orders SET {set_clause} WHERE client_order_id = ?", values)


def get_order_by_client_id(client_order_id: str) -> dict | None:
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM orders WHERE client_order_id = ?", (client_order_id,)
        ).fetchone()
    return dict(row) if row else None


def get_orders_by_ticker(ticker: str) -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM orders WHERE ticker = ? ORDER BY placed_at DESC", (ticker,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_orders_by_status(status: str) -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM orders WHERE status = ? ORDER BY placed_at DESC", (status,)
        ).fetchall()
    return [dict(r) for r in rows]


def has_active_order_for_ticker(ticker: str) -> bool:
    """True if there is a pending or resting order for this ticker."""
    with _conn() as c:
        row = c.execute(
            "SELECT 1 FROM orders WHERE ticker = ? AND status IN ('pending','resting') LIMIT 1",
            (ticker,)
        ).fetchone()
    return row is not None


def has_executed_order_for_ticker(ticker: str) -> bool:
    """True if we have an executed (filled) order for this ticker."""
    with _conn() as c:
        row = c.execute(
            "SELECT 1 FROM orders WHERE ticker = ? AND status = 'executed' LIMIT 1",
            (ticker,)
        ).fetchone()
    return row is not None


def next_sequence_for_ticker_today(ticker: str) -> int:
    """Return the next sequence number for a ticker on today's date (for client_order_id)."""
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    with _conn() as c:
        row = c.execute(
            "SELECT COUNT(*) as n FROM orders WHERE ticker = ? AND placed_at LIKE ?",
            (ticker, f"{today}%")
        ).fetchone()
    return (row["n"] if row else 0) + 1


# ── Positions ─────────────────────────────────────────────────────────────────

def upsert_position(ticker: str, side: str, contracts: int, avg_entry_cents: float):
    with _conn() as c:
        c.execute("""
            INSERT INTO positions (ticker, side, contracts, avg_entry_cents, opened_at, updated_at)
            VALUES (?,?,?,?,?,?)
            ON CONFLICT(ticker) DO UPDATE SET
                side            = excluded.side,
                contracts       = excluded.contracts,
                avg_entry_cents = excluded.avg_entry_cents,
                updated_at      = excluded.updated_at
        """, (ticker, side, contracts, avg_entry_cents, _now(), _now()))


def remove_position(ticker: str):
    with _conn() as c:
        c.execute("DELETE FROM positions WHERE ticker = ?", (ticker,))


def get_all_positions() -> list[dict]:
    with _conn() as c:
        rows = c.execute("SELECT * FROM positions").fetchall()
    return [dict(r) for r in rows]


def get_position(ticker: str) -> dict | None:
    with _conn() as c:
        row = c.execute("SELECT * FROM positions WHERE ticker = ?", (ticker,)).fetchone()
    return dict(row) if row else None


def total_exposure_cents() -> float:
    """Sum of (contracts × avg_entry_cents) across all open positions."""
    with _conn() as c:
        row = c.execute(
            "SELECT SUM(contracts * avg_entry_cents) as total FROM positions"
        ).fetchone()
    return float(row["total"] or 0)


# ── P&L history ───────────────────────────────────────────────────────────────

def record_pnl(ticker: str, side: str, contracts: int,
               entry_cents: float, exit_cents: float, pnl_cents: float):
    with _conn() as c:
        c.execute("""
            INSERT INTO pnl_history
                (ticker, side, contracts, entry_cents, exit_cents, pnl_cents, closed_at)
            VALUES (?,?,?,?,?,?,?)
        """, (ticker, side, contracts, entry_cents, exit_cents, pnl_cents, _now()))


def total_pnl_cents() -> float:
    with _conn() as c:
        row = c.execute("SELECT SUM(pnl_cents) as total FROM pnl_history").fetchone()
    return float(row["total"] or 0)


def get_pnl_history(limit: int = 50) -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM pnl_history ORDER BY closed_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


# ── Trade analytics ───────────────────────────────────────────────────────────

def insert_trade_analytics(
    ticker: str, side: str, sport: str, matched_event_id: str,
    regime: str, exec_mode: str, confidence_score: float,
    fair_probability: float, market_probability: float,
    raw_edge: float, edge_after_slip: float, spread_at_entry: float,
    entry_price: int, fill_price: float, slippage_cents: float,
    contracts: int, partial: bool,
    market_type:  str   = "misc",
    model_name:   str   = "",
    parsed_line:  float = None,
    signal_reason: str  = "",
    # v4 portfolio fields
    overlap_level:                  str   = "none",
    concentration_score:            float = 0.0,
    allocation_rank:                float = 0.0,
    portfolio_event_exposure_cents: float = 0.0,
    portfolio_sport_exposure_cents: float = 0.0,
    # v5 timing fields
    entry_timing_classification:    str   = "aligned",
    urgency_score:                  float = 0.0,
    staged_entry_flag:              bool  = False,
    is_add_entry:                   bool  = False,
    missed_edge_cents:              float = 0.0,
) -> int:
    """Insert entry-side trade record. Returns the new row id."""
    with _conn() as c:
        cur = c.execute("""
            INSERT INTO trade_analytics
                (ticker, side, sport, matched_event_id, regime, exec_mode,
                 confidence_score, fair_probability, market_probability,
                 raw_edge, edge_after_slip, spread_at_entry, entry_price,
                 fill_price, slippage_cents, contracts, partial, market_type,
                 model_name, parsed_line, signal_reason,
                 overlap_level, concentration_score, allocation_rank,
                 portfolio_event_exposure_cents, portfolio_sport_exposure_cents,
                 entry_timing_classification, urgency_score,
                 staged_entry_flag, is_add_entry, missed_edge_cents,
                 entry_at, outcome)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (ticker, side, sport, matched_event_id, regime, exec_mode,
              confidence_score, fair_probability, market_probability,
              raw_edge, edge_after_slip, spread_at_entry, entry_price,
              fill_price, slippage_cents, contracts, int(partial), market_type,
              model_name or "", parsed_line, signal_reason or "",
              overlap_level or "none", concentration_score, allocation_rank,
              portfolio_event_exposure_cents, portfolio_sport_exposure_cents,
              entry_timing_classification or "aligned", urgency_score,
              int(staged_entry_flag), int(is_add_entry), missed_edge_cents,
              _now(), "open"))
        return cur.lastrowid


def update_trade_analytics_exit(
    ticker: str,
    exit_price: float,
    exit_reason: str,
    pnl_cents: float,
    outcome: str,          # "win" | "loss" | "push"
):
    """Update the most recent open trade for ticker with exit data."""
    now = _now()
    with _conn() as c:
        # Find the most recent open trade for this ticker
        row = c.execute("""
            SELECT id, entry_at FROM trade_analytics
            WHERE ticker = ? AND outcome = 'open'
            ORDER BY entry_at DESC LIMIT 1
        """, (ticker,)).fetchone()
        if row is None:
            return
        row_id   = row["id"]
        entry_ts = row["entry_at"]
        try:
            from datetime import datetime, timezone
            entry_dt  = datetime.fromisoformat(entry_ts)
            exit_dt   = datetime.fromisoformat(now)
            hold_secs = (exit_dt - entry_dt).total_seconds()
        except Exception:
            hold_secs = 0.0

        c.execute("""
            UPDATE trade_analytics
            SET exit_price=?, exit_at=?, exit_reason=?,
                hold_seconds=?, pnl_cents=?, outcome=?
            WHERE id=?
        """, (exit_price, now, exit_reason, hold_secs, pnl_cents, outcome, row_id))


def get_all_trade_analytics() -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM trade_analytics ORDER BY entry_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_closed_trade_analytics() -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM trade_analytics WHERE outcome != 'open' ORDER BY entry_at"
        ).fetchall()
    return [dict(r) for r in rows]


def get_open_trade_analytics() -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM trade_analytics WHERE outcome = 'open' ORDER BY entry_at"
        ).fetchall()
    return [dict(r) for r in rows]


# ── Reconciliation log ────────────────────────────────────────────────────────

def log_reconciliation(orders_synced: int, positions_synced: int, notes: str = ""):
    with _conn() as c:
        c.execute("""
            INSERT INTO reconciliation_log (run_at, orders_synced, positions_synced, notes)
            VALUES (?,?,?,?)
        """, (_now(), orders_synced, positions_synced, notes))
