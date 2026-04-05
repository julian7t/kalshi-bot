"""
weather_paper.py — Paper tracking for weather strategy signals.

Stores every signal to a SQLite table (weather_signals) so we can:
  - review signal history
  - later add actual settlement outcomes for backtesting
  - compute running summaries

Schema:
    weather_signals
        id              INTEGER  PK AUTOINCREMENT
        recorded_at     TEXT     ISO-8601 UTC timestamp
        ticker          TEXT
        city            TEXT
        contract_type   TEXT     range | above | below
        lower           REAL     null for above/below
        upper           REAL     null for above/below
        threshold       REAL     null for range
        yes_bid         INTEGER
        yes_ask         INTEGER
        forecast_high   REAL
        model_prob      REAL
        market_prob     REAL
        raw_edge        REAL
        net_edge        REAL
        confidence      REAL
        reason          TEXT
        outcome_yes     INTEGER  null until settlement confirmed (1=yes,0=no)

Public API:
    record_weather_signal(signal, market)  → None
    summarize_weather_signals()            → dict
    mark_outcome(ticker, recorded_at, outcome_yes)  → None
"""

import logging
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

from config import DB_PATH

logger = logging.getLogger("kalshi_bot.weather_paper")

# ── Schema ────────────────────────────────────────────────────────────────────

_DDL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS weather_signals (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    recorded_at   TEXT    NOT NULL,
    ticker        TEXT    NOT NULL,
    city          TEXT,
    contract_type TEXT,
    lower         REAL,
    upper         REAL,
    threshold     REAL,
    yes_bid       INTEGER,
    yes_ask       INTEGER,
    forecast_high REAL,
    model_prob    REAL,
    market_prob   REAL,
    raw_edge      REAL,
    net_edge      REAL,
    confidence    REAL,
    reason        TEXT,
    outcome_yes   INTEGER  -- NULL until settled; 1=yes won, 0=no won
);

CREATE INDEX IF NOT EXISTS idx_wsig_ticker ON weather_signals(ticker);
CREATE INDEX IF NOT EXISTS idx_wsig_city   ON weather_signals(city);
"""


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH, timeout=10)
    c.row_factory = sqlite3.Row
    return c


def _ensure_table() -> None:
    """Create the weather_signals table if it doesn't exist yet."""
    try:
        with _conn() as c:
            c.executescript(_DDL)
    except Exception as exc:
        logger.error("[WEATHER PAPER] Table init failed: %s", exc)


# Ensure schema exists at import time
_ensure_table()


# ── Write ─────────────────────────────────────────────────────────────────────

def record_weather_signal(signal: dict, market: dict) -> None:
    """
    Persist a weather signal to the database.

    Parameters
    ----------
    signal : dict   Output of evaluate_weather_market()
    market : dict   Raw Kalshi market dict (for bid/ask/contract metadata)
    """
    try:
        # Contract shape from parser (re-parse to get thresholds)
        from weather_parser import parse_weather_market
        contract = parse_weather_market(market) or {}

        now = datetime.now(timezone.utc).isoformat()
        with _conn() as c:
            c.execute(
                """
                INSERT INTO weather_signals (
                    recorded_at, ticker, city, contract_type,
                    lower, upper, threshold,
                    yes_bid, yes_ask,
                    forecast_high, model_prob, market_prob,
                    raw_edge, net_edge, confidence, reason
                ) VALUES (
                    ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?,
                    ?, ?, ?,
                    ?, ?, ?, ?
                )
                """,
                (
                    now,
                    signal.get("ticker", ""),
                    signal.get("city", ""),
                    contract.get("type", ""),
                    contract.get("lower"),
                    contract.get("upper"),
                    contract.get("threshold"),
                    market.get("yes_bid"),
                    market.get("yes_ask"),
                    signal.get("forecast_high"),
                    signal.get("model_prob"),
                    signal.get("market_prob"),
                    signal.get("raw_edge"),
                    signal.get("net_edge"),
                    signal.get("confidence"),
                    signal.get("reason", ""),
                ),
            )
        logger.debug("[WEATHER PAPER] Recorded signal: %s", signal.get("ticker", ""))
    except Exception as exc:
        logger.error("[WEATHER PAPER] record_weather_signal failed: %s", exc)


def mark_outcome(ticker: str, recorded_at: str, outcome_yes: bool) -> None:
    """
    Set the settlement outcome for a specific signal row.
    Called externally once a market resolves.

    Parameters
    ----------
    ticker      : Kalshi ticker
    recorded_at : ISO-8601 timestamp string of the signal row
    outcome_yes : True if the YES contract won, False if NO won
    """
    try:
        with _conn() as c:
            c.execute(
                "UPDATE weather_signals SET outcome_yes=? "
                "WHERE ticker=? AND recorded_at=?",
                (1 if outcome_yes else 0, ticker, recorded_at),
            )
        logger.info("[WEATHER PAPER] Marked outcome %s for %s @ %s",
                    outcome_yes, ticker, recorded_at)
    except Exception as exc:
        logger.error("[WEATHER PAPER] mark_outcome failed: %s", exc)


# ── Read / summary ────────────────────────────────────────────────────────────

def load_weather_signal_records() -> list[dict]:
    """Return all stored weather signal rows as plain dicts."""
    try:
        with _conn() as c:
            rows = c.execute(
                "SELECT * FROM weather_signals ORDER BY recorded_at DESC"
            ).fetchall()
            return [dict(r) for r in rows]
    except Exception as exc:
        logger.error("[WEATHER PAPER] load_weather_signal_records failed: %s", exc)
        return []


def summarize_weather_signals() -> dict:
    """
    Return a high-level summary of all recorded weather signals.

    Returns
    -------
    dict with keys:
        total_signals     int
        avg_raw_edge      float
        avg_net_edge      float
        by_city           dict[str, int]
        by_contract_type  dict[str, int]
    """
    records = load_weather_signal_records()

    if not records:
        return {
            "total_signals":    0,
            "avg_raw_edge":     0.0,
            "avg_net_edge":     0.0,
            "by_city":          {},
            "by_contract_type": {},
        }

    raw_edges = [r["raw_edge"] for r in records if r.get("raw_edge") is not None]
    net_edges = [r["net_edge"] for r in records if r.get("net_edge") is not None]

    by_city: dict[str, int]  = defaultdict(int)
    by_type: dict[str, int]  = defaultdict(int)
    for r in records:
        by_city[r.get("city", "Unknown")] += 1
        by_type[r.get("contract_type", "?")] += 1

    return {
        "total_signals":    len(records),
        "avg_raw_edge":     round(sum(raw_edges) / len(raw_edges), 4) if raw_edges else 0.0,
        "avg_net_edge":     round(sum(net_edges) / len(net_edges), 4) if net_edges else 0.0,
        "by_city":          dict(by_city),
        "by_contract_type": dict(by_type),
    }
