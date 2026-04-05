"""
weather_risk.py — Risk approval and position sizing for live weather trading.

Used only when weather_config.WEATHER_LIVE_ENABLED is True.
Reads directly from the DB so it always reflects current state —
no external state dict required from the caller.

Public API
----------
    approve_weather_trade(signal)         -> (bool, reason: str)
    compute_weather_position_size(signal, balance_dollars) -> int
"""

import logging
import sqlite3
from datetime import datetime, timezone

import weather_config as cfg
from config import DB_PATH

logger = logging.getLogger("kalshi_bot.weather_risk")


# ── DB helpers ────────────────────────────────────────────────────────────────

def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH, timeout=10)
    c.row_factory = sqlite3.Row
    return c


def _count_open_weather_positions() -> int:
    """
    Count weather orders currently in a pending or resting state.
    Uses the orders table (strategy='weather', status IN pending/resting).
    Falls back to 0 on any DB error.
    """
    try:
        with _conn() as c:
            row = c.execute(
                "SELECT COUNT(*) AS n FROM orders "
                "WHERE strategy='weather' AND status IN ('pending','resting')"
            ).fetchone()
            return int(row["n"]) if row else 0
    except Exception as exc:
        logger.warning("[WEATHER RISK] _count_open_weather_positions error: %s", exc)
        return 0


def _daily_weather_loss_cents() -> float:
    """
    Sum of negative pnl_cents from today's settled weather positions.
    Matches pnl_history rows whose ticker appears in the weather_signals table
    and whose closed_at is today (UTC).
    Returns 0.0 (not negative) — caller subtracts from limit.
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        with _conn() as c:
            row = c.execute(
                """
                SELECT COALESCE(SUM(p.pnl_cents), 0) AS total
                FROM   pnl_history p
                WHERE  p.closed_at LIKE ?
                AND    p.pnl_cents < 0
                AND    p.ticker IN (SELECT DISTINCT ticker FROM weather_signals)
                """,
                (f"{today}%",),
            ).fetchone()
            return abs(float(row["total"])) if row else 0.0
    except Exception as exc:
        logger.warning("[WEATHER RISK] _daily_weather_loss_cents error: %s", exc)
        return 0.0


def _current_weather_exposure_cents() -> float:
    """
    Estimate current dollar exposure from open weather orders.
    Approximation: count × WEATHER_MAX_CONTRACTS_PER_TRADE × entry_price,
    using the average yes_price of resting/pending weather orders.
    """
    try:
        with _conn() as c:
            row = c.execute(
                """
                SELECT COALESCE(SUM(count * yes_price), 0) AS exposure
                FROM   orders
                WHERE  strategy='weather' AND status IN ('pending','resting')
                """
            ).fetchone()
            return float(row["exposure"]) if row else 0.0
    except Exception as exc:
        logger.warning("[WEATHER RISK] _current_weather_exposure_cents error: %s", exc)
        return 0.0


# ── Approval ──────────────────────────────────────────────────────────────────

def approve_weather_trade(signal: dict) -> tuple[bool, str]:
    """
    Gate-check a weather signal before placing a live order.

    Checks (in order):
      1. Confidence >= WEATHER_MIN_CONF_FOR_LIVE
      2. Net edge   >= MIN_NET_EDGE
      3. Spread     <= MAX_SPREAD_CENTS
      4. Open weather positions < WEATHER_MAX_CONCURRENT_POSITIONS
      5. Daily weather loss < WEATHER_DAILY_LOSS_LIMIT_DOLLARS
      6. Current exposure < WEATHER_MAX_DOLLAR_EXPOSURE

    Returns
    -------
    (True,  "")           — approved
    (False, reason_str)   — blocked; reason is human-readable
    """
    ticker     = signal.get("ticker", "?")
    confidence = float(signal.get("confidence", 0.0))
    net_edge   = float(signal.get("net_edge",   0.0))
    spread     = float(signal.get("spread_cents", 99.0))

    # 1. Confidence gate (higher bar for live than for paper)
    if confidence < cfg.WEATHER_MIN_CONF_FOR_LIVE:
        reason = (f"conf {confidence:.2f} < live minimum {cfg.WEATHER_MIN_CONF_FOR_LIVE:.2f}")
        logger.info("[WEATHER RISK] BLOCKED %s — %s", ticker, reason)
        return False, reason

    # 2. Net edge gate
    if net_edge < cfg.MIN_NET_EDGE:
        reason = f"net_edge {net_edge:.3f} < {cfg.MIN_NET_EDGE:.3f}"
        logger.info("[WEATHER RISK] BLOCKED %s — %s", ticker, reason)
        return False, reason

    # 3. Spread gate
    if spread > cfg.MAX_SPREAD_CENTS:
        reason = f"spread {spread:.0f}¢ > max {cfg.MAX_SPREAD_CENTS}¢"
        logger.info("[WEATHER RISK] BLOCKED %s — %s", ticker, reason)
        return False, reason

    # 4. Concurrent positions cap
    open_count = _count_open_weather_positions()
    if open_count >= cfg.WEATHER_MAX_CONCURRENT_POSITIONS:
        reason = (f"open weather positions {open_count} >= "
                  f"max {cfg.WEATHER_MAX_CONCURRENT_POSITIONS}")
        logger.info("[WEATHER RISK] BLOCKED %s — %s", ticker, reason)
        return False, reason

    # 5. Daily loss limit
    daily_loss_cents = _daily_weather_loss_cents()
    daily_loss_limit = cfg.WEATHER_DAILY_LOSS_LIMIT_DOLLARS * 100.0
    if daily_loss_cents >= daily_loss_limit:
        reason = (f"daily weather loss ${daily_loss_cents/100:.2f} >= "
                  f"limit ${cfg.WEATHER_DAILY_LOSS_LIMIT_DOLLARS:.2f}")
        logger.info("[WEATHER RISK] BLOCKED %s — %s", ticker, reason)
        return False, reason

    # 6. Dollar exposure cap
    exposure_cents = _current_weather_exposure_cents()
    exposure_limit = cfg.WEATHER_MAX_DOLLAR_EXPOSURE * 100.0
    if exposure_cents >= exposure_limit:
        reason = (f"weather exposure ${exposure_cents/100:.2f} >= "
                  f"limit ${cfg.WEATHER_MAX_DOLLAR_EXPOSURE:.2f}")
        logger.info("[WEATHER RISK] BLOCKED %s — %s", ticker, reason)
        return False, reason

    logger.info(
        "[WEATHER RISK] APPROVED %s | conf=%.2f net=%.3f spread=%.0f¢ "
        "open=%d daily_loss=$%.2f exposure=$%.2f",
        ticker, confidence, net_edge, spread,
        open_count, daily_loss_cents / 100, exposure_cents / 100,
    )
    return True, ""


# ── Sizing ────────────────────────────────────────────────────────────────────

def compute_weather_position_size(signal: dict, balance_dollars: float) -> int:
    """
    Return the number of contracts to trade for a weather signal.

    Sizing logic (conservative):
      - Always starts at 1 contract (the hard minimum).
      - May scale to WEATHER_MAX_CONTRACTS_PER_TRADE only when:
          confidence >= WEATHER_MIN_CONF_FOR_LIVE + 0.10
          AND net_edge >= MIN_NET_EDGE * 2
      - Never exceeds WEATHER_MAX_CONTRACTS_PER_TRADE regardless.
      - Never exceeds what the balance can support.

    Parameters
    ----------
    signal          : signal dict from evaluate_weather_market
    balance_dollars : current account balance in dollars
    """
    max_contracts = cfg.WEATHER_MAX_CONTRACTS_PER_TRADE
    if max_contracts <= 1:
        return 1

    confidence = float(signal.get("confidence", 0.0))
    net_edge   = float(signal.get("net_edge",   0.0))
    yes_price  = int(signal.get("yes_price", 50))

    high_conf = confidence >= cfg.WEATHER_MIN_CONF_FOR_LIVE + 0.10
    high_edge = net_edge   >= cfg.MIN_NET_EDGE * 2.0

    if high_conf and high_edge:
        candidate = max_contracts
    else:
        candidate = 1

    # Balance safety: never risk more than 5% of balance on weather
    max_by_balance = max(1, int((balance_dollars * 0.05 * 100) / max(yes_price, 1)))
    size = min(candidate, max_by_balance, max_contracts)

    logger.debug(
        "[WEATHER RISK] Sizing %s → %d contract(s) "
        "(conf=%.2f edge=%.3f balance=$%.2f yes_price=%d¢)",
        signal.get("ticker", "?"), size,
        confidence, net_edge, balance_dollars, yes_price,
    )
    return max(1, size)
