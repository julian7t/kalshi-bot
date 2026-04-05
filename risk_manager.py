"""
risk_manager.py — Hard risk rules enforced before every trade.

All limits are configurable via environment variables (see config.py).
The check_trade_allowed() function is the single gate — if it returns
False, the order_manager must not place the order.

Rules:
  1. Minimum balance — don't trade on a near-empty account
  2. Max exposure per market — cap risk to any single market
  3. Max total exposure — cap overall capital at risk
  4. Max position size — cap contract count per position
  5. P&L kill switch — halt all trading after a drawdown threshold
"""

import logging
from config import (
    MIN_BALANCE_CENTS,
    MAX_EXPOSURE_PER_MARKET,
    MAX_TOTAL_EXPOSURE,
    MAX_POSITION_SIZE,
    PNL_KILL_SWITCH_CENTS,
)
import db

logger = logging.getLogger("kalshi_bot.risk")


class RiskViolation(Exception):
    """Raised when a hard risk rule blocks a trade."""


def check_trade_allowed(
    ticker: str,
    side: str,
    count: int,
    entry_price_cents: float,
    balance_cents: float,
) -> tuple[bool, str]:
    """
    Run all risk checks for a proposed trade.

    Returns (True, "") if allowed, (False, reason) if blocked.
    Callers should log the reason and skip the order.
    """
    trade_cost_cents = count * entry_price_cents

    # 1 — Minimum balance
    if balance_cents < MIN_BALANCE_CENTS:
        return False, (
            f"Balance ${balance_cents/100:.2f} is below minimum "
            f"${MIN_BALANCE_CENTS/100:.2f}"
        )

    # 2 — P&L kill switch
    session_pnl = db.total_pnl_cents()
    if session_pnl < PNL_KILL_SWITCH_CENTS:
        return False, (
            f"Kill switch triggered: session P&L ${session_pnl/100:.2f} "
            f"< threshold ${PNL_KILL_SWITCH_CENTS/100:.2f}"
        )

    # 3 — Max position size (contracts)
    if count > MAX_POSITION_SIZE:
        return False, (
            f"Position size {count} contracts exceeds max {MAX_POSITION_SIZE}"
        )

    # 4 — Max exposure per market
    existing = db.get_position(ticker)
    existing_cost = 0.0
    if existing:
        existing_cost = existing["contracts"] * existing["avg_entry_cents"]
    market_exposure = existing_cost + trade_cost_cents
    if market_exposure > MAX_EXPOSURE_PER_MARKET:
        return False, (
            f"Market exposure ${market_exposure/100:.2f} would exceed "
            f"limit ${MAX_EXPOSURE_PER_MARKET/100:.2f} for {ticker}"
        )

    # 5 — Max total exposure across all markets
    total = db.total_exposure_cents() + trade_cost_cents
    if total > MAX_TOTAL_EXPOSURE:
        return False, (
            f"Total exposure ${total/100:.2f} would exceed "
            f"limit ${MAX_TOTAL_EXPOSURE/100:.2f}"
        )

    return True, ""


def log_blocked(ticker: str, reason: str):
    logger.warning("[RISK BLOCKED] [%s] %s", ticker, reason)


def summarize() -> str:
    """Return a one-line risk summary for logging."""
    total_exp  = db.total_exposure_cents()
    session_pnl = db.total_pnl_cents()
    return (
        f"TotalExposure=${total_exp/100:.2f}  "
        f"SessionP&L=${session_pnl/100:.2f}  "
        f"KillSwitch@${PNL_KILL_SWITCH_CENTS/100:.2f}"
    )
