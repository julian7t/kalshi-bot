"""
strategy.py — Signal detection: find high-confidence opportunities.

A trade signal fires when ALL THREE conditions are met:
  1. Implied win probability >= MIN_CONFIDENCE (default 85%)
  2. Market is >= GAME_PROGRESS_THRESHOLD through its time window (default 65%)
  3. Market is still active and unsettled

Kalshi prices come back as dollar strings (e.g. "0.1500" = 15¢).

Price / confidence rules:
  - Ask-only market, yes_ask=6¢  → NO has 94% implied prob → BUY NO signal
  - Ask-only market, yes_ask=90¢ → YES has 90% implied prob → BUY YES signal
  - Both sides present: use mid-price

Order placement note:
  The `yes_price` field in the Kalshi API is ALWAYS the YES-side price (1–99):
    BUY YES @ 90¢ → yes_price=90, side="yes"
    BUY NO  @ 94¢ → yes_price=6,  side="no"  (entry cost = 100 - 6 = 94¢)
"""

import logging
from datetime import datetime, timezone

from config import (
    MIN_CONFIDENCE,
    GAME_PROGRESS_THRESHOLD,
    BET_FRACTION,
    MIN_CONTRACTS,
    MAX_CONTRACTS,
)

logger = logging.getLogger("kalshi_bot.strategy")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cents(val) -> float | None:
    """Convert Kalshi dollar string/float to cents (0–100 scale)."""
    if val is None:
        return None
    try:
        c = float(val) * 100.0
        return c if c >= 0 else None
    except (ValueError, TypeError):
        return None


def _parse_time(s: str) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def _game_progress(market: dict) -> float | None:
    """
    Fraction of the market's time window that has elapsed (0.0–1.0).
    Returns None if timestamps are missing or invalid.
    """
    open_t  = _parse_time(market.get("open_time"))
    close_t = _parse_time(market.get("close_time"))
    if not open_t or not close_t:
        return None
    total = (close_t - open_t).total_seconds()
    if total <= 0:
        return None
    elapsed = (datetime.now(timezone.utc) - open_t).total_seconds()
    return max(0.0, min(1.0, elapsed / total))


# ── Signal detection ──────────────────────────────────────────────────────────

def find_signal(market: dict) -> dict | None:
    """
    Analyse one market and return a signal dict, or None.

    Signal dict keys:
      ticker, title, side, yes_price, confidence, progress
    """
    ticker = market.get("ticker", "")
    title  = market.get("title", ticker)
    status = market.get("status", "")

    # Must be active and not yet settled
    if status not in ("active", "open"):
        return None
    if market.get("result"):
        return None

    # Timing gate — must be >= 65% through the event window
    progress = _game_progress(market)
    if progress is None or progress < GAME_PROGRESS_THRESHOLD:
        return None

    yes_bid_c = _cents(market.get("yes_bid_dollars"))
    yes_ask_c = _cents(market.get("yes_ask_dollars"))
    has_bid   = yes_bid_c is not None and yes_bid_c > 0
    has_ask   = yes_ask_c is not None and yes_ask_c > 0

    if not has_bid and not has_ask:
        return None

    threshold = MIN_CONFIDENCE * 100  # e.g. 85.0

    def sig(side, yes_price, confidence):
        return {
            "ticker":     ticker,
            "title":      title,
            "side":       side,
            "yes_price":  max(1, min(99, int(round(yes_price)))),
            "confidence": confidence,
            "progress":   progress,
        }

    # Both sides quoted — use mid-price
    if has_bid and has_ask:
        if yes_ask_c >= 99 or yes_bid_c >= 99:
            return None
        mid     = (yes_bid_c + yes_ask_c) / 2.0
        no_prob = 100.0 - mid
        if mid >= threshold:
            return sig("yes", yes_ask_c, mid / 100.0)
        if no_prob >= threshold:
            return sig("no", yes_bid_c, no_prob / 100.0)
        return None

    # Ask-only
    if has_ask and not has_bid:
        if yes_ask_c >= 99:
            return None
        no_prob = 100.0 - yes_ask_c
        if yes_ask_c >= threshold:
            return sig("yes", yes_ask_c, yes_ask_c / 100.0)
        if no_prob >= threshold:
            return sig("no", yes_ask_c, no_prob / 100.0)
        return None

    # Bid-only
    if has_bid and not has_ask:
        if yes_bid_c >= 99:
            return None
        no_prob = 100.0 - yes_bid_c
        if yes_bid_c >= threshold:
            return sig("yes", yes_bid_c, yes_bid_c / 100.0)
        if no_prob >= threshold:
            return sig("no", yes_bid_c, no_prob / 100.0)
        return None

    return None


def scan_markets(markets: list) -> list:
    """Return a confidence-sorted list of signals from a market batch."""
    signals = [s for m in markets if (s := find_signal(m))]
    signals.sort(key=lambda s: s["confidence"], reverse=True)
    return signals


def calc_contracts(
    portfolio_cents:    float,
    entry_price_cents:  float,
    size_multiplier:    float = 1.0,
) -> int:
    """
    Calculate contract count = (BET_FRACTION × portfolio) / entry_price,
    scaled by the portfolio-aware size_multiplier (0.25–1.0).

    The multiplier is applied BEFORE clamping so that the minimum of
    MIN_CONTRACTS is still respected.  Multiplier never exceeds 1.0 —
    it can only reduce size, never inflate it.
    """
    if entry_price_cents <= 0 or portfolio_cents <= 0:
        return MIN_CONTRACTS
    multiplier = max(0.01, min(1.0, size_multiplier))
    bet        = portfolio_cents * BET_FRACTION * multiplier
    contracts  = int(bet / entry_price_cents)
    return max(MIN_CONTRACTS, min(MAX_CONTRACTS, contracts))
