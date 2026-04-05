"""
metrics.py — Performance metrics tracking.

Writes two outputs:
  metrics/trades.csv    — one row per fill or settlement event (append-only)
  metrics/summary.json  — aggregate stats, rewritten after every event

CSV columns:
  event_time, event_type, ticker, side, strategy,
  yes_price, filled_price, slippage_cents, contracts,
  pnl_cents, hold_time_seconds, outcome

event_type values:
  FILL         — order fully filled
  PARTIAL_FILL — order partially filled (remaining canceled)
  SETTLEMENT   — position settled (market resolved yes/no)

Slippage = filled_price − yes_price  (positive = paid more than quoted)

Hold time = seconds from order placed_at to settlement closed_at
"""

import csv
import json
import logging
import os
from collections import defaultdict
from datetime import datetime, timezone

logger = logging.getLogger("kalshi_bot.metrics")

# ── Paths ─────────────────────────────────────────────────────────────────────

_BASE    = os.path.join(os.path.dirname(__file__), "metrics")
CSV_PATH = os.path.join(_BASE, "trades.csv")
JSON_PATH = os.path.join(_BASE, "summary.json")

_CSV_HEADERS = [
    "event_time", "event_type", "ticker", "side", "strategy",
    "yes_price", "filled_price", "slippage_cents",
    "contracts", "pnl_cents", "hold_time_seconds", "outcome",
]

# In-memory event log (also persisted to CSV)
_events: list[dict] = []


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_dir():
    os.makedirs(_BASE, exist_ok=True)


def _ensure_csv():
    _ensure_dir()
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=_CSV_HEADERS).writeheader()


def _append_csv(row: dict):
    _ensure_csv()
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_HEADERS, extrasaction="ignore")
        writer.writerow(row)


# ── Public recording API ──────────────────────────────────────────────────────

def record_fill(
    ticker: str,
    side: str,
    strategy: str,
    yes_price: int,
    filled_price: float | None,
    contracts: int,
    partial: bool = False,
):
    """
    Record a fill event (full or partial).

    filled_price — avg_yes_price from Kalshi order response.
    slippage     — filled_price − yes_price (positive = worse than expected).
    """
    if filled_price is None:
        filled_price = float(yes_price)

    slippage = round(filled_price - yes_price, 4)
    event_type = "PARTIAL_FILL" if partial else "FILL"

    row = {
        "event_time":      _now(),
        "event_type":      event_type,
        "ticker":          ticker,
        "side":            side,
        "strategy":        strategy,
        "yes_price":       yes_price,
        "filled_price":    round(filled_price, 4),
        "slippage_cents":  slippage,
        "contracts":       contracts,
        "pnl_cents":       "",
        "hold_time_seconds": "",
        "outcome":         "",
    }

    _events.append(row)
    _append_csv(row)
    _write_summary()

    logger.info(
        "[METRICS] %s [%s] side=%s contracts=%d filled@%.1f¢ slippage=%+.1f¢",
        event_type, ticker, side, contracts, filled_price, slippage
    )


def record_settlement(
    ticker: str,
    side: str,
    strategy: str,
    yes_price: int,
    contracts: int,
    pnl_cents: float,
    hold_time_seconds: float,
    outcome: str,           # "win" | "loss" | "push"
):
    """
    Record a settlement event.

    outcome       — 'win' if the side we held was the winner, 'loss' otherwise.
    hold_time     — seconds from order placement to settlement.
    pnl_cents     — realised P&L in cents (+ve = profit).
    """
    row = {
        "event_time":        _now(),
        "event_type":        "SETTLEMENT",
        "ticker":            ticker,
        "side":              side,
        "strategy":          strategy,
        "yes_price":         yes_price,
        "filled_price":      "",
        "slippage_cents":    "",
        "contracts":         contracts,
        "pnl_cents":         round(pnl_cents, 2),
        "hold_time_seconds": round(hold_time_seconds, 1),
        "outcome":           outcome,
    }

    _events.append(row)
    _append_csv(row)
    _write_summary()

    logger.info(
        "[METRICS] SETTLEMENT [%s] %s  pnl=$%.2f  hold=%.0fs  outcome=%s",
        ticker, side, pnl_cents / 100, hold_time_seconds, outcome
    )


# ── Summary ───────────────────────────────────────────────────────────────────

def _write_summary():
    """
    Recompute aggregate metrics from the CSV and write summary.json.
    Reads from disk to catch events from previous sessions.
    """
    try:
        all_events = _load_all_events()
        summary = _compute_summary(all_events)
        _ensure_dir()
        with open(JSON_PATH, "w") as f:
            json.dump(summary, f, indent=2)
    except Exception as e:
        logger.warning("[METRICS] Could not write summary: %s", e)


def _load_all_events() -> list[dict]:
    if not os.path.exists(CSV_PATH):
        return []
    with open(CSV_PATH, "r", newline="") as f:
        return list(csv.DictReader(f))


def _compute_summary(events: list[dict]) -> dict:
    fills       = [e for e in events if e["event_type"] in ("FILL", "PARTIAL_FILL")]
    settlements = [e for e in events if e["event_type"] == "SETTLEMENT"]

    wins   = [s for s in settlements if s["outcome"] == "win"]
    losses = [s for s in settlements if s["outcome"] == "loss"]

    def avg(vals):
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    slippages    = [float(f["slippage_cents"]) for f in fills if f["slippage_cents"] != ""]
    pnls         = [float(s["pnl_cents"])      for s in settlements if s["pnl_cents"] != ""]
    hold_times   = [float(s["hold_time_seconds"]) for s in settlements if s["hold_time_seconds"] != ""]

    # Per-strategy breakdown
    by_strategy: dict = defaultdict(lambda: {"fills": 0, "wins": 0, "losses": 0, "total_pnl_cents": 0.0})
    for e in events:
        strat = e.get("strategy", "unknown") or "unknown"
        if e["event_type"] in ("FILL", "PARTIAL_FILL"):
            by_strategy[strat]["fills"] += 1
        if e["event_type"] == "SETTLEMENT":
            pnl = float(e["pnl_cents"]) if e["pnl_cents"] != "" else 0.0
            by_strategy[strat]["total_pnl_cents"] += pnl
            if e["outcome"] == "win":
                by_strategy[strat]["wins"] += 1
            elif e["outcome"] == "loss":
                by_strategy[strat]["losses"] += 1

    total_settled = len(wins) + len(losses)

    return {
        "generated_at":         datetime.now(timezone.utc).isoformat(),
        "total_fills":          len(fills),
        "total_settlements":    total_settled,
        "wins":                 len(wins),
        "losses":               len(losses),
        "win_rate_pct":         round(len(wins) / total_settled * 100, 1) if total_settled else 0.0,
        "total_pnl_cents":      round(sum(pnls), 2),
        "total_pnl_dollars":    round(sum(pnls) / 100, 2),
        "avg_pnl_per_trade_cents": avg(pnls),
        "avg_slippage_cents":   avg(slippages),
        "avg_hold_time_seconds": avg(hold_times),
        "max_hold_time_seconds": max(hold_times) if hold_times else 0,
        "min_hold_time_seconds": min(hold_times) if hold_times else 0,
        "by_strategy":          {k: dict(v) for k, v in by_strategy.items()},
    }


def print_summary():
    """Log a human-readable metrics summary."""
    try:
        events  = _load_all_events()
        summary = _compute_summary(events)
        logger.info(
            "[METRICS SUMMARY] fills=%d  settled=%d  win_rate=%.1f%%  "
            "total_pnl=$%.2f  avg_slippage=%+.1f¢  avg_hold=%.0fs",
            summary["total_fills"],
            summary["total_settlements"],
            summary["win_rate_pct"],
            summary["total_pnl_dollars"],
            summary["avg_slippage_cents"],
            summary["avg_hold_time_seconds"],
        )
    except Exception as e:
        logger.warning("[METRICS] Could not print summary: %s", e)
