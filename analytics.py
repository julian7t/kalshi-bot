"""
analytics.py — Rich trade analytics layer.

Sits between the execution system and reporting.  Records every
filled trade with its full signal context (sport, regime, exec_mode,
model probabilities, edge estimates) and every settlement outcome.

Usage:
    # In bot.py, just before place_order_safe:
    analytics.register_entry_context(ticker, context_dict)

    # In order_manager._record_fill (after confirmed fill):
    analytics.record_fill_event(ticker, fill_price, contracts, partial)

    # In bot.py settlement check:
    analytics.record_settlement_event(ticker, exit_price, pnl_cents, outcome, reason)

Public API:
    register_entry_context(ticker, ctx)   — store pending context for a ticker
    record_fill_event(...)                — commit entry row to trade_analytics
    record_settlement_event(...)          — update row with exit/settlement data
    get_stats()                           — return dict of summary statistics
    log_summary()                         — log stats at INFO level
"""

import logging
import time
from datetime import datetime, timezone
from typing import Optional

import db

logger = logging.getLogger("kalshi_bot.analytics")

# ── In-memory pending context store ──────────────────────────────────────────
# Keyed by ticker. Holds signal + execution context for the trade that was
# just placed. Cleared after the fill event is recorded.
_pending: dict[str, dict] = {}


def register_entry_context(ticker: str, ctx: dict):
    """
    Store rich context for a trade that is about to be placed.

    Expected ctx keys (all optional — missing ones default gracefully):
      side, sport, matched_event_id, regime, exec_mode,
      confidence_score, fair_probability, market_probability,
      raw_edge, edge_after_slip, spread_at_entry, entry_price
    """
    _pending[ticker] = dict(ctx)
    logger.debug("[ANALYTICS] Registered entry context for %s: %s",
                 ticker, {k: v for k, v in ctx.items() if k not in ("market_data",)})


def record_fill_event(
    ticker:      str,
    fill_price:  float,
    contracts:   int,
    partial:     bool = False,
):
    """
    Called after a confirmed fill.  Combines fill data with stored context
    and writes one row to trade_analytics.
    Clears the pending context for this ticker.
    """
    ctx         = _pending.pop(ticker, {})
    entry_price = ctx.get("entry_price", int(fill_price))
    slippage    = fill_price - entry_price

    row_id = db.insert_trade_analytics(
        ticker             = ticker,
        side               = ctx.get("side", "?"),
        sport              = ctx.get("sport", "?"),
        matched_event_id   = ctx.get("matched_event_id", ""),
        regime             = ctx.get("regime", "unknown"),
        exec_mode          = ctx.get("exec_mode", "unset"),
        confidence_score   = float(ctx.get("confidence_score", 0) or 0),
        fair_probability   = float(ctx.get("fair_probability", 0) or 0),
        market_probability = float(ctx.get("market_probability", 0) or 0),
        raw_edge           = float(ctx.get("raw_edge", 0) or 0),
        edge_after_slip    = float(ctx.get("edge_after_slip", 0) or 0),
        spread_at_entry    = float(ctx.get("spread_at_entry", 0) or 0),
        entry_price        = entry_price,
        fill_price         = fill_price,
        slippage_cents     = slippage,
        contracts          = contracts,
        partial            = partial,
        market_type        = ctx.get("market_type", "misc"),
        model_name         = ctx.get("model_name", ""),
        parsed_line        = ctx.get("parsed_line"),
        signal_reason      = ctx.get("signal_reason", ""),
        # v4 portfolio fields
        overlap_level                  = ctx.get("overlap_level", "none"),
        concentration_score            = float(ctx.get("concentration_score", 0) or 0),
        allocation_rank                = float(ctx.get("allocation_rank", 0) or 0),
        portfolio_event_exposure_cents = float(ctx.get("portfolio_event_exposure_cents", 0) or 0),
        portfolio_sport_exposure_cents = float(ctx.get("portfolio_sport_exposure_cents", 0) or 0),
        # v5 timing fields
        entry_timing_classification    = ctx.get("entry_timing_classification", "aligned"),
        urgency_score                  = float(ctx.get("urgency_score", 0) or 0),
        staged_entry_flag              = bool(ctx.get("staged_entry_flag", False)),
        is_add_entry                   = bool(ctx.get("is_add_entry", False)),
        missed_edge_cents              = float(ctx.get("missed_edge_cents", 0) or 0),
    )

    logger.info(
        "[ANALYTICS] Trade entry #%d: %s %s %d@%.1f¢ "
        "sport=%s mtype=%s model=%s regime=%s mode=%s fair=%.0f%% edge=%+.0f%%",
        row_id, ctx.get("side","?").upper(), ticker, contracts, fill_price,
        ctx.get("sport","?"), ctx.get("market_type", "misc"),
        ctx.get("model_name", "—"),
        ctx.get("regime","?"), ctx.get("exec_mode","?"),
        float(ctx.get("fair_probability", 0) or 0) * 100,
        float(ctx.get("raw_edge", 0) or 0) * 100,
    )


def record_settlement_event(
    ticker:      str,
    side:        str,
    exit_price:  float,
    pnl_cents:   float,
    outcome:     str,    # "win" | "loss" | "push"
    exit_reason: str = "settlement",
):
    """
    Called when a position closes (market resolution or manual exit).
    Updates the most recent open trade_analytics row for this ticker.
    """
    db.update_trade_analytics_exit(
        ticker=ticker,
        exit_price=exit_price,
        exit_reason=exit_reason,
        pnl_cents=pnl_cents,
        outcome=outcome,
    )
    logger.info(
        "[ANALYTICS] Trade exit: %s %s pnl=$%.2f outcome=%s reason=%s",
        ticker, side, pnl_cents / 100, outcome, exit_reason,
    )


# ── Summary statistics ────────────────────────────────────────────────────────

def get_stats() -> dict:
    """
    Compute full analytics summary from the trade_analytics table.
    Returns a dict suitable for JSON serialisation.
    """
    all_trades    = db.get_all_trade_analytics()
    closed_trades = [t for t in all_trades if t.get("outcome") != "open"]
    open_trades   = [t for t in all_trades if t.get("outcome") == "open"]

    wins   = [t for t in closed_trades if t.get("outcome") == "win"]
    losses = [t for t in closed_trades if t.get("outcome") == "loss"]

    def _avg(vals: list) -> float:
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    def _median(vals: list) -> float:
        if not vals:
            return 0.0
        s = sorted(vals)
        n = len(s)
        return round((s[n // 2] + s[~(n // 2)]) / 2, 4)

    def _safe(t, key, default=0.0):
        v = t.get(key)
        try:
            return float(v) if v is not None else default
        except (TypeError, ValueError):
            return default

    # Core metrics
    pnls        = [_safe(t, "pnl_cents")      for t in closed_trades]
    hold_times  = [_safe(t, "hold_seconds")   for t in closed_trades]
    slippages   = [_safe(t, "slippage_cents") for t in all_trades if t.get("fill_price")]
    raw_edges   = [_safe(t, "raw_edge")       for t in all_trades if t.get("fill_price")]
    adj_edges   = [_safe(t, "edge_after_slip")for t in all_trades if t.get("fill_price")]

    total_settled = len(closed_trades)
    win_rate      = round(len(wins) / total_settled * 100, 1) if total_settled else 0.0

    # Breakdowns by dimension
    by_sport        = _breakdown(closed_trades, "sport",        pnl_key="pnl_cents")
    by_regime       = _breakdown(closed_trades, "regime",       pnl_key="pnl_cents")
    by_mode         = _breakdown(closed_trades, "exec_mode",    pnl_key="pnl_cents")
    by_market_type  = _breakdown(closed_trades, "market_type",  pnl_key="pnl_cents")
    by_model_name   = _breakdown_model_name(all_trades, closed_trades)
    by_hour         = _breakdown_by_hour(closed_trades)

    # Sport × market_type cross-breakdown
    by_sport_and_market_type = _breakdown_sport_market_type(closed_trades)

    # Portfolio-aware breakdowns
    by_overlap_level        = _breakdown(closed_trades, "overlap_level",   pnl_key="pnl_cents")
    by_concentration_bucket = _breakdown_concentration_bucket(closed_trades)
    by_event_exposure       = _breakdown_event_exposure(closed_trades)

    # Timing breakdowns
    by_timing_classification = _breakdown(closed_trades, "entry_timing_classification", pnl_key="pnl_cents")
    by_urgency_bucket        = _breakdown_urgency_bucket(closed_trades)
    by_staged_flag           = _breakdown_staged_flag(closed_trades)

    # Edge calibration bins (raw_edge in 5% buckets)
    edge_bins  = _edge_calibration_bins(closed_trades)

    # Confidence calibration bins (fair_probability in 10% buckets)
    conf_bins  = _probability_calibration_bins(closed_trades)

    return {
        "generated_at":               _now(),
        "total_entries":              len(all_trades),
        "total_filled":               len(all_trades),
        "open_positions":             len(open_trades),
        "total_settled":              total_settled,
        "wins":                       len(wins),
        "losses":                     len(losses),
        "win_rate_pct":               win_rate,
        "total_pnl_cents":            round(sum(pnls), 2),
        "total_pnl_dollars":          round(sum(pnls) / 100, 2),
        "avg_pnl_cents":              _avg(pnls),
        "median_pnl_cents":           _median(pnls),
        "avg_hold_seconds":           _avg(hold_times),
        "median_hold_seconds":        _median(hold_times),
        "avg_slippage_cents":         _avg(slippages),
        "avg_raw_edge_pct":           round(_avg(raw_edges) * 100, 2),
        "avg_adj_edge_pct":           round(_avg(adj_edges) * 100, 2),
        "pnl_by_sport":               by_sport,
        "pnl_by_regime":              by_regime,
        "pnl_by_exec_mode":           by_mode,
        "pnl_by_market_type":         by_market_type,
        "pnl_by_model_name":          by_model_name,
        "pnl_by_sport_and_market_type": by_sport_and_market_type,
        "pnl_by_hour":                by_hour,
        "pnl_by_overlap_level":           by_overlap_level,
        "pnl_by_concentration_bucket":    by_concentration_bucket,
        "pnl_by_event_exposure":          by_event_exposure,
        "pnl_by_timing_classification":   by_timing_classification,
        "pnl_by_urgency_bucket":          by_urgency_bucket,
        "pnl_by_staged_flag":             by_staged_flag,
        "edge_calibration_bins":          edge_bins,
        "prob_calibration_bins":          conf_bins,
    }


def log_summary():
    """Log a one-line analytics summary at INFO level."""
    try:
        s = get_stats()
        logger.info(
            "[ANALYTICS] fills=%d  settled=%d  win_rate=%.1f%%  "
            "pnl=$%.2f  avg_slip=%+.1f¢  avg_edge=%+.0f%%",
            s["total_filled"], s["total_settled"], s["win_rate_pct"],
            s["total_pnl_dollars"], s["avg_slippage_cents"],
            s["avg_raw_edge_pct"],
        )
    except Exception as e:
        logger.warning("[ANALYTICS] Could not compute summary: %s", e)


# ── Breakdown helpers ─────────────────────────────────────────────────────────

def _breakdown(trades: list[dict], group_key: str, pnl_key: str = "pnl_cents") -> dict:
    """Group trades by a string field and compute stats per group."""
    groups: dict[str, list] = {}
    for t in trades:
        k = str(t.get(group_key) or "unknown")
        groups.setdefault(k, []).append(t)

    result = {}
    for k, ts in groups.items():
        pnls  = [float(t.get(pnl_key) or 0) for t in ts]
        wins  = sum(1 for t in ts if t.get("outcome") == "win")
        total = len(ts)
        result[k] = {
            "count":        total,
            "wins":         wins,
            "win_rate_pct": round(wins / total * 100, 1) if total else 0.0,
            "total_pnl_cents": round(sum(pnls), 2),
            "avg_pnl_cents":   round(sum(pnls) / total, 2) if total else 0.0,
        }
    return result


def _breakdown_model_name(all_trades: list[dict], closed_trades: list[dict]) -> dict:
    """
    Per-model-name breakdown.

    Includes:
      count (fills), settled, wins, win_rate_pct, total_pnl_cents,
      avg_pnl_cents, avg_slippage_cents, avg_raw_edge_pct, fill_rate_pct.

    fill_rate = settled / filled (how often a signal becomes a real trade).
    """
    # Filled count per model (all trades with a fill_price)
    filled_by_model: dict[str, int] = {}
    for t in all_trades:
        if t.get("fill_price"):
            k = str(t.get("model_name") or "unknown")
            filled_by_model[k] = filled_by_model.get(k, 0) + 1

    groups: dict[str, list] = {}
    for t in closed_trades:
        k = str(t.get("model_name") or "unknown")
        groups.setdefault(k, []).append(t)

    result = {}
    for k, ts in groups.items():
        pnls  = [float(t.get("pnl_cents") or 0) for t in ts]
        slips = [float(t.get("slippage_cents") or 0) for t in ts if t.get("fill_price")]
        edges = [float(t.get("raw_edge") or 0) for t in ts if t.get("fill_price")]
        wins  = sum(1 for t in ts if t.get("outcome") == "win")
        total = len(ts)
        filled = filled_by_model.get(k, total)
        result[k] = {
            "fills":             filled,
            "settled":           total,
            "wins":              wins,
            "win_rate_pct":      round(wins / total * 100, 1) if total else 0.0,
            "fill_rate_pct":     round(total / filled * 100, 1) if filled else 0.0,
            "total_pnl_cents":   round(sum(pnls), 2),
            "avg_pnl_cents":     round(sum(pnls) / total, 2) if total else 0.0,
            "avg_slippage_cents": round(sum(slips) / len(slips), 2) if slips else 0.0,
            "avg_raw_edge_pct":  round(sum(edges) / len(edges) * 100, 2) if edges else 0.0,
        }
    return result


def _breakdown_sport_market_type(trades: list[dict]) -> dict:
    """
    Cross-breakdown by (sport, market_type).
    Returns {"Basketball:game_winner": {...stats...}, ...}
    """
    groups: dict[str, list] = {}
    for t in trades:
        sport = str(t.get("sport") or "unknown")
        mtype = str(t.get("market_type") or "misc")
        key   = f"{sport}:{mtype}"
        groups.setdefault(key, []).append(t)

    result = {}
    for key, ts in groups.items():
        pnls  = [float(t.get("pnl_cents") or 0) for t in ts]
        wins  = sum(1 for t in ts if t.get("outcome") == "win")
        total = len(ts)
        slips = [float(t.get("slippage_cents") or 0) for t in ts if t.get("fill_price")]
        result[key] = {
            "count":            total,
            "wins":             wins,
            "win_rate_pct":     round(wins / total * 100, 1) if total else 0.0,
            "total_pnl_cents":  round(sum(pnls), 2),
            "avg_pnl_cents":    round(sum(pnls) / total, 2) if total else 0.0,
            "avg_slippage":     round(sum(slips) / len(slips), 2) if slips else 0.0,
        }
    return result


def _breakdown_by_hour(trades: list[dict]) -> dict:
    """Group trades by hour of day (UTC) of entry."""
    groups: dict[str, list] = {}
    for t in trades:
        entry_at = t.get("entry_at", "")
        try:
            hour = datetime.fromisoformat(entry_at).hour
            k    = f"{hour:02d}:00"
        except Exception:
            k = "unknown"
        groups.setdefault(k, []).append(t)

    result = {}
    for k, ts in sorted(groups.items()):
        pnls  = [float(t.get("pnl_cents") or 0) for t in ts]
        wins  = sum(1 for t in ts if t.get("outcome") == "win")
        total = len(ts)
        result[k] = {
            "count":        total,
            "wins":         wins,
            "total_pnl_cents": round(sum(pnls), 2),
        }
    return result


def _edge_calibration_bins(trades: list[dict]) -> list[dict]:
    """
    Bin trades by raw_edge (5% buckets) and show realized win rate and PnL.
    Answers: does higher predicted edge → better outcomes?
    """
    bins: dict[str, list] = {}
    for t in trades:
        edge = float(t.get("raw_edge") or 0)
        b    = f"{int(edge * 20) * 5:+d}% to {int(edge * 20) * 5 + 5:+d}%"
        bins.setdefault(b, []).append(t)

    result = []
    for b, ts in sorted(bins.items()):
        pnls  = [float(t.get("pnl_cents") or 0) for t in ts]
        wins  = sum(1 for t in ts if t.get("outcome") == "win")
        total = len(ts)
        result.append({
            "edge_bin":         b,
            "count":            total,
            "win_rate_pct":     round(wins / total * 100, 1) if total else 0.0,
            "avg_pnl_cents":    round(sum(pnls) / total, 2) if total else 0.0,
        })
    return result


def _probability_calibration_bins(trades: list[dict]) -> list[dict]:
    """
    Bin trades by fair_probability (10% buckets) and show realized win rate.
    Answers: is the model's probability estimate calibrated?
    """
    bins: dict[int, list] = {}
    for t in trades:
        fair = float(t.get("fair_probability") or 0.5)
        b    = int(fair * 10) * 10   # 0, 10, 20, ..., 90
        bins.setdefault(b, []).append(t)

    result = []
    for b, ts in sorted(bins.items()):
        wins  = sum(1 for t in ts if t.get("outcome") == "win")
        total = len(ts)
        pnls  = [float(t.get("pnl_cents") or 0) for t in ts]
        result.append({
            "fair_prob_bin":    f"{b}%–{b+10}%",
            "count":            total,
            "win_rate_pct":     round(wins / total * 100, 1) if total else 0.0,
            "avg_pnl_cents":    round(sum(pnls) / total, 2) if total else 0.0,
            "calibration_err":  round(wins / total - (b + 5) / 100, 3) if total else 0.0,
        })
    return result


def _breakdown_urgency_bucket(trades: list[dict]) -> dict:
    """
    Break trades into urgency quartile buckets.
    Answers: does entering at high urgency produce better outcomes?
    """
    def _bucket(u: float) -> str:
        if u < 0.25: return "low (0-25%)"
        if u < 0.50: return "moderate (25-50%)"
        if u < 0.75: return "high (50-75%)"
        return "very-high (75-100%)"

    groups: dict[str, list] = {}
    for t in trades:
        u = float(t.get("urgency_score") or 0)
        groups.setdefault(_bucket(u), []).append(t)

    result = {}
    for b in ["low (0-25%)", "moderate (25-50%)", "high (50-75%)", "very-high (75-100%)"]:
        ts    = groups.get(b, [])
        pnls  = [float(t.get("pnl_cents") or 0) for t in ts]
        wins  = sum(1 for t in ts if t.get("outcome") == "win")
        total = len(ts)
        result[b] = {
            "count":           total,
            "wins":            wins,
            "win_rate_pct":    round(wins / total * 100, 1) if total else 0.0,
            "total_pnl_cents": round(sum(pnls), 2),
            "avg_pnl_cents":   round(sum(pnls) / total, 2) if total else 0.0,
        }
    return result


def _breakdown_staged_flag(trades: list[dict]) -> dict:
    """
    Compare staged entries (partial size) vs full-size entries.
    Answers: does staged entry improve or hurt PnL?
    """
    groups: dict[str, list] = {"staged": [], "full": [], "add": []}
    for t in trades:
        is_add    = bool(int(t.get("is_add_entry") or 0))
        is_staged = bool(int(t.get("staged_entry_flag") or 0))
        if is_add:
            groups["add"].append(t)
        elif is_staged:
            groups["staged"].append(t)
        else:
            groups["full"].append(t)

    result = {}
    for label, ts in groups.items():
        pnls  = [float(t.get("pnl_cents") or 0) for t in ts]
        wins  = sum(1 for t in ts if t.get("outcome") == "win")
        total = len(ts)
        result[label] = {
            "count":           total,
            "wins":            wins,
            "win_rate_pct":    round(wins / total * 100, 1) if total else 0.0,
            "total_pnl_cents": round(sum(pnls), 2),
            "avg_pnl_cents":   round(sum(pnls) / total, 2) if total else 0.0,
        }
    return result


def _breakdown_concentration_bucket(trades: list[dict]) -> dict:
    """
    Break trades into four concentration buckets based on concentration_score
    at the time of entry.  Answers: do more isolated trades outperform?
    """
    try:
        from portfolio import concentration_bucket
    except ImportError:
        def concentration_bucket(s):
            if s < 0.25: return "0-25%"
            if s < 0.50: return "25-50%"
            if s < 0.75: return "50-75%"
            return "75-100%"

    groups: dict[str, list] = {}
    for t in trades:
        score = float(t.get("concentration_score") or 0)
        b     = concentration_bucket(score)
        groups.setdefault(b, []).append(t)

    result = {}
    for b in ["0-25%", "25-50%", "50-75%", "75-100%"]:
        ts = groups.get(b, [])
        pnls = [float(t.get("pnl_cents") or 0) for t in ts]
        wins = sum(1 for t in ts if t.get("outcome") == "win")
        total = len(ts)
        result[b] = {
            "count":            total,
            "wins":             wins,
            "win_rate_pct":     round(wins / total * 100, 1) if total else 0.0,
            "total_pnl_cents":  round(sum(pnls), 2),
            "avg_pnl_cents":    round(sum(pnls) / total, 2) if total else 0.0,
        }
    return result


def _breakdown_event_exposure(trades: list[dict]) -> dict:
    """
    Break trades by event-exposure quartile at the time of entry.
    Answers: do trades entered when event exposure is already high perform worse?
    Quartiles are computed dynamically from the data.
    """
    import config as _cfg
    max_ev = getattr(_cfg, "PORTFOLIO_MAX_EXPOSURE_PER_EVENT", 2000) or 2000

    def _bucket(raw_cents: float) -> str:
        ratio = raw_cents / max_ev
        if ratio < 0.25:  return "low (<25%)"
        if ratio < 0.50:  return "moderate (25-50%)"
        if ratio < 0.75:  return "high (50-75%)"
        return "maxed (>75%)"

    groups: dict[str, list] = {}
    for t in trades:
        ev_exp = float(t.get("portfolio_event_exposure_cents") or 0)
        groups.setdefault(_bucket(ev_exp), []).append(t)

    result = {}
    for b in ["low (<25%)", "moderate (25-50%)", "high (50-75%)", "maxed (>75%)"]:
        ts    = groups.get(b, [])
        pnls  = [float(t.get("pnl_cents") or 0) for t in ts]
        wins  = sum(1 for t in ts if t.get("outcome") == "win")
        total = len(ts)
        result[b] = {
            "count":           total,
            "wins":            wins,
            "win_rate_pct":    round(wins / total * 100, 1) if total else 0.0,
            "total_pnl_cents": round(sum(pnls), 2),
            "avg_pnl_cents":   round(sum(pnls) / total, 2) if total else 0.0,
        }
    return result


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
