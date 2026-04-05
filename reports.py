"""
reports.py — Analytics report generation.

Reads from trade_analytics (via analytics.py) and writes all output files.

Output files (written to metrics/ directory):
  analytics_summary.json   — comprehensive analytics summary
  trades.csv               — enriched trade log (one row per trade)
  pnl_by_sport.csv
  pnl_by_regime.csv
  execution_quality.csv    — fill rate, slippage, and edge by exec mode
  model_calibration.csv    — realized win rate vs predicted probability
  backtest_summary.json    — results of scenario backtests

All writes are idempotent and crash-safe (write to temp then rename).
"""

import csv
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from typing import Any

import analytics
import db
from backtest import BacktestEngine

logger = logging.getLogger("kalshi_bot.reports")

_BASE    = os.path.join(os.path.dirname(__file__), "metrics")
os.makedirs(_BASE, exist_ok=True)


# ── File paths ────────────────────────────────────────────────────────────────

ANALYTICS_JSON      = os.path.join(_BASE, "analytics_summary.json")
TRADES_CSV          = os.path.join(_BASE, "trades.csv")
PNL_SPORT_CSV       = os.path.join(_BASE, "pnl_by_sport.csv")
PNL_REGIME_CSV      = os.path.join(_BASE, "pnl_by_regime.csv")
EXEC_QUALITY_CSV    = os.path.join(_BASE, "execution_quality.csv")
CALIBRATION_CSV     = os.path.join(_BASE, "model_calibration.csv")
BACKTEST_JSON       = os.path.join(_BASE, "backtest_summary.json")
PNL_MODEL_CSV       = os.path.join(_BASE, "pnl_by_model_name.csv")
PNL_MODEL_JSON      = os.path.join(_BASE, "pnl_by_model_name.json")
PNL_OVERLAP_CSV     = os.path.join(_BASE, "pnl_by_overlap_level.csv")
PNL_CONCBKT_CSV     = os.path.join(_BASE, "pnl_by_concentration_bucket.csv")
PNL_EVEXP_CSV       = os.path.join(_BASE, "pnl_by_event_exposure.csv")
PNL_TIMING_CSV      = os.path.join(_BASE, "pnl_by_timing_classification.csv")
PNL_URGENCY_CSV     = os.path.join(_BASE, "pnl_by_urgency_bucket.csv")
PNL_STAGED_CSV      = os.path.join(_BASE, "pnl_by_staged_flag.csv")
TIMING_BACKTEST_JSON= os.path.join(_BASE, "timing_backtest_summary.json")


# ── Column definitions ────────────────────────────────────────────────────────

_TRADES_HEADERS = [
    "id", "ticker", "side", "sport", "matched_event_id",
    "market_type", "model_name", "parsed_line",
    "regime", "exec_mode", "confidence_score",
    "fair_probability", "market_probability", "raw_edge", "edge_after_slip",
    "spread_at_entry", "entry_price", "fill_price", "slippage_cents",
    "contracts", "partial",
    "entry_at", "exit_at", "exit_reason", "hold_seconds",
    "pnl_cents", "pnl_dollars", "outcome",
]

_PNL_BREAKDOWN_HEADERS = [
    "group", "count", "wins", "losses", "win_rate_pct",
    "total_pnl_cents", "total_pnl_dollars", "avg_pnl_cents",
]

_EXEC_QUALITY_HEADERS = [
    "exec_mode", "total_trades", "filled", "canceled_pct",
    "avg_slippage_cents", "avg_raw_edge_pct", "avg_adj_edge_pct",
    "wins", "losses", "win_rate_pct", "total_pnl_cents",
]

_CALIBRATION_HEADERS = [
    "fair_prob_bin", "count", "predicted_win_pct",
    "actual_win_rate_pct", "calibration_error",
    "avg_pnl_cents",
]

_MODEL_PERF_HEADERS = [
    "model_name", "fills", "settled", "wins", "win_rate_pct",
    "fill_rate_pct", "total_pnl_cents", "total_pnl_dollars",
    "avg_pnl_cents", "avg_slippage_cents", "avg_raw_edge_pct",
]


# ── Safe write helper ─────────────────────────────────────────────────────────

def _safe_write_json(path: str, data: Any):
    """Write JSON atomically (temp file + rename)."""
    dir_ = os.path.dirname(path)
    os.makedirs(dir_, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile("w", dir=dir_, suffix=".tmp",
                                         delete=False) as f:
            json.dump(data, f, indent=2, default=str)
            tmp = f.name
        os.replace(tmp, path)
    except Exception as e:
        logger.warning("[REPORTS] Failed to write %s: %s", path, e)


def _safe_write_csv(path: str, headers: list, rows: list[dict]):
    """Write CSV atomically."""
    dir_ = os.path.dirname(path)
    os.makedirs(dir_, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile("w", dir=dir_, suffix=".tmp",
                                         delete=False, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            tmp = f.name
        os.replace(tmp, path)
    except Exception as e:
        logger.warning("[REPORTS] Failed to write %s: %s", path, e)


# ── Report generators ─────────────────────────────────────────────────────────

def write_analytics_summary():
    """Write analytics_summary.json from current analytics stats."""
    stats = analytics.get_stats()
    _safe_write_json(ANALYTICS_JSON, stats)
    logger.info("[REPORTS] analytics_summary.json written (%d settled trades)",
                stats.get("total_settled", 0))


def write_trades_csv():
    """Write enriched trades.csv with all trade_analytics columns."""
    all_trades = db.get_all_trade_analytics()
    rows = []
    for t in all_trades:
        pnl_cents = t.get("pnl_cents")
        rows.append({
            **{k: t.get(k, "") for k in _TRADES_HEADERS if k != "pnl_dollars"},
            "pnl_dollars": round(float(pnl_cents)/100, 4) if pnl_cents else "",
        })
    _safe_write_csv(TRADES_CSV, _TRADES_HEADERS, rows)
    logger.info("[REPORTS] trades.csv written (%d rows)", len(rows))


def write_pnl_by_sport():
    """Write pnl_by_sport.csv."""
    stats = analytics.get_stats()
    breakdown = stats.get("pnl_by_sport", {})
    rows = _breakdown_to_rows(breakdown)
    _safe_write_csv(PNL_SPORT_CSV, _PNL_BREAKDOWN_HEADERS, rows)
    logger.info("[REPORTS] pnl_by_sport.csv written (%d groups)", len(rows))


def write_pnl_by_regime():
    """Write pnl_by_regime.csv."""
    stats = analytics.get_stats()
    breakdown = stats.get("pnl_by_regime", {})
    rows = _breakdown_to_rows(breakdown)
    _safe_write_csv(PNL_REGIME_CSV, _PNL_BREAKDOWN_HEADERS, rows)
    logger.info("[REPORTS] pnl_by_regime.csv written (%d groups)", len(rows))


def write_pnl_by_model_name():
    """
    Write pnl_by_model_name.csv and pnl_by_model_name.json.

    Columns: model_name, fills, settled, wins, win_rate_pct, fill_rate_pct,
             total_pnl_cents, total_pnl_dollars, avg_pnl_cents,
             avg_slippage_cents, avg_raw_edge_pct.

    Sorted by total_pnl_cents descending so best-performing models are first.
    """
    stats     = analytics.get_stats()
    breakdown = stats.get("pnl_by_model_name", {})

    rows: list[dict] = []
    for model_name, m in sorted(
        breakdown.items(),
        key=lambda kv: kv[1].get("total_pnl_cents", 0),
        reverse=True,
    ):
        pnl_cents = m.get("total_pnl_cents", 0)
        rows.append({
            "model_name":        model_name,
            "fills":             m.get("fills", 0),
            "settled":           m.get("settled", 0),
            "wins":              m.get("wins", 0),
            "win_rate_pct":      m.get("win_rate_pct", 0.0),
            "fill_rate_pct":     m.get("fill_rate_pct", 0.0),
            "total_pnl_cents":   round(pnl_cents, 2),
            "total_pnl_dollars": round(pnl_cents / 100, 4),
            "avg_pnl_cents":     m.get("avg_pnl_cents", 0.0),
            "avg_slippage_cents":m.get("avg_slippage_cents", 0.0),
            "avg_raw_edge_pct":  m.get("avg_raw_edge_pct", 0.0),
        })

    _safe_write_csv(PNL_MODEL_CSV, _MODEL_PERF_HEADERS, rows)
    _safe_write_json(PNL_MODEL_JSON, {
        "generated_at":  stats.get("generated_at", ""),
        "total_settled": stats.get("total_settled", 0),
        "by_model_name": breakdown,
    })
    logger.info("[REPORTS] pnl_by_model_name written (%d models)", len(rows))


def write_execution_quality():
    """
    Write execution_quality.csv.
    Measures:
      - intended entry vs actual fill (slippage)
      - edge lost to slippage
      - fill rate by execution mode
      - win rate by execution mode
    """
    all_trades = db.get_all_trade_analytics()
    modes: dict[str, dict] = {}

    for t in all_trades:
        mode = t.get("exec_mode") or "unset"
        if mode not in modes:
            modes[mode] = {
                "total": 0, "fills": 0, "wins": 0, "losses": 0,
                "slippages": [], "raw_edges": [], "adj_edges": [], "pnls": [],
            }
        m = modes[mode]
        m["total"] += 1
        if t.get("fill_price"):
            m["fills"] += 1
            slip = t.get("slippage_cents")
            re   = t.get("raw_edge")
            ae   = t.get("edge_after_slip")
            if slip is not None: m["slippages"].append(float(slip))
            if re   is not None: m["raw_edges"].append(float(re))
            if ae   is not None: m["adj_edges"].append(float(ae))
        if t.get("outcome") == "win":  m["wins"]   += 1
        if t.get("outcome") == "loss": m["losses"] += 1
        if t.get("pnl_cents") is not None:
            m["pnls"].append(float(t["pnl_cents"]))

    def avg(v): return round(sum(v)/len(v), 3) if v else 0.0

    rows = []
    for mode, m in sorted(modes.items()):
        settled = m["wins"] + m["losses"]
        cancel_pct = round((m["total"] - m["fills"]) / m["total"] * 100, 1) if m["total"] else 0.0
        rows.append({
            "exec_mode":          mode,
            "total_trades":       m["total"],
            "filled":             m["fills"],
            "canceled_pct":       cancel_pct,
            "avg_slippage_cents": avg(m["slippages"]),
            "avg_raw_edge_pct":   round(avg(m["raw_edges"]) * 100, 2),
            "avg_adj_edge_pct":   round(avg(m["adj_edges"]) * 100, 2),
            "wins":               m["wins"],
            "losses":             m["losses"],
            "win_rate_pct":       round(m["wins"]/settled*100,1) if settled else 0.0,
            "total_pnl_cents":    round(sum(m["pnls"]), 2),
        })

    _safe_write_csv(EXEC_QUALITY_CSV, _EXEC_QUALITY_HEADERS, rows)
    logger.info("[REPORTS] execution_quality.csv written (%d modes)", len(rows))


def write_model_calibration():
    """
    Write model_calibration.csv.
    Shows whether fair_probability is well-calibrated — does the model's
    probability estimate match realized win rates?
    Also includes confidence_score calibration.
    """
    stats = analytics.get_stats()
    bins  = stats.get("prob_calibration_bins", [])
    rows  = []
    for b in bins:
        mid_pct = float(b["fair_prob_bin"].split("%")[0]) + 5
        rows.append({
            "fair_prob_bin":      b["fair_prob_bin"],
            "count":              b["count"],
            "predicted_win_pct":  round(mid_pct, 1),
            "actual_win_rate_pct":b["win_rate_pct"],
            "calibration_error":  b.get("calibration_err", 0),
            "avg_pnl_cents":      b["avg_pnl_cents"],
        })
    _safe_write_csv(CALIBRATION_CSV, _CALIBRATION_HEADERS, rows)
    logger.info("[REPORTS] model_calibration.csv written (%d bins)", len(rows))


def write_backtest_summary():
    """
    Run all scenario backtests and write backtest_summary.json.
    This is the most compute-intensive report; run less frequently.
    """
    engine   = BacktestEngine()
    results  = engine.run_all_scenarios()
    summary  = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scenarios":    [r.to_dict() for r in results],
        "comparison": _scenario_comparison(results),
    }
    _safe_write_json(BACKTEST_JSON, summary)
    logger.info("[REPORTS] backtest_summary.json written (%d scenarios)", len(results))
    for r in results:
        engine.log_result(r)


# ── Main entry point ──────────────────────────────────────────────────────────

def write_pnl_by_timing_classification():
    """Write pnl_by_timing_classification.csv — lagging/aligned/overreacting/noisy."""
    stats     = analytics.get_stats()
    breakdown = stats.get("pnl_by_timing_classification", {})
    rows      = _breakdown_to_rows(breakdown)
    if not rows:
        return
    _safe_write_csv(PNL_TIMING_CSV, _PNL_BREAKDOWN_HEADERS, rows)
    logger.info("[REPORTS] pnl_by_timing_classification written (%d classes)", len(rows))


def write_pnl_by_urgency_bucket():
    """Write pnl_by_urgency_bucket.csv — urgency quartile vs PnL outcomes."""
    stats     = analytics.get_stats()
    breakdown = stats.get("pnl_by_urgency_bucket", {})
    rows      = _breakdown_to_rows(breakdown)
    if not rows:
        return
    _safe_write_csv(PNL_URGENCY_CSV, _PNL_BREAKDOWN_HEADERS, rows)
    logger.info("[REPORTS] pnl_by_urgency_bucket written (%d buckets)", len(rows))


def write_pnl_by_staged_flag():
    """Write pnl_by_staged_flag.csv — staged vs full vs add entries."""
    stats     = analytics.get_stats()
    breakdown = stats.get("pnl_by_staged_flag", {})
    rows      = _breakdown_to_rows(breakdown)
    if not rows:
        return
    _safe_write_csv(PNL_STAGED_CSV, _PNL_BREAKDOWN_HEADERS, rows)
    logger.info("[REPORTS] pnl_by_staged_flag written (%d groups)", len(rows))


def write_timing_backtest_summary():
    """
    Write timing_backtest_summary.json comparing:
    - immediate entry (current behavior)
    - staged entry simulation (50% size)
    - chase-protection on vs off
    - wait-for-better-entry (higher urgency threshold)
    """
    from backtest import BacktestEngine
    engine = BacktestEngine()
    try:
        scenarios = engine.run_timing_scenarios()
        data = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "scenarios":    [s.to_dict() for s in scenarios],
        }
        _safe_write_json(TIMING_BACKTEST_JSON, data)
        logger.info("[REPORTS] timing_backtest_summary written (%d scenarios)", len(scenarios))
    except Exception as e:
        logger.warning("[REPORTS] timing backtest failed: %s", e)


def write_pnl_by_overlap_level():
    """Write pnl_by_overlap_level.csv — portfolio correlation impact on PnL."""
    stats     = analytics.get_stats()
    breakdown = stats.get("pnl_by_overlap_level", {})
    rows      = _breakdown_to_rows(breakdown)
    if not rows:
        return
    _safe_write_csv(PNL_OVERLAP_CSV, _PNL_BREAKDOWN_HEADERS, rows)
    logger.info("[REPORTS] pnl_by_overlap_level written (%d levels)", len(rows))


def write_pnl_by_concentration_bucket():
    """Write pnl_by_concentration_bucket.csv — portfolio concentration vs PnL."""
    stats     = analytics.get_stats()
    breakdown = stats.get("pnl_by_concentration_bucket", {})
    rows      = _breakdown_to_rows(breakdown)
    if not rows:
        return
    _safe_write_csv(PNL_CONCBKT_CSV, _PNL_BREAKDOWN_HEADERS, rows)
    logger.info("[REPORTS] pnl_by_concentration_bucket written (%d buckets)", len(rows))


def write_pnl_by_event_exposure():
    """Write pnl_by_event_exposure.csv — event exposure quartile vs PnL."""
    stats     = analytics.get_stats()
    breakdown = stats.get("pnl_by_event_exposure", {})
    rows      = _breakdown_to_rows(breakdown)
    if not rows:
        return
    _safe_write_csv(PNL_EVEXP_CSV, _PNL_BREAKDOWN_HEADERS, rows)
    logger.info("[REPORTS] pnl_by_event_exposure written (%d buckets)", len(rows))


def generate_all(include_backtest: bool = False):
    """
    Generate all reports.  Call periodically from bot.py.
    include_backtest=True runs the full scenario battery (slower).
    """
    closed = db.get_closed_trade_analytics()
    if not closed:
        logger.debug("[REPORTS] No closed trades yet — skipping report generation.")
        return

    write_analytics_summary()
    write_trades_csv()
    write_pnl_by_sport()
    write_pnl_by_regime()
    write_pnl_by_model_name()
    write_execution_quality()
    write_model_calibration()
    write_pnl_by_overlap_level()
    write_pnl_by_concentration_bucket()
    write_pnl_by_event_exposure()
    write_pnl_by_timing_classification()
    write_pnl_by_urgency_bucket()
    write_pnl_by_staged_flag()

    if include_backtest:
        write_backtest_summary()
        write_timing_backtest_summary()

    logger.info("[REPORTS] All reports generated. Output: %s", _BASE)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _breakdown_to_rows(breakdown: dict) -> list[dict]:
    rows = []
    for group, stats in sorted(breakdown.items()):
        wins   = stats.get("wins", 0)
        count  = stats.get("count", 0)
        losses = count - wins
        total_pnl = stats.get("total_pnl_cents", 0)
        rows.append({
            "group":            group,
            "count":            count,
            "wins":             wins,
            "losses":           losses,
            "win_rate_pct":     stats.get("win_rate_pct", 0),
            "total_pnl_cents":  total_pnl,
            "total_pnl_dollars":round(total_pnl / 100, 2),
            "avg_pnl_cents":    stats.get("avg_pnl_cents", 0),
        })
    return rows


# ── Optimization report writers ───────────────────────────────────────────────

OPT_RESULTS_CSV  = os.path.join(_BASE, "optimization_results.csv")
OPT_SUMMARY_JSON = os.path.join(_BASE, "optimization_summary.json")
BEST_PARAMS_JSON = os.path.join(_BASE, "best_parameters.json")

_OPT_RESULTS_HEADERS = [
    "rank",
    "combined_score", "in_sample_score", "out_of_sample_score",
    "min_raw_edge", "min_confidence", "max_spread_cents",
    "take_profit_cents", "slippage_buffer_cents",
    "aggressive_edge_threshold", "only_regime",
    "train_trades", "train_win_rate_pct", "train_pnl_dollars", "train_avg_pnl_cents",
    "test_trades",  "test_win_rate_pct",  "test_pnl_dollars",  "test_avg_pnl_cents",
    "train_avg_slippage", "test_avg_slippage",
    "weaknesses",
]


def write_optimization_results(all_results: list[dict]):
    """
    Write optimization_results.csv — one row per parameter combination,
    sorted by combined_score descending.  All columns are flat (no nested dicts).
    """
    rows = []
    for rank, r in enumerate(all_results, start=1):
        rows.append({
            "rank":                        rank,
            "combined_score":              r.get("combined_score"),
            "in_sample_score":             r.get("in_sample_score"),
            "out_of_sample_score":         r.get("out_of_sample_score"),
            "min_raw_edge":                r.get("min_raw_edge"),
            "min_confidence":              r.get("min_confidence"),
            "max_spread_cents":            r.get("max_spread_cents"),
            "take_profit_cents":           r.get("take_profit_cents"),
            "slippage_buffer_cents":       r.get("slippage_buffer_cents"),
            "aggressive_edge_threshold":   r.get("aggressive_edge_threshold"),
            "only_regime":                 r.get("only_regime"),
            "train_trades":                r.get("train_trades"),
            "train_win_rate_pct":          r.get("train_win_rate_pct"),
            "train_pnl_dollars":           r.get("train_pnl_dollars"),
            "train_avg_pnl_cents":         r.get("train_avg_pnl_cents"),
            "test_trades":                 r.get("test_trades"),
            "test_win_rate_pct":           r.get("test_win_rate_pct"),
            "test_pnl_dollars":            r.get("test_pnl_dollars"),
            "test_avg_pnl_cents":          r.get("test_avg_pnl_cents"),
            "train_avg_slippage":          r.get("train_avg_slippage"),
            "test_avg_slippage":           r.get("test_avg_slippage"),
            "weaknesses":                  r.get("weaknesses", ""),
        })
    _safe_write_csv(OPT_RESULTS_CSV, _OPT_RESULTS_HEADERS, rows)
    logger.info("[REPORTS] optimization_results.csv written (%d rows)", len(rows))


def write_optimization_summary(
    top_results: list[dict],
    best:        dict,
    sensitivity: dict,
    all_trades:  list[dict],
):
    """
    Write optimization_summary.json — top N results, best params,
    sensitivity analysis, and metadata.
    """
    # Strip non-serialisable nested 'params' key from each result
    def _clean(r: dict) -> dict:
        return {k: v for k, v in r.items()
                if k not in ("params", "score_components") and not isinstance(v, dict)}

    summary = {
        "generated_at":        datetime.now(timezone.utc).isoformat(),
        "total_trades_used":   len(all_trades),
        "total_combinations_evaluated": len(top_results),
        "best_combined_score": best.get("combined_score"),
        "best_params":         best.get("params"),
        "best_metrics": {
            "test_pnl_dollars":  best.get("test_pnl_dollars"),
            "test_win_rate_pct": best.get("test_win_rate_pct"),
            "test_trades":       best.get("test_trades"),
            "test_avg_slippage": best.get("test_avg_slippage"),
            "weaknesses":        best.get("weaknesses"),
        },
        "top_20_results": [_clean(r) for r in top_results],
        "sensitivity_analysis": {
            param: {
                "impact":     s["impact"],
                "direction":  s["direction"],
                "best_value": s["best_value"],
                "notes":      s["notes"],
                "values_tested": s["values_tested"],
            }
            for param, s in sensitivity.items()
        },
        "WARNING": (
            "These results are derived from limited historical data. "
            "Review carefully before changing any live trading parameters."
        ),
    }
    _safe_write_json(OPT_SUMMARY_JSON, summary)
    logger.info("[REPORTS] optimization_summary.json written")


def write_best_parameters(best: dict):
    """
    Write best_parameters.json — the single best parameter set in a format
    that can be reviewed and manually applied to config.py or env vars.

    NEVER auto-applied.  A human must explicitly apply these values.
    """
    params = best.get("params", {})
    out = {
        "generated_at":        datetime.now(timezone.utc).isoformat(),
        "combined_score":      best.get("combined_score"),
        "in_sample_score":     best.get("in_sample_score"),
        "out_of_sample_score": best.get("out_of_sample_score"),
        "test_pnl_dollars":    best.get("test_pnl_dollars"),
        "test_win_rate_pct":   best.get("test_win_rate_pct"),
        "test_trades":         best.get("test_trades"),
        "weaknesses":          best.get("weaknesses"),
        "WARNING": (
            "NOT auto-applied. Manual review and explicit config change required "
            "before any setting takes effect in live trading."
        ),
        "HOW_TO_APPLY": (
            "Set the corresponding environment variable (see config.py) OR "
            "pass --preset best to bot.py (not yet implemented) OR "
            "call config.load_optimized_params() in a dry-run context."
        ),
        "parameters": {
            "MIN_NET_EDGE_pct":           round(float(params.get("min_raw_edge", 0)) * 100, 1),
            "MIN_CONFIDENCE_pct":         round(float(params.get("min_confidence", 0)) * 100, 1),
            "MAX_SPREAD_CENTS":           params.get("max_spread_cents"),
            "TAKE_PROFIT_CENTS":          params.get("take_profit_cents"),
            "SLIPPAGE_BUFFER_CENTS":      params.get("slippage_buffer_cents"),
            "AGGRESSIVE_EDGE_THRESHOLD_pct": (
                round(float(params.get("aggressive_edge_threshold", 0)) * 100, 1)
                if params.get("aggressive_edge_threshold") is not None else None
            ),
            "REGIME_FILTER":              params.get("only_regime"),
        },
        "env_var_equivalents": {
            "MIN_CONFIDENCE":          str(params.get("min_confidence", "")),
            "MAX_SPREAD_CENTS":        str(int(params.get("max_spread_cents") or 0)),
            "TAKE_PROFIT_CENTS":       str(int(params.get("take_profit_cents") or 0))
                                       if params.get("take_profit_cents") else "0",
            "SLIPPAGE_BUFFER_CENTS":   str(params.get("slippage_buffer_cents", "")),
            "AGGRESSIVE_EDGE_THRESHOLD": str(params.get("aggressive_edge_threshold", "")),
        },
    }
    _safe_write_json(BEST_PARAMS_JSON, out)
    logger.info("[REPORTS] best_parameters.json written (score=%.3f)", best.get("combined_score", 0))


# ── Sport-specific optimization report writers ────────────────────────────────

OPT_RESULTS_SPORT_CSV    = os.path.join(_BASE, "optimization_results_by_sport.csv")
OPT_SUMMARY_SPORT_JSON   = os.path.join(_BASE, "optimization_summary_by_sport.json")
BEST_PARAMS_SPORT_JSON   = os.path.join(_BASE, "best_parameters_by_sport.json")
SPORT_COMPARISON_CSV     = os.path.join(_BASE, "sport_config_comparison.csv")

_SPORT_RESULTS_HEADERS = [
    "sport", "status", "trade_count",
    "combined_score", "in_sample_score", "out_of_sample_score",
    "min_raw_edge", "min_confidence", "max_spread_cents",
    "take_profit_cents", "slippage_buffer_cents",
    "aggressive_edge_threshold", "only_regime",
    "test_trades", "test_win_rate_pct", "test_pnl_dollars",
    "test_avg_pnl_cents", "test_avg_slippage",
    "weaknesses", "note",
]

_SPORT_COMPARISON_HEADERS = [
    "sport", "trade_count", "status",
    "global_score",  "global_win_rate_pct",  "global_pnl_dollars",
    "global_avg_pnl_cents", "global_avg_slippage",
    "sport_score",   "sport_win_rate_pct",   "sport_pnl_dollars",
    "sport_avg_pnl_cents",  "sport_avg_slippage",
    "score_improvement", "pnl_improvement_dollars",
    "recommendation", "weaknesses",
    "sport_min_raw_edge", "sport_min_confidence",
    "sport_max_spread_cents", "sport_take_profit_cents", "sport_only_regime",
]


def write_optimization_results_by_sport(sport_results: dict):
    """
    Write optimization_results_by_sport.csv — one row per sport showing the
    best parameter set found for that sport (or the global fallback).
    """
    rows = []
    for sport, sr in sorted(sport_results.items()):
        params   = sr.get("params", {})
        metrics  = sr.get("metrics", {})
        best_row = (sr.get("top_results") or [{}])[0] if sr.get("status") == "ok" else {}
        rows.append({
            "sport":                      sport,
            "status":                     sr.get("status", "unknown"),
            "trade_count":                sr.get("trade_count", 0),
            "combined_score":             sr.get("combined_score", ""),
            "in_sample_score":            best_row.get("in_sample_score", ""),
            "out_of_sample_score":        best_row.get("out_of_sample_score", ""),
            "min_raw_edge":               params.get("min_raw_edge"),
            "min_confidence":             params.get("min_confidence"),
            "max_spread_cents":           params.get("max_spread_cents"),
            "take_profit_cents":          params.get("take_profit_cents"),
            "slippage_buffer_cents":      params.get("slippage_buffer_cents"),
            "aggressive_edge_threshold":  params.get("aggressive_edge_threshold"),
            "only_regime":                params.get("only_regime"),
            "test_trades":                metrics.get("test_trades", ""),
            "test_win_rate_pct":          metrics.get("test_win_rate_pct", ""),
            "test_pnl_dollars":           metrics.get("test_pnl_dollars", ""),
            "test_avg_pnl_cents":         best_row.get("test_avg_pnl_cents", ""),
            "test_avg_slippage":          metrics.get("test_avg_slippage", ""),
            "weaknesses":                 metrics.get("weaknesses", ""),
            "note":                       sr.get("note", ""),
        })
    _safe_write_csv(OPT_RESULTS_SPORT_CSV, _SPORT_RESULTS_HEADERS, rows)
    logger.info("[REPORTS] optimization_results_by_sport.csv written (%d sports)", len(rows))


def write_optimization_summary_by_sport(sport_results: dict):
    """
    Write optimization_summary_by_sport.json — full per-sport summary
    including status, metrics, and top parameter sets.
    """
    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sports": {},
        "WARNING": (
            "Sport-specific results are derived from limited historical data. "
            "A sport with few trades has high variance — prefer global params "
            "until sample size grows."
        ),
    }

    for sport, sr in sorted(sport_results.items()):
        entry = {
            "status":      sr.get("status"),
            "trade_count": sr.get("trade_count", 0),
        }
        if sr.get("status") == "ok":
            entry.update({
                "combined_score":  sr.get("combined_score"),
                "best_params":     sr.get("params"),
                "metrics":         sr.get("metrics"),
                "top_5_results": [
                    {k: v for k, v in r.items()
                     if k not in ("params", "score_components")}
                    for r in (sr.get("top_results") or [])[:5]
                ],
            })
        else:
            entry.update({
                "min_required":   sr.get("min_required"),
                "fallback":       sr.get("fallback"),
                "fallback_params":sr.get("params"),
                "note":           sr.get("note"),
            })
        out["sports"][sport] = entry

    _safe_write_json(OPT_SUMMARY_SPORT_JSON, out)
    logger.info("[REPORTS] optimization_summary_by_sport.json written (%d sports)",
                len(sport_results))


def write_best_parameters_by_sport(sport_results: dict, global_best_params: dict):
    """
    Write best_parameters_by_sport.json — machine-readable recommended params
    per sport.  Includes env-var equivalents for manual application.
    NEVER auto-applied.
    """
    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "WARNING": (
            "NOT auto-applied. Manual review required before changing any "
            "live trading configuration."
        ),
        "HOW_TO_USE": (
            "For each sport, call config.load_optimized_params_for_sport(sport) "
            "to retrieve these values for review. Apply only via explicit config "
            "change or environment variable update."
        ),
        "global_fallback_params": global_best_params,
        "sports": {},
    }

    for sport, sr in sorted(sport_results.items()):
        params = sr.get("params", global_best_params) or {}
        entry: dict = {
            "status":      sr.get("status"),
            "trade_count": sr.get("trade_count", 0),
        }
        if sr.get("status") == "ok":
            entry.update({
                "combined_score": sr.get("combined_score"),
                "metrics":        sr.get("metrics"),
                "parameters": {
                    "MIN_NET_EDGE_pct":     round(float(params.get("min_raw_edge", 0)) * 100, 1),
                    "MIN_CONFIDENCE_pct":   round(float(params.get("min_confidence", 0)) * 100, 1),
                    "MAX_SPREAD_CENTS":     params.get("max_spread_cents"),
                    "TAKE_PROFIT_CENTS":    params.get("take_profit_cents"),
                    "SLIPPAGE_BUFFER_CENTS":params.get("slippage_buffer_cents"),
                    "AGGRESSIVE_EDGE_THRESHOLD_pct": (
                        round(float(params.get("aggressive_edge_threshold", 0)) * 100, 1)
                        if params.get("aggressive_edge_threshold") is not None else None
                    ),
                    "REGIME_FILTER":        params.get("only_regime"),
                },
                "raw_params": params,
            })
        else:
            entry.update({
                "note":           sr.get("note"),
                "fallback":       "global",
                "raw_params":     global_best_params,
                "parameters":     {"note": "Using global fallback — insufficient sport data."},
            })
        out["sports"][sport] = entry

    _safe_write_json(BEST_PARAMS_SPORT_JSON, out)
    logger.info("[REPORTS] best_parameters_by_sport.json written (%d sports)",
                len(sport_results))


def write_sport_config_comparison(comparison_rows: list[dict]):
    """
    Write sport_config_comparison.csv — one row per sport comparing
    global-config performance vs sport-specific-config performance.

    Columns: sport, trade_count, global_*, sport_*, score_improvement,
             pnl_improvement_dollars, recommendation.
    """
    if not comparison_rows:
        logger.debug("[REPORTS] No comparison rows — skipping sport_config_comparison.csv")
        return
    _safe_write_csv(SPORT_COMPARISON_CSV, _SPORT_COMPARISON_HEADERS, comparison_rows)
    n_better = sum(1 for r in comparison_rows if r.get("recommendation") == "use_sport_specific")
    n_global = sum(1 for r in comparison_rows if r.get("recommendation") == "use_global")
    logger.info(
        "[REPORTS] sport_config_comparison.csv written (%d sports: "
        "%d favour sport-specific, %d favour global, %d marginal/insufficient)",
        len(comparison_rows), n_better, n_global,
        len(comparison_rows) - n_better - n_global,
    )


# ── Market-type optimization report writers ───────────────────────────────────

OPT_RESULTS_MT_CSV   = os.path.join(_BASE, "optimization_results_by_market_type.csv")
OPT_SUMMARY_MT_JSON  = os.path.join(_BASE, "optimization_summary_by_market_type.json")
BEST_PARAMS_MT_JSON  = os.path.join(_BASE, "best_parameters_by_market_type.json")
MT_COMPARISON_CSV    = os.path.join(_BASE, "market_type_config_comparison.csv")

_MT_RESULTS_HEADERS = [
    "bucket", "sport", "market_type", "status", "trade_count", "min_required",
    "combined_score",
    "min_raw_edge", "min_confidence", "max_spread_cents",
    "take_profit_cents", "slippage_buffer_cents",
    "aggressive_edge_threshold", "only_regime",
    "test_trades", "test_win_rate_pct", "test_pnl_dollars",
    "test_avg_slippage", "weaknesses",
    "fallback", "fallback_reason",
]

_MT_COMPARISON_HEADERS = [
    "bucket", "sport", "market_type", "trade_count", "mt_status",
    # Global config
    "global_score",  "global_win_rate_pct",  "global_pnl_dollars", "global_avg_slippage",
    # Sport config
    "sport_score",   "sport_win_rate_pct",   "sport_pnl_dollars",  "sport_avg_slippage",
    # Market-type config
    "mt_score",      "mt_win_rate_pct",      "mt_pnl_dollars",     "mt_avg_slippage",
    # Deltas and recommendation
    "sport_score_delta", "mt_score_delta", "mt_pnl_delta", "recommendation",
    "fallback_reason",
    # Best market-type params
    "mt_min_raw_edge", "mt_min_confidence", "mt_max_spread_cents",
    "mt_take_profit_cents", "mt_only_regime",
]


def write_optimization_results_by_market_type(mt_results: dict):
    """
    Write optimization_results_by_market_type.csv — one row per (sport,
    market_type) bucket, showing best params found or the fallback params.
    """
    rows = []
    for key, mr in sorted(mt_results.items()):
        params   = mr.get("params", {})
        metrics  = mr.get("metrics", {})
        best_row = (mr.get("top_results") or [{}])[0] if mr.get("status") == "ok" else {}
        rows.append({
            "bucket":                   key,
            "sport":                    mr.get("sport", ""),
            "market_type":              mr.get("market_type", ""),
            "status":                   mr.get("status", "unknown"),
            "trade_count":              mr.get("trade_count", 0),
            "min_required":             mr.get("min_required", ""),
            "combined_score":           mr.get("combined_score", ""),
            "min_raw_edge":             params.get("min_raw_edge"),
            "min_confidence":           params.get("min_confidence"),
            "max_spread_cents":         params.get("max_spread_cents"),
            "take_profit_cents":        params.get("take_profit_cents"),
            "slippage_buffer_cents":    params.get("slippage_buffer_cents"),
            "aggressive_edge_threshold":params.get("aggressive_edge_threshold"),
            "only_regime":              params.get("only_regime"),
            "test_trades":              metrics.get("test_trades", ""),
            "test_win_rate_pct":        metrics.get("test_win_rate_pct", ""),
            "test_pnl_dollars":         metrics.get("test_pnl_dollars", ""),
            "test_avg_slippage":        metrics.get("test_avg_slippage", ""),
            "weaknesses":               metrics.get("weaknesses", ""),
            "fallback":                 mr.get("fallback", ""),
            "fallback_reason":          mr.get("fallback_reason", ""),
        })
    _safe_write_csv(OPT_RESULTS_MT_CSV, _MT_RESULTS_HEADERS, rows)
    logger.info(
        "[REPORTS] optimization_results_by_market_type.csv written (%d buckets: "
        "%d optimized, %d fallback).",
        len(rows),
        sum(1 for r in rows if r["status"] == "ok"),
        sum(1 for r in rows if r["status"] == "insufficient_data"),
    )


def write_optimization_summary_by_market_type(mt_results: dict):
    """
    Write optimization_summary_by_market_type.json — full per-bucket summary
    with top-5 param sets for each optimized bucket.
    """
    output: dict = {
        "generated_at":  _now(),
        "total_buckets": len(mt_results),
        "optimized":     sum(1 for r in mt_results.values() if r.get("status") == "ok"),
        "insufficient":  sum(1 for r in mt_results.values() if r.get("status") == "insufficient_data"),
        "buckets": {},
    }
    for key, mr in sorted(mt_results.items()):
        params  = mr.get("params", {})
        metrics = mr.get("metrics", {})
        entry: dict = {
            "status":      mr.get("status", "unknown"),
            "sport":       mr.get("sport", ""),
            "market_type": mr.get("market_type", ""),
            "trade_count": mr.get("trade_count", 0),
        }
        if mr.get("status") == "ok":
            entry.update({
                "combined_score": mr.get("combined_score"),
                "best_params":    params,
                "metrics":        metrics,
                "top_5_results":  (mr.get("top_results") or [])[:5],
            })
        else:
            entry.update({
                "fallback":         mr.get("fallback", "global"),
                "fallback_reason":  mr.get("fallback_reason", ""),
                "fallback_params":  params,
                "min_required":     mr.get("min_required", 5),
                "note": (
                    f"Only {mr.get('trade_count', 0)} closed trades in "
                    f"{key}; need {mr.get('min_required', 5)}+ for reliable "
                    f"bucket-specific estimate. Fallback: {mr.get('fallback', 'global')} params."
                ),
            })
        output["buckets"][key] = entry

    _safe_write_json(OPT_SUMMARY_MT_JSON, output)
    logger.info("[REPORTS] optimization_summary_by_market_type.json written (%d buckets).",
                len(mt_results))


def write_best_parameters_by_market_type(mt_results: dict, global_best_params: dict):
    """
    Write best_parameters_by_market_type.json — machine-readable param file.

    Structure:
      {
        "generated_at":         ...,
        "global_fallback_params": {...},
        "buckets": {
          "Basketball:game_winner": {
            "status":       "ok" | "insufficient_data",
            "sport":        "Basketball",
            "market_type":  "game_winner",
            "raw_params":   {...},   # ← apply these
            "parameters":   {...},   # human-readable with env var names
            "fallback":     "sport" | "global",
            "fallback_reason": "...",
            "NOT_AUTO_APPLIED": "..."
          }
        }
      }
    """
    _WARNING = (
        "These parameters are NOT automatically applied to live trading. "
        "Human review required before use."
    )
    buckets: dict = {}
    for key, mr in sorted(mt_results.items()):
        params = mr.get("params", {})
        entry: dict = {
            "status":           mr.get("status", "unknown"),
            "sport":            mr.get("sport", ""),
            "market_type":      mr.get("market_type", ""),
            "trade_count":      mr.get("trade_count", 0),
            "NOT_AUTO_APPLIED": _WARNING,
        }
        if mr.get("status") == "ok":
            entry.update({
                "combined_score": mr.get("combined_score"),
                "raw_params":     params,
                "parameters": {
                    "MIN_RAW_EDGE":              params.get("min_raw_edge"),
                    "MIN_CONFIDENCE":            params.get("min_confidence"),
                    "MAX_SPREAD_CENTS":          params.get("max_spread_cents"),
                    "TAKE_PROFIT_CENTS":         params.get("take_profit_cents"),
                    "SLIPPAGE_BUFFER_CENTS":     params.get("slippage_buffer_cents"),
                    "AGGRESSIVE_EDGE_THRESHOLD": params.get("aggressive_edge_threshold"),
                    "ONLY_REGIME":               params.get("only_regime"),
                },
            })
        else:
            entry.update({
                "fallback":        mr.get("fallback", "global"),
                "fallback_reason": mr.get("fallback_reason", ""),
                "raw_params":      params,
                "note": (
                    f"Insufficient data for bucket {key}. "
                    f"Showing {mr.get('fallback', 'global')} params as recommended fallback."
                ),
            })
        buckets[key] = entry

    output = {
        "generated_at":          _now(),
        "NOT_AUTO_APPLIED":      _WARNING,
        "global_fallback_params": global_best_params,
        "total_buckets":         len(buckets),
        "buckets":               buckets,
    }
    _safe_write_json(BEST_PARAMS_MT_JSON, output)
    logger.info("[REPORTS] best_parameters_by_market_type.json written (%d buckets).",
                len(buckets))


def write_market_type_config_comparison(comparison_rows: list[dict]):
    """
    Write market_type_config_comparison.csv — one row per (sport, market_type)
    bucket comparing global vs sport-specific vs market-type-specific config.
    """
    if not comparison_rows:
        logger.debug("[REPORTS] No rows — skipping market_type_config_comparison.csv")
        return
    _safe_write_csv(MT_COMPARISON_CSV, _MT_COMPARISON_HEADERS, comparison_rows)
    n_mt     = sum(1 for r in comparison_rows if r.get("recommendation") == "use_market_type")
    n_sport  = sum(1 for r in comparison_rows if r.get("recommendation") == "use_sport")
    n_global = sum(1 for r in comparison_rows if r.get("recommendation") == "use_global")
    n_insuff = sum(1 for r in comparison_rows if r.get("recommendation") == "insufficient_data")
    logger.info(
        "[REPORTS] market_type_config_comparison.csv written (%d buckets: "
        "market_type=%d  sport=%d  global=%d  insufficient=%d).",
        len(comparison_rows), n_mt, n_sport, n_global, n_insuff,
    )
