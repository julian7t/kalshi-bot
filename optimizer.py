"""
optimizer.py — Evidence-based parameter optimization for the Kalshi trading bot.

Uses grid search (and optional random search) over configurable strategy/execution
parameters.  Each combination is evaluated via BacktestEngine.run_on_trades() on
train and test subsets of historical trade_analytics records.

Overfitting guardrails:
  - 70 / 30 train-test split on entry_at timestamps
  - Scoring function rewards PnL, win rate, fill rate; penalizes low sample size,
    sport/regime concentration, and high slippage
  - Combined score = 0.4 × in-sample + 0.6 × out-of-sample
  - Sensitivity analysis quantifies which parameters actually mattered

Output (via reports.write_optimization_*):
  metrics/optimization_results.csv
  metrics/optimization_summary.json
  metrics/best_parameters.json

Usage:
    from optimizer import ParameterOptimizer
    opt = ParameterOptimizer()
    summary = opt.run()

    # optional: random search on top of grid
    opt2 = ParameterOptimizer(search_mode="random", random_samples=200)
    summary2 = opt2.run()

IMPORTANT:
  The best parameter set is written to best_parameters.json for manual review.
  It is NEVER auto-applied to live trading.  A human must review and update
  config.py (or set environment variables) explicitly.
"""

import itertools
import logging
import random
import time
from datetime import datetime, timezone
from typing import Any, Optional

import config
import db
import reports
from backtest import BacktestEngine, BacktestResult, compute_max_drawdown

# ── Known sport labels (must match model.py sport tags stored in trade_analytics) ──
KNOWN_SPORTS: list[str] = [
    "Basketball",
    "Football",
    "Baseball",
    "Hockey",
    "Soccer",
    "Generic",
]

logger = logging.getLogger("kalshi_bot.optimizer")

# ── Parameter search space ────────────────────────────────────────────────────
#
# GRID_SPACE defines the curated grid used for grid search.
# Values are chosen to be meaningfully different without overfitting to
# fine-grained distinctions that the data cannot support.
#
# RANDOM_SPACE defines the full continuous ranges for random search.

GRID_SPACE: dict[str, list] = {
    "min_raw_edge":              [0.03, 0.05, 0.07, 0.10],
    "min_confidence":            [0.35, 0.50, 0.65, 0.75],
    "max_spread_cents":          [6.0,  8.0,  10.0, 15.0],
    "take_profit_cents":         [None, 8.0,  15.0, 25.0],
    "slippage_buffer_cents":     [0.0,  1.0,  2.0],
    "aggressive_edge_threshold": [0.08, 0.10, 0.12],
    "only_regime":               [None, "calm", "volatile"],
}

RANDOM_SPACE: dict[str, Any] = {
    "min_raw_edge":              (0.02, 0.15),
    "min_confidence":            (0.25, 0.85),
    "max_spread_cents":          (4.0,  18.0),
    "take_profit_cents":         [None, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0],
    "slippage_buffer_cents":     (0.0,  4.0),
    "aggressive_edge_threshold": (0.06, 0.18),
    "only_regime":               [None, "calm", "volatile", "trending"],
}

# Minimum settled trade count for a result to receive a non-zero score
MIN_SETTLED_FOR_SCORE = 5


# ── Scoring function ──────────────────────────────────────────────────────────

def score_result(result: BacktestResult, n_total_trades: int, params: dict) -> dict:
    """
    Multi-objective score for a BacktestResult.  Returns a dict with:
      score         — combined score (used for ranking)
      components    — breakdown of each scoring term
      weaknesses    — list of human-readable weakness notes
    """
    d        = result.to_dict()
    n_filled = d["filled_trades"]
    settled  = d["wins"] + d["losses"]
    weaknesses: list[str] = []

    if settled < MIN_SETTLED_FOR_SCORE:
        weaknesses.append(f"insufficient data (settled={settled}, need≥{MIN_SETTLED_FOR_SCORE})")
        return {
            "score":      -1000.0,
            "components": {},
            "weaknesses": weaknesses,
        }

    wr        = d["wins"] / settled
    avg_pnl   = d["avg_pnl_cents"]
    avg_slip  = abs(d["avg_slippage_cents"])
    fill_rate = n_filled / n_total_trades if n_total_trades > 0 else 0.0

    # Sport concentration (HHI: 0 = perfectly distributed, 1 = all one sport)
    sport_counts = [v["count"] for v in d["by_sport"].values()]
    hhi_sport    = sum((c / n_filled) ** 2 for c in sport_counts) if n_filled > 0 else 1.0

    # Regime concentration
    regime_counts = [v["count"] for v in d["by_regime"].values()]
    hhi_regime    = sum((c / n_filled) ** 2 for c in regime_counts) if n_filled > 0 else 1.0

    # Max drawdown
    max_dd = compute_max_drawdown(result.pnl_list) / 100.0  # convert to dollars

    # Sample size discount (ramps up linearly, full weight at MIN_SETTLED_FOR_SCORE * 3)
    sample_factor = min(settled / (MIN_SETTLED_FOR_SCORE * 3), 1.0)

    # Regime filter penalty — restricting to one regime reduces breadth
    regime_filter_penalty = 15.0 if params.get("only_regime") else 0.0

    # Score components
    pnl_term        =  avg_pnl * 2.5
    wr_term         = (wr - 0.50) * 120.0        # +/- 60 at extremes
    fill_rate_term  =  fill_rate  * 30.0          # reward breadth
    slippage_term   = -avg_slip   * 2.5
    sport_conc_term = -hhi_sport  * 35.0
    regime_conc_term= -hhi_regime * 25.0
    drawdown_term   = -max_dd     * 5.0           # dollars
    regime_pen_term = -regime_filter_penalty

    raw_score = (
        pnl_term + wr_term + fill_rate_term +
        slippage_term + sport_conc_term + regime_conc_term +
        drawdown_term + regime_pen_term
    ) * sample_factor

    # Weakness notes
    if avg_slip > 4.0:
        weaknesses.append(f"high slippage ({avg_slip:.1f}¢)")
    if fill_rate < 0.20:
        weaknesses.append(f"low fill rate ({fill_rate:.0%})")
    if hhi_sport > 0.70:
        dominant = max(d["by_sport"].items(), key=lambda x: x[1]["count"])[0]
        weaknesses.append(f"sport-concentrated ({dominant} HHI={hhi_sport:.2f})")
    if hhi_regime > 0.80:
        dominant = max(d["by_regime"].items(), key=lambda x: x[1]["count"])[0]
        weaknesses.append(f"regime-concentrated ({dominant} HHI={hhi_regime:.2f})")
    if max_dd > 5.0:
        weaknesses.append(f"max drawdown ${max_dd:.2f}")
    if params.get("only_regime"):
        weaknesses.append(f"regime-filtered ({params['only_regime']} only)")

    return {
        "score": round(raw_score, 4),
        "components": {
            "pnl_term":         round(pnl_term,         3),
            "wr_term":          round(wr_term,           3),
            "fill_rate_term":   round(fill_rate_term,    3),
            "slippage_term":    round(slippage_term,     3),
            "sport_conc_term":  round(sport_conc_term,   3),
            "regime_conc_term": round(regime_conc_term,  3),
            "drawdown_term":    round(drawdown_term,     3),
            "regime_pen_term":  round(regime_pen_term,   3),
            "sample_factor":    round(sample_factor,     3),
        },
        "weaknesses": weaknesses,
    }


# ── Optimizer ─────────────────────────────────────────────────────────────────

class ParameterOptimizer:
    """
    Grid (or random) search over the parameter space, evaluated via
    BacktestEngine.run_on_trades() on train/test subsets.

    search_mode:     "grid" (default) or "random"
    random_samples:  number of random combinations to try (only for "random")
    train_fraction:  fraction of trades used for in-sample evaluation (default 0.70)
    """

    def __init__(
        self,
        search_mode:     str   = "grid",
        random_samples:  int   = 150,
        train_fraction:  float = 0.70,
    ):
        self.search_mode    = search_mode
        self.random_samples = random_samples
        self.train_fraction = train_fraction
        self.engine         = BacktestEngine()

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self) -> dict:
        """
        Run the full optimization.  Returns a summary dict.

        Steps:
          1. Fetch all closed trades
          2. Train / test split
          3. Build parameter combinations
          4. Evaluate each combination (train score + test score)
          5. Sort by combined score
          6. Sensitivity analysis on best config
          7. Write reports (optimization_results.csv, optimization_summary.json,
                            best_parameters.json)
        """
        t0     = time.monotonic()
        all_trades = db.get_closed_trade_analytics()

        if len(all_trades) < MIN_SETTLED_FOR_SCORE:
            logger.warning(
                "[OPT] Not enough closed trades (%d) to optimize. "
                "Need at least %d.  Skipping.",
                len(all_trades), MIN_SETTLED_FOR_SCORE,
            )
            return {"status": "insufficient_data", "count": len(all_trades)}

        logger.info("[OPT] Starting optimization on %d closed trades.", len(all_trades))

        train_trades, test_trades = self._split(all_trades)
        logger.info(
            "[OPT] Train/test split: %d / %d (%.0f%% / %.0f%%)",
            len(train_trades), len(test_trades),
            self.train_fraction * 100, (1 - self.train_fraction) * 100,
        )

        param_sets = self._build_param_sets()
        logger.info("[OPT] Evaluating %d parameter combinations (%s search).",
                    len(param_sets), self.search_mode)

        all_results:  list[dict] = []
        best_score = -float("inf")
        best_idx   = 0

        for i, params in enumerate(param_sets):
            row = self._evaluate(params, train_trades, test_trades, all_trades)
            all_results.append(row)

            if row["combined_score"] > best_score:
                best_score = row["combined_score"]
                best_idx   = i
                logger.info(
                    "[OPT] New best score=%.3f (set #%d): edge≥%.0f%% conf≥%.0f%% "
                    "spread≤%.0f¢ tp=%s regime=%s",
                    best_score, i + 1,
                    float(params.get("min_raw_edge", 0)) * 100,
                    float(params.get("min_confidence", 0)) * 100,
                    float(params.get("max_spread_cents", 0)),
                    params.get("take_profit_cents"),
                    params.get("only_regime") or "all",
                )

        all_results.sort(key=lambda r: r["combined_score"], reverse=True)
        best = all_results[0]

        logger.info(
            "[OPT] Optimization complete: %d sets evaluated in %.1fs.  "
            "Best score=%.3f, pnl=$%.2f, win_rate=%.1f%%, trades=%d",
            len(all_results),
            time.monotonic() - t0,
            best["combined_score"],
            best["test_pnl_dollars"],
            best["test_win_rate_pct"],
            best["test_trades"],
        )

        # Sensitivity analysis on best config
        sensitivity = self._sensitivity_analysis(best["params"], all_trades)

        # Write all reports
        reports.write_optimization_results(all_results)
        reports.write_optimization_summary(all_results[:20], best, sensitivity, all_trades)
        reports.write_best_parameters(best)

        logger.info("[OPT] Optimization report generated.")

        return {
            "status":       "ok",
            "combinations": len(all_results),
            "best_params":  best["params"],
            "best_score":   best["combined_score"],
            "elapsed_s":    round(time.monotonic() - t0, 1),
            "sensitivity":  sensitivity,
        }

    # ── Train / test split ────────────────────────────────────────────────────

    def _split(self, trades: list[dict]) -> tuple[list[dict], list[dict]]:
        """
        Sort trades by entry_at timestamp and split 70 / 30.
        Trades without a valid timestamp go into training.
        """
        def ts_key(t):
            try:
                return datetime.fromisoformat(t.get("entry_at", "")).timestamp()
            except Exception:
                return 0.0

        sorted_trades = sorted(trades, key=ts_key)
        cut = max(1, int(len(sorted_trades) * self.train_fraction))
        return sorted_trades[:cut], sorted_trades[cut:]

    # ── Parameter set generation ──────────────────────────────────────────────

    def _build_param_sets(self) -> list[dict]:
        if self.search_mode == "random":
            return self._random_param_sets()
        return self._grid_param_sets()

    def _grid_param_sets(self) -> list[dict]:
        keys   = list(GRID_SPACE.keys())
        values = list(GRID_SPACE.values())
        sets   = []
        for combo in itertools.product(*values):
            sets.append(dict(zip(keys, combo)))
        return sets

    def _random_param_sets(self) -> list[dict]:
        sets = []
        for _ in range(self.random_samples):
            params = {}
            for key, space in RANDOM_SPACE.items():
                if isinstance(space, list):
                    params[key] = random.choice(space)
                elif isinstance(space, tuple) and len(space) == 2:
                    lo, hi = space
                    params[key] = round(random.uniform(lo, hi), 3)
                else:
                    params[key] = space
            sets.append(params)
        return sets

    # ── Single combination evaluation ─────────────────────────────────────────

    def _evaluate(
        self,
        params:        dict,
        train_trades:  list[dict],
        test_trades:   list[dict],
        all_trades:    list[dict],
    ) -> dict:
        """Evaluate a single param set on train + test splits.  Returns a flat row dict."""
        train_r = self.engine.run_on_trades(train_trades, params, label="train")
        test_r  = self.engine.run_on_trades(test_trades,  params, label="test")

        train_scored = score_result(train_r, len(train_trades), params)
        test_scored  = score_result(test_r,  len(test_trades),  params)

        in_score  = train_scored["score"]
        out_score = test_scored["score"]
        combined  = round(0.4 * in_score + 0.6 * out_score, 4)

        td = train_r.to_dict()
        od = test_r.to_dict()

        weaknesses = list(set(train_scored["weaknesses"] + test_scored["weaknesses"]))

        return {
            # --- params
            "params":                  params,
            "min_raw_edge":            params.get("min_raw_edge"),
            "min_confidence":          params.get("min_confidence"),
            "max_spread_cents":        params.get("max_spread_cents"),
            "take_profit_cents":       params.get("take_profit_cents"),
            "slippage_buffer_cents":   params.get("slippage_buffer_cents"),
            "aggressive_edge_threshold": params.get("aggressive_edge_threshold"),
            "only_regime":             params.get("only_regime"),
            # --- combined scores
            "in_sample_score":         in_score,
            "out_of_sample_score":     out_score,
            "combined_score":          combined,
            # --- train metrics
            "train_trades":            td["filled_trades"],
            "train_win_rate_pct":      td["win_rate_pct"],
            "train_pnl_dollars":       td["total_pnl_dollars"],
            "train_avg_pnl_cents":     td["avg_pnl_cents"],
            "train_avg_slippage":      td["avg_slippage_cents"],
            # --- test metrics
            "test_trades":             od["filled_trades"],
            "test_win_rate_pct":       od["win_rate_pct"],
            "test_pnl_dollars":        od["total_pnl_dollars"],
            "test_avg_pnl_cents":      od["avg_pnl_cents"],
            "test_avg_slippage":       od["avg_slippage_cents"],
            # --- score components (test set)
            "score_components":        test_scored["components"],
            # --- weaknesses
            "weaknesses":              "; ".join(weaknesses) if weaknesses else "none",
        }

    # ── Sensitivity analysis ──────────────────────────────────────────────────

    def _sensitivity_analysis(
        self,
        best_params: dict,
        all_trades:  list[dict],
    ) -> dict:
        """
        Vary each parameter one at a time (holding others at best_params values).
        Returns a dict of {param_name: analysis_dict}.

        Analysis dict contains:
          values_tested   — list of {value, score, pnl, trades}
          best_value      — value that achieved highest score in isolation
          worst_value     — value that achieved lowest score in isolation
          impact          — score range across values tested (higher = more sensitive)
          direction       — "higher_better" | "lower_better" | "flat"
          notes           — human-readable interpretation
        """
        logger.info("[OPT] Running sensitivity analysis on best config...")
        sensitivity: dict[str, dict] = {}

        for param_name, grid_values in GRID_SPACE.items():
            tested = []
            for val in grid_values:
                test_params = {**best_params, param_name: val}
                result      = self.engine.run_on_trades(all_trades, test_params,
                                                        label=f"sens_{param_name}")
                scored      = score_result(result, len(all_trades), test_params)
                tested.append({
                    "value":   val,
                    "score":   scored["score"],
                    "pnl":     round(result.total_pnl_cents / 100, 2),
                    "trades":  result.filled_trades,
                    "win_rate":result.to_dict()["win_rate_pct"],
                })

            scores = [x["score"] for x in tested if x["score"] > -999]
            impact = round(max(scores) - min(scores), 3) if len(scores) >= 2 else 0.0

            best_t  = max(tested, key=lambda x: x["score"])
            worst_t = min(tested, key=lambda x: x["score"])

            # Direction heuristic: compare first and last grid values
            if len(scores) >= 2:
                if scores[0] < scores[-1]:
                    direction = "higher_better"
                elif scores[0] > scores[-1]:
                    direction = "lower_better"
                else:
                    direction = "flat"
            else:
                direction = "flat"

            # Human-readable note
            note = _sensitivity_note(param_name, best_t, worst_t, impact, direction)

            sensitivity[param_name] = {
                "values_tested": tested,
                "best_value":    best_t["value"],
                "worst_value":   worst_t["value"],
                "impact":        impact,
                "direction":     direction,
                "notes":         note,
            }

            logger.info(
                "[OPT] Sensitivity %s: impact=%.2f  best=%s  direction=%s",
                param_name, impact, best_t["value"], direction,
            )

        # Sort by impact descending
        sensitivity = dict(
            sorted(sensitivity.items(), key=lambda x: x[1]["impact"], reverse=True)
        )
        return sensitivity

    # ── Sport-specific optimization ───────────────────────────────────────────

    def run_sport_optimizations(
        self,
        global_best_params: Optional[dict] = None,
    ) -> dict:
        """
        Run per-sport optimizations.

        For each sport found in trade_analytics:
          - If closed trade count ≥ config.OPT_MIN_TRADES_PER_SPORT: run full
            grid/random optimization on that sport's trades only.
          - Otherwise: mark as insufficient_data and recommend global params.

        Also generates a global-vs-sport comparison report.

        global_best_params: the raw param dict from a prior run() call.
          If None, a quick global pass is performed to obtain it.

        Returns dict {sport_label: result_dict}.
        NEVER auto-applies any settings.
        """
        t0         = time.monotonic()
        all_trades = db.get_closed_trade_analytics()

        if not all_trades:
            logger.info("[OPT] No closed trades — skipping sport optimization.")
            return {}

        # ── Obtain global baseline params ────────────────────────────────────
        if global_best_params is None:
            logger.info("[OPT] No global params provided — running quick global pass.")
            global_best_params = self._quick_global_best(all_trades)

        # ── Group closed trades by sport ─────────────────────────────────────
        by_sport: dict[str, list] = {}
        for t in all_trades:
            sport = t.get("sport") or "Generic"
            by_sport.setdefault(sport, []).append(t)

        logger.info(
            "[OPT] Sport breakdown: %s",
            {s: len(ts) for s, ts in sorted(by_sport.items())},
        )

        min_trades = config.OPT_MIN_TRADES_PER_SPORT
        sport_results: dict[str, dict] = {}

        for sport, trades in sorted(by_sport.items()):
            if len(trades) < min_trades:
                logger.info(
                    "[OPT] %s: insufficient data (%d trades, need %d) "
                    "— falling back to global params.",
                    sport, len(trades), min_trades,
                )
                sport_results[sport] = {
                    "status":       "insufficient_data",
                    "sport":        sport,
                    "trade_count":  len(trades),
                    "min_required": min_trades,
                    "fallback":     "global",
                    "params":       global_best_params,
                    "note": (
                        f"Only {len(trades)} closed trades — need {min_trades}+ "
                        f"for a reliable sport-specific estimate. "
                        f"Using global recommended params as fallback."
                    ),
                    "metrics": {},
                }
            else:
                sport_results[sport] = self._run_single_sport(
                    sport, trades, global_best_params
                )

        # ── Comparison: global params vs sport-specific params ────────────────
        comparison = self._run_sport_comparison(by_sport, sport_results, global_best_params)

        # ── Write reports ─────────────────────────────────────────────────────
        reports.write_optimization_results_by_sport(sport_results)
        reports.write_optimization_summary_by_sport(sport_results)
        reports.write_best_parameters_by_sport(sport_results, global_best_params)
        reports.write_sport_config_comparison(comparison)

        logger.info(
            "[OPT] Sport optimization complete in %.1fs.  "
            "Sports: %d (%d optimized, %d insufficient).",
            time.monotonic() - t0,
            len(sport_results),
            sum(1 for r in sport_results.values() if r["status"] == "ok"),
            sum(1 for r in sport_results.values() if r["status"] == "insufficient_data"),
        )
        return sport_results

    def _quick_global_best(self, all_trades: list[dict]) -> dict:
        """
        Run a grid pass over all trades to get global best params.
        Used internally when no global_best_params are provided.
        """
        if len(all_trades) < MIN_SETTLED_FOR_SCORE:
            return {}
        train, test = self._split(all_trades)
        best_score  = -float("inf")
        best_params: dict = {}
        for params in self._grid_param_sets():
            row = self._evaluate(params, train, test, all_trades)
            if row["combined_score"] > best_score:
                best_score  = row["combined_score"]
                best_params = params
        return best_params

    def _run_single_sport(
        self,
        sport:              str,
        trades:             list[dict],
        global_best_params: dict,
    ) -> dict:
        """
        Full optimization for one sport.  Returns a result dict.
        """
        logger.info(
            "[OPT] %s: optimizing on %d closed trades (%s search)...",
            sport, len(trades), self.search_mode,
        )
        train, test  = self._split(trades)
        param_sets   = self._build_param_sets()
        all_results: list[dict] = []
        best_score   = -float("inf")

        for params in param_sets:
            row = self._evaluate(params, train, test, trades)
            all_results.append(row)
            if row["combined_score"] > best_score:
                best_score = row["combined_score"]
                logger.debug(
                    "[OPT] %s: new best score=%.3f edge=%.0f%% conf=%.0f%%",
                    sport, best_score,
                    float(params.get("min_raw_edge", 0)) * 100,
                    float(params.get("min_confidence", 0)) * 100,
                )

        all_results.sort(key=lambda r: r["combined_score"], reverse=True)
        best = all_results[0]

        logger.info(
            "[OPT] %s: best score=%.3f  pnl=$%.2f  win=%.1f%%  trades=%d",
            sport, best["combined_score"], best["test_pnl_dollars"],
            best["test_win_rate_pct"], best["test_trades"],
        )

        return {
            "status":         "ok",
            "sport":          sport,
            "trade_count":    len(trades),
            "combined_score": best["combined_score"],
            "params":         best["params"],
            "metrics": {
                "test_pnl_dollars":   best["test_pnl_dollars"],
                "test_win_rate_pct":  best["test_win_rate_pct"],
                "test_trades":        best["test_trades"],
                "test_avg_slippage":  best["test_avg_slippage"],
                "weaknesses":         best["weaknesses"],
            },
            "top_results": all_results[:5],
        }

    def _run_sport_comparison(
        self,
        by_sport:           dict[str, list],
        sport_results:      dict[str, dict],
        global_params:      dict,
    ) -> list[dict]:
        """
        For every sport with at least 1 closed trade, evaluate:
          1. Performance under global best params
          2. Performance under sport-specific best params
        Returns a list of comparison row dicts (one per sport), sorted by improvement.
        """
        rows: list[dict] = []

        for sport, trades in sorted(by_sport.items()):
            if not trades:
                continue

            # Global config on this sport
            g_result = self.engine.run_on_trades(trades, global_params,
                                                 label=f"global_{sport}")
            g_scored = score_result(g_result, len(trades), global_params)
            gd       = g_result.to_dict()

            # Sport-specific config (may be global fallback if insufficient data)
            sr           = sport_results.get(sport, {})
            sport_params = sr.get("params", global_params)
            s_result     = self.engine.run_on_trades(trades, sport_params,
                                                     label=f"sport_{sport}")
            s_scored     = score_result(s_result, len(trades), sport_params)
            sd           = s_result.to_dict()

            status      = sr.get("status", "unknown")
            improvement = round(s_scored["score"] - g_scored["score"], 3)
            pnl_delta   = round(
                sd["total_pnl_dollars"] - gd["total_pnl_dollars"], 2
            )

            # Recommendation logic
            if status == "insufficient_data":
                recommendation = "insufficient_data"
            elif improvement > 2.0:
                recommendation = "use_sport_specific"
            elif improvement < -1.0:
                recommendation = "use_global"
            else:
                recommendation = "marginal_difference"

            rows.append({
                "sport":                sport,
                "trade_count":          len(trades),
                "status":               status,
                # Global config metrics
                "global_score":         round(g_scored["score"], 3),
                "global_win_rate_pct":  gd["win_rate_pct"],
                "global_pnl_dollars":   gd["total_pnl_dollars"],
                "global_avg_pnl_cents": gd["avg_pnl_cents"],
                "global_avg_slippage":  gd["avg_slippage_cents"],
                # Sport-specific config metrics
                "sport_score":          round(s_scored["score"], 3),
                "sport_win_rate_pct":   sd["win_rate_pct"],
                "sport_pnl_dollars":    sd["total_pnl_dollars"],
                "sport_avg_pnl_cents":  sd["avg_pnl_cents"],
                "sport_avg_slippage":   sd["avg_slippage_cents"],
                # Delta
                "score_improvement":       improvement,
                "pnl_improvement_dollars": pnl_delta,
                "recommendation":          recommendation,
                "weaknesses":              sr.get("metrics", {}).get("weaknesses", ""),
                # Best sport-specific params (key fields only)
                "sport_min_raw_edge":        sport_params.get("min_raw_edge"),
                "sport_min_confidence":      sport_params.get("min_confidence"),
                "sport_max_spread_cents":    sport_params.get("max_spread_cents"),
                "sport_take_profit_cents":   sport_params.get("take_profit_cents"),
                "sport_only_regime":         sport_params.get("only_regime"),
            })

        rows.sort(key=lambda r: r["score_improvement"], reverse=True)
        return rows

    # ── Market-type optimizations ──────────────────────────────────────────────

    def run_market_type_optimizations(
        self,
        global_best_params: dict,
        sport_results:       Optional[dict] = None,
    ) -> dict:
        """
        Run per-(sport, market_type) optimizations.

        For each bucket found in trade_analytics:
          - If closed trade count ≥ config.OPT_MIN_TRADES_PER_MARKET_TYPE: full
            grid/random optimization on that bucket's trades only.
          - Otherwise: fall back to sport-specific params, then global params.

        Writes 4 report files:
          metrics/optimization_results_by_market_type.csv
          metrics/optimization_summary_by_market_type.json
          metrics/best_parameters_by_market_type.json
          metrics/market_type_config_comparison.csv

        Returns dict {"sport:market_type": result_dict}.
        NEVER auto-applies any settings.
        """
        t0         = time.monotonic()
        all_trades = db.get_closed_trade_analytics()

        if not all_trades:
            logger.info("[OPT] No closed trades — skipping market-type optimization.")
            return {}

        if sport_results is None:
            sport_results = {}

        # ── Group closed trades by (sport, market_type) ───────────────────────
        by_bucket: dict[str, list] = {}
        for t in all_trades:
            sport = t.get("sport")       or "Generic"
            mtype = t.get("market_type") or "misc"
            key   = f"{sport}:{mtype}"
            by_bucket.setdefault(key, []).append(t)

        logger.info(
            "[OPT] Market-type bucket breakdown: %s",
            {k: len(v) for k, v in sorted(by_bucket.items())},
        )

        min_trades = config.OPT_MIN_TRADES_PER_MARKET_TYPE
        mt_results: dict[str, dict] = {}

        for key, trades in sorted(by_bucket.items()):
            sport, mtype = key.split(":", 1)

            if len(trades) < min_trades:
                # Determine fallback: sport-specific > global
                sr = sport_results.get(sport, {})
                if sr.get("status") == "ok":
                    fallback_params = sr["params"]
                    fallback        = "sport"
                    fallback_reason = (
                        f"Only {len(trades)} trades for {key}; "
                        f"need {min_trades}+. Using sport-specific params."
                    )
                else:
                    fallback_params = global_best_params
                    fallback        = "global"
                    fallback_reason = (
                        f"Only {len(trades)} trades for {key}; "
                        f"need {min_trades}+. No sport params available. "
                        f"Using global params."
                    )
                logger.info(
                    "[OPT] %s: insufficient data (%d trades, need %d) "
                    "— fallback=%s.",
                    key, len(trades), min_trades, fallback,
                )
                mt_results[key] = {
                    "status":         "insufficient_data",
                    "sport":          sport,
                    "market_type":    mtype,
                    "trade_count":    len(trades),
                    "min_required":   min_trades,
                    "fallback":       fallback,
                    "fallback_reason": fallback_reason,
                    "params":         fallback_params,
                    "metrics":        {},
                }
            else:
                mt_results[key] = self._run_single_bucket(
                    sport, mtype, trades, global_best_params
                )

        # ── Comparison: global vs sport vs market-type ────────────────────────
        comparison = self._run_market_type_comparison(
            by_bucket, mt_results, global_best_params, sport_results
        )

        # ── Write reports ─────────────────────────────────────────────────────
        reports.write_optimization_results_by_market_type(mt_results)
        reports.write_optimization_summary_by_market_type(mt_results)
        reports.write_best_parameters_by_market_type(mt_results, global_best_params)
        reports.write_market_type_config_comparison(comparison)

        n_ok   = sum(1 for r in mt_results.values() if r["status"] == "ok")
        n_fall = sum(1 for r in mt_results.values() if r["status"] == "insufficient_data")
        logger.info(
            "[OPT] Market-type optimization complete in %.1fs.  "
            "Buckets: %d (%d optimized, %d fallback).",
            time.monotonic() - t0, len(mt_results), n_ok, n_fall,
        )
        return mt_results

    def _run_single_bucket(
        self,
        sport:              str,
        market_type:        str,
        trades:             list[dict],
        global_best_params: dict,
    ) -> dict:
        """
        Full optimization for one (sport, market_type) bucket.
        Mirrors _run_single_sport but labels logs with sport:market_type.
        """
        key = f"{sport}:{market_type}"
        logger.info(
            "[OPT] %s: optimizing on %d closed trades (%s search)...",
            key, len(trades), self.search_mode,
        )
        train, test  = self._split(trades)
        param_sets   = self._build_param_sets()
        all_results: list[dict] = []
        best_score   = -float("inf")

        for params in param_sets:
            row = self._evaluate(params, train, test, trades)
            all_results.append(row)
            if row["combined_score"] > best_score:
                best_score = row["combined_score"]
                logger.debug(
                    "[OPT] %s: new best score=%.3f edge=%.0f%% conf=%.0f%%",
                    key, best_score,
                    float(params.get("min_raw_edge", 0)) * 100,
                    float(params.get("min_confidence", 0)) * 100,
                )

        all_results.sort(key=lambda r: r["combined_score"], reverse=True)
        best = all_results[0]

        logger.info(
            "[OPT] %s: best score=%.3f  pnl=$%.2f  win=%.1f%%  trades=%d",
            key, best["combined_score"], best["test_pnl_dollars"],
            best["test_win_rate_pct"], best["test_trades"],
        )
        return {
            "status":         "ok",
            "sport":          sport,
            "market_type":    market_type,
            "trade_count":    len(trades),
            "combined_score": best["combined_score"],
            "params":         best["params"],
            "metrics": {
                "test_pnl_dollars":  best["test_pnl_dollars"],
                "test_win_rate_pct": best["test_win_rate_pct"],
                "test_trades":       best["test_trades"],
                "test_avg_slippage": best["test_avg_slippage"],
                "weaknesses":        best["weaknesses"],
            },
            "top_results": all_results[:5],
        }

    def _run_market_type_comparison(
        self,
        by_bucket:     dict[str, list],
        mt_results:    dict[str, dict],
        global_params: dict,
        sport_results: dict,
    ) -> list[dict]:
        """
        For every (sport, market_type) bucket, evaluate performance under:
          1. global params
          2. sport-specific params (from sport_results)
          3. market-type-specific params (from mt_results)

        Returns sorted list of comparison row dicts.
        """
        rows: list[dict] = []

        for key, trades in sorted(by_bucket.items()):
            if not trades:
                continue

            sport, mtype = key.split(":", 1)
            sr           = sport_results.get(sport, {})
            mr           = mt_results.get(key,   {})

            sport_params = sr.get("params", global_params)
            mt_params    = mr.get("params", sport_params)

            # Evaluate all three configs on this bucket
            g_result = self.engine.run_on_trades(trades, global_params,  label=f"global_{key}")
            s_result = self.engine.run_on_trades(trades, sport_params,   label=f"sport_{key}")
            m_result = self.engine.run_on_trades(trades, mt_params,      label=f"mt_{key}")

            g_scored = score_result(g_result, len(trades), global_params)
            s_scored = score_result(s_result, len(trades), sport_params)
            m_scored = score_result(m_result, len(trades), mt_params)

            gd = g_result.to_dict()
            sd = s_result.to_dict()
            md = m_result.to_dict()

            mt_status  = mr.get("status", "unknown")
            sport_status = sr.get("status", "unknown")

            # Best overall score among the three
            best_score  = max(g_scored["score"], s_scored["score"], m_scored["score"])
            if m_scored["score"] == best_score and mt_status == "ok":
                recommendation = "use_market_type"
            elif s_scored["score"] == best_score and sport_status == "ok":
                recommendation = "use_sport"
            else:
                recommendation = "use_global"

            if mt_status == "insufficient_data":
                recommendation = "insufficient_data"

            rows.append({
                "sport":                   sport,
                "market_type":             mtype,
                "bucket":                  key,
                "trade_count":             len(trades),
                "mt_status":               mt_status,
                "fallback_reason":         mr.get("fallback_reason", ""),
                # Global config
                "global_score":            round(g_scored["score"], 3),
                "global_win_rate_pct":     gd["win_rate_pct"],
                "global_pnl_dollars":      gd["total_pnl_dollars"],
                "global_avg_slippage":     gd["avg_slippage_cents"],
                # Sport-specific config
                "sport_score":             round(s_scored["score"], 3),
                "sport_win_rate_pct":      sd["win_rate_pct"],
                "sport_pnl_dollars":       sd["total_pnl_dollars"],
                "sport_avg_slippage":      sd["avg_slippage_cents"],
                # Market-type config
                "mt_score":                round(m_scored["score"], 3),
                "mt_win_rate_pct":         md["win_rate_pct"],
                "mt_pnl_dollars":          md["total_pnl_dollars"],
                "mt_avg_slippage":         md["avg_slippage_cents"],
                # Deltas vs global
                "sport_score_delta":       round(s_scored["score"] - g_scored["score"], 3),
                "mt_score_delta":          round(m_scored["score"] - g_scored["score"], 3),
                "mt_pnl_delta":            round(md["total_pnl_dollars"] - gd["total_pnl_dollars"], 2),
                "recommendation":          recommendation,
                # Best market-type params
                "mt_min_raw_edge":         mt_params.get("min_raw_edge"),
                "mt_min_confidence":       mt_params.get("min_confidence"),
                "mt_max_spread_cents":     mt_params.get("max_spread_cents"),
                "mt_take_profit_cents":    mt_params.get("take_profit_cents"),
                "mt_only_regime":          mt_params.get("only_regime"),
            })

        rows.sort(key=lambda r: r["mt_score_delta"], reverse=True)
        return rows


# ── Sensitivity note generator ────────────────────────────────────────────────

def run_standalone(
    search_mode:    str  = "grid",
    random_samples: int  = 150,
    run_sports:     bool = True,
):
    """
    Entrypoint for running the optimizer as a standalone script:
        cd kalshi-bot && python3 optimizer.py
        cd kalshi-bot && python3 optimizer.py --mode random --samples 200
        cd kalshi-bot && python3 optimizer.py --no-sports
    Results are written to metrics/ and logged to stdout.
    NEVER modifies live trading state.
    """
    import db as _db
    _db.init_db()

    opt = ParameterOptimizer(
        search_mode=search_mode,
        random_samples=random_samples,
    )

    # Global optimization
    summary = opt.run()
    if summary.get("status") == "ok":
        print(f"\n── Global Optimization ──────────────────────────────────")
        print(f"  Combinations evaluated: {summary['combinations']}")
        print(f"  Best combined score:    {summary['best_score']:.3f}")
        print(f"  Elapsed:                {summary['elapsed_s']}s")
        print(f"\n  Best parameters:")
        for k, v in summary["best_params"].items():
            print(f"    {k}: {v}")
        print(f"\n  Top sensitivity drivers:")
        for param, s in list(summary["sensitivity"].items())[:5]:
            print(f"    {param}: impact={s['impact']:.2f}  "
                  f"best={s['best_value']}  direction={s['direction']}")
            print(f"    → {s['notes'][:100]}")
    else:
        print(f"Global optimization skipped: {summary}")

    # Sport-specific optimization
    sport_results: dict = {}
    if run_sports:
        global_params = summary.get("best_params", {})
        print(f"\n── Sport-Specific Optimization ──────────────────────────")
        sport_results = opt.run_sport_optimizations(global_best_params=global_params)
        for sport, sr in sorted(sport_results.items()):
            if sr.get("status") == "ok":
                m = sr.get("metrics", {})
                print(f"  {sport}: score={sr.get('combined_score', '?'):.3f}  "
                      f"pnl=${m.get('test_pnl_dollars', '?')}  "
                      f"win={m.get('test_win_rate_pct', '?')}%  "
                      f"trades={sr.get('trade_count', '?')}")
            else:
                print(f"  {sport}: insufficient data ({sr.get('trade_count', 0)} trades)")

    # Market-type optimization
    if run_sports:
        global_params = summary.get("best_params", {})
        print(f"\n── Market-Type Optimization ─────────────────────────────")
        mt_results = opt.run_market_type_optimizations(
            global_best_params=global_params,
            sport_results=sport_results,
        )
        for key, mr in sorted(mt_results.items()):
            if mr.get("status") == "ok":
                m = mr.get("metrics", {})
                print(f"  {key}: score={mr.get('combined_score', '?'):.3f}  "
                      f"pnl=${m.get('test_pnl_dollars', '?')}  "
                      f"win={m.get('test_win_rate_pct', '?')}%  "
                      f"trades={mr.get('trade_count', '?')}")
            else:
                fb = mr.get("fallback", "global")
                print(f"  {key}: insufficient data ({mr.get('trade_count', 0)} trades, "
                      f"fallback={fb})")

    print(f"\nAll reports written to kalshi-bot/metrics/")


def _sensitivity_note(
    param: str,
    best: dict,
    worst: dict,
    impact: float,
    direction: str,
) -> str:
    """Generate a plain-English interpretation of sensitivity results."""
    if impact < 1.0:
        return f"{param}: low sensitivity (score range {impact:.1f}) — this parameter has little effect on outcomes."

    bv = best["value"]
    wv = worst["value"]
    bd = best["pnl"]
    wd = worst["pnl"]

    if param == "min_raw_edge":
        bv_str = f"{float(bv)*100:.0f}%" if bv is not None else "none"
        wv_str = f"{float(wv)*100:.0f}%" if wv is not None else "none"
        return (
            f"min_raw_edge: setting to {bv_str} scored best (pnl=${bd:.2f}).  "
            f"Setting to {wv_str} scored worst (pnl=${wd:.2f}).  "
            f"Impact score range: {impact:.1f}."
        )
    if param == "min_confidence":
        bv_str = f"{float(bv)*100:.0f}%" if bv is not None else "none"
        return (
            f"min_confidence: {bv_str} scored best (pnl=${bd:.2f}).  "
            f"Direction: {direction}.  Impact: {impact:.1f}."
        )
    if param == "max_spread_cents":
        return (
            f"max_spread_cents: {bv}¢ scored best (pnl=${bd:.2f}).  "
            f"{wv}¢ scored worst.  "
            + ("Tightening the spread filter too much may hurt fill rate."
               if direction == "lower_better" else
               "Wider spread tolerance increases fills but costs PnL.")
        )
    if param == "take_profit_cents":
        bv_str = f"{bv}¢" if bv is not None else "hold-to-settle"
        return (
            f"take_profit_cents: {bv_str} scored best (pnl=${bd:.2f}).  "
            f"Impact: {impact:.1f}.  "
            + ("An early exit target helps lock in gains." if bv is not None else
               "Holding to settlement outperformed early exits.")
        )
    if param == "slippage_buffer_cents":
        return (
            f"slippage_buffer_cents: {bv}¢ scored best.  "
            f"Higher buffers reduce fill success rate; lower buffers expose to fill risk."
        )
    if param == "aggressive_edge_threshold":
        bv_str = f"{float(bv)*100:.0f}%" if bv is not None else "?"
        return (
            f"aggressive_edge_threshold: {bv_str} scored best.  "
            f"Above this edge, simulating aggressive fills changed PnL by ${bd - wd:.2f}.  "
            f"Impact: {impact:.1f}."
        )
    if param == "only_regime":
        bv_str = str(bv) if bv else "all regimes"
        return (
            f"only_regime: restricting to '{bv_str}' scored best (pnl=${bd:.2f}).  "
            f"Impact: {impact:.1f}.  "
            + ("Regime filtering concentrates on stronger setups but reduces breadth."
               if bv else
               "No regime filter scored best — broader entry is preferable.")
        )
    return f"{param}: best={bv} (pnl=${bd:.2f}), worst={wv} (pnl=${wd:.2f}), impact={impact:.1f}."


if __name__ == "__main__":
    import argparse
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
    )
    parser = argparse.ArgumentParser(description="Kalshi bot parameter optimizer")
    parser.add_argument("--mode",      default="grid", choices=["grid", "random"],
                        help="Search mode: grid (default) or random")
    parser.add_argument("--samples",   default=150, type=int,
                        help="Random search sample count (only for --mode random)")
    parser.add_argument("--no-sports", action="store_true",
                        help="Skip per-sport optimization (global only)")
    args = parser.parse_args()
    run_standalone(
        search_mode=args.mode,
        random_samples=args.samples,
        run_sports=not args.no_sports,
    )
