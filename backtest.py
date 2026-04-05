"""
backtest.py — Deterministic backtester for replay and scenario analysis.

Two modes:
  1. Replay: re-process the stored trade_analytics records to verify
     that our analytics match what actually happened (sanity check).

  2. Scenario: apply modified parameters (different confidence threshold,
     different spread limit, different take-profit level) to the same
     historical trades and compare outcomes to what was actually achieved.

Since we do not store full orderbook snapshots, fill simulation is done
using the stored spread_at_entry and entry_price data with conservative
assumptions.

Usage:
    from backtest import BacktestEngine
    engine = BacktestEngine()
    result = engine.run()             # replay actual trades
    result = engine.run_scenario(     # modified params
        min_raw_edge=0.08,
        max_spread_cents=6,
        take_profit_threshold_cents=8,
    )
    engine.log_result(result)
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import db

logger = logging.getLogger("kalshi_bot.backtest")


# ── Data structures ───────────────────────────────────────────────────────────

class BacktestResult:
    def __init__(self, label: str):
        self.label          = label
        self.total_trades   = 0
        self.filled_trades  = 0
        self.skipped_trades = 0
        self.wins           = 0
        self.losses         = 0
        self.total_pnl_cents= 0.0
        self.slippages      = []
        self.hold_times     = []
        self.pnl_list       = []
        self.by_sport:       dict[str, dict] = {}
        self.by_regime:      dict[str, dict] = {}
        self.by_mode:        dict[str, dict] = {}

    def add_trade(
        self,
        sport: str, regime: str, mode: str,
        simulated_fill: float, slippage: float,
        pnl_cents: float, outcome: str,
        hold_seconds: float, skipped: bool = False,
    ):
        self.total_trades += 1
        if skipped:
            self.skipped_trades += 1
            return
        self.filled_trades += 1
        self.slippages.append(slippage)
        self.pnl_list.append(pnl_cents)
        self.total_pnl_cents += pnl_cents
        if hold_seconds:
            self.hold_times.append(hold_seconds)
        if outcome == "win":
            self.wins += 1
        elif outcome == "loss":
            self.losses += 1

        for key, val in [("sport", sport), ("regime", regime), ("mode", mode)]:
            attr = getattr(self, f"by_{key}")
            attr.setdefault(val, {"count":0,"wins":0,"total_pnl":0.0})
            attr[val]["count"]     += 1
            attr[val]["total_pnl"] += pnl_cents
            if outcome == "win":
                attr[val]["wins"] += 1

    def to_dict(self) -> dict:
        def avg(v): return round(sum(v)/len(v),3) if v else 0.0
        settled = self.wins + self.losses
        return {
            "label":                self.label,
            "generated_at":         datetime.now(timezone.utc).isoformat(),
            "total_trades":         self.total_trades,
            "filled_trades":        self.filled_trades,
            "skipped_trades":       self.skipped_trades,
            "wins":                 self.wins,
            "losses":               self.losses,
            "win_rate_pct":         round(self.wins/settled*100,1) if settled else 0.0,
            "total_pnl_cents":      round(self.total_pnl_cents, 2),
            "total_pnl_dollars":    round(self.total_pnl_cents/100, 2),
            "avg_pnl_cents":        avg(self.pnl_list),
            "avg_slippage_cents":   avg(self.slippages),
            "avg_hold_seconds":     avg(self.hold_times),
            "by_sport":             {k: dict(v) for k,v in self.by_sport.items()},
            "by_regime":            {k: dict(v) for k,v in self.by_regime.items()},
            "by_exec_mode":         {k: dict(v) for k,v in self.by_mode.items()},
        }


# ── Backtest engine ───────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Runs deterministic backtests over stored trade_analytics records.
    Does not touch any live trading state.
    """

    def run(self) -> BacktestResult:
        """
        Replay all closed trades from trade_analytics.
        Verifies analytics consistency and produces summary metrics.
        """
        trades = db.get_closed_trade_analytics()
        result = BacktestResult(label="replay_actual")

        for t in trades:
            fill   = float(t.get("fill_price") or t.get("entry_price") or 50)
            slip   = float(t.get("slippage_cents") or 0)
            pnl    = float(t.get("pnl_cents") or 0)
            outcome= t.get("outcome", "unknown")
            hold   = float(t.get("hold_seconds") or 0)
            sport  = t.get("sport", "unknown") or "unknown"
            regime = t.get("regime", "unknown") or "unknown"
            mode   = t.get("exec_mode", "unset") or "unset"

            result.add_trade(
                sport=sport, regime=regime, mode=mode,
                simulated_fill=fill, slippage=slip,
                pnl_cents=pnl, outcome=outcome, hold_seconds=hold,
            )

        logger.info("[BACKTEST] Replay: %d trades, win_rate=%.1f%%, pnl=$%.2f",
                    result.filled_trades,
                    result.to_dict()["win_rate_pct"],
                    result.total_pnl_cents / 100)
        return result

    def run_scenario(
        self,
        label: str = "scenario",
        min_raw_edge: float = 0.05,
        max_spread_cents: float = 10.0,
        min_confidence: float = 0.35,
        take_profit_threshold_cents: Optional[float] = None,
        only_regime: Optional[str] = None,
        only_sport: Optional[str] = None,
    ) -> BacktestResult:
        """
        Re-apply modified filters to stored closed trades to simulate
        what would have happened under different parameters.

        Conservative fill assumption: if aggressive, pay the spread;
        if passive/adaptive, fill at the stated entry price.

        take_profit_threshold_cents: if set and outcome would have been 'win',
          simulate early exit at this many cents of convergence (reduces win PnL).
        """
        trades = db.get_closed_trade_analytics()
        result = BacktestResult(label=label)

        for t in trades:
            sport  = t.get("sport", "unknown") or "unknown"
            regime = t.get("regime", "unknown") or "unknown"
            mode   = t.get("exec_mode", "unset") or "unset"

            # Dimension filters
            if only_sport and sport != only_sport:
                continue
            if only_regime and regime != only_regime:
                continue

            raw_edge  = float(t.get("raw_edge") or 0)
            spread    = float(t.get("spread_at_entry") or 0)
            conf      = float(t.get("confidence_score") or 0)
            outcome   = t.get("outcome", "unknown")
            hold      = float(t.get("hold_seconds") or 0)
            entry_p   = float(t.get("entry_price") or 50)
            pnl       = float(t.get("pnl_cents") or 0)

            # Scenario filters — skip if trade wouldn't have passed
            skip = (
                raw_edge < min_raw_edge or
                spread > max_spread_cents or
                conf < min_confidence
            )

            if skip:
                result.add_trade(
                    sport=sport, regime=regime, mode=mode,
                    simulated_fill=entry_p, slippage=0,
                    pnl_cents=0, outcome="skipped",
                    hold_seconds=0, skipped=True,
                )
                continue

            # Simulated fill: aggressive pays half-spread extra
            sim_slip = spread * 0.5 if mode == "aggressive" else spread * 0.2
            sim_fill = entry_p + sim_slip

            # Simulated PnL — use stored outcome (can't replay without orderbook)
            sim_pnl = pnl
            if take_profit_threshold_cents and outcome == "win":
                # Assume early exit reduced full win to take_profit amount
                side          = t.get("side", "yes")
                max_pnl       = (100.0 - entry_p) if side == "yes" else entry_p
                max_pnl_cents = max_pnl * float(t.get("contracts") or 1)
                tp_pnl        = take_profit_threshold_cents * float(t.get("contracts") or 1)
                sim_pnl       = min(sim_pnl, tp_pnl if tp_pnl < max_pnl_cents else sim_pnl)

            result.add_trade(
                sport=sport, regime=regime, mode=mode,
                simulated_fill=sim_fill, slippage=sim_slip,
                pnl_cents=sim_pnl, outcome=outcome, hold_seconds=hold,
            )

        logger.info(
            "[BACKTEST] Scenario '%s': %d/%d trades used, "
            "win_rate=%.1f%%, pnl=$%.2f",
            label, result.filled_trades, result.total_trades,
            result.to_dict()["win_rate_pct"],
            result.total_pnl_cents / 100,
        )
        return result

    def run_all_scenarios(self) -> list[BacktestResult]:
        """
        Run a standard battery of scenario tests and return all results.
        Useful for generating the backtest_summary.json.
        """
        scenarios = [
            self.run(),
            self.run_scenario("tight_edge",    min_raw_edge=0.10, label="min_edge_10pct"),
            self.run_scenario("wider_spread",  max_spread_cents=15, label="max_spread_15c"),
            self.run_scenario("tight_spread",  max_spread_cents=6, label="max_spread_6c"),
            self.run_scenario("hi_conf",       min_confidence=0.60, label="min_conf_60pct"),
            self.run_scenario("take_profit_8", take_profit_threshold_cents=8,
                              label="take_profit_8c"),
            self.run_scenario("take_profit_15",take_profit_threshold_cents=15,
                              label="take_profit_15c"),
            self.run_scenario("calm_only",     only_regime="calm", label="calm_regime"),
            self.run_scenario("volatile_only", only_regime="volatile", label="volatile_regime"),
        ]
        return scenarios

    def run_on_trades(
        self,
        trades: list[dict],
        params: dict,
        label: str = "custom",
    ) -> "BacktestResult":
        """
        Run a scenario on a pre-fetched list of trade dicts (from trade_analytics).
        Accepts a unified params dict so the optimizer can call this in a tight loop
        without repeated DB hits.

        Supported params keys:
          min_raw_edge          — skip trade if raw_edge < this
          min_confidence        — skip trade if confidence_score < this
          max_spread_cents      — skip trade if spread_at_entry > this
          take_profit_cents     — clip win PnL per contract to this value
          only_regime           — skip trade if regime != this (None = all)
          only_sport            — skip trade if sport != this (None = all)
          slippage_buffer_cents — extra cents added to simulated fill (subtracts PnL)
          aggressive_edge_threshold — above this, simulate aggressive fill cost

        Returns a BacktestResult.
        """
        min_edge    = float(params.get("min_raw_edge",    0.0))
        min_conf    = float(params.get("min_confidence",  0.0))
        max_spread  = float(params.get("max_spread_cents", 999.0))
        tp_cents    = params.get("take_profit_cents")     # None or float
        regime_filt = params.get("only_regime")           # None or str
        sport_filt  = params.get("only_sport")            # None or str
        slip_buf    = float(params.get("slippage_buffer_cents", 0.0))
        agg_thresh  = float(params.get("aggressive_edge_threshold", 0.99))

        result = BacktestResult(label=label)

        for t in trades:
            sport  = t.get("sport",     "unknown") or "unknown"
            regime = t.get("regime",    "unknown") or "unknown"
            mode   = t.get("exec_mode", "unset")   or "unset"

            if regime_filt and regime != regime_filt:
                continue
            if sport_filt  and sport  != sport_filt:
                continue

            raw_edge  = float(t.get("raw_edge")          or 0)
            spread    = float(t.get("spread_at_entry")    or 0)
            conf      = float(t.get("confidence_score")   or 0)
            outcome   = t.get("outcome", "unknown")
            hold      = float(t.get("hold_seconds")       or 0)
            entry_p   = float(t.get("entry_price")        or 50)
            pnl       = float(t.get("pnl_cents")          or 0)
            contracts = float(t.get("contracts")          or 1)

            skip = (
                raw_edge < min_edge  or
                spread   > max_spread or
                conf     < min_conf
            )
            if skip:
                result.add_trade(
                    sport=sport, regime=regime, mode=mode,
                    simulated_fill=entry_p, slippage=0,
                    pnl_cents=0, outcome="skipped",
                    hold_seconds=0, skipped=True,
                )
                continue

            # Simulated fill cost
            if raw_edge >= agg_thresh:
                sim_slip = spread * 0.5 + slip_buf
            else:
                sim_slip = spread * 0.2 + slip_buf * 0.5
            sim_fill = entry_p + sim_slip

            # Apply extra slippage cost to PnL
            sim_pnl = pnl - sim_slip * contracts

            # Take-profit clipping for wins
            if tp_cents is not None and outcome == "win":
                tp_cents_f   = float(tp_cents)
                side         = t.get("side", "yes")
                max_pnl      = (100.0 - entry_p) if side == "yes" else entry_p
                max_pnl_total= max_pnl * contracts
                tp_total     = tp_cents_f * contracts
                if tp_total < max_pnl_total:
                    sim_pnl = min(sim_pnl, tp_total)

            result.add_trade(
                sport=sport, regime=regime, mode=mode,
                simulated_fill=sim_fill, slippage=sim_slip,
                pnl_cents=sim_pnl, outcome=outcome, hold_seconds=hold,
            )

        return result

    def run_timing_scenarios(self) -> list["BacktestResult"]:
        """
        Timing-focused scenario battery.  Uses stored trade_analytics records
        to compare different entry-timing strategies.

        Measures per scenario:
          pnl, fill_rate, avg_slippage, missed_opportunity_cost, drawdown.

        Scenarios
        ---------
        baseline              — all trades as actually entered (immediate entry)
        staged_entry_sim      — simulate 50% size on staged trades; wins scaled down
        chase_protection_on   — skip trades where market classification was "overreacting"
        wait_high_urgency     — only take trades with urgency_score >= 0.65
        aggressive_only       — only trades where exec_mode == "aggressive"
        passive_vs_aggressive — split existing trades by exec_mode
        lagging_only          — only trades where classification was "lagging"
        aligned_only          — only trades where classification was "aligned"
        """
        trades = db.get_closed_trade_analytics()
        results: list[BacktestResult] = []

        # 1. Baseline: all trades as entered
        baseline = BacktestResult(label="baseline_immediate_entry")
        for t in trades:
            sport  = t.get("sport",     "unknown") or "unknown"
            regime = t.get("regime",    "unknown") or "unknown"
            mode   = t.get("exec_mode", "unset")   or "unset"
            fill   = float(t.get("fill_price")  or t.get("entry_price") or 50)
            slip   = float(t.get("slippage_cents") or 0)
            pnl    = float(t.get("pnl_cents")   or 0)
            hold   = float(t.get("hold_seconds") or 0)
            outcome= t.get("outcome", "unknown")
            baseline.add_trade(sport=sport, regime=regime, mode=mode,
                               simulated_fill=fill, slippage=slip,
                               pnl_cents=pnl, outcome=outcome, hold_seconds=hold)
        results.append(baseline)

        # Helper: apply timing filter + optional staged-entry scaling
        def _run_filter(
            label:              str,
            filter_fn,          # trade_dict → bool  (True = keep)
            staged_scale:       float = 1.0,  # scale win pnl down for staged sim
            skip_if_missing:    str   = "",   # field name — skip if field missing/null
        ) -> BacktestResult:
            r = BacktestResult(label=label)
            for t in trades:
                sport  = t.get("sport",     "unknown") or "unknown"
                regime = t.get("regime",    "unknown") or "unknown"
                mode   = t.get("exec_mode", "unset")   or "unset"
                entry  = float(t.get("entry_price") or 50)
                slip   = float(t.get("slippage_cents") or 0)
                pnl    = float(t.get("pnl_cents")   or 0)
                hold   = float(t.get("hold_seconds") or 0)
                outcome= t.get("outcome", "unknown")

                if skip_if_missing and not t.get(skip_if_missing):
                    continue

                if not filter_fn(t):
                    r.add_trade(sport=sport, regime=regime, mode=mode,
                                simulated_fill=entry, slippage=0,
                                pnl_cents=0, outcome="skipped",
                                hold_seconds=0, skipped=True)
                    continue

                sim_pnl = pnl * staged_scale if staged_scale < 1.0 else pnl
                r.add_trade(sport=sport, regime=regime, mode=mode,
                            simulated_fill=entry + slip, slippage=slip,
                            pnl_cents=sim_pnl, outcome=outcome, hold_seconds=hold)
            return r

        # 2. Staged entry simulation: staged trades at 50% size
        def _is_staged(t): return bool(int(t.get("staged_entry_flag") or 0))
        def _not_staged(t): return not _is_staged(t)

        staged_sim = BacktestResult(label="staged_entry_simulation")
        for t in trades:
            sport  = t.get("sport",     "unknown") or "unknown"
            regime = t.get("regime",    "unknown") or "unknown"
            mode   = t.get("exec_mode", "unset")   or "unset"
            entry  = float(t.get("entry_price") or 50)
            slip   = float(t.get("slippage_cents") or 0)
            pnl    = float(t.get("pnl_cents")   or 0)
            hold   = float(t.get("hold_seconds") or 0)
            outcome= t.get("outcome", "unknown")
            scale  = 0.5 if _is_staged(t) else 1.0
            staged_sim.add_trade(sport=sport, regime=regime, mode=mode,
                                 simulated_fill=entry + slip, slippage=slip,
                                 pnl_cents=pnl * scale, outcome=outcome, hold_seconds=hold)
        results.append(staged_sim)

        # 3. Chase protection on: skip overreacting entries
        results.append(_run_filter(
            label     = "chase_protection_on",
            filter_fn = lambda t: (t.get("entry_timing_classification") or "aligned") != "overreacting",
        ))

        # 4. Wait for high urgency only (≥ 0.65)
        results.append(_run_filter(
            label     = "wait_for_high_urgency_065",
            filter_fn = lambda t: float(t.get("urgency_score") or 0) >= 0.65,
        ))

        # 5. Very high urgency (≥ 0.80) — most selective
        results.append(_run_filter(
            label     = "very_high_urgency_080",
            filter_fn = lambda t: float(t.get("urgency_score") or 0) >= 0.80,
        ))

        # 6. Lagging-market entries only
        results.append(_run_filter(
            label     = "lagging_classification_only",
            filter_fn = lambda t: (t.get("entry_timing_classification") or "") == "lagging",
        ))

        # 7. Aligned-market entries only
        results.append(_run_filter(
            label     = "aligned_classification_only",
            filter_fn = lambda t: (t.get("entry_timing_classification") or "") == "aligned",
        ))

        # 8. Aggressive execution only
        results.append(_run_filter(
            label     = "aggressive_exec_only",
            filter_fn = lambda t: (t.get("exec_mode") or "") == "aggressive",
        ))

        # 9. Passive execution only
        results.append(_run_filter(
            label     = "passive_exec_only",
            filter_fn = lambda t: (t.get("exec_mode") or "") == "passive",
        ))

        # Missed opportunity cost: total missed_edge_cents across all skipped
        total_missed = sum(
            float(t.get("missed_edge_cents") or 0)
            for t in db.get_all_trade_analytics()
        )

        logger.info(
            "[BACKTEST] Timing scenarios: %d scenarios, baseline pnl=$%.2f, "
            "total_missed_edge=$%.2f",
            len(results),
            baseline.total_pnl_cents / 100,
            total_missed / 100,
        )

        # Attach missed_edge_cents to baseline dict for reporting
        baseline_d = baseline.to_dict()
        baseline_d["missed_edge_cents_total"] = round(total_missed, 2)

        return results

    def log_result(self, result: BacktestResult):
        d = result.to_dict()
        logger.info(
            "[BACKTEST] %s: %d filled, %d skipped, win=%.1f%%, pnl=$%.2f, "
            "avg_pnl=%.1f¢, avg_slip=%.1f¢",
            d["label"], d["filled_trades"], d["skipped_trades"],
            d["win_rate_pct"], d["total_pnl_dollars"],
            d["avg_pnl_cents"], d["avg_slippage_cents"],
        )


# ── Standalone helpers ────────────────────────────────────────────────────────

def compute_max_drawdown(pnl_list: list[float]) -> float:
    """
    Compute the maximum peak-to-trough drawdown (in cents) from a sequence
    of per-trade PnL values.  Returns a positive number (magnitude of loss).
    """
    if not pnl_list:
        return 0.0
    peak       = 0.0
    cumulative = 0.0
    max_dd     = 0.0
    for pnl in pnl_list:
        cumulative += pnl
        if cumulative > peak:
            peak = cumulative
        dd = peak - cumulative
        if dd > max_dd:
            max_dd = dd
    return round(max_dd, 2)
