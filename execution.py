"""
execution.py — Orderbook-aware execution intelligence layer.

Sits between the scanner (which finds edge) and order_manager
(which places orders). Decides HOW to enter, not whether to.

Responsibilities:
  • Microstructure analysis  — spread, liquidity, depth
  • Market regime detection  — calm / trending / volatile / illiquid
  • Execution mode selection — aggressive / passive / adaptive
  • Entry price optimisation — best limit price for the chosen mode
  • Slippage estimation      — pre-trade cost model
  • Edge-after-slippage gate — do not trade if slippage kills the edge
  • Entry timing guard       — avoid chasing spikes, detect market lag
  • Order cooldowns          — per-ticker quiet periods after events
  • Exit signal generation   — take-profit / model-flip / hold-to-settle
  • Execution audit logging  — one structured line per decision

Does NOT place orders.  Does NOT touch DB.  Does NOT modify risk rules.
"""

import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import config

logger = logging.getLogger("kalshi_bot.execution")

# ── Tunables (all env-overridable) ────────────────────────────────────────────

MAX_SPREAD_CENTS          = int(os.environ.get("MAX_SPREAD_CENTS",           "10"))
MIN_DEPTH_CONTRACTS       = int(os.environ.get("MIN_DEPTH_CONTRACTS",         "1"))
SLIPPAGE_BUFFER_CENTS     = float(os.environ.get("SLIPPAGE_BUFFER_CENTS",    "1.5"))
AGGRESSIVE_EDGE_THRESHOLD = float(os.environ.get("AGGRESSIVE_EDGE_THRESHOLD","0.12"))
PASSIVE_IMPROVEMENT_CENTS = int(os.environ.get("PASSIVE_IMPROVEMENT_CENTS",  "2"))
TRENDING_MOMENTUM_THRESH  = float(os.environ.get("TRENDING_MOMENTUM_THRESH", "0.06"))
ILLIQUID_SPREAD_THRESH    = int(os.environ.get("ILLIQUID_SPREAD_THRESH",     "15"))
COOLDOWN_AFTER_ENTRY_SECS = int(os.environ.get("COOLDOWN_AFTER_ENTRY_SECS", "120"))
COOLDOWN_AFTER_CANCEL_SECS= int(os.environ.get("COOLDOWN_AFTER_CANCEL_SECS","60"))
COOLDOWN_AFTER_LOSS_SECS  = int(os.environ.get("COOLDOWN_AFTER_LOSS_SECS",  "300"))
EXIT_MODE                 = os.environ.get("EXIT_MODE", "settle")
TAKE_PROFIT_CENTS         = int(os.environ.get("TAKE_PROFIT_CENTS",          "12"))
MAX_HOLD_SECONDS          = int(os.environ.get("MAX_HOLD_SECONDS",           "0"))
SPIKE_LOOKBACK_CENTS      = int(os.environ.get("SPIKE_LOOKBACK_CENTS",       "8"))
MIN_EDGE_AFTER_SLIP       = float(os.environ.get("MIN_EDGE_AFTER_SLIP",      "0.03"))
EDGE_DISAPPEAR_THRESHOLD  = float(os.environ.get("EDGE_DISAPPEAR_THRESHOLD", "0.04"))


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Microstructure:
    """Parsed orderbook snapshot from a raw Kalshi market dict."""
    yes_bid:        Optional[float]    # cents (1–99) or None
    yes_ask:        Optional[float]    # cents (1–99) or None
    spread_cents:   float              # yes_ask - yes_bid (0 if one side missing)
    mid_cents:      float              # midpoint (0 if no data)
    has_both_sides: bool
    regime:         str = "unknown"    # set after classify_regime()


@dataclass
class ExecutionDecision:
    """Result of execution analysis for one signal."""
    ok:               bool
    reason:           str
    mode:             str             # "aggressive" | "passive" | "adaptive" | "skipped"
    entry_price:      int             # final limit yes_price to submit (cents)
    expected_fill:    float           # estimated actual fill price (cents)
    slippage_est:     float           # estimated slippage cost (cents)
    edge_after_slip:  float           # net edge after slippage (fraction 0–1)
    regime:           str             # "calm" | "trending" | "volatile" | "illiquid"
    spread_cents:     float
    mid_cents:        float
    raw_edge:         float           # from signal (fraction)
    net_edge:         float           # from signal (fraction)


# ── Cooldown tracker ──────────────────────────────────────────────────────────

class CooldownTracker:
    """Per-ticker quiet periods after entries, cancels, and losses."""

    def __init__(self):
        self._expiry: dict[str, tuple[float, str]] = {}   # ticker → (expiry_ts, label)

    def set(self, ticker: str, seconds: int, label: str = ""):
        expiry = time.time() + seconds
        old    = self._expiry.get(ticker, (0, ""))
        if expiry > old[0]:   # only extend, never shorten
            self._expiry[ticker] = (expiry, label)
            logger.info("[COOLDOWN] %s → %ds (%s)", ticker, seconds, label)

    def is_cooling(self, ticker: str) -> tuple[bool, str]:
        entry = self._expiry.get(ticker)
        if not entry:
            return False, ""
        expiry, label = entry
        remaining = expiry - time.time()
        if remaining <= 0:
            del self._expiry[ticker]
            return False, ""
        return True, f"cooldown '{label}' {remaining:.0f}s remaining"

    def clear(self, ticker: str):
        self._expiry.pop(ticker, None)

    def active_count(self) -> int:
        now = time.time()
        return sum(1 for exp, _ in self._expiry.values() if exp > now)


# ── Microstructure analysis ───────────────────────────────────────────────────

def parse_microstructure(market: dict) -> Microstructure:
    """Extract yes_bid, yes_ask, spread, mid from a raw Kalshi market dict."""
    def _c(key) -> Optional[float]:
        v = market.get(key)
        if v is None:
            return None
        try:
            c = float(v) * 100.0
            return c if 0 < c < 100 else None
        except (ValueError, TypeError):
            return None

    bid = _c("yes_bid_dollars")
    ask = _c("yes_ask_dollars")

    if bid is not None and ask is not None:
        spread = max(0.0, ask - bid)
        mid    = (bid + ask) / 2.0
        both   = True
    elif ask is not None:
        spread = 0.0
        mid    = ask
        both   = False
    elif bid is not None:
        spread = 0.0
        mid    = bid
        both   = False
    else:
        spread = 0.0
        mid    = 50.0
        both   = False

    return Microstructure(
        yes_bid=bid, yes_ask=ask,
        spread_cents=spread, mid_cents=mid,
        has_both_sides=both,
    )


# ── Regime detection ──────────────────────────────────────────────────────────

def classify_regime(ms: Microstructure, volatility: float, momentum: float) -> str:
    """
    Classify current market conditions.
    Returns: "calm" | "trending" | "volatile" | "illiquid"
    """
    if not ms.has_both_sides:
        return "illiquid"
    if ms.spread_cents > ILLIQUID_SPREAD_THRESH:
        return "illiquid"
    if volatility > config.MAX_TOTAL_EXPOSURE / 100000:   # reuse vol threshold
        return "volatile"
    if abs(momentum) > TRENDING_MOMENTUM_THRESH:
        return "trending"
    return "calm"


def classify_regime_full(
    ms: Microstructure,
    volatility: float,
    momentum: float,
    max_vol: float,
) -> str:
    """Full regime classifier using explicit thresholds."""
    if not ms.has_both_sides:
        return "illiquid"
    if ms.spread_cents > ILLIQUID_SPREAD_THRESH:
        return "illiquid"
    if volatility > max_vol:
        return "volatile"
    if abs(momentum) > TRENDING_MOMENTUM_THRESH:
        return "trending"
    return "calm"


# ── Mode selection ────────────────────────────────────────────────────────────

def choose_mode(regime: str, net_edge: float, raw_edge: float) -> str:
    """
    Select execution mode based on market regime and edge strength.

    aggressive: large edge, time-sensitive → take the ask immediately
    passive:    calm market, moderate edge  → rest inside the spread
    adaptive:   default / balanced case     → mid or slight improvement
    """
    if regime == "illiquid":
        return "passive"   # don't chase in thin markets

    if abs(raw_edge) >= AGGRESSIVE_EDGE_THRESHOLD:
        if regime in ("calm", "trending"):
            return "aggressive"

    if regime == "calm":
        return "passive"

    if regime == "volatile":
        return "adaptive"  # don't overpay in volatile markets

    return "adaptive"


# ── Entry price optimisation ──────────────────────────────────────────────────

def compute_entry_price(
    side:    str,
    ms:      Microstructure,
    mode:    str,
) -> int:
    """
    Compute the yes_price to submit (1–99 cents) for the given side and mode.

    YES orders: we pay yes_price per contract, get $1 if YES resolves.
    NO  orders: we pay (100 - yes_price) per contract, get $1 if NO resolves.
    """
    bid = ms.yes_bid or ms.mid_cents
    ask = ms.yes_ask or ms.mid_cents

    if side == "yes":
        if mode == "aggressive":
            price = ask                                   # take the offer
        elif mode == "passive":
            price = max(bid, ask - PASSIVE_IMPROVEMENT_CENTS)  # inside spread
        else:
            price = (bid + ask) / 2.0                    # at mid
    else:
        # Buying NO: we want a lower yes_price (we pay 100-yes_price for NO)
        if mode == "aggressive":
            price = bid                                   # hit the bid (sell YES side)
        elif mode == "passive":
            price = min(ask, bid + PASSIVE_IMPROVEMENT_CENTS)  # inside spread
        else:
            price = (bid + ask) / 2.0

    return max(1, min(99, int(round(price))))


# ── Slippage estimation ───────────────────────────────────────────────────────

def estimate_slippage(mode: str, ms: Microstructure, volatility: float) -> float:
    """
    Estimate expected slippage in cents (positive = pay more than quoted).

    Aggressive: worst-case = half the spread + vol buffer
    Passive:    expected improvement = some fraction of passive gap
    Adaptive:   half-spread + small vol buffer
    """
    vol_contrib = volatility * 100 * 0.5   # vol as fraction → cents

    if mode == "aggressive":
        return min(ms.spread_cents * 0.5 + vol_contrib, ms.spread_cents) + SLIPPAGE_BUFFER_CENTS
    elif mode == "passive":
        return -PASSIVE_IMPROVEMENT_CENTS * 0.5 + vol_contrib  # expect slight improvement
    else:
        return ms.spread_cents * 0.25 + vol_contrib + SLIPPAGE_BUFFER_CENTS * 0.5


# ── Entry timing guards ───────────────────────────────────────────────────────

def check_entry_timing(
    ms:         Microstructure,
    momentum:   float,
    signal:     dict,
) -> tuple[bool, str]:
    """
    Check for bad entry timing patterns.
    Returns (ok, reason).

    Blocks:
    - Price just spiked hard in the same direction we want to trade
    - Market has already moved so far that our edge has been consumed
    """
    side     = signal.get("side", "yes")
    raw_edge = signal.get("raw_edge", signal.get("net_edge", 0))

    # If we're buying YES and momentum is also sharply rising,
    # the market may already be reacting to what we see.
    # Threshold: if momentum > SPIKE_LOOKBACK_CENTS worth in the same direction
    if side == "yes" and momentum > SPIKE_LOOKBACK_CENTS / 100:
        return False, (
            f"market already spiked in YES direction (momentum={momentum:+.3f}) "
            f"— avoid chasing"
        )
    if side == "no" and momentum < -(SPIKE_LOOKBACK_CENTS / 100):
        return False, (
            f"market already spiked in NO direction (momentum={momentum:+.3f}) "
            f"— avoid chasing"
        )

    return True, ""


# ── Main analysis entry point ─────────────────────────────────────────────────

class ExecutionEngine:
    """
    Long-lived execution intelligence engine.
    Create once; call analyze() for every candidate signal.
    Holds the CooldownTracker so cooldowns persist across scans.
    """

    def __init__(self):
        self.cooldowns = CooldownTracker()
        self._max_vol  = float(os.environ.get("MAX_VOLATILITY_TRADE", "0.12"))
        logger.info(
            "[EXEC] Initialised. mode=adaptive  max_spread=%d¢  "
            "aggressive_thresh=%.0f%%  exit=%s",
            MAX_SPREAD_CENTS, AGGRESSIVE_EDGE_THRESHOLD * 100, EXIT_MODE,
        )

    def analyze(
        self,
        market:     dict,
        signal:     dict,
        volatility: float = 0.0,
        momentum:   float = 0.0,
    ) -> ExecutionDecision:
        """
        Full pre-trade execution analysis.

        Args:
            market:     raw Kalshi market dict
            signal:     enriched signal from scanner (includes fair_prob, net_edge, etc.)
            volatility: price std-dev fraction from scanner's price history
            momentum:   signed price trend fraction from scanner's price history

        Returns ExecutionDecision.  If .ok is False, do not place the order.
        """
        ticker   = signal.get("ticker", "?")
        side     = signal.get("side", "yes")
        raw_edge = signal.get("raw_edge", 0.0)
        net_edge = signal.get("net_edge", 0.0)

        def skip(reason: str, mode: str = "skipped") -> ExecutionDecision:
            d = ExecutionDecision(
                ok=False, reason=reason, mode=mode,
                entry_price=signal.get("yes_price", 50),
                expected_fill=0, slippage_est=0,
                edge_after_slip=0, regime="unknown",
                spread_cents=0, mid_cents=0,
                raw_edge=raw_edge, net_edge=net_edge,
            )
            _log_exec_audit(ticker, signal, d, None)
            return d

        # ── 1. Cooldown check ─────────────────────────────────────────────
        cooling, cool_reason = self.cooldowns.is_cooling(ticker)
        if cooling:
            return skip(cool_reason)

        # ── 2. Microstructure ─────────────────────────────────────────────
        ms = parse_microstructure(market)

        # ── 3. Basic liquidity / spread check ─────────────────────────────
        if ms.spread_cents > MAX_SPREAD_CENTS and ms.has_both_sides:
            return skip(
                f"spread {ms.spread_cents:.1f}¢ > max {MAX_SPREAD_CENTS}¢"
            )

        if not ms.has_both_sides and ms.mid_cents <= 0:
            return skip("no valid price data in market")

        # ── 4. Regime ─────────────────────────────────────────────────────
        regime = classify_regime_full(ms, volatility, momentum, self._max_vol)
        ms.regime = regime

        if regime == "illiquid":
            return skip(f"illiquid market (spread={ms.spread_cents:.1f}¢, no_both_sides={not ms.has_both_sides})")

        # ── 5. Entry timing guard ─────────────────────────────────────────
        timing_ok, timing_reason = check_entry_timing(ms, momentum, signal)
        if not timing_ok:
            return skip(timing_reason)

        # ── 6. Choose mode ────────────────────────────────────────────────
        mode = choose_mode(regime, net_edge, raw_edge)

        # In volatile regime: downgrade to adaptive if mode was aggressive
        if regime == "volatile" and mode == "aggressive":
            mode = "adaptive"
            logger.debug("[EXEC] %s downgraded to adaptive (volatile regime)", ticker)

        # ── 7. Compute entry price ────────────────────────────────────────
        entry_price  = compute_entry_price(side, ms, mode)
        expected_fill = float(entry_price)   # limit order: we fill at our price or better

        # ── 8. Slippage estimation ────────────────────────────────────────
        slippage_est   = estimate_slippage(mode, ms, volatility)
        slip_frac      = slippage_est / 100.0
        edge_after_slip = raw_edge - slip_frac

        # ── 9. Edge-after-slippage gate ───────────────────────────────────
        if edge_after_slip < MIN_EDGE_AFTER_SLIP:
            return skip(
                f"edge after slippage too thin: "
                f"raw={raw_edge:+.0%} slip={slippage_est:.1f}¢ "
                f"adj={edge_after_slip:+.0%} < {MIN_EDGE_AFTER_SLIP:.0%}"
            )

        decision = ExecutionDecision(
            ok=True, reason="approved",
            mode=mode,
            entry_price=entry_price,
            expected_fill=expected_fill,
            slippage_est=slippage_est,
            edge_after_slip=edge_after_slip,
            regime=regime,
            spread_cents=ms.spread_cents,
            mid_cents=ms.mid_cents,
            raw_edge=raw_edge,
            net_edge=net_edge,
        )

        _log_exec_audit(ticker, signal, decision, ms)
        return decision

    # ── Exit signal ───────────────────────────────────────────────────────────

    def should_exit(
        self,
        ticker:           str,
        entry_price:      float,   # cents we paid per contract
        side:             str,
        current_market:   dict,
        placed_at:        float,   # unix timestamp
        fair_probability: float = 0.5,
        model_flipped:    bool  = False,
        urgency_current:  float = 0.5,
    ) -> tuple[bool, str]:
        """
        Decide whether to exit an existing position.
        Returns (should_exit, reason).  Caller places the closing order.

        EXIT_MODE options:
          settle      — never exit early (always False)
          take_profit — exit if price has converged by TAKE_PROFIT_CENTS;
                        also trims early if price overshoots fair value
          model_flip  — exit if the model now predicts opposite outcome
          adaptive    — combines take_profit + overshoot trim + urgency flip
        """
        if EXIT_MODE == "settle":
            return False, "exit_mode=settle"

        # Max hold time guard (hard limit)
        if MAX_HOLD_SECONDS > 0:
            held_secs = time.time() - placed_at
            if held_secs > MAX_HOLD_SECONDS:
                return True, f"max hold time exceeded ({held_secs:.0f}s > {MAX_HOLD_SECONDS}s)"

        ms = parse_microstructure(current_market)
        current_mid = ms.mid_cents if ms.mid_cents > 0 else entry_price

        if EXIT_MODE in ("take_profit", "adaptive"):
            if side == "yes":
                current_price       = ms.yes_bid or current_mid
                profit_per_contract = current_price - entry_price
            else:
                current_no_price    = 100.0 - (ms.yes_ask or current_mid)
                entry_no_price      = 100.0 - entry_price
                profit_per_contract = current_no_price - entry_no_price

            # Use enhanced timing exit check (overshoot trim + convergence hold)
            try:
                from timing import check_exit_timing
                action, t_reason = check_exit_timing(
                    entry_price       = entry_price,
                    current_price     = current_mid,
                    fair_probability  = fair_probability,
                    side              = side,
                    take_profit_cents = float(TAKE_PROFIT_CENTS),
                    urgency_current   = urgency_current,
                )
                if action in ("trim", "exit"):
                    return True, t_reason
                if action == "hold":
                    return False, t_reason
            except ImportError:
                # Fall back to simple take-profit if timing module unavailable
                if profit_per_contract >= TAKE_PROFIT_CENTS:
                    return True, (
                        f"take profit: {profit_per_contract:.1f}¢ ≥ {TAKE_PROFIT_CENTS}¢"
                    )

        if EXIT_MODE == "model_flip" and model_flipped:
            return True, "model now predicts opposite outcome"

        # adaptive: also exit if urgency flipped negative (position vs current model)
        if EXIT_MODE == "adaptive" and urgency_current < 0.15:
            return True, f"adaptive exit: urgency flipped ({urgency_current:.2f})"

        return False, ""

    # ── Cooldown management ───────────────────────────────────────────────────

    def after_order_placed(self, ticker: str):
        self.cooldowns.set(ticker, COOLDOWN_AFTER_ENTRY_SECS, "entry")

    def after_order_canceled(self, ticker: str):
        self.cooldowns.set(ticker, COOLDOWN_AFTER_CANCEL_SECS, "cancel")

    def after_loss(self, ticker: str):
        self.cooldowns.set(ticker, COOLDOWN_AFTER_LOSS_SECS, "loss")

    def edge_still_valid(
        self,
        ticker:          str,
        side:            str,
        fair_probability: float,
        current_market:  dict,
    ) -> bool:
        """
        Called during fill polling to check if edge is still alive.
        If the market has moved against our position by EDGE_DISAPPEAR_THRESHOLD,
        returns False (caller should cancel the resting order).
        """
        ms = parse_microstructure(current_market)
        if ms.mid_cents <= 0:
            return True   # no data → do not cancel speculatively

        current_yes_prob = ms.mid_cents / 100.0

        if side == "yes":
            remaining_edge = fair_probability - current_yes_prob
        else:
            fair_no_prob    = 1.0 - fair_probability
            current_no_prob = 1.0 - current_yes_prob
            remaining_edge  = fair_no_prob - current_no_prob

        if remaining_edge < -EDGE_DISAPPEAR_THRESHOLD:
            logger.info(
                "[EXEC] Edge disappeared for %s: fair=%.0f%% current=%.0f%% "
                "remaining_edge=%+.0f%%",
                ticker, fair_probability * 100, current_yes_prob * 100,
                remaining_edge * 100,
            )
            return False

        return True


# ── Execution audit log ───────────────────────────────────────────────────────

def _log_exec_audit(
    ticker:   str,
    signal:   dict,
    decision: ExecutionDecision,
    ms:       Optional[Microstructure],
):
    """
    One structured [EXEC_AUDIT] line for every trade decision.
    """
    sport    = signal.get("sport", "?")
    fair_p   = signal.get("fair_probability", 0)
    mkt_p    = signal.get("market_probability", signal.get("yes_price", 50) / 100)
    side     = signal.get("side", "?")

    bid_str  = f"{ms.yes_bid:.0f}¢" if ms and ms.yes_bid else "—"
    ask_str  = f"{ms.yes_ask:.0f}¢" if ms and ms.yes_ask else "—"

    logger.info(
        "[EXEC_AUDIT] %-24s | %-4s | side=%-3s "
        "| bid=%-4s ask=%-4s spread=%.1f¢ mid=%.1f¢ "
        "| fair=%.0f%% mkt=%.0f%% raw_edge=%+.0f%% "
        "| slip=%.1f¢ edge_adj=%+.0f%% "
        "| regime=%-10s mode=%-11s entry=%d¢ "
        "| %s: %s",
        ticker, sport, side,
        bid_str, ask_str,
        decision.spread_cents, decision.mid_cents,
        fair_p * 100, mkt_p * 100, decision.raw_edge * 100,
        decision.slippage_est, decision.edge_after_slip * 100,
        decision.regime, decision.mode, decision.entry_price,
        "TRADE" if decision.ok else "SKIP", decision.reason,
    )
