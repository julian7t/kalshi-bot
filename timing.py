"""
timing.py — Entry timing model and edge-exploitation layer.

Answers the question: WHEN should the bot enter, and HOW MUCH?

Architecture
------------
  TimingClassifier.classify() is called after the portfolio layer approves a
  candidate signal and before the execution layer places an order.  It returns
  a TimingDecision that overrides or augments the execution decision:

    - entry_mode:       now | stage | wait | skip
    - staged_fraction:  fraction of target contracts to place now (0.5 = half)
    - urgency_score:    0–1 composite urgency
    - market_classification: lagging | aligned | overreacting | noisy
    - chase_protected:  True → skip because the move already happened
    - skip_reason:      explanation if entry_mode == "skip"

Inputs (all deterministic, no market anchoring)
---------
  fair_probability      — model's fair probability (not market price)
  market_probability    — current yes_mid / 100
  momentum              — signed price trend fraction (from scanner price history)
  volatility            — price std-dev fraction (from scanner)
  spread_cents          — current bid-ask spread
  regime                — "calm" | "trending" | "volatile" | "illiquid"
  time_to_resolution    — seconds until market closes (0 if unknown)
  net_edge              — fair_prob - market_prob (signed, from signal)
  confidence            — model confidence (0–1)
  side                  — "yes" | "no"

NO modifications to auth, reconciliation, or risk controls.
Settings are logged but NEVER auto-applied.
"""

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger("kalshi_bot.timing")

# ── Tunables (all env-overridable; never auto-applied to live params) ─────────

# Market move classification thresholds (fraction of fair probability)
LAG_THRESHOLD        = float(os.environ.get("TIMING_LAG_THRESHOLD",       "0.04"))  # 4%
OVERREACT_THRESHOLD  = float(os.environ.get("TIMING_OVERREACT_THRESHOLD", "0.04"))  # 4%
NOISE_VOL_THRESHOLD  = float(os.environ.get("TIMING_NOISE_VOL_THRESHOLD", "0.10"))  # 10%

# Chase protection: block entry if the market already moved most of the way
CHASE_MOMENTUM_THRESH   = float(os.environ.get("TIMING_CHASE_MOMENTUM",   "0.05"))  # 5¢/cycle
CHASE_CONSUMED_FRACTION = float(os.environ.get("TIMING_CHASE_CONSUMED",   "0.70"))  # 70% of edge consumed

# Urgency mode thresholds
URGENCY_NOW_THRESH    = float(os.environ.get("TIMING_URGENCY_NOW",   "0.68"))  # above → enter now
URGENCY_STAGE_THRESH  = float(os.environ.get("TIMING_URGENCY_STAGE", "0.38"))  # above → staged entry
URGENCY_WAIT_THRESH   = float(os.environ.get("TIMING_URGENCY_WAIT",  "0.18"))  # above → wait/passive
# Below URGENCY_WAIT_THRESH → skip

# Staged entry: fraction of target size to enter immediately
STAGE_ENTRY_FRACTION  = float(os.environ.get("TIMING_STAGE_FRACTION", "0.50"))  # 50%

# Time sensitivity: urgency boost when time is running out
TIME_URGENCY_MAX_SECS = int(os.environ.get("TIMING_TIME_MAX_SECS", "600"))    # 10 min
TIME_URGENCY_BOOST    = float(os.environ.get("TIMING_TIME_BOOST",   "0.20"))  # +20% urgency at limit

# Edge-decay urgency: if edge is eroding (momentum away from us), raise urgency
EDGE_DECAY_URGENCY_BOOST = float(os.environ.get("TIMING_EDGE_DECAY_BOOST", "0.10"))

# Regime urgency multipliers
_REGIME_MULT = {
    "calm":     1.00,
    "trending": 0.85,
    "volatile": 0.70,
    "illiquid": 0.30,
    "unknown":  0.75,
}

# Overreaction skip: how much above fair before we skip entirely
OVERREACT_SKIP_FRAC = float(os.environ.get("TIMING_OVERREACT_SKIP_FRAC", "0.08"))  # 8%


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class TimingDecision:
    """
    Complete timing decision for one candidate signal.

    Fields
    ------
    market_classification : lagging | aligned | overreacting | noisy
      Describes how the current market price relates to model fair value.

    entry_mode : now | stage | wait | skip
      How the bot should act on this signal right now.
        now   — enter full position aggressively
        stage — enter staged_fraction now, rest later if price improves
        wait  — place passive resting order or skip this cycle
        skip  — do not enter at all this cycle

    urgency_score : 0.0–1.0
      Composite urgency: high → act now; low → wait or skip.

    staged_fraction : 0.0–1.0
      Fraction of the target contract count to place now.
      1.0 if entry_mode is "now" or "wait"; STAGE_ENTRY_FRACTION if "stage".

    is_staged_entry : bool
      True if staged_fraction < 1.0 and entry_mode == "stage".

    chase_protected : bool
      True if the move already happened (market moved most of the way toward
      fair value before we entered).  entry_mode will be "skip" or "wait".

    skip_reason : str
      Non-empty only when entry_mode == "skip". Human-readable explanation.

    rationale : str
      Full explanation for the [TIMING_AUDIT] log line.
    """
    market_classification: str
    entry_mode:            str
    urgency_score:         float
    staged_fraction:       float
    is_staged_entry:       bool
    chase_protected:       bool
    skip_reason:           str
    rationale:             str


# ── Core classifier ───────────────────────────────────────────────────────────

class TimingClassifier:
    """
    Stateless entry timing model.

    Create once; call classify() for every approved candidate signal.
    All logic is deterministic and uses only the inputs passed in —
    no market anchoring, no external state.
    """

    # ── Main entry point ──────────────────────────────────────────────────────

    def classify(
        self,
        signal:               dict,
        momentum:             float,
        volatility:           float,
        spread_cents:         float,
        regime:               str,
        time_to_resolution:   float = 0.0,
    ) -> TimingDecision:
        """
        Produce a TimingDecision for a single candidate signal.

        Args:
            signal              : enriched signal dict (must have fair_probability,
                                  market_probability, net_edge, confidence, side)
            momentum            : signed price trend (fraction, from scanner)
            volatility          : price std-dev fraction (from scanner)
            spread_cents        : current bid-ask spread in cents
            regime              : market regime from execution layer
            time_to_resolution  : seconds until market close (0 = unknown)

        Returns:
            TimingDecision — see dataclass docs above.
        """
        fair_prob  = float(signal.get("fair_probability", 0.5) or 0.5)
        mkt_prob   = float(signal.get("market_probability",
                           signal.get("yes_price", 50) / 100) or 0.5)
        net_edge   = float(signal.get("net_edge",  0.0) or 0.0)
        confidence = float(signal.get("confidence_score", 0.0) or
                           signal.get("confidence", 0.0) or 0.0)
        side       = signal.get("side", "yes")
        ticker     = signal.get("ticker", "?")

        # Signed direction: positive = market under-priced (favor YES)
        directional_edge = net_edge if side == "yes" else -net_edge

        # ── 1. Market move classification ─────────────────────────────────────
        classification = self._classify_market_move(
            fair_prob, mkt_prob, volatility, side, directional_edge,
        )

        # ── 2. Chase protection ───────────────────────────────────────────────
        chase_protected, chase_reason = self._check_chase(
            momentum, directional_edge, net_edge, side,
        )

        # ── 3. Hard skip: overreaction so severe we would be chasing ──────────
        overreact_skip = False
        if classification == "overreacting":
            overshoot = abs(mkt_prob - fair_prob) - OVERREACT_THRESHOLD
            if overshoot >= OVERREACT_SKIP_FRAC:
                overreact_skip = True

        # ── 4. Urgency score ──────────────────────────────────────────────────
        urgency = self._compute_urgency(
            net_edge, confidence, spread_cents, volatility,
            regime, time_to_resolution, classification, momentum, side,
        )

        # ── 5. Entry mode ─────────────────────────────────────────────────────
        skip_reason = ""
        if regime == "illiquid":
            entry_mode  = "skip"
            skip_reason = "illiquid market — timing skip"
        elif overreact_skip:
            entry_mode  = "skip"
            skip_reason = (
                f"market overreacted past fair by "
                f"{abs(mkt_prob - fair_prob):.0%} — "
                f"expected edge already consumed"
            )
        elif chase_protected:
            # Chase protection: prefer wait over hard skip (edge still exists,
            # just the best entry moment has passed this cycle)
            if urgency >= URGENCY_NOW_THRESH:
                entry_mode = "wait"   # slow down but don't skip entirely
            else:
                entry_mode  = "skip"
                skip_reason = chase_reason
        elif classification == "overreacting":
            # Light overreaction: wait for mean-reversion
            entry_mode = "wait"
        elif urgency >= URGENCY_NOW_THRESH:
            entry_mode = "now"
        elif urgency >= URGENCY_STAGE_THRESH:
            entry_mode = "stage"
        elif urgency >= URGENCY_WAIT_THRESH:
            entry_mode = "wait"
        else:
            entry_mode  = "skip"
            skip_reason = f"urgency too low ({urgency:.2f} < {URGENCY_WAIT_THRESH})"

        # ── 6. Staged sizing ──────────────────────────────────────────────────
        if entry_mode == "stage":
            staged_fraction  = STAGE_ENTRY_FRACTION
            is_staged_entry  = True
        else:
            staged_fraction  = 1.0
            is_staged_entry  = False

        # ── 7. Rationale string ───────────────────────────────────────────────
        rationale = (
            f"class={classification} urgency={urgency:.2f} "
            f"mode={entry_mode} frac={staged_fraction:.2f} "
            f"regime={regime} fair={fair_prob:.0%} mkt={mkt_prob:.0%} "
            f"edge={net_edge:+.0%} mom={momentum:+.3f} vol={volatility:.3f} "
            f"spread={spread_cents:.1f}¢ ttres={time_to_resolution:.0f}s "
            f"chase={chase_protected}"
        )

        decision = TimingDecision(
            market_classification = classification,
            entry_mode            = entry_mode,
            urgency_score         = round(urgency, 4),
            staged_fraction       = round(staged_fraction, 3),
            is_staged_entry       = is_staged_entry,
            chase_protected       = chase_protected,
            skip_reason           = skip_reason,
            rationale             = rationale,
        )

        self._log_audit(ticker, decision)
        return decision

    # ── Market move classification ────────────────────────────────────────────

    def _classify_market_move(
        self,
        fair_prob:        float,
        mkt_prob:         float,
        volatility:       float,
        side:             str,
        directional_edge: float,
    ) -> str:
        """
        Classify the current market-price move relative to model fair value.

        lagging      — market is behind our model; good entry window
        aligned      — market is near our fair; normal conditions
        overreacting — market has moved past our fair; wait for reversion
        noisy        — high volatility, signal unclear
        """
        if volatility > NOISE_VOL_THRESHOLD and abs(directional_edge) < LAG_THRESHOLD:
            return "noisy"

        gap = fair_prob - mkt_prob   # positive = market under-prices YES

        if side == "yes":
            # We want to buy YES; market is lagging if it's cheap vs fair
            if gap >= LAG_THRESHOLD:
                return "lagging"
            if gap <= -OVERREACT_THRESHOLD:
                return "overreacting"
        else:
            # We want to buy NO; market is lagging if YES is expensive vs fair
            # (i.e. fair < mkt_prob — no side is cheap)
            if gap <= -LAG_THRESHOLD:
                return "lagging"
            if gap >= OVERREACT_THRESHOLD:
                return "overreacting"

        return "aligned"

    # ── Chase protection ──────────────────────────────────────────────────────

    def _check_chase(
        self,
        momentum:         float,
        directional_edge: float,
        net_edge:         float,
        side:             str,
    ) -> tuple[bool, str]:
        """
        Detect if the market is already sprinting toward fair value —
        meaning we would be chasing a move that is already mostly done.

        Returns (chase_protected: bool, reason: str).
        """
        # Momentum is in the same direction as our trade
        if side == "yes":
            momentum_favorable = momentum > 0   # price rising, we want YES
        else:
            momentum_favorable = momentum < 0   # price falling, we want NO

        if not momentum_favorable:
            return False, ""

        abs_momentum = abs(momentum)
        if abs_momentum < CHASE_MOMENTUM_THRESH:
            return False, ""

        # Estimate how much of the original edge has been consumed by the move.
        # momentum is a price change fraction; net_edge is the remaining gap.
        # If momentum (recent move) already exceeds the remaining edge times a
        # threshold, the move happened before us.
        if net_edge <= 0:
            return True, (
                "edge exhausted (net_edge ≤ 0) and momentum is running in "
                "our direction — missed the entry"
            )

        consumed_fraction = abs_momentum / (abs_momentum + abs(net_edge))
        if consumed_fraction >= CHASE_CONSUMED_FRACTION:
            return True, (
                f"market already moved {consumed_fraction:.0%} of the "
                f"expected edge (momentum={abs_momentum:+.3f}, "
                f"remaining_edge={net_edge:+.0%}) — chase protection"
            )

        return False, ""

    # ── Urgency score ─────────────────────────────────────────────────────────

    def _compute_urgency(
        self,
        net_edge:           float,
        confidence:         float,
        spread_cents:       float,
        volatility:         float,
        regime:             str,
        time_to_resolution: float,
        classification:     str,
        momentum:           float,
        side:               str,
    ) -> float:
        """
        Compute a 0–1 urgency score.

        Urgency is high when:
          • Edge is large and confidence is high
          • Time to resolution is short (market closing soon)
          • Regime is calm (low friction)
          • Market is lagging (good entry window, unlikely to close fast)

        Urgency is low when:
          • Spread is wide (high friction)
          • Volatility is high (hard to time well)
          • Market is overreacting (better to wait)
          • Regime is illiquid or volatile
        """
        # Base: net_edge drives urgency (normalized to ~20% max edge)
        base = min(1.0, abs(net_edge) / 0.18)

        # Confidence multiplier (0–1)
        conf_mult = max(0.3, min(1.0, confidence))

        # Spread penalty (wide spread → less urgent to enter now)
        spread_penalty = max(0.4, 1.0 - spread_cents / 24.0)

        # Volatility penalty (high vol → wait for better price)
        vol_penalty = max(0.4, 1.0 - volatility * 6.0)

        # Regime multiplier
        regime_mult = _REGIME_MULT.get(regime, 0.75)

        # Time sensitivity: if resolution is soon, urgency rises sharply
        time_boost = 0.0
        if time_to_resolution > 0:
            remaining_fraction = max(0.0, min(1.0,
                1.0 - time_to_resolution / (TIME_URGENCY_MAX_SECS * 6)
            ))
            time_boost = TIME_URGENCY_BOOST * remaining_fraction

        # Classification adjustment
        class_mult = {
            "lagging":     1.10,   # great window — act sooner
            "aligned":     1.00,
            "overreacting": 0.60,  # wait for reversion
            "noisy":       0.80,
        }.get(classification, 1.00)

        # Edge decay boost: if momentum is running AWAY from us, enter now or
        # our edge will erode (opposing momentum creates urgency)
        edge_decay_boost = 0.0
        if side == "yes" and momentum < -0.02:
            edge_decay_boost = EDGE_DECAY_URGENCY_BOOST * min(1.0, abs(momentum) / 0.10)
        elif side == "no" and momentum > 0.02:
            edge_decay_boost = EDGE_DECAY_URGENCY_BOOST * min(1.0, abs(momentum) / 0.10)

        urgency = (
            base
            * conf_mult
            * spread_penalty
            * vol_penalty
            * regime_mult
            * class_mult
            + time_boost
            + edge_decay_boost
        )
        return max(0.0, min(1.0, urgency))

    # ── Audit log ─────────────────────────────────────────────────────────────

    def _log_audit(self, ticker: str, d: TimingDecision):
        """One structured [TIMING_AUDIT] line per evaluated candidate."""
        decision_str = d.entry_mode.upper()
        if d.skip_reason:
            decision_str += f": {d.skip_reason}"
        logger.info(
            "[TIMING_AUDIT] %-24s | class=%-12s | urgency=%.2f | %-5s | "
            "staged=%s frac=%.2f | chase=%s | %s",
            ticker,
            d.market_classification,
            d.urgency_score,
            d.entry_mode.upper(),
            d.is_staged_entry,
            d.staged_fraction,
            d.chase_protected,
            d.rationale,
        )


# ── Re-entry / add-to-position logic ─────────────────────────────────────────

# Re-entry rules: may the bot enter again into a ticker it already holds?
ALLOW_ADD_SAME_TICKER   = os.environ.get("TIMING_ALLOW_ADD", "false").lower() == "true"
ADD_MIN_PRICE_IMPROVE   = float(os.environ.get("TIMING_ADD_MIN_PRICE_IMPROVE", "3.0"))  # cents
ADD_MIN_EDGE_IMPROVE    = float(os.environ.get("TIMING_ADD_MIN_EDGE_IMPROVE",  "0.02")) # 2% more edge
ADD_COOLDOWN_SECS       = int(os.environ.get("TIMING_ADD_COOLDOWN_SECS", "180"))        # 3 min


def can_add_to_position(
    ticker:            str,
    side:              str,
    current_entry:     float,    # cents we paid originally
    new_price:         float,    # current market ask/bid (cents)
    original_net_edge: float,    # edge when we entered
    current_net_edge:  float,    # edge right now
    last_entry_ts:     float,    # unix timestamp of last entry
    cancel_cooldown:   bool,     # True if ticker is in cancel cooldown
    import_time:       "float",  # current time.time()
) -> tuple[bool, str]:
    """
    Determine whether to add to an existing position on the same ticker.

    Conservative rules:
      1. TIMING_ALLOW_ADD must be true (default: false)
      2. Not in cancel cooldown
      3. Minimum time since last entry
      4. Price improved by at least ADD_MIN_PRICE_IMPROVE cents vs original
      5. Current edge is materially stronger than at entry

    Returns (allowed: bool, reason: str).
    """
    if not ALLOW_ADD_SAME_TICKER:
        return False, "add-to-position disabled (TIMING_ALLOW_ADD=false)"

    if cancel_cooldown:
        return False, "cancel cooldown active — no add"

    secs_since_entry = import_time - last_entry_ts
    if secs_since_entry < ADD_COOLDOWN_SECS:
        return False, (
            f"add cooldown: {secs_since_entry:.0f}s since last entry "
            f"(min={ADD_COOLDOWN_SECS}s)"
        )

    # Price must have improved (we're getting a better deal than last time)
    if side == "yes":
        price_improvement = current_entry - new_price   # paying less YES cents
    else:
        price_improvement = new_price - current_entry   # paying less NO cents

    if price_improvement < ADD_MIN_PRICE_IMPROVE:
        return False, (
            f"price not improved enough: {price_improvement:.1f}¢ "
            f"< {ADD_MIN_PRICE_IMPROVE}¢"
        )

    # Edge must have strengthened
    edge_improvement = current_net_edge - original_net_edge
    if edge_improvement < ADD_MIN_EDGE_IMPROVE:
        return False, (
            f"edge not improved enough: {edge_improvement:+.0%} "
            f"< {ADD_MIN_EDGE_IMPROVE:.0%}"
        )

    return True, (
        f"add approved: price_improve={price_improvement:.1f}¢ "
        f"edge_improve={edge_improvement:+.0%}"
    )


# ── Exit timing helpers ───────────────────────────────────────────────────────

TRIM_OVERSHOOT_THRESHOLD = float(os.environ.get("TIMING_TRIM_OVERSHOOT", "0.05"))  # 5% past fair
CONVERGENCE_HOLD_MIN     = float(os.environ.get("TIMING_CONV_HOLD_MIN",  "0.60"))  # 60% of TP target


def check_exit_timing(
    entry_price:      float,    # cents we paid (YES side perspective)
    current_price:    float,    # current market yes-mid (cents)
    fair_probability: float,    # model fair (0–1)
    side:             str,
    take_profit_cents: float,
    urgency_current:  float = 0.5,  # current urgency (if computed)
) -> tuple[str, str]:
    """
    Enhanced exit timing check.

    Returns (action, reason) where action is one of:
      hold    — wait for further convergence
      trim    — exit now: price overshot fair value
      exit    — full exit: take-profit or urgency-flip trigger
      stay    — no exit signal

    Callers should still respect the hard EXIT_MODE setting from config.
    This function only makes a *recommendation* based on timing.
    """
    fair_cents = fair_probability * 100

    if side == "yes":
        profit_per_contract = current_price - entry_price
        # Overshoot: current price blew past fair value
        overshoot_cents     = current_price - fair_cents
    else:
        entry_no  = 100.0 - entry_price
        current_no = 100.0 - current_price
        profit_per_contract = current_no - entry_no
        overshoot_cents     = (100.0 - current_price) - (100.0 - fair_cents)
        overshoot_cents     = -overshoot_cents  # flip sign for NO side

    # Trim early if price overshoots fair value by more than threshold
    overshoot_frac = overshoot_cents / 100.0
    if overshoot_frac >= TRIM_OVERSHOOT_THRESHOLD:
        return "trim", (
            f"price overshot fair by {overshoot_frac:.0%} "
            f"(current={current_price:.1f}¢, fair={fair_cents:.1f}¢) — trim early"
        )

    # Take-profit: converged enough
    if profit_per_contract >= take_profit_cents:
        return "exit", f"take profit: {profit_per_contract:.1f}¢ ≥ {take_profit_cents:.1f}¢"

    # Hold if convergence is incomplete (< CONVERGENCE_HOLD_MIN of target)
    convergence_frac = profit_per_contract / take_profit_cents if take_profit_cents > 0 else 0
    if convergence_frac < CONVERGENCE_HOLD_MIN:
        return "hold", (
            f"convergence {convergence_frac:.0%} < {CONVERGENCE_HOLD_MIN:.0%} "
            f"of target — hold for further convergence"
        )

    # Urgency flip: if urgency dropped sharply (model changed its mind), reduce
    if urgency_current < 0.20:
        return "exit", f"urgency flipped negative ({urgency_current:.2f}) — reduce exposure"

    return "stay", ""


# ── Missed-edge estimator ─────────────────────────────────────────────────────

def estimate_missed_edge(
    entry_mode_chosen:   str,   # "wait" | "skip"
    net_edge_at_skip:    float, # edge at time of skip/wait decision
    price_at_skip:       float, # market mid at time of skip (cents)
    price_current:       float, # market mid right now (cents)
    side:                str,
) -> float:
    """
    Estimate how much edge was missed if the bot chose wait/skip and the
    market subsequently moved favorably without a fill.

    Returns missed_edge_cents (positive = we missed money).
    This is purely informational — stored in analytics for review.
    """
    if entry_mode_chosen not in ("wait", "skip"):
        return 0.0

    if side == "yes":
        favorable_move = price_current - price_at_skip   # price rose
    else:
        favorable_move = price_at_skip - price_current   # price fell

    if favorable_move <= 0:
        return 0.0

    # Missed cents = how much the market moved in our direction post-skip
    # Capped at the original net_edge to avoid unrealistic estimates
    max_claimable = abs(net_edge_at_skip) * 100
    return round(min(favorable_move, max_claimable), 2)
