"""
model.py — Independent sport-specific fair-probability model (v3).

Design contract:
  - fair_probability is computed ENTIRELY from game state.
    It is NEVER blended with the Kalshi market price.
  - market_probability (Kalshi price) is used ONLY for:
      • edge = fair_probability - market_probability
      • confidence adjustment (agreement ↑ confidence slightly)
      • opportunity ranking
  - All logic is deterministic and traceable — no ML, no black-box.
  - Each sport has its own probability function.
  - Market type is classified per-market; specialized models price
    totals, spreads, segment winners, and game winners differently.
  - Confidence is penalised for missing data, chaotic game states,
    early game, and poor data freshness.
  - No-trade zones are enforced before returning a result.

Public API:
    result = evaluate_signal(signal, game_state, volatility=0.0)
    # result is None → hard-block (reason logged)
    # result is dict → keys: fair_probability, market_probability,
    #   raw_edge, net_edge, confidence_score, model_used, model_name,
    #   market_type, parsed_line, projected_value, signal_reason,
    #   sport, reason

Internal sport helpers (_basketball_prob etc.) are also imported by
signal_models.GameWinnerModel — keep them stable.
"""

import logging
import math
import os
from typing import Optional

from game_state import GameState, _parse_clock_seconds, _period_minutes

logger = logging.getLogger("kalshi_bot.model")

# ── Tunables (env-overridable) ────────────────────────────────────────────────

MIN_PROGRESS_FOR_MODEL  = float(os.environ.get("MIN_PROGRESS_FOR_MODEL",  "0.50"))
FINAL_SECONDS_NO_TRADE  = int(os.environ.get("FINAL_SECONDS_NO_TRADE",    "90"))
MIN_RAW_EDGE            = float(os.environ.get("MIN_RAW_EDGE",             "0.05"))  # 5¢
HIGH_VOLATILITY_PENALTY = float(os.environ.get("HIGH_VOLATILITY_PENALTY", "0.15"))
MAX_VOLATILITY_TRADE    = float(os.environ.get("MAX_VOLATILITY_TRADE",     "0.12"))  # 12¢

# Sport family groupings
_BASKETBALL = {"NBA", "NCAAB"}
_FOOTBALL   = {"NFL", "NCAAF"}
_BASEBALL   = {"MLB"}
_HOCKEY     = {"NHL"}
_SOCCER     = {"MLS", "EPL", "UEFA", "SOCCER"}


# ── Public entry point ────────────────────────────────────────────────────────

def evaluate_signal(
    signal:     dict,
    game_state: Optional[GameState],
    volatility: float = 0.0,
) -> Optional[dict]:
    """
    Evaluate one strategy signal using a market-type specialized model.

    Returns enriched dict or None (trade blocked).

    Output keys (in addition to all signal keys):
        fair_probability   float   from game-state model only — never anchored to market price
        market_probability float   Kalshi implied YES prob
        raw_edge           float   fair - market (positive = market underprices YES)
        net_edge           float   raw_edge * confidence
        confidence_score   float
        volatility         float   passed in from scanner's price history
        model_used         str     model_name from the specialized model
        model_name         str     same as model_used (explicit new key)
        market_type        str     classified market type
        parsed_line        float   spread/total line value if parsed, else None
        projected_value    float   projected total or margin if applicable, else None
        signal_reason      str     full reason string from the specialized model
        sport              str
        reason             str     same as signal_reason (backward compat)
    """
    # Lazy imports to avoid circular dependencies at module load time
    import classifier    as _cls
    import parser        as _prs
    import signal_models as _sm

    ticker          = signal.get("ticker", "?")
    title           = signal.get("title",  ticker)
    side            = signal.get("side",   "yes")
    yes_price       = signal.get("yes_price", 50)
    market_yes_prob = yes_price / 100.0

    # ── No game state ─────────────────────────────────────────────────────────
    if game_state is None:
        reason = "no matched game state — model cannot compute independent fair probability"
        logger.info("[MODEL] SKIP %s — %s", ticker, reason)
        return None

    # ── No-trade zones (stale clock, final seconds, high volatility) ──────────
    block = _check_no_trade_zones(game_state, volatility, ticker)
    if block:
        return None

    sport = _classify_sport(game_state.league)

    # ── Classify market type ──────────────────────────────────────────────────
    # Honour a pre-set market_type if the scanner already attached one;
    # otherwise run the classifier now so the result is always present.
    market_type = signal.get("market_type") or ""
    if not market_type:
        cls_result  = _cls.classify(ticker, title, sport)
        market_type = cls_result["market_type"]
        logger.debug(
            "[MODEL] classified %s → %s (conf=%.0f%% src=%s)",
            ticker, market_type,
            cls_result.get("confidence", 0) * 100,
            cls_result.get("source", "?"),
        )
    else:
        logger.debug("[MODEL] pre-classified %s → %s", ticker, market_type)

    # ── Parse market line (spread/total/period ref) ───────────────────────────
    parse_info = _prs.parse_for_market_type(market_type, ticker, title)

    # Hard block: totals and spread markets MUST have a parseable line
    if market_type in ("totals", "spread") and not parse_info.get("parsed"):
        reason = (
            f"{market_type} line parse failed — {parse_info.get('reason', 'unknown')} "
            f"— cannot price market without the line"
        )
        logger.info("[MODEL] SKIP %s — %s", ticker, reason)
        return None

    # ── Dispatch to specialized model ─────────────────────────────────────────
    model_result = _sm.dispatch(
        market_type=market_type,
        game_state=game_state,
        sport=sport,
        ticker=ticker,
        side=side,
        market_yes_prob=market_yes_prob,
        parse_info=parse_info,
        title=title,
    )

    if not model_result.can_trade:
        logger.info("[MODEL] SKIP %s — %s", ticker, model_result.no_trade_reason)
        return None

    fair_prob  = model_result.fair_probability
    model_conf = model_result.confidence

    # ── Edge check ────────────────────────────────────────────────────────────
    raw_edge = fair_prob - market_yes_prob
    if abs(raw_edge) < MIN_RAW_EDGE:
        reason = (
            f"edge too small: fair={fair_prob:.0%} market={market_yes_prob:.0%} "
            f"raw_edge={raw_edge:+.0%} < {MIN_RAW_EDGE:.0%}"
        )
        logger.info("[MODEL] SKIP %s — %s", ticker, reason)
        return None

    # ── Final confidence (applies volatility, freshness, agreement bonuses) ───
    confidence = _compute_confidence(
        model_conf, game_state, market_yes_prob, fair_prob, volatility
    )
    net_edge = raw_edge * confidence

    # ── Assemble enriched signal ──────────────────────────────────────────────
    enriched = dict(signal)
    enriched.update({
        "fair_probability":   round(fair_prob, 4),
        "market_probability": round(market_yes_prob, 4),
        "raw_edge":           round(raw_edge, 4),
        "net_edge":           round(net_edge, 4),
        "confidence_score":   round(confidence, 4),
        "volatility":         round(volatility, 4),
        # Model metadata
        "model_used":         model_result.model_name,
        "model_name":         model_result.model_name,
        "market_type":        market_type,
        "parsed_line":        model_result.parsed_line,
        "projected_value":    model_result.projected_value,
        "signal_reason":      model_result.reason,
        # Sport & reason (backward compat)
        "sport":              sport,
        "reason":             model_result.reason,
    })

    logger.info(
        "[MODEL] PASS %-28s sport=%-10s mtype=%-12s model=%-20s "
        "fair=%.0f%% mkt=%.0f%% edge=%+.0f%% net=%+.0f%% conf=%.0f%% vol=%.3f",
        ticker, sport, market_type, model_result.model_name,
        fair_prob * 100, market_yes_prob * 100,
        raw_edge * 100, net_edge * 100, confidence * 100, volatility,
    )

    return enriched


# ── No-trade zones ────────────────────────────────────────────────────────────

def _check_no_trade_zones(
    gs: GameState,
    volatility: float,
    ticker: str,
) -> Optional[str]:
    """
    Return a reason string if trading should be blocked, None if OK.
    Logs the block reason at INFO level.
    """
    if not gs.is_active:
        reason = f"game not active (status={gs.status})"
        logger.info("[MODEL] BLOCK %s — %s", ticker, reason)
        return reason

    progress = gs.game_progress
    if progress < MIN_PROGRESS_FOR_MODEL:
        reason = f"game too early: progress={progress:.0%} < {MIN_PROGRESS_FOR_MODEL:.0%}"
        logger.info("[MODEL] BLOCK %s — %s", ticker, reason)
        return reason

    # Final seconds — too risky to execute
    clock_secs   = _parse_clock_seconds(gs.game_clock)
    period_mins  = _period_minutes(gs.league, gs.total_periods)
    is_last_period = gs.period >= gs.total_periods

    if is_last_period and clock_secs is not None and clock_secs <= FINAL_SECONDS_NO_TRADE:
        reason = (
            f"final seconds no-trade zone: {clock_secs:.0f}s remaining "
            f"(threshold={FINAL_SECONDS_NO_TRADE}s)"
        )
        logger.info("[MODEL] BLOCK %s — %s", ticker, reason)
        return reason

    # High volatility
    if volatility > MAX_VOLATILITY_TRADE:
        reason = (
            f"volatility too high: {volatility:.3f} > {MAX_VOLATILITY_TRADE:.3f}"
        )
        logger.info("[MODEL] BLOCK %s — %s", ticker, reason)
        return reason

    return None


# ── Sport classifier ──────────────────────────────────────────────────────────

def _classify_sport(league: str) -> str:
    lg = league.upper()
    if lg in _BASKETBALL:  return "BASKETBALL"
    if lg in _FOOTBALL:    return "FOOTBALL"
    if lg in _BASEBALL:    return "BASEBALL"
    if lg in _HOCKEY:      return "HOCKEY"
    if lg in _SOCCER:      return "SOCCER"
    return "GENERIC"


# ── Dispatcher ────────────────────────────────────────────────────────────────

def _compute_fair_probability(
    gs: GameState,
    sport: str,
    ticker: str,
) -> Optional[tuple[float, float, list[str]]]:
    """
    Dispatch to sport-specific model.
    Returns (home_win_prob, model_confidence, [factor strings]) or None.
    """
    fn = {
        "BASKETBALL": _basketball_prob,
        "FOOTBALL":   _football_prob,
        "BASEBALL":   _baseball_prob,
        "HOCKEY":     _hockey_prob,
        "SOCCER":     _soccer_prob,
        "GENERIC":    _generic_prob,
    }.get(sport, _generic_prob)

    try:
        return fn(gs)
    except Exception as e:
        logger.warning("[MODEL] Sport model error (%s) for %s: %s", sport, ticker, e)
        return None


# ── Basketball (NBA / NCAAB) ──────────────────────────────────────────────────

def _basketball_prob(gs: GameState) -> Optional[tuple[float, float, list[str]]]:
    diff         = gs.score_differential
    clock_secs   = _parse_clock_seconds(gs.game_clock)
    period       = gs.period
    total_p      = gs.total_periods
    period_mins  = _period_minutes(gs.league, total_p) or 12.0

    # Time remaining in whole game (seconds)
    periods_left = max(0, total_p - period)
    if clock_secs is not None:
        total_remaining = periods_left * period_mins * 60 + clock_secs
        clock_known = True
    else:
        total_remaining = periods_left * period_mins * 60 + period_mins * 30
        clock_known = False

    total_game_secs = total_p * period_mins * 60
    time_remaining_frac = max(0.001, total_remaining / total_game_secs)

    # Possession adds ~1 point worth of advantage
    poss_adj = 0.0
    if gs.possession == "home":
        poss_adj = 1.0
    elif gs.possession == "away":
        poss_adj = -1.0

    # Late-game sensitivity: Q4 last 5 min → much sharper
    is_last_period = period >= total_p
    late_game      = is_last_period and (clock_secs or 999) <= 300
    sensitivity    = 0.15 / time_remaining_frac
    if late_game:
        sensitivity *= 2.0   # double sensitivity in last 5 minutes
    sensitivity = min(sensitivity, 8.0)

    effective_diff = diff + poss_adj
    x = effective_diff * sensitivity / 10.0   # normalise to ~[-3, 3] range
    home_win_prob = _sigmoid(x)
    home_win_prob = max(0.03, min(0.97, home_win_prob))

    factors = [
        f"diff={diff:+d}",
        f"poss={gs.possession or 'unk'}(+{poss_adj:.0f})",
        f"trm={total_remaining:.0f}s",
        f"sens={sensitivity:.2f}",
        f"late={'Y' if late_game else 'N'}",
    ]

    # Confidence: penalise missing clock, very early in last period
    conf = 0.75 if clock_known else 0.45
    if is_last_period and not late_game:
        conf *= 0.85   # mid-Q4 is uncertain
    if gs.possession is None:
        conf *= 0.90

    return home_win_prob, conf, factors


# ── Football (NFL / NCAAF) ────────────────────────────────────────────────────

def _football_prob(gs: GameState) -> Optional[tuple[float, float, list[str]]]:
    diff        = gs.score_differential
    clock_secs  = _parse_clock_seconds(gs.game_clock)
    period      = gs.period
    total_p     = gs.total_periods
    period_mins = _period_minutes(gs.league, total_p) or 15.0

    periods_left = max(0, total_p - period)
    if clock_secs is not None:
        total_remaining = periods_left * period_mins * 60 + clock_secs
        clock_known = True
    else:
        total_remaining = periods_left * period_mins * 60 + period_mins * 30
        clock_known = False

    total_game_secs     = total_p * period_mins * 60
    time_remaining_frac = max(0.001, total_remaining / total_game_secs)

    # Possession matters a lot in final 2 minutes (2-minute drill)
    poss_adj = 0.0
    if gs.possession == "home":
        poss_adj = 2.0 if (clock_secs or 999) <= 120 and period >= total_p else 0.5
    elif gs.possession == "away":
        poss_adj = -2.0 if (clock_secs or 999) <= 120 and period >= total_p else -0.5

    # Football scoring: 3 pts (FG), 7 pts (TD+PAT), 8 pts (2pt conv)
    # A 2-score lead (9+) is much more comfortable late
    two_score_safety  = abs(diff) >= 9
    one_score_game    = abs(diff) <= 8
    blowout           = abs(diff) >= 21

    sensitivity = 0.12 / time_remaining_frac
    if blowout:
        sensitivity *= 1.5
    elif two_score_safety:
        sensitivity *= 1.2
    sensitivity = min(sensitivity, 6.0)

    effective_diff = diff + poss_adj
    x = effective_diff * sensitivity / 14.0   # normalised; 14 = ~2 TD lead
    home_win_prob = _sigmoid(x)
    home_win_prob = max(0.03, min(0.97, home_win_prob))

    factors = [
        f"diff={diff:+d}",
        f"poss={gs.possession or 'unk'}",
        f"trm={total_remaining:.0f}s",
        f"2score={'Y' if two_score_safety else 'N'}",
        f"sens={sensitivity:.2f}",
    ]
    conf = 0.70 if clock_known else 0.40
    if not two_score_safety and period >= total_p:
        conf *= 0.75   # one-score games in 4th are volatile
    if gs.possession is None:
        conf *= 0.85

    return home_win_prob, conf, factors


# ── Baseball (MLB) ────────────────────────────────────────────────────────────

def _baseball_prob(gs: GameState) -> Optional[tuple[float, float, list[str]]]:
    diff         = gs.score_differential
    inning       = gs.period           # ESPN maps inning to period
    total_inn    = gs.total_periods    # 9
    inning_frac  = min(1.0, inning / max(total_inn, 1))

    # Is it the bottom half? ESPN game_clock often contains "Top"/"Bot"
    is_bottom = "bot" in gs.game_clock.lower() or "bottom" in gs.game_clock.lower()

    # Home team bats in the bottom of the inning — advantage if they're losing
    # and still have at-bats left
    home_batting_adj = 0.0
    if is_bottom and gs.home_score < gs.away_score and inning >= 9:
        home_batting_adj = 0.5   # home team can still tie/win

    # Late-inning sensitivity: 9th+ inning with a big lead is decisive
    late_inning = inning >= 7
    sensitivity = 0.5 + inning_frac * 1.5   # grows from 0.5 to 2.0

    effective_diff = diff + home_batting_adj
    x = effective_diff * sensitivity / 3.0   # 3 run lead in 9th ≈ near-certain
    home_win_prob = _sigmoid(x)
    home_win_prob = max(0.03, min(0.97, home_win_prob))

    factors = [
        f"inning={inning}/{total_inn}",
        f"diff={diff:+d}",
        f"bottom={'Y' if is_bottom else 'N'}",
        f"late={'Y' if late_inning else 'N'}",
        f"home_bat_adj={home_batting_adj:+.1f}",
    ]
    conf = 0.65 if late_inning else 0.50
    if total_inn < 9:
        conf *= 0.80   # shortened game = less certainty

    return home_win_prob, conf, factors


# ── Hockey (NHL) ──────────────────────────────────────────────────────────────

def _hockey_prob(gs: GameState) -> Optional[tuple[float, float, list[str]]]:
    diff        = gs.score_differential
    clock_secs  = _parse_clock_seconds(gs.game_clock)
    period      = gs.period
    total_p     = gs.total_periods    # 3
    period_mins = _period_minutes(gs.league, total_p) or 20.0

    periods_left = max(0, total_p - period)
    if clock_secs is not None:
        total_remaining = periods_left * period_mins * 60 + clock_secs
        clock_known = True
    else:
        total_remaining = periods_left * period_mins * 60 + period_mins * 30
        clock_known = False

    total_game_secs     = total_p * period_mins * 60
    time_remaining_frac = max(0.001, total_remaining / total_game_secs)

    # Empty-net proxy: if winning by 2+ in last 2 min, near-certain
    is_final_period     = period >= total_p
    empty_net_likely    = is_final_period and abs(diff) >= 2 and (clock_secs or 999) <= 120

    sensitivity = 0.18 / time_remaining_frac
    if empty_net_likely:
        sensitivity *= 2.0
    sensitivity = min(sensitivity, 8.0)

    x = diff * sensitivity / 2.0   # 2-goal lead is a blowout in hockey
    home_win_prob = _sigmoid(x)
    home_win_prob = max(0.03, min(0.97, home_win_prob))

    factors = [
        f"diff={diff:+d}",
        f"trm={total_remaining:.0f}s",
        f"empty_net={'Y' if empty_net_likely else 'N'}",
        f"sens={sensitivity:.2f}",
    ]
    conf = 0.70 if clock_known else 0.45
    if is_final_period and abs(diff) == 1 and not empty_net_likely:
        conf *= 0.70   # one-goal game in 3rd is volatile

    return home_win_prob, conf, factors


# ── Soccer (MLS / EPL / etc.) ─────────────────────────────────────────────────

def _soccer_prob(gs: GameState) -> Optional[tuple[float, float, list[str]]]:
    diff       = gs.score_differential
    # Soccer: game_clock usually shows minutes elapsed ("73'" etc.)
    # period: 1 or 2 (halves)
    period     = gs.period
    total_p    = gs.total_periods    # 2
    period_mins = 45.0

    # Parse elapsed minutes from game clock
    elapsed_min = _parse_soccer_clock(gs.game_clock)
    if elapsed_min is None:
        elapsed_min = (period - 1) * 45.0 + 22.5   # fallback: mid-half
    elapsed_min = min(elapsed_min, 95.0)

    total_mins          = 90.0
    time_remaining_min  = max(0.5, total_mins - elapsed_min)
    progress            = elapsed_min / total_mins

    # Late-game with a lead: exponentially more certain
    stoppage_time = gs.game_clock.lower().count("+") > 0

    sensitivity = 0.08 / (time_remaining_min / 90.0 + 0.01)
    if stoppage_time:
        sensitivity *= 1.5
    sensitivity = min(sensitivity, 6.0)

    x = diff * sensitivity / 1.5   # 2-goal lead ≈ near-certain late
    home_win_prob = _sigmoid(x)
    home_win_prob = max(0.03, min(0.97, home_win_prob))

    factors = [
        f"diff={diff:+d}",
        f"min={elapsed_min:.0f}",
        f"trm={time_remaining_min:.0f}min",
        f"stoppage={'Y' if stoppage_time else 'N'}",
        f"sens={sensitivity:.2f}",
    ]
    conf = 0.65 if elapsed_min > 45 else 0.45
    if abs(diff) == 0 and time_remaining_min < 15:
        conf *= 0.70   # draws are chaotic late

    return home_win_prob, conf, factors


# ── Generic (fallback) ────────────────────────────────────────────────────────

def _generic_prob(gs: GameState) -> Optional[tuple[float, float, list[str]]]:
    diff     = gs.score_differential
    progress = gs.game_progress
    urgency  = progress

    blowout = 15
    x = (diff / blowout) * urgency * 2.5
    home_win_prob = _sigmoid(x)
    home_win_prob = max(0.03, min(0.97, home_win_prob))

    factors = [f"diff={diff:+d}", f"progress={progress:.0%}", f"urgency={urgency:.2f}"]
    conf = 0.40 + progress * 0.25   # generic gets lower base confidence

    return home_win_prob, conf, factors


# ── Confidence scoring ────────────────────────────────────────────────────────

def _compute_confidence(
    model_conf:        float,
    gs:                GameState,
    market_yes_prob:   float,
    fair_prob:         float,
    volatility:        float,
) -> float:
    """
    Assemble final confidence score.

    Starts from model_conf (sport-specific), then:
      + Agreement bonus: model and market directionally agree
      − Volatility penalty: high recent price swings
      − Late chaos penalty: one-score/one-goal game in final minutes
      − Missing possession penalty
    """
    conf = model_conf

    # Agreement: market leaning same direction as model (but not anchoring on value)
    model_dir  = 1 if fair_prob > 0.5 else -1
    market_dir = 1 if market_yes_prob > 0.5 else -1
    if model_dir == market_dir:
        conf *= 1.10   # slight boost for directional agreement
    else:
        conf *= 0.80   # penalty for disagreement

    # Volatility penalty
    if volatility > 0:
        vol_penalty = min(0.40, volatility * HIGH_VOLATILITY_PENALTY * 10)
        conf -= vol_penalty

    # Data freshness: if game state is close to stale
    age = gs.last_updated
    import time as _time
    data_age_secs = _time.time() - age
    if data_age_secs > 60:
        conf *= max(0.5, 1.0 - (data_age_secs - 60) / 120)

    return max(0.05, min(0.95, conf))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    """Standard logistic sigmoid: maps any real → (0, 1)."""
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def _parse_soccer_clock(clock_str: str) -> Optional[float]:
    """Parse soccer minute display ('73', '73+2', '45+1') → float minutes."""
    if not clock_str:
        return None
    import re
    m = re.match(r"(\d+)(?:\+(\d+))?", clock_str.strip())
    if not m:
        return None
    base = float(m.group(1))
    extra = float(m.group(2)) if m.group(2) else 0.0
    return base + extra
