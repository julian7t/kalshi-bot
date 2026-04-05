"""
signal_models.py — Market-type specialized fair-probability models.

Each model produces an independent fair probability from game state.
Fair probability is NEVER anchored to the Kalshi market price.

Models:
  GameWinnerModel    — full-game moneyline / outright winner
  SegmentWinnerModel — quarter / half / period / inning / set winner
  TotalsModel        — over / under total score
  SpreadModel        — point spread / handicap
  PlayerPropModel    — player-level markets (unsupported — no-trade)
  MiscModel          — fallback for unclassifiable markets

Public API:
    result = dispatch(market_type, game_state, sport, ticker, side,
                      market_yes_prob, parse_info, title="")
    # result is always a ModelResult; check result.can_trade before using
"""

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("kalshi_bot.signal_models")

# Minimum confidence to trade a FUTURE segment market.
# Env var takes priority at call-time (supports runtime overrides and tests);
# falls back to config.py constant, then hardcoded default 0.50.
def _min_future_seg_conf() -> float:
    raw = os.environ.get("MIN_FUTURE_SEGMENT_CONFIDENCE")
    if raw is not None:
        return float(raw)
    try:
        from config import MIN_FUTURE_SEGMENT_CONFIDENCE
        return MIN_FUTURE_SEGMENT_CONFIDENCE
    except Exception:
        return 0.50

# ── Scoring baselines by sport (used for pace estimates) ─────────────────────
# ppg = single-team points per game; variance = single-team std dev
_SCORING = {
    "BASKETBALL": {"ppg": 112.0, "variance": 10.0},   # NBA ~225 combined
    "FOOTBALL":   {"ppg":  23.0, "variance":  7.0},   # NFL ~46 combined
    "BASEBALL":   {"ppg":   4.5, "variance":  2.5},   # MLB ~9 combined
    "HOCKEY":     {"ppg":   3.0, "variance":  1.5},   # NHL ~6 combined
    "SOCCER":     {"ppg":   1.4, "variance":  1.0},   # MLS/EPL ~2.8 combined
    "GENERIC":    {"ppg":  20.0, "variance":  8.0},
}

# Market types that need a segment reference
_SEGMENT_TYPES = frozenset({
    "quarter_winner", "half_winner", "period_winner",
    "inning_winner", "set_winner",
})


# ── ModelResult ───────────────────────────────────────────────────────────────

@dataclass
class ModelResult:
    """Return type for every model.evaluate() call."""
    fair_probability:  float
    confidence:        float
    model_name:        str
    reason:            str
    can_trade:         bool
    no_trade_reason:   Optional[str]
    parsed_line:       Optional[float] = None
    projected_value:   Optional[float] = None
    factors:           list = field(default_factory=list)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def _clamp(v: float, lo: float = 0.04, hi: float = 0.96) -> float:
    return max(lo, min(hi, v))


def _no_trade(model_name: str, reason: str) -> ModelResult:
    logger.info("[SIGNAL_MODELS] no-trade %s — %s", model_name, reason)
    return ModelResult(
        fair_probability=0.5, confidence=0.0,
        model_name=model_name, reason=reason,
        can_trade=False, no_trade_reason=reason,
    )


# ── Game Winner Model ─────────────────────────────────────────────────────────

class GameWinnerModel:
    """
    Full-game moneyline / outright winner.

    Delegates to the sport-specific probability functions in model.py
    via a lazy import (avoids circular imports at module load time).
    """
    NAME = "game_winner"

    def evaluate(
        self,
        game_state,
        sport:           str,
        ticker:          str,
        side:            str,
        market_yes_prob: float,
    ) -> ModelResult:
        import model as _m
        result = _m._compute_fair_probability(game_state, sport, ticker)
        if result is None:
            return _no_trade(self.NAME, "sport model returned None")

        home_win_prob, model_conf, factors = result

        # Map home_win_prob to YES-side probability
        fair_prob = home_win_prob if market_yes_prob >= 0.5 else 1.0 - home_win_prob
        fair_prob = _clamp(fair_prob)

        gs = game_state
        reason = (
            f"game_winner/{sport}: {gs.home_abbr} {gs.home_score}–"
            f"{gs.away_score} {gs.away_abbr} | "
            f"{gs.game_clock} P{gs.period}/{gs.total_periods} | "
            + " | ".join(factors)
        )
        logger.debug("[SIGNAL_MODELS] game_winner %s fair=%.0f%% conf=%.0f%%",
                     ticker, fair_prob * 100, model_conf * 100)
        return ModelResult(
            fair_probability=round(fair_prob, 4),
            confidence=round(model_conf, 4),
            model_name=self.NAME,
            reason=reason,
            can_trade=True,
            no_trade_reason=None,
            factors=factors,
        )


# ── Segment Winner Model ──────────────────────────────────────────────────────

class SegmentWinnerModel:
    """
    Intra-game segment winner: quarter, half, period, inning, set.

    Three cases based on how the target segment relates to the current one:

    CURRENT segment  — short-horizon model using score differential and time
                       remaining in the segment.  Higher sensitivity than full game.

    FUTURE segment   — use full-game team strength as a proxy, with a large
                       confidence penalty (we can't see the future segment state).

    PAST segment     — no-trade; the market should already be settled.

    Period reference (period_ref) comes from parser.parse_period_ref().
    If it could not be parsed, fall back to GameWinner with a confidence penalty.
    """
    NAME = "segment_winner"

    def evaluate(
        self,
        game_state,
        sport:           str,
        ticker:          str,
        side:            str,
        market_yes_prob: float,
        period_ref:      Optional[int],
        period_type:     str,
        market_type:     str,
    ) -> ModelResult:
        from game_state import _parse_clock_seconds, _period_minutes

        model_name     = f"segment_{market_type}"
        current_period = game_state.period
        total_periods  = game_state.total_periods
        diff           = game_state.score_differential
        clock_secs     = _parse_clock_seconds(game_state.game_clock)
        period_mins    = _period_minutes(game_state.league, total_periods) or 12.0

        # ── No period reference ───────────────────────────────────────────────
        if period_ref is None:
            # Log and fall back to game_winner with penalty
            logger.info(
                "[SIGNAL_MODELS] %s %s: period_ref unknown — confidence-penalized fallback",
                model_name, ticker,
            )
            min_conf  = _min_future_seg_conf()
            gw        = GameWinnerModel().evaluate(game_state, sport, ticker, side, market_yes_prob)
            penalized = round(max(0.10, gw.confidence * 0.45), 4)
            can_trade = penalized >= min_conf
            return ModelResult(
                fair_probability=gw.fair_probability,
                confidence=penalized,
                model_name=model_name,
                reason=(
                    f"{model_name}: period ref unknown; fallback to game_winner with penalty | "
                    f"conf={penalized:.0%} threshold={min_conf:.0%}"
                ),
                can_trade=can_trade,
                no_trade_reason=(
                    None if can_trade else
                    f"segment market: period ref unknown, proxy confidence {penalized:.0%} < "
                    f"MIN_FUTURE_SEGMENT_CONFIDENCE {min_conf:.0%}"
                ),
                factors=gw.factors,
            )

        # ── Past segment ──────────────────────────────────────────────────────
        if period_ref < current_period:
            return _no_trade(
                model_name,
                f"market is for past segment (target={period_ref} current={current_period})"
            )

        # ── Current segment ───────────────────────────────────────────────────
        if period_ref == current_period:
            total_seg_secs = period_mins * 60
            if clock_secs is not None:
                seg_remaining  = clock_secs
                clock_known    = True
            else:
                seg_remaining  = total_seg_secs * 0.50  # mid-segment estimate
                clock_known    = False

            seg_elapsed_frac   = max(0.001, (total_seg_secs - seg_remaining) / total_seg_secs)
            seg_remaining_frac = max(0.001, seg_remaining / total_seg_secs)

            # Short-horizon sensitivity: rises steeply as segment ends
            sensitivity = 0.22 / seg_remaining_frac
            sensitivity = min(sensitivity, 12.0)

            x = diff * sensitivity / 10.0
            home_seg_prob = _sigmoid(x)
            home_seg_prob = _clamp(home_seg_prob)
            fair_prob = home_seg_prob if market_yes_prob >= 0.5 else 1.0 - home_seg_prob

            # Confidence: higher as segment progresses; penalise missing clock
            conf = 0.45 if clock_known else 0.25
            conf += seg_elapsed_frac * 0.25   # grows 0→0.25 through segment
            conf = _clamp(conf, 0.10, 0.80)

            factors = [
                f"seg={period_ref}/{total_periods}",
                f"diff={diff:+d}",
                f"seg_rem={seg_remaining:.0f}s",
                f"sens={sensitivity:.2f}",
                f"clock_known={clock_known}",
            ]
            logger.debug("[SIGNAL_MODELS] %s %s current-seg fair=%.0f%% conf=%.0f%%",
                         model_name, ticker, fair_prob * 100, conf * 100)
            return ModelResult(
                fair_probability=round(fair_prob, 4),
                confidence=round(conf, 4),
                model_name=model_name,
                reason=(
                    f"{model_name}/live: seg {period_ref}/{total_periods} | "
                    + " | ".join(factors)
                ),
                can_trade=True,
                no_trade_reason=None,
                factors=factors,
            )

        # ── Future segment ────────────────────────────────────────────────────
        min_conf  = _min_future_seg_conf()
        gw        = GameWinnerModel().evaluate(game_state, sport, ticker, side, market_yes_prob)
        adj_conf  = round(max(0.10, gw.confidence * 0.45), 4)
        can_trade = adj_conf >= min_conf
        no_trade_reason = (
            None if can_trade else
            f"future segment: proxy confidence {adj_conf:.0%} < "
            f"MIN_FUTURE_SEGMENT_CONFIDENCE {min_conf:.0%}"
        )
        reason = (
            f"{model_name}/future: target seg {period_ref} (current={current_period}); "
            f"team-strength proxy | conf={adj_conf:.0%} threshold={min_conf:.0%}"
        )
        logger.debug(
            "[SIGNAL_MODELS] %s %s future-seg fair=%.0f%% conf=%.0f%% "
            "threshold=%.0f%% trade=%s",
            model_name, ticker, gw.fair_probability * 100,
            adj_conf * 100, min_conf * 100, can_trade,
        )
        return ModelResult(
            fair_probability=gw.fair_probability,
            confidence=adj_conf,
            model_name=model_name,
            reason=reason,
            can_trade=can_trade,
            no_trade_reason=no_trade_reason,
            factors=gw.factors,
        )


# ── Totals Model ──────────────────────────────────────────────────────────────

class TotalsModel:
    """
    Over / under total combined score model.

    Algorithm:
      1. current_total = home_score + away_score
      2. pace_projection = current_total / elapsed_fraction
      3. blend pace with sport average (pace weight grows as game progresses)
      4. gap = projected_total - total_line
      5. fair_over_prob = sigmoid(gap / uncertainty)
      6. Map to YES side using market implied direction
    """
    NAME = "totals"

    def evaluate(
        self,
        game_state,
        sport:           str,
        ticker:          str,
        side:            str,
        market_yes_prob: float,
        total_line:      Optional[float],
    ) -> ModelResult:
        if total_line is None:
            return _no_trade(
                self.NAME,
                "total line not parseable from ticker/title — cannot price over-under market",
            )

        gs             = game_state
        current_total  = gs.home_score + gs.away_score
        elapsed_frac   = max(0.001, gs.game_progress)
        remaining_frac = max(0.001, 1.0 - elapsed_frac)

        baseline     = _SCORING.get(sport, _SCORING["GENERIC"])
        sport_avg    = baseline["ppg"] * 2           # combined ppg
        sport_var    = baseline["variance"] * math.sqrt(2)  # combined std dev

        # Pace projection: how fast the game is scoring
        if elapsed_frac > 0.10:
            pace_proj = current_total / elapsed_frac
        else:
            pace_proj = sport_avg   # too early — trust league average

        # Blend: weight pace heavier as game goes on
        blend_wt      = min(0.92, elapsed_frac * 1.8)
        projected     = pace_proj * blend_wt + sport_avg * (1.0 - blend_wt)

        gap           = projected - total_line
        uncertainty   = remaining_frac * sport_var + 2.0   # floor uncertainty

        fair_over_p   = _sigmoid(gap / uncertainty)
        fair_over_p   = _clamp(fair_over_p, 0.05, 0.95)

        # Map to YES side:
        # If market_yes_prob >= 0.5 → market prices YES as likely → YES is the "over" side
        # If market_yes_prob < 0.5  → YES is the "under" side
        fair_yes_p = fair_over_p if market_yes_prob >= 0.5 else (1.0 - fair_over_p)

        # Confidence
        conf = 0.45
        conf += 0.12 if elapsed_frac > 0.25 else 0.0
        conf += 0.10 if elapsed_frac > 0.55 else 0.0
        conf -= 0.12 if remaining_frac < 0.12 else 0.0  # late chaos
        if gs.total_periods <= 0:
            conf = 0.20   # no game structure

        factors = [
            f"current={current_total}",
            f"line={total_line}",
            f"projected={projected:.1f}",
            f"gap={gap:+.1f}",
            f"elapsed={elapsed_frac:.0%}",
            f"uncertainty={uncertainty:.1f}",
        ]
        reason = (
            f"totals/{sport}: projected={projected:.1f} vs line={total_line} "
            f"(gap={gap:+.1f}) | " + " | ".join(factors)
        )
        logger.debug("[SIGNAL_MODELS] totals %s line=%.1f proj=%.1f fair_over=%.0f%% conf=%.0f%%",
                     ticker, total_line, projected, fair_over_p * 100, conf * 100)
        return ModelResult(
            fair_probability=round(fair_yes_p, 4),
            confidence=round(_clamp(conf, 0.15, 0.80), 4),
            model_name=self.NAME,
            reason=reason,
            can_trade=True,
            no_trade_reason=None,
            parsed_line=total_line,
            projected_value=round(projected, 1),
            factors=factors,
        )


# ── Spread Model ──────────────────────────────────────────────────────────────

class SpreadModel:
    """
    Point spread / handicap model.

    Algorithm:
      1. diff = home_score - away_score  (current margin)
      2. projected_margin = diff * (1 - remaining_frac * reversion) + home_edge
      3. gap = projected_margin - needed_margin_to_cover
      4. fair_cover_prob = sigmoid(gap / uncertainty)
      5. Map to YES side using market implied direction
    """
    NAME = "spread"

    # Slight home-field advantage proxy (in points)
    _HOME_EDGE = 1.5

    def evaluate(
        self,
        game_state,
        sport:           str,
        ticker:          str,
        side:            str,
        market_yes_prob: float,
        spread_line:     Optional[float],
    ) -> ModelResult:
        if spread_line is None:
            return _no_trade(
                self.NAME,
                "spread line not parseable from ticker/title — cannot price spread market",
            )

        gs             = game_state
        diff           = gs.score_differential        # home - away
        elapsed_frac   = max(0.001, gs.game_progress)
        remaining_frac = max(0.001, 1.0 - elapsed_frac)

        baseline     = _SCORING.get(sport, _SCORING["GENERIC"])
        sport_var    = baseline["variance"]

        # Projected margin regresses slightly toward home-edge as time remains
        reversion        = 0.25   # how much current margin regresses per remaining fraction
        projected_margin = (
            diff * (1.0 - remaining_frac * reversion)
            + self._HOME_EDGE * remaining_frac * reversion
        )

        # "Cover the spread" for home team:
        #   spread_line = -7 means home is favored by 7 (must win by >7)
        #   needed = 7 (positive)
        needed = -spread_line   # positive = home must win by this many
        gap    = projected_margin - needed

        uncertainty = remaining_frac * sport_var + 1.5
        fair_cover  = _sigmoid(gap / uncertainty)
        fair_cover  = _clamp(fair_cover, 0.05, 0.95)

        # Map to YES side
        fair_yes_p = fair_cover if market_yes_prob >= 0.5 else (1.0 - fair_cover)

        conf = 0.48
        conf += 0.12 if elapsed_frac > 0.35 else 0.0
        conf += 0.08 if elapsed_frac > 0.65 else 0.0
        conf -= 0.12 if remaining_frac < 0.10 else 0.0

        factors = [
            f"diff={diff:+d}",
            f"spread_line={spread_line:+.1f}",
            f"proj_margin={projected_margin:+.1f}",
            f"gap={gap:+.1f}",
            f"elapsed={elapsed_frac:.0%}",
        ]
        reason = (
            f"spread/{sport}: proj_margin={projected_margin:+.1f} vs line={spread_line:+.1f} "
            f"(gap={gap:+.1f}) | " + " | ".join(factors)
        )
        logger.debug("[SIGNAL_MODELS] spread %s line=%+.1f proj=%+.1f fair_cover=%.0f%% conf=%.0f%%",
                     ticker, spread_line, projected_margin, fair_cover * 100, conf * 100)
        return ModelResult(
            fair_probability=round(fair_yes_p, 4),
            confidence=round(_clamp(conf, 0.15, 0.80), 4),
            model_name=self.NAME,
            reason=reason,
            can_trade=True,
            no_trade_reason=None,
            parsed_line=spread_line,
            projected_value=round(projected_margin, 1),
            factors=factors,
        )


# ── Player Prop Model ─────────────────────────────────────────────────────────

class PlayerPropModel:
    """
    Player prop — explicitly unsupported.

    We don't have player-level performance data, so we cannot price these
    markets independently.  Always returns no-trade.
    """
    NAME = "player_prop"

    def evaluate(self, game_state, sport, ticker, side, market_yes_prob, **kwargs) -> ModelResult:
        return _no_trade(
            self.NAME,
            "player props: no player-level data available — market type not supported",
        )


# ── Misc / Fallback Model ─────────────────────────────────────────────────────

class MiscModel:
    """
    Fallback for unclassifiable or explicitly unsupported market types.

    Uses game_winner logic with a significant confidence penalty.
    Returns no-trade if penalized confidence is too low.
    """
    NAME = "misc_fallback"

    def evaluate(self, game_state, sport, ticker, side, market_yes_prob, **kwargs) -> ModelResult:
        gw  = GameWinnerModel().evaluate(game_state, sport, ticker, side, market_yes_prob)
        pen = round(max(0.05, gw.confidence * 0.45), 4)
        return ModelResult(
            fair_probability=gw.fair_probability,
            confidence=pen,
            model_name=self.NAME,
            reason=f"misc_fallback: {gw.reason[:120]}",
            can_trade=pen >= 0.35,
            no_trade_reason=(
                None if pen >= 0.35
                else "misc market: confidence after penalty too low to trade"
            ),
            factors=gw.factors,
        )


# ── Dispatcher ────────────────────────────────────────────────────────────────

def dispatch(
    market_type:     str,
    game_state,
    sport:           str,
    ticker:          str,
    side:            str,
    market_yes_prob: float,
    parse_info:      dict,
    title:           str = "",
) -> ModelResult:
    """
    Route to the correct model based on market_type.

    Args:
        market_type     — from classifier.classify()
        game_state      — GameState object (must not be None — caller checks)
        sport           — sport label, e.g. "BASKETBALL"
        ticker          — Kalshi ticker string
        side            — "yes" | "no"
        market_yes_prob — Kalshi implied YES probability (not used as anchor)
        parse_info      — from parser.parse_for_market_type()
        title           — market title string (for logging)

    Returns ModelResult.  Always returns a value; caller checks .can_trade.
    NEVER auto-applies anything to live trading.
    """
    logger.debug(
        "[SIGNAL_MODELS] dispatch ticker=%s market_type=%s sport=%s",
        ticker, market_type, sport,
    )

    if market_type == "game_winner":
        return GameWinnerModel().evaluate(game_state, sport, ticker, side, market_yes_prob)

    if market_type in _SEGMENT_TYPES:
        return SegmentWinnerModel().evaluate(
            game_state=game_state,
            sport=sport,
            ticker=ticker,
            side=side,
            market_yes_prob=market_yes_prob,
            period_ref=parse_info.get("period"),
            period_type=parse_info.get("period_type", "unknown"),
            market_type=market_type,
        )

    if market_type == "totals":
        return TotalsModel().evaluate(
            game_state=game_state,
            sport=sport,
            ticker=ticker,
            side=side,
            market_yes_prob=market_yes_prob,
            total_line=parse_info.get("line"),
        )

    if market_type == "spread":
        return SpreadModel().evaluate(
            game_state=game_state,
            sport=sport,
            ticker=ticker,
            side=side,
            market_yes_prob=market_yes_prob,
            spread_line=parse_info.get("line"),
        )

    if market_type == "player_prop":
        return PlayerPropModel().evaluate(
            game_state=game_state, sport=sport,
            ticker=ticker, side=side, market_yes_prob=market_yes_prob,
        )

    # Unknown / misc
    logger.debug("[SIGNAL_MODELS] unknown market_type=%s → misc_fallback", market_type)
    return MiscModel().evaluate(
        game_state=game_state, sport=sport,
        ticker=ticker, side=side, market_yes_prob=market_yes_prob,
    )
