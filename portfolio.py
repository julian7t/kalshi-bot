"""
portfolio.py — Portfolio construction and correlation-aware allocation layer.

Purpose
-------
  1. Build a live snapshot of current exposure per sport / event / market_type.
  2. For every candidate signal, compute:
       * overlap classification  (none / low / medium / high)
       * concentration score     (0 = isolated, 1 = fully concentrated)
       * allocation rank         (adjusted edge accounting for portfolio fit)
       * portfolio-aware size    (size_multiplier: 0.25–1.0)
       * approved / rejected     (with explicit reason)
  3. Rank all candidates so the best portfolio-fit subset is evaluated first.
     Within the scan, tentatively accepted candidates update the snapshot so
     later candidates see the revised exposure before their own evaluation.
  4. Emit a structured [PORTFOLIO AUDIT] log row for every candidate.

Interaction with existing systems
----------------------------------
  * READ-ONLY from DB.  Never modifies orders, positions, or trade records.
  * Does NOT replace risk_manager — hard risk rules still apply afterward.
  * Does NOT auto-apply optimization results.
  * Called from bot.py after scanner.scan() and before the execution loop.

Public API
----------
    analyzer = PortfolioAnalyzer()
    snapshot = analyzer.build_snapshot()               # once per scan
    ranked   = analyzer.rank_and_evaluate(signals, snapshot)
    for sig, ev in ranked:
        if not ev.approved:
            continue
        contracts = max(MIN_CONTRACTS, int(base_contracts * ev.size_multiplier))
        analytics.register_entry_context(ticker, {
            ...,
            "overlap_level":                  ev.overlap_level,
            "concentration_score":            ev.concentration_score,
            "allocation_rank":                ev.allocation_rank,
            "portfolio_event_exposure_cents": ev.current_event_exposure_cents,
            "portfolio_sport_exposure_cents": ev.current_sport_exposure_cents,
        })
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import db

logger = logging.getLogger("kalshi_bot.portfolio")


# ── Portfolio config (imported from config.py, fallback to env / defaults) ────

def _int(k, d):  return int(os.environ.get(k, str(d)))
def _bool(k, d): return os.environ.get(k, str(d)).lower() == "true"

try:
    from config import (
        PORTFOLIO_MAX_EXPOSURE_PER_EVENT,
        PORTFOLIO_MAX_EXPOSURE_PER_SPORT,
        PORTFOLIO_MAX_EXPOSURE_PER_MTYPE,
        PORTFOLIO_MAX_POSITIONS_PER_EVENT,
        PORTFOLIO_REJECT_HIGH_OVERLAP,
    )
except ImportError:
    PORTFOLIO_MAX_EXPOSURE_PER_EVENT  = _int("PORTFOLIO_MAX_EXPOSURE_PER_EVENT",  2000)
    PORTFOLIO_MAX_EXPOSURE_PER_SPORT  = _int("PORTFOLIO_MAX_EXPOSURE_PER_SPORT",  2000)
    PORTFOLIO_MAX_EXPOSURE_PER_MTYPE  = _int("PORTFOLIO_MAX_EXPOSURE_PER_MTYPE",  1500)
    PORTFOLIO_MAX_POSITIONS_PER_EVENT = _int("PORTFOLIO_MAX_POSITIONS_PER_EVENT",    3)
    PORTFOLIO_REJECT_HIGH_OVERLAP     = _bool("PORTFOLIO_REJECT_HIGH_OVERLAP",  True)


# ── Overlap rules ─────────────────────────────────────────────────────────────
# Pairs of market_types that share the same underlying game script
# (i.e. the same signal sources the same directional bet on the same outcome).

_HIGHLY_CORRELATED = frozenset([
    frozenset(["game_winner",   "spread"]),
    frozenset(["game_winner",   "quarter_winner"]),
    frozenset(["game_winner",   "half_winner"]),
    frozenset(["game_winner",   "period_winner"]),
    frozenset(["game_winner",   "inning_winner"]),
    frozenset(["game_winner",   "set_winner"]),
    frozenset(["spread",        "quarter_winner"]),
    frozenset(["spread",        "half_winner"]),
    frozenset(["spread",        "period_winner"]),
])

_MODERATELY_CORRELATED = frozenset([
    frozenset(["game_winner",   "totals"]),
    frozenset(["spread",        "totals"]),
    frozenset(["quarter_winner","half_winner"]),
    frozenset(["quarter_winner","period_winner"]),
    frozenset(["inning_winner", "half_winner"]),
])

_OVERLAP_PENALTY = {"none": 0.0, "low": 0.05, "medium": 0.20, "high": 0.50}
_OVERLAP_ORDER   = ["none", "low", "medium", "high"]


def _pair_overlap(
    cand_mtype:    str,
    cand_side:     str,
    exist_mtype:   str,
    exist_side:    str,
    same_event:    bool,
) -> str:
    """
    Return the overlap level for one (candidate, existing) position pair.
    Deterministic — no randomness, no market anchoring.
    """
    if not same_event:
        return "none"

    pair = frozenset([cand_mtype, exist_mtype])

    # Exact same market_type, same directional side → duplicate idea
    if cand_mtype == exist_mtype and cand_side == exist_side:
        return "high"

    # Same side, highly correlated types → also high
    if cand_side == exist_side and pair in _HIGHLY_CORRELATED:
        return "high"

    # Same side, moderately correlated types
    if cand_side == exist_side and pair in _MODERATELY_CORRELATED:
        return "medium"

    # Opposite side, same market type (potential hedge)
    if cand_mtype == exist_mtype and cand_side != exist_side:
        return "low"

    # Same event, different market type, different side
    if cand_side != exist_side:
        return "low"

    # Same event, same side, uncorrelated types
    return "medium"


def _worst_overlap(levels: list[str]) -> str:
    if not levels:
        return "none"
    return max(levels, key=lambda l: _OVERLAP_ORDER.index(l))


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class PositionContext:
    """Enriched view of one open position (live DB position + analytics context)."""
    ticker:      str
    side:        str
    contracts:   int
    cost_cents:  float    # contracts × avg_entry_cents
    sport:       str  = "unknown"
    event_id:    str  = ""
    market_type: str  = "misc"


@dataclass
class PortfolioSnapshot:
    """
    Immutable-ish view of current portfolio, updated in-memory
    within a scan as candidates are tentatively accepted.
    """
    positions: list = field(default_factory=list)  # list[PositionContext]

    # Aggregates — rebuilt on every change
    _event_exposure:  dict = field(default_factory=dict, repr=False)
    _sport_exposure:  dict = field(default_factory=dict, repr=False)
    _mtype_exposure:  dict = field(default_factory=dict, repr=False)
    _event_positions: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self._rebuild()

    def _rebuild(self):
        self._event_exposure  = {}
        self._sport_exposure  = {}
        self._mtype_exposure  = {}
        self._event_positions = {}
        for p in self.positions:
            ev  = p.event_id    or ""
            sp  = p.sport       or "unknown"
            mt  = p.market_type or "misc"
            self._event_exposure[ev]  = self._event_exposure.get(ev, 0.0)  + p.cost_cents
            self._sport_exposure[sp]  = self._sport_exposure.get(sp, 0.0)  + p.cost_cents
            self._mtype_exposure[mt]  = self._mtype_exposure.get(mt, 0.0)  + p.cost_cents
            self._event_positions[ev] = self._event_positions.get(ev, 0)   + 1

    def event_exposure(self, event_id: str) -> float:
        return self._event_exposure.get(event_id or "", 0.0)

    def sport_exposure(self, sport: str) -> float:
        return self._sport_exposure.get(sport or "unknown", 0.0)

    def mtype_exposure(self, market_type: str) -> float:
        return self._mtype_exposure.get(market_type or "misc", 0.0)

    def positions_in_event(self, event_id: str) -> int:
        return self._event_positions.get(event_id or "", 0)

    def add_tentative(self, pos: PositionContext):
        """
        Register a just-accepted candidate so subsequent candidates within
        the same scan see updated exposure.  Call after approving a signal.
        """
        self.positions.append(pos)
        self._rebuild()


@dataclass
class CandidateEvaluation:
    """Full portfolio evaluation result for one candidate signal."""
    ticker:      str
    sport:       str
    event_id:    str
    market_type: str
    side:        str

    # Portfolio state at evaluation time
    current_event_exposure_cents:  float
    current_sport_exposure_cents:  float
    current_mtype_exposure_cents:  float
    positions_in_event:            int

    # Overlap
    overlap_level:  str     # none | low | medium | high
    overlap_detail: str     # human-readable explanation

    # Scores
    concentration_score: float  # 0 = isolated, 1 = fully concentrated
    allocation_rank:     float  # higher = better candidate
    size_multiplier:     float  # 0.25–1.0 applied to contract count

    # Decision
    approved:         bool
    rejection_reason: str  # "" if approved


# ── Main analyser ─────────────────────────────────────────────────────────────

class PortfolioAnalyzer:
    """
    Portfolio construction and correlation-aware allocation layer.

    Create once per bot instance.  Every scan cycle:
      1. Call build_snapshot() to read current DB state.
      2. Call rank_and_evaluate(signals, snapshot) to rank + filter candidates.
      3. Use ev.size_multiplier in contract sizing for approved candidates.
    """

    # ── Snapshot construction ─────────────────────────────────────────────────

    def build_snapshot(self) -> PortfolioSnapshot:
        """
        Build a portfolio snapshot from DB positions + trade_analytics context.
        Always call at the start of each scan, before rank_and_evaluate().
        """
        raw_positions = db.get_all_positions()
        if not raw_positions:
            return PortfolioSnapshot(positions=[])

        # Enrich position with sport / event / market_type from trade_analytics.
        # Use the most-recent open analytics record for each ticker.
        open_analytics = db.get_open_trade_analytics()
        ctx_by_ticker: dict[str, dict] = {}
        for ta in open_analytics:
            t = ta.get("ticker", "")
            if t and t not in ctx_by_ticker:
                ctx_by_ticker[t] = ta

        enriched: list[PositionContext] = []
        for p in raw_positions:
            ticker    = p.get("ticker", "?")
            side      = p.get("side", "yes")
            contracts = int(p.get("contracts") or 0)
            entry_c   = float(p.get("avg_entry_cents") or 0)
            cost      = contracts * entry_c
            ta        = ctx_by_ticker.get(ticker, {})
            enriched.append(PositionContext(
                ticker      = ticker,
                side        = side,
                contracts   = contracts,
                cost_cents  = cost,
                sport       = ta.get("sport",            "unknown") or "unknown",
                event_id    = ta.get("matched_event_id", "")        or "",
                market_type = ta.get("market_type",      "misc")    or "misc",
            ))

        snap = PortfolioSnapshot(positions=enriched)
        logger.debug(
            "[PORTFOLIO] Snapshot: %d positions | events=%s | sports=%s",
            len(enriched),
            {k: f"${v/100:.2f}" for k, v in snap._event_exposure.items() if k},
            {k: f"${v/100:.2f}" for k, v in snap._sport_exposure.items() if k},
        )
        return snap

    # ── Rank + evaluate ───────────────────────────────────────────────────────

    def rank_and_evaluate(
        self,
        signals:  list[dict],
        snapshot: PortfolioSnapshot,
    ) -> list[tuple[dict, CandidateEvaluation]]:
        """
        Evaluate all candidate signals against the portfolio snapshot.
        Returns (signal, CandidateEvaluation) pairs sorted by allocation_rank
        descending (best portfolio fit first).

        Approved candidates are added tentatively to the snapshot so later
        candidates within the same scan see updated exposure.
        """
        if not signals:
            return []

        # First pass — score with current snapshot (for initial sort only)
        scored: list[tuple[dict, CandidateEvaluation]] = [
            (sig, self._evaluate_candidate(sig, snapshot)) for sig in signals
        ]
        scored.sort(key=lambda se: se[1].allocation_rank, reverse=True)

        # Second pass — re-evaluate in rank order, updating snapshot per approval
        final: list[tuple[dict, CandidateEvaluation]] = []
        for sig, _ in scored:
            ev = self._evaluate_candidate(sig, snapshot)
            self._log_audit(sig, ev)
            if ev.approved:
                # Tentatively register so subsequent candidates see the exposure
                entry_c   = sig.get("yes_price", 50)
                if sig.get("side", "yes") == "no":
                    entry_c = 100 - entry_c
                snapshot.add_tentative(PositionContext(
                    ticker      = ev.ticker,
                    side        = ev.side,
                    contracts   = 1,          # placeholder; real size set by bot.py
                    cost_cents  = float(entry_c),
                    sport       = ev.sport,
                    event_id    = ev.event_id,
                    market_type = ev.market_type,
                ))
            final.append((sig, ev))

        approved_n = sum(1 for _, ev in final if ev.approved)
        logger.info(
            "[PORTFOLIO] %d/%d candidates approved after portfolio evaluation.",
            approved_n, len(final),
        )
        return final

    # ── Per-candidate evaluation ──────────────────────────────────────────────

    def _evaluate_candidate(
        self,
        sig:      dict,
        snapshot: PortfolioSnapshot,
    ) -> CandidateEvaluation:
        ticker      = sig.get("ticker", "?")
        sport       = sig.get("sport", "unknown") or "unknown"
        event_id    = sig.get("matched_event_id") or ""
        market_type = sig.get("market_type") or "misc"
        side        = sig.get("side", "yes")
        net_edge    = float(sig.get("net_edge", 0))
        confidence  = float(sig.get("confidence_score", 0))

        # Current exposure for this candidate's dimensions
        ev_exp    = snapshot.event_exposure(event_id)
        sp_exp    = snapshot.sport_exposure(sport)
        mt_exp    = snapshot.mtype_exposure(market_type)
        pos_count = snapshot.positions_in_event(event_id)

        # Overlap detection — evaluate against every existing position
        overlap_levels:  list[str] = []
        overlap_details: list[str] = []
        for pos in snapshot.positions:
            same_event = bool(event_id and event_id == pos.event_id)
            ol = _pair_overlap(market_type, side, pos.market_type, pos.side, same_event)
            if ol != "none":
                overlap_levels.append(ol)
                overlap_details.append(
                    f"{pos.ticker}({pos.market_type}/{pos.side})→{ol}"
                )

        worst_ol    = _worst_overlap(overlap_levels)
        overlap_str = "; ".join(overlap_details) if overlap_details else "none"

        # Concentration score (0 = isolated, 1 = maxed out)
        ev_ratio = (ev_exp / PORTFOLIO_MAX_EXPOSURE_PER_EVENT) if PORTFOLIO_MAX_EXPOSURE_PER_EVENT else 0.0
        sp_ratio = (sp_exp / PORTFOLIO_MAX_EXPOSURE_PER_SPORT) if PORTFOLIO_MAX_EXPOSURE_PER_SPORT else 0.0
        mt_ratio = (mt_exp / PORTFOLIO_MAX_EXPOSURE_PER_MTYPE) if PORTFOLIO_MAX_EXPOSURE_PER_MTYPE else 0.0
        conc_score = min(1.0, max(ev_ratio, sp_ratio, mt_ratio) + _OVERLAP_PENALTY[worst_ol])

        # Allocation rank — higher = better portfolio fit
        # net_edge × confidence × diversification bonus
        divers_bonus = max(0.0, 1.0 - conc_score)
        alloc_rank   = net_edge * confidence * (0.5 + 0.5 * divers_bonus)

        # Portfolio-aware size multiplier
        # Reduces size as concentration rises; never below 0.25 × baseline.
        size_mult = round(max(0.25, 1.0 - conc_score * 0.75), 3)

        # Hard portfolio rejection checks (in priority order)
        rejection = ""
        if PORTFOLIO_REJECT_HIGH_OVERLAP and worst_ol == "high":
            rejection = (
                f"portfolio overlap=HIGH ({overlap_str}) — "
                f"same idea already captured"
            )
        elif event_id and pos_count >= PORTFOLIO_MAX_POSITIONS_PER_EVENT:
            rejection = (
                f"portfolio: {pos_count} positions in event {event_id!r} "
                f"(max={PORTFOLIO_MAX_POSITIONS_PER_EVENT})"
            )
        elif event_id and ev_exp >= PORTFOLIO_MAX_EXPOSURE_PER_EVENT:
            rejection = (
                f"portfolio: event exposure ${ev_exp/100:.2f} >= "
                f"limit ${PORTFOLIO_MAX_EXPOSURE_PER_EVENT/100:.2f}"
            )
        elif sp_exp >= PORTFOLIO_MAX_EXPOSURE_PER_SPORT:
            rejection = (
                f"portfolio: sport exposure ${sp_exp/100:.2f} >= "
                f"limit ${PORTFOLIO_MAX_EXPOSURE_PER_SPORT/100:.2f} ({sport})"
            )

        return CandidateEvaluation(
            ticker                       = ticker,
            sport                        = sport,
            event_id                     = event_id,
            market_type                  = market_type,
            side                         = side,
            current_event_exposure_cents = ev_exp,
            current_sport_exposure_cents = sp_exp,
            current_mtype_exposure_cents = mt_exp,
            positions_in_event           = pos_count,
            overlap_level                = worst_ol,
            overlap_detail               = overlap_str,
            concentration_score          = round(conc_score, 4),
            allocation_rank              = round(alloc_rank, 6),
            size_multiplier              = size_mult,
            approved                     = not rejection,
            rejection_reason             = rejection,
        )

    # ── Audit logging ─────────────────────────────────────────────────────────

    def _log_audit(self, sig: dict, ev: CandidateEvaluation):
        """Emit one structured [PORTFOLIO AUDIT] log row per evaluated candidate."""
        decision = "APPROVED" if ev.approved else "REJECTED"
        logger.info(
            "[PORTFOLIO AUDIT] %-24s | %-10s | event=%-18s | mtype=%-16s | side=%s"
            " | ev_exp=$%.2f sp_exp=$%.2f pos=%d"
            " | overlap=%-6s conc=%.0f%% rank=%.4f size=%.2f"
            " | %s%s",
            ev.ticker, ev.sport, ev.event_id or "—", ev.market_type, ev.side,
            ev.current_event_exposure_cents / 100,
            ev.current_sport_exposure_cents / 100,
            ev.positions_in_event,
            ev.overlap_level,
            ev.concentration_score * 100,
            ev.allocation_rank,
            ev.size_multiplier,
            decision,
            f": {ev.rejection_reason}" if ev.rejection_reason else "",
        )


# ── Concentration bucket helpers (used by analytics.py) ──────────────────────

def concentration_bucket(score: float) -> str:
    """Map a concentration score (0–1) to a human-readable bucket label."""
    if score < 0.25:
        return "0-25%"
    if score < 0.50:
        return "25-50%"
    if score < 0.75:
        return "50-75%"
    return "75-100%"
