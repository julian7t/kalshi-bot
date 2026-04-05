"""
scanner.py — Orchestrated scan pipeline (v2).

Each scan cycle:
  1. Receive Kalshi markets (fetched by bot.py)
  2. Refresh live sports data → update GameStateStore
  3. Update per-ticker price history → compute volatility + momentum
  4. For each base strategy signal:
       a. Match to a live game (or reject)
       b. Run sport-specific fair-probability model
       c. Apply no-trade zones (volatility, end-of-game, data freshness)
       d. Emit full [AUDIT] log row
  5. Return approved signals sorted by net_edge

Backward-compatible with bot.py: signals still include all original keys
(ticker, title, side, yes_price, confidence, progress) plus new keys.

New keys on approved signals:
  fair_probability    float   model's YES prob (no market anchoring)
  market_probability  float   Kalshi implied YES prob
  raw_edge            float   fair - market
  net_edge            float   raw_edge * confidence
  confidence_score    float
  volatility          float   recent price std-dev (0 if insufficient history)
  momentum            float   recent price trend (positive = rising)
  sport               str
  matched_event_id    str|None
  model_used          str
  reason              str
"""

import logging
import os
import statistics
import time
from collections import deque
from typing import Optional

import config
import strategy
from event_matcher import EventMatcher
from game_state import GameState, GameStateStore
from model import evaluate_signal, MAX_VOLATILITY_TRADE
from sports_data import SportsDataAdapter, get_adapter

logger = logging.getLogger("kalshi_bot.scanner")


# ── Tunables ──────────────────────────────────────────────────────────────────

SPORTS_STALE_SECONDS = int(os.environ.get("SPORTS_STALE_SECONDS",   "120"))
REQUIRE_GAME_MATCH   = os.environ.get("REQUIRE_GAME_MATCH", "false").lower() == "true"
MIN_MODEL_CONFIDENCE = float(os.environ.get("MIN_MODEL_CONFIDENCE",  "0.35"))
MIN_NET_EDGE         = float(os.environ.get("MIN_NET_EDGE",           "0.03"))  # 3¢ after confidence
FORCE_MOCK_SPORTS    = os.environ.get("FORCE_MOCK_SPORTS",  "false").lower() == "true"
PRICE_HISTORY_LEN    = int(os.environ.get("PRICE_HISTORY_LEN",        "8"))    # last N prices
SPORTS_LEAGUES       = [
    l.strip() for l in
    os.environ.get("SPORTS_LEAGUES", "NBA,NFL,MLB,NHL,NCAAF,NCAAB,MLS").split(",")
    if l.strip()
]


# ── Price history tracking ────────────────────────────────────────────────────

class _PriceHistory:
    """
    Rolling window of recent YES prices for one ticker.
    Used to compute volatility and momentum.
    """
    __slots__ = ("_prices",)

    def __init__(self):
        self._prices: deque[float] = deque(maxlen=PRICE_HISTORY_LEN)

    def update(self, yes_price: int):
        self._prices.append(float(yes_price))

    @property
    def count(self) -> int:
        return len(self._prices)

    @property
    def volatility(self) -> float:
        """
        Standard deviation of recent prices expressed as a fraction (0–1).
        Returns 0.0 if fewer than 3 data points.
        Represents how much the market has been swinging.
        """
        if self.count < 3:
            return 0.0
        prices = list(self._prices)
        stdev  = statistics.stdev(prices)
        return stdev / 100.0   # convert cent std-dev to 0–1 fraction

    @property
    def momentum(self) -> float:
        """
        Signed momentum: latest price − oldest price in window, as fraction.
        Positive = market moving toward YES.  Negative = moving toward NO.
        """
        if self.count < 2:
            return 0.0
        prices = list(self._prices)
        return (prices[-1] - prices[0]) / 100.0

    @property
    def recent_jump(self) -> float:
        """Absolute change between last two observations (fraction)."""
        if self.count < 2:
            return 0.0
        prices = list(self._prices)
        return abs(prices[-1] - prices[-2]) / 100.0

    def classify(self) -> str:
        """Human-readable momentum label."""
        m = self.momentum
        v = self.volatility
        if v > MAX_VOLATILITY_TRADE:
            return "VOLATILE"
        if m > 0.05:
            return "RISING"
        if m < -0.05:
            return "FALLING"
        return "STABLE"


# ── Main Scanner ──────────────────────────────────────────────────────────────

class Scanner:
    """
    Long-lived scanner — create once, call scan() every iteration.
    Holds GameStateStore (accumulates across cycles) and price histories.
    """

    def __init__(self):
        self._store:         GameStateStore               = GameStateStore()
        self._matcher:       EventMatcher                 = EventMatcher()
        self._adapter:       SportsDataAdapter            = get_adapter(force_mock=FORCE_MOCK_SPORTS)
        self._price_hist:    dict[str, _PriceHistory]     = {}
        self._market_lookup: dict[str, dict]             = {}
        self._last_sports_fetch: float                    = 0.0
        self._sports_fetch_failures: int                  = 0

        logger.info(
            "[SCANNER] Initialised. adapter=%s  leagues=%s  "
            "require_match=%s  min_conf=%.0f%%  min_net_edge=%.0f%%",
            self._adapter.name, ",".join(SPORTS_LEAGUES),
            REQUIRE_GAME_MATCH, MIN_MODEL_CONFIDENCE * 100, MIN_NET_EDGE * 100,
        )

    # ── Main entry ────────────────────────────────────────────────────────────

    def scan(self, markets: list[dict]) -> list[dict]:
        """
        Full scan pipeline. Returns approved signals sorted by net_edge desc.
        """
        self._refresh_sports_data()

        sports_ok    = not self._sports_data_stale()
        active_games = len(self._store.get_active())

        logger.info(
            "[SCANNER] Sports: %s | %d active games | require_match=%s",
            "OK" if sports_ok else "STALE", active_games, REQUIRE_GAME_MATCH,
        )

        # Base strategy pass — unchanged gating on confidence + progress
        base_signals = strategy.scan_markets(markets)
        if not base_signals:
            logger.info(
                "[SCANNER] No base signals (conf≥%.0f%%, progress≥%.0f%%).",
                config.MIN_CONFIDENCE * 100, config.GAME_PROGRESS_THRESHOLD * 100,
            )
            return []

        logger.info("[SCANNER] %d base signal(s) to evaluate.", len(base_signals))

        # Update price histories from current market batch
        self._update_price_histories(markets)

        # Enrich each signal
        approved: list[dict] = []
        for sig in base_signals:
            enriched = self._evaluate(sig, sports_ok)
            if enriched is not None:
                approved.append(enriched)

        # Sort by net_edge (strongest independent edge first)
        approved.sort(key=lambda s: s.get("net_edge", 0), reverse=True)

        logger.info(
            "[SCANNER] %d/%d signal(s) approved. "
            "Top edge: %s",
            len(approved), len(base_signals),
            f"{approved[0]['ticker']} net_edge={approved[0]['net_edge']:+.0%}" if approved else "—",
        )
        return approved

    # ── Sports data refresh ───────────────────────────────────────────────────

    def _refresh_sports_data(self):
        try:
            raw = self._adapter.fetch_games()
            self._store.update_from_raw(raw)
            self._last_sports_fetch = time.time()
            self._sports_fetch_failures = 0
        except Exception as e:
            self._sports_fetch_failures += 1
            logger.warning("[SCANNER] Sports fetch failed (#%d): %s",
                           self._sports_fetch_failures, e)

    def _sports_data_stale(self) -> bool:
        return (self._last_sports_fetch == 0 or
                (time.time() - self._last_sports_fetch) > SPORTS_STALE_SECONDS)

    # ── Price history update ──────────────────────────────────────────────────

    def _update_price_histories(self, markets: list[dict]):
        """
        Feed current YES prices into per-ticker rolling windows.
        Also builds the market lookup dict so _evaluate can embed raw market data.
        """
        self._market_lookup: dict[str, dict] = {}
        for m in markets:
            ticker = m.get("ticker")
            if not ticker:
                continue
            self._market_lookup[ticker] = m
            try:
                ask = m.get("yes_ask_dollars")
                bid = m.get("yes_bid_dollars")
                if ask:
                    price = int(float(ask) * 100)
                elif bid:
                    price = int(float(bid) * 100)
                else:
                    continue
                if price <= 0 or price >= 100:
                    continue
                if ticker not in self._price_hist:
                    self._price_hist[ticker] = _PriceHistory()
                self._price_hist[ticker].update(price)
            except Exception:
                pass

    def _get_market_volatility(self, ticker: str) -> tuple[float, float, str]:
        """Return (volatility, momentum, label) for a ticker."""
        hist = self._price_hist.get(ticker)
        if hist is None:
            return 0.0, 0.0, "NO_HISTORY"
        return hist.volatility, hist.momentum, hist.classify()

    # ── Per-signal evaluation ─────────────────────────────────────────────────

    def _evaluate(self, sig: dict, sports_ok: bool) -> Optional[dict]:
        """
        Attempt to approve one signal. Returns enriched dict or None.
        Always emits an [AUDIT] log row.
        """
        ticker   = sig["ticker"]
        yes_price = sig["yes_price"]

        volatility, momentum, vol_label = self._get_market_volatility(ticker)

        # ── Match game ────────────────────────────────────────────────────
        game_state: Optional[GameState] = None
        match_id   = None

        if sports_ok:
            market_stub = {"ticker": ticker, "title": sig.get("title", ticker)}
            gs, match_reason = self._matcher.match(market_stub, self._store)

            if gs is not None:
                if gs.is_stale(SPORTS_STALE_SECONDS):
                    reject = f"matched game data stale ({gs.event_id})"
                    _log_audit(sig, None, None, volatility, momentum, vol_label, reject)
                    logger.info("[SCANNER] SKIP %s — %s", ticker, reject)
                    return None
                game_state = gs
                match_id   = gs.event_id
            else:
                if REQUIRE_GAME_MATCH:
                    reject = f"unmatched (require_match=true): {match_reason}"
                    _log_audit(sig, None, None, volatility, momentum, vol_label, reject)
                    logger.info("[SCANNER] SKIP %s — %s", ticker, reject)
                    return None
                # Continue to model without game state (will be blocked there)
        else:
            if REQUIRE_GAME_MATCH:
                reject = "sports feed stale and require_match=true"
                _log_audit(sig, None, None, volatility, momentum, vol_label, reject)
                logger.info("[SCANNER] SKIP %s — %s", ticker, reject)
                return None

        # ── Model evaluation ──────────────────────────────────────────────
        enriched = evaluate_signal(sig, game_state, volatility=volatility)
        if enriched is None:
            _log_audit(sig, game_state, None, volatility, momentum, vol_label,
                       "model blocked")
            return None

        enriched["matched_event_id"] = match_id
        enriched["momentum"]         = round(momentum, 4)
        enriched.setdefault("volatility", round(volatility, 4))

        # ── Confidence gate ───────────────────────────────────────────────
        model_conf = enriched["confidence_score"]
        if model_conf < MIN_MODEL_CONFIDENCE:
            reject = (f"confidence {model_conf:.0%} < min {MIN_MODEL_CONFIDENCE:.0%}")
            _log_audit(sig, game_state, enriched, volatility, momentum, vol_label, reject)
            logger.info("[SCANNER] SKIP %s — %s", ticker, reject)
            return None

        # ── Net edge gate ─────────────────────────────────────────────────
        net_edge = enriched.get("net_edge", 0.0)
        if abs(net_edge) < MIN_NET_EDGE:
            reject = (f"net_edge {net_edge:+.0%} < min {MIN_NET_EDGE:.0%}")
            _log_audit(sig, game_state, enriched, volatility, momentum, vol_label, reject)
            logger.info("[SCANNER] SKIP %s — %s", ticker, reject)
            return None

        # ── Volatile market gate ──────────────────────────────────────────
        if vol_label == "VOLATILE":
            reject = (f"market volatile: vol={volatility:.3f} > {MAX_VOLATILITY_TRADE:.3f}")
            _log_audit(sig, game_state, enriched, volatility, momentum, vol_label, reject)
            logger.info("[SCANNER] SKIP %s — %s", ticker, reject)
            return None

        enriched["market_data"] = self._market_lookup.get(ticker, {})
        _log_audit(sig, game_state, enriched, volatility, momentum, vol_label, None)
        return enriched


# ── Audit logging ─────────────────────────────────────────────────────────────

def _log_audit(
    sig:       dict,
    gs:        Optional[GameState],
    enriched:  Optional[dict],
    volatility: float,
    momentum:   float,
    vol_label:  str,
    rejected:   Optional[str],
):
    """
    One structured [AUDIT] line per evaluated market.
    Fields: ticker | sport | teams | score | clock |
            fair% | mkt% | edge | net_edge | conf% | vol | momentum | decision | reason
    """
    ticker    = sig.get("ticker", "?")
    side      = sig.get("side", "?")
    yes_price = sig.get("yes_price", 0)

    if gs:
        score_str = f"{gs.home_abbr} {gs.home_score}-{gs.away_score} {gs.away_abbr}"
        clock_str = f"P{gs.period}/{gs.total_periods} {gs.game_clock}"
        ev_id     = gs.event_id
        sport     = f"{gs.league}"
    else:
        score_str = "no_game"
        clock_str = "—"
        ev_id     = "—"
        sport     = "?"

    if enriched:
        fair_p   = enriched.get("fair_probability", 0)
        mkt_p    = enriched.get("market_probability", yes_price / 100)
        raw_edge = enriched.get("raw_edge", 0)
        net_edge = enriched.get("net_edge", 0)
        m_conf   = enriched.get("confidence_score", 0)
        reason   = enriched.get("reason", "")[:100]
        sport    = enriched.get("sport", sport)
    else:
        fair_p   = 0
        mkt_p    = yes_price / 100
        raw_edge = 0
        net_edge = 0
        m_conf   = 0
        reason   = rejected or ""

    decision = "SKIP" if rejected else "TRADE"

    logger.info(
        "[AUDIT] %-24s | %-4s | side=%-3s | %s | %s "
        "| fair=%.0f%% mkt=%.0f%% edge=%+.0f%% net=%+.0f%% "
        "| conf=%.0f%% vol=%.3f(%s) mom=%+.3f "
        "| %s: %s",
        ticker, sport, side,
        score_str, clock_str,
        fair_p * 100, mkt_p * 100, raw_edge * 100, net_edge * 100,
        m_conf * 100, volatility, vol_label, momentum,
        decision, reason[:80],
    )
