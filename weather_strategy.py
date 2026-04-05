"""
weather_strategy.py — Signal generation for Kalshi weather/temperature markets.

Pipeline per market:
  1. Parse contract (weather_parser)
  2. Spread hard-cap check (fail fast before any API call)
  3. Fetch forecast (weather_data)
  4. Apply confidence deductions (spread, stale data)
  5. Compute Gaussian model probability for YES and NO sides
  6. Choose the side with the greater net edge
  7. Apply uncertainty buffer + spread penalty → net_edge
  8. Directional sanity checks
  9. Emit signal dict or None

Returned signal dict:
    {
        "strategy":       "weather",
        "ticker":         str,
        "side":           "yes" | "no",
        "city":           str,
        "contract_type":  str,          # "above" | "below" | "range"
        "forecast_high":  float,
        "sigma_f":        float,
        "model_prob":     float,        # P(chosen side wins)
        "market_prob":    float,        # implied probability of chosen side
        "raw_edge":       float,        # model_prob - market_prob
        "net_edge":       float,        # raw_edge - uncertainty_buffer - spread_penalty
        "confidence":     float,        # adjusted downward for spread / stale data
        "spread_cents":   float,
        "reason":         str,
    }

NOTE: alerts and paper tracking only unless WEATHER_LIVE_ENABLED is True.
"""

import logging
import math
from datetime import datetime, timezone
from typing import Optional

import weather_config as cfg
from weather_data import get_weather_forecast
from weather_parser import parse_weather_market

logger = logging.getLogger("kalshi_bot.weather_strategy")


# ── Gaussian helpers ──────────────────────────────────────────────────────────

def _cdf(x: float, mu: float, sigma: float) -> float:
    """Cumulative distribution function for N(mu, sigma)."""
    return 0.5 * (1.0 + math.erf((x - mu) / (sigma * math.sqrt(2))))


def _sigma_for(city: str) -> float:
    """Return city-specific sigma or the global default."""
    return cfg.CITY_SIGMA_OVERRIDES.get(city, cfg.DEFAULT_SIGMA_F)


def _model_prob_yes(contract: dict, forecast_high: float, sigma: float) -> float:
    """
    P(YES contract resolves YES) under a Gaussian temperature distribution.

    Range : P(lower <= high <= upper)  — ±0.5 discrete correction
    Above : P(high >= threshold)
    Below : P(high < threshold)
    """
    ct = contract["type"]

    if ct == "range":
        lo = contract["lower"]
        hi = contract["upper"]
        p  = _cdf(hi + 0.5, forecast_high, sigma) - _cdf(lo - 0.5, forecast_high, sigma)

    elif ct == "above":
        p = 1.0 - _cdf(contract["threshold"] - 0.5, forecast_high, sigma)

    elif ct == "below":
        p = _cdf(contract["threshold"] - 0.5, forecast_high, sigma)

    else:
        return 0.5

    return max(0.01, min(0.99, p))


# ── Spread helpers ────────────────────────────────────────────────────────────

def _spread_cents(market: dict) -> Optional[float]:
    """Return yes_ask - yes_bid in cents, or None if data missing."""
    yes_bid = market.get("yes_bid")
    yes_ask = market.get("yes_ask")
    if yes_bid is None or yes_ask is None:
        return None
    return float(yes_ask) - float(yes_bid)


def _spread_penalty(spread: float) -> float:
    """
    Incremental net-edge penalty for spread above the free threshold.
    Spread of 2¢ or less → no penalty.
    Each cent above that costs SPREAD_PENALTY_PER_CENT.
    """
    excess = max(0.0, spread - cfg.SPREAD_FREE_THRESH_CENTS)
    return round(excess * cfg.SPREAD_PENALTY_PER_CENT, 4)


# ── Confidence adjustments ────────────────────────────────────────────────────

def _adjusted_confidence(base_conf: float, spread: float,
                          forecast_timestamp: Optional[datetime]) -> float:
    """
    Reduce confidence for illiquid or stale signals.

    Deductions:
      - wide spread (> 4¢):          -CONF_DEDUCT_WIDE_SPREAD
      - stale forecast (> 4 hours):  -CONF_DEDUCT_STALE_FORECAST
    """
    conf = base_conf

    if spread > 4:
        conf -= cfg.CONF_DEDUCT_WIDE_SPREAD

    if forecast_timestamp is not None:
        try:
            age_secs = (datetime.now(timezone.utc) - forecast_timestamp).total_seconds()
            if age_secs > cfg.FORECAST_STALE_SECS:
                conf -= cfg.CONF_DEDUCT_STALE_FORECAST
        except Exception:
            pass

    return round(max(0.10, min(0.99, conf)), 4)


# ── Reason string ─────────────────────────────────────────────────────────────

def _make_reason(contract: dict, side: str, forecast_high: float, sigma: float,
                 model_prob: float, market_prob: float, raw_edge: float,
                 net_edge: float, spread: float, confidence: float,
                 source: str) -> str:
    ct = contract["type"]
    if ct == "range":
        spec = f"{contract['lower']}–{contract['upper']}°F"
    elif ct == "above":
        spec = f"≥{contract['threshold']}°F"
    else:
        spec = f"<{contract['threshold']}°F"

    direction = "underpriced" if model_prob > market_prob else "overpriced"
    return (
        f"{contract['city']} forecast {forecast_high:.1f}°F (σ={sigma:.1f}°F) | "
        f"contract {spec} side={side.upper()} | "
        f"model={model_prob:.3f} market={market_prob:.3f} | "
        f"raw={raw_edge:.3f} net={net_edge:.3f} spread={spread:.0f}¢ | "
        f"{direction} | conf={confidence:.2f} src={source}"
    )


# ── Public API ────────────────────────────────────────────────────────────────

def evaluate_weather_market(market: dict) -> Optional[dict]:
    """
    Evaluate a single Kalshi market dict for a weather edge on either side.

    Chooses YES or NO, whichever has the higher net edge, and returns a signal
    dict if net_edge >= MIN_NET_EDGE and all quality gates pass.

    Returns None when the market is not a weather contract, data is missing,
    or no edge clears the threshold.  Never raises.
    """
    ticker = market.get("ticker", "?")
    try:
        # 1. Parse — skip non-weather markets immediately
        contract = parse_weather_market(market)
        if contract is None:
            return None

        city = contract["city"]

        # Directional sanity: require city and appropriate thresholds
        if not city:
            logger.debug("[WEATHER] %s — empty city, skipping", ticker)
            return None
        ct = contract.get("type", "")
        if ct == "range" and (contract.get("lower") is None or contract.get("upper") is None):
            logger.debug("[WEATHER] %s — range contract missing bounds, skipping", ticker)
            return None
        if ct in ("above", "below") and contract.get("threshold") is None:
            logger.debug("[WEATHER] %s — threshold missing, skipping", ticker)
            return None

        # 2. Spread hard-cap (fail fast before API call)
        spread = _spread_cents(market)
        if spread is None:
            logger.debug("[WEATHER] %s — spread data missing, skipping", ticker)
            return None
        if spread > cfg.MAX_SPREAD_CENTS:
            logger.debug("[WEATHER] %s — spread %.0f¢ > max %d¢", ticker, spread,
                         cfg.MAX_SPREAD_CENTS)
            return None

        # 3. Fetch forecast
        forecast = get_weather_forecast(city)
        if forecast is None:
            logger.debug("[WEATHER] %s — no forecast data", ticker)
            return None

        forecast_high = float(forecast["forecast_high"])
        base_conf     = float(forecast["confidence"])
        forecast_ts   = forecast.get("timestamp")
        source        = "openweathermap" if base_conf >= 0.75 else "seasonal_baseline"

        # 4. Adjusted confidence (spread, staleness)
        confidence = _adjusted_confidence(base_conf, spread, forecast_ts)
        if confidence < cfg.MIN_CONFIDENCE:
            logger.debug("[WEATHER] %s — adjusted conf %.2f < %.2f", ticker,
                         confidence, cfg.MIN_CONFIDENCE)
            return None

        # 5. Model probabilities for both sides
        sigma     = _sigma_for(city)
        mp_yes    = _model_prob_yes(contract, forecast_high, sigma)
        mp_no     = 1.0 - mp_yes

        yes_ask   = int(market.get("yes_ask") or 50)
        yes_bid   = int(market.get("yes_bid") or 50)

        # Market implied probability for each side
        mkt_yes   = round(yes_ask / 100.0, 4)          # cost to buy YES
        mkt_no    = round((100 - yes_bid) / 100.0, 4)  # cost to buy NO (no_ask = 100 - yes_bid)

        # Spread penalty shared by both sides (same spread)
        s_penalty = _spread_penalty(spread)

        # Raw and net edge for each side
        raw_yes = round(mp_yes - mkt_yes, 4)
        net_yes = round(raw_yes - cfg.UNCERTAINTY_BUFFER - s_penalty, 4)

        raw_no  = round(mp_no - mkt_no, 4)
        net_no  = round(raw_no - cfg.UNCERTAINTY_BUFFER - s_penalty, 4)

        logger.debug(
            "[WEATHER] %s city=%s type=%s forecast=%.1f°F σ=%.1f | "
            "YES model=%.3f mkt=%.3f raw=%.3f net=%.3f | "
            "NO  model=%.3f mkt=%.3f raw=%.3f net=%.3f | "
            "spread=%.0f¢ conf=%.2f",
            ticker, city, ct, forecast_high, sigma,
            mp_yes, mkt_yes, raw_yes, net_yes,
            mp_no,  mkt_no,  raw_no,  net_no,
            spread, confidence,
        )

        # 6. Pick the better side
        if net_yes >= net_no and net_yes >= cfg.MIN_NET_EDGE:
            side       = "yes"
            model_prob = round(mp_yes, 4)
            market_prob = mkt_yes
            raw_edge   = raw_yes
            net_edge   = net_yes
            yes_price  = yes_ask   # entry cost for YES

        elif net_no > net_yes and net_no >= cfg.MIN_NET_EDGE:
            side        = "no"
            model_prob  = round(mp_no, 4)
            market_prob = mkt_no
            raw_edge    = raw_no
            net_edge    = net_no
            yes_price   = yes_bid  # YES-price equivalent when entering NO side

        else:
            # Neither side clears the threshold
            return None

        reason = _make_reason(
            contract, side, forecast_high, sigma,
            model_prob, market_prob, raw_edge, net_edge,
            spread, confidence, source,
        )

        signal = {
            "strategy":      "weather",
            "ticker":        ticker,
            "side":          side,
            "city":          city,
            "contract_type": ct,
            "forecast_high": forecast_high,
            "sigma_f":       sigma,
            "model_prob":    model_prob,
            "market_prob":   market_prob,
            "raw_edge":      raw_edge,
            "net_edge":      net_edge,
            "confidence":    confidence,
            "spread_cents":  spread,
            "yes_price":     yes_price,   # order entry price in cents
            "reason":        reason,
        }

        logger.info(
            "[WEATHER SIGNAL] %s | city=%s | side=%s | forecast=%.1f°F | "
            "raw=%.3f net=%.3f model=%.3f market=%.3f spread=%.0f¢ conf=%.2f",
            ticker, city, side.upper(), forecast_high,
            raw_edge, net_edge, model_prob, market_prob, spread, confidence,
        )

        return signal

    except Exception as exc:
        logger.warning("[WEATHER] evaluate_weather_market error on %s: %s", ticker, exc)
        return None
