"""
weather_parser.py — Detect and parse Kalshi weather/temperature markets.

Returns a structured dict or None.

Output format:
    {
        "city":      str,
        "metric":    "temp_high",
        "type":      "range" | "above" | "below",
        "lower":     int | None,   # lower bound for range contracts
        "upper":     int | None,   # upper bound for range contracts
        "threshold": int | None,   # threshold for above/below contracts
    }
"""

import re
import logging
from typing import Optional

logger = logging.getLogger("kalshi_bot.weather_parser")

# ── Weather market detection ──────────────────────────────────────────────────

_WEATHER_TITLE_KEYWORDS = (
    "high temp", "temperature", "high between", "high at or above",
    "high below", "degrees", "fahrenheit",
)
_WEATHER_TICKER_PREFIXES = ("HIGH", "KXHIGH", "TEMP", "WEATHER")

# ── Contract title patterns ───────────────────────────────────────────────────

_RE_RANGE = re.compile(
    r"high\s+(?:temp(?:erature)?\s+)?between\s+(\d+)\s+and\s+(\d+)",
    re.IGNORECASE,
)
_RE_ABOVE = re.compile(
    r"high\s+(?:temp(?:erature)?\s+)?at\s+or\s+above\s+(\d+)",
    re.IGNORECASE,
)
_RE_BELOW = re.compile(
    r"high\s+(?:temp(?:erature)?\s+)?below\s+(\d+)",
    re.IGNORECASE,
)

# ── City extraction ───────────────────────────────────────────────────────────

_CITY_ALIASES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"NYC?",       re.IGNORECASE), "New York"),
    (re.compile(r"\bNY\b",     re.IGNORECASE), "New York"),
    (re.compile(r"New\s+York", re.IGNORECASE), "New York"),
    (re.compile(r"\bLA\b",     re.IGNORECASE), "Los Angeles"),
    (re.compile(r"Los\s+Angeles", re.IGNORECASE), "Los Angeles"),
    (re.compile(r"CHI(?:CAGO)?", re.IGNORECASE), "Chicago"),
    (re.compile(r"MIA(?:MI)?", re.IGNORECASE), "Miami"),
    (re.compile(r"DAL(?:LAS)?", re.IGNORECASE), "Dallas"),
    (re.compile(r"SEA(?:TTLE)?", re.IGNORECASE), "Seattle"),
    (re.compile(r"\bSF\b|San\s+Francisco", re.IGNORECASE), "San Francisco"),
    (re.compile(r"BOS(?:TON)?", re.IGNORECASE), "Boston"),
    (re.compile(r"DEN(?:VER)?", re.IGNORECASE), "Denver"),
    (re.compile(r"ATL(?:ANTA)?", re.IGNORECASE), "Atlanta"),
    (re.compile(r"PHX|PHOENIX", re.IGNORECASE), "Phoenix"),
    (re.compile(r"\bDC\b|WASHINGTON", re.IGNORECASE), "Washington DC"),
    (re.compile(r"LAS\s*VEGAS", re.IGNORECASE), "Las Vegas"),
    (re.compile(r"HOU(?:STON)?", re.IGNORECASE), "Houston"),
    (re.compile(r"PHI(?:LA(?:DELPHIA)?)?", re.IGNORECASE), "Philadelphia"),
]


def _extract_city(ticker: str, title: str) -> str:
    """
    Extract city name from title first (more readable), then ticker segments.
    Ticker is split on hyphens so short codes like NY/LA in HIGHNY are found
    by stripping the HIGH/KXHIGH prefix first.
    """
    # 1. Title (free text — most reliable)
    for pattern, city in _CITY_ALIASES:
        if pattern.search(title):
            return city

    # 2. Each hyphen-delimited segment of the ticker
    for segment in ticker.split("-"):
        for pattern, city in _CITY_ALIASES:
            if pattern.search(segment):
                return city

    # 3. Strip common prefixes then search the remainder (handles HIGHNY, HIGHLA)
    tail = re.sub(r"^(?:KXHIGH|HIGH|TEMP|WEATHER)", "", ticker, flags=re.IGNORECASE)
    for pattern, city in _CITY_ALIASES:
        if pattern.search(tail):
            return city

    return "Unknown"


def _is_weather_market(ticker: str, title: str) -> bool:
    ticker_up = ticker.upper()
    if any(ticker_up.startswith(p) for p in _WEATHER_TICKER_PREFIXES):
        return True
    title_lower = title.lower()
    return any(kw in title_lower for kw in _WEATHER_TITLE_KEYWORDS)


def parse_weather_market(market: dict) -> Optional[dict]:
    """
    Parse a raw Kalshi market dict.

    Returns a structured dict on success, None if the market is not a
    weather/temperature market or if the title format is unrecognised.
    """
    ticker = market.get("ticker", "")
    title  = market.get("title", "") or market.get("subtitle", "")

    if not _is_weather_market(ticker, title):
        return None

    city = _extract_city(ticker, title)

    # Range: "high between X and Y"
    m = _RE_RANGE.search(title)
    if m:
        lower, upper = int(m.group(1)), int(m.group(2))
        logger.debug("[WEATHER PARSER] %s → range %d–%d°F  city=%s", ticker, lower, upper, city)
        return {
            "city":      city,
            "metric":    "temp_high",
            "type":      "range",
            "lower":     lower,
            "upper":     upper,
            "threshold": None,
        }

    # Above: "high at or above X"
    m = _RE_ABOVE.search(title)
    if m:
        threshold = int(m.group(1))
        logger.debug("[WEATHER PARSER] %s → above %d°F  city=%s", ticker, threshold, city)
        return {
            "city":      city,
            "metric":    "temp_high",
            "type":      "above",
            "lower":     None,
            "upper":     None,
            "threshold": threshold,
        }

    # Below: "high below X"
    m = _RE_BELOW.search(title)
    if m:
        threshold = int(m.group(1))
        logger.debug("[WEATHER PARSER] %s → below %d°F  city=%s", ticker, threshold, city)
        return {
            "city":      city,
            "metric":    "temp_high",
            "type":      "below",
            "lower":     None,
            "upper":     None,
            "threshold": threshold,
        }

    logger.debug("[WEATHER PARSER] %s — weather market but unparseable title: %.80s",
                 ticker, title)
    return None
