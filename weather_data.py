"""
weather_data.py — Fetch temperature forecast data for a given city.

Primary source: OpenWeatherMap current conditions API (free tier).
  Set OPENWEATHER_API_KEY environment variable to enable.

Fallback: seasonal monthly baseline (no API key required, lower confidence).

Cache: results are reused for 30 minutes per city to avoid rate-limit issues.

Public interface:
    get_weather_forecast(city: str) -> dict
        {
            "forecast_high": float,    # predicted daily high °F
            "confidence":    float,    # 0.0–1.0
            "timestamp":     datetime,
        }
"""

import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional

import requests

logger = logging.getLogger("kalshi_bot.weather_data")

OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "")
_OWM_URL            = "https://api.openweathermap.org/data/2.5/weather"
_CACHE_TTL_SECS     = 1800   # 30 minutes

# city → (unix_timestamp, result_dict)
_cache: dict[str, tuple[float, dict]] = {}

# ── City coordinates ──────────────────────────────────────────────────────────

_CITY_COORDS: dict[str, tuple[float, float]] = {
    "New York":      (40.71, -74.01),
    "Los Angeles":   (34.05, -118.24),
    "Chicago":       (41.88, -87.63),
    "Miami":         (25.77, -80.19),
    "Dallas":        (32.78, -96.80),
    "Seattle":       (47.61, -122.33),
    "San Francisco": (37.77, -122.42),
    "Boston":        (42.36, -71.06),
    "Denver":        (39.74, -104.98),
    "Atlanta":       (33.75, -84.39),
    "Phoenix":       (33.45, -112.07),
    "Washington DC": (38.91, -77.04),
    "Las Vegas":     (36.17, -115.14),
    "Houston":       (29.76, -95.37),
    "Philadelphia":  (39.95, -75.16),
    "Unknown":       (40.71, -74.01),
}

# ── Seasonal baselines: average daily high °F, indexed by month (0=Jan) ──────

_SEASONAL: dict[str, list[float]] = {
    "New York":      [38, 41, 50, 61, 71, 80, 85, 83, 76, 64, 53, 42],
    "Los Angeles":   [68, 69, 71, 74, 77, 82, 85, 87, 85, 79, 73, 67],
    "Chicago":       [32, 36, 47, 59, 70, 80, 84, 82, 75, 63, 49, 36],
    "Miami":         [77, 79, 82, 85, 88, 90, 91, 92, 90, 86, 82, 78],
    "Dallas":        [56, 61, 69, 77, 84, 92, 96, 97, 89, 79, 67, 57],
    "Seattle":       [47, 51, 56, 60, 67, 73, 79, 80, 74, 63, 52, 46],
    "San Francisco": [57, 60, 62, 63, 64, 67, 67, 68, 70, 68, 62, 57],
    "Boston":        [36, 38, 46, 57, 67, 76, 82, 80, 73, 62, 51, 40],
    "Denver":        [45, 48, 55, 63, 72, 82, 88, 86, 78, 66, 53, 45],
    "Atlanta":       [52, 57, 65, 73, 80, 87, 89, 88, 83, 73, 63, 53],
    "Phoenix":       [66, 70, 78, 86, 95, 104, 106, 104, 99, 88, 74, 65],
    "Washington DC": [43, 46, 56, 67, 76, 85, 89, 87, 80, 69, 58, 46],
    "Las Vegas":     [58, 63, 71, 80, 90, 100, 105, 103, 96, 82, 67, 57],
    "Houston":       [62, 66, 73, 79, 86, 92, 94, 95, 90, 82, 72, 63],
    "Philadelphia":  [40, 43, 52, 63, 73, 82, 87, 85, 78, 67, 56, 44],
    "Unknown":       [60, 62, 65, 70, 75, 80, 82, 81, 78, 72, 65, 61],
}


def _seasonal_fallback(city: str) -> dict:
    month = datetime.now(timezone.utc).month - 1
    baseline = _SEASONAL.get(city, _SEASONAL["Unknown"])
    return {
        "forecast_high": baseline[month],
        "confidence":    0.50,
        "timestamp":     datetime.now(timezone.utc),
    }


def _fetch_owm(city: str) -> Optional[dict]:
    """Fetch current conditions from OpenWeatherMap. Returns None on any error."""
    coords = _CITY_COORDS.get(city, _CITY_COORDS["Unknown"])
    try:
        resp = requests.get(
            _OWM_URL,
            params={
                "lat":   coords[0],
                "lon":   coords[1],
                "appid": OPENWEATHER_API_KEY,
                "units": "imperial",
            },
            timeout=8,
        )
        resp.raise_for_status()
        data = resp.json()
        main = data.get("main", {})
        high = float(main.get("temp_max", main.get("temp", 70)))
        logger.debug("[WEATHER DATA] OWM %s → %.1f°F", city, high)
        return {
            "forecast_high": high,
            "confidence":    0.80,
            "timestamp":     datetime.now(timezone.utc),
        }
    except Exception as exc:
        logger.warning("[WEATHER DATA] OWM fetch failed for %s: %s", city, exc)
        return None


def get_weather_forecast(city: str) -> dict:
    """
    Return forecast dict for city, using cache when available.

    Never raises — falls back to seasonal baseline on any error.
    """
    now = time.time()

    cached = _cache.get(city)
    if cached and (now - cached[0]) < _CACHE_TTL_SECS:
        logger.debug("[WEATHER DATA] Cache hit for %s", city)
        return cached[1]

    result = None
    if OPENWEATHER_API_KEY:
        result = _fetch_owm(city)

    if result is None:
        result = _seasonal_fallback(city)

    _cache[city] = (now, result)
    return result
