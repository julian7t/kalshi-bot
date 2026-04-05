"""
weather_config.py — Central configuration for the weather strategy module.

All weather-specific tunables live here so they can be adjusted without
touching strategy logic. Import this module wherever weather constants
are needed — never hard-code these values in other files.
"""

# ── Probability model ─────────────────────────────────────────────────────────

# Default Gaussian std-dev (°F) used when no city-specific override exists.
DEFAULT_SIGMA_F: float = 2.5

# Per-city sigma overrides.  Cities with more variable weather get a wider
# distribution; stable climates (e.g. LA, Phoenix) get a tighter one.
# Leave empty to use DEFAULT_SIGMA_F for all cities.
CITY_SIGMA_OVERRIDES: dict[str, float] = {
    # "Chicago":     3.5,   # high day-to-day variability
    # "Los Angeles": 1.8,   # stable marine climate
    # "Phoenix":     2.0,   # dry desert heat, predictable
}

# ── Signal quality filters ────────────────────────────────────────────────────

# Minimum forecast confidence required to emit any signal (paper or live).
MIN_CONFIDENCE: float = 0.60

# Flat uncertainty buffer subtracted from raw edge.
# Compensates for model error and execution friction.
UNCERTAINTY_BUFFER: float = 0.03

# Spread penalty rate: each cent of spread above the free threshold reduces
# net_edge by this amount.  Penalises illiquid markets gradually.
SPREAD_FREE_THRESH_CENTS: int   = 2      # spreads ≤ this get no penalty
SPREAD_PENALTY_PER_CENT:  float = 0.004  # penalty per cent above free threshold

# Minimum *net* edge (after UNCERTAINTY_BUFFER + spread penalty) for a signal.
MIN_NET_EDGE: float = 0.10

# Hard cap — skip the market entirely if spread exceeds this.
MAX_SPREAD_CENTS: int = 8

# Confidence deduction for a wide spread (spread > 4¢).
CONF_DEDUCT_WIDE_SPREAD: float = 0.05

# Confidence deduction when forecast timestamp is older than this many seconds.
FORECAST_STALE_SECS:       int   = 14_400   # 4 hours
CONF_DEDUCT_STALE_FORECAST: float = 0.05

# ── Live trading gate ─────────────────────────────────────────────────────────

# Master switch.  Set True ONLY when you are ready for live weather execution.
# When False: signal alerts and paper recording happen, no orders placed.
WEATHER_LIVE_ENABLED: bool = True

# ── Live weather risk limits ──────────────────────────────────────────────────

# Maximum contracts per individual weather trade.
WEATHER_MAX_CONTRACTS_PER_TRADE: int = 1

# Maximum simultaneous open weather positions (resting + pending orders).
WEATHER_MAX_CONCURRENT_POSITIONS: int = 2

# Absolute max dollar exposure across all open weather positions combined.
WEATHER_MAX_DOLLAR_EXPOSURE: float = 25.0

# Daily weather loss ceiling.  If realised weather P&L today drops below
# this, no further weather trades are placed until the next calendar day.
WEATHER_DAILY_LOSS_LIMIT_DOLLARS: float = 10.0

# Minimum confidence required to place a LIVE weather order (higher bar
# than the paper signal threshold of MIN_CONFIDENCE).
WEATHER_MIN_CONF_FOR_LIVE: float = 0.70
