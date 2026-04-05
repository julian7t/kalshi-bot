"""
config.py — All bot settings, loaded from environment variables with safe defaults.
Every tunable parameter lives here. Never hardcode values in other modules.
"""
import os

_e = os.environ.get

# ── Mode ─────────────────────────────────────────────────────────────────────
PAPER_MODE  = _e("LIVE_MODE", "false").lower() != "true"

# ── Signal / strategy ─────────────────────────────────────────────────────────
MIN_CONFIDENCE          = float(_e("MIN_CONFIDENCE",          "0.65"))  # 65%
GAME_PROGRESS_THRESHOLD = float(_e("GAME_PROGRESS_THRESHOLD", "0.30"))  # 30% through window
BET_FRACTION            = float(_e("BET_FRACTION",            "0.03"))  # 3% of portfolio

# ── Position sizing ───────────────────────────────────────────────────────────
MIN_CONTRACTS = int(_e("MIN_CONTRACTS", "1"))
MAX_CONTRACTS = int(_e("MAX_CONTRACTS", "20"))

# ── Scan ──────────────────────────────────────────────────────────────────────
MARKETS_TO_SCAN    = int(_e("MARKETS_TO_SCAN",    "40"))
SCAN_INTERVAL      = int(_e("SCAN_INTERVAL",       "15"))   # seconds
MAX_OPEN_POSITIONS = int(_e("MAX_OPEN_POSITIONS",  "4"))

# ── Risk management ───────────────────────────────────────────────────────────
MIN_BALANCE_CENTS       = int(_e("MIN_BALANCE_CENTS",       "500"))   # $5.00
MAX_EXPOSURE_PER_MARKET = int(_e("MAX_EXPOSURE_PER_MARKET", "1000"))  # $10.00
MAX_TOTAL_EXPOSURE      = int(_e("MAX_TOTAL_EXPOSURE",      "3500"))  # $35.00
MAX_POSITION_SIZE       = int(_e("MAX_POSITION_SIZE",       "10"))    # contracts
PNL_KILL_SWITCH_CENTS   = int(_e("PNL_KILL_SWITCH_CENTS",   "-2000")) # -$20.00

# ── Order execution ───────────────────────────────────────────────────────────
STALE_ORDER_SECONDS       = int(_e("STALE_ORDER_SECONDS",   "45"))
FILL_POLL_INTERVAL        = int(_e("FILL_POLL_INTERVAL",    "2"))
FILL_POLL_TIMEOUT         = int(_e("FILL_POLL_TIMEOUT",     "45"))
EXCHANGE_DELAY_WAIT       = int(_e("EXCHANGE_DELAY_WAIT",   "3"))   # seconds to re-fetch after cancel

# ── Rate limiting ─────────────────────────────────────────────────────────────
API_RATE_PER_SECOND = float(_e("API_RATE_PER_SECOND", "5.0"))
API_RATE_BURST      = float(_e("API_RATE_BURST",      "10.0"))
API_MAX_RETRIES     = int(_e("API_MAX_RETRIES",       "4"))

# ── Data freshness ────────────────────────────────────────────────────────────
# If the last successful market fetch is older than this, skip trading and alert.
DATA_FRESHNESS_TIMEOUT      = int(_e("DATA_FRESHNESS_TIMEOUT",      "45"))   # seconds
# Number of consecutive failed scans before sending an API failure alert.
API_FAILURE_ALERT_THRESHOLD = int(_e("API_FAILURE_ALERT_THRESHOLD", "3"))

# ── Execution layer ───────────────────────────────────────────────────────────
MAX_SPREAD_CENTS           = int(_e("MAX_SPREAD_CENTS",           "10"))   # skip if spread > 10¢
MIN_DEPTH_CONTRACTS        = int(_e("MIN_DEPTH_CONTRACTS",         "1"))   # min quote depth
SLIPPAGE_BUFFER_CENTS      = float(_e("SLIPPAGE_BUFFER_CENTS",    "1.5"))  # extra slippage allowance
AGGRESSIVE_EDGE_THRESHOLD  = float(_e("AGGRESSIVE_EDGE_THRESHOLD","0.12")) # 12%+ → aggressive mode
PASSIVE_IMPROVEMENT_CENTS  = int(_e("PASSIVE_IMPROVEMENT_CENTS",  "2"))    # price improve for passive
COOLDOWN_AFTER_ENTRY_SECS  = int(_e("COOLDOWN_AFTER_ENTRY_SECS", "120"))  # 2 min after placement
COOLDOWN_AFTER_CANCEL_SECS = int(_e("COOLDOWN_AFTER_CANCEL_SECS","60"))   # 1 min after cancel
COOLDOWN_AFTER_LOSS_SECS   = int(_e("COOLDOWN_AFTER_LOSS_SECS",  "300"))  # 5 min after loss
EXIT_MODE                  = _e("EXIT_MODE", "settle")   # settle | take_profit | model_flip
TAKE_PROFIT_CENTS          = int(_e("TAKE_PROFIT_CENTS",          "12"))   # ¢ convergence to exit
MAX_HOLD_SECONDS           = int(_e("MAX_HOLD_SECONDS",           "0"))    # 0 = no time limit
MIN_EDGE_AFTER_SLIP        = float(_e("MIN_EDGE_AFTER_SLIP",      "0.03")) # 3% min edge after slip
EDGE_DISAPPEAR_THRESHOLD   = float(_e("EDGE_DISAPPEAR_THRESHOLD", "0.04")) # cancel if edge gone > 4%

# ── Alerting ──────────────────────────────────────────────────────────────────
# Set these in Replit Secrets (not env vars), they are read by alerting.py directly.
# DISCORD_WEBHOOK_URL, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
ALERT_COOLDOWN_SECONDS = int(_e("ALERT_COOLDOWN_SECONDS", "300"))  # 5 min between same alert type

# ── Paths ─────────────────────────────────────────────────────────────────────
_base = os.path.dirname(__file__)
DB_PATH           = os.path.join(_base, "trading.db")
PAPER_LEDGER_PATH = os.path.join(_base, "paper_trades.json")
LOG_PATH          = os.path.join(_base, "logs", "bot.log")
METRICS_DIR       = os.path.join(_base, "metrics")
STRATEGY_NAME     = "momentum_v1"
HEALTH_FILE_PATH  = os.path.join(_base, "metrics", "health.json")

# ── Health monitoring ──────────────────────────────────────────────────────────
# Separate consecutive-failure thresholds per data feed.
KALSHI_FAILURE_THRESHOLD  = int(_e("KALSHI_FAILURE_THRESHOLD",  "3"))
SPORTS_FAILURE_THRESHOLD  = int(_e("SPORTS_FAILURE_THRESHOLD",  "5"))
# Consecutive order-placement failures before safe mode activates.
MAX_ORDER_FAILURE_STREAK  = int(_e("MAX_ORDER_FAILURE_STREAK",  "3"))
# Seconds of stale data before trading is halted (vs just alerting at DATA_FRESHNESS_TIMEOUT).
# Must be >= DATA_FRESHNESS_TIMEOUT.  Default: 60s.
DATA_STALE_HALT_SECS      = int(_e("DATA_STALE_HALT_SECS",      "60"))
# If True, safe mode activates automatically on critical failures.
# Set False to disable auto-safe-mode (not recommended in production).
SAFE_MODE_AUTO            = _e("SAFE_MODE_AUTO", "true").lower() == "true"

# ── Watchdog ──────────────────────────────────────────────────────────────────
# Seconds without a heartbeat before the watchdog fires a CRITICAL alert.
WATCHDOG_STALL_SECS       = int(_e("WATCHDOG_STALL_SECS",       "120"))
# How often the watchdog checks (seconds).  Should be << WATCHDOG_STALL_SECS.
WATCHDOG_CHECK_INTERVAL   = int(_e("WATCHDOG_CHECK_INTERVAL",   "30"))
# Consecutive idle cycles (no candidates found) before the watchdog warns.
# 200 cycles × 15s ≈ 50 min of silence.  Set 0 to disable.
WATCHDOG_IDLE_CYCLES      = int(_e("WATCHDOG_IDLE_CYCLES",      "200"))

# ── Report generation throttle ────────────────────────────────────────────────
# How often (in scan iterations) to run the heavy CSV/JSON report writers.
# Default: 20 iterations ≈ every 5 minutes at a 15s scan cadence.
# Set to 1 to write every cycle (not recommended for production).
# Light metrics (analytics.log_summary) always run every cycle regardless.
REPORT_INTERVAL_ITERS = int(_e("REPORT_INTERVAL_ITERS", "20"))
# How often (in REPORT_INTERVAL_ITERS multiples) to include backtest scenarios.
# Default: every 50 report runs ≈ every ~4 hours.
# Set to 0 to disable scheduled backtest reports entirely.
BACKTEST_INTERVAL_REPORTS = int(_e("BACKTEST_INTERVAL_REPORTS", "50"))

# ── Parameter optimization ────────────────────────────────────────────────────
# How often (in scan iterations) to run the optimizer in the background.
# Set to 0 to disable automatic optimization runs.
OPT_INTERVAL_ITERS    = int(_e("OPT_INTERVAL_ITERS", "100"))
# Minimum closed trades required before the optimizer will run.
OPT_MIN_TRADES        = int(_e("OPT_MIN_TRADES", "20"))
# Search mode: "grid" (curated 4-value grid) or "random" (random sampling).
OPT_SEARCH_MODE       = _e("OPT_SEARCH_MODE", "grid")
# Random search samples (only used when OPT_SEARCH_MODE=random).
OPT_RANDOM_SAMPLES    = int(_e("OPT_RANDOM_SAMPLES", "150"))
# Minimum closed trades PER SPORT before a sport-specific config is recommended.
# Sports below this threshold fall back to the global recommended params.
OPT_MIN_TRADES_PER_SPORT = int(_e("OPT_MIN_TRADES_PER_SPORT", "10"))
# Whether to run sport-specific optimization alongside the global run.
OPT_RUN_SPORT_SPECIFIC   = _e("OPT_RUN_SPORT_SPECIFIC", "true").lower() == "true"
# Minimum closed trades per (sport, market_type) bucket for dedicated optimization.
# Buckets below this threshold fall back to sport params, then global params.
OPT_MIN_TRADES_PER_MARKET_TYPE = int(_e("OPT_MIN_TRADES_PER_MARKET_TYPE", "5"))
# Whether to run per-(sport, market_type) optimization after sport-specific run.
OPT_RUN_MARKET_TYPE      = _e("OPT_RUN_MARKET_TYPE", "true").lower() == "true"

# ── Signal model thresholds ───────────────────────────────────────────────────
# Minimum confidence required before trading a FUTURE segment market
# (e.g. Q4 winner market while the game is still in Q3).
# These markets rely on a full-game team-strength proxy, which is inherently
# less reliable than a current-segment model.  Default 0.50.
# Set env var MIN_FUTURE_SEGMENT_CONFIDENCE to override.
MIN_FUTURE_SEGMENT_CONFIDENCE = float(_e("MIN_FUTURE_SEGMENT_CONFIDENCE", "0.50"))

# ── Portfolio construction ────────────────────────────────────────────────────
# Max total capital committed to a single game/event (all positions combined).
PORTFOLIO_MAX_EXPOSURE_PER_EVENT  = int(_e("PORTFOLIO_MAX_EXPOSURE_PER_EVENT",  "2000"))  # $20.00
# Max total capital committed to a single sport at any time.
PORTFOLIO_MAX_EXPOSURE_PER_SPORT  = int(_e("PORTFOLIO_MAX_EXPOSURE_PER_SPORT",  "2000"))  # $20.00
# Max total capital committed to a single market_type (e.g. all totals bets).
PORTFOLIO_MAX_EXPOSURE_PER_MTYPE  = int(_e("PORTFOLIO_MAX_EXPOSURE_PER_MTYPE",  "1500"))  # $15.00
# Max number of simultaneous open positions within one event/game.
PORTFOLIO_MAX_POSITIONS_PER_EVENT = int(_e("PORTFOLIO_MAX_POSITIONS_PER_EVENT",    "3"))
# If True, HIGH-overlap candidates (same idea already in book) are rejected outright.
# Set False to allow but just reduce size.
PORTFOLIO_REJECT_HIGH_OVERLAP     = _e("PORTFOLIO_REJECT_HIGH_OVERLAP", "true").lower() == "true"


# ── Entry timing layer ────────────────────────────────────────────────────────
# All thresholds are informational defaults; never auto-applied to live config.
# Market move classification
TIMING_LAG_THRESHOLD       = float(_e("TIMING_LAG_THRESHOLD",       "0.04"))  # 4% gap = lagging
TIMING_OVERREACT_THRESHOLD = float(_e("TIMING_OVERREACT_THRESHOLD", "0.04"))  # 4% gap = overreacting
TIMING_NOISE_VOL_THRESHOLD = float(_e("TIMING_NOISE_VOL_THRESHOLD", "0.10"))  # 10% vol = noisy
# Chase protection
TIMING_CHASE_MOMENTUM      = float(_e("TIMING_CHASE_MOMENTUM",      "0.05"))  # 5¢/cycle
TIMING_CHASE_CONSUMED      = float(_e("TIMING_CHASE_CONSUMED",      "0.70"))  # 70% of edge consumed
# Urgency thresholds → entry mode
TIMING_URGENCY_NOW         = float(_e("TIMING_URGENCY_NOW",         "0.68"))  # ≥ → enter now
TIMING_URGENCY_STAGE       = float(_e("TIMING_URGENCY_STAGE",       "0.38"))  # ≥ → staged entry
TIMING_URGENCY_WAIT        = float(_e("TIMING_URGENCY_WAIT",        "0.18"))  # ≥ → wait/passive
# Staged entry: fraction of target contracts to place immediately
TIMING_STAGE_FRACTION      = float(_e("TIMING_STAGE_FRACTION",      "0.50"))  # 50%
# Time sensitivity window
TIMING_TIME_MAX_SECS       = int(_e("TIMING_TIME_MAX_SECS",         "600"))   # 10 min
# Re-entry / add logic
TIMING_ALLOW_ADD           = _e("TIMING_ALLOW_ADD", "false").lower() == "true"
TIMING_ADD_MIN_PRICE_IMPROVE = float(_e("TIMING_ADD_MIN_PRICE_IMPROVE", "3.0"))  # cents
TIMING_ADD_MIN_EDGE_IMPROVE  = float(_e("TIMING_ADD_MIN_EDGE_IMPROVE",  "0.02")) # 2%
TIMING_ADD_COOLDOWN_SECS     = int(_e("TIMING_ADD_COOLDOWN_SECS",       "180"))  # 3 min
# Exit timing
TIMING_TRIM_OVERSHOOT      = float(_e("TIMING_TRIM_OVERSHOOT",      "0.05"))  # 5% past fair → trim
TIMING_CONV_HOLD_MIN       = float(_e("TIMING_CONV_HOLD_MIN",       "0.60"))  # 60% convergence → hold


def load_optimized_params_for_sport(sport: str) -> dict:
    """
    Read best_parameters_by_sport.json and return the recommended param dict
    for the given sport.

    Falls back to load_optimized_params() (global) if:
      - The file doesn't exist
      - The sport is not found in the file
      - The sport's status is "insufficient_data"

    IMPORTANT: This function only READS — it does NOT apply anything to live
    config values.  The caller is responsible for reviewing before use.

    Returns {} if neither sport nor global params are available.
    """
    import json
    path = os.path.join(METRICS_DIR, "best_parameters_by_sport.json")
    if not os.path.exists(path):
        return load_optimized_params()
    try:
        with open(path) as f:
            data = json.load(f)
        sports = data.get("sports", {})
        entry  = sports.get(sport)
        if entry is None:
            # Sport not in file — use global
            return load_optimized_params()
        if entry.get("status") == "ok":
            params = entry.get("raw_params", {})
            import logging
            logging.getLogger("kalshi_bot.config").info(
                "[CONFIG] Sport-specific params for %s (score=%s): %s",
                sport, entry.get("combined_score", "n/a"), params,
            )
            return params
        else:
            # Insufficient data — fall back to global
            import logging
            logging.getLogger("kalshi_bot.config").info(
                "[CONFIG] %s: %s — using global params.",
                sport, entry.get("note", "insufficient data"),
            )
            return data.get("global_fallback_params") or load_optimized_params()
    except Exception as e:
        import logging
        logging.getLogger("kalshi_bot.config").warning(
            "[CONFIG] Could not load sport params for %s: %s", sport, e
        )
        return load_optimized_params()


def load_optimized_params_for_market_type(sport: str, market_type: str) -> dict:
    """
    Priority chain for parameter lookup:
      1. sport + market_type specific  (best_parameters_by_market_type.json)
      2. sport specific                (best_parameters_by_sport.json)
      3. global                        (best_parameters.json)

    IMPORTANT: This function only READS — it does NOT apply anything to live
    config values.  The caller is responsible for reviewing before use.

    Returns {} if no file exists at any level of the chain.
    """
    import json
    _log = __import__("logging").getLogger("kalshi_bot.config")

    # Level 1: sport + market_type
    mt_path = os.path.join(METRICS_DIR, "best_parameters_by_market_type.json")
    if os.path.exists(mt_path):
        try:
            with open(mt_path) as f:
                mt_data = json.load(f)
            key   = f"{sport}:{market_type}"
            entry = mt_data.get("buckets", {}).get(key)
            if entry and entry.get("status") == "ok":
                params = entry.get("raw_params", {})
                _log.info(
                    "[CONFIG] Market-type params for %s (score=%s): %s",
                    key, entry.get("combined_score", "n/a"), params,
                )
                return params
            elif entry:
                _log.info(
                    "[CONFIG] %s: %s — trying sport fallback.",
                    key, entry.get("fallback_reason", "insufficient data"),
                )
        except Exception as e:
            _log.warning("[CONFIG] Could not read market_type params: %s", e)

    # Level 2: sport
    return load_optimized_params_for_sport(sport)


def load_optimized_params() -> dict:
    """
    Read best_parameters.json and return the recommended parameter dict.

    IMPORTANT: This function only READS the file — it does NOT apply anything
    to live config values.  The caller is responsible for reviewing and deciding
    whether to use these values.

    Returns {} if the file does not exist or cannot be parsed.
    """
    import json
    path = os.path.join(METRICS_DIR, "best_parameters.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
        params = data.get("parameters", {})
        score  = data.get("combined_score", "n/a")
        gen_at = data.get("generated_at", "unknown")
        # Log but never auto-apply
        import logging
        logging.getLogger("kalshi_bot.config").info(
            "[CONFIG] Optimized params available (score=%s, generated=%s): %s",
            score, gen_at, params,
        )
        return params
    except Exception as e:
        import logging
        logging.getLogger("kalshi_bot.config").warning(
            "[CONFIG] Could not load optimized params: %s", e
        )
        return {}
