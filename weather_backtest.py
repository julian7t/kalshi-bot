"""
weather_backtest.py — Replay and evaluate historical weather signals.

Usage
-----
    from weather_backtest import run_weather_backtest, summarize_weather_backtest
    from weather_paper import load_weather_signal_records

    records = load_weather_signal_records()
    result  = run_weather_backtest(records)
    summary = summarize_weather_backtest(result)
    print(summary)

Each record dict must have at minimum:
    ticker, city, contract_type, model_prob, market_prob,
    raw_edge, net_edge, yes_ask, outcome_yes (None|1|0)

Records without outcome_yes are counted as 'skipped' (not yet settled).

P&L model (paper entry at yes_ask):
    win  → pnl_cents = 100 - yes_ask   (bought YES at yes_ask, settled at 100)
    loss → pnl_cents = -yes_ask        (bought YES at yes_ask, settled at 0)
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("kalshi_bot.weather_backtest")


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class CityStats:
    signals:    int   = 0
    settled:    int   = 0
    wins:       int   = 0
    losses:     int   = 0
    pnl_cents:  float = 0.0


@dataclass
class WeatherBacktestResult:
    """Aggregated results from run_weather_backtest()."""
    total_signals:   int
    settled_signals: int
    wins:            int
    losses:          int
    skipped:         int
    avg_raw_edge:    float
    avg_net_edge:    float
    total_pnl_cents: float
    by_city:         dict = field(default_factory=dict)   # city → CityStats
    by_contract_type:dict = field(default_factory=dict)   # type → {"signals":int,"wins":int,"pnl":float}


# ── Core backtest ─────────────────────────────────────────────────────────────

def run_weather_backtest(records: list[dict]) -> WeatherBacktestResult:
    """
    Replay a list of weather signal records and compute aggregate performance.

    Parameters
    ----------
    records : list[dict]
        Rows from weather_paper.load_weather_signal_records() or any list of
        dicts matching the schema described in the module docstring.

    Returns
    -------
    WeatherBacktestResult
    """
    total          = len(records)
    settled        = 0
    wins           = 0
    losses         = 0
    skipped        = 0
    total_pnl      = 0.0
    raw_edges:  list[float] = []
    net_edges:  list[float] = []

    by_city: dict[str, CityStats] = defaultdict(CityStats)
    by_type: dict[str, dict]      = defaultdict(
        lambda: {"signals": 0, "settled": 0, "wins": 0, "losses": 0, "pnl_cents": 0.0}
    )

    for rec in records:
        city          = rec.get("city", "Unknown")
        ctype         = rec.get("contract_type", "?")
        yes_ask       = int(rec.get("yes_ask") or 50)
        outcome_yes   = rec.get("outcome_yes")   # 1 | 0 | None

        if rec.get("raw_edge") is not None:
            raw_edges.append(float(rec["raw_edge"]))
        if rec.get("net_edge") is not None:
            net_edges.append(float(rec["net_edge"]))

        by_city[city].signals    += 1
        by_type[ctype]["signals"] += 1

        if outcome_yes is None:
            skipped += 1
            continue

        settled += 1
        by_city[city].settled    += 1
        by_type[ctype]["settled"] += 1

        if outcome_yes == 1 or outcome_yes is True:
            pnl = float(100 - yes_ask)
            wins += 1
            by_city[city].wins    += 1
            by_type[ctype]["wins"] += 1
        else:
            pnl = float(-yes_ask)
            losses += 1
            by_city[city].losses    += 1
            by_type[ctype]["losses"] += 1

        total_pnl                   += pnl
        by_city[city].pnl_cents     += pnl
        by_type[ctype]["pnl_cents"]  += pnl

    avg_raw = round(sum(raw_edges) / len(raw_edges), 4) if raw_edges else 0.0
    avg_net = round(sum(net_edges) / len(net_edges), 4) if net_edges else 0.0

    return WeatherBacktestResult(
        total_signals    = total,
        settled_signals  = settled,
        wins             = wins,
        losses           = losses,
        skipped          = skipped,
        avg_raw_edge     = avg_raw,
        avg_net_edge     = avg_net,
        total_pnl_cents  = round(total_pnl, 2),
        by_city          = dict(by_city),
        by_contract_type = dict(by_type),
    )


# ── Summary ───────────────────────────────────────────────────────────────────

def summarize_weather_backtest(result: WeatherBacktestResult) -> dict:
    """
    Convert a WeatherBacktestResult into a human-readable summary dict.

    Returns
    -------
    dict with:
        win_rate_pct       float   (0–100)
        avg_raw_edge       float
        avg_net_edge       float
        total_pnl_dollars  float
        pnl_per_signal     float   (based on settled signals)
        by_city            dict    city → {signals, settled, wins, losses, pnl_dollars}
        by_contract_type   dict    type → {signals, settled, wins, losses, pnl_dollars}
    """
    settled = result.settled_signals or 1   # avoid div-by-zero
    win_rate = round(result.wins / settled * 100, 1)

    by_city = {}
    for city, stats in result.by_city.items():
        by_city[city] = {
            "signals":    stats.signals,
            "settled":    stats.settled,
            "wins":       stats.wins,
            "losses":     stats.losses,
            "pnl_dollars": round(stats.pnl_cents / 100, 2),
        }

    by_type = {}
    for ctype, stats in result.by_contract_type.items():
        by_type[ctype] = {
            "signals":    stats["signals"],
            "settled":    stats["settled"],
            "wins":       stats["wins"],
            "losses":     stats["losses"],
            "pnl_dollars": round(stats["pnl_cents"] / 100, 2),
        }

    return {
        "win_rate_pct":      win_rate,
        "avg_raw_edge":      result.avg_raw_edge,
        "avg_net_edge":      result.avg_net_edge,
        "total_pnl_dollars": round(result.total_pnl_cents / 100, 2),
        "pnl_per_signal":    round(result.total_pnl_cents / settled / 100, 4),
        "by_city":           by_city,
        "by_contract_type":  by_type,
    }


# ── Convenience loader ────────────────────────────────────────────────────────

def load_weather_signal_records() -> list[dict]:
    """
    Load signal records from the database via weather_paper.
    Safe wrapper — returns [] if weather_paper is unavailable.
    """
    try:
        from weather_paper import load_weather_signal_records as _load
        return _load()
    except Exception as exc:
        logger.error("[WEATHER BACKTEST] Could not load records: %s", exc)
        return []
