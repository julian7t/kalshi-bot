"""
weather_reports.py — Human-readable reporting for weather paper-trade performance.

Uses data already stored by weather_paper.py and computed by weather_backtest.py.
Read-only — no orders, no DB writes, no sports changes.

Run from the command line:
    cd kalshi-bot && python3 weather_reports.py
"""

import logging
from weather_backtest import run_weather_backtest, summarize_weather_backtest
from weather_paper import load_weather_signal_records, summarize_weather_signals

logger = logging.getLogger("kalshi_bot.weather_reports")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _divider(char: str = "─", width: int = 60) -> str:
    return char * width


def _load() -> tuple[list[dict], dict, dict]:
    """Load records once and return (records, paper_summary, backtest_summary)."""
    records  = load_weather_signal_records()
    p_sum    = summarize_weather_signals()
    bt_res   = run_weather_backtest(records)
    bt_sum   = summarize_weather_backtest(bt_res)
    return records, p_sum, bt_sum


# ── Report functions ──────────────────────────────────────────────────────────

def print_weather_signal_summary() -> None:
    """Print overall signal and backtest performance to stdout."""
    records, p_sum, bt_sum = _load()

    total    = p_sum["total_signals"]
    settled  = total - sum(
        1 for r in records if r.get("outcome_yes") is None
    )
    skipped  = total - settled

    print()
    print(_divider("═"))
    print("  WEATHER SIGNAL SUMMARY")
    print(_divider("═"))
    print(f"  Total signals recorded : {total}")
    print(f"  Settled signals        : {settled}")
    print(f"  Pending (not settled)  : {skipped}")
    print()
    print(f"  Average raw edge       : {p_sum['avg_raw_edge']:+.4f}")
    print(f"  Average net edge       : {p_sum['avg_net_edge']:+.4f}")
    print()

    if settled > 0:
        print(f"  Win rate               : {bt_sum['win_rate_pct']:.1f}%")
        print(f"  Total P&L              : ${bt_sum['total_pnl_dollars']:+.2f}")
        print(f"  P&L per settled signal : ${bt_sum['pnl_per_signal']:+.4f}")
    else:
        print("  Win rate               : n/a (no settled signals yet)")
        print("  Total P&L              : n/a")

    print(_divider())


def print_weather_backtest_summary() -> None:
    """Print full backtest performance breakdown to stdout."""
    records, _, bt_sum = _load()
    bt_res = run_weather_backtest(records)

    print()
    print(_divider("═"))
    print("  WEATHER BACKTEST SUMMARY")
    print(_divider("═"))
    print(f"  Total signals    : {bt_res.total_signals}")
    print(f"  Settled          : {bt_res.settled_signals}")
    print(f"  Wins             : {bt_res.wins}")
    print(f"  Losses           : {bt_res.losses}")
    print(f"  Skipped (none)   : {bt_res.skipped}")
    print()

    if bt_res.settled_signals > 0:
        print(f"  Win rate         : {bt_sum['win_rate_pct']:.1f}%")
        print(f"  Avg raw edge     : {bt_sum['avg_raw_edge']:+.4f}")
        print(f"  Avg net edge     : {bt_sum['avg_net_edge']:+.4f}")
        print(f"  Total P&L        : ${bt_sum['total_pnl_dollars']:+.2f}")
        print(f"  P&L per signal   : ${bt_sum['pnl_per_signal']:+.4f}")
    else:
        print("  No settled signals — add outcome_yes data to run backtest.")

    print(_divider())


def print_top_weather_cities() -> None:
    """Print per-city signal counts and P&L."""
    _, p_sum, bt_sum = _load()

    by_city_signals = p_sum.get("by_city", {})
    by_city_bt      = bt_sum.get("by_city", {})

    # Merge: all cities seen in either source
    all_cities = sorted(
        set(by_city_signals) | set(by_city_bt),
        key=lambda c: by_city_signals.get(c, 0),
        reverse=True,
    )

    print()
    print(_divider("═"))
    print("  WEATHER PERFORMANCE BY CITY")
    print(_divider("═"))
    print(f"  {'City':<18} {'Signals':>8} {'Settled':>8} {'Wins':>6} {'Losses':>7} {'Win%':>7} {'P&L':>9}")
    print(_divider())

    for city in all_cities:
        sigs    = by_city_signals.get(city, 0)
        bt_data = by_city_bt.get(city, {})
        settled = bt_data.get("settled", 0)
        wins    = bt_data.get("wins", 0)
        losses  = bt_data.get("losses", 0)
        pnl     = bt_data.get("pnl_dollars", 0.0)
        win_pct = f"{wins/settled*100:.1f}%" if settled > 0 else "n/a"
        print(f"  {city:<18} {sigs:>8} {settled:>8} {wins:>6} {losses:>7} {win_pct:>7} {pnl:>+9.2f}")

    if not all_cities:
        print("  No data yet.")

    print(_divider())


def print_top_weather_contract_types() -> None:
    """Print per-contract-type signal counts and P&L."""
    _, p_sum, bt_sum = _load()

    by_type_signals = p_sum.get("by_contract_type", {})
    by_type_bt      = bt_sum.get("by_contract_type", {})

    all_types = sorted(
        set(by_type_signals) | set(by_type_bt),
        key=lambda t: by_type_signals.get(t, 0),
        reverse=True,
    )

    print()
    print(_divider("═"))
    print("  WEATHER PERFORMANCE BY CONTRACT TYPE")
    print(_divider("═"))
    print(f"  {'Type':<10} {'Signals':>8} {'Settled':>8} {'Wins':>6} {'Losses':>7} {'Win%':>7} {'P&L':>9}")
    print(_divider())

    for ctype in all_types:
        sigs    = by_type_signals.get(ctype, 0)
        bt_data = by_type_bt.get(ctype, {})
        settled = bt_data.get("settled", 0)
        wins    = bt_data.get("wins", 0)
        losses  = bt_data.get("losses", 0)
        pnl     = bt_data.get("pnl_dollars", 0.0)
        win_pct = f"{wins/settled*100:.1f}%" if settled > 0 else "n/a"
        print(f"  {ctype:<10} {sigs:>8} {settled:>8} {wins:>6} {losses:>7} {win_pct:>7} {pnl:>+9.2f}")

    if not all_types:
        print("  No data yet.")

    print(_divider())


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)   # suppress debug noise in CLI

    print_weather_signal_summary()
    print_weather_backtest_summary()
    print_top_weather_cities()
    print_top_weather_contract_types()
