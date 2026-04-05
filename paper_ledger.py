"""
Paper trading ledger — persists trades to a local JSON file.
Tracks open positions, closed positions, and running P&L.
"""

import json
import os
from datetime import datetime, timezone
from config import PAPER_LEDGER_PATH


def _load() -> dict:
    if os.path.exists(PAPER_LEDGER_PATH):
        with open(PAPER_LEDGER_PATH, "r") as f:
            return json.load(f)
    return {"open": {}, "closed": [], "stats": {"total_paper_pl": 0, "wins": 0, "losses": 0}}


def _save(data: dict):
    with open(PAPER_LEDGER_PATH, "w") as f:
        json.dump(data, f, indent=2)


def is_open(ticker: str) -> bool:
    data = _load()
    return ticker in data["open"]


def open_position(ticker: str, side: str, yes_price: int, contracts: int, title: str):
    """
    Record opening a paper position.
    yes_price is the YES-side price in the Kalshi order (1–99).
    For YES trades: we pay yes_price cents per contract.
    For NO trades: we pay (100 - yes_price) cents per contract.
    """
    data = _load()
    if ticker in data["open"]:
        return
    entry_price = yes_price if side == "yes" else (100 - yes_price)
    data["open"][ticker] = {
        "ticker": ticker,
        "title": title,
        "side": side,
        "yes_price": yes_price,
        "entry_price_cents": entry_price,
        "contracts": contracts,
        "cost_cents": entry_price * contracts,
        "opened_at": datetime.now(timezone.utc).isoformat(),
    }
    _save(data)
    cost = entry_price * contracts / 100
    print(f"  [PAPER] Opened {side.upper()} on [{ticker}]")
    print(f"          Entry: {entry_price}¢ x{contracts} contract(s) = ${cost:.2f} notional")


def settle_position(ticker: str, result: str):
    """
    Settle a paper position given the market result ('yes' or 'no').
    Winning side receives 100¢ per contract.
    """
    data = _load()
    if ticker not in data["open"]:
        return

    pos = data["open"].pop(ticker)
    side = pos["side"]
    contracts = pos["contracts"]
    cost_cents = pos["cost_cents"]

    won = side == result
    payout_cents = 100 * contracts if won else 0
    pl_cents = payout_cents - cost_cents

    pos["result"] = result
    pos["won"] = won
    pos["pl_cents"] = pl_cents
    pos["settled_at"] = datetime.now(timezone.utc).isoformat()

    data["closed"].append(pos)
    data["stats"]["total_paper_pl"] += pl_cents
    if won:
        data["stats"]["wins"] += 1
    else:
        data["stats"]["losses"] += 1

    _save(data)

    pl_str = f"+${pl_cents/100:.2f}" if pl_cents >= 0 else f"-${abs(pl_cents)/100:.2f}"
    outcome = "WON" if won else "LOST"
    print(f"  [PAPER SETTLED] [{ticker}] {outcome}  P&L: {pl_str}")


def print_summary():
    data = _load()
    stats = data["stats"]
    open_count = len(data["open"])
    closed_count = len(data["closed"])
    total = stats["wins"] + stats["losses"]
    win_rate = (stats["wins"] / total * 100) if total > 0 else 0.0
    pl_cents = stats["total_paper_pl"]
    pl_str = f"+${pl_cents/100:.2f}" if pl_cents >= 0 else f"-${abs(pl_cents)/100:.2f}"

    print(f"\n{'='*52}")
    print(f"  PAPER TRADING SUMMARY")
    print(f"{'='*52}")
    print(f"  Open positions:   {open_count}")
    print(f"  Settled trades:   {closed_count}  ({stats['wins']}W / {stats['losses']}L)")
    print(f"  Win rate:         {win_rate:.1f}%")
    print(f"  Total P&L:        {pl_str}")
    print(f"{'='*52}\n")

    if data["open"]:
        print("  Open paper positions:")
        for ticker, pos in data["open"].items():
            ep = pos["entry_price_cents"]
            print(f"    [{ticker[:40]}] {pos['side'].upper()} @ {ep}¢ x{pos['contracts']} — {pos['opened_at'][:10]}")
        print()


def get_open_tickers() -> list:
    data = _load()
    return list(data["open"].keys())
