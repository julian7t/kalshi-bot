"""
LiveGuard — prevents duplicate live orders and confirms fills.

On startup in live mode, this syncs all existing Kalshi open orders
and positions so a restart never places a second order for the same market.

After placing an order it polls until the order fills, expires, or times out.
"""

import time
from kalshi_client import get_open_orders, get_order, get_live_positions


FILL_POLL_INTERVAL = 3   # seconds between fill checks
FILL_POLL_TIMEOUT  = 90  # seconds before giving up on a fill


class LiveGuard:
    def __init__(self):
        self._tickers: set[str] = set()
        self.sync_from_kalshi()

    def sync_from_kalshi(self):
        """Pull all open orders + live positions from Kalshi and index by ticker."""
        synced = set()

        # Open (unfilled) orders
        try:
            orders = get_open_orders()
            for o in orders:
                t = o.get("ticker")
                if t:
                    synced.add(t)
            print(f"  [LiveGuard] Synced {len(orders)} open order(s) from Kalshi.")
        except Exception as e:
            print(f"  [LiveGuard] Warning: could not fetch open orders: {e}")

        # Settled positions still held
        try:
            positions = get_live_positions()
            for t, pos in positions.items():
                if pos.get("position", 0) != 0:
                    synced.add(t)
            print(f"  [LiveGuard] Synced {len(positions)} live position(s) from Kalshi.")
        except Exception as e:
            print(f"  [LiveGuard] Warning: could not fetch positions: {e}")

        self._tickers = synced

    def has_position(self, ticker: str) -> bool:
        """True if we already have an open order or position for this ticker."""
        return ticker in self._tickers

    def record(self, ticker: str):
        """Mark a ticker as traded so we don't duplicate it this session."""
        self._tickers.add(ticker)

    def confirm_fill(self, order_id: str, ticker: str) -> bool:
        """
        Poll the order until it fills (status='executed') or becomes unfillable.
        Returns True if filled, False otherwise.
        """
        deadline = time.time() + FILL_POLL_TIMEOUT
        attempt = 0
        while time.time() < deadline:
            attempt += 1
            try:
                data = get_order(order_id)
                order = data.get("order", data)
                status = order.get("status", "")

                if status == "executed":
                    filled = order.get("filled_count", "?")
                    avg_price = order.get("avg_yes_price", "?")
                    print(f"  [FILL CONFIRMED] [{ticker}] order {order_id[:8]} filled {filled} contract(s) @ {avg_price}¢")
                    return True

                if status in ("canceled", "expired"):
                    print(f"  [ORDER {status.upper()}] [{ticker}] order {order_id[:8]} did not fill.")
                    self._tickers.discard(ticker)
                    return False

                # Still resting — wait and retry
                print(f"  [LiveGuard] Order {order_id[:8]} status={status!r}, waiting... (attempt {attempt})")
            except Exception as e:
                print(f"  [LiveGuard] Error polling order {order_id[:8]}: {e}")

            time.sleep(FILL_POLL_INTERVAL)

        print(f"  [LiveGuard] Fill timeout for {order_id[:8]} — order left resting, will check next scan.")
        return False
