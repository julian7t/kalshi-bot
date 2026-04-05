"""
order_manager.py — Idempotent order placement with full fill lifecycle handling.

client_order_id format (preserved from v1):
  {YYYYMMDD}_{ticker_short}_{side}_{strategy}_{seq:04d}
  e.g. 20260404_KXMVESPORTS_no_momentum_v1_0003

Fill lifecycle:
  pending  → resting    — order acknowledged by Kalshi
  resting  → executed   — fully filled (happy path)
  resting  → partially filled, then canceled  — partial cancel (see below)
  resting  → canceled   — no fill
  pending  → failed     — API error on placement

Partial fill handling:
  1. Every poll we compare filled_count with the previous value.
  2. If the order is canceled (or expired) with filled_count > 0:
       a. Wait EXCHANGE_DELAY_WAIT seconds (Kalshi sometimes updates fill
          count after sending the cancel confirmation — "exchange delay").
       b. Re-fetch the order to get the definitive fill count.
       c. Record ONLY the confirmed filled portion as a real position.
       d. Fire an alert so you know a partial happened.
  3. Positions are NEVER updated from assumptions — only from avg_yes_price
     and filled_count returned by Kalshi's order endpoint.
"""

import logging
import time
from datetime import datetime, timezone

import alerting
import analytics
import db
import kalshi_client as kc
import metrics
from config import (
    EXCHANGE_DELAY_WAIT,
    FILL_POLL_INTERVAL,
    FILL_POLL_TIMEOUT,
    STALE_ORDER_SECONDS,
    STRATEGY_NAME,
)

# Edge-monitor: check if edge is still valid every N polls during fill wait
_EDGE_CHECK_EVERY_N_POLLS = 5

logger = logging.getLogger("kalshi_bot.orders")


# ── client_order_id ───────────────────────────────────────────────────────────

def _make_client_order_id(ticker: str, side: str) -> str:
    date    = datetime.now(timezone.utc).strftime("%Y%m%d")
    t_short = ticker[:20].replace("-", "")
    seq     = db.next_sequence_for_ticker_today(ticker)
    return f"{date}_{t_short}_{side}_{STRATEGY_NAME}_{seq:04d}"


# ── Duplicate detection ───────────────────────────────────────────────────────

def _already_traded(ticker: str, live_open_orders: list, live_positions: dict) -> tuple[bool, str]:
    """Three-source duplicate check: local DB, Kalshi open orders, Kalshi positions."""
    if db.has_active_order_for_ticker(ticker):
        return True, "active order in local DB"
    if db.has_executed_order_for_ticker(ticker):
        return True, "executed order in local DB"
    for o in live_open_orders:
        if o.get("ticker") == ticker:
            return True, f"open order on Kalshi (id={o.get('order_id','')})"
    if ticker in live_positions and (live_positions[ticker].get("position", 0) != 0):
        return True, "existing position on Kalshi"
    return False, ""


# ── Order placement ───────────────────────────────────────────────────────────

def place_order_safe(
    ticker: str,
    side: str,
    count: int,
    yes_price: int,
    live_open_orders: list,
    live_positions: dict,
    exec_decision=None,   # Optional[ExecutionDecision] from execution.py
    engine=None,          # Optional[ExecutionEngine] for mid-poll edge checks
    fair_probability: float = 0.5,
) -> str | None:
    """
    Idempotent order placement.

    Writes to DB as 'pending' before touching the API — so a crash between
    placement and response confirmation won't cause a re-entry on restart.

    If exec_decision is provided, its entry_price overrides yes_price.
    If engine is provided, the fill poller will cancel orders when edge disappears.

    Returns kalshi_order_id on success, None on skip or failure.
    """
    dup, reason = _already_traded(ticker, live_open_orders, live_positions)
    if dup:
        logger.debug("[SKIP] [%s] already traded — %s", ticker, reason)
        return None

    # Use execution-layer optimised price if available
    actual_price = exec_decision.entry_price if exec_decision else yes_price
    exec_mode    = exec_decision.mode if exec_decision else "unset"
    exec_regime  = exec_decision.regime if exec_decision else "unknown"

    client_id = _make_client_order_id(ticker, side)

    # --- Write 'pending' to DB before API call (crash safety) ---
    db.save_order(
        client_order_id=client_id,
        ticker=ticker, side=side, count=count,
        yes_price=actual_price, strategy=STRATEGY_NAME,
        status="pending",
    )
    logger.info(
        "[ORDER] Placing %s %s %d@%d¢ (mode=%s regime=%s)  client_id=%s",
        side.upper(), ticker, count, actual_price,
        exec_mode, exec_regime, client_id,
    )

    try:
        resp      = kc.place_order(ticker, side, count, actual_price, client_order_id=client_id)
        order     = resp.get("order", resp)
        kalshi_id = order.get("order_id", "")

        db.update_order(client_id, kalshi_order_id=kalshi_id, status="resting")
        logger.info("[ORDER PLACED] kalshi_id=%s  client_id=%s  ticker=%s",
                    kalshi_id, client_id, ticker)

        # Register in in-memory dedup cache so subsequent signals this scan skip it
        live_open_orders.append({"ticker": ticker, "order_id": kalshi_id})
        alerting.alert_order_placed(ticker, side, count, actual_price, client_id)

        # Register cooldown immediately after placing
        if engine is not None:
            engine.after_order_placed(ticker)

        _confirm_fill(
            kalshi_id, client_id, ticker, side, count, actual_price,
            fair_probability=fair_probability, engine=engine,
        )
        return kalshi_id

    except Exception as e:
        logger.error("[ORDER FAILED] [%s] %s", ticker, e)
        db.update_order(client_id, status="failed")
        alerting.send_alert("ORDER_FAILED",
                            f"Order placement failed for {ticker}: {e}",
                            level="WARNING")
        return None


# ── Fill confirmation with partial-fill handling ──────────────────────────────

def _confirm_fill(
    kalshi_id: str,
    client_id: str,
    ticker: str,
    side: str,
    count: int,
    yes_price: int,
    fair_probability: float = 0.5,
    engine=None,
):
    """
    Poll the Kalshi order until:
      - Fully executed  → record fill, update position
      - Canceled/expired with partial fill  → handle exchange delay, record partial
      - Canceled with no fill  → mark canceled
      - Edge disappears while resting → cancel the order and trigger cooldown
      - Timeout  → leave resting, stale cleanup will cancel it later

    Positions are updated ONLY from confirmed fill data (avg_yes_price, filled_count).
    Edge monitoring: every _EDGE_CHECK_EVERY_N_POLLS polls we re-fetch the market
    and ask the execution engine whether the original edge is still alive.
    """
    deadline        = time.time() + FILL_POLL_TIMEOUT
    attempt         = 0
    last_fill_count = 0
    placed_at       = time.time()
    edge_canceled   = False

    while time.time() < deadline:
        attempt += 1
        try:
            data   = kc.get_order(kalshi_id)
            order  = data.get("order", data)
            status = order.get("status", "")
            filled = int(order.get("filled_count", 0) or 0)
            avg_px = order.get("avg_yes_price")

            # Log any new partial fill progress
            if filled > last_fill_count:
                logger.info("[PARTIAL FILL] [%s] %d/%d contract(s) filled  status=%s",
                            ticker, filled, count, status)
                last_fill_count = filled

            # ── Full execution ─────────────────────────────────────────────
            if status == "executed":
                _record_fill(client_id, ticker, side, count, filled, avg_px, yes_price,
                             placed_at, partial=False)
                return

            # ── Canceled / expired ─────────────────────────────────────────
            if status in ("canceled", "expired"):
                if filled > 0:
                    logger.warning(
                        "[PARTIAL CANCEL] [%s] %d/%d filled before %s — "
                        "waiting %ds for exchange delay...",
                        ticker, filled, count, status, EXCHANGE_DELAY_WAIT
                    )
                    time.sleep(EXCHANGE_DELAY_WAIT)
                    try:
                        data2      = kc.get_order(kalshi_id)
                        order2     = data2.get("order", data2)
                        final_fill = int(order2.get("filled_count", filled) or filled)
                        final_px   = order2.get("avg_yes_price", avg_px)
                        if final_fill != filled:
                            logger.info(
                                "[EXCHANGE DELAY] [%s] Fill count updated %d → %d after re-fetch",
                                ticker, filled, final_fill
                            )
                        filled = final_fill
                        avg_px = final_px
                    except Exception as e:
                        logger.warning("[EXCHANGE DELAY RE-FETCH FAILED] [%s] %s — using %d",
                                       ticker, e, filled)
                    _record_fill(client_id, ticker, side, count, filled, avg_px, yes_price,
                                 placed_at, partial=True)
                    alerting.alert_partial_cancel(ticker, side, filled, count, kalshi_id)
                else:
                    db.update_order(client_id, status="canceled", filled_count=0)
                    logger.info("[ORDER CANCELED] [%s] status=%s  kalshi_id=%s",
                                ticker, status, kalshi_id[:8])
                    if edge_canceled and engine is not None:
                        engine.after_order_canceled(ticker)
                return

            # ── Mid-poll edge monitor ──────────────────────────────────────
            if engine is not None and attempt % _EDGE_CHECK_EVERY_N_POLLS == 0:
                try:
                    mkt_data = kc.get_market(ticker)
                    market   = mkt_data.get("market", mkt_data)
                    if not engine.edge_still_valid(ticker, side, fair_probability, market):
                        logger.warning(
                            "[EDGE GONE] [%s] Edge disappeared while resting "
                            "— canceling order %s", ticker, kalshi_id[:8]
                        )
                        kc.cancel_order(kalshi_id)
                        alerting.send_alert(
                            "EDGE_DISAPPEARED",
                            f"Order {kalshi_id[:8]} canceled for {ticker}: edge gone while resting",
                            level="INFO",
                        )
                        edge_canceled = True
                        # Loop continues; next poll will catch the canceled status
                except Exception as e:
                    logger.debug("[EDGE CHECK FAILED] [%s] %s", ticker, e)

            logger.debug("[POLL] [%s] status=%s attempt=%d filled=%d/%d",
                         ticker, status, attempt, filled, count)

        except Exception as e:
            logger.warning("[POLL ERROR] [%s] %s (attempt %d)", ticker, e, attempt)

        time.sleep(FILL_POLL_INTERVAL)

    logger.warning(
        "[FILL TIMEOUT] [%s] order %s still resting after %ds — stale cleanup will handle it",
        ticker, kalshi_id[:8], FILL_POLL_TIMEOUT
    )


def _record_fill(
    client_id: str,
    ticker: str,
    side: str,
    requested: int,
    filled: int,
    avg_px,
    yes_price: int,
    placed_at: float,
    partial: bool,
):
    """
    Commit a confirmed fill to the DB, update the position, and record metrics.
    ONLY called with confirmed fill data from Kalshi — no assumptions.
    """
    if filled == 0:
        db.update_order(client_id, status="canceled", filled_count=0)
        return

    # Entry cost from our perspective
    fill_price = float(avg_px) if avg_px is not None else float(yes_price)
    if side == "no":
        entry_cents = 100.0 - fill_price
    else:
        entry_cents = fill_price

    slippage = fill_price - yes_price   # positive = paid more than quoted

    db.update_order(
        client_id,
        status="executed" if not partial else "partial",
        filled_count=filled,
        avg_fill_price=fill_price,
    )
    db.upsert_position(ticker, side, filled, entry_cents)

    hold_so_far = time.time() - placed_at

    logger.info(
        "[FILL RECORDED] [%s] %s %d/%d contract(s) @ %.1f¢ "
        "(asked %d¢, slippage %+.1f¢)  partial=%s",
        ticker, side.upper(), filled, requested, fill_price,
        yes_price, slippage, partial
    )

    alerting.alert_order_filled(ticker, side, filled, fill_price, slippage)
    metrics.record_fill(
        ticker=ticker, side=side, strategy=STRATEGY_NAME,
        yes_price=yes_price, filled_price=fill_price,
        contracts=filled, partial=partial,
    )
    # Record into rich analytics table (context was registered by bot.py)
    analytics.record_fill_event(ticker, fill_price, filled, partial=partial)


# ── Stale order cleanup ───────────────────────────────────────────────────────

def cleanup_stale_orders():
    """
    Cancel resting orders older than STALE_ORDER_SECONDS.
    After canceling, re-fetches the order to capture any partial fills
    that occurred before the cancellation reached the exchange.
    """
    resting  = db.get_orders_by_status("resting")
    now      = time.time()
    canceled = 0

    for order in resting:
        placed_at_str = order.get("placed_at", "")
        if not placed_at_str:
            continue
        try:
            placed_ts = datetime.fromisoformat(placed_at_str).timestamp()
        except Exception:
            continue

        age = now - placed_ts
        if age < STALE_ORDER_SECONDS:
            continue

        kalshi_id = order.get("kalshi_order_id")
        ticker    = order.get("ticker", "?")
        client_id = order["client_order_id"]
        side      = order.get("side", "?")
        yes_price = order.get("yes_price", 0)
        count     = order.get("count", 0)

        if not kalshi_id:
            db.update_order(client_id, status="canceled")
            continue

        try:
            kc.cancel_order(kalshi_id)
            logger.info("[STALE CANCEL] [%s] order %s resting for %.0fs — canceling",
                        ticker, kalshi_id[:8], age)

            # Re-fetch after cancel to catch any partial fill from exchange delay
            time.sleep(EXCHANGE_DELAY_WAIT)
            try:
                data      = kc.get_order(kalshi_id)
                o         = data.get("order", data)
                filled    = int(o.get("filled_count", 0) or 0)
                avg_px    = o.get("avg_yes_price")
                if filled > 0:
                    logger.info(
                        "[STALE PARTIAL] [%s] %d contract(s) filled before stale cancel",
                        ticker, filled
                    )
                    _record_fill(client_id, ticker, side, count, filled, avg_px,
                                 yes_price, placed_ts, partial=True)
                    alerting.alert_partial_cancel(ticker, side, filled, count, kalshi_id)
                else:
                    db.update_order(client_id, status="canceled", filled_count=0)
            except Exception as e:
                logger.warning("[STALE CANCEL RE-FETCH FAILED] [%s] %s", ticker, e)
                db.update_order(client_id, status="canceled")

            canceled += 1
        except Exception as e:
            logger.warning("[STALE CANCEL FAILED] [%s] %s", ticker, e)

    if canceled:
        logger.info("[STALE CLEANUP] Canceled %d stale order(s)", canceled)
