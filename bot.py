"""
bot.py — Production-safe Kalshi trading bot (v4).

Architecture:
  kalshi_client  — authenticated API calls, rate limiting, retries
  db             — SQLite persistent state (orders, positions, P&L)
  order_manager  — idempotent placement, partial-fill handling, stale cleanup
  risk_manager   — hard pre-trade risk rules + kill switch
  strategy       — base signal detection (confidence + timing gates)
  scanner        — sports data integration + fair-probability model + audit log
  alerting       — Discord / Telegram structured alerts
  metrics        — CSV + JSON performance dashboard
  paper_ledger   — paper trading log (paper mode only)
  sports_data    — ESPN live scores adapter (mock fallback)
  game_state     — normalised per-event state store
  event_matcher  — fuzzy market-to-game matching
  model          — deterministic fair-probability blending

Startup (live mode):
  1. Init DB + run migrations
  2. Create Scanner (initialises sports adapter + game state store)
  3. Send BOT_STARTED alert
  4. Reconcile — fetch all Kalshi orders/positions, sync into DB

Every scan iteration:
  1. Per-loop reconciliation (compare DB resting orders vs Kalshi live)
  2. Freshness guard — skip trading if market data is stale
  3. Cancel stale orders
  4. Fetch + scan markets via Scanner (sports data + model evaluation)
  5. For each approved signal: risk check → idempotency check → place → confirm fill
  6. Update metrics + summary

Kill switch: if P&L < PNL_KILL_SWITCH_CENTS, alert and stop placing orders.
"""

import logging
import os
import sys
import threading
import time
from datetime import datetime, timezone

import alerting
import analytics
import classifier
import config
import db
import kalshi_client as kc
import metrics
import order_manager as om
import optimizer as opt_module
import paper_ledger
import reports
import risk_manager as rm
import strategy
from execution import ExecutionEngine
from health import HealthMonitor
from portfolio import PortfolioAnalyzer
from scanner import Scanner
from timing import TimingClassifier
from watchdog import Watchdog
import weather_config as wcfg
from weather_paper import record_weather_signal
from weather_risk import approve_weather_trade, compute_weather_position_size
from weather_strategy import evaluate_weather_market


# ── Logging ───────────────────────────────────────────────────────────────────

def _setup_logging():
    os.makedirs(os.path.dirname(config.LOG_PATH), exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(levelname)-8s] %(name)s — %(message)s")

    fh = logging.FileHandler(config.LOG_PATH, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    root = logging.getLogger("kalshi_bot")
    root.setLevel(logging.DEBUG)
    root.addHandler(fh)
    root.addHandler(ch)
    return root


logger = _setup_logging()

BANNER = """
╔══════════════════════════════════════════════════╗
║         KALSHI TRADING BOT  — PRODUCTION v4      ║
╚══════════════════════════════════════════════════╝"""


# ── Startup reconciliation ────────────────────────────────────────────────────

def startup_reconcile() -> tuple[list, dict, bool]:
    """
    Fetch all live Kalshi state and sync into local DB.
    Returns (live_open_orders, live_positions, reconcile_clean).
    reconcile_clean=False means at least one step had an error — safe mode
    will be activated until the operator confirms the state is correct.
    """
    logger.info("=== STARTUP RECONCILIATION ===")
    kalshi_open     = []
    positions       = {}
    reconcile_clean = True

    # Open (resting) orders
    try:
        kalshi_open = kc.get_open_orders()
        for o in kalshi_open:
            kalshi_id = o.get("order_id", "")
            ticker    = o.get("ticker", "")
            known     = {r["kalshi_order_id"] for r in db.get_orders_by_ticker(ticker)
                         if r.get("kalshi_order_id")}
            if kalshi_id not in known:
                db.save_order(
                    client_order_id=f"reconcile_{kalshi_id[:16]}",
                    ticker=ticker, side=o.get("side",""), count=o.get("count",0),
                    yes_price=o.get("yes_price",0), strategy="reconciled",
                    status="resting", kalshi_order_id=kalshi_id,
                )
                logger.info("  Synced open order %s [%s]", kalshi_id[:8], ticker)
        logger.info("  Open orders: %d", len(kalshi_open))
    except Exception as e:
        reconcile_clean = False
        logger.error("  Open orders reconciliation failed: %s", e)
        alerting.send_alert("AUTH_ERROR" if "auth" in str(e).lower() else "API_FAILURE",
                            f"Startup open-order reconciliation failed: {e}", level="CRITICAL")

    # Filled orders (recent)
    try:
        filled = kc.get_filled_orders(limit=50)
        for o in filled:
            kalshi_id = o.get("order_id", "")
            ticker    = o.get("ticker", "")
            known     = {r["kalshi_order_id"] for r in db.get_orders_by_ticker(ticker)
                         if r.get("kalshi_order_id")}
            if kalshi_id not in known:
                db.save_order(
                    client_order_id=f"reconcile_{kalshi_id[:16]}",
                    ticker=ticker, side=o.get("side",""),
                    count=o.get("filled_count",0),
                    yes_price=int(o.get("avg_yes_price") or 0),
                    strategy="reconciled", status="executed",
                    kalshi_order_id=kalshi_id,
                )
                logger.info("  Synced filled order %s [%s]", kalshi_id[:8], ticker)
        logger.info("  Filled orders: %d", len(filled))
    except Exception as e:
        reconcile_clean = False
        logger.error("  Filled orders reconciliation failed: %s", e)

    # Live positions
    try:
        positions = kc.get_live_positions()
        for ticker, pos in positions.items():
            contracts = abs(pos.get("position", 0))
            if contracts == 0:
                continue
            side = "yes" if pos.get("position", 0) > 0 else "no"
            db.upsert_position(ticker, side, contracts, avg_entry_cents=0.0)
            logger.info("  Synced position [%s] %s %d contract(s)", ticker, side, contracts)
        logger.info("  Positions: %d", len(positions))
    except Exception as e:
        reconcile_clean = False
        logger.error("  Position reconciliation failed: %s", e)

    db.log_reconciliation(
        orders_synced=len(kalshi_open),
        positions_synced=len(positions),
        notes="startup",
    )
    status = "CLEAN" if reconcile_clean else "HAD ERRORS — safe mode will activate"
    logger.info("=== RECONCILIATION COMPLETE (%s) ===\n", status)
    return kalshi_open, positions, reconcile_clean


# ── Per-loop reconciliation ───────────────────────────────────────────────────

def loop_reconcile(live_open_orders: list, live_positions: dict):
    """
    Run at the start of every scan to keep DB in sync with Kalshi's truth.

    Detects "ghost" resting orders — orders our DB thinks are resting but
    Kalshi has already settled or canceled (e.g., filled while bot was restarting).

    Updates live_open_orders and live_positions IN PLACE so the caller's
    dedup caches stay current.
    """
    mismatches     = []
    kalshi_open_ids = {o.get("order_id") for o in live_open_orders}

    # Check DB resting orders against Kalshi
    for order in db.get_orders_by_status("resting"):
        kalshi_id = order.get("kalshi_order_id")
        if not kalshi_id or kalshi_id in kalshi_open_ids:
            continue  # Known to Kalshi — no mismatch

        # DB says resting but Kalshi's open-order list doesn't show it
        ticker    = order.get("ticker", "?")
        client_id = order["client_order_id"]
        try:
            data   = kc.get_order(kalshi_id)
            remote = data.get("order", data)
            status = remote.get("status", "")
            filled = int(remote.get("filled_count", 0) or 0)
            avg_px = remote.get("avg_yes_price")

            if status == "executed":
                side  = order.get("side", "?")
                yp    = order.get("yes_price", 0)
                entry = float(avg_px or yp) if side == "yes" else (100.0 - float(avg_px or yp))
                db.update_order(client_id, status="executed",
                                filled_count=filled, avg_fill_price=avg_px)
                if filled > 0:
                    db.upsert_position(ticker, side, filled, entry)
                mismatches.append(f"{ticker}: DB=resting Kalshi=executed filled={filled}")
                logger.info("[LOOP RECONCILE] %s executed outside our poll window, filled=%d",
                            kalshi_id[:8], filled)

            elif status in ("canceled", "expired"):
                db.update_order(client_id, status="canceled", filled_count=filled)
                if filled > 0:
                    side  = order.get("side", "?")
                    yp    = order.get("yes_price", 0)
                    entry = float(avg_px or yp) if side == "yes" else (100.0 - float(avg_px or yp))
                    db.upsert_position(ticker, side, filled, entry)
                    mismatches.append(f"{ticker}: DB=resting Kalshi={status} partial_filled={filled}")
                    logger.warning("[LOOP RECONCILE] %s canceled with partial fill=%d",
                                   kalshi_id[:8], filled)
                else:
                    mismatches.append(f"{ticker}: DB=resting Kalshi={status}")
                    logger.info("[LOOP RECONCILE] %s was %s (0 filled)", kalshi_id[:8], status)

        except Exception as e:
            logger.warning("[LOOP RECONCILE] Could not fetch order %s: %s", kalshi_id[:8], e)

    if mismatches:
        alerting.alert_reconciliation_mismatch(len(mismatches), "; ".join(mismatches))

    # Refresh live_open_orders in place
    try:
        fresh = kc.get_open_orders()
        live_open_orders.clear()
        live_open_orders.extend(fresh)
    except Exception as e:
        logger.warning("[LOOP RECONCILE] Could not refresh open orders: %s", e)

    # Refresh live_positions in place
    try:
        fresh_pos = kc.get_live_positions()
        live_positions.clear()
        live_positions.update(fresh_pos)
    except Exception as e:
        logger.warning("[LOOP RECONCILE] Could not refresh positions: %s", e)


# ── Freshness guard ───────────────────────────────────────────────────────────

class _ScanState:
    """Mutable state shared across scan iterations (avoids globals)."""
    def __init__(
        self,
        scanner:             Scanner,
        exec_engine:         ExecutionEngine,
        portfolio_analyzer:  PortfolioAnalyzer,
        timing_classifier:   TimingClassifier,
        health_monitor:      HealthMonitor,
    ):
        self.kill_switch_fired:     bool  = False
        self.scanner:               Scanner           = scanner
        self.exec_engine:           ExecutionEngine   = exec_engine
        self.portfolio_analyzer:    PortfolioAnalyzer = portfolio_analyzer
        self.timing_classifier:     TimingClassifier  = timing_classifier
        self.health:                HealthMonitor     = health_monitor
        # Scan-level counters for timing skips
        self.timing_chase_skips:    int   = 0
        self.timing_total_skips:    int   = 0


# ── State integrity check ─────────────────────────────────────────────────────

def _check_state_integrity(
    live_open_orders: list,
    live_positions:   dict,
    health:           HealthMonitor,
) -> None:
    """
    Lightweight consistency check run at the start of each live-mode scan.

    Detects:
      - DB positions with 0 contracts (stale rows)
      - DB resting orders for tickers that already have a live position
        but whose DB status wasn't updated (orphan orders)
      - Duplicate resting DB orders for the same ticker

    Reports diffs to the health monitor (which alerts on repeated failures).
    This is complementary to loop_reconcile() — it does not fix anything,
    just surfaces anomalies.
    """
    issues: list[str] = []

    # 1. DB positions with 0 contracts
    for pos in db.get_all_positions():
        if int(pos.get("contracts") or 0) == 0:
            issues.append(f"zero-contract DB position: {pos['ticker']}")

    # 2. Orphan resting orders — same ticker has both a DB resting order
    #    AND a live Kalshi position (means the order actually filled but
    #    DB wasn't updated).
    live_tickers = set(live_positions.keys())
    resting_tickers: dict[str, int] = {}
    for order in db.get_orders_by_status("resting"):
        t = order.get("ticker", "?")
        resting_tickers[t] = resting_tickers.get(t, 0) + 1
        if t in live_tickers:
            issues.append(f"orphan resting order + live position: {t}")

    # 3. Duplicate resting orders for the same ticker
    for t, count in resting_tickers.items():
        if count > 1:
            issues.append(f"duplicate resting DB orders ({count}×): {t}")

    if issues:
        health.record_integrity_fail("; ".join(issues))
    else:
        health.record_integrity_ok()


# ── Paper mode settlement ─────────────────────────────────────────────────────

def _check_settled_paper():
    for ticker in paper_ledger.get_open_tickers():
        result = kc.get_settled_result(ticker)
        if result:
            paper_ledger.settle_position(ticker, result)


# ── Live mode settlement ───────────────────────────────────────────────────────

def _check_settled_live():
    """
    Check all open DB positions for settlement.  For each settled market:
      - compute realised PnL
      - record in analytics, metrics, pnl_history
      - remove the position from DB
    Does NOT touch auth, risk controls, or order placement.
    """
    positions = db.get_all_positions()
    for pos in positions:
        ticker     = pos["ticker"]
        side       = pos["side"]
        contracts  = int(pos.get("contracts") or 0)
        entry_c    = float(pos.get("avg_entry_cents") or 0)

        if contracts == 0:
            db.remove_position(ticker)
            continue

        try:
            outcome = kc.get_settled_result(ticker)
        except Exception as e:
            logger.debug("[SETTLE] Could not fetch result for %s: %s", ticker, e)
            continue

        if outcome not in ("yes", "no"):
            continue   # not yet settled

        # Compute PnL
        won            = (side == outcome)
        pnl_per        = (100.0 - entry_c) if won else (-entry_c)
        pnl_cents      = round(pnl_per * contracts, 2)
        exit_price     = 100.0 if won else 0.0
        outcome_label  = "win" if won else "loss"

        logger.info(
            "[SETTLE] %s → %s | side=%s entry=%.1f¢ contracts=%d | pnl=$%.2f (%s)",
            ticker, outcome, side, entry_c, contracts, pnl_cents/100, outcome_label,
        )

        # Analytics
        analytics.record_settlement_event(
            ticker=ticker, side=side,
            exit_price=exit_price, pnl_cents=pnl_cents,
            outcome=outcome_label, exit_reason="settlement",
        )

        # Legacy metrics (hold_time from positions table opened_at)
        try:
            opened_at  = pos.get("opened_at", "")
            from datetime import datetime, timezone
            opened_dt  = datetime.fromisoformat(opened_at)
            hold_secs  = (datetime.now(timezone.utc) - opened_dt).total_seconds()
        except Exception:
            hold_secs  = 0.0

        metrics.record_settlement(
            ticker=ticker, side=side, strategy=config.STRATEGY_NAME,
            yes_price=int(entry_c), contracts=contracts,
            pnl_cents=pnl_cents, hold_time_seconds=hold_secs,
            outcome=outcome_label,
        )

        # Persist PnL event and remove position
        db.record_pnl(ticker, side, contracts, entry_c, exit_price, pnl_cents)
        db.remove_position(ticker)

        alerting.send_alert(
            "POSITION_SETTLED",
            f"{ticker} settled {outcome.upper()} → {outcome_label.upper()} "
            f"pnl=${pnl_cents/100:.2f} ({contracts} contracts @ {entry_c:.1f}¢)",
            level="INFO",
        )


# ── Main scan ─────────────────────────────────────────────────────────────────

def run_scan(live_open_orders: list, live_positions: dict, state: _ScanState):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    logger.info("── Scan at %s ──", now)

    state.health.record_cycle_start()

    # ── Per-loop reconciliation + integrity check (live mode) ─────────────────
    if not config.PAPER_MODE:
        try:
            loop_reconcile(live_open_orders, live_positions)
        except Exception as e:
            logger.error("Loop reconciliation error: %s", e)
        try:
            _check_state_integrity(live_open_orders, live_positions, state.health)
        except Exception as e:
            logger.warning("State integrity check error: %s", e)

    # ── Balance ───────────────────────────────────────────────────────────────
    balance_cents   = 0.0
    portfolio_cents = 0.0
    try:
        bal             = kc.get_balance()
        balance_cents   = float(bal.get("balance", 0) or 0)
        portfolio_cents = float(bal.get("portfolio_value", balance_cents) or balance_cents)
        # Track current exposure for heartbeat (portfolio_value - balance = exposure)
        state.health.cycle.exposure_cents = max(0.0, portfolio_cents - balance_cents)
        logger.info("Balance: $%.2f  |  Portfolio: $%.2f  |  %s",
                    balance_cents / 100, portfolio_cents / 100, rm.summarize())
    except Exception as e:
        logger.error("Could not fetch balance: %s", e)

    # ── Kill switch check ─────────────────────────────────────────────────────
    if not config.PAPER_MODE:
        session_pnl = db.total_pnl_cents()
        if session_pnl < config.PNL_KILL_SWITCH_CENTS and not state.kill_switch_fired:
            state.kill_switch_fired = True
            alerting.alert_kill_switch(session_pnl, config.PNL_KILL_SWITCH_CENTS)
            logger.critical("[KILL SWITCH] Session P&L $%.2f < threshold $%.2f — halting trades",
                            session_pnl / 100, config.PNL_KILL_SWITCH_CENTS / 100)
        if state.kill_switch_fired:
            logger.warning("[KILL SWITCH ACTIVE] Skipping trade placement this scan.")
            metrics.print_summary()
            return

    # ── Stale order cleanup (live) ────────────────────────────────────────────
    if not config.PAPER_MODE:
        try:
            om.cleanup_stale_orders()
        except Exception as e:
            logger.error("Stale order cleanup error: %s", e)

    # ── Settlement checks ─────────────────────────────────────────────────────
    if config.PAPER_MODE:
        try:
            _check_settled_paper()
        except Exception as e:
            logger.error("Paper settlement error: %s", e)
    else:
        try:
            _check_settled_live()
        except Exception as e:
            logger.error("Live settlement error: %s", e)

    # ── Fetch markets ─────────────────────────────────────────────────────────
    # Warn early if the previous fetch was stale (soft alert, still try again)
    if state.health.last_kalshi_fetch_ok > 0:
        data_age = time.time() - state.health.last_kalshi_fetch_ok
        if data_age > config.DATA_FRESHNESS_TIMEOUT:
            alerting.alert_stale_data(data_age)
            logger.warning("[STALE DATA] Last market fetch was %.0fs ago (warn=%ds)",
                           data_age, config.DATA_FRESHNESS_TIMEOUT)

    logger.info("Scanning up to %d markets...", config.MARKETS_TO_SCAN)
    fetch_start = time.time()
    try:
        markets = kc.get_all_open_markets(max_markets=config.MARKETS_TO_SCAN)
        state.health.record_kalshi_fetch_ok()
        fetch_ms = (time.time() - fetch_start) * 1000
        logger.info("Fetched %d markets in %.0fms.", len(markets), fetch_ms)
        state.health.cycle.markets_fetched = len(markets)
    except Exception as e:
        state.health.record_kalshi_fetch_fail(e)
        logger.error("Could not fetch markets (failure #%d): %s",
                     state.health.kalshi_fail_count, e)
        if state.health.kalshi_fail_count >= config.API_FAILURE_ALERT_THRESHOLD:
            alerting.alert_api_failure(state.health.kalshi_fail_count, e)
        return

    # ── Safe-to-trade gate ────────────────────────────────────────────────────
    # Checked AFTER a successful market fetch so data_freshness is current.
    # Safe mode (API failures, order failures, stale data) blocks signal loop.
    if not config.PAPER_MODE:
        can_trade, trade_block_reason = state.health.is_safe_to_trade()
        if not can_trade:
            logger.warning("[SAFE MODE] Skipping signal evaluation: %s", trade_block_reason)
            alerting.send_alert("SAFE_MODE",
                                f"Trade gate blocked: {trade_block_reason}",
                                level="WARNING")
            metrics.print_summary()
            return

    # ── Weather strategy scan ────────────────────────────────────────────────
    #
    # Flow:
    #   evaluate → paper-record always → alert → [if live enabled] risk-check → order
    #
    # Sports execution below this block is NOT touched.
    # ─────────────────────────────────────────────────────────────────────────
    weather_signals_found = 0
    weather_orders_placed = 0

    for _mkt in markets:
        try:
            w_signal = evaluate_weather_market(_mkt)
        except Exception as _we:
            logger.warning("[WEATHER] evaluate error: %s", _we)
            continue

        if w_signal is None:
            continue

        weather_signals_found += 1
        _wticker = w_signal["ticker"]

        # ── 1. Paper-record (always, regardless of live flag) ─────────────
        try:
            record_weather_signal(w_signal, _mkt)
        except Exception as _wpe:
            logger.warning("[WEATHER] paper record failed: %s", _wpe)

        # ── 2. WEATHER_SIGNAL alert ───────────────────────────────────────
        alerting.send_alert(
            "WEATHER_SIGNAL",
            (
                f"{w_signal['city']} {_wticker} side={w_signal['side'].upper()} | "
                f"forecast={w_signal['forecast_high']:.1f}F | "
                f"model={w_signal['model_prob']:.2f} market={w_signal['market_prob']:.2f} | "
                f"raw={w_signal['raw_edge']:.2f} net={w_signal['net_edge']:.2f} | "
                f"spread={w_signal['spread_cents']:.0f}¢ conf={w_signal['confidence']:.2f}"
            ),
            level="INFO",
            details={
                "ticker":        _wticker,
                "city":          w_signal["city"],
                "side":          w_signal["side"].upper(),
                "forecast_high": f"{w_signal['forecast_high']:.1f}°F",
                "sigma_f":       f"{w_signal['sigma_f']:.1f}°F",
                "model_prob":    f"{w_signal['model_prob']:.3f}",
                "market_prob":   f"{w_signal['market_prob']:.3f}",
                "raw_edge":      f"{w_signal['raw_edge']:.3f}",
                "net_edge":      f"{w_signal['net_edge']:.3f}",
                "spread_cents":  f"{w_signal['spread_cents']:.0f}¢",
                "confidence":    f"{w_signal['confidence']:.2f}",
                "reason":        w_signal["reason"],
            },
        )

        # ── 3. Live execution branch ──────────────────────────────────────
        if not wcfg.WEATHER_LIVE_ENABLED:
            # Emit once per boot cycle, not per signal, to avoid spam
            logger.debug("[WEATHER] live disabled — signal alerted and paper-recorded only")
            continue

        if config.PAPER_MODE:
            # Sports paper mode → weather also stays in paper mode
            logger.debug("[WEATHER] PAPER_MODE active — skipping live weather order")
            continue

        # ── 3a. Risk approval ─────────────────────────────────────────────
        try:
            approved, block_reason = approve_weather_trade(w_signal)
        except Exception as _wr_exc:
            logger.error("[WEATHER] risk approval error: %s", _wr_exc)
            approved, block_reason = False, f"risk check error: {_wr_exc}"

        if not approved:
            alerting.send_alert(
                "WEATHER_RISK_BLOCKED",
                f"{_wticker} — {block_reason}",
                level="WARNING",
                details={
                    "ticker":       _wticker,
                    "block_reason": block_reason,
                    "net_edge":     f"{w_signal['net_edge']:.3f}",
                    "confidence":   f"{w_signal['confidence']:.2f}",
                },
            )
            continue

        # ── 3b. Position sizing ───────────────────────────────────────────
        try:
            contracts = compute_weather_position_size(w_signal, balance_cents / 100.0)
        except Exception as _sz_exc:
            logger.error("[WEATHER] sizing error: %s", _sz_exc)
            contracts = 1

        side      = w_signal["side"]
        yes_price = int(w_signal.get("yes_price", w_signal["market_prob"] * 100))

        # ── 3c. Place order via existing order manager ────────────────────
        try:
            result = om.place_order_safe(
                ticker           = _wticker,
                side             = side,
                count            = contracts,
                yes_price        = yes_price,
                live_open_orders = live_open_orders,
                live_positions   = live_positions,
                exec_decision    = None,
                engine           = None,
                fair_probability = w_signal["model_prob"],
            )
        except Exception as _ord_exc:
            logger.error("[WEATHER] order placement error: %s", _ord_exc)
            result = None

        if result:
            weather_orders_placed += 1
            alerting.send_alert(
                "WEATHER_ORDER_PLACED",
                (
                    f"{_wticker} {side.upper()} {contracts}@{yes_price}¢ | "
                    f"city={w_signal['city']} net={w_signal['net_edge']:.3f} "
                    f"conf={w_signal['confidence']:.2f}"
                ),
                level="INFO",
                details={
                    "ticker":    _wticker,
                    "side":      side.upper(),
                    "contracts": str(contracts),
                    "yes_price": f"{yes_price}¢",
                    "city":      w_signal["city"],
                    "net_edge":  f"{w_signal['net_edge']:.3f}",
                    "conf":      f"{w_signal['confidence']:.2f}",
                },
            )

    if weather_signals_found:
        logger.info(
            "[WEATHER] %d signal(s) found — %d paper-recorded, %d live order(s) placed.",
            weather_signals_found, weather_signals_found, weather_orders_placed,
        )
    elif not wcfg.WEATHER_LIVE_ENABLED:
        logger.debug("[WEATHER] 0 signals this scan (live disabled).")

    # ── WEATHER_LIVE_DISABLED notice (once per scan, not per market) ──────
    if not wcfg.WEATHER_LIVE_ENABLED and weather_signals_found > 0:
        alerting.send_alert(
            "WEATHER_LIVE_DISABLED",
            f"{weather_signals_found} signal(s) suppressed — set WEATHER_LIVE_ENABLED=True to trade",
            level="INFO",
        )

    # ── Signal scan (sports-data-enriched) ───────────────────────────────────
    try:
        signals = state.scanner.scan(markets)
        state.health.record_sports_fetch_ok()
    except Exception as e:
        state.health.record_sports_fetch_fail(e)
        logger.error("Scanner.scan() failed: %s", e)
        signals = []

    if not signals:
        logger.info("No approved signals after model evaluation.")
    else:
        logger.info("Found %d raw signal(s) — running portfolio evaluation.", len(signals))

    # ── Portfolio evaluation (correlation-aware allocation) ───────────────────
    port_snapshot = state.portfolio_analyzer.build_snapshot()
    ranked_signals = state.portfolio_analyzer.rank_and_evaluate(signals, port_snapshot)
    state.health.cycle.candidates_evaluated = len(ranked_signals)

    open_paper = set(paper_ledger.get_open_tickers()) if config.PAPER_MODE else set()
    acted = 0

    for sig, port_eval in ranked_signals:
        if not port_eval.approved:
            logger.info(
                "[PORTFOLIO SKIP] [%s] %s",
                sig["ticker"], port_eval.rejection_reason,
            )
            state.health.cycle.trades_skipped += 1
            continue

        ticker    = sig["ticker"]
        side      = sig["side"]
        conf      = sig["confidence"]
        progress  = sig["progress"]
        title     = sig["title"]
        yes_price = sig["yes_price"]
        entry_c   = yes_price if side == "yes" else (100 - yes_price)
        contracts = strategy.calc_contracts(portfolio_cents, entry_c, port_eval.size_multiplier)
        bet_usd   = contracts * entry_c / 100.0

        logger.info(
            "SIGNAL [%s] side=%s conf=%.1f%% progress=%.1f%% "
            "entry=%d¢ x%d contracts ($%.2f)",
            ticker, side.upper(), conf * 100, progress * 100,
            entry_c, contracts, bet_usd
        )
        logger.debug("  %s", title[:100])

        # Position cap
        if len(open_paper) + acted >= config.MAX_OPEN_POSITIONS:
            logger.info("Max positions (%d) reached.", config.MAX_OPEN_POSITIONS)
            break

        # ── PAPER MODE ────────────────────────────────────────────────────────
        if config.PAPER_MODE:
            if ticker in open_paper:
                continue
            paper_ledger.open_position(ticker, side, yes_price, contracts, title)
            open_paper.add(ticker)
            acted += 1
            continue

        # ── LIVE MODE ─────────────────────────────────────────────────────────
        allowed, reason = rm.check_trade_allowed(
            ticker, side, contracts, entry_c, balance_cents
        )
        if not allowed:
            rm.log_blocked(ticker, reason)
            alerting.send_alert("RISK_BLOCKED",
                                f"Trade blocked: {ticker} — {reason}",
                                level="INFO")
            continue

        # ── Execution layer analysis ───────────────────────────────────────
        exec_engine  = state.exec_engine
        market_data  = sig.get("market_data", {})
        volatility   = sig.get("volatility", 0.0)
        momentum     = sig.get("momentum", 0.0)
        fair_prob    = sig.get("fair_probability", 0.5)

        exec_decision = exec_engine.analyze(
            market=market_data,
            signal=sig,
            volatility=volatility,
            momentum=momentum,
        )
        if not exec_decision.ok:
            logger.info("[EXEC SKIP] [%s] %s", ticker, exec_decision.reason)
            continue

        # ── Timing layer ───────────────────────────────────────────────────
        _time_to_res = float(sig.get("time_to_resolution_secs") or
                             sig.get("seconds_to_close") or 0)
        timing_decision = state.timing_classifier.classify(
            signal           = sig,
            momentum         = momentum,
            volatility       = volatility,
            spread_cents     = exec_decision.spread_cents,
            regime           = exec_decision.regime,
            time_to_resolution = _time_to_res,
        )

        if timing_decision.entry_mode == "skip":
            state.timing_total_skips  += 1
            state.timing_chase_skips  += int(
                timing_decision.market_classification == "overreacting"
            )
            logger.info(
                "[TIMING SKIP] [%s] cls=%s urgency=%.3f reason=%s",
                ticker, timing_decision.market_classification,
                timing_decision.urgency_score, timing_decision.skip_reason,
            )
            continue

        if timing_decision.entry_mode == "wait":
            state.timing_total_skips += 1
            logger.info(
                "[TIMING WAIT] [%s] cls=%s urgency=%.3f — holding for better entry",
                ticker, timing_decision.market_classification,
                timing_decision.urgency_score,
            )
            continue

        # Apply staged-fraction scaling if requested
        _is_staged = timing_decision.entry_mode == "stage"
        if _is_staged and timing_decision.staged_fraction < 1.0:
            contracts = max(1, int(contracts * timing_decision.staged_fraction))
            bet_usd   = contracts * entry_c / 100.0
            logger.info(
                "[TIMING STAGE] [%s] urgency=%.3f → staged %.0f%% = %d contracts",
                ticker, timing_decision.urgency_score,
                timing_decision.staged_fraction * 100, contracts,
            )

        # Market type: evaluate_signal() classifies and injects it into the signal.
        # Fall back to classifier if somehow absent (backward compat).
        _mkt_title   = market_data.get("title", "") or ticker
        _mkt_sport   = sig.get("sport", "Generic") or "Generic"
        _market_type = sig.get("market_type") or ""
        if not _market_type:
            _mkt_cls     = classifier.classify(ticker, _mkt_title, _mkt_sport)
            _market_type = _mkt_cls["market_type"]
        logger.debug(
            "[CLASSIFY] %s → mtype=%s model=%s timing=%s urgency=%.3f",
            ticker, _market_type, sig.get("model_name", "—"),
            timing_decision.market_classification, timing_decision.urgency_score,
        )

        # Register rich context so analytics can capture it when the fill
        # is confirmed (analytics.record_fill_event is called in order_manager)
        analytics.register_entry_context(ticker, {
            "side":               side,
            "sport":              _mkt_sport,
            "market_type":        _market_type,
            "model_name":         sig.get("model_name", ""),
            "parsed_line":        sig.get("parsed_line"),
            "signal_reason":      sig.get("signal_reason", ""),
            "matched_event_id":   sig.get("matched_event_id", ""),
            "regime":             exec_decision.regime,
            "exec_mode":          exec_decision.mode,
            "confidence_score":   sig.get("confidence_score", 0),
            "fair_probability":   sig.get("fair_probability", 0.5),
            "market_probability": sig.get("market_probability", yes_price / 100),
            "raw_edge":           sig.get("raw_edge", 0),
            "edge_after_slip":    exec_decision.edge_after_slip,
            "spread_at_entry":    exec_decision.spread_cents,
            "entry_price":        exec_decision.entry_price,
            # v4 portfolio fields
            "overlap_level":                  port_eval.overlap_level,
            "concentration_score":            port_eval.concentration_score,
            "allocation_rank":                port_eval.allocation_rank,
            "portfolio_event_exposure_cents": port_eval.current_event_exposure_cents,
            "portfolio_sport_exposure_cents": port_eval.current_sport_exposure_cents,
            # v5 timing fields
            "entry_timing_classification": timing_decision.market_classification,
            "urgency_score":               timing_decision.urgency_score,
            "staged_entry_flag":           int(_is_staged),
            "is_add_entry":                int(bool(sig.get("is_add_entry", False))),
            "missed_edge_cents":           0.0,
        })

        result = om.place_order_safe(
            ticker=ticker, side=side, count=contracts,
            yes_price=yes_price,
            live_open_orders=live_open_orders,
            live_positions=live_positions,
            exec_decision=exec_decision,
            engine=exec_engine,
            fair_probability=fair_prob,
        )
        if result:
            acted += 1
            state.health.cycle.trades_placed += 1
            state.health.record_order_placed()

    # ── Summary ───────────────────────────────────────────────────────────────
    if config.PAPER_MODE:
        paper_ledger.print_summary()
    else:
        positions = db.get_all_positions()
        pnl       = db.total_pnl_cents()
        logger.info("Scan complete. Open positions: %d  |  Session P&L: $%.2f",
                    len(positions), pnl / 100)
        metrics.print_summary()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(BANNER)
    mode = "LIVE TRADING" if not config.PAPER_MODE else "PAPER TRADING (no real money)"
    logger.info("Mode:           %s", mode)
    logger.info("Confidence:     %.0f%%  |  Progress gate: %.0f%%",
                config.MIN_CONFIDENCE * 100, config.GAME_PROGRESS_THRESHOLD * 100)
    logger.info("Bet size:       %.0f%% of portfolio  |  Scan: every %ds",
                config.BET_FRACTION * 100, config.SCAN_INTERVAL)
    logger.info("Logs:           %s", config.LOG_PATH)
    logger.info("Metrics:        %s", config.METRICS_DIR)

    db.init_db()
    os.makedirs(config.METRICS_DIR, exist_ok=True)

    # Health monitor created first — tracks all liveness from this point on
    health_monitor = HealthMonitor()

    # Create long-lived objects once — they accumulate state across cycles
    scanner            = Scanner()
    exec_engine        = ExecutionEngine()
    portfolio_analyzer = PortfolioAnalyzer()
    timing_classifier  = TimingClassifier()

    live_open_orders: list = []
    live_positions:   dict = {}
    reconcile_clean:  bool = True

    if not config.PAPER_MODE:
        logger.warning("LIVE MODE ACTIVE — real orders will be placed")
        live_open_orders, live_positions, reconcile_clean = startup_reconcile()

        # Full system snapshot + restart alert (before regular BOT_STARTED)
        health_monitor.log_system_snapshot(
            mode            = "LIVE",
            open_orders     = len(live_open_orders),
            open_positions  = len(live_positions),
            db_positions    = len(db.get_all_positions()),
            session_pnl     = db.total_pnl_cents(),
            reconcile_clean = reconcile_clean,
        )
        alerting.alert_bot_started(
            mode="LIVE",
            confidence=config.MIN_CONFIDENCE,
            progress=config.GAME_PROGRESS_THRESHOLD,
        )
    else:
        alerting.alert_bot_started(
            mode="PAPER",
            confidence=config.MIN_CONFIDENCE,
            progress=config.GAME_PROGRESS_THRESHOLD,
        )

    state            = _ScanState(
        scanner, exec_engine, portfolio_analyzer, timing_classifier, health_monitor
    )
    iteration        = 0
    report_run_count = 0
    _report_thread: threading.Thread | None = None

    # Start watchdog — monitors heartbeat from now on
    watchdog = Watchdog(health_monitor)
    watchdog.start()

    while True:
        iteration += 1
        logger.info("─" * 50)
        logger.info("Iteration #%d", iteration)

        try:
            run_scan(live_open_orders, live_positions, state)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt — shutting down.")
            watchdog.stop()
            break
        except Exception as e:
            logger.error("Unhandled error in scan #%d: %s", iteration, e, exc_info=True)
            alerting.send_alert("BOT_ERROR",
                                f"Unhandled error in scan #{iteration}: {e}",
                                level="CRITICAL")
            logger.info("Waiting 10s before retrying...")
            time.sleep(10)
            # Still emit heartbeat so watchdog doesn't fire during error recovery
            state.health.last_heartbeat = time.time()
            continue

        # ── Heartbeat + cycle stats (every iteration) ─────────────────────────
        try:
            state.health.record_cycle_end()
        except Exception as e:
            logger.warning("[HEALTH] record_cycle_end error: %s", e)

        # ── Light metrics: every iteration (fast, in-memory) ─────────────────
        if not config.PAPER_MODE:
            try:
                analytics.log_summary()
            except Exception as e:
                logger.warning("[ANALYTICS] log_summary error: %s", e)

        # ── Heavy reports: throttled + non-blocking ───────────────────────────
        # Only run every REPORT_INTERVAL_ITERS iterations (default 20 = ~5 min).
        # Launched in a daemon thread so disk I/O never stalls the scan loop.
        # If the previous thread is still running we skip to avoid pile-up.
        if (
            not config.PAPER_MODE
            and config.REPORT_INTERVAL_ITERS > 0
            and iteration % config.REPORT_INTERVAL_ITERS == 0
        ):
            if _report_thread is not None and _report_thread.is_alive():
                logger.debug("[REPORTS] Previous report thread still running — skipping this cycle.")
            else:
                report_run_count += 1
                _include_bt = (
                    config.BACKTEST_INTERVAL_REPORTS > 0
                    and report_run_count % config.BACKTEST_INTERVAL_REPORTS == 0
                )

                def _run_reports(include_bt: bool = _include_bt) -> None:
                    try:
                        reports.generate_all(include_backtest=include_bt)
                        logger.debug(
                            "[REPORTS] Run #%d complete (backtest=%s)",
                            report_run_count, include_bt,
                        )
                    except Exception as exc:
                        logger.warning("[REPORTS] Generation error: %s", exc)

                _report_thread = threading.Thread(
                    target=_run_reports, daemon=True, name="report-writer"
                )
                _report_thread.start()
                logger.debug(
                    "[REPORTS] Launched report thread (run #%d, backtest=%s)",
                    report_run_count, _include_bt,
                )

        # ── Periodic parameter optimization ───────────────────────────────────
        # Runs every OPT_INTERVAL_ITERS iterations (default: every 100).
        # Writes optimization_results.csv, optimization_summary.json,
        # and best_parameters.json — but NEVER auto-applies any settings.
        if (
            not config.PAPER_MODE
            and config.OPT_INTERVAL_ITERS > 0
            and iteration % config.OPT_INTERVAL_ITERS == 0
        ):
            try:
                closed_count = len(db.get_closed_trade_analytics())
                if closed_count >= config.OPT_MIN_TRADES:
                    logger.info(
                        "[OPT] Running parameter optimization on %d closed trades "
                        "(every %d iters)...",
                        closed_count, config.OPT_INTERVAL_ITERS,
                    )
                    optimizer = opt_module.ParameterOptimizer(
                        search_mode=config.OPT_SEARCH_MODE,
                        random_samples=config.OPT_RANDOM_SAMPLES,
                    )
                    result = optimizer.run()
                    if result.get("status") == "ok":
                        logger.info(
                            "[OPT] Best global config: score=%.3f  "
                            "edge≥%.0f%% conf≥%.0f%% spread≤%.0f¢ tp=%s regime=%s",
                            result["best_score"],
                            float(result["best_params"].get("min_raw_edge", 0)) * 100,
                            float(result["best_params"].get("min_confidence", 0)) * 100,
                            float(result["best_params"].get("max_spread_cents", 0)),
                            result["best_params"].get("take_profit_cents"),
                            result["best_params"].get("only_regime") or "all",
                        )
                        # Sport-specific optimization using global best as baseline
                        sport_results: dict = {}
                        if config.OPT_RUN_SPORT_SPECIFIC:
                            sport_results = optimizer.run_sport_optimizations(
                                global_best_params=result["best_params"]
                            )
                            logger.info(
                                "[OPT] Sport optimization: %d sports evaluated "
                                "(%d optimized, %d fallback-to-global).",
                                len(sport_results),
                                sum(1 for r in sport_results.values()
                                    if r.get("status") == "ok"),
                                sum(1 for r in sport_results.values()
                                    if r.get("status") == "insufficient_data"),
                            )

                        # Market-type optimization using both baselines
                        if config.OPT_RUN_MARKET_TYPE:
                            mt_results = optimizer.run_market_type_optimizations(
                                global_best_params=result["best_params"],
                                sport_results=sport_results,
                            )
                            logger.info(
                                "[OPT] Market-type optimization: %d buckets "
                                "(%d optimized, %d fallback).",
                                len(mt_results),
                                sum(1 for r in mt_results.values()
                                    if r.get("status") == "ok"),
                                sum(1 for r in mt_results.values()
                                    if r.get("status") == "insufficient_data"),
                            )
                else:
                    logger.info(
                        "[OPT] Skipping optimization: only %d closed trades "
                        "(need %d).",
                        closed_count, config.OPT_MIN_TRADES,
                    )
            except Exception as e:
                logger.warning("[OPT] Optimization error: %s", e)

        logger.info("Sleeping %ds...", config.SCAN_INTERVAL)
        time.sleep(config.SCAN_INTERVAL)


if __name__ == "__main__":
    main()
