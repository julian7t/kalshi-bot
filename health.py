"""
health.py — Bot health monitoring, liveness tracking, and safe mode.

HealthMonitor is the single source of truth for whether the bot is healthy
enough to place trades.  It tracks:

  - Last successful fetch timestamps (Kalshi markets, sports data)
  - Consecutive failure counts per data feed
  - Safe mode state  (scan / log only — no order placement)
  - Loop heartbeat timestamp  (checked by the watchdog thread)
  - Per-cycle counters for the structured heartbeat log line
  - Health status JSON written atomically each cycle to metrics/health.json

Safe mode activates automatically when (and only when SAFE_MODE_AUTO=true):
  - Kalshi API fails >= KALSHI_FAILURE_THRESHOLD consecutive times
  - Sports data fails >= SPORTS_FAILURE_THRESHOLD consecutive times
  - Market data is stale beyond DATA_STALE_HALT_SECS
  - Order placement fails >= MAX_ORDER_FAILURE_STREAK consecutive times

Safe mode deactivates automatically when the failing feed recovers AND no
other feed is still failing.  If the startup reconciliation had errors, safe
mode is activated until the operator manually deactivates by restarting.

Thread safety: all write operations happen in the main loop thread.  The
watchdog daemon only reads last_heartbeat (float), which is GIL-safe.
"""

import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import config

logger = logging.getLogger("kalshi_bot.health")


# ── Atomic JSON writer ─────────────────────────────────────────────────────────

def _atomic_write_json(path: str, data: dict) -> None:
    """Write data atomically using temp-file + rename (never corrupts the file)."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            dir=os.path.dirname(path), suffix=".tmp", prefix=".health_"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except Exception as e:
        logger.debug("[HEALTH] Could not write health file: %s", e)


# ── Cycle stats (reset each scan) ─────────────────────────────────────────────

@dataclass
class CycleStats:
    start_time:           float = 0.0
    markets_fetched:      int   = 0
    candidates_evaluated: int   = 0
    trades_placed:        int   = 0
    trades_skipped:       int   = 0
    exposure_cents:       float = 0.0


# ── Health monitor ─────────────────────────────────────────────────────────────

class HealthMonitor:
    """
    Central health monitor.  Create once in main(); store in _ScanState.

    Call order each cycle:
        start = monitor.record_cycle_start()
        ... run_scan, incrementing monitor.cycle.X counters as you go ...
        monitor.record_cycle_end()
    """

    def __init__(self):
        # Watchdog heartbeat (monotonic; safe to read from watchdog thread)
        self.last_heartbeat:          float = time.time()

        # Feed success timestamps (0 = never fetched this session)
        self.last_kalshi_fetch_ok:    float = 0.0
        self.last_sports_fetch_ok:    float = 0.0
        self.last_order_placed:       float = 0.0

        # Consecutive failure counters (reset to 0 on first success)
        self.kalshi_fail_count:       int   = 0
        self.sports_fail_count:       int   = 0
        self.order_fail_streak:       int   = 0

        # Safe mode
        self._safe_mode:              bool  = False
        self._safe_mode_reason:       str   = ""

        # State integrity
        self.last_integrity_ok:       bool  = True
        self.integrity_fail_count:    int   = 0

        # Session-level counters (never reset)
        self.total_cycles:            int   = 0
        self.total_trades:            int   = 0
        self.total_skipped:           int   = 0
        self.consecutive_idle_cycles: int   = 0

        # Current-cycle mutable stats (reset by record_cycle_start)
        self.cycle: CycleStats = CycleStats()

    # ── Cycle boundary ────────────────────────────────────────────────────────

    def record_cycle_start(self) -> float:
        """Call at the very top of each scan iteration.  Returns wall-clock start."""
        self.cycle       = CycleStats(start_time=time.time())
        self.total_cycles += 1
        return self.cycle.start_time

    def record_cycle_end(self) -> None:
        """
        Call at the END of each iteration (after run_scan() returns).

        Reads counters from self.cycle (mutated during run_scan), emits the
        structured heartbeat log line, and writes health.json.
        """
        now     = time.time()
        loop_ms = (now - self.cycle.start_time) * 1000 if self.cycle.start_time else 0

        self.last_heartbeat  = now
        self.total_trades   += self.cycle.trades_placed
        self.total_skipped  += self.cycle.trades_skipped

        if self.cycle.candidates_evaluated == 0 and self.cycle.trades_placed == 0:
            self.consecutive_idle_cycles += 1
        else:
            self.consecutive_idle_cycles  = 0

        logger.info(
            "[HEARTBEAT] cycle=%d loop=%.0fms markets=%d candidates=%d "
            "placed=%d skipped=%d exposure=$%.2f safe=%s",
            self.total_cycles,
            loop_ms,
            self.cycle.markets_fetched,
            self.cycle.candidates_evaluated,
            self.cycle.trades_placed,
            self.cycle.trades_skipped,
            self.cycle.exposure_cents / 100,
            "YES" if self._safe_mode else "no",
        )

        self._write_health_file(loop_ms)

    # ── Kalshi feed tracking ──────────────────────────────────────────────────

    def record_kalshi_fetch_ok(self) -> None:
        was_failing = self.kalshi_fail_count >= config.KALSHI_FAILURE_THRESHOLD
        self.last_kalshi_fetch_ok = time.time()
        self.kalshi_fail_count    = 0
        if was_failing:
            logger.info("[HEALTH] Kalshi API recovered.")
            self._maybe_deactivate_safe_mode("kalshi_api")

    def record_kalshi_fetch_fail(self, error: Exception) -> None:
        self.kalshi_fail_count += 1
        logger.warning("[HEALTH] Kalshi API failure #%d: %s", self.kalshi_fail_count, error)
        if (
            config.SAFE_MODE_AUTO
            and self.kalshi_fail_count >= config.KALSHI_FAILURE_THRESHOLD
        ):
            self.activate_safe_mode(
                f"Kalshi API: {self.kalshi_fail_count} consecutive failures — {error}"
            )

    # ── Sports data feed tracking ─────────────────────────────────────────────

    def record_sports_fetch_ok(self) -> None:
        was_failing = self.sports_fail_count >= config.SPORTS_FAILURE_THRESHOLD
        self.last_sports_fetch_ok = time.time()
        self.sports_fail_count    = 0
        if was_failing:
            logger.info("[HEALTH] Sports data feed recovered.")
            self._maybe_deactivate_safe_mode("sports_data")

    def record_sports_fetch_fail(self, error: Exception) -> None:
        self.sports_fail_count += 1
        logger.warning("[HEALTH] Sports data failure #%d: %s", self.sports_fail_count, error)
        if (
            config.SAFE_MODE_AUTO
            and self.sports_fail_count >= config.SPORTS_FAILURE_THRESHOLD
        ):
            self.activate_safe_mode(
                f"Sports data: {self.sports_fail_count} consecutive failures — {error}"
            )

    # ── Order tracking ────────────────────────────────────────────────────────

    def record_order_placed(self) -> None:
        """Call each time an order is successfully placed."""
        self.last_order_placed = time.time()
        self.order_fail_streak = 0

    def record_order_failed(self, error: Exception) -> None:
        """Call when order placement itself fails (API error, not a risk skip)."""
        import alerting
        self.order_fail_streak += 1
        level = (
            "CRITICAL"
            if self.order_fail_streak >= config.MAX_ORDER_FAILURE_STREAK
            else "WARNING"
        )
        logger.warning("[HEALTH] Order failure streak=%d: %s", self.order_fail_streak, error)
        alerting.send_alert(
            "ORDER_FAILURE_STREAK",
            f"Order placement failure #{self.order_fail_streak}",
            level=level,
            details={"error": str(error)[:200], "streak": self.order_fail_streak},
        )
        if (
            config.SAFE_MODE_AUTO
            and self.order_fail_streak >= config.MAX_ORDER_FAILURE_STREAK
        ):
            self.activate_safe_mode(
                f"Order failures: {self.order_fail_streak} consecutive — last: {error}"
            )

    # ── Freshness / stale-data gate ───────────────────────────────────────────

    def check_data_freshness(self) -> tuple[bool, str]:
        """
        Return (ok: bool, reason: str).
        ok=False means data is critically stale — halt trading immediately.
        """
        now = time.time()

        # Kalshi market data
        if self.last_kalshi_fetch_ok > 0:
            age = now - self.last_kalshi_fetch_ok
            if age > config.DATA_STALE_HALT_SECS:
                return (
                    False,
                    f"Kalshi market data stale {age:.0f}s "
                    f"(halt threshold={config.DATA_STALE_HALT_SECS}s)",
                )

        # Sports data: 2× lenient — it enriches signals but is not required
        # for order routing.  Only halt if the model truly has nothing.
        if self.last_sports_fetch_ok > 0:
            sports_limit = config.DATA_STALE_HALT_SECS * 2
            age = now - self.last_sports_fetch_ok
            if age > sports_limit:
                return (
                    False,
                    f"Sports data stale {age:.0f}s (halt threshold={sports_limit}s)",
                )

        return True, ""

    # ── State integrity ───────────────────────────────────────────────────────

    def record_integrity_ok(self) -> None:
        if not self.last_integrity_ok:
            logger.info("[HEALTH] State integrity restored.")
        self.last_integrity_ok    = True
        self.integrity_fail_count = 0

    def record_integrity_fail(self, diff_summary: str) -> None:
        """
        Call when the per-cycle integrity check detects a mismatch.
        Does NOT activate safe mode — loop_reconcile() already corrects state.
        """
        import alerting
        self.last_integrity_ok     = False
        self.integrity_fail_count += 1
        logger.error(
            "[HEALTH] State integrity failure #%d: %s",
            self.integrity_fail_count, diff_summary,
        )
        alerting.send_alert(
            "INTEGRITY_MISMATCH",
            f"State integrity check failed (#{self.integrity_fail_count})",
            level="CRITICAL" if self.integrity_fail_count >= 3 else "WARNING",
            details={"diff": diff_summary[:300]},
        )

    # ── is_safe_to_trade ──────────────────────────────────────────────────────

    def is_safe_to_trade(self) -> tuple[bool, str]:
        """
        Master trade-eligibility gate.  Call before entering the signal loop.

        Returns (can_trade: bool, reason: str).
        reason is empty when can_trade=True.
        """
        if self._safe_mode:
            return False, f"safe_mode: {self._safe_mode_reason}"

        fresh_ok, stale_reason = self.check_data_freshness()
        if not fresh_ok:
            return False, f"stale_data: {stale_reason}"

        return True, ""

    # ── Safe mode ─────────────────────────────────────────────────────────────

    @property
    def safe_mode(self) -> bool:
        return self._safe_mode

    def activate_safe_mode(self, reason: str) -> None:
        """Enter safe mode.  Idempotent — safe to call repeatedly."""
        import alerting
        if not self._safe_mode:
            self._safe_mode        = True
            self._safe_mode_reason = reason
            logger.critical("[SAFE MODE ACTIVATED] %s", reason)
            alerting.send_alert(
                "SAFE_MODE",
                "Bot entered SAFE MODE — scanning only, NO orders will be placed",
                level="CRITICAL",
                details={"reason": reason[:300]},
            )

    def deactivate_safe_mode(self, reason: str = "operator request") -> None:
        """Manually deactivate safe mode (e.g., after operator review)."""
        import alerting
        if self._safe_mode:
            self._safe_mode        = False
            self._safe_mode_reason = ""
            logger.info("[SAFE MODE CLEARED] %s", reason)
            alerting.send_alert(
                "SAFE_MODE_CLEARED",
                "Bot exited safe mode — trading resumed",
                level="INFO",
                details={"reason": reason},
            )

    def _maybe_deactivate_safe_mode(self, recovered_feed: str) -> None:
        """
        Auto-deactivate if ALL feeds are now healthy.
        Conservative: stays in safe mode if any feed is still failing.
        """
        if not self._safe_mode:
            return

        if self.kalshi_fail_count > 0 or self.sports_fail_count > 0:
            logger.info(
                "[SAFE MODE] %s recovered but other failures remain — staying in safe mode.",
                recovered_feed,
            )
            return

        fresh_ok, _ = self.check_data_freshness()
        if not fresh_ok:
            logger.info(
                "[SAFE MODE] %s recovered but data still stale — staying in safe mode.",
                recovered_feed,
            )
            return

        self.deactivate_safe_mode(f"all feeds healthy after {recovered_feed} recovery")

    # ── Startup snapshot ──────────────────────────────────────────────────────

    def log_system_snapshot(
        self,
        mode:            str,
        open_orders:     int,
        open_positions:  int,
        db_positions:    int,
        session_pnl:     float,
        reconcile_clean: bool,
    ) -> None:
        """
        Called once at startup to log full state and alert operators.
        If reconcile wasn't clean, activates safe mode until operator restarts cleanly.
        """
        import alerting
        logger.info(
            "[STARTUP SNAPSHOT] mode=%s open_orders=%d kalshi_positions=%d "
            "db_positions=%d session_pnl=$%.2f reconcile_clean=%s",
            mode, open_orders, open_positions, db_positions,
            session_pnl / 100, reconcile_clean,
        )
        alerting.send_alert(
            "BOT_RESTARTED",
            f"Bot restarted — {mode} mode",
            level="WARNING",
            details={
                "open_orders":     open_orders,
                "kalshi_positions": open_positions,
                "db_positions":    db_positions,
                "session_pnl":     f"${session_pnl/100:.2f}",
                "reconcile_clean": reconcile_clean,
            },
        )
        if not reconcile_clean:
            self.activate_safe_mode(
                "Startup reconciliation had errors — manual review required before trading."
            )

    # ── Health file ───────────────────────────────────────────────────────────

    def _write_health_file(self, loop_ms: float) -> None:
        now = datetime.now(timezone.utc)

        def _iso(ts: float) -> Optional[str]:
            return (
                datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
                if ts else None
            )

        data = {
            "ts":                      now.isoformat(),
            "status":                  "safe_mode" if self._safe_mode else "ok",
            "safe_mode":               self._safe_mode,
            "safe_mode_reason":        self._safe_mode_reason,
            "cycle":                   self.total_cycles,
            "loop_ms":                 round(loop_ms, 1),
            "markets_fetched":         self.cycle.markets_fetched,
            "candidates_evaluated":    self.cycle.candidates_evaluated,
            "trades_placed_this_cycle": self.cycle.trades_placed,
            "trades_total":            self.total_trades,
            "exposure_cents":          round(self.cycle.exposure_cents, 2),
            "consecutive_idle":        self.consecutive_idle_cycles,
            "last_kalshi_fetch_ok":    _iso(self.last_kalshi_fetch_ok),
            "last_sports_fetch_ok":    _iso(self.last_sports_fetch_ok),
            "last_order_placed":       _iso(self.last_order_placed),
            "last_heartbeat":          now.isoformat(),
            "kalshi_fail_count":       self.kalshi_fail_count,
            "sports_fail_count":       self.sports_fail_count,
            "order_fail_streak":       self.order_fail_streak,
            "integrity_fail_count":    self.integrity_fail_count,
            "integrity_ok":            self.last_integrity_ok,
        }
        _atomic_write_json(config.HEALTH_FILE_PATH, data)
