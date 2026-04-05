"""
watchdog.py — Background stall-detection thread.

Checks that the main loop's heartbeat has updated within WATCHDOG_STALL_SECS.
If not, sends a CRITICAL alert.  Also alerts when the bot has been idle
(no candidates found) for too many consecutive cycles, which may indicate the
scanner is misconfigured or markets aren't loading.

Does NOT attempt auto-restart — that requires operator intervention.

Design:
  - Single daemon thread, created in main() after the first scan completes
  - Reads health.last_heartbeat and health.consecutive_idle_cycles (GIL-safe)
  - Uses threading.Event for a clean shutdown path
  - All alert calls are fire-and-forget and never raise
"""

import logging
import threading
import time

import alerting
import config

logger = logging.getLogger("kalshi_bot.watchdog")


class Watchdog:
    """
    Daemon thread that monitors main-loop liveness.

    Usage:
        wd = Watchdog(health_monitor)
        wd.start()          # call once, after startup reconciliation
        ...
        wd.stop()           # on clean shutdown (KeyboardInterrupt)
    """

    def __init__(self, health) -> None:
        self._health  = health
        self._stop    = threading.Event()
        self._thread  = threading.Thread(
            target=self._run, daemon=True, name="watchdog"
        )

    def start(self) -> None:
        self._thread.start()
        logger.info(
            "[WATCHDOG] Started — stall_threshold=%ds  check_interval=%ds  idle_alert=%d cycles",
            config.WATCHDOG_STALL_SECS,
            config.WATCHDOG_CHECK_INTERVAL,
            config.WATCHDOG_IDLE_CYCLES,
        )

    def stop(self) -> None:
        self._stop.set()
        logger.info("[WATCHDOG] Stop requested.")

    def _run(self) -> None:
        """Main watchdog loop — runs on the daemon thread."""
        while not self._stop.is_set():
            # Sleep in small chunks so stop() is responsive
            self._stop.wait(config.WATCHDOG_CHECK_INTERVAL)
            if self._stop.is_set():
                break

            self._check_stall()
            self._check_idle()

    # ── Checks ────────────────────────────────────────────────────────────────

    def _check_stall(self) -> None:
        """Alert if the heartbeat hasn't updated within WATCHDOG_STALL_SECS."""
        age = time.time() - self._health.last_heartbeat
        if age <= config.WATCHDOG_STALL_SECS:
            return

        logger.critical(
            "[WATCHDOG] Main loop stalled — no heartbeat for %.0fs (threshold=%ds)",
            age, config.WATCHDOG_STALL_SECS,
        )
        alerting.send_alert(
            "WATCHDOG_STALL",
            f"Main loop stalled for {age:.0f}s — bot may be hung",
            level="CRITICAL",
            details={
                "heartbeat_age_secs": f"{age:.0f}",
                "threshold_secs":     config.WATCHDOG_STALL_SECS,
            },
        )

    def _check_idle(self) -> None:
        """Alert if the scanner has produced no candidates for too many cycles."""
        if config.WATCHDOG_IDLE_CYCLES <= 0:
            return

        idle = self._health.consecutive_idle_cycles
        if idle < config.WATCHDOG_IDLE_CYCLES:
            return

        # Only alert at each new multiple of the threshold to avoid flooding
        if idle % config.WATCHDOG_IDLE_CYCLES != 0:
            return

        logger.warning(
            "[WATCHDOG] %d consecutive idle cycles — scanner may be misconfigured",
            idle,
        )
        alerting.send_alert(
            "WATCHDOG_IDLE",
            f"Bot idle for {idle} consecutive scan cycles with no candidates",
            level="WARNING",
            details={
                "idle_cycles": idle,
                "threshold":   config.WATCHDOG_IDLE_CYCLES,
            },
        )
