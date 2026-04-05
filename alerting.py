"""
alerting.py — Structured alerts via Discord webhook and/or Telegram.

Supports both channels simultaneously. Configure with environment variables:
  DISCORD_WEBHOOK_URL   — full Discord webhook URL
  TELEGRAM_BOT_TOKEN    — bot token from @BotFather
  TELEGRAM_CHAT_ID      — numeric chat ID to send messages to

Alert types:
  BOT_STARTED           — bot came online
  KILL_SWITCH           — P&L kill switch triggered, trading halted
  AUTH_ERROR            — Kalshi authentication failure
  API_FAILURE           — N consecutive API failures
  RECONCILIATION_MISMATCH — DB vs Kalshi state divergence detected
  ORDER_PLACED          — live order submitted
  ORDER_FILLED          — order confirmed filled
  ORDER_PARTIAL         — order partially filled
  ORDER_CANCELED        — order canceled (including partial cancels)
  ORDER_FAILED          — placement API call failed
  STALE_DATA            — market data hasn't refreshed within timeout
  RISK_BLOCKED          — trade blocked by risk manager
  BOT_ERROR             — unhandled exception in main loop

Rate limiting: same alert_type will not fire more than once per ALERT_COOLDOWN_SECONDS
to prevent flooding on repeated issues. CRITICAL alerts bypass the cooldown.
"""

import logging
import os
import time
from datetime import datetime, timezone

import requests

logger = logging.getLogger("kalshi_bot.alerting")

# ── Configuration ─────────────────────────────────────────────────────────────

DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")
TELEGRAM_BOT_TOKEN  = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID    = os.environ.get("TELEGRAM_CHAT_ID", "")
ALERT_COOLDOWN      = int(os.environ.get("ALERT_COOLDOWN_SECONDS", "300"))  # 5 minutes

# Color map for Discord embeds
_DISCORD_COLORS = {
    "INFO":     0x00CC66,   # green
    "WARNING":  0xFFAA00,   # amber
    "CRITICAL": 0xFF3333,   # red
}

# Rate limit state: {alert_type: last_sent_timestamp}
_last_sent: dict[str, float] = {}


# ── Public API ────────────────────────────────────────────────────────────────

def send_alert(alert_type: str, message: str, level: str = "INFO", details: dict = None):
    """
    Send a structured alert to all configured channels.

    Never raises — a failed alert must never crash the bot.
    CRITICAL alerts bypass rate limiting.

    Args:
        alert_type: One of the defined alert type constants (str key)
        message:    Human-readable summary of what happened
        level:      "INFO" | "WARNING" | "CRITICAL"
        details:    Optional dict of key→value fields appended to the message
    """
    # Rate limit check (CRITICAL bypasses)
    if level != "CRITICAL":
        last = _last_sent.get(alert_type, 0)
        if time.time() - last < ALERT_COOLDOWN:
            logger.debug("[ALERT SUPPRESSED] %s (cooldown active)", alert_type)
            return

    _last_sent[alert_type] = time.time()

    ts  = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    full_msg = message
    if details:
        detail_lines = "\n".join(f"  {k}: {v}" for k, v in details.items())
        full_msg = f"{message}\n{detail_lines}"

    logger.info("[ALERT][%s][%s] %s", level, alert_type, full_msg)

    # Send to all configured channels in parallel (fire-and-forget, errors swallowed)
    if DISCORD_WEBHOOK_URL:
        _send_discord(alert_type, full_msg, level, ts)

    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        _send_telegram(alert_type, full_msg, level, ts)


# ── Discord ───────────────────────────────────────────────────────────────────

def _send_discord(alert_type: str, message: str, level: str, ts: str):
    """Send a formatted embed to a Discord webhook."""
    try:
        payload = {
            "embeds": [{
                "title":       f"[{level}] {alert_type}",
                "description": message[:4000],   # Discord embed limit
                "color":       _DISCORD_COLORS.get(level, 0xAAAAAA),
                "footer":      {"text": f"Kalshi Bot  •  {ts}"},
            }]
        }
        resp = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=5)
        if not resp.ok:
            logger.warning("[DISCORD ALERT FAILED] HTTP %d: %s", resp.status_code, resp.text[:200])
    except Exception as e:
        logger.warning("[DISCORD ALERT ERROR] %s", e)


# ── Telegram ──────────────────────────────────────────────────────────────────

_LEVEL_EMOJI = {"INFO": "ℹ️", "WARNING": "⚠️", "CRITICAL": "🚨"}

def _send_telegram(alert_type: str, message: str, level: str, ts: str):
    """Send a Markdown message to a Telegram chat."""
    try:
        emoji = _LEVEL_EMOJI.get(level, "📌")
        text  = (
            f"{emoji} *{_escape_md(alert_type)}* \\[{_escape_md(level)}\\]\n"
            f"{_escape_md(message)}\n"
            f"_{_escape_md(ts)}_"
        )
        url  = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        resp = requests.post(
            url,
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "MarkdownV2"},
            timeout=5,
        )
        if not resp.ok:
            logger.warning("[TELEGRAM ALERT FAILED] HTTP %d: %s", resp.status_code, resp.text[:200])
    except Exception as e:
        logger.warning("[TELEGRAM ALERT ERROR] %s", e)


def _escape_md(text: str) -> str:
    """Escape special MarkdownV2 characters for Telegram."""
    specials = r"\_*[]()~`>#+-=|{}.!"
    for ch in specials:
        text = text.replace(ch, f"\\{ch}")
    return text


# ── Convenience helpers ───────────────────────────────────────────────────────

def alert_order_placed(ticker: str, side: str, count: int, yes_price: int, client_id: str):
    send_alert("ORDER_PLACED",
               f"Placed {side.upper()} order on {ticker}",
               level="INFO",
               details={"contracts": count, "yes_price": f"{yes_price}¢",
                        "client_id": client_id})


def alert_order_filled(ticker: str, side: str, filled: int, avg_price, slippage: float):
    send_alert("ORDER_FILLED",
               f"Filled {side.upper()} {ticker}: {filled} contract(s)",
               level="INFO",
               details={"avg_fill_price": f"{avg_price}¢",
                        "slippage": f"{slippage:+.1f}¢"})


def alert_partial_cancel(ticker: str, side: str, filled: int, requested: int, kalshi_id: str):
    send_alert("ORDER_PARTIAL",
               f"Partial fill on {ticker}: {filled}/{requested} contracts filled before cancel",
               level="WARNING",
               details={"side": side.upper(), "kalshi_id": kalshi_id[:8]})


def alert_kill_switch(pnl_cents: float, threshold_cents: float):
    send_alert("KILL_SWITCH",
               f"Trading halted — P&L kill switch triggered",
               level="CRITICAL",
               details={"session_pnl": f"${pnl_cents/100:.2f}",
                        "threshold":   f"${threshold_cents/100:.2f}"})


def alert_api_failure(failures: int, error: str):
    send_alert("API_FAILURE",
               f"{failures} consecutive API failures — bot may be degraded",
               level="CRITICAL",
               details={"last_error": str(error)[:200]})


def alert_stale_data(age_seconds: float):
    send_alert("STALE_DATA",
               f"Market data is {age_seconds:.0f}s old — skipping trading this iteration",
               level="WARNING")


def alert_reconciliation_mismatch(count: int, description: str):
    send_alert("RECONCILIATION_MISMATCH",
               f"{count} DB/Kalshi state mismatch(es) corrected",
               level="WARNING",
               details={"details": description[:300]})


def alert_bot_started(mode: str, confidence: float, progress: float):
    send_alert("BOT_STARTED",
               f"Kalshi bot online in {mode} mode",
               level="INFO",
               details={"min_confidence": f"{confidence*100:.0f}%",
                        "progress_gate":  f"{progress*100:.0f}%"})


def alert_safe_mode_activated(reason: str):
    send_alert("SAFE_MODE",
               "Bot entered SAFE MODE — scanning only, no orders will be placed",
               level="CRITICAL",
               details={"reason": reason[:300]})


def alert_safe_mode_cleared(recovered_feed: str = ""):
    send_alert("SAFE_MODE_CLEARED",
               "Bot exited safe mode — trading resumed",
               level="INFO",
               details={"recovered": recovered_feed} if recovered_feed else {})


def alert_watchdog_stall(age_secs: float, threshold_secs: int):
    send_alert("WATCHDOG_STALL",
               f"Main loop stalled for {age_secs:.0f}s — bot may be hung",
               level="CRITICAL",
               details={"heartbeat_age_secs": f"{age_secs:.0f}",
                        "threshold_secs":     threshold_secs})


def alert_watchdog_idle(idle_cycles: int, threshold: int):
    send_alert("WATCHDOG_IDLE",
               f"Bot idle for {idle_cycles} consecutive cycles with no candidates",
               level="WARNING",
               details={"idle_cycles": idle_cycles, "threshold": threshold})


def alert_data_stale_halt(feed: str, age_secs: float, threshold_secs: int):
    send_alert("DATA_STALE_HALT",
               f"{feed} data is {age_secs:.0f}s old — trading halted",
               level="CRITICAL",
               details={"feed": feed, "age_secs": f"{age_secs:.0f}",
                        "threshold_secs": threshold_secs})


def alert_order_failure_streak(count: int, last_error: str):
    send_alert("ORDER_FAILURE_STREAK",
               f"{count} consecutive order placement failures",
               level="CRITICAL",
               details={"streak": count, "last_error": last_error[:200]})


def alert_integrity_mismatch(count: int, diff: str):
    send_alert("INTEGRITY_MISMATCH",
               f"State integrity check failed ({count} time(s))",
               level="CRITICAL" if count >= 3 else "WARNING",
               details={"diff": diff[:300]})


def alert_bot_restarted(mode: str, open_orders: int, open_positions: int,
                         db_positions: int, session_pnl: float, reconcile_clean: bool):
    send_alert("BOT_RESTARTED",
               f"Bot restarted — {mode} mode",
               level="WARNING",
               details={"open_orders":      open_orders,
                        "kalshi_positions": open_positions,
                        "db_positions":     db_positions,
                        "session_pnl":      f"${session_pnl/100:.2f}",
                        "reconcile_clean":  reconcile_clean})
