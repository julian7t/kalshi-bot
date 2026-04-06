"""
kalshi_client.py — Authenticated Kalshi API client.

Features:
  - RSA-PSS request signing (api.elections.kalshi.com)
  - Token-bucket rate limiter (5 req/s, burst 10)
  - Exponential backoff on 429 / 5xx / connection errors
  - Structured logging for every API error

Auth note:
  Sign only the URL path (not query string). Timestamp is milliseconds.
  Padding: PSS with salt_length=DIGEST_LENGTH.
"""

import base64
import datetime
import logging
import os
import threading
import time
from urllib.parse import urlparse

import requests
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from config import API_RATE_BURST, API_RATE_PER_SECOND, API_MAX_RETRIES

logger = logging.getLogger("kalshi_bot.client")

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"


# ── Rate limiter (token bucket) ───────────────────────────────────────────────

class _TokenBucket:
    def __init__(self, rate: float, capacity: float):
        self._rate     = rate
        self._capacity = capacity
        self._tokens   = capacity
        self._last     = time.monotonic()
        self._lock     = threading.Lock()

    def consume(self, tokens: float = 1.0) -> bool:
        with self._lock:
            now = time.monotonic()
            self._tokens = min(
                self._capacity,
                self._tokens + (now - self._last) * self._rate
            )
            self._last = now
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    def wait_and_consume(self, tokens: float = 1.0):
        while not self.consume(tokens):
            time.sleep(0.05)


_limiter = _TokenBucket(rate=API_RATE_PER_SECOND, capacity=API_RATE_BURST)


# ── Key loading ───────────────────────────────────────────────────────────────

def _load_private_key():
    raw = os.environ.get("KALSHI_PRIVATE_KEY", "")
    if not raw:
        raise ValueError("KALSHI_PRIVATE_KEY is not set")
    raw = raw.replace("\\n", "\n")

    # Try raw PEM first
    try:
        return serialization.load_pem_private_key(
            raw.encode(), password=None, backend=default_backend()
        )
    except Exception:
        pass

    # Strip headers and reformat
    clean = raw.strip()
    for h in [
        "-----BEGIN PRIVATE KEY-----", "-----BEGIN RSA PRIVATE KEY-----",
        "-----END PRIVATE KEY-----",   "-----END RSA PRIVATE KEY-----",
    ]:
        clean = clean.replace(h, "")
    clean = clean.replace("\n", "").replace(" ", "").strip()

    for header, footer in [
        ("-----BEGIN PRIVATE KEY-----",     "-----END PRIVATE KEY-----"),
        ("-----BEGIN RSA PRIVATE KEY-----", "-----END RSA PRIVATE KEY-----"),
    ]:
        pem = (header + "\n"
               + "\n".join(clean[i:i+64] for i in range(0, len(clean), 64))
               + "\n" + footer)
        try:
            return serialization.load_pem_private_key(
                pem.encode(), password=None, backend=default_backend()
            )
        except Exception:
            continue

    raise ValueError("Could not parse KALSHI_PRIVATE_KEY as a PEM RSA key")


def _auth_headers(method: str, path: str) -> dict:
    key_id = os.environ.get("KALSHI_API_KEY_ID", "")
    if not key_id:
        raise ValueError("KALSHI_API_KEY_ID is not set")
    ts_ms = str(int(datetime.datetime.now().timestamp() * 1000))
    sign_path = urlparse(BASE_URL + path).path
    message   = f"{ts_ms}{method.upper()}{sign_path}".encode()
    private_key = _load_private_key()
    sig = private_key.sign(
        message,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )
    return {
        "KALSHI-ACCESS-KEY":       key_id,
        "KALSHI-ACCESS-TIMESTAMP": ts_ms,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
        "Content-Type":            "application/json",
    }


# ── Core request dispatcher ───────────────────────────────────────────────────

def _request(method: str, path: str, params: dict = None, body: dict = None) -> dict:
    """
    Make an authenticated API call with rate limiting and exponential backoff.

    Retries on:
      - HTTP 429 (rate limited) — waits 2^attempt seconds
      - HTTP 5xx (server error) — waits 2^attempt seconds
      - Connection / timeout errors — waits 2^attempt seconds

    Raises on:
      - HTTP 4xx (client error, not 429)
      - Exhausted retries
    """
    url = BASE_URL + path

    for attempt in range(1, API_MAX_RETRIES + 1):
        _limiter.wait_and_consume()
        try:
            headers = _auth_headers(method, path)
            if method == "GET":
                resp = requests.get(url,  headers=headers, params=params, timeout=15)
            elif method == "POST":
                resp = requests.post(url, headers=headers, json=body or {}, timeout=15)
            elif method == "DELETE":
                resp = requests.delete(url, headers=headers, timeout=15)
            else:
                raise ValueError(f"Unsupported method: {method}")

            if resp.status_code == 429:
                wait = 2 ** attempt
                logger.warning("Rate limited (HTTP 429) on %s %s — waiting %ds", method, path, wait)
                time.sleep(wait)
                continue

            if resp.status_code >= 500:
                wait = 2 ** attempt
                logger.warning("Server error %d on %s %s — waiting %ds (attempt %d/%d)",
                               resp.status_code, method, path, wait, attempt, API_MAX_RETRIES)
                if attempt < API_MAX_RETRIES:
                    time.sleep(wait)
                    continue
                resp.raise_for_status()

            resp.raise_for_status()
            return resp.json()

        except (requests.ConnectionError, requests.Timeout) as e:
            wait = 2 ** attempt
            logger.warning("Request error on %s %s: %s — waiting %ds (attempt %d/%d)",
                           method, path, e, wait, attempt, API_MAX_RETRIES)
            if attempt == API_MAX_RETRIES:
                raise
            time.sleep(wait)

    raise RuntimeError(f"Exhausted {API_MAX_RETRIES} retries for {method} {path}")


def api_get(path: str, params: dict = None) -> dict:
    return _request("GET", path, params=params)

def api_post(path: str, body: dict = None) -> dict:
    return _request("POST", path, body=body)

def api_delete(path: str) -> dict:
    return _request("DELETE", path)


# ── Account ───────────────────────────────────────────────────────────────────

def get_balance() -> dict:
    return api_get("/portfolio/balance")


# ── Markets ───────────────────────────────────────────────────────────────────

def get_markets(limit: int = 100, cursor: str = None, series_ticker: str = None) -> dict:
    params = {"limit": limit}
    if cursor:
        params["cursor"] = cursor
    if series_ticker:
        params["series_ticker"] = series_ticker
    return api_get("/markets", params=params)


def get_all_open_markets(max_markets: int = 200) -> list:
    markets, cursor = [], None
    while len(markets) < max_markets:
        batch = min(100, max_markets - len(markets))
        data  = get_markets(limit=batch, cursor=cursor)
        page  = data.get("markets", [])
        markets.extend(page)
        cursor = data.get("cursor")
        if not cursor or not page:
            break
    return markets


def get_sports_markets(series_list: list[str], max_per_series: int = 50) -> list:
    """
    Fetch open markets across a list of sports series tickers.
    Returns a deduplicated list of all markets found.

    Uses series_ticker filter so we only pull sports markets — not political,
    financial, or weather markets that dominate the default /markets sort order.
    """
    import logging
    log = logging.getLogger("kalshi_bot.client")

    seen:    set  = set()
    results: list = []

    for series in series_list:
        try:
            cursor = None
            fetched = 0
            while fetched < max_per_series:
                batch = min(100, max_per_series - fetched)
                data  = get_markets(limit=batch, cursor=cursor, series_ticker=series)
                page  = data.get("markets", [])
                for m in page:
                    t = m.get("ticker")
                    if t and t not in seen:
                        seen.add(t)
                        results.append(m)
                fetched += len(page)
                cursor   = data.get("cursor")
                if not cursor or not page:
                    break
            log.debug("[CLIENT] series=%-24s → %d markets", series, fetched)
        except Exception as exc:
            log.warning("[CLIENT] Failed to fetch series %s: %s", series, exc)

    return results


def get_market(ticker: str) -> dict:
    return api_get(f"/markets/{ticker}")


def get_settled_result(ticker: str) -> str | None:
    """Return 'yes', 'no', or None if the market hasn't settled yet."""
    try:
        data   = get_market(ticker)
        market = data.get("market", data)
        result = market.get("result", "")
        status = market.get("status", "")
        if result in ("yes", "no"):
            return result
        if status in ("settled", "finalized", "closed") and result:
            return result.lower() if result.lower() in ("yes", "no") else None
    except Exception:
        pass
    return None


# ── Portfolio / positions ─────────────────────────────────────────────────────

def get_positions() -> dict:
    return api_get("/portfolio/positions")


def get_live_positions() -> dict:
    """Return {ticker: position_obj} for all held positions."""
    data = get_positions()
    return {
        pos["ticker"]: pos
        for pos in data.get("market_positions", [])
        if pos.get("ticker")
    }


# ── Orders ────────────────────────────────────────────────────────────────────

def place_order(ticker: str, side: str, count: int,
                yes_price: int, client_order_id: str = None) -> dict:
    body = {
        "ticker":    ticker,
        "action":    "buy",
        "side":      side,
        "count":     count,
        "type":      "limit",
        "yes_price": yes_price,
    }
    if client_order_id:
        body["client_order_id"] = client_order_id
    return api_post("/portfolio/orders", body)


def get_orders(status: str = None, ticker: str = None, limit: int = 100) -> list:
    """Fetch orders filtered by status and/or ticker."""
    params: dict = {"limit": limit}
    if status:
        params["status"] = status
    if ticker:
        params["ticker"] = ticker
    data = api_get("/portfolio/orders", params=params)
    return data.get("orders", [])


def get_open_orders(ticker: str = None) -> list:
    return get_orders(status="resting", ticker=ticker)


def get_filled_orders(limit: int = 100) -> list:
    return get_orders(status="executed", limit=limit)


def get_order(order_id: str) -> dict:
    return api_get(f"/portfolio/orders/{order_id}")


def cancel_order(order_id: str) -> dict:
    return api_delete(f"/portfolio/orders/{order_id}")
