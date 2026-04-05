"""
parser.py — Deterministic market-line parser for Kalshi tickers and titles.

Extracts numeric line values needed by market-type models.
All parsing is deterministic and returns a reason string on failure.
No ML, no heuristics — only explicit pattern matching.

Public API:
    parse_total_line(ticker, title)    → dict with "line", "side", "parsed", "reason"
    parse_spread_line(ticker, title)   → dict with "line", "parsed", "reason"
    parse_period_ref(ticker, title)    → dict with "period", "period_type", "parsed"
    parse_prop_threshold(ticker, title)→ dict with "threshold", "stat", "parsed"
    parse_for_market_type(mtype, ticker, title) → unified call
"""

import logging
import re
from typing import Optional

logger = logging.getLogger("kalshi_bot.parser")

# ── Number helpers ─────────────────────────────────────────────────────────────
_NUM        = r"(\d{1,3}(?:\.\d+)?)"        # positive number, e.g. 220.5
_SIGNED_NUM = r"([+-]?\d{1,3}(?:\.\d+)?)"   # signed number, e.g. -5.5


# ── Total / over-under line ───────────────────────────────────────────────────

def parse_total_line(ticker: str, title: str) -> dict:
    """
    Extract the numeric over/under total line and the side ("over"/"under").

    Priority: title → ticker

    Returns:
        line   — float or None
        side   — "over" | "under" | "unknown"
        parsed — True if a number was found
        reason — explanation string
    """
    title_s  = (title  or "").strip()
    ticker_s = (ticker or "").strip().upper()

    # ── Title patterns ────────────────────────────────────────────────────────
    title_rules: list[tuple] = [
        # "exceed 220.5", "over 220.5", "above 220.5", "more than 220.5"
        (re.compile(r"(?:exceed|over|above|more than|at least)\s+" + _NUM, re.IGNORECASE), "over"),
        # "under 220.5", "below 220.5", "less than 220.5", "fewer than 220.5"
        (re.compile(r"(?:under|below|less than|fewer than|at most)\s+" + _NUM, re.IGNORECASE), "under"),
        # "220.5 or more"
        (re.compile(_NUM + r"\s+or\s+more", re.IGNORECASE), "over"),
        # "220.5 or fewer", "220.5 or less"
        (re.compile(_NUM + r"\s+or\s+(?:fewer|less)", re.IGNORECASE), "under"),
        # "total of 220.5", "total exceeds 220.5", "total over 220.5"
        (re.compile(r"total\s+(?:of\s+|is\s+|exceeds\s+|over\s+|above\s+)?" + _NUM, re.IGNORECASE), "over"),
        # "total under 220.5", "total below 220.5"
        (re.compile(r"total\s+(?:under|below|less than)\s+" + _NUM, re.IGNORECASE), "under"),
        # "combined score of 220.5 or more"
        (re.compile(r"combined\s+(?:score|total)\s+(?:of\s+)?" + _NUM + r"\s+or\s+more", re.IGNORECASE), "over"),
        (re.compile(r"combined\s+(?:score|total)\s+(?:of\s+)?" + _NUM + r"\s+or\s+(?:fewer|less)", re.IGNORECASE), "under"),
    ]
    for pattern, side in title_rules:
        m = pattern.search(title_s)
        if m:
            try:
                val = float(m.group(1))
                if 0.5 <= val <= 999:  # sanity check: realistic totals
                    logger.debug("[PARSER] total_line %.1f (%s) from title: %s",
                                 val, side, m.group(0)[:40])
                    return {"line": val, "side": side, "parsed": True,
                            "reason": f"title match: '{m.group(0)[:40]}'"}
            except (ValueError, IndexError):
                pass

    # ── Ticker patterns ───────────────────────────────────────────────────────
    # e.g. NBAOVER-220.5, MLBUNDER-8.5, NFLTOTAL-48, O220, U8.5
    ticker_rules: list[tuple] = [
        (re.compile(r"OVER-?" + _NUM),           "over"),
        (re.compile(r"OVR-?" + _NUM),            "over"),
        (re.compile(r"UNDER-?" + _NUM),          "under"),
        (re.compile(r"UND-?" + _NUM),            "under"),
        (re.compile(r"TOTAL-?" + _NUM),          "over"),
        # O220 / U8.5 (require 2+ digits before decimal to avoid collision with period refs)
        (re.compile(r"\bO(\d{2,3}(?:\.\d+)?)\b"), "over"),
        (re.compile(r"\bU(\d{2,3}(?:\.\d+)?)\b"), "under"),
    ]
    for pattern, side in ticker_rules:
        m = pattern.search(ticker_s)
        if m:
            try:
                val = float(m.group(1))
                if 0.5 <= val <= 999:
                    logger.debug("[PARSER] total_line %.1f (%s) from ticker: %s",
                                 val, side, m.group(0)[:20])
                    return {"line": val, "side": side, "parsed": True,
                            "reason": f"ticker match: '{m.group(0)[:20]}'"}
            except (ValueError, IndexError):
                pass

    return {
        "line": None, "side": "unknown", "parsed": False,
        "reason": "no total line found in ticker or title",
    }


# ── Spread / handicap line ────────────────────────────────────────────────────

def parse_spread_line(ticker: str, title: str) -> dict:
    """
    Extract the spread/handicap numeric value.

    Convention: negative spread = home team favored (must win by more).
    Positive spread = home team gets points added (underdog).

    Returns:
        line   — float or None (negative = home must cover)
        parsed — True if a number was found
        reason — explanation string
    """
    title_s  = (title  or "").strip()
    ticker_s = (ticker or "").strip().upper()

    # ── Title patterns ────────────────────────────────────────────────────────
    title_rules = [
        re.compile(r"cover\s+" + _SIGNED_NUM, re.IGNORECASE),
        re.compile(r"spread\s+(?:of\s+)?" + _SIGNED_NUM, re.IGNORECASE),
        re.compile(r"handicap\s+(?:of\s+)?" + _SIGNED_NUM, re.IGNORECASE),
        # "([+-]N point" or "[+-]N.5 point"
        re.compile(r"([+-]\d{1,2}(?:\.\d+)?)\s*(?:point|run|goal)s?", re.IGNORECASE),
        # "win by 7 or more"
        re.compile(r"win\s+by\s+" + _NUM, re.IGNORECASE),
    ]
    for pattern in title_rules:
        m = pattern.search(title_s)
        if m:
            try:
                val = float(m.group(1))
                logger.debug("[PARSER] spread_line %+.1f from title: %s", val, m.group(0)[:40])
                return {"line": val, "parsed": True,
                        "reason": f"title match: '{m.group(0)[:40]}'"}
            except (ValueError, IndexError):
                pass

    # ── Ticker patterns ───────────────────────────────────────────────────────
    ticker_rules = [
        re.compile(r"SPREAD-?" + _SIGNED_NUM),
        re.compile(r"HCAP-?" + _SIGNED_NUM),
        re.compile(r"HDCP-?" + _SIGNED_NUM),
        re.compile(r"ATS-?" + _SIGNED_NUM),
        re.compile(r"COVER-?" + _SIGNED_NUM),
    ]
    for pattern in ticker_rules:
        m = pattern.search(ticker_s)
        if m:
            try:
                val = float(m.group(1))
                logger.debug("[PARSER] spread_line %+.1f from ticker: %s", val, m.group(0)[:20])
                return {"line": val, "parsed": True,
                        "reason": f"ticker match: '{m.group(0)[:20]}'"}
            except (ValueError, IndexError):
                pass

    return {
        "line": None, "parsed": False,
        "reason": "no spread line found in ticker or title",
    }


# ── Period / quarter / half / inning / set reference ─────────────────────────

def parse_period_ref(ticker: str, title: str) -> dict:
    """
    Extract which segment (quarter/half/period/inning/set) the market is about.

    Returns:
        period      — int (1-based) or None
        period_type — "quarter" | "half" | "period" | "inning" | "set" | "unknown"
        parsed      — True if a reference was found
    """
    title_s  = (title  or "").strip()
    ticker_s = (ticker or "").strip().upper()

    # ── Quarter ───────────────────────────────────────────────────────────────
    m = re.search(r"\b([1-4])(?:st|nd|rd|th)?\s+quarter\b", title_s, re.IGNORECASE)
    if not m:
        m = re.search(r"\bQ([1-4])\b", title_s, re.IGNORECASE)
    if not m:
        m = re.search(r"\bQ([1-4])\b", ticker_s)
    if m:
        return {"period": int(m.group(1)), "period_type": "quarter", "parsed": True}

    # ── Half ──────────────────────────────────────────────────────────────────
    m = re.search(r"\b(1st|first|2nd|second)\s+half\b", title_s, re.IGNORECASE)
    if not m:
        m = re.search(r"\b(1|2)H\b", ticker_s)
    if m:
        half_num = 1 if m.group(1).upper() in ("1ST", "FIRST", "1") else 2
        return {"period": half_num, "period_type": "half", "parsed": True}

    # ── Period (NHL / soccer) ─────────────────────────────────────────────────
    m = re.search(r"\b([1-3])(?:st|nd|rd|th)?\s+period\b", title_s, re.IGNORECASE)
    if not m:
        m = re.search(r"\bPER?([1-3])\b", ticker_s)
    if not m:
        m = re.search(r"\bP([1-3])\b", ticker_s)
    if m:
        return {"period": int(m.group(1)), "period_type": "period", "parsed": True}

    # ── Inning ────────────────────────────────────────────────────────────────
    m = re.search(r"\b([1-9])(?:st|nd|rd|th)?\s+inning\b", title_s, re.IGNORECASE)
    if not m:
        m = re.search(r"\bINN?([1-9])\b", ticker_s)
    if m:
        return {"period": int(m.group(1)), "period_type": "inning", "parsed": True}

    # ── Set (tennis) ──────────────────────────────────────────────────────────
    m = re.search(r"\b([1-5])(?:st|nd|rd|th)?\s+set\b", title_s, re.IGNORECASE)
    if not m:
        m = re.search(r"\bSET([1-5])\b", ticker_s)
    if m:
        return {"period": int(m.group(1)), "period_type": "set", "parsed": True}

    return {"period": None, "period_type": "unknown", "parsed": False}


# ── Player prop threshold ─────────────────────────────────────────────────────

def parse_prop_threshold(ticker: str, title: str) -> dict:
    """
    Extract player prop threshold and stat type from the title.

    Returns:
        threshold — float or None
        stat      — normalized stat name or "unknown"
        parsed    — True if a threshold was found
        reason    — explanation string
    """
    title_s = (title or "").strip()

    # Stat keyword → canonical name
    stat_map = [
        (re.compile(r"points?\b", re.IGNORECASE), "points"),
        (re.compile(r"rebounds?\b", re.IGNORECASE), "rebounds"),
        (re.compile(r"assists?\b", re.IGNORECASE), "assists"),
        (re.compile(r"goals?\b", re.IGNORECASE), "goals"),
        (re.compile(r"strikeouts?\b", re.IGNORECASE), "strikeouts"),
        (re.compile(r"home\s+runs?\b", re.IGNORECASE), "home_runs"),
        (re.compile(r"rushing\s+yards?\b", re.IGNORECASE), "rushing_yards"),
        (re.compile(r"receiving\s+yards?\b", re.IGNORECASE), "receiving_yards"),
        (re.compile(r"passing\s+yards?\b", re.IGNORECASE), "passing_yards"),
        (re.compile(r"touchdowns?\b", re.IGNORECASE), "touchdowns"),
        (re.compile(r"sacks?\b", re.IGNORECASE), "sacks"),
        (re.compile(r"three[- ]pointers?\b", re.IGNORECASE), "three_pointers"),
        (re.compile(r"steals?\b", re.IGNORECASE), "steals"),
        (re.compile(r"blocks?\b", re.IGNORECASE), "blocks"),
    ]

    for stat_re, stat_name in stat_map:
        if stat_re.search(title_s):
            # Find a number near this stat keyword
            num_m = re.search(
                r"(?:over|more than|at least|exceed[s]?|above)?\s*"
                + _NUM + r"\s*" + stat_re.pattern,
                title_s, re.IGNORECASE,
            )
            if not num_m:
                # Try the reverse: "records N+ assists"
                num_m = re.search(
                    stat_re.pattern + r"\s*(?:of|:)?\s*(?:over|more than|at least)?\s*" + _NUM,
                    title_s, re.IGNORECASE,
                )
            if num_m:
                try:
                    val = float(num_m.group(1))
                    logger.debug("[PARSER] prop threshold=%.1f stat=%s from title", val, stat_name)
                    return {
                        "threshold": val, "stat": stat_name, "parsed": True,
                        "reason": f"title: {num_m.group(0)[:40]}",
                    }
                except (ValueError, IndexError):
                    pass

    return {
        "threshold": None, "stat": "unknown", "parsed": False,
        "reason": "no prop threshold found in title",
    }


# ── Unified parser ────────────────────────────────────────────────────────────

def parse_for_market_type(market_type: str, ticker: str, title: str) -> dict:
    """
    Return the most relevant parse result for the given market_type.

    Callers inspect:
      - "line"      for totals and spread
      - "period"    for segment winners
      - "threshold" for player props
    """
    if market_type == "totals":
        return parse_total_line(ticker, title)
    if market_type == "spread":
        return parse_spread_line(ticker, title)
    if market_type in ("quarter_winner", "half_winner", "period_winner",
                       "inning_winner", "set_winner"):
        return parse_period_ref(ticker, title)
    if market_type == "player_prop":
        return parse_prop_threshold(ticker, title)
    # game_winner, misc — no line to parse
    return {"parsed": True, "reason": f"no line needed for market_type={market_type}"}
