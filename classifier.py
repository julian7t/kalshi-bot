"""
classifier.py — Deterministic market-type classifier for Kalshi trade analytics.

Classifies each market into one of these canonical types based on ticker and
title string matching.  Logic is fully deterministic and explainable — every
decision records a reason string.

Market types:
  game_winner    — moneyline / full-game / match winner
  half_winner    — 1st or 2nd half winner
  quarter_winner — Q1 / Q2 / Q3 / Q4 winner
  period_winner  — P1 / P2 / P3 winner (NHL, soccer)
  inning_winner  — inning or first-N-innings markets (MLB)
  set_winner     — set or game winner (tennis)
  totals         — over / under points / runs / goals
  spread         — point spread / handicap / against-the-spread
  player_prop    — player-specific market (stats, milestones)
  misc           — anything that does not match the above

Classification priority:
  1. Title keyword matching   (confidence 0.75 – 0.95)
  2. Ticker pattern matching  (confidence 0.70 – 0.82)
  3. Sport-based fallback     (confidence 0.40 – 0.55)

Usage:
    from classifier import classify
    info = classify(ticker, title, sport)
    market_type = info["market_type"]
"""

import logging
import re

logger = logging.getLogger("kalshi_bot.classifier")

# ── Canonical market type labels ──────────────────────────────────────────────

MARKET_TYPES: list[str] = [
    "game_winner",
    "half_winner",
    "quarter_winner",
    "period_winner",
    "inning_winner",
    "set_winner",
    "totals",
    "spread",
    "player_prop",
    "misc",
]


# ── Title keyword rules ───────────────────────────────────────────────────────
# Each rule: (market_type, compiled_pattern, confidence, reason)
# Checked in ORDER — first match wins.  More specific rules come first.

_TITLE_RULES: list[tuple] = [

    # ── Player props — specific stat terms (check before totals) ─────────────
    ("player_prop", re.compile(
        r"\b(points|rebounds|assists|goals?|strikeouts?|home runs?|yards|"
        r"touchdowns?|field goals?|three.pointers?|steals?|blocks?|aces?|"
        r"double.doubles?|triple.doubles?|batting average|era\b|rushing|receiving|"
        r"passing yards|sacks?|interceptions?|saves?|shutouts?|"
        r"rushing yards|receiving yards|completions?|turnovers?|"
        r"free throws?|threes? made|points? scored|minutes? played)\b",
        re.IGNORECASE
    ), 0.92, "player stat keyword in title"),

    # ── Totals / over-under ────────────────────────────────────────────────────
    ("totals", re.compile(
        r"\b(over|under|o/u|combined score|combined total|"
        r"total runs|total goals|total points|total score|"
        r"more than \d|fewer than \d|at least \d|at most \d|"
        r"o\d{1,3}\.?\d*|u\d{1,3}\.?\d*)\b",
        re.IGNORECASE
    ), 0.92, "over/under keyword in title"),

    # ── Spread / handicap ─────────────────────────────────────────────────────
    ("spread", re.compile(
        r"\b(spread|handicap|cover the spread|against the spread|\bats\b|"
        r"cover by|[+-]\d+\.?\d* (?:point|run|goal)|"
        r"win by \d|lose by \d|margin of victory|by more than)\b",
        re.IGNORECASE
    ), 0.90, "spread / handicap keyword in title"),

    # ── Quarter winner ─────────────────────────────────────────────────────────
    ("quarter_winner", re.compile(
        r"\b(q1|q2|q3|q4|1st quarter|2nd quarter|3rd quarter|4th quarter|"
        r"quarter winner|win (?:the )?\d.. quarter|lead (?:at|after) (?:the )?\d.. quarter)\b",
        re.IGNORECASE
    ), 0.93, "quarter keyword in title"),

    # ── Half winner ────────────────────────────────────────────────────────────
    ("half_winner", re.compile(
        r"\b(1st half|2nd half|first half|second half|halftime|half winner|"
        r"win (?:the )?(?:first|second|1st|2nd) half|lead at half|"
        r"\b1h\b|\b2h\b|half.?time winner)\b",
        re.IGNORECASE
    ), 0.93, "half keyword in title"),

    # ── Period winner (NHL / soccer) ───────────────────────────────────────────
    ("period_winner", re.compile(
        r"\b(1st period|2nd period|3rd period|first period|second period|third period|"
        r"period winner|win (?:the )?\d.. period|lead (?:at|after) (?:the )?\d.. period|"
        r"\bp1\b|\bp2\b|\bp3\b)\b",
        re.IGNORECASE
    ), 0.91, "period keyword in title"),

    # ── Inning / first N innings (MLB) ─────────────────────────────────────────
    ("inning_winner", re.compile(
        r"\b(\d.. inning|first inning|inning winner|run line|"
        r"first [3-9] innings|first \d innings|f5\b|first five innings|"
        r"1st inning|score in the \d.. inning)\b",
        re.IGNORECASE
    ), 0.90, "inning keyword in title"),

    # ── Set winner (tennis) ────────────────────────────────────────────────────
    ("set_winner", re.compile(
        r"\b(\d.. set|first set|second set|third set|fourth set|fifth set|"
        r"set winner|win (?:the )?\d.. set|tiebreak|tie.break|"
        r"win in straight sets)\b",
        re.IGNORECASE
    ), 0.90, "set keyword in title"),

    # ── Game / match winner (broad — last before fallback) ─────────────────────
    ("game_winner", re.compile(
        r"\b(win the game|win the match|win the series|series winner|"
        r"outright winner|match winner|game winner|moneyline|"
        r"will .{2,40} win\b|championship winner|advance to|"
        r"reach the (?:final|semi|quarter)|make the playoffs)\b",
        re.IGNORECASE
    ), 0.82, "game winner keyword in title"),
]

# ── Ticker pattern rules ──────────────────────────────────────────────────────
# Kalshi tickers are upper-case alphanumeric with dashes.
# E.g.: KXNBAMONO-24-11-08-GSWS, NBAQS1-..., MLBOVER-..., NFLFG-...
# Checked in ORDER — first match wins.

_TICKER_RULES: list[tuple] = [
    ("totals",         re.compile(r"(OVER|UNDER|OVR|UND|TOTAL|OU\d|OVUN)", re.IGNORECASE),         0.82, "totals ticker pattern"),
    ("quarter_winner", re.compile(r"(Q[1-4][-_]|QTR|QUAR|1STQTR|2NDQTR|3RDQTR|4THQTR)", re.IGNORECASE), 0.82, "quarter ticker pattern"),
    ("half_winner",    re.compile(r"(HALF|[12]H-|HLF|HALFT|1STHALF|2NDHALF)", re.IGNORECASE),      0.80, "half ticker pattern"),
    ("period_winner",  re.compile(r"(PER[123]|P[123]-|PRD[123]|1STPER|2NDPER|3RDPER)", re.IGNORECASE), 0.80, "period ticker pattern"),
    ("inning_winner",  re.compile(r"(INN|INNING|F5-|FIRST5|1STINT|RUNLINE)", re.IGNORECASE),         0.80, "inning ticker pattern"),
    ("set_winner",     re.compile(r"(SET[123]|S[123]WIN|1STSET|2NDSET)", re.IGNORECASE),             0.78, "set ticker pattern"),
    ("spread",         re.compile(r"(SPREAD|HCAP|HDCP|ATS|COVER)", re.IGNORECASE),                   0.80, "spread ticker pattern"),
    ("player_prop",    re.compile(r"(PROP|PLR|PLAYER|STAT|PTS-|REB-|AST-|YDS-)", re.IGNORECASE),     0.80, "player prop ticker pattern"),
    ("game_winner",    re.compile(r"(WIN|WINNER|ML-|MONO|CHAMP|OUTRGHT|SERIES)", re.IGNORECASE),     0.72, "game winner ticker pattern"),
]

# ── Sport-based fallback market type ─────────────────────────────────────────
# When no title or ticker pattern matches, lean on sport context.
_SPORT_FALLBACK: dict[str, tuple[str, float]] = {
    "Basketball": ("game_winner", 0.52),
    "Football":   ("game_winner", 0.52),
    "Baseball":   ("game_winner", 0.50),
    "Hockey":     ("game_winner", 0.50),
    "Soccer":     ("game_winner", 0.48),
    "Tennis":     ("game_winner", 0.50),
}


# ── Public API ────────────────────────────────────────────────────────────────

def classify(ticker: str, title: str, sport: str = "Generic") -> dict:
    """
    Classify a Kalshi market into a canonical market_type.

    Args:
        ticker: Kalshi ticker string (e.g. "KXNBAMONO-24-11-08-GSWS")
        title:  Human-readable market title (e.g. "Will the Lakers win?")
        sport:  Sport label from model.py (e.g. "Basketball")

    Returns dict:
        market_type  — one of MARKET_TYPES
        confidence   — float 0.0–1.0
        reason       — plain-English explanation
        source       — "title" | "ticker" | "fallback"
    """
    title_s  = (title  or "").strip()
    ticker_s = (ticker or "").strip()
    sport_s  = (sport  or "Generic").strip()

    # 1. Title keyword matching (highest signal quality)
    for mtype, pattern, conf, reason in _TITLE_RULES:
        if pattern.search(title_s):
            if conf < 0.85:
                logger.debug(
                    "[CLASSIFIER] %s → %s (conf=%.0f%%, low-conf): %s | title=%s",
                    ticker_s, mtype, conf * 100, reason, title_s[:60],
                )
            return {
                "market_type": mtype,
                "confidence":  conf,
                "reason":      reason,
                "source":      "title",
            }

    # 2. Ticker pattern matching
    for mtype, pattern, conf, reason in _TICKER_RULES:
        if pattern.search(ticker_s.upper()):
            logger.debug(
                "[CLASSIFIER] %s → %s (conf=%.0f%%, ticker): %s",
                ticker_s, mtype, conf * 100, reason,
            )
            return {
                "market_type": mtype,
                "confidence":  conf,
                "reason":      reason,
                "source":      "ticker",
            }

    # 3. Sport-based fallback
    if sport_s in _SPORT_FALLBACK:
        mtype, conf = _SPORT_FALLBACK[sport_s]
        reason = f"no pattern matched; assuming {mtype} for {sport_s}"
    else:
        mtype, conf = "misc", 0.40
        reason = "no pattern matched; unknown market type"

    logger.debug(
        "[CLASSIFIER] %s → %s (conf=%.0f%%, fallback) | title=%s",
        ticker_s, mtype, conf * 100, title_s[:60],
    )
    return {
        "market_type": mtype,
        "confidence":  conf,
        "reason":      reason,
        "source":      "fallback",
    }


def classify_batch(markets: list[dict]) -> dict[str, dict]:
    """
    Classify a list of market dicts (each with 'ticker', 'title', 'sport').
    Returns {ticker: classification_dict}.
    """
    return {
        m["ticker"]: classify(
            m.get("ticker", ""),
            m.get("title",  ""),
            m.get("sport",  "Generic"),
        )
        for m in markets
    }
