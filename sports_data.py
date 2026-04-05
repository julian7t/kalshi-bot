"""
sports_data.py — Live sports data adapters.

Primary adapter: ESPN public scoreboard API (no API key required).
Fallback:        MockSportsAdapter (deterministic fake games for testing).

Usage:
    adapter = get_adapter()
    games = adapter.fetch_games()   # -> list[dict]  (raw normalised game dicts)

Each raw game dict has these guaranteed keys (None if unavailable):
    event_id        str   unique event identifier
    league          str   e.g. "NBA", "NFL", "MLB", "NHL", "NCAAF", "NCAAB"
    home_team       str   full team name
    away_team       str   full team name
    home_abbr       str   short abbreviation
    away_abbr       str   short abbreviation
    home_score      int   current score / runs / goals
    away_score      int
    game_clock      str   display clock ("4:32", "End of 3rd", etc.)
    period          int   current quarter / period / inning (0 if pregame)
    total_periods   int   4 for NFL/NBA, 3 for NHL, 9 for MLB, etc.
    possession      str | None  "home" | "away" | None
    status          str   "scheduled" | "in_progress" | "final" | "halftime" | "unknown"
    start_time      str   ISO-8601 UTC
    last_updated    float unix timestamp
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Optional

import requests

logger = logging.getLogger("kalshi_bot.sports_data")

# ── ESPN league configs ────────────────────────────────────────────────────────

_ESPN_ENDPOINTS = {
    "NBA":   "basketball/nba",
    "NFL":   "football/nfl",
    "MLB":   "baseball/mlb",
    "NHL":   "hockey/nhl",
    "NCAAF": "football/college-football",
    "NCAAB": "basketball/mens-college-basketball",
    "MLS":   "soccer/usa.1",
}

_TOTAL_PERIODS = {
    "NBA": 4, "NCAAB": 2,
    "NFL": 4, "NCAAF": 4,
    "MLB": 9,
    "NHL": 3,
    "MLS": 2,
}

_ESPN_STATUS_MAP = {
    "STATUS_SCHEDULED":    "scheduled",
    "STATUS_IN_PROGRESS":  "in_progress",
    "STATUS_FINAL":        "final",
    "STATUS_HALFTIME":     "halftime",
    "STATUS_END_PERIOD":   "in_progress",
    "STATUS_DELAYED":      "scheduled",
    "STATUS_POSTPONED":    "scheduled",
    "STATUS_CANCELED":     "final",
    "STATUS_SUSPENDED":    "in_progress",
}

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports"
REQUEST_TIMEOUT = 8  # seconds


# ── Abstract base ─────────────────────────────────────────────────────────────

class SportsDataAdapter(ABC):
    """All adapters must implement fetch_games()."""

    @abstractmethod
    def fetch_games(self) -> list[dict]:
        """Return a list of normalised raw game dicts."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


# ── ESPN adapter ──────────────────────────────────────────────────────────────

class ESPNSportsAdapter(SportsDataAdapter):
    """
    Pulls live scores from ESPN's public (no-auth) scoreboard API.
    Fetches all configured leagues in sequence; failures on individual
    leagues are logged but do not abort the others.
    """

    name = "ESPN"

    def __init__(self, leagues: list[str] | None = None):
        self._leagues = leagues or list(_ESPN_ENDPOINTS.keys())

    def fetch_games(self) -> list[dict]:
        games = []
        for league in self._leagues:
            path = _ESPN_ENDPOINTS.get(league)
            if not path:
                logger.warning("[ESPN] Unknown league: %s", league)
                continue
            try:
                url  = f"{ESPN_BASE}/{path}/scoreboard"
                resp = requests.get(url, timeout=REQUEST_TIMEOUT)
                resp.raise_for_status()
                data = resp.json()
                parsed = _parse_espn_scoreboard(data, league)
                games.extend(parsed)
                logger.debug("[ESPN] %s: %d events", league, len(parsed))
            except requests.exceptions.RequestException as e:
                logger.warning("[ESPN] Fetch failed for %s: %s", league, e)
            except Exception as e:
                logger.warning("[ESPN] Parse error for %s: %s", league, e)
        logger.info("[ESPN] Fetched %d total live/recent games across %d leagues",
                    len(games), len(self._leagues))
        return games


def _parse_espn_scoreboard(data: dict, league: str) -> list[dict]:
    """Convert ESPN scoreboard JSON into normalised game dicts."""
    events = data.get("events") or []
    results = []
    now = time.time()
    total_periods = _TOTAL_PERIODS.get(league, 4)

    for ev in events:
        try:
            ev_id  = str(ev.get("id", ""))
            status_block = ev.get("status", {})
            status_type  = status_block.get("type", {})
            status_name  = status_type.get("name", "STATUS_SCHEDULED")
            status       = _ESPN_STATUS_MAP.get(status_name, "unknown")
            game_clock   = status_block.get("displayClock", "")
            period       = int(status_block.get("period") or 0)

            comps = ev.get("competitions") or []
            if not comps:
                continue
            comp = comps[0]
            competitors = comp.get("competitors") or []

            home = _find_team(competitors, "home")
            away = _find_team(competitors, "away")
            if not home or not away:
                continue

            home_score = _safe_int(home.get("score"))
            away_score = _safe_int(away.get("score"))

            # Possession (football / some other sports)
            situation = comp.get("situation") or {}
            possession = _parse_possession(situation, competitors)

            results.append({
                "event_id":      f"{league}_{ev_id}",
                "league":        league,
                "home_team":     home["team"].get("displayName", ""),
                "away_team":     away["team"].get("displayName", ""),
                "home_abbr":     home["team"].get("abbreviation", ""),
                "away_abbr":     away["team"].get("abbreviation", ""),
                "home_score":    home_score,
                "away_score":    away_score,
                "game_clock":    game_clock,
                "period":        period,
                "total_periods": total_periods,
                "possession":    possession,
                "status":        status,
                "start_time":    ev.get("date", ""),
                "last_updated":  now,
            })
        except Exception as e:
            logger.debug("[ESPN] Skipped event parse error: %s", e)

    return results


def _find_team(competitors: list, side: str) -> dict | None:
    for c in competitors:
        if c.get("homeAway") == side:
            return c
    return None


def _safe_int(val) -> int:
    try:
        return int(val or 0)
    except (ValueError, TypeError):
        return 0


def _parse_possession(situation: dict, competitors: list) -> Optional[str]:
    """Return 'home' or 'away' from ESPN situation block, or None."""
    poss_id = str(situation.get("possession") or situation.get("possessionText") or "")
    if not poss_id:
        return None
    for c in competitors:
        team = c.get("team", {})
        if str(team.get("id", "")) == poss_id:
            return c.get("homeAway")
    return None


# ── Mock adapter ──────────────────────────────────────────────────────────────

class MockSportsAdapter(SportsDataAdapter):
    """
    Deterministic fake game generator for testing when no API is available.
    Creates a set of plausible in-progress games across major leagues.
    Scores advance slightly each call (based on real clock) so the data
    looks live.
    """

    name = "MOCK"

    _TEMPLATES = [
        # (league, home_team, home_abbr, away_team, away_abbr, total_periods)
        ("NBA",   "Los Angeles Lakers",      "LAL", "Golden State Warriors",   "GSW", 4),
        ("NBA",   "Boston Celtics",          "BOS", "Miami Heat",              "MIA", 4),
        ("NFL",   "Kansas City Chiefs",      "KC",  "San Francisco 49ers",     "SF",  4),
        ("NFL",   "Dallas Cowboys",          "DAL", "Philadelphia Eagles",     "PHI", 4),
        ("MLB",   "New York Yankees",        "NYY", "Boston Red Sox",          "BOS", 9),
        ("MLB",   "Los Angeles Dodgers",     "LAD", "San Francisco Giants",    "SF",  9),
        ("NHL",   "Toronto Maple Leafs",     "TOR", "Montreal Canadiens",      "MTL", 3),
        ("NCAAB", "Duke Blue Devils",        "DUKE","North Carolina Tar Heels","UNC", 2),
        ("NCAAF", "Alabama Crimson Tide",    "ALA", "Georgia Bulldogs",        "UGA", 4),
    ]

    def fetch_games(self) -> list[dict]:
        now    = time.time()
        minute = int(now / 60) % 48   # cycles 0–47 → simulate game clock
        games  = []

        for i, (league, ht, ha, at, aa, tp) in enumerate(self._TEMPLATES):
            seed    = (i * 7 + int(now / 300)) % 100
            h_score = (seed + minute // 3) % (12 * tp)
            a_score = ((seed + 5) + minute // 4) % (12 * tp)
            period  = min(tp, 1 + minute // (48 // tp))
            secs_in_period = (minute % (48 // max(tp, 1))) * 60
            clock_mins = max(0, (48 // tp) - secs_in_period // 60)
            clock_secs = (60 - secs_in_period % 60) % 60

            games.append({
                "event_id":      f"{league}_MOCK_{i:02d}",
                "league":        league,
                "home_team":     ht,
                "away_team":     at,
                "home_abbr":     ha,
                "away_abbr":     aa,
                "home_score":    h_score,
                "away_score":    a_score,
                "game_clock":    f"{clock_mins}:{clock_secs:02d}",
                "period":        period,
                "total_periods": tp,
                "possession":    "home" if seed % 2 == 0 else "away",
                "status":        "in_progress",
                "start_time":    "",
                "last_updated":  now,
            })

        logger.debug("[MOCK] Generated %d fake games", len(games))
        return games


# ── Factory ───────────────────────────────────────────────────────────────────

def get_adapter(force_mock: bool = False) -> SportsDataAdapter:
    """
    Return the best available adapter.
    Falls back to mock if ESPN is unreachable or force_mock=True.
    """
    if force_mock:
        logger.info("[SPORTS] Using MOCK adapter")
        return MockSportsAdapter()

    # Quick connectivity probe
    try:
        r = requests.get(
            f"{ESPN_BASE}/basketball/nba/scoreboard",
            timeout=5,
        )
        r.raise_for_status()
        logger.info("[SPORTS] ESPN adapter available")
        return ESPNSportsAdapter()
    except Exception as e:
        logger.warning("[SPORTS] ESPN probe failed (%s) — falling back to MOCK", e)
        return MockSportsAdapter()
