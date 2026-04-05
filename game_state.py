"""
game_state.py — Normalised internal game state store.

One GameState object per live event, keyed by event_id.
Updated on each polling cycle from raw sports adapter output.
Detects and rejects malformed data.  Staleness is exposed as a
boolean so callers can gate trading on freshness.

Usage:
    store = GameStateStore()
    store.update_from_raw(raw_games)          # call each cycle
    gs = store.get("NBA_MOCK_00")             # returns GameState or None
    if gs and not gs.is_stale(120):
        print(gs.score_differential)
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("kalshi_bot.game_state")

# How long (seconds) before we consider a game state stale
DEFAULT_STALE_SECONDS = 120


@dataclass
class GameState:
    """
    Normalised, validated snapshot of a single live game.
    All numeric fields are integers / floats.  String fields are
    stripped and lowercased where comparison is needed.
    """
    event_id:      str
    league:        str
    home_team:     str
    away_team:     str
    home_abbr:     str
    away_abbr:     str
    home_score:    int
    away_score:    int
    game_clock:    str          # display string, e.g. "4:32"
    period:        int          # current Q/period/inning (0 = pregame)
    total_periods: int          # 4 / 3 / 9 / 2 etc.
    possession:    Optional[str]  # "home" | "away" | None
    status:        str          # "scheduled" | "in_progress" | "halftime" | "final" | "unknown"
    start_time:    str          # ISO-8601 string
    last_updated:  float = field(default_factory=time.time)

    # ── Computed helpers ──────────────────────────────────────────────────────

    @property
    def score_differential(self) -> int:
        """Positive = home winning, negative = away winning."""
        return self.home_score - self.away_score

    @property
    def is_active(self) -> bool:
        return self.status in ("in_progress", "halftime")

    @property
    def is_final(self) -> bool:
        return self.status == "final"

    @property
    def period_fraction(self) -> float:
        """Fraction of total periods completed (0.0 – 1.0)."""
        if self.total_periods <= 0:
            return 0.0
        # current period minus one gives completed periods
        completed = max(0, self.period - 1)
        return min(1.0, completed / self.total_periods)

    @property
    def game_progress(self) -> float:
        """
        Rough fraction of the game elapsed (0.0 – 1.0).
        For in-progress games: uses period + clock when parseable.
        For final: 1.0.  For scheduled: 0.0.
        """
        if self.is_final:
            return 1.0
        if not self.is_active:
            return 0.0
        if self.total_periods <= 0:
            return 0.0

        period_length_min = _period_minutes(self.league, self.total_periods)
        clock_remaining   = _parse_clock_seconds(self.game_clock)

        if clock_remaining is None or period_length_min is None:
            # Fall back to period-level granularity
            return self.period_fraction

        total_seconds   = self.total_periods * period_length_min * 60
        elapsed_periods = max(0, self.period - 1) * period_length_min * 60
        elapsed_clock   = (period_length_min * 60) - clock_remaining
        elapsed_total   = elapsed_periods + elapsed_clock
        return min(1.0, max(0.0, elapsed_total / total_seconds))

    def is_stale(self, threshold_seconds: float = DEFAULT_STALE_SECONDS) -> bool:
        return (time.time() - self.last_updated) > threshold_seconds

    def summary(self) -> str:
        return (
            f"[{self.league}] {self.home_abbr} {self.home_score}–"
            f"{self.away_score} {self.away_abbr} | "
            f"P{self.period}/{self.total_periods} {self.game_clock} | "
            f"{self.status} | progress={self.game_progress:.0%}"
        )


# ── Store ─────────────────────────────────────────────────────────────────────

class GameStateStore:
    """
    Thread-safe-ish dictionary of GameState objects keyed by event_id.
    Safe for single-threaded use (our bot is single-threaded).
    """

    def __init__(self):
        self._states: dict[str, GameState] = {}

    def update_from_raw(self, raw_games: list[dict]) -> int:
        """
        Ingest a list of raw game dicts (from sports adapter).
        Validates each entry; silently skips malformed records.
        Returns number of states updated.
        """
        updated = 0
        for raw in raw_games:
            gs = _validate_and_build(raw)
            if gs is None:
                continue
            self._states[gs.event_id] = gs
            updated += 1

        # Prune events that haven't been seen in a long time (3 × stale threshold)
        cutoff = time.time() - (DEFAULT_STALE_SECONDS * 3)
        stale_keys = [k for k, v in self._states.items() if v.last_updated < cutoff]
        for k in stale_keys:
            del self._states[k]
            logger.debug("[GAME STATE] Pruned stale event: %s", k)

        logger.debug("[GAME STATE] Store has %d events after update (added/updated=%d)",
                     len(self._states), updated)
        return updated

    def get(self, event_id: str) -> Optional[GameState]:
        return self._states.get(event_id)

    def get_all(self) -> list[GameState]:
        return list(self._states.values())

    def get_active(self) -> list[GameState]:
        return [gs for gs in self._states.values() if gs.is_active]

    def __len__(self) -> int:
        return len(self._states)


# ── Validation ────────────────────────────────────────────────────────────────

_REQUIRED_KEYS = {
    "event_id", "league", "home_team", "away_team",
    "home_score", "away_score", "status",
}

_VALID_STATUSES = {"scheduled", "in_progress", "halftime", "final", "unknown"}


def _validate_and_build(raw: dict) -> Optional[GameState]:
    """
    Validate a raw dict and return a GameState, or None if invalid.
    """
    if not isinstance(raw, dict):
        logger.debug("[GAME STATE] Rejected non-dict: %r", raw)
        return None

    missing = _REQUIRED_KEYS - raw.keys()
    if missing:
        logger.debug("[GAME STATE] Rejected record missing keys %s: %r", missing, raw)
        return None

    try:
        event_id = str(raw["event_id"]).strip()
        league   = str(raw.get("league", "")).upper().strip()
        if not event_id or not league:
            return None

        home_score = int(raw.get("home_score") or 0)
        away_score = int(raw.get("away_score") or 0)
        period     = int(raw.get("period") or 0)
        total_per  = int(raw.get("total_periods") or 4)
        status     = str(raw.get("status", "unknown")).lower().strip()

        if status not in _VALID_STATUSES:
            status = "unknown"

        possession = raw.get("possession")
        if possession not in (None, "home", "away"):
            possession = None

        return GameState(
            event_id      = event_id,
            league        = league,
            home_team     = str(raw.get("home_team", "")).strip(),
            away_team     = str(raw.get("away_team", "")).strip(),
            home_abbr     = str(raw.get("home_abbr", "")).upper().strip(),
            away_abbr     = str(raw.get("away_abbr", "")).upper().strip(),
            home_score    = home_score,
            away_score    = away_score,
            game_clock    = str(raw.get("game_clock", "") or "").strip(),
            period        = max(0, period),
            total_periods = max(1, total_per),
            possession    = possession,
            status        = status,
            start_time    = str(raw.get("start_time", "") or "").strip(),
            last_updated  = float(raw.get("last_updated") or time.time()),
        )
    except Exception as e:
        logger.debug("[GAME STATE] Validation exception: %s  raw=%r", e, raw)
        return None


# ── Clock parsing helpers ─────────────────────────────────────────────────────

def _parse_clock_seconds(clock_str: str) -> Optional[float]:
    """
    Convert display clock ('4:32', '0:07.3') to total seconds remaining.
    Returns None if unparseable.
    """
    if not clock_str:
        return None
    try:
        parts = clock_str.split(":")
        if len(parts) == 2:
            mins = float(parts[0])
            secs = float(parts[1])
            return mins * 60 + secs
        if len(parts) == 1:
            return float(parts[0])
    except (ValueError, IndexError):
        pass
    return None


def _period_minutes(league: str, total_periods: int) -> Optional[float]:
    """Return standard period length in minutes for known leagues."""
    mapping = {
        "NBA": 12.0, "NCAAB": 20.0,
        "NFL": 15.0, "NCAAF": 15.0,
        "NHL": 20.0,
        "MLB": None,   # no clock in baseball
        "MLS": 45.0,
    }
    return mapping.get(league.upper())
