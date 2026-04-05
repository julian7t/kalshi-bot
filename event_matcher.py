"""
event_matcher.py — Match Kalshi market tickers/titles to live sports events.

Matching strategy:
  1. Normalise the Kalshi title — strip stop words, expand abbreviations
  2. For each active game in the store, check if both team tokens appear
     in the normalised title
  3. Return the best match (highest token overlap) above a score threshold

Unmatched markets are never traded; the reason is always logged.

Usage:
    from event_matcher import EventMatcher
    from game_state import GameStateStore

    matcher = EventMatcher()
    gs, reason = matcher.match(market, store)
    if gs is None:
        logger.info("Unmatched: %s — %s", ticker, reason)
"""

import logging
import re
from typing import Optional

from game_state import GameState, GameStateStore

logger = logging.getLogger("kalshi_bot.event_matcher")

# ── Normalisation maps ────────────────────────────────────────────────────────

# Common abbreviations → canonical tokens used in team names
_ABBREV_EXPANSION = {
    # NFL
    "kc": "kansas city", "chiefs": "kansas city chiefs",
    "ne": "new england", "patriots": "new england patriots",
    "sf": "san francisco", "niners": "san francisco 49ers",
    "gb": "green bay", "packers": "green bay packers",
    "dal": "dallas", "cowboys": "dallas cowboys",
    "phi": "philadelphia", "eagles": "philadelphia eagles",
    "buf": "buffalo", "bills": "buffalo bills",
    "lv": "las vegas", "raiders": "las vegas raiders",
    "pit": "pittsburgh", "steelers": "pittsburgh steelers",
    "sea": "seattle", "seahawks": "seattle seahawks",
    "lac": "los angeles chargers", "chargers": "los angeles chargers",
    "lar": "los angeles rams", "rams": "los angeles rams",
    "mia": "miami", "dolphins": "miami dolphins",
    "den": "denver", "broncos": "denver broncos",
    "min": "minnesota", "vikings": "minnesota vikings",
    "det": "detroit", "lions": "detroit lions",
    "chi": "chicago", "bears": "chicago bears",
    "atl": "atlanta", "falcons": "atlanta falcons",
    "no": "new orleans", "saints": "new orleans saints",
    "tb": "tampa bay", "buccaneers": "tampa bay buccaneers",
    "car": "carolina", "panthers": "carolina panthers",
    "bal": "baltimore", "ravens": "baltimore ravens",
    "cle": "cleveland", "browns": "cleveland browns",
    "cin": "cincinnati", "bengals": "cincinnati bengals",
    "ten": "tennessee", "titans": "tennessee titans",
    "ind": "indianapolis", "colts": "indianapolis colts",
    "jax": "jacksonville", "jaguars": "jacksonville jaguars",
    "hou": "houston", "texans": "houston texans",
    "ari": "arizona", "cardinals": "arizona cardinals",
    "nyg": "new york giants", "giants": "new york giants",
    "nyj": "new york jets", "jets": "new york jets",
    "wsh": "washington", "commanders": "washington commanders",

    # NBA
    "lal": "los angeles lakers", "lakers": "los angeles lakers",
    "gsw": "golden state warriors", "warriors": "golden state warriors",
    "bos": "boston celtics", "celtics": "boston celtics",
    "bkn": "brooklyn nets", "nets": "brooklyn nets",
    "nyk": "new york knicks", "knicks": "new york knicks",
    "phi76": "philadelphia 76ers", "sixers": "philadelphia 76ers",
    "mil": "milwaukee bucks", "bucks": "milwaukee bucks",
    "chi": "chicago bulls", "bulls": "chicago bulls",
    "tor": "toronto raptors", "raptors": "toronto raptors",
    "ind": "indiana pacers", "pacers": "indiana pacers",
    "cle": "cleveland cavaliers", "cavs": "cleveland cavaliers",
    "det": "detroit pistons", "pistons": "detroit pistons",
    "orl": "orlando magic", "magic": "orlando magic",
    "was": "washington wizards", "wizards": "washington wizards",
    "atl": "atlanta hawks", "hawks": "atlanta hawks",
    "mia": "miami heat", "heat": "miami heat",
    "cha": "charlotte hornets", "hornets": "charlotte hornets",
    "den": "denver nuggets", "nuggets": "denver nuggets",
    "min": "minnesota timberwolves", "wolves": "minnesota timberwolves",
    "okc": "oklahoma city thunder", "thunder": "oklahoma city thunder",
    "uta": "utah jazz", "jazz": "utah jazz",
    "por": "portland trail blazers", "blazers": "portland trail blazers",
    "sac": "sacramento kings", "kings": "sacramento kings",
    "lac": "los angeles clippers", "clippers": "los angeles clippers",
    "phx": "phoenix suns", "suns": "phoenix suns",
    "dal": "dallas mavericks", "mavs": "dallas mavericks",
    "hou": "houston rockets", "rockets": "houston rockets",
    "mem": "memphis grizzlies", "grizzlies": "memphis grizzlies",
    "nop": "new orleans pelicans", "pelicans": "new orleans pelicans",
    "san": "san antonio spurs", "spurs": "san antonio spurs",

    # MLB
    "nyy": "new york yankees", "yankees": "new york yankees",
    "bos": "boston red sox", "sox": "boston red sox",
    "lad": "los angeles dodgers", "dodgers": "los angeles dodgers",
    "sf": "san francisco giants",
    "hou": "houston astros", "astros": "houston astros",
    "atl": "atlanta braves", "braves": "atlanta braves",
    "stl": "st louis cardinals",
    "chc": "chicago cubs", "cubs": "chicago cubs",
    "cws": "chicago white sox",
    "tb": "tampa bay rays", "rays": "tampa bay rays",
    "tor": "toronto blue jays", "blue jays": "toronto blue jays",
    "min": "minnesota twins", "twins": "minnesota twins",
    "oak": "oakland athletics", "athletics": "oakland athletics",
    "sea": "seattle mariners", "mariners": "seattle mariners",
    "sd": "san diego padres", "padres": "san diego padres",
    "col": "colorado rockies", "rockies": "colorado rockies",
    "ari": "arizona diamondbacks", "dbacks": "arizona diamondbacks",
    "phi": "philadelphia phillies", "phillies": "philadelphia phillies",
    "nym": "new york mets", "mets": "new york mets",
    "wsh": "washington nationals", "nats": "washington nationals",
    "mia": "miami marlins", "marlins": "miami marlins",
    "mil": "milwaukee brewers", "brewers": "milwaukee brewers",
    "cin": "cincinnati reds", "reds": "cincinnati reds",
    "pit": "pittsburgh pirates", "pirates": "pittsburgh pirates",
    "cle": "cleveland guardians", "guardians": "cleveland guardians",
    "det": "detroit tigers", "tigers": "detroit tigers",
    "kc": "kansas city royals", "royals": "kansas city royals",
    "bal": "baltimore orioles", "orioles": "baltimore orioles",
    "tex": "texas rangers", "rangers": "texas rangers",
    "laa": "los angeles angels", "angels": "los angeles angels",

    # NHL
    "tor": "toronto maple leafs", "leafs": "toronto maple leafs",
    "mtl": "montreal canadiens", "habs": "montreal canadiens",
    "bos": "boston bruins", "bruins": "boston bruins",
    "buf": "buffalo sabres", "sabres": "buffalo sabres",
    "det": "detroit red wings", "wings": "detroit red wings",
    "tbl": "tampa bay lightning", "lightning": "tampa bay lightning",
    "fla": "florida panthers",
    "ott": "ottawa senators", "senators": "ottawa senators",
    "car": "carolina hurricanes", "canes": "carolina hurricanes",
    "col": "colorado avalanche", "avs": "colorado avalanche",
    "edm": "edmonton oilers", "oilers": "edmonton oilers",
    "cgy": "calgary flames", "flames": "calgary flames",
    "van": "vancouver canucks", "canucks": "vancouver canucks",
    "wpg": "winnipeg jets",
    "min": "minnesota wild", "wild": "minnesota wild",
    "chi": "chicago blackhawks", "hawks": "chicago blackhawks",
    "stl": "st louis blues", "blues": "st louis blues",
    "nsh": "nashville predators", "preds": "nashville predators",
    "dal": "dallas stars", "stars": "dallas stars",
    "lak": "los angeles kings",
    "ana": "anaheim ducks", "ducks": "anaheim ducks",
    "sjs": "san jose sharks", "sharks": "san jose sharks",
    "vgk": "vegas golden knights", "knights": "vegas golden knights",
    "sea": "seattle kraken", "kraken": "seattle kraken",
    "ari": "arizona coyotes", "coyotes": "arizona coyotes",
    "cbj": "columbus blue jackets",
    "nyr": "new york rangers",
    "nyi": "new york islanders", "islanders": "new york islanders",
    "njd": "new jersey devils", "devils": "new jersey devils",
    "phi": "philadelphia flyers", "flyers": "philadelphia flyers",
    "pit": "pittsburgh penguins", "penguins": "pittsburgh penguins",
    "wsh": "washington capitals", "caps": "washington capitals",

    # College football
    "ala": "alabama crimson tide", "bama": "alabama",
    "uga": "georgia bulldogs",
    "unc": "north carolina tar heels",
    "ohio st": "ohio state buckeyes",
    "tcu": "tcu horned frogs",
    "okla": "oklahoma sooners",
}

# Words to strip from market titles before matching
_STOP_WORDS = {
    "will", "the", "win", "beat", "lose", "to", "vs", "at", "in",
    "by", "over", "tonight", "today", "game", "match", "nfl", "nba",
    "mlb", "nhl", "ncaaf", "ncaab", "moneyline", "spread", "ml",
    "playoff", "playoffs", "series", "against", "who", "which", "team",
    "cover", "super", "bowl", "championship", "finals", "final",
    "series", "wildcard", "wild", "card", "round",
}

_MIN_MATCH_SCORE = 1  # at least 1 team token must match


# ── Matcher ───────────────────────────────────────────────────────────────────

class EventMatcher:
    """
    Matches a Kalshi market dict to a GameState from the store.
    """

    def match(
        self,
        market: dict,
        store: GameStateStore,
    ) -> tuple[Optional[GameState], str]:
        """
        Try to match a market to a live game.

        Returns:
            (GameState, "")              on success
            (None,      rejection_reason) on failure
        """
        ticker = market.get("ticker", "")
        title  = market.get("title", ticker)

        tokens = _tokenize(title)
        if len(tokens) < 2:
            return None, f"title too short to match: '{title[:60]}'"

        candidates = store.get_active()
        if not candidates:
            return None, "no active games in state store"

        best_gs    = None
        best_score = 0

        for gs in candidates:
            score = _match_score(tokens, gs)
            if score > best_score:
                best_score = score
                best_gs    = gs

        if best_gs is None or best_score < _MIN_MATCH_SCORE:
            logger.debug("[MATCHER] No match for '%s' (tokens=%s)", ticker, list(tokens)[:6])
            return None, f"no game matched title tokens (best_score={best_score})"

        logger.debug("[MATCHER] %s → %s (score=%d)", ticker, best_gs.event_id, best_score)
        return best_gs, ""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tokenize(title: str) -> set[str]:
    """
    Lower-case, strip punctuation, expand abbreviations, remove stop words.
    Returns a set of normalised tokens.
    """
    text   = title.lower()
    text   = re.sub(r"[^a-z0-9 ]", " ", text)
    words  = text.split()

    tokens: set[str] = set()
    for w in words:
        if w in _STOP_WORDS:
            continue
        # expand abbreviation if known
        expanded = _ABBREV_EXPANSION.get(w)
        if expanded:
            for part in expanded.split():
                if part not in _STOP_WORDS:
                    tokens.add(part)
        tokens.add(w)

    return tokens


def _game_tokens(gs: GameState) -> set[str]:
    """Build the set of tokens we expect to find in a matching market title."""
    tokens: set[str] = set()
    for name in (gs.home_team, gs.away_team, gs.home_abbr, gs.away_abbr):
        if not name:
            continue
        parts = re.sub(r"[^a-z0-9 ]", " ", name.lower()).split()
        for p in parts:
            if p and p not in _STOP_WORDS:
                tokens.add(p)
    return tokens


def _match_score(market_tokens: set[str], gs: GameState) -> int:
    """
    Return overlap count between market tokens and game tokens.
    Higher is better; at least 1 required for a valid match.
    """
    game_tok = _game_tokens(gs)
    return len(market_tokens & game_tok)
