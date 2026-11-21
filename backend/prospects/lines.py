"""Season lines and bios from the Stats API, cached under cache/prospects/.

Three sources share one MLBAM person id: the LMB (league 125), the LMP winter
league (league 132, labeled by the year it starts: season 2025 runs October
2025 to January 2026) and the US minor leagues (sports 11-16; the LMB's own
rows under sport 11 through 2019 are dropped there). A season file is fetched
once it is complete and never again; an open season is refetched on request."""

from datetime import date

import pandas as pd

from backend.config import CACHE_DIR
from backend.statsapi.client import get
from backend.statsapi.league import LEAGUE_ID, sport_id_for

PROSPECTS_DIR = CACHE_DIR / "prospects"
LINES_DIR = PROSPECTS_DIR / "lines"
PEOPLE_PATH = PROSPECTS_DIR / "people.parquet"
TEAM_GAMES_PATH = PROSPECTS_DIR / "team_games.parquet"

LMP_LEAGUE_ID = 132
LMP_SPORT_ID = 17
MILB_SPORT_IDS = (11, 12, 13, 14, 15, 16)
FIRST_LMB, FIRST_LMP = 2005, 2006
PAGE = 2000

STAT_FIELDS = {
    "plateAppearances": "pa",
    "atBats": "ab",
    "hits": "h",
    "doubles": "d2",
    "triples": "d3",
    "homeRuns": "hr",
    "baseOnBalls": "bb",
    "hitByPitch": "hbp",
    "strikeOuts": "so",
    "battersFaced": "bf",
    "outs": "outs",
    "gamesStarted": "gs",
    "gamesPlayed": "g",
}


def lmb_seasons(through: int) -> list[int]:
    """LMB seasons played: no 2020."""
    return [y for y in range(FIRST_LMB, through + 1) if y != 2020]


def _splits(response: dict) -> list[dict]:
    """A season with no games yet comes back with an empty stats list."""
    stats = response.get("stats") or []
    return stats[0].get("splits", []) if stats else []


def _stat_seasons(season: int) -> list[str]:
    """2018 is stored as two short seasons in the LMB."""
    return ["2018.1", "2018.2"] if season == 2018 else [str(season)]


def season_complete(source: str, season: int, today: date) -> bool:
    if source == "lmp":
        return today >= date(season + 1, 2, 15)
    return today >= date(season, 10, 1)


def _fetch(source: str, season: int, open_season: bool) -> pd.DataFrame:
    """One season's lines. An open season may have none yet (a league is added
    on April 1 or October 1 and opens about two weeks later); a complete
    season with none is an error."""
    if source == "lmb":
        queries = [(sport_id_for(season), LEAGUE_ID, s) for s in _stat_seasons(season)]
    elif source == "lmp":
        queries = [(LMP_SPORT_ID, LMP_LEAGUE_ID, str(season))]
    elif source == "milb":
        queries = [(sport, None, str(season)) for sport in MILB_SPORT_IDS]
    else:
        raise ValueError(f"Unknown source {source}")
    rows = []
    for sport, league, stat_season in queries:
        for group in ("hitting", "pitching"):
            offset = 0
            while True:
                params = dict(
                    stats="season",
                    group=group,
                    sportId=sport,
                    season=stat_season,
                    playerPool="ALL",
                    limit=PAGE,
                    offset=offset,
                )
                if league:
                    params["leagueId"] = league
                splits = _splits(get("/v1/stats", **params))
                for split in splits:
                    stat = split.get("stat", {})
                    row = {
                        "player": split["player"]["id"],
                        "name": split["player"].get("fullName"),
                        "team": split.get("team", {}).get("id"),
                        "league": split.get("league", {}).get("id"),
                        "sport": sport,
                        "group": group,
                    }
                    row.update({col: stat.get(key, 0) or 0 for key, col in STAT_FIELDS.items()})
                    rows.append(row)
                if len(splits) < PAGE:
                    break
                offset += PAGE
    frame = pd.DataFrame(rows)
    if frame.empty:
        if open_season:
            return frame
        raise RuntimeError(f"No {source} lines returned for {season}")
    if source == "milb":
        frame = frame[frame.league != LEAGUE_ID]
    frame["source"] = source
    frame["season"] = season
    return frame


def load_lines(seasons: dict[str, list[int]], today: date, refresh_open: bool = False) -> pd.DataFrame:
    """Season lines for {source: seasons}. Missing files are fetched; an open
    season's file is refetched only when `refresh_open` is set."""
    LINES_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for source, years in seasons.items():
        for season in years:
            path = LINES_DIR / f"{source}_{season}.parquet"
            open_season = not season_complete(source, season, today)
            if (refresh_open and open_season) or not path.exists():
                fetched = _fetch(source, season, open_season)
                if fetched.empty:  # not cached, so the next run asks again
                    continue
                fetched.to_parquet(path, index=False)
            frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True)


def _people(ids: list[int]) -> list[dict]:
    """Person records for `ids`; a batch the API refuses is retried one id at
    a time, and an id refused alone has no record."""
    try:
        return get("/v1/people", personIds=",".join(map(str, ids)))["people"]
    except RuntimeError:
        if len(ids) == 1:
            return []
        return [p for i in ids for p in _people([i])]


def load_people(player_ids) -> pd.DataFrame:
    """Birthdate, birth country, position, handedness and MLB debut, fetched
    once per player. A few line ids have no person record (the API answers
    404 when a whole batch is unknown); they are stored with no bio, so they
    are never asked for again and never become prospects (no age)."""
    people = pd.read_parquet(PEOPLE_PATH) if PEOPLE_PATH.exists() else pd.DataFrame(columns=["player"])
    missing = sorted(set(int(p) for p in player_ids) - set(people.player.astype(int)))
    rows = []
    for i in range(0, len(missing), 150):
        chunk = missing[i : i + 150]
        found = _people(chunk)
        rows += [{"player": p} for p in set(chunk) - {f["id"] for f in found}]
        for p in found:
            rows.append(
                {
                    "player": p["id"],
                    "full_name": p.get("fullName"),
                    "birth_date": p.get("birthDate"),
                    "birth_country": p.get("birthCountry"),
                    "position": p.get("primaryPosition", {}).get("abbreviation"),
                    "throws": p.get("pitchHand", {}).get("code"),
                    "mlb_debut": p.get("mlbDebutDate"),
                }
            )
    if rows:
        people = pd.concat([people, pd.DataFrame(rows)], ignore_index=True) if len(people) else pd.DataFrame(rows)
        PROSPECTS_DIR.mkdir(parents=True, exist_ok=True)
        people.to_parquet(PEOPLE_PATH, index=False)
    return people


def load_team_games(seasons: list[int], today: date) -> pd.Series:
    """Median games played per LMB team by season (2018 halves summed), the
    yardstick that scales the regular-season playing-time threshold."""
    cached = pd.read_parquet(TEAM_GAMES_PATH) if TEAM_GAMES_PATH.exists() else pd.DataFrame(columns=["season", "games"])
    known = {int(s): float(g) for s, g in zip(cached.season, cached.games) if season_complete("lmb", int(s), today)}
    for season in seasons:
        if season in known:
            continue
        total = 0.0
        for stat_season in _stat_seasons(season):
            splits = _splits(
                get(
                    "/v1/teams/stats",
                    stats="season",
                    group="hitting",
                    season=stat_season,
                    sportIds=sport_id_for(season),
                    leagueIds=LEAGUE_ID,
                )
            )
            if splits:
                total += float(pd.Series([s["stat"]["gamesPlayed"] for s in splits]).median())
        if total:  # an open season with no games yet is left out
            known[season] = total
    PROSPECTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"season": list(known), "games": list(known.values())}).to_parquet(TEAM_GAMES_PATH, index=False)
    return pd.Series(known).sort_index()


def load_inputs(today: date, refresh_open: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, list[int]]:
    """Every season line through `today`, the bios, LMB team games, and the
    complete LMB seasons (which bound outcome windows)."""
    summer = [y for y in range(FIRST_LMB, today.year + 1) if y != 2020 and today >= date(y, 4, 1)]
    winter = [y for y in range(FIRST_LMP, today.year + 1) if today >= date(y, 10, 1)]
    lines = load_lines({"lmb": summer, "lmp": winter, "milb": summer}, today, refresh_open)
    people = load_people(lines.player.unique())
    team_games = load_team_games(summer, today)
    played = [y for y in summer if season_complete("lmb", y, today)]
    return lines, people, team_games, played
