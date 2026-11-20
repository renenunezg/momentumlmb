"""LMB games from the Stats API schedule, cached per season.

Seasons are pulled by date range (March 1 to October 31), never by `season=`,
because a season query silently drops the 2025 and 2026 postseasons."""

from datetime import date
from pathlib import Path

import pandas as pd

from backend.config import CACHE_DIR
from backend.statsapi.client import get
from backend.statsapi.league import GAME_TYPES, LEAGUE_ID, sport_id_for

GAMES_DIR = CACHE_DIR / "games"
COLUMNS = [
    "game_pk",
    "official_date",
    "start_time_utc",
    "season",
    "game_type",
    "home_team_id",
    "away_team_id",
    "venue_id",
    "scheduled_innings",
    "double_header",
    "game_number",
    "status",
    "home_score",
    "away_score",
]


def fetch_games(start: date, end: date) -> pd.DataFrame:
    """Every LMB game of the listed game types scheduled from start to end."""
    payload = get(
        "/v1/schedule",
        sportId=sport_id_for(start.year),
        leagueId=LEAGUE_ID,
        startDate=start.isoformat(),
        endDate=end.isoformat(),
        gameType=",".join(GAME_TYPES),
    )
    rows = []
    for day in payload.get("dates", []):
        for g in day["games"]:
            if g["gameType"] not in GAME_TYPES:
                continue
            rows.append(
                {
                    "game_pk": g["gamePk"],
                    "official_date": g["officialDate"],
                    "start_time_utc": g["gameDate"],
                    "season": int(g["season"]),
                    "game_type": g["gameType"],
                    "home_team_id": g["teams"]["home"]["team"]["id"],
                    "away_team_id": g["teams"]["away"]["team"]["id"],
                    "venue_id": g["venue"]["id"],
                    "scheduled_innings": g.get("scheduledInnings", 9),
                    "double_header": g.get("doubleHeader", "N"),
                    "game_number": g.get("gameNumber", 1),
                    "status": g["status"]["detailedState"],
                    "home_score": g["teams"]["home"].get("score"),
                    "away_score": g["teams"]["away"].get("score"),
                }
            )
    frame = pd.DataFrame(rows, columns=COLUMNS)
    # A suspended game can appear on two dates; keep its latest listing.
    frame = frame.drop_duplicates("game_pk", keep="last")
    frame["official_date"] = pd.to_datetime(frame["official_date"]).dt.date
    frame["start_time_utc"] = pd.to_datetime(frame["start_time_utc"], utc=True)
    return frame.reset_index(drop=True)


def build_games_cache(seasons: list[int]) -> Path:
    GAMES_DIR.mkdir(parents=True, exist_ok=True)
    for season in seasons:
        frame = fetch_games(date(season, 3, 1), date(season, 10, 31))
        frame.to_parquet(GAMES_DIR / f"{season}.parquet", index=False)
    return GAMES_DIR


def load_games(seasons: list[int]) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(GAMES_DIR / f"{s}.parquet") for s in seasons], ignore_index=True)
