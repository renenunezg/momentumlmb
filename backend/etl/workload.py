"""Pitcher appearances per game from the boxscore, the input to bullpen
queues (starter workload, reliever rest)."""

from pathlib import Path

import pandas as pd

from backend.config import CACHE_DIR
from backend.etl.pa_frame import feed, final_games

WORKLOAD_DIR = CACHE_DIR / "workload"
COLUMNS = ["game_pk", "official_date", "team_id", "pitcher_id", "is_starter", "order", "outs", "pitches", "runs"]


def pitcher_appearances(box: dict, game: pd.Series) -> pd.DataFrame:
    rows = []
    for side in ("home", "away"):
        team = box["teams"][side]
        team_id = game[f"{side}_team_id"]
        for order, pitcher_id in enumerate(team.get("pitchers", [])):
            stats = team["players"].get(f"ID{pitcher_id}", {}).get("stats", {}).get("pitching", {})
            rows.append(
                {
                    "game_pk": game["game_pk"],
                    "official_date": game["official_date"],
                    "team_id": team_id,
                    "pitcher_id": pitcher_id,
                    "is_starter": order == 0,
                    "order": order,
                    "outs": int(stats.get("outs", 0)),
                    "pitches": int(stats.get("numberOfPitches", stats.get("pitchesThrown", 0))),
                    "runs": int(stats.get("runs", 0)),
                }
            )
    return pd.DataFrame(rows, columns=COLUMNS)


def build_workload_cache(seasons: list[int]) -> Path:
    WORKLOAD_DIR.mkdir(parents=True, exist_ok=True)
    for season in seasons:
        games = final_games([season])
        frames = [
            pitcher_appearances(feed(g.game_pk)["liveData"]["boxscore"], pd.Series(g._asdict()))
            for g in games.itertuples(index=False)
        ]
        pd.concat(frames, ignore_index=True).to_parquet(WORKLOAD_DIR / f"{season}.parquet", index=False)
    return WORKLOAD_DIR


def load_workload(seasons: list[int], missing_ok: bool = False) -> pd.DataFrame:
    """Cached seasons; with missing_ok, seasons without a cache yet (a new
    season before its first final game) are skipped."""
    paths = [WORKLOAD_DIR / f"{s}.parquet" for s in seasons]
    if missing_ok:
        paths = [p for p in paths if p.exists()]
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
