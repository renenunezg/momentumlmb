"""One row per plate appearance, rebuilt from Stats API play-by-play.

LMB has no Statcast, so this replaces mlbmodel's Statcast PA frame with the
same column names. Every column the forecast reads is in the public feed:
event, batter and pitcher with handedness, inning, outs and base state when
the PA started, score before and after, and the scorer's batted-ball type.
Base and out state are carried play to play inside each half-inning, so
steals, pickoffs and wild pitches between PAs move the state the next PA
starts from, as in Statcast."""

import gzip
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from backend.config import CACHE_DIR
from backend.etl.games import load_games
from backend.statsapi.client import get

FEEDS_DIR = CACHE_DIR / "feeds"
PA_DIR = CACHE_DIR / "pa"

OUTCOMES = ("K", "BB", "HBP", "1B", "2B", "3B", "HR", "OUT")

# Stats API eventType values share Statcast's event names (mlbmodel v2/data/pa_dataset.py).
EVENT_TO_OUTCOME = {
    "single": "1B",
    "double": "2B",
    "triple": "3B",
    "home_run": "HR",
    "strikeout": "K",
    "strikeout_double_play": "K",
    "strikeout_triple_play": "K",
    "walk": "BB",
    "intent_walk": "BB",
    "hit_by_pitch": "HBP",
    "field_out": "OUT",
    "force_out": "OUT",
    "grounded_into_double_play": "OUT",
    "grounded_into_triple_play": "OUT",
    "double_play": "OUT",
    "triple_play": "OUT",
    "fielders_choice": "OUT",
    "fielders_choice_out": "OUT",
    "sac_fly": "OUT",
    "sac_fly_double_play": "OUT",
    "sac_bunt": "OUT",
    "sac_bunt_double_play": "OUT",
    "field_error": "OUT",
    "catcher_interf": "OUT",
}

EVENT_TO_OUT_SUBTYPE = {
    "field_out": "field_out",
    "force_out": "force_out",
    "grounded_into_double_play": "gidp",
    "grounded_into_triple_play": "tp",
    "double_play": "dp",
    "triple_play": "tp",
    "fielders_choice": "fc",
    "fielders_choice_out": "fc",
    "sac_fly": "sac_fly",
    "sac_fly_double_play": "sac_fly_dp",
    "sac_bunt": "sac_bunt",
    "sac_bunt_double_play": "sac_bunt_dp",
    "field_error": "roe",
    "catcher_interf": "ci",
    "strikeout_double_play": "k_dp",
    "strikeout_triple_play": "k_dp",
}

# The scorer's trajectory; bunts fold into their batted-ball family.
TRAJECTORY_TO_BB_TYPE = {
    "ground_ball": "ground_ball",
    "bunt_grounder": "ground_ball",
    "line_drive": "line_drive",
    "bunt_line_drive": "line_drive",
    "fly_ball": "fly_ball",
    "popup": "popup",
    "bunt_popup": "popup",
}

BASES = ("1B", "2B", "3B")

PA_FRAME_COLS = [
    "game_pk",
    "official_date",
    "season",
    "game_type",
    "venue_id",
    "home_team_id",
    "away_team_id",
    "scheduled_innings",
    "batter",
    "pitcher",
    "stand",
    "p_throws",
    "inning",
    "inning_topbot",
    "at_bat_number",
    "outs_when_up",
    "on_1b",
    "on_2b",
    "on_3b",
    "bat_score",
    "post_bat_score",
    "events",
    "outcome",
    "out_subtype",
    "bb_type",
]


def _bases_after(play: dict, before: set[str]) -> set[str]:
    """Occupied bases once the play is over: runners who moved end where their
    last movement ends; runners who never moved stay put."""
    moved_from: set[str] = set()
    occupied_after: set[str] = set()
    last: dict[int, dict] = {}
    for runner in play.get("runners", []):
        rid = runner["details"]["runner"]["id"]
        origin = runner["movement"].get("originBase")
        if origin in BASES and rid not in last:
            moved_from.add(origin)
        last[rid] = runner["movement"]
    for movement in last.values():
        if not movement.get("isOut") and movement.get("end") in BASES:
            occupied_after.add(movement["end"])
    return (before - moved_from) | occupied_after


def _bases_at_start(play: dict, carried: set[str]) -> set[str]:
    """Carried state plus any runner whose first movement in this play starts
    on a base (the extra-innings runner appears this way)."""
    seen: set[int] = set()
    start = set(carried)
    for runner in play.get("runners", []):
        rid = runner["details"]["runner"]["id"]
        if rid in seen:
            continue
        seen.add(rid)
        origin = runner["movement"].get("originBase")
        if origin in BASES:
            start.add(origin)
    return start


def _bb_type(play: dict) -> str | None:
    for event in reversed(play.get("playEvents", [])):
        trajectory = (event.get("hitData") or {}).get("trajectory")
        if trajectory:
            return TRAJECTORY_TO_BB_TYPE.get(trajectory)
    return None


def pa_rows(feed: dict) -> pd.DataFrame:
    game = feed["gameData"]
    meta = {
        "game_pk": feed["gamePk"],
        "official_date": pd.Timestamp(game["datetime"]["officialDate"]).date(),
        "season": int(game["game"]["season"]),
        "game_type": game["game"]["type"],
        "venue_id": game["venue"]["id"],
        "home_team_id": game["teams"]["home"]["id"],
        "away_team_id": game["teams"]["away"]["id"],
        "scheduled_innings": feed["liveData"]["linescore"].get("scheduledInnings", 9),
    }
    rows = []
    half_key = None
    bases: set[str] = set()
    outs = 0
    score = {"top": 0, "bottom": 0}
    for play in feed["liveData"]["plays"]["allPlays"]:
        about = play["about"]
        if not about.get("isComplete", True):
            continue
        half = about["halfInning"]
        key = (about["inning"], half)
        if key != half_key:
            half_key, bases, outs = key, set(), 0
        start = _bases_at_start(play, bases)
        result = play["result"]
        bat_score = score[half]
        post = result["awayScore"] if half == "top" else result["homeScore"]
        event = result.get("eventType")
        outcome = EVENT_TO_OUTCOME.get(event)
        if outcome is not None:
            matchup = play["matchup"]
            rows.append(
                {
                    **meta,
                    "batter": matchup["batter"]["id"],
                    "pitcher": matchup["pitcher"]["id"],
                    "stand": matchup["batSide"]["code"],
                    "p_throws": matchup["pitchHand"]["code"],
                    "inning": about["inning"],
                    "inning_topbot": half,
                    "at_bat_number": about["atBatIndex"] + 1,
                    "outs_when_up": outs,
                    "on_1b": "1B" in start,
                    "on_2b": "2B" in start,
                    "on_3b": "3B" in start,
                    "bat_score": bat_score,
                    "post_bat_score": post,
                    "events": event,
                    "outcome": outcome,
                    "out_subtype": EVENT_TO_OUT_SUBTYPE.get(event),
                    "bb_type": _bb_type(play),
                }
            )
        bases = _bases_after(play, start)
        outs = play["count"]["outs"]
        score[half] = post
    return pd.DataFrame(rows, columns=PA_FRAME_COLS)


def _trim(payload: dict) -> dict:
    """Keep what the PA frame and workloads read; drop per-pitch tracking
    blocks (empty for the LMB) and the player directory."""
    game = payload["gameData"]
    box = payload["liveData"]["boxscore"]["teams"]
    plays = payload["liveData"]["plays"]["allPlays"]
    for play in plays:
        for event in play["playEvents"]:
            event.pop("pitchData", None)
    return {
        "gamePk": payload["gamePk"],
        "gameData": {
            "game": game["game"],
            "datetime": game["datetime"],
            "status": game["status"],
            "venue": {"id": game["venue"]["id"], "name": game["venue"]["name"]},
            "teams": {s: {"id": game["teams"][s]["id"], "name": game["teams"][s]["name"]} for s in ("home", "away")},
            "probablePitchers": game.get("probablePitchers", {}),
        },
        "liveData": {
            "plays": {"allPlays": plays},
            "linescore": payload["liveData"]["linescore"],
            "boxscore": {
                "teams": {
                    s: {
                        "teamStats": box[s]["teamStats"],
                        "pitchers": box[s].get("pitchers", []),
                        "battingOrder": box[s].get("battingOrder", []),
                        "players": {
                            k: {
                                "person": {"id": v["person"]["id"]},
                                "position": v.get("position", {}),
                                "battingOrder": v.get("battingOrder"),
                                "stats": {"pitching": v.get("stats", {}).get("pitching", {})},
                            }
                            for k, v in box[s].get("players", {}).items()
                        },
                    }
                    for s in ("home", "away")
                }
            },
        },
    }


def feed(game_pk: int) -> dict:
    """Live feed for a final game, fetched once and cached trimmed and gzipped."""
    FEEDS_DIR.mkdir(parents=True, exist_ok=True)
    path = FEEDS_DIR / f"{game_pk}.json.gz"
    if path.exists():
        return json.loads(gzip.decompress(path.read_bytes()))
    payload = _trim(get(f"/v1.1/game/{game_pk}/feed/live"))
    if payload["gameData"]["status"]["abstractGameState"] == "Final":
        path.write_bytes(gzip.compress(json.dumps(payload, separators=(",", ":")).encode()))
    return payload


FINAL_STATES = ("Final", "Completed Early")


def final_games(seasons: list[int]) -> pd.DataFrame:
    games = load_games(seasons)
    return games[games["status"].isin(FINAL_STATES)]


def build_pa_cache(seasons: list[int]) -> Path:
    PA_DIR.mkdir(parents=True, exist_ok=True)
    for season in seasons:
        with ThreadPoolExecutor(max_workers=4) as pool:
            feeds = list(pool.map(feed, final_games([season])["game_pk"]))
        frames = [pa_rows(f) for f in feeds]
        pd.concat(frames, ignore_index=True).to_parquet(PA_DIR / f"{season}.parquet", index=False)
    return PA_DIR


def load_pa(seasons: list[int], missing_ok: bool = False) -> pd.DataFrame:
    """Cached seasons; with missing_ok, seasons without a cache yet (a new
    season before its first final game) are skipped."""
    paths = [PA_DIR / f"{s}.parquet" for s in seasons]
    if missing_ok:
        paths = [p for p in paths if p.exists()]
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
