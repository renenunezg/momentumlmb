"""Play-by-play events for player valuation, parsed from the cached live feeds.

The game model's PA frame keeps the state at the start of each plate
appearance. Valuation needs more: the state just before the PA-ending event
(steals, wild passes and pickoffs during the PA are their own events), the
state after it, who stood on each base, who was fielding each position, the
scorer's batted-ball record (trajectory, hardness, spray coordinates) and the
fielders credited. One row per event goes to `events`, one row per runner per
event to `moves`.

The defensive alignment starts from the box score's starters and follows the
substitution and switch events inside each play."""

import gzip
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from backend.config import CACHE_DIR
from backend.etl.pa_frame import EVENT_TO_OUTCOME, FEEDS_DIR, TRAJECTORY_TO_BB_TYPE, final_games

EVENTS_DIR = CACHE_DIR / "players" / "events"
BASES = ("1B", "2B", "3B")
FIELD_POSITIONS = ("C", "1B", "2B", "3B", "SS", "LF", "CF", "RF")
FIELDING_CREDITS = (
    "f_putout",
    "f_assist",
    "f_fielded_ball",
    "f_fielding_error",
    "f_throwing_error",
    "f_deflection",
    "f_assist_of",
    "f_touch",
)
ERROR_CREDITS = ("f_fielding_error", "f_throwing_error")


def _bases_mask(occupant: dict[str, int]) -> int:
    return sum(1 << i for i, b in enumerate(BASES) if b in occupant)


def _side_of_players(box: dict) -> dict[int, str]:
    return {v["person"]["id"]: side for side in ("home", "away") for v in box[side]["players"].values()}


def _first_credit_positions(plays: list[dict]) -> dict[int, str]:
    """Each player's position on his first fielding credit, read only up to
    the first substitution or switch that names him (after it his position may
    have changed)."""
    first: dict[int, str] = {}
    moved: set[int] = set()
    for play in plays:
        for e in play.get("playEvents", []):
            if e.get("type") == "action" and e.get("isSubstitution"):
                for key in ("player", "replacedPlayer"):
                    pid = (e.get(key) or {}).get("id")
                    if pid is not None:
                        moved.add(pid)
        for r in play.get("runners", []):
            for c in r.get("credits", []):
                pid, pos = c["player"]["id"], c["position"]["abbreviation"]
                if c["credit"] in FIELDING_CREDITS and pos in FIELD_POSITIONS and pid not in moved:
                    first.setdefault(pid, pos)
    return first


def _starting_alignment(box: dict, side: str, first_pos: dict[int, str]) -> dict[str, int]:
    """Starters at the position of their first fielding credit; the box
    score's listed position (sometimes the player's usual position rather
    than the one he started at) fills in the rest."""
    starters = [
        (v["person"]["id"], (v.get("position") or {}).get("abbreviation"))
        for v in box[side]["players"].values()
        if (v.get("battingOrder") or "").endswith("00")
    ]
    align: dict[str, int] = {}
    for pid, _ in starters:
        pos = first_pos.get(pid)
        if pos is not None and pos not in align:
            align[pos] = pid
    placed = set(align.values())
    for pid, pos in starters:
        if pid not in placed and pos in FIELD_POSITIONS and pos not in align:
            align[pos] = pid
    return align


def _correct_from_credit(team: dict[str, int], pos: str, pid: int) -> None:
    """A credit shows `pid` at `pos`: put him there, and move whoever stood
    there to the spot he left."""
    if team.get(pos) == pid:
        return
    old = next((p for p, x in team.items() if x == pid), None)
    prev = team.get(pos)
    team[pos] = pid
    if old is not None:
        if prev is not None:
            team[old] = prev
        else:
            del team[old]


def _apply_substitution(event: dict, align: dict[str, dict[str, int]], side_of: dict[int, str]) -> None:
    kind = event["details"].get("eventType")
    if kind not in ("defensive_substitution", "defensive_switch"):
        return
    pos = (event.get("position") or {}).get("abbreviation")
    player = (event.get("player") or {}).get("id")
    side = side_of.get(player)
    if pos not in FIELD_POSITIONS or side is None:
        return
    team = align[side]
    for p, pid in list(team.items()):
        if pid == player:
            del team[p]
    team[pos] = player


def _hit_data(play: dict) -> dict:
    for event in reversed(play.get("playEvents", [])):
        hd = event.get("hitData")
        if hd and hd.get("trajectory"):
            coords = hd.get("coordinates") or {}
            return {
                "bb_type": TRAJECTORY_TO_BB_TYPE.get(hd["trajectory"]),
                "hardness": hd.get("hardness"),
                "location": hd.get("location"),
                "coord_x": coords.get("coordX"),
                "coord_y": coords.get("coordY"),
            }
    return {"bb_type": None, "hardness": None, "location": None, "coord_x": None, "coord_y": None}


def _first_fielder(movements: list[dict]) -> tuple[str | None, int | None, str | None, int | None]:
    """Position and id of the first fielder credited, and of the fielder
    charged with an error, on the movements of one event."""
    first = (None, None)
    error = (None, None)
    for m in movements:
        for c in m.get("credits", []):
            if c["credit"] not in FIELDING_CREDITS:
                continue
            pos, pid = c["position"]["abbreviation"], c["player"]["id"]
            if first[0] is None:
                first = (pos, pid)
            if c["credit"] in ERROR_CREDITS and error[0] is None:
                error = (pos, pid)
    return first[0], first[1], error[0], error[1]


def game_events(feed: dict) -> tuple[list[dict], list[dict]]:
    game = feed["gameData"]
    box = feed["liveData"]["boxscore"]["teams"]
    side_of = _side_of_players(box)
    plays = feed["liveData"]["plays"]["allPlays"]
    first_pos = _first_credit_positions(plays)
    align = {s: _starting_alignment(box, s, first_pos) for s in ("home", "away")}
    meta = {
        "game_pk": feed["gamePk"],
        "season": int(game["game"]["season"]),
        "game_type": game["game"]["type"],
        "official_date": game["datetime"]["officialDate"],
        "venue_id": game["venue"]["id"],
        "home_team_id": game["teams"]["home"]["id"],
        "away_team_id": game["teams"]["away"]["id"],
        "scheduled_innings": feed["liveData"]["linescore"].get("scheduledInnings", 9),
    }
    events: list[dict] = []
    moves: list[dict] = []
    half_key = None
    occupant: dict[str, int] = {}
    outs = 0
    score = {"top": 0, "bottom": 0}
    for play in plays:
        about = play["about"]
        if not about.get("isComplete", True):
            continue
        half = about["halfInning"]
        key = (about["inning"], half)
        if key != half_key:
            half_key, occupant, outs = key, {}, 0
        fielding_side = "home" if half == "top" else "away"
        batting_side = "away" if half == "top" else "home"
        batter = play["matchup"]["batter"]["id"]
        pitcher = play["matchup"]["pitcher"]["id"]
        result_type = play["result"].get("eventType")
        runners = play.get("runners", [])

        # Runners placed on base without a prior movement (the extra-innings runner).
        seen: set[int] = set()
        for r in runners:
            rid = r["details"]["runner"]["id"]
            if rid in seen:
                continue
            seen.add(rid)
            start = r["movement"].get("originBase")
            if start in BASES and start not in occupant and rid not in occupant.values():
                occupant[start] = rid

        batter_moves = [r for r in runners if r["details"]["runner"]["id"] == batter]
        is_pa = result_type in EVENT_TO_OUTCOME and bool(batter_moves)
        final_idx = batter_moves[0]["details"]["playIndex"] if is_pa else None
        groups: dict[int, list[dict]] = {}
        for r in runners:
            idx = r["details"]["playIndex"]
            if final_idx is not None and idx > final_idx:
                idx = final_idx
            groups.setdefault(idx, []).append(r)
        if is_pa and final_idx not in groups:
            groups[final_idx] = []
        substitutions = [
            (i, e)
            for i, e in enumerate(play.get("playEvents", []))
            if e.get("type") == "action" and e.get("isSubstitution")
        ]

        for idx in sorted(groups):
            for i, e in substitutions:
                if i <= idx:
                    _apply_substitution(e, align, side_of)
            substitutions = [(i, e) for i, e in substitutions if i > idx]
            group = groups[idx]
            for r in group:
                for c in r.get("credits", []):
                    pos, pid = c["position"]["abbreviation"], c["player"]["id"]
                    if c["credit"] in FIELDING_CREDITS and pos in FIELD_POSITIONS and side_of.get(pid) == fielding_side:
                        _correct_from_credit(align[fielding_side], pos, pid)
            pa_event = is_pa and idx == final_idx
            bases_before, outs_before = _bases_mask(occupant), outs
            runners_before = {f"runner_{b}": occupant.get(b) for b in BASES}
            runs = 0
            consolidated: dict[int, dict] = {}
            for r in group:
                rid = r["details"]["runner"]["id"]
                mv = r["movement"]
                start, end = mv.get("start"), mv.get("end")
                if start in BASES and occupant.get(start) == rid:
                    del occupant[start]
                entry = consolidated.setdefault(
                    rid, {"start": mv.get("originBase") if rid != batter or not pa_event else None, "end": None}
                )
                if mv.get("isOut"):
                    outs += 1
                    entry["end"] = "out"
                elif end == "score":
                    runs += 1
                    entry["end"] = "score"
                elif end in BASES:
                    occupant[end] = rid
                    entry["end"] = end
                entry["responsible_pitcher"] = (r["details"].get("responsiblePitcher") or {}).get("id")
                entry["event_type"] = r["details"].get("eventType")
            outs = min(outs, 3)
            first_pos, first_id, err_pos, err_id = _first_fielder(group)
            defense = align[fielding_side]
            event_id = f"{meta['game_pk']}-{about['atBatIndex']}-{idx}"
            row = {
                **meta,
                "event_id": event_id,
                "at_bat_index": about["atBatIndex"],
                "play_index": idx,
                "inning": about["inning"],
                "half": half,
                "kind": "pa" if pa_event else "br",
                "event_type": result_type if pa_event else (group[0]["details"].get("eventType") if group else None),
                "batter": batter,
                "pitcher": pitcher,
                "bat_pos": next((p for p, pid in align[batting_side].items() if pid == batter), "DH"),
                "outs_before": outs_before,
                "bases_before": bases_before,
                "outs_after": outs,
                "bases_after": _bases_mask(occupant),
                "runs": runs,
                "bat_score_before": score[half],
                **runners_before,
                "fielder_pos": first_pos,
                "fielder_id": first_id,
                "error_pos": err_pos,
                "error_id": err_id,
                **{f"def_{p}": defense.get(p) for p in FIELD_POSITIONS},
            }
            if pa_event:
                row["outcome"] = EVENT_TO_OUTCOME[result_type]
                row.update(_hit_data(play))
            events.append(row)
            score[half] += runs
            for rid, entry in consolidated.items():
                moves.append(
                    {
                        "event_id": event_id,
                        "runner": rid,
                        "is_batter": rid == batter and pa_event,
                        "start": entry["start"],
                        "end": entry["end"],
                        "responsible_pitcher": entry.get("responsible_pitcher"),
                        "move_type": entry.get("event_type"),
                    }
                )
        # Trust the feed's own count at the end of each play.
        outs = min(play["count"]["outs"], 3)
    return events, moves


def _parse(game_pk: int) -> tuple[list[dict], list[dict]]:
    path: Path = FEEDS_DIR / f"{game_pk}.json.gz"
    return game_events(json.loads(gzip.decompress(path.read_bytes())))


def build_events(season: int, workers: int = 8) -> None:
    games = final_games([season])["game_pk"].tolist()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        parsed = list(pool.map(_parse, games, chunksize=16))
    events = pd.DataFrame([e for ev, _ in parsed for e in ev])
    moves = pd.DataFrame([m for _, mv in parsed for m in mv])
    events["official_date"] = pd.to_datetime(events["official_date"]).dt.date
    EVENTS_DIR.mkdir(parents=True, exist_ok=True)
    events.to_parquet(EVENTS_DIR / f"events_{season}.parquet", index=False)
    moves.to_parquet(EVENTS_DIR / f"moves_{season}.parquet", index=False)


def load_events(seasons: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    ev = pd.concat([pd.read_parquet(EVENTS_DIR / f"events_{s}.parquet") for s in seasons], ignore_index=True)
    mv = pd.concat([pd.read_parquet(EVENTS_DIR / f"moves_{s}.parquet") for s in seasons], ignore_index=True)
    for col in ("fielder_id", "error_id", *[f"def_{p}" for p in FIELD_POSITIONS], *[f"runner_{b}" for b in BASES]):
        ev[col] = ev[col].astype("Int64")
    return ev, mv


def spray_angle(coord_x: pd.Series, coord_y: pd.Series) -> np.ndarray:
    """Degrees from straightaway center, negative toward left field, from the
    scorer's hit-chart coordinates (home plate at about 125.4, 198.3)."""
    return np.degrees(np.arctan2(coord_x.astype(float) - 125.42, 198.27 - coord_y.astype(float)))
