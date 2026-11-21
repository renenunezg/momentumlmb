"""Season lines, schedules, managers and winners behind the LMB award boards.

Everything comes from the Stats API and is cached under cache/awards, so a refit
touches no network. The LMB publishes one winner per award and no votes, so
winners are the only targets. 2018 was played as two short seasons stored as
"2018.1" and "2018.2"; each is kept apart here and merged by `year_lines`.
"""

from __future__ import annotations

import json

import pandas as pd

from backend.config import CACHE_DIR
from backend.statsapi.client import get

CACHE = CACHE_DIR / "awards"
FIRST_YEAR = 2005
AWARD_CODES = {"MVP": "MEXMVP", "POY": "MEXPOY", "REL": "MEXREOY", "ROY": "MEXROY", "MOY": "MEXMOY"}
_COUNTS = (
    "plateAppearances",
    "atBats",
    "hits",
    "doubles",
    "triples",
    "homeRuns",
    "rbi",
    "baseOnBalls",
    "intentionalWalks",
    "hitByPitch",
    "sacFlies",
    "stolenBases",
    "outs",
    "earnedRuns",
    "runs",
    "wins",
    "losses",
    "saves",
    "holds",
    "strikeOuts",
    "gamesPitched",
    "gamesStarted",
    "gamesPlayed",
    "battersFaced",
)


def season_keys(year: int) -> list[str]:
    if year == 2020 or year < FIRST_YEAR:
        raise ValueError(f"No LMB season {year}")
    return ["2018.1", "2018.2"] if year == 2018 else [str(year)]


def _sport(key: str) -> int:
    return 23 if int(key[:4]) >= 2021 else 11


def _cached(name: str, path: str, **params) -> dict:
    """Finished seasons never change, so a response is kept for good; the board's own
    season is refetched only when its cache file is deleted."""
    target = CACHE / f"{name}.json"
    if target.exists():
        return json.loads(target.read_text())
    payload = get(path, **params)
    CACHE.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload))
    return payload


def lines(key: str, group: str) -> pd.DataFrame:
    """One row per player for one API season: his totals across LMB clubs, the club he ended with."""
    (block,) = _cached(
        f"lines_{group}_{key}",
        "/v1/stats",
        stats="season",
        group=group,
        leagueId=125,
        sportId=_sport(key),
        season=key,
        limit=5000,
        playerPool="ALL",
        gameType="R",
    )["stats"]
    if len(block["splits"]) != block["totalSplits"]:
        raise ValueError(f"Truncated {group} lines for {key}")
    frame = pd.DataFrame(
        [
            {
                **s["stat"],
                "player_id": s["player"]["id"],
                "player_name": s["player"].get("fullName"),
                "team_id": s["team"]["id"],
                "team": s["team"]["name"],
            }
            for s in block["splits"]
        ]
    )
    if frame.player_id.duplicated().any():
        raise ValueError(f"Duplicate players in {group} lines for {key}")
    for column in _COUNTS:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    return frame


def year_lines(year: int, group: str) -> pd.DataFrame:
    """A year's lines, with 2018's two seasons summed; a player keeps the club he ended the year with."""
    parts = [lines(key, group) for key in season_keys(year)]
    if len(parts) == 1:
        return parts[0]
    both = pd.concat(parts, ignore_index=True)
    counts = [c for c in _COUNTS if c in both]
    return both.groupby("player_id", as_index=False, sort=False).agg(
        {**{c: "sum" for c in counts}, "player_name": "last", "team_id": "last", "team": "last"}
    )


def games(key: str) -> pd.DataFrame:
    """Final regular-season games of one API season, one row per gamePk."""
    payload = _cached(f"schedule_{key}", "/v1/schedule", sportId=_sport(key), leagueId=125, season=key, gameType="R")
    rows = [
        {
            "game_pk": g["gamePk"],
            "venue_id": g["venue"]["id"],
            "home_team_id": g["teams"]["home"]["team"]["id"],
            "away_team_id": g["teams"]["away"]["team"]["id"],
            "home_score": g["teams"]["home"].get("score"),
            "away_score": g["teams"]["away"].get("score"),
        }
        for day in payload["dates"]
        for g in day["games"]
        if g["status"]["abstractGameState"] == "Final" and g["teams"]["home"].get("score") is not None
    ]
    return pd.DataFrame(rows).drop_duplicates("game_pk", keep="last").reset_index(drop=True)


def standings(year: int) -> pd.DataFrame:
    """Wins and losses per club over the whole year, counted from final games (the standings
    endpoint lists only one division for the first 2018 season)."""
    g = pd.concat([games(key) for key in season_keys(year)], ignore_index=True)
    g = g[g.home_score != g.away_score]
    home_won = g.home_score > g.away_score
    winners = pd.concat([g.home_team_id[home_won], g.away_team_id[~home_won]])
    losers = pd.concat([g.away_team_id[home_won], g.home_team_id[~home_won]])
    frame = pd.DataFrame({"wins": winners.value_counts(), "losses": losers.value_counts()}).fillna(0)
    frame = frame.rename_axis("team_id").reset_index()
    if len(frame) < 14:
        raise ValueError(f"Incomplete standings for {year}")
    return frame.assign(win_pct=frame.wins / (frame.wins + frame.losses))


def managers(year: int) -> pd.DataFrame:
    """Every manager the Stats API lists for a club and year. A club that changed managers lists
    them all, in no reliable order, so each is credited with the club's whole record."""
    rows = []
    for team_id in standings(year).team_id:
        payload = _cached(f"coaches_{team_id}_{year}", f"/v1/teams/{team_id}/coaches", season=year)
        listed = [c for c in payload.get("roster", []) if c.get("job") == "Manager"]
        rows += [
            {"team_id": team_id, "player_id": c["person"]["id"], "player_name": c["person"]["fullName"]} for c in listed
        ]
    return pd.DataFrame(rows, columns=["team_id", "player_id", "player_name"])


def people(ids: list[int]) -> pd.DataFrame:
    """Birth country and date, cached per person."""
    missing = [i for i in ids if not (CACHE / "people" / f"{i}.json").exists()]
    for start in range(0, len(missing), 150):
        chunk = missing[start : start + 150]
        payload = get("/v1/people", personIds=",".join(map(str, chunk)))
        (CACHE / "people").mkdir(parents=True, exist_ok=True)
        for person in payload["people"]:
            (CACHE / "people" / f"{person['id']}.json").write_text(json.dumps(person))
        for absent in set(chunk) - {p["id"] for p in payload["people"]}:
            (CACHE / "people" / f"{absent}.json").write_text(json.dumps({"id": absent}))
    rows = [json.loads((CACHE / "people" / f"{i}.json").read_text()) for i in ids]
    return pd.DataFrame(
        [{"player_id": p["id"], "birth_country": p.get("birthCountry"), "birth_date": p.get("birthDate")} for p in rows]
    )


def recipients(award: str) -> pd.DataFrame:
    """Every listed winner of one award, with the season he won it. Two 2022 winners are filed
    under season "2002"; their award date gives the real year."""
    payload = _cached(f"recipients_{AWARD_CODES[award]}", f"/v1/awards/{AWARD_CODES[award]}/recipients")
    rows = []
    for a in payload["awards"]:
        year = int(a["season"])
        if year < FIRST_YEAR:
            year = int(a["date"][:4])
            if year < FIRST_YEAR or year == 2020:
                raise ValueError(f"Unplaceable {award} recipient {a}")
        notes = a.get("notes") or ""
        key = "2018.1" if "First" in notes else "2018.2" if "Second" in notes else str(year)
        rows.append(
            {"year": year, "key": key, "player_id": a["player"]["id"], "player_name": a["player"].get("nameFirstLast")}
        )
    return pd.DataFrame(rows)
