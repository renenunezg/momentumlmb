"""Pregame inputs from the Stats API: probable starters, posted lineups and
active rosters."""

from datetime import date

from backend.statsapi.client import get
from backend.statsapi.league import LEAGUE_ID, sport_id_for


def probables(day: date) -> dict[int, dict[str, int | None]]:
    payload = get(
        "/v1/schedule",
        sportId=sport_id_for(day.year),
        leagueId=LEAGUE_ID,
        date=day.isoformat(),
        hydrate="probablePitcher",
    )
    out = {}
    for d in payload.get("dates", []):
        for g in d["games"]:
            out[g["gamePk"]] = {
                side: (g["teams"][side].get("probablePitcher") or {}).get("id") for side in ("home", "away")
            }
    return out


def posted_lineups(game_pk: int) -> dict[str, list[int] | None]:
    """Batting orders once posted (nine players), else None per side."""
    box = get(f"/v1/game/{game_pk}/boxscore")["teams"]
    out = {}
    for side in ("home", "away"):
        order = box[side].get("battingOrder", [])
        out[side] = [int(x) for x in order[:9]] if len(order) >= 9 else None
    return out


def active_roster(team_id: int, day: date) -> dict[str, list[int]]:
    """Active roster split into pitchers and position players."""
    roster = get(f"/v1/teams/{team_id}/roster", rosterType="active", date=day.isoformat()).get("roster", [])
    pitchers = [p["person"]["id"] for p in roster if p["position"]["type"] == "Pitcher"]
    hitters = [p["person"]["id"] for p in roster if p["position"]["type"] != "Pitcher"]
    return {"pitchers": pitchers, "hitters": hitters}
