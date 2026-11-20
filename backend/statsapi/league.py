"""LMB identity in the MLB Stats API."""

from dataclasses import dataclass

import pandas as pd

from backend.config import STATIC_DIR

LEAGUE_ID = 125
ZONE_BY_DIVISION = {222: "Norte", 223: "Sur"}
GAME_TYPES = ("R", "F", "D", "L", "W")


def sport_id_for(season: int) -> int:
    """The LMB sat under Triple-A (sportId 11) through 2019 and under
    Independent Leagues (sportId 23) from 2021. 2020 was cancelled."""
    if season == 2020:
        raise ValueError("The 2020 LMB season was cancelled")
    if season < 2005:
        raise ValueError("The Stats API has no LMB data before 2005")
    return 11 if season <= 2019 else 23


@dataclass(frozen=True)
class Team:
    id: int
    code: str
    name: str
    display_name: str
    zone: str


def _load_teams() -> dict[int, Team]:
    frame = pd.read_csv(STATIC_DIR / "teams.csv")
    return {int(r.id): Team(int(r.id), r.code, r.name, r.display_name, r.zone) for r in frame.itertuples()}


TEAMS: dict[int, Team] = _load_teams()


def refresh_teams_csv(season: int = 2026) -> None:
    """Rebuild teams.csv from the Stats API (run when the league changes).
    Accented display names are kept for teams already in the file; new teams
    start with the API name and need their accents added by hand."""
    from backend.statsapi.client import get

    path = STATIC_DIR / "teams.csv"
    display = dict(pd.read_csv(path)[["id", "display_name"]].itertuples(index=False))

    rows = [
        {
            "id": t["id"],
            "code": t["abbreviation"],
            "name": t["name"],
            "display_name": display.get(t["id"], t["name"]),
            "zone": ZONE_BY_DIVISION[t["division"]["id"]],
        }
        for t in get("/v1/teams", sportId=sport_id_for(season), season=season, leagueIds=LEAGUE_ID)[
            "teams"
        ]
    ]
    pd.DataFrame(rows).sort_values("id").to_csv(path, index=False)
