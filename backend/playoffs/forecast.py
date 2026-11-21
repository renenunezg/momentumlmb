"""The day's playoff odds for every team, as rows for lmb.playoff_odds."""

from datetime import date, timedelta

import numpy as np
import pandas as pd

from backend.playoffs.simulate import STAGES, simulate
from backend.playoffs.state import ZONES, build_state, title_decided_on, zones
from backend.playoffs.strength import season_ratings
from backend.statsapi.league import TEAMS

MODEL = "lmb-playoffs-v1"
N_SIMS = 20_000


def skip_reason(games: pd.DataFrame, day: date) -> str | None:
    """Why `day` gets no odds, or None. Outside the season the odds would not
    move, so nothing is written before opening day or after the morning after
    the title is decided. A season with a schedule but no zones raises."""
    season = day.year
    schedule = games[games.season == season]
    if schedule.empty:
        return f"no {season} schedule yet"
    zones(season)
    opening = schedule[schedule.game_type == "R"].official_date.min()
    if day < opening:
        return f"the {season} season opens {opening}"
    decided = title_decided_on(schedule, season)
    if decided is not None and day > decided + timedelta(days=1):
        return f"the {season} title was decided on {decided}"
    return None


def odds_rows(games: pd.DataFrame, day: date, n_sims: int = N_SIMS) -> list[dict]:
    """Odds on the morning of `day` (games dated before it are counted).
    `games`: this season's schedule and last season's games. The seed is the
    date, so a rerun on unchanged results gives identical rows."""
    season = day.year
    state = build_state(games, season, day)
    ratings = season_ratings(games, season, state.teams, day)
    odds = simulate(state, ratings, n_sims, np.random.default_rng(day.toordinal()))
    rows = []
    for i, team in enumerate(state.teams):
        rows.append(
            {
                "season": season,
                "as_of": day,
                "team": TEAMS[int(team)].code if int(team) in TEAMS else str(int(team)),
                "zone": ZONES[state.zone[i]],
                "wins": int(state.wins[i]),
                "losses": int(state.losses[i]),
                "proj_wins": round(float(odds.expected_wins[i]), 1),
                "proj_losses": round(float(odds.expected_losses[i]), 1),
            }
            | {f"p_{s}": int(round(float(p) * 10_000)) for s, p in zip(STAGES, odds.reach[i])}
        )
    return rows
