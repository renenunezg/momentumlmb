"""The day's playoff odds for every team, as rows for lmb.playoff_odds."""

from datetime import date

import numpy as np
import pandas as pd

from backend.playoffs.simulate import STAGES, simulate
from backend.playoffs.state import ZONES, build_state
from backend.playoffs.strength import season_ratings
from backend.statsapi.league import TEAMS

MODEL = "lmb-playoffs-v1"
N_SIMS = 20_000


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
