"""Benchmarks for the walk-forward evaluation, never published.

home_field: the home team's win rate in all prior games.
team_strength: each team's runs scored and allowed per game this season,
shrunk toward the league with SHRINK_GAMES phantom games, scaled by the
venue's run factor from prior seasons, played out on independent Poisson
run counts; ties go to the home team at the prior home win rate."""

from datetime import date

import numpy as np
import pandas as pd
from scipy.stats import poisson

SHRINK_GAMES = 10
MAX_RUNS = 30


def home_field(history: pd.DataFrame, game_date: date) -> float:
    prior = history[history.official_date < game_date]
    return float(prior.home_win.mean())


def _venue_factors(prior: pd.DataFrame) -> pd.Series:
    rpg = (prior.home_score + prior.away_score).groupby(prior.venue_id).mean()
    return rpg / (prior.home_score + prior.away_score).mean()


def team_strength(history: pd.DataFrame, game: pd.Series) -> float:
    prior = history[history.official_date < game.official_date]
    season = prior[prior.season == game.season]
    league = (prior.home_score + prior.away_score).mean() / 2
    venues = _venue_factors(prior[prior.season < game.season]) if (prior.season < game.season).any() else pd.Series()

    def rates(team: int) -> tuple[float, float]:
        home = season[season.home_team_id == team]
        away = season[season.away_team_id == team]
        n = len(home) + len(away)
        scored = home.home_score.sum() + away.away_score.sum()
        allowed = home.away_score.sum() + away.home_score.sum()
        return (scored + SHRINK_GAMES * league) / (n + SHRINK_GAMES), (allowed + SHRINK_GAMES * league) / (
            n + SHRINK_GAMES
        )

    h_off, h_def = rates(int(game.home_team_id))
    a_off, a_def = rates(int(game.away_team_id))
    park = float(venues.get(int(game.venue_id), 1.0))
    lam_home = h_off * a_def / league * park
    lam_away = a_off * h_def / league * park
    k = np.arange(MAX_RUNS + 1)
    ph, pa = poisson.pmf(k, lam_home), poisson.pmf(k, lam_away)
    joint = np.outer(ph, pa)
    win = np.tril(joint, -1).sum()
    tie = np.trace(joint)
    return float(win + tie * home_field(history, game.official_date))
