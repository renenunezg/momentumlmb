"""Team strength for games beyond today: a Bradley-Terry rating per team from
this season's wins and losses, shrunk toward a share of last season's rating,
with a Laplace posterior so each simulated season draws its own ratings.

logit P(home wins) = home + r_home - r_away, with a Gaussian prior
r ~ N(PRIOR_WEIGHT * r_last_season, 1 / (2 * PRIOR_PRECISION)).

Measured against ratings fitted to the game simulator's own probabilities on
the remaining 2024-2025 regular-season games: this rating had the lower log
loss from every as-of date, so the simulator is not used for future games.
PRIOR_PRECISION and PRIOR_WEIGHT are fitted on 2022-2023 (backend.playoffs.backtest)."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

from backend.playoffs.state import PLAYED

PRIOR_PRECISION = 4.35
PRIOR_WEIGHT = 0.50


@dataclass(frozen=True)
class Ratings:
    mean: np.ndarray  # (T,) per team, in SeasonState.teams order
    cov: np.ndarray  # (T, T)
    home: float  # logit home advantage

    def draw(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.multivariate_normal(self.mean, self.cov, size=n, method="cholesky")

    def p_home(self, home: np.ndarray, away: np.ndarray) -> np.ndarray:
        return expit(self.home + self.mean[home] - self.mean[away])


def _results(games: pd.DataFrame, idx: dict[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    games = games[games.home_team_id.isin(idx) & games.away_team_id.isin(idx)]
    games = games[games.home_score != games.away_score]
    return (
        games.home_team_id.map(idx).to_numpy(),
        games.away_team_id.map(idx).to_numpy(),
        (games.home_score > games.away_score).to_numpy(float),
    )


def fit(
    games: pd.DataFrame,
    teams: np.ndarray,
    prior_mean: np.ndarray,
    home: float,
    precision: float = PRIOR_PRECISION,
) -> Ratings:
    """MAP ratings and their Laplace covariance from played games (schedule
    frame rows with scores) at a fixed home advantage."""
    idx = {int(t): i for i, t in enumerate(teams)}
    h, a, y = _results(games, idx)
    n = len(teams)

    def objective(r):
        p = expit(home + r[h] - r[a])
        d = r - prior_mean
        loss = -np.sum(y * np.log(p) + (1 - y) * np.log1p(-p)) + precision * d @ d
        g = p - y
        grad = 2 * precision * d
        np.add.at(grad, h, g)
        np.add.at(grad, a, -g)
        return loss, grad

    r = minimize(objective, prior_mean.copy(), jac=True, method="L-BFGS-B").x
    w = expit(home + r[h] - r[a])
    w = w * (1 - w)
    hess = 2 * precision * np.eye(n)
    np.add.at(hess, (h, h), w)
    np.add.at(hess, (a, a), w)
    np.add.at(hess, (h, a), -w)
    np.add.at(hess, (a, h), -w)
    cov = np.linalg.inv(hess)
    return Ratings(mean=r, cov=(cov + cov.T) / 2, home=home)


def home_advantage(prior_games: pd.DataFrame) -> float:
    """logit of the home win rate in earlier regular-season games."""
    decided = prior_games[prior_games.home_score != prior_games.away_score]
    return float(logit((decided.home_score > decided.away_score).mean()))


def season_ratings(
    games: pd.DataFrame,
    season: int,
    teams: np.ndarray,
    as_of,
    precision: float = PRIOR_PRECISION,
    weight: float = PRIOR_WEIGHT,
) -> Ratings:
    """Ratings on the morning of `as_of`. `games`: regular-season games of
    `season` and earlier seasons with final scores; only games before `as_of`
    are read. Last season's rating (same model, no prior of its own) is the
    prior mean; a team new to the league starts at average."""
    played = games[(games.game_type == "R") & games.status.isin(PLAYED) & (games.official_date < as_of)]
    current = played[played.season == season]
    last = played[played.season == season - 1]
    prior = np.zeros(len(teams))
    if not last.empty:
        last_teams = np.array(sorted(set(last.home_team_id) | set(last.away_team_id)))
        last_home = home_advantage(last)
        last_fit = fit(last, last_teams, np.zeros(len(last_teams)), last_home, precision)
        by_team = dict(zip(last_teams, last_fit.mean))
        prior = weight * np.array([by_team.get(int(t), 0.0) for t in teams])
    return fit(current, teams, prior, home_advantage(played), precision)
