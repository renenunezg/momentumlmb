"""Playoff-odds backtests, never published.

tune: PRIOR_PRECISION and PRIOR_WEIGHT by the log loss of every remaining
regular-season game from weekly as-of dates, on development seasons only.
calibrate: odds on the morning of every day from opening day to the last
postseason game, against what happened, with the same simulation run on
standings alone (every team rated average) as the benchmark.

    poetry run python -m backend.playoffs.backtest tune --seasons 2022 2023
    poetry run python -m backend.playoffs.backtest calibrate --seasons 2024 2025 --out odds.parquet
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from backend.etl.games import load_games
from backend.evaluation.metrics import log_loss
from backend.playoffs.simulate import STAGES, simulate
from backend.playoffs.state import PLAYED, build_state
from backend.playoffs.strength import Ratings, season_ratings

N_SIMS = 10_000
TUNE_STEP_DAYS = 7
BINS = np.array([0, 0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98, 1.0])


def _history(season: int) -> pd.DataFrame:
    return load_games([s for s in (season - 1, season) if s >= 2021])


def forward_log_loss(seasons: list[int], precision: float, weight: float) -> float:
    """Mean log loss over every (as-of date, remaining regular-season game) pair."""
    losses = []
    for season in seasons:
        games = _history(season)
        regular = games[(games.season == season) & (games.game_type == "R") & games.status.isin(PLAYED)]
        regular = regular[regular.home_score != regular.away_score]
        teams = np.array(sorted(set(regular.home_team_id) | set(regular.away_team_id)))
        idx = {int(t): i for i, t in enumerate(teams)}
        day, last = regular.official_date.min(), regular.official_date.max()
        while day <= last:
            ahead = regular[regular.official_date >= day]
            ratings = season_ratings(games, season, teams, day, precision, weight)
            p = ratings.p_home(ahead.home_team_id.map(idx).to_numpy(), ahead.away_team_id.map(idx).to_numpy())
            losses.append(log_loss(p, (ahead.home_score > ahead.away_score).to_numpy()))
            day += timedelta(days=TUNE_STEP_DAYS)
    return float(np.concatenate(losses).mean())


def tune(seasons: list[int]) -> tuple[float, float, float]:
    """(precision, weight, log loss) minimizing forward_log_loss."""

    def objective(x):
        return forward_log_loss(seasons, float(np.exp(x[0])), float(expit(x[1])))

    start = np.array([np.log(30.0), 0.0])
    best = minimize(objective, start, method="Nelder-Mead", options={"xatol": 0.02, "fatol": 1e-6})
    return float(np.exp(best.x[0])), float(expit(best.x[1])), float(best.fun)


def actual_reach(games: pd.DataFrame, teams: np.ndarray) -> np.ndarray:
    """(T, len(STAGES)) 0/1: which stages each team actually reached."""
    idx = {int(t): i for i, t in enumerate(teams)}
    out = np.zeros((len(teams), len(STAGES)))
    for k, round_ in enumerate("FDLW"):
        rows = games[games.game_type == round_]
        for t in set(rows.home_team_id) | set(rows.away_team_id):
            out[idx[int(t)], k] = 1
    final = games[(games.game_type == "W") & games.status.isin(PLAYED)]
    winners = np.where(final.home_score > final.away_score, final.home_team_id, final.away_team_id)
    out[idx[int(pd.Series(winners).value_counts().idxmax())], len(STAGES) - 1] = 1
    return out


def _odds_for_day(args) -> list[dict]:
    season, day, n_sims = args
    games = _history(season)
    state = build_state(games, season, day)
    ratings = season_ratings(games, season, state.teams, day)
    flat = Ratings(np.zeros(state.n), np.eye(state.n) * 1e-12, ratings.home)
    rows = []
    for model, r in (("ratings", ratings), ("standings", flat)):
        odds = simulate(state, r, n_sims, np.random.default_rng(day.toordinal()))
        for i, team in enumerate(state.teams):
            rows.append(
                {"season": season, "as_of": day, "model": model, "team_id": int(team)}
                | dict(zip(STAGES, odds.reach[i].round(5)))
            )
    return rows


def daily_odds(seasons: list[int], n_sims: int = N_SIMS, workers: int = 10) -> pd.DataFrame:
    jobs = []
    for season in seasons:
        games = load_games([season])
        day, last = games.official_date.min(), games.official_date.max()
        while day <= last:
            jobs.append((season, day, n_sims))
            day += timedelta(days=1)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = [r for part in pool.map(_odds_for_day, jobs, chunksize=4) for r in part]
    frame = pd.DataFrame(rows)
    truth = []
    for season in seasons:
        teams = np.array(sorted(frame[frame.season == season].team_id.unique()))
        reached = actual_reach(load_games([season]), teams)
        truth.append(pd.DataFrame(reached, columns=[f"y_{s}" for s in STAGES]).assign(season=season, team_id=teams))
    return frame.merge(pd.concat(truth), on=["season", "team_id"], how="left")


def _brier_interval(frame: pd.DataFrame, stage: str, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """90% interval of Brier(ratings) - Brier(standings), resampling team-seasons."""
    wide = frame.pivot_table(index=["season", "team_id", "as_of"], columns="model", values=stage)
    y = frame[frame.model == "ratings"].set_index(["season", "team_id", "as_of"])[f"y_{stage}"].loc[wide.index]
    diff = ((wide.ratings - y) ** 2 - (wide.standings - y) ** 2).groupby(level=[0, 1]).agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(diff), size=(n, len(diff)))
    means = diff["sum"].to_numpy()[pick].sum(1) / diff["count"].to_numpy()[pick].sum(1)
    return float(np.quantile(means, 0.05)), float(np.quantile(means, 0.95))


def summarize(frame: pd.DataFrame) -> str:
    lines = []
    for stage in STAGES:
        y = f"y_{stage}"
        scores = {
            m: (((g[stage] - g[y]) ** 2).mean(), log_loss(g[stage], g[y]).mean()) for m, g in frame.groupby("model")
        }
        lo, hi = _brier_interval(frame, stage)
        lines.append(
            f"{stage}: Brier ratings {scores['ratings'][0]:.4f} standings {scores['standings'][0]:.4f} "
            f"(diff 90% [{lo:+.4f}, {hi:+.4f}]); log loss {scores['ratings'][1]:.4f} vs {scores['standings'][1]:.4f}"
        )
        rated = frame[frame.model == "ratings"]
        binned = rated.groupby(pd.cut(rated[stage], BINS, include_lowest=True), observed=True)
        table = binned.agg(n=(y, "size"), predicted=(stage, "mean"), observed=(y, "mean"))
        lines.append(table.round(3).to_string())
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["tune", "calibrate"])
    parser.add_argument("--seasons", type=int, nargs="+", required=True)
    parser.add_argument("--sims", type=int, default=N_SIMS)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if 2026 in args.seasons:
        raise SystemExit("2026 is the holdout season: no tuning or calibration on it")
    if args.command == "tune":
        precision, weight, loss = tune(args.seasons)
        print(f"precision {precision:.2f} weight {weight:.3f} forward log loss {loss:.5f}")
        return
    frame = daily_odds(args.seasons, args.sims)
    if args.out:
        frame.to_parquet(args.out, index=False)
    for season, part in frame.groupby("season"):
        print(f"== {season}\n{summarize(part)}")
    print(f"== pooled {args.seasons}\n{summarize(frame)}")


if __name__ == "__main__":
    main()
