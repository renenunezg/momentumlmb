"""Walk-forward evaluation: refit on everything before a date, then score the
games after it with the lineups, starters and bullpens as they stood, and
never read anything on or after a game's date.

Historical inputs reconstruct what the live scorer sees:
- lineups: the posted batting order (boxscore), which live scoring also waits
  for before publishing a pick;
- starter: the first pitcher of the boxscore;
- bullpen candidates: pitchers who appeared for the team in the previous
  ROSTER_LOOKBACK_DAYS (the Stats API's active roster by date would cost one
  call per team per day).
"""

import gzip
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from backend.config import ARTIFACTS_DIR, CACHE_DIR
from backend.etl.games import load_games
from backend.etl.pa_frame import FINAL_STATES, load_pa
from backend.etl.workload import load_workload
from backend.sim.baserunner import load_advancement_table, load_out_subtype_table
from backend.sim.bullpen import queue_from_workload
from backend.sim.game import GameInputs, simulate_draws
from backend.sim.gbq import load_gb_quartiles
from backend.sim.posteriors import load_posterior_draws
from backend.sim.rules import game_rules
from backend.sim.tables import build_tables
from backend.skills.fit import fit_all

WALKFORWARD_DIR = ARTIFACTS_DIR / "walkforward"
RESULTS_DIR = CACHE_DIR / "walkforward"
ROSTER_LOOKBACK_DAYS = 15
N_DRAWS = 10
N_PER_DRAW = 300
WF_DRAWS = 500
WF_TUNE = 500


@dataclass(frozen=True)
class Setting:
    seasons_back: int
    refit_every_days: int
    weather_hr_coef: float = 0.0  # HR logit shift per 0.01 kg/m3 of air density below the venue median

    @property
    def key(self) -> str:
        return f"sb{self.seasons_back}_r{self.refit_every_days}"


def refit_dates(season_games: pd.DataFrame, every: int) -> list[date]:
    start = season_games.official_date.min()
    end = season_games.official_date.max()
    out, d = [], start
    while d <= end:
        out.append(d)
        d += timedelta(days=every)
    return out


def _feed(game_pk: int) -> dict:
    return json.loads(gzip.decompress((CACHE_DIR / "feeds" / f"{game_pk}.json.gz").read_bytes()))


def _lineup(feed: dict, side: str) -> list[int] | None:
    order = feed["liveData"]["boxscore"]["teams"][side].get("battingOrder", [])
    return [int(x) for x in order[:9]] if len(order) >= 9 else None


def game_inputs(game: pd.Series, workload: pd.DataFrame, throws: dict[int, str], hr_shift: float) -> GameInputs | None:
    feed = _feed(int(game.game_pk))
    box = feed["liveData"]["boxscore"]["teams"]
    lineups = {s: _lineup(feed, s) for s in ("home", "away")}
    if None in lineups.values() or not box["home"].get("pitchers") or not box["away"].get("pitchers"):
        return None
    queues = {}
    for side in ("home", "away"):
        team_id = int(game[f"{side}_team_id"])
        recent = workload[
            (workload.team_id == team_id)
            & (workload.official_date < game.official_date)
            & (workload.official_date >= game.official_date - timedelta(days=ROSTER_LOOKBACK_DAYS))
        ]
        queues[side] = queue_from_workload(
            game.official_date, team_id, int(box[side]["pitchers"][0]), workload, recent.pitcher_id.unique().tolist()
        )
    return GameInputs(
        home_lineup=np.array(lineups["home"]),
        away_lineup=np.array(lineups["away"]),
        home_queue=queues["home"],
        away_queue=queues["away"],
        venue_id=int(game.venue_id),
        season=int(game.season),
        month=pd.Timestamp(game.official_date).month,
        home_throws=throws,
        away_throws=throws,
        rules=game_rules(int(game.season), game.game_type, int(game.scheduled_innings)),
        hr_logit_shift=hr_shift,
    )


def _score_block(args) -> list[dict]:
    bundle, games, workload, throws, hr_shifts, seed = args
    rng = np.random.default_rng(seed)
    draws = load_posterior_draws(bundle, N_DRAWS, rng)
    adv, sub, gbq = load_advancement_table(bundle), load_out_subtype_table(bundle), load_gb_quartiles(bundle)
    rows = []
    for _, game in games.iterrows():
        inputs = game_inputs(game, workload, throws, hr_shifts.get(int(game.game_pk), 0.0))
        if inputs is None:
            continue
        result = simulate_draws(rng, draws, adv, sub, gbq, inputs, N_PER_DRAW)
        rows.append(
            {
                "game_pk": int(game.game_pk),
                "p_home": float((result.home_runs > result.away_runs).mean()),
                "exp_home_runs": float(result.home_runs.mean()),
                "exp_away_runs": float(result.away_runs.mean()),
            }
        )
    return rows


def hr_shifts_for(games: pd.DataFrame, weather: pd.DataFrame, coef: float) -> dict[int, float]:
    """HR logit shift per game from air density below the venue's median."""
    if coef == 0.0:
        return {}
    w = games[["game_pk", "venue_id"]].merge(weather[["game_pk", "air_density"]], on="game_pk")
    median = w.groupby("venue_id").air_density.transform("median")
    return dict(zip(w.game_pk, coef * (median - w.air_density) / 0.01))


def run(season: int, setting: Setting, weather: pd.DataFrame | None = None, workers: int = 6) -> pd.DataFrame:
    """Walk-forward forecasts for every final game of `season` (regular and postseason)."""
    out_path = RESULTS_DIR / f"{season}_{setting.key}_w{setting.weather_hr_coef:g}.parquet"
    if out_path.exists():
        return pd.read_parquet(out_path)
    seasons = list(range(2021, season + 1))
    pa = load_pa(seasons)
    games = load_games([season])
    games = games[games.status.isin(FINAL_STATES)].sort_values("start_time_utc")
    workload = load_workload(seasons)
    throws = pa.groupby("pitcher").p_throws.agg(lambda s: s.mode().iloc[0]).to_dict()
    hr_shifts = hr_shifts_for(games, weather, setting.weather_hr_coef) if weather is not None else {}

    jobs = []
    dates = refit_dates(games[games.game_type == "R"], setting.refit_every_days)
    for i, d in enumerate(dates):
        end = dates[i + 1] if i + 1 < len(dates) else date(season, 12, 31)
        block = games[(games.official_date >= d) & (games.official_date < end)]
        if block.empty:
            continue
        bundle = WALKFORWARD_DIR / f"sb{setting.seasons_back}" / d.isoformat()
        if not (bundle / "diagnostics.json").exists():
            build_tables(pa, d, bundle)
            fit_all(pa, d, bundle, setting.seasons_back, draws=WF_DRAWS, tune=WF_TUNE, enforce_gate=False)
        for j, chunk in enumerate(np.array_split(block, max(1, min(workers, len(block) // 10)))):
            jobs.append((bundle, chunk, workload, throws, hr_shifts, hash((season, d.toordinal(), j)) % 2**32))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = [r for part in pool.map(_score_block, jobs) for r in part]
    result = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out_path, index=False)
    return result


def outcomes(seasons: list[int]) -> pd.DataFrame:
    games = load_games(seasons)
    games = games[games.status.isin(FINAL_STATES)]
    return games.assign(home_win=(games.home_score > games.away_score).astype(int))[
        ["game_pk", "season", "game_type", "official_date", "home_win", "home_score", "away_score"]
    ]
