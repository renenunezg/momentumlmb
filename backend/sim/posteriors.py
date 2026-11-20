"""Posterior draws of the skill models, laid out for vectorized simulation
(ported from mlbmodel v2/simulator/posteriors.py).

Unknown players fall back to a zero offset (league average). An unknown
season uses the pooled league mean; an unknown venue uses its altitude
prior (elevation slope times elevation)."""

import json
from dataclasses import dataclass
from pathlib import Path

import arviz as az
import numpy as np

from backend.etl.pa_frame import OUTCOMES
from backend.skills.common import MODEL_VERSION, NON_REF_IDX
from backend.venues import venue


@dataclass(frozen=True)
class PosteriorDraw:
    intercept: dict[int, np.ndarray]  # season -> (K_FREE,)
    intercept_mu: np.ndarray  # (K_FREE,) pooled mean, for unseen seasons
    batter_offset: np.ndarray  # (n_b + 1, K_FREE), last row zeros
    platoon_offset: np.ndarray  # (n_b + 1, K_FREE), last row zeros
    pitcher_offset: np.ndarray  # (n_p + 1, K_FREE), last row zeros
    park: dict[int, float]  # venue_id -> park logit coefficient
    elevation_slope: float
    batter_ids: np.ndarray
    pitcher_ids: np.ndarray
    woba_free: np.ndarray  # (K_FREE,)

    def encode_batter(self, ids: np.ndarray) -> np.ndarray:
        return _encode(ids, self.batter_ids)

    def encode_pitcher(self, ids: np.ndarray) -> np.ndarray:
        return _encode(ids, self.pitcher_ids)

    def season_intercept(self, season: int) -> np.ndarray:
        return self.intercept.get(season, self.intercept_mu)

    def park_log(self, venue_id: int) -> float:
        if venue_id in self.park:
            return self.park[venue_id]
        return self.elevation_slope * float(venue(venue_id).elevation_m) / 1000.0


def _encode(ids: np.ndarray, sorted_ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(ids, dtype=np.int64)
    if len(sorted_ids) == 0:
        return np.zeros(len(ids), dtype=np.int64)
    pos = np.searchsorted(sorted_ids, ids)
    clipped = np.clip(pos, 0, len(sorted_ids) - 1)
    hit = (pos < len(sorted_ids)) & (sorted_ids[clipped] == ids)
    return np.where(hit, clipped, len(sorted_ids)).astype(np.int64)


def training_max_date(bundle: Path) -> str:
    return json.loads((bundle / "diagnostics.json").read_text())["training_max_date"]


def load_posterior_draws(bundle: Path, n_draws: int, rng: np.random.Generator) -> list[PosteriorDraw]:
    pit = az.from_netcdf(bundle / "pitcher.nc")
    bat = az.from_netcdf(bundle / "batter.nc")
    for trace in (pit, bat):
        if trace.posterior.attrs.get("model_version") != MODEL_VERSION:
            raise ValueError(f"{bundle} holds a different model version; refit")
    if pit.posterior.attrs.get("training_max_date") != bat.posterior.attrs.get("training_max_date"):
        raise ValueError(f"{bundle} mixes training cutoffs; refit")
    weights = json.loads((bundle / "woba.json").read_text())
    woba_free = np.array([weights[OUTCOMES[i]] for i in NON_REF_IDX])

    ps = pit.posterior.stack(sample=("chain", "draw"))
    bs = bat.posterior.stack(sample=("chain", "draw"))
    n = ps.sizes["sample"]
    if n_draws > n:
        raise ValueError(f"{n_draws} draws requested, {n} available")
    picks = rng.choice(n, size=n_draws, replace=False)

    roles = pit.constant_data["pitcher_role"].values.astype(int)
    seasons = ps["season"].values.astype(int)
    venues = ps["venue"].values.astype(int)
    pitcher_ids = ps["pitcher"].values.astype(np.int64)
    batter_ids = bs["batter"].values.astype(np.int64)
    zeros = np.zeros((1, len(NON_REF_IDX)))

    draws = []
    for i in picks:
        p = ps.isel(sample=i)
        b = bs.isel(sample=i)
        beta_pitcher = p["sigma_pitcher"].values[roles] * p["z_pitcher"].values
        beta_main = b["sigma_batter"].values[None, :] * b["z_batter"].values
        beta_platoon = b["sigma_platoon"].values[None, :] * b["z_platoon"].values
        draws.append(
            PosteriorDraw(
                intercept={int(s): p["intercept"].values[j] for j, s in enumerate(seasons)},
                intercept_mu=p["intercept_mu"].values,
                batter_offset=np.vstack([beta_main, zeros]),
                platoon_offset=np.vstack([beta_platoon, zeros]),
                pitcher_offset=np.vstack([beta_pitcher, zeros]),
                park={int(v): float(p["park_log"].values[j]) for j, v in enumerate(venues)},
                elevation_slope=float(p["elevation_slope"].values),
                batter_ids=batter_ids,
                pitcher_ids=pitcher_ids,
                woba_free=woba_free,
            )
        )
    return draws
