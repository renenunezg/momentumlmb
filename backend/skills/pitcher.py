"""Hierarchical multinomial-logit pitcher model with season run levels and
venue park effects pooled toward altitude (ported from mlbmodel
v2/bayesian/pitcher_skill.py).

Cells are (pitcher, season, venue). Each cell's free logits are
    intercept[season] + beta_pitcher[pitcher] + park[venue] * woba
with OUT as the reference. Season intercepts are partially pooled around a
league mean, so a season with few games leans on recent seasons. Park
effects live here rather than in a separate stage because LMB parks are
extreme (run factors about 0.63 to 1.33): fitting them jointly keeps a
Mexico City pitcher's skill from absorbing his home park. Each park's prior
mean is a fitted slope on elevation in km."""

import time

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from backend.etl.pa_frame import OUTCOMES
from backend.skills.common import (
    K_FREE,
    NON_REF_IDX,
    NON_REF_LABELS,
    REF_IDX,
    ActorIndex,
    encode_outcomes,
    league_log_p,
)

ROLES = ("SP", "RP")
POSITION_PLAYER_PA_THRESHOLD = 50
PARK_BOUND = 2.0


def classify_roles(pa: pd.DataFrame) -> pd.Series:
    """Per-pitcher SP/RP from the majority of game appearances as the first-inning pitcher."""
    inning1 = pa[pa["inning"] == 1]
    side = np.where(inning1["inning_topbot"] == "top", "away", "home")
    starters = (
        inning1.assign(side=side)
        .groupby(["game_pk", "side"])["pitcher"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
        .dropna()
        .astype("int64")
    )
    starts = starters.value_counts()
    games = pa.groupby("pitcher")["game_pk"].nunique()
    share = (starts / games).fillna(0.0)
    return pd.Series(np.where(share >= 0.5, "SP", "RP"), index=games.index, name="role")


def filter_position_player_pitching(pa: pd.DataFrame) -> tuple[pd.DataFrame, set[int]]:
    batter_pa = pa.groupby("batter").size()
    real_batters = set(batter_pa[batter_pa >= POSITION_PLAYER_PA_THRESHOLD].index)
    drop = set(pa["pitcher"].unique()) & real_batters
    return pa[~pa["pitcher"].isin(drop)].reset_index(drop=True), drop


def cell_counts(pa: pd.DataFrame, keys: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    """Long-format multinomial cells: one row per distinct key combination,
    with an (n_cells, 8) count matrix."""
    codes = encode_outcomes(pa["outcome"])
    frame = pa[keys].copy()
    frame["code"] = codes
    table = frame.groupby([*keys, "code"]).size().unstack("code", fill_value=0)
    table = table.reindex(columns=range(len(OUTCOMES)), fill_value=0)
    return table.index.to_frame(index=False), table.to_numpy(dtype=np.int64)


def full_logits(logit_free):
    import pytensor.tensor as pt

    n = logit_free.shape[0]
    cols = []
    free = iter(range(K_FREE))
    for i in range(len(OUTCOMES)):
        if i == REF_IDX:
            cols.append(pt.zeros((n, 1)))
        else:
            fi = next(free)
            cols.append(logit_free[:, fi : fi + 1])
    return pt.concatenate(cols, axis=1)


def build_model(pa: pd.DataFrame, venues: pd.DataFrame, woba_free: np.ndarray) -> tuple[pm.Model, dict]:
    log_p = league_log_p(encode_outcomes(pa["outcome"]))
    intercept_prior = log_p[NON_REF_IDX] - log_p[REF_IDX]

    pitchers = ActorIndex.from_series(pa["pitcher"])
    seasons = np.sort(pa["season"].unique())
    venue_ids = np.sort(pa["venue_id"].unique())
    elevation_km = venues.loc[venue_ids, "elevation_m"].to_numpy(dtype=float) / 1000.0

    cells, counts = cell_counts(pa, ["pitcher", "season", "venue_id"])
    p_code = pitchers.encode(cells["pitcher"].to_numpy())
    s_code = np.searchsorted(seasons, cells["season"].to_numpy())
    v_code = np.searchsorted(venue_ids, cells["venue_id"].to_numpy())

    roles = classify_roles(pa).reindex(pitchers.ids).fillna("RP")
    role_slot = (roles.to_numpy() == "RP").astype(np.int64)  # 0 = SP, 1 = RP

    coords = {
        "outcome_free": NON_REF_LABELS,
        "pitcher": pitchers.ids,
        "season": seasons,
        "venue": venue_ids,
        "role": list(ROLES),
    }
    with pm.Model(coords=coords) as model:
        pm.Data("pitcher_role", role_slot, dims="pitcher")
        mu = pm.Normal("intercept_mu", mu=intercept_prior, sigma=0.5, dims="outcome_free")
        tau = pm.HalfNormal("intercept_tau", sigma=0.2, dims="outcome_free")
        z_season = pm.Normal("z_season", 0.0, 1.0, dims=("season", "outcome_free"))
        intercept = pm.Deterministic(
            "intercept", mu[None, :] + tau[None, :] * z_season, dims=("season", "outcome_free")
        )

        sigma_pitcher = pm.HalfNormal("sigma_pitcher", sigma=0.6, dims=("role", "outcome_free"))
        z_pitcher = pm.Normal("z_pitcher", 0.0, 1.0, dims=("pitcher", "outcome_free"))
        beta_pitcher = sigma_pitcher[role_slot] * z_pitcher

        elevation_slope = pm.Normal("elevation_slope", 0.0, 0.5)
        park_sigma = pm.HalfNormal("park_sigma", 0.25)
        z_park = pm.Normal("z_park", 0.0, 1.0, dims="venue")
        park = pm.Deterministic("park_log", elevation_slope * elevation_km + park_sigma * z_park, dims="venue")

        logit_free = intercept[s_code] + beta_pitcher[p_code] + park[v_code][:, None] * woba_free[None, :]
        pm.Multinomial(
            "outcome_counts", n=counts.sum(axis=1), p=pm.math.softmax(full_logits(logit_free), axis=1), observed=counts
        )

    meta = {
        "n_pa": int(len(pa)),
        "n_pitchers": int(pitchers.n),
        "n_sp": int((roles == "SP").sum()),
        "n_rp": int((roles == "RP").sum()),
        "seasons": seasons.tolist(),
        "venues": venue_ids.tolist(),
        "n_cells": int(len(cells)),
    }
    return model, meta


def fit(
    pa: pd.DataFrame, venues: pd.DataFrame, woba_free: np.ndarray, *, draws: int, tune: int, chains: int, seed: int
) -> tuple[az.InferenceData, dict, float]:
    model, meta = build_model(pa, venues, woba_free)
    t0 = time.time()
    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=0.95,
            nuts_sampler="numpyro",
            nuts_sampler_kwargs={"chain_method": "vectorized"},
            random_seed=seed,
            progressbar=False,
        )
    return idata, meta, time.time() - t0


GATE_VARS = ("intercept_mu", "intercept_tau", "intercept", "sigma_pitcher", "elevation_slope", "park_sigma", "park_log")
