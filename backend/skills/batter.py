"""Hierarchical multinomial-logit batter model with a platoon split (ported
from mlbmodel v2/bayesian/batter_skill.py).

Cells are (batter, vs-left-handed-pitcher, season, venue). The season
intercepts and park effects are frozen at the pitcher model's posterior
means, so both halves share one league baseline and one park scale."""

import time

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from backend.skills.common import NON_REF_LABELS, ActorIndex
from backend.skills.pitcher import cell_counts, full_logits

HANDS = ("vs_R", "vs_L")


def build_model(pa: pd.DataFrame, intercept: pd.DataFrame, park: pd.DataFrame) -> tuple[pm.Model, dict]:
    """`intercept` and `park` are indexed by season and venue_id with K_FREE columns."""
    batters = ActorIndex.from_series(pa["batter"])
    frame = pa.assign(vs_lhp=(pa["p_throws"] == "L").astype(np.int64))
    cells, counts = cell_counts(frame, ["batter", "vs_lhp", "season", "venue_id"])
    b_code = batters.encode(cells["batter"].to_numpy())
    lhp = cells["vs_lhp"].to_numpy(dtype=float)[:, None]
    offset = intercept.loc[cells["season"]].to_numpy() + park.loc[cells["venue_id"]].to_numpy()

    coords = {"outcome_free": NON_REF_LABELS, "batter": batters.ids}
    with pm.Model(coords=coords) as model:
        sigma_batter = pm.HalfNormal("sigma_batter", sigma=0.6, dims="outcome_free")
        z_batter = pm.Normal("z_batter", 0.0, 1.0, dims=("batter", "outcome_free"))
        sigma_platoon = pm.HalfNormal("sigma_platoon", sigma=0.3, dims="outcome_free")
        z_platoon = pm.Normal("z_platoon", 0.0, 1.0, dims=("batter", "outcome_free"))
        beta_main = sigma_batter[None, :] * z_batter
        beta_platoon = sigma_platoon[None, :] * z_platoon
        logit_free = offset + beta_main[b_code] + lhp * beta_platoon[b_code]
        pm.Multinomial(
            "outcome_counts", n=counts.sum(axis=1), p=pm.math.softmax(full_logits(logit_free), axis=1), observed=counts
        )
    meta = {"n_pa": int(len(pa)), "n_batters": int(batters.n), "n_cells": int(len(cells))}
    return model, meta


def fit(
    pa: pd.DataFrame,
    intercept: pd.DataFrame,
    park: pd.Series,
    *,
    draws: int,
    tune: int,
    chains: int,
    seed: int,
) -> tuple[az.InferenceData, dict, float]:
    model, meta = build_model(pa, intercept, park)
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


GATE_VARS = ("sigma_batter", "sigma_platoon")
