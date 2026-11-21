"""Neutral run values of the simulator's skill posteriors: each batter's
batting runs per plate appearance, and each pitcher's runs saved per batter
faced, against a league-average opponent in an average park and month, at the
latest training season's run level, through a given season's linear weights."""

from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

from backend.etl.pa_frame import OUTCOMES
from backend.skills.common import NON_REF_IDX, REF_IDX

N_DRAWS = 200
LHP_SHARE = 0.28  # LMB share of plate appearances against left-handed pitchers


def _softmax_value(free_logits: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Expected linear weight of free logits (..., K_FREE)."""
    shape = free_logits.shape[:-1] + (len(OUTCOMES),)
    logits = np.zeros(shape)
    logits[..., NON_REF_IDX] = free_logits
    logits[..., REF_IDX] = 0.0
    logits -= logits.max(axis=-1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(axis=-1, keepdims=True)
    return (p * weights).sum(axis=-1)


def neutral_skills(bundle: Path, lw: pd.Series, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(batters, pitchers) with posterior mean and sd of runs per PA above
    average (`bat_post`, `bat_post_sd`; `pitch_post`, `pitch_post_sd`)."""
    weights = lw.reindex(list(OUTCOMES)).to_numpy()
    pit = az.from_netcdf(bundle / "pitcher.nc").posterior.stack(sample=("chain", "draw"))
    bat = az.from_netcdf(bundle / "batter.nc").posterior.stack(sample=("chain", "draw"))
    rng = np.random.default_rng(seed)
    picks = rng.choice(pit.sizes["sample"], size=min(N_DRAWS, pit.sizes["sample"]), replace=False)
    last = int(pit["season"].values.max())
    base = pit["intercept"].sel(season=last).isel(sample=picks).transpose("sample", "outcome_free").values

    roles = az.from_netcdf(bundle / "pitcher.nc").constant_data["pitcher_role"].values.astype(int)
    sig_p = pit["sigma_pitcher"].isel(sample=picks).transpose("sample", "role", "outcome_free").values
    z_p = pit["z_pitcher"].isel(sample=picks).transpose("sample", "pitcher", "outcome_free").values
    beta_p = sig_p[:, roles, :] * z_p
    sig_b = bat["sigma_batter"].isel(sample=picks).transpose("sample", "outcome_free").values
    z_b = bat["z_batter"].isel(sample=picks).transpose("sample", "batter", "outcome_free").values
    sig_l = bat["sigma_platoon"].isel(sample=picks).transpose("sample", "outcome_free").values
    z_l = bat["z_platoon"].isel(sample=picks).transpose("sample", "batter", "outcome_free").values
    beta_b = sig_b[:, None, :] * z_b + LHP_SHARE * sig_l[:, None, :] * z_l

    league = _softmax_value(base, weights)[:, None]
    bat_runs = _softmax_value(base[:, None, :] + beta_b, weights) - league
    pitch_runs = league - _softmax_value(base[:, None, :] + beta_p, weights)
    batters = pd.DataFrame(
        {"player": bat["batter"].values.astype(np.int64), "bat_post": bat_runs.mean(0), "bat_post_sd": bat_runs.std(0)}
    )
    pitchers = pd.DataFrame(
        {
            "player": pit["pitcher"].values.astype(np.int64),
            "pitch_post": pitch_runs.mean(0),
            "pitch_post_sd": pitch_runs.std(0),
        }
    )
    return batters, pitchers
