"""Fit the pitcher model (with season levels and parks), then the batter
model against its frozen baseline, and write traces plus diagnostics."""

from datetime import date
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

from backend.etl.pa_frame import OUTCOMES
from backend.skills import batter, pitcher
from backend.skills.common import MODEL_VERSION, NON_REF_IDX, evaluate_gate, woba_weights, write_diagnostics
from backend.venues import load_venues

DRAWS = 1000
TUNE = 1000
CHAINS = 4
SEED = 20261119


def _gate(idata: az.InferenceData, var_names: tuple[str, ...]) -> dict:
    summary = az.summary(idata, var_names=list(var_names))
    n_div = int(idata.sample_stats["diverging"].sum().item())
    rhat, ess = float(summary["r_hat"].max()), float(summary["ess_bulk"].min())
    return {"max_rhat": rhat, "min_ess_bulk": ess, "n_divergent": n_div, "gate_passed": evaluate_gate(rhat, ess)}


def training_frame(pa: pd.DataFrame, through: date, seasons_back: int) -> pd.DataFrame:
    """Regular-season PAs strictly before `through`'s day, from the last
    `seasons_back` seasons up to and including through's season."""
    rows = pa[(pa["game_type"] == "R") & (pd.to_datetime(pa["official_date"]).dt.date < through)]
    first = through.year - seasons_back + 1
    return rows[rows["season"] >= first].reset_index(drop=True)


def fit_all(
    pa: pd.DataFrame,
    through: date,
    out_dir: Path,
    seasons_back: int,
    *,
    draws: int = DRAWS,
    tune: int = TUNE,
    chains: int = CHAINS,
    enforce_gate: bool = True,
) -> Path:
    """Fit on regular-season PAs before `through` and write pitcher.nc,
    batter.nc, woba.json and diagnostics.json. Raises if any gate fails, unless enforce_gate is off
    (walk-forward evaluation records the miss and carries on)."""
    train = training_frame(pa, through, seasons_back)
    if train.empty:
        raise ValueError(f"No training PAs before {through}")
    weights = woba_weights(train)
    woba_free = np.array([weights[OUTCOMES[i]] for i in NON_REF_IDX])
    venues = load_venues()

    pit_pa, dropped = pitcher.filter_position_player_pitching(train)
    pit, pit_meta, pit_s = pitcher.fit(pit_pa, venues, woba_free, draws=draws, tune=tune, chains=chains, seed=SEED)
    pit_diag = _gate(pit, pitcher.GATE_VARS)

    post = pit.posterior
    intercept = pd.DataFrame(post["intercept"].mean(("chain", "draw")).values, index=post["season"].values)
    park = pd.Series(post["park_log"].mean(("chain", "draw")).values, index=post["venue"].values)
    bat, bat_meta, bat_s = batter.fit(
        train, intercept, park, woba_free, draws=draws, tune=tune, chains=chains, seed=SEED
    )
    bat_diag = _gate(bat, batter.GATE_VARS)

    max_date = str(pd.to_datetime(train["official_date"]).max().date())
    for trace in (pit, bat):
        trace.posterior.attrs["training_max_date"] = max_date
        trace.posterior.attrs["model_version"] = MODEL_VERSION
    out_dir.mkdir(parents=True, exist_ok=True)
    pit.to_netcdf(out_dir / "pitcher.nc")
    bat.to_netcdf(out_dir / "batter.nc")
    write_diagnostics(out_dir / "woba.json", weights)
    passed = pit_diag["gate_passed"] and bat_diag["gate_passed"]
    write_diagnostics(
        out_dir / "diagnostics.json",
        {
            "model_version": MODEL_VERSION,
            "through": str(through),
            "seasons_back": seasons_back,
            "training_max_date": max_date,
            "n_pa": int(len(train)),
            "pitcher": {**pit_diag, **pit_meta, "fit_seconds": pit_s, "n_dropped_position_players": len(dropped)},
            "batter": {**bat_diag, **bat_meta, "fit_seconds": bat_s},
            "all_gates_passed": passed,
        },
    )
    if enforce_gate and not passed:
        raise RuntimeError(f"Skill fit failed its convergence gate; see {out_dir / 'diagnostics.json'}")
    return out_dir
