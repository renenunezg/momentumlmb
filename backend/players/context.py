"""League context of every plate appearance: the expected outcome mix of a
league-average batter against a league-average pitcher for the season, month
and park. It comes from the game model's own pitcher fit (season intercepts,
month effects, per-outcome park effects pooled toward elevation), so the WAR
adjusts for parks with the same posterior the simulator uses."""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from backend.config import ARTIFACTS_DIR
from backend.etl.pa_frame import OUTCOMES, load_pa
from backend.skills import pitcher
from backend.skills.common import NON_REF_IDX, REF_IDX, season_month
from backend.skills.fit import training_frame
from backend.venues import load_venues

CONTEXT_DIR = ARTIFACTS_DIR / "players" / "context"
DRAWS = 500
TUNE = 500
CHAINS = 4
SEED = 20261120


def context_path(last_season: int) -> Path:
    return CONTEXT_DIR / f"through_{last_season}.parquet"


def fit_context(first_season: int, last_season: int) -> Path:
    """Fit the pitcher model on the regular seasons first..last and keep the
    posterior means of the season, month and park logits (long format)."""
    pa = load_pa(list(range(first_season, last_season + 1)))
    through = date(last_season + 1, 1, 1)
    # training_frame keeps seasons from through.year - seasons_back + 1.
    train = training_frame(pa, through, through.year - first_season + 1)
    if sorted(train["season"].unique()) != list(range(first_season, last_season + 1)):
        raise ValueError(f"context training frame does not cover {first_season}-{last_season}")
    train, _ = pitcher.filter_position_player_pitching(train)
    idata, _, _ = pitcher.fit(train, load_venues(), draws=DRAWS, tune=TUNE, chains=CHAINS, seed=SEED)
    post = idata.posterior
    labels = [OUTCOMES[i] for i in NON_REF_IDX]
    frames = []
    for name, dim in (("intercept", "season"), ("park_log", "venue")):
        values = post[name].mean(("chain", "draw")).values
        keys = post[dim].values
        frames.append(pd.DataFrame(values, columns=labels).assign(kind=dim, key=keys.astype(int)))
    month = post["month_effect"].mean(("chain", "draw")).values.T
    frames.append(pd.DataFrame(month, columns=labels).assign(kind="month", key=post["month"].values.astype(int)))
    out = pd.concat(frames, ignore_index=True)
    path = context_path(last_season)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return path


def context_probs(pa: pd.DataFrame, last_season: int) -> np.ndarray:
    """(n_pa, 8) league outcome probabilities at each PA's season, month and park,
    in OUTCOMES order."""
    table = pd.read_parquet(context_path(last_season))
    labels = [OUTCOMES[i] for i in NON_REF_IDX]

    def block(kind: str, keys: np.ndarray) -> np.ndarray:
        t = table[table.kind == kind].set_index("key")[labels]
        missing = set(np.unique(keys)) - set(t.index)
        if missing:
            raise KeyError(f"context fit through {last_season} has no {kind} {sorted(missing)}")
        return t.loc[keys].to_numpy()

    free = (
        block("season", pa["season"].to_numpy())
        + block("month", season_month(pa["official_date"]))
        + block("venue", pa["venue_id"].to_numpy())
    )
    logits = np.zeros((len(pa), len(OUTCOMES)))
    logits[:, NON_REF_IDX] = free
    logits[:, REF_IDX] = 0.0
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(axis=1, keepdims=True)
