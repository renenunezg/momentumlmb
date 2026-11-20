"""Shared pieces of the skill layer (ported from mlbmodel v2/bayesian/_common.py)."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from backend.etl.pa_frame import OUTCOMES

REF_IDX = OUTCOMES.index("OUT")
NON_REF_IDX = [i for i in range(len(OUTCOMES)) if i != REF_IDX]
NON_REF_LABELS = [OUTCOMES[i] for i in NON_REF_IDX]
K_FREE = len(NON_REF_IDX)
MODEL_VERSION = "lmb-sim-v1"


@dataclass(frozen=True)
class ActorIndex:
    ids: np.ndarray
    n: int

    def encode(self, raw: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.ids, raw)

    @classmethod
    def from_series(cls, s: pd.Series) -> "ActorIndex":
        ids = np.sort(s.unique())
        return cls(ids=ids, n=len(ids))


def encode_outcomes(outcome_col: pd.Series) -> np.ndarray:
    cat_to_code = {o: i for i, o in enumerate(OUTCOMES)}
    return outcome_col.map(cat_to_code).to_numpy(dtype=np.int32)


def league_log_p(outcome_codes: np.ndarray) -> np.ndarray:
    counts = np.bincount(outcome_codes, minlength=len(OUTCOMES))
    p = np.clip(counts / counts.sum(), 1e-6, None)
    return np.log(p)


def write_diagnostics(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def evaluate_gate(rhat_max: float, ess_min: float, threshold_rhat: float = 1.01, threshold_ess: float = 400) -> bool:
    return bool(rhat_max <= threshold_rhat and ess_min > threshold_ess)


def _state(frame: pd.DataFrame) -> np.ndarray:
    return (
        frame["on_1b"].astype(int).to_numpy()
        + 2 * frame["on_2b"].astype(int).to_numpy()
        + 4 * frame["on_3b"].astype(int).to_numpy()
    )


def woba_weights(pa: pd.DataFrame) -> dict[str, float]:
    """LMB linear weights: each outcome's average change in run expectancy
    (plus runs on the play) from a base-out run-expectancy table built on the
    same PAs, expressed relative to an out, so OUT and K sit at zero as in
    FanGraphs wOBA. Used only to tilt logits by a park coefficient."""
    df = pa.sort_values(["game_pk", "inning", "inning_topbot", "at_bat_number"]).copy()
    df["state"] = _state(df)
    keys = ["game_pk", "inning", "inning_topbot"]
    half_final = df.groupby(keys)["post_bat_score"].transform("max")
    df["runs_rest"] = half_final - df["bat_score"]
    re = df.groupby(["state", "outs_when_up"])["runs_rest"].mean()
    df["re_now"] = re.reindex(pd.MultiIndex.from_arrays([df["state"], df["outs_when_up"]])).to_numpy()
    nxt = df.groupby(keys)["re_now"].shift(-1).fillna(0.0)
    df["value"] = nxt - df["re_now"] + (df["post_bat_score"] - df["bat_score"])
    by_outcome = df.groupby("outcome")["value"].mean()
    out_value = float(by_outcome["OUT"])
    weights = {o: float(by_outcome[o]) - out_value for o in OUTCOMES}
    weights["OUT"] = 0.0
    weights["K"] = 0.0
    return weights
