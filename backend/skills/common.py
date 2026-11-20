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
MODEL_VERSION = "lmb-sim-v4"
# Calendar months the season effect distinguishes; earlier and later dates
# (spring openers, September playoffs) take the nearest month's effect.
MONTHS = (4, 5, 6, 7, 8)


def season_month(dates) -> np.ndarray:
    return np.clip(pd.to_datetime(pd.Series(dates)).dt.month.to_numpy(), MONTHS[0], MONTHS[-1])


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
