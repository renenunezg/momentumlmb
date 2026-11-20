"""Plate-appearance outcome logits (ported from mlbmodel v2/simulator/pa_sim.py):

    logits[k] = intercept[season, k] + month[m, k] + batter[b, k]
              + vs_lhp * platoon[b, k] + pitcher[p, k] + park[venue, k]

with OUT spliced back in at logit 0."""

import numpy as np

from backend.etl.pa_frame import OUTCOMES
from backend.sim.posteriors import PosteriorDraw
from backend.skills.common import REF_IDX

HR_IDX = OUTCOMES.index("HR")


def build_full_logits(free: np.ndarray) -> np.ndarray:
    return np.insert(free, REF_IDX, 0.0, axis=1)


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def sample_categorical(rng: np.random.Generator, probs: np.ndarray) -> np.ndarray:
    cdf = np.cumsum(probs, axis=1)
    u = rng.random(size=(probs.shape[0], 1))
    return (u < cdf).argmax(axis=1).astype(np.int64)


def pa_logits_batch(
    draw: PosteriorDraw,
    batter_ids: np.ndarray,
    pitcher_ids: np.ndarray,
    vs_lhp: np.ndarray,
    season: int,
    month: int,
    venue_id: int,
) -> np.ndarray:
    b = draw.encode_batter(batter_ids)
    p = draw.encode_pitcher(pitcher_ids)
    free = (
        draw.season_intercept(season)[None, :]
        + draw.month_log(month)[None, :]
        + draw.batter_offset[b]
        + np.asarray(vs_lhp, dtype=float)[:, None] * draw.platoon_offset[b]
        + draw.pitcher_offset[p]
        + draw.park_log(venue_id)[None, :]
    )
    return build_full_logits(free)
