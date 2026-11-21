"""A conditional logit per award: a softmax over each race's candidates, fitted on winners.

The LMB publishes no votes, so each race contributes its winner alone. The ridge
penalty comes from leaving one training year out in turn, and the reported skill
from leaving each year out of the whole procedure (penalty choice included), so a
held-out year never tunes its own forecast.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_softmax, softmax

from backend.awards.features import Race

PENALTIES = np.logspace(-2, 2, 9)
SEED = 20261120


@dataclass(frozen=True)
class Fit:
    beta: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    penalty: float

    def logits(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.center) / self.scale) @ self.beta

    def win(self, x: np.ndarray) -> np.ndarray:
        return softmax(self.logits(x))


def _won(races: list[Race]) -> list[Race]:
    return [r for r in races if r.y.sum() > 0]


def _coefficients(races: list[Race], center: np.ndarray, scale: np.ndarray, penalty: float) -> np.ndarray:
    scaled = [((r.x - center) / scale, r.y) for r in _won(races)]

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        value, gradient = penalty / 2 * beta @ beta, penalty * beta
        for x, y in scaled:
            logits = x @ beta
            value -= y @ log_softmax(logits)
            gradient = gradient + x.T @ (softmax(logits) - y)
        return value, gradient

    result = minimize(objective, np.zeros(scale.size), jac=True, method="L-BFGS-B")
    if not result.success and "ABNORMAL" in str(result.message):
        raise RuntimeError(f"Award fit failed: {result.message}")
    return result.x


def _fit_at(races: list[Race], penalty: float) -> Fit:
    stacked = np.vstack([r.x for r in races])
    center, deviation = stacked.mean(axis=0), stacked.std(axis=0)
    scale = np.where(deviation > 0, deviation, 1.0)
    return Fit(_coefficients(races, center, scale, penalty), center, scale, float(penalty))


def _log_loss(fit: Fit, race: Race) -> float:
    return float(-race.y @ log_softmax(fit.logits(race.x)))


def _held_out_loss(races: list[Race], penalty: float) -> float:
    total = 0.0
    for year in sorted({r.year for r in races}):
        model = _fit_at([r for r in races if r.year != year], penalty)
        total += sum(_log_loss(model, r) for r in _won(races) if r.year == year)
    return total


def fit(races: list[Race]) -> Fit:
    best = min(PENALTIES, key=lambda p: _held_out_loss(races, p))
    return _fit_at(races, best)


def _leader_share(values: np.ndarray, winners: set[int]) -> float:
    """A tie for the lead counts as an even share of a hit."""
    leaders = set(np.flatnonzero(values == values.max()))
    return len(leaders & winners) / len(leaders)


def validate(races: list[Race], baseline_column: int) -> dict:
    """Leave each year out of everything: the model, a one-input logit on the baseline column,
    the baseline leader, and a uniform pick, scored on the same races."""
    rows = []
    single = [Race(r.key, r.year, r.pool, r.x[:, [baseline_column]], r.y) for r in races]
    for year in sorted({r.year for r in races}):
        model = fit([r for r in races if r.year != year])
        base = fit([r for r in single if r.year != year])
        for race, one in ((r, s) for r, s in zip(races, single) if r.year == year):
            if race.y.sum() == 0:
                raise ValueError(f"Winner outside the {race.key} pool")
            p = model.win(race.x)
            winners = set(np.flatnonzero(race.y))
            rows.append(
                {
                    "key": race.key,
                    "year": year,
                    "candidates": len(race.y),
                    "log_loss": _log_loss(model, race),
                    "baseline_log_loss": _log_loss(base, one),
                    "uniform_log_loss": float(np.log(len(race.y))),
                    "hit": int(p.argmax()) in winners,
                    "top3": bool(winners & set(np.argsort(-p)[:3])),
                    "baseline_hit": _leader_share(race.x[:, baseline_column], winners),
                    "winner_probability": float(p[list(winners)].sum()),
                    "top_probability": float(p.max()),
                }
            )
    diff = np.array([r["log_loss"] - r["baseline_log_loss"] for r in rows])
    years = np.array([r["year"] for r in rows])
    unique = np.unique(years)
    rng = np.random.default_rng(SEED)
    draws = [np.mean(np.concatenate([diff[years == y] for y in rng.choice(unique, unique.size)])) for _ in range(2000)]

    def mean(k: str) -> float:
        return float(np.mean([r[k] for r in rows]))

    return {
        "races": len(rows),
        "years": [int(y) for y in unique],
        "hits": sum(r["hit"] for r in rows),
        "top3": sum(r["top3"] for r in rows),
        "baseline_hits": sum(r["baseline_hit"] for r in rows),
        "log_loss": mean("log_loss"),
        "baseline_log_loss": mean("baseline_log_loss"),
        "uniform_log_loss": mean("uniform_log_loss"),
        "log_loss_minus_baseline": float(diff.mean()),
        "interval_90": [float(np.quantile(draws, 0.05)), float(np.quantile(draws, 0.95))],
        "mean_winner_probability": mean("winner_probability"),
        "mean_top_probability": mean("top_probability"),
        "rows": rows,
    }
