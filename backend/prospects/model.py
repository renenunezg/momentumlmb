"""Ridge-penalized logistic regression, one per role (hitters, pitchers).

About 240 players in the 2006-2019 cohorts became LMB regulars, too few for
mlbmodel's gradient-boosted trees, so the prospect model is the size of its
KATOH-style baseline. The penalty is chosen by player-grouped cross-validation
on the fitting rows, never on validation or holdout rows."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from backend.evaluation.metrics import log_loss

PENALTIES = np.logspace(-3, 2, 21)  # inverse strength C, log-spaced
FOLDS = 5


@dataclass(frozen=True)
class Logit:
    features: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray  # intercept first
    penalty: float

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        x = (frame[list(self.features)].to_numpy(dtype=float) - self.mean) / self.scale
        return 1.0 / (1.0 + np.exp(-(self.weights[0] + x @ self.weights[1:])))

    def contributions(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Per-feature standardized log-odds shifts relative to the training mean."""
        x = (frame[list(self.features)].to_numpy(dtype=float) - self.mean) / self.scale
        return pd.DataFrame(x * self.weights[1:], index=frame.index, columns=list(self.features))

    def to_dict(self) -> dict:
        return {
            "features": list(self.features),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "weights": self.weights.tolist(),
            "penalty": self.penalty,
        }


def _solve(x: np.ndarray, y: np.ndarray, penalty: float) -> np.ndarray:
    xb = np.c_[np.ones(len(x)), x]

    def objective(w):
        z = xb @ w
        loss = np.sum(np.logaddexp(0.0, z) - y * z) + 0.5 / penalty * np.sum(w[1:] ** 2)
        grad = xb.T @ (1.0 / (1.0 + np.exp(-z)) - y)
        grad[1:] += w[1:] / penalty
        return loss, grad

    result = minimize(objective, np.zeros(xb.shape[1]), jac=True, method="L-BFGS-B")
    if not result.success:
        raise RuntimeError(f"Logistic fit did not converge: {result.message}")
    return result.x


def _fit_at(frame: pd.DataFrame, features: tuple[str, ...], penalty: float) -> Logit:
    x = frame[list(features)].to_numpy(dtype=float)
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale == 0] = 1.0
    w = _solve((x - mean) / scale, frame.success.to_numpy(dtype=float), penalty)
    return Logit(tuple(features), mean, scale, w, float(penalty))


def fit(frame: pd.DataFrame, features: tuple[str, ...], seed: int = 0) -> Logit:
    """Fit on matured rows, choosing the penalty by grouped CV log loss."""
    players = frame.player.unique()
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(players, rng.permutation(len(players)) % FOLDS))
    folds = frame.player.map(fold_of).to_numpy()
    cv = []
    for penalty in PENALTIES:
        losses = []
        for k in range(FOLDS):
            model = _fit_at(frame[folds != k], features, penalty)
            held = frame[folds == k]
            losses.append(log_loss(model.predict(held), held.success.to_numpy()))
        cv.append(np.concatenate(losses).mean())
    return _fit_at(frame, features, PENALTIES[int(np.argmin(cv))])
