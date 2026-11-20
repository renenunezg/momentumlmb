"""Small one-parameter fits on walk-forward forecasts (development seasons
only): home field on the sim, the sim's weight beyond the market for pick
pricing, and the log-loss-optimal anchor weight (reported, not adopted)."""

import numpy as np
from scipy.optimize import minimize, minimize_scalar


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _offset_logistic(offset: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """MLE of b in P(y) = sigmoid(offset + b * x), with its standard error."""

    def nll(b):
        z = offset + b[0] * x
        return -(y * z - np.logaddexp(0, z)).sum()

    b = float(minimize(nll, [0.0]).x[0])
    z = offset + b * x
    p = 1 / (1 + np.exp(-z))
    info = float((p * (1 - p) * x * x).sum())
    return b, 1 / np.sqrt(info)


def home_field_logit(p_sim: np.ndarray, home_win: np.ndarray) -> tuple[float, float]:
    """Constant logit shift on the sim's home probability."""
    return _offset_logistic(_logit(p_sim), np.ones(len(p_sim)), np.asarray(home_win, dtype=float))


def sim_lean(p_market: np.ndarray, p_sim_shifted: np.ndarray, home_win: np.ndarray) -> tuple[float, float]:
    """The sim's coefficient beyond the de-vigged market logit (mlbmodel's PICK_MODEL_WEIGHT)."""
    lm = _logit(p_market)
    return _offset_logistic(lm, _logit(p_sim_shifted) - lm, np.asarray(home_win, dtype=float))


def optimal_anchor_weight(p_market: np.ndarray, p_sim_shifted: np.ndarray, home_win: np.ndarray) -> float:
    lm, ls, y = _logit(p_market), _logit(p_sim_shifted), np.asarray(home_win, dtype=float)

    def loss(w):
        z = w * ls + (1 - w) * lm
        return -(y * z - np.logaddexp(0, z)).mean()

    return float(minimize_scalar(loss, bounds=(0.0, 1.0), method="bounded").x)


def hr_density_coefficient(pa, weather) -> tuple[float, float]:
    """Home-run logit change per 0.01 kg/m3 of air thinner than the venue-season
    median, from regular-season PAs (offset: the venue-season HR rate)."""
    d = pa[pa.game_type == "R"].merge(weather[["game_pk", "air_density"]], on="game_pk")
    x = ((d.groupby(["venue_id", "season"]).air_density.transform("median") - d.air_density) / 0.01).to_numpy()
    y = (d.outcome == "HR").astype(float).to_numpy()
    rate = d.groupby(["venue_id", "season"]).outcome.transform(lambda s: (s == "HR").mean()).clip(1e-4, 1 - 1e-4)
    return _offset_logistic(np.log(rate / (1 - rate)).to_numpy(), x, y)
