"""Forecast scores and paired comparison intervals."""

import numpy as np


def log_loss(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-game log loss."""
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def brier(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2


def paired_bootstrap_ci(loss_a: np.ndarray, loss_b: np.ndarray, level: float = 0.90, n: int = 10_000, seed: int = 0):
    """Interval for mean(loss_a - loss_b) by resampling games."""
    diff = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), size=(n, len(diff)))].mean(axis=1)
    tail = (1 - level) / 2
    return float(np.quantile(means, tail)), float(np.quantile(means, 1 - tail))
