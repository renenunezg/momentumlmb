"""Market probabilities and the blends that consume them (ported from
mlbmodel v2/markets/probs.py, moneyline only)."""

import math


def logit(p: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def devig_pair(p_home_raw: float, p_away_raw: float) -> float:
    """Proportional de-vig of a two-way price pair; returns the home probability."""
    return p_home_raw / (p_home_raw + p_away_raw)


def anchor_home_prob(p_home_sim: float, p_market_home: float | None, w_model: float, hfa_logit: float) -> float:
    """Published win probability: the home-field-shifted sim blended with the
    market on the logit scale. No market, and the shifted sim passes through."""
    shifted = logit(p_home_sim) + hfa_logit
    if p_market_home is None:
        return sigmoid(shifted)
    return sigmoid(w_model * shifted + (1.0 - w_model) * logit(p_market_home))


def pick_prob(p_market: float, p_sim: float, lean: float) -> float:
    """Probability used to price a pick: the market plus a small lean toward the sim."""
    return sigmoid(logit(p_market) + lean * (logit(p_sim) - logit(p_market)))
