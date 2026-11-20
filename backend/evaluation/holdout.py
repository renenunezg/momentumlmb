"""The one-time 2026 holdout read (AGENTS.md). Run once, after every setting
is committed; nothing may be tuned on its output.

Populations, reported separately and never pooled:
- all final 2026 games: sim vs home-field and team-strength baselines;
- games with a matched BetExplorer close: sim (shifted, as published before
  anchoring) vs the de-vigged average close;
- games with a matched 2026 Kalshi quote: sim vs the Kalshi midpoint.
The launch rule: picks are off if the 90% paired bootstrap interval of
per-game log loss (sim minus team-strength) lies wholly above zero."""

import json

import numpy as np
import pandas as pd

from backend import strategy
from backend.config import CACHE_DIR, HOLDOUT_SEASON
from backend.etl.games import load_games
from backend.evaluation import baselines
from backend.evaluation.metrics import brier, log_loss, paired_bootstrap_ci
from backend.evaluation.walkforward import Setting, outcomes, run
from backend.markets.betexplorer import CLOSES
from backend.markets.kalshi import KALSHI_2026
from backend.markets.probs import devig_pair, logit, sigmoid

REPORT = CACHE_DIR / "holdout_2026.json"


def _block(name: str, p_model: np.ndarray, p_other: np.ndarray, y: np.ndarray, other: str) -> dict:
    lm, lo = log_loss(p_model, y), log_loss(p_other, y)
    lo_ci, hi_ci = paired_bootstrap_ci(lm, lo)
    return {
        "population": name,
        "games": int(len(y)),
        "sim_log_loss": float(lm.mean()),
        f"{other}_log_loss": float(lo.mean()),
        "sim_brier": float(brier(p_model, y).mean()),
        f"{other}_brier": float(brier(p_other, y).mean()),
        "log_loss_diff_ci90": [lo_ci, hi_ci],
    }


def read_holdout() -> dict:
    if REPORT.exists():
        raise RuntimeError(f"The holdout was already read ({REPORT}); it is read once")
    setting = Setting(strategy.SEASONS_BACK, strategy.REFIT_EVERY_DAYS, strategy.WEATHER_HR_COEF)
    forecasts = run(HOLDOUT_SEASON, setting)
    games = load_games(list(range(2021, HOLDOUT_SEASON + 1)))
    history = games.merge(outcomes(list(range(2021, HOLDOUT_SEASON + 1)))[["game_pk", "home_win"]], on="game_pk")
    df = forecasts.merge(history, on="game_pk")
    df = df[df.home_score != df.away_score]
    df["p_sim"] = [sigmoid(logit(p) + strategy.HOME_FIELD_LOGIT) for p in df.p_home]
    df["p_home_field"] = [baselines.home_field(history, d) for d in df.official_date]
    df["p_team"] = [baselines.team_strength(history, g) for _, g in df.iterrows()]
    y = df.home_win.to_numpy()

    report = {"setting": setting.__dict__, "home_field_logit": strategy.HOME_FIELD_LOGIT, "blocks": []}
    report["blocks"].append(
        _block("all games vs home field", df.p_sim.to_numpy(), df.p_home_field.to_numpy(), y, "home_field")
    )
    team = _block("all games vs team strength", df.p_sim.to_numpy(), df.p_team.to_numpy(), y, "team_strength")
    report["blocks"].append(team)

    closes = pd.read_parquet(CLOSES).merge(df, on="game_pk")
    p_close = np.array([devig_pair(1 / h, 1 / a) for h, a in zip(closes.home_decimal, closes.away_decimal)])
    report["blocks"].append(
        _block("games with a BetExplorer close", closes.p_sim.to_numpy(), p_close, closes.home_win.to_numpy(), "close")
    )

    kalshi = (
        pd.read_parquet(KALSHI_2026)
        .dropna(subset=["home_bid", "home_ask", "away_bid", "away_ask"])
        .merge(df, on="game_pk")
    )
    mid_h = (kalshi.home_bid + kalshi.home_ask) / 2
    mid_a = (kalshi.away_bid + kalshi.away_ask) / 2
    report["blocks"].append(
        _block(
            "games with a Kalshi quote",
            kalshi.p_sim.to_numpy(),
            (mid_h / (mid_h + mid_a)).to_numpy(),
            kalshi.home_win.to_numpy(),
            "kalshi",
        )
    )

    low, _ = team["log_loss_diff_ci90"]
    report["picks_enabled"] = not (low > 0)
    REPORT.write_text(json.dumps(report, indent=2, default=str))
    return report
