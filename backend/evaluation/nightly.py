"""Nightly grading of frozen forecasts: log loss, Brier, accuracy, expected
runs error, the Kalshi benchmark on games with a closing quote, the pick
ledger, and calibration. The Kalshi columns cover a different population
from the all-games columns and are never pooled with them."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import text

from backend.db import get_engine
from backend.evaluation.metrics import brier, log_loss

WINDOWS = {"day": 1, "7d": 7, "30d": 30, "season": None}
FINAL = ("Final", "Completed Early", "Game Over")


def _graded(conn, season: int) -> pd.DataFrame:
    return pd.read_sql(
        text(
            """
            SELECT m.game_pk, m.date, m.win_prob, h.expected_runs AS home_exp, a.expected_runs AS away_exp,
                   g.home_score, g.away_score,
                   c.home_yes_bid, c.home_yes_ask, c.away_yes_bid, c.away_yes_ask
            FROM lmb.model_outputs_season m
            JOIN lmb.games g ON g.game_pk = m.game_pk AND g.game_date = m.date
            JOIN lmb.model_outputs_season h ON h.game_pk = m.game_pk AND h.is_home
            JOIN lmb.model_outputs_season a ON a.game_pk = m.game_pk AND NOT a.is_home
            LEFT JOIN lmb.closing_lines c ON c.game_pk = m.game_pk
            WHERE m.is_home AND g.season = :season AND g.status = ANY(:final)
              AND g.home_score IS DISTINCT FROM g.away_score
            """
        ),
        conn,
        params={"season": season, "final": list(FINAL)},
    )


def _kalshi_prob(df: pd.DataFrame) -> pd.Series:
    cols = ["home_yes_bid", "home_yes_ask", "away_yes_bid", "away_yes_ask"]
    q = df[cols].astype(float)
    mid_home = (q.home_yes_bid + q.home_yes_ask) / 2
    mid_away = (q.away_yes_bid + q.away_yes_ask) / 2
    return mid_home / (mid_home + mid_away)


def window_row(df: pd.DataFrame, ledger: pd.DataFrame, name: str, as_of: date, season: int) -> dict:
    y = (df.home_score > df.away_score).astype(int).to_numpy()
    p = df.win_prob.to_numpy()
    k = _kalshi_prob(df)
    has_k = k.notna().to_numpy()
    runs_err = np.concatenate([(df.home_exp - df.home_score).abs(), (df.away_exp - df.away_score).abs()])
    graded = ledger.dropna(subset=["units"])
    return {
        "window": name,
        "as_of": as_of,
        "season": season,
        "n_games": int(len(df)),
        "log_loss": float(log_loss(p, y).mean()) if len(df) else None,
        "brier": float(brier(p, y).mean()) if len(df) else None,
        "accuracy": float(((p > 0.5) == y).mean()) if len(df) else None,
        "kalshi_games": int(has_k.sum()),
        "model_log_loss_on_kalshi": float(log_loss(p[has_k], y[has_k]).mean()) if has_k.any() else None,
        "kalshi_log_loss": float(log_loss(k[has_k], y[has_k]).mean()) if has_k.any() else None,
        "model_brier_on_kalshi": float(brier(p[has_k], y[has_k]).mean()) if has_k.any() else None,
        "kalshi_brier": float(brier(k[has_k], y[has_k]).mean()) if has_k.any() else None,
        "pick_count": int(len(graded)),
        "pick_wins": int(graded.won.sum()) if len(graded) else 0,
        "pick_units": float(graded.units.sum()) if len(graded) else 0.0,
        "expected_runs_mae": float(runs_err.mean()) if len(df) else None,
    }


def evaluate(as_of: date) -> None:
    season = as_of.year
    with get_engine().begin() as conn:
        graded = _graded(conn, season)
        ledger = pd.read_sql(text("SELECT * FROM lmb.bet_ledger_v WHERE season = :s"), conn, params={"s": season})
    graded["date"] = pd.to_datetime(graded.date).dt.date
    ledger["date"] = pd.to_datetime(ledger.date).dt.date if len(ledger) else ledger.get("date")
    rows = []
    for name, days in WINDOWS.items():
        lo = as_of - timedelta(days=days - 1) if days else date(season, 1, 1)
        in_window = graded[(graded.date >= lo) & (graded.date <= as_of)]
        led = ledger[(ledger.date >= lo) & (ledger.date <= as_of)] if len(ledger) else ledger
        rows.append(window_row(in_window, led, name, as_of, season))

    bins = np.clip((graded.win_prob * 10).astype(int), 0, 9)
    y = (graded.home_score > graded.away_score).astype(int)
    calibration = [
        {"season": season, "bin": int(b), "n": int(len(g)), "mean_pred": float(g.win_prob.mean()),
         "mean_obs": float(y[g.index].mean())}
        for b, g in graded.groupby(bins)
    ]  # fmt: skip

    columns = list(rows[0])
    with get_engine().begin() as conn:
        conn.execute(
            text(
                f'INSERT INTO lmb.model_evaluation ("window", {", ".join(columns[1:])}, updated_at) '
                f"VALUES ({', '.join(':' + c for c in columns)}, now()) "
                f'ON CONFLICT ("window") DO UPDATE SET '
                + ", ".join(f"{c} = EXCLUDED.{c}" for c in columns[1:])
                + ", updated_at = now()"
            ),
            rows,
        )
        conn.execute(text("DELETE FROM lmb.model_calibration WHERE season = :s"), {"s": season})
        if calibration:
            conn.execute(
                text("INSERT INTO lmb.model_calibration VALUES (:season, :bin, :n, :mean_pred, :mean_obs, now())"),
                calibration,
            )
    print(f"evaluated {len(graded)} games through {as_of}")
