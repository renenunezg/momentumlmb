"""LMB award boards: validate, show or publish one season's frozen board, grade it.

    python -m backend.awards.run validate --season 2026
    python -m backend.awards.run board --season 2026          (no database writes)
    python -m backend.awards.run publish --season 2026 [--after-announcement]
    python -m backend.awards.run grade --season 2026

A board is fitted on every year before its season and reads no winner of its own
season. Each award shows the model's probabilities only when they beat a one-input
logit on its baseline stat in leave-one-year-out log loss; otherwise the one-input
logit; and a ranking without probabilities when neither beats a uniform pick. The
rule was fixed before the first validation (2026-11-20).
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from backend.awards import data, features, model

MODEL_VERSION = "lmb-awards-v1"
FIRST_YEAR = {
    "MVP": 2005,
    "POY": 2005,
    "REL": 2005,
    "ROY": features.FIRST_ROOKIE_YEAR,
    "MOY": features.FIRST_MANAGER_YEAR,
}
BOARD_SIZE = 10


def _validation_path(season: int):
    return data.CACHE / f"validation_{MODEL_VERSION}_{season}.json"


def validate(season: int) -> dict:
    """Leave-one-year-out results on every year before the season, cached per version and season."""
    path = _validation_path(season)
    if path.exists():
        return json.loads(path.read_text())
    out = {}
    for award, columns in features.FEATURES.items():
        races = features.races(award, FIRST_YEAR[award], season - 1)
        out[award] = model.validate(races, columns.index(features.BASELINE[award]))
    path.write_text(json.dumps(out))
    return out


def choice(result: dict) -> str:
    """Which board an award earns from its validation: 'model', 'baseline' or 'rank'."""
    best = min(result["log_loss"], result["baseline_log_loss"])
    if best >= result["uniform_log_loss"]:
        return "rank"
    return "model" if result["log_loss"] <= result["baseline_log_loss"] else "baseline"


def board(season: int) -> tuple[pd.DataFrame, list[dict]]:
    """The season's board and one model row per award. Reads winners of earlier years only."""
    checks = validate(season)
    rows, models = [], []
    for award, columns in features.FEATURES.items():
        kind = choice(checks[award])
        used = columns if kind == "model" else [features.BASELINE[award]]
        races = features.races(award, FIRST_YEAR[award], season - 1)
        index = [columns.index(c) for c in used]
        races = [features.Race(r.key, r.year, r.pool, r.x[:, index], r.y) for r in races]
        pool = features.pool(award, season)
        x = pool[used].to_numpy(float)
        if kind == "rank":
            probability, order = None, np.argsort(-x[:, 0], kind="stable")
        else:
            fit = model.fit(races)
            probability = fit.win(x)
            order = np.argsort(-probability, kind="stable")
        for rank, i in enumerate(order[:BOARD_SIZE], start=1):
            line = pool.iloc[i]
            rows.append(
                {
                    "season": season,
                    "award": award,
                    "player_id": int(line.player_id),
                    "player_name": line.player_name,
                    "team": line.team,
                    "rank": rank,
                    "win_probability": None if probability is None else round(float(probability[i]), 4),
                    "stat_line": features.STAT_LINE[award](line),
                }
            )
        summary = {k: v for k, v in checks[award].items() if k != "rows"}
        models.append(
            {
                "season": season,
                "award": award,
                "board": kind,
                "inputs": used,
                "coefficients": None if kind == "rank" else dict(zip(used, np.round(fit.beta / fit.scale, 6).tolist())),
                "penalty": None if kind == "rank" else fit.penalty,
                "candidates": len(pool),
                "validation": summary,
                "validation_races": [
                    {k: r[k] for k in ("key", "candidates", "winner_probability", "hit")} for r in checks[award]["rows"]
                ],
            }
        )
    return pd.DataFrame(rows), models


def publish(season: int, after_announcement: bool) -> None:
    from sqlalchemy import text

    from backend.db import get_engine

    rows, models = board(season)
    as_of = datetime.now(UTC)
    with get_engine().begin() as conn:
        for m in models:
            conn.execute(
                text(
                    "INSERT INTO lmb.award_models (season, award, as_of, model_version, after_announcement, "
                    "backtest) VALUES (:season, :award, :as_of, :version, :after, CAST(:backtest AS jsonb))"
                ),
                {
                    "season": season,
                    "award": m["award"],
                    "as_of": as_of,
                    "version": MODEL_VERSION,
                    "after": after_announcement,
                    "backtest": json.dumps(m),
                },
            )
        conn.execute(
            text(
                "INSERT INTO lmb.award_boards (season, award, player_id, player_name, team, rank, "
                "win_probability, stat_line) VALUES (:season, :award, :player_id, :player_name, :team, "
                ":rank, :win_probability, :stat_line)"
            ),
            rows.to_dict("records"),
        )
    print(f"Published {len(rows)} board rows for {season}")


def grade(season: int) -> None:
    """Set won once per award whose winner the Stats API lists; a winner off the board is added unranked."""
    from sqlalchemy import text

    from backend.db import get_engine

    with get_engine().begin() as conn:
        for award in features.FEATURES:
            winners = data.recipients(award)
            winners = winners[winners.year.eq(season)]
            if winners.empty:
                print(f"{award} {season}: no winner listed yet, skipped")
                continue
            shown = {
                r.player_id
                for r in conn.execute(
                    text("SELECT player_id FROM lmb.award_boards WHERE season = :s AND award = :a"),
                    {"s": season, "a": award},
                )
            }
            for w in winners.itertuples():
                if w.player_id not in shown:
                    conn.execute(
                        text(
                            "INSERT INTO lmb.award_boards (season, award, player_id, player_name, "
                            "stat_line, won) VALUES (:s, :a, :p, :n, '', true)"
                        ),
                        {"s": season, "a": award, "p": int(w.player_id), "n": w.player_name},
                    )
            conn.execute(
                text(
                    "UPDATE lmb.award_boards SET won = (player_id = ANY(:ids)) "
                    "WHERE season = :s AND award = :a AND won IS NULL"
                ),
                {"s": season, "a": award, "ids": [int(i) for i in winners.player_id]},
            )
            print(f"{award} {season}: graded")


def _print(rows: pd.DataFrame, models: list[dict]) -> None:
    for m in models:
        v = m["validation"]
        print(
            f"\n{m['award']} ({m['board']}, {m['candidates']} candidates): held out {v['races']} races, "
            f"top pick won {v['hits']}, top 3 {v['top3']}, baseline leader {v['baseline_hits']:.1f}; "
            f"log loss {v['log_loss']:.3f} vs baseline {v['baseline_log_loss']:.3f} "
            f"[{v['interval_90'][0]:+.3f}, {v['interval_90'][1]:+.3f}] vs uniform {v['uniform_log_loss']:.3f}"
        )
        for r in rows[rows.award.eq(m["award"])].itertuples():
            p = "" if r.win_probability is None or pd.isna(r.win_probability) else f"{r.win_probability:6.1%}"
            print(f"  {r.rank:2d}. {p:>6} {r.player_name} ({r.team}): {r.stat_line}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="backend.awards.run")
    parser.add_argument("command", choices=["validate", "board", "publish", "grade"])
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument(
        "--after-announcement", action="store_true", help="label a board frozen after the league announced its winners"
    )
    args = parser.parse_args()
    if args.command == "validate":
        for award, result in validate(args.season).items():
            print(award, choice(result), {k: v for k, v in result.items() if k != "rows"})
    elif args.command == "board":
        _print(*board(args.season))
        print("\nNo database writes")
    elif args.command == "publish":
        publish(args.season, args.after_announcement)
    else:
        grade(args.season)


if __name__ == "__main__":
    main()
