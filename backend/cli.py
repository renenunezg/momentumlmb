"""momentumlmb command line. Heavy imports stay inside each subcommand so
startup is instant."""

import argparse
from datetime import UTC, date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

LMB_TZ = ZoneInfo("America/Mexico_City")


def _today() -> date:
    return datetime.now(UTC).astimezone(LMB_TZ).date()


def main() -> None:
    parser = argparse.ArgumentParser(prog="backend")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("train", "fit skills and tables on everything before the date"),
        ("score", "score the date's open games, publish, record Kalshi quotes and closes"),
        ("nightly", "refresh caches and grade the previous day"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--date", type=date.fromisoformat, default=None)
    replay = sub.add_parser("replay", help="score a past date with no database writes")
    replay.add_argument("--date", type=date.fromisoformat, required=True)
    replay.add_argument("--bundle", type=Path, required=True)
    prospects = sub.add_parser("prospects", help="prospect board: development report, holdout read, or score")
    prospects.add_argument("action", choices=("report", "holdout", "score"))
    prospects.add_argument("--date", type=date.fromisoformat, default=None)
    prospects.add_argument("--write", action="store_true", help="write the scored board to lmb.prospects")
    args = parser.parse_args()

    if args.command == "prospects":
        _prospects(args.action, args.date or _today(), args.write)
        return

    from backend import pipeline

    day = args.date or _today()
    if args.command == "train":
        pipeline.train(day)
    elif args.command == "score":
        pipeline.score_and_publish(day)
    elif args.command == "nightly":
        pipeline.nightly(day)
    elif args.command == "replay":
        rows = pipeline.replay(day, args.bundle)
        for r in rows:
            if r["is_home"]:
                pick = r["prediction_context"].get("pick")
                print(
                    f"{r['game_pk']} {r['opponent']}@{r['team']} home win {r['win_prob']:.3f} "
                    f"(sim {r['sim_win_prob']:.3f}) runs {r['expected_runs']:.2f} lineup {r['lineup_source']}"
                    + (f" pick {pick['team']} {pick['tier']}" if pick else "")
                )
        print(f"{len(rows) // 2} games scored, no database writes")


def _prospects(action: str, day: date, write: bool) -> None:
    import json

    from backend.prospects import evaluate, publish

    if action == "report":
        print(json.dumps(evaluate.development_report(day), indent=2))
    elif action == "holdout":
        print(json.dumps(evaluate.read_holdout(day), indent=2))
    else:
        board = publish.score(day)
        for role, part in board.groupby("role"):
            print(part.head(15)[["role", "role_rank", "name", "age", "team", "level"]].to_string(index=False))
        print(f"{len(board)} prospects scored for {board.snapshot_season.iloc[0]}")
        if write:
            publish.write(board)
            print("written to lmb.prospects")


if __name__ == "__main__":
    main()
