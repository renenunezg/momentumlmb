"""Score the current prospect board and write it to lmb.prospects.

The board season is the latest finished summer (LMB and minor leagues); the
LMP winter that starts that autumn counts as it stands. Each role's model is
the spec chosen on validation, fitted on every matured snapshot, and a role is
written only if it passed the holdout rule (PUBLISHED_ROLES).

The board is ranks only: role_rank orders each role. The model output is kept
as raw_score for context and is not a calibrated chance (see PUBLISHED_ROLES);
it is never to be shown as a probability, and it is not recalibrated on the
2021-2022 holdout cohorts."""

from datetime import UTC, date, datetime

import pandas as pd
from sqlalchemy import text

from backend.db import get_engine
from backend.prospects import evaluate, lines, model, snapshots
from backend.statsapi.client import get

MODEL_VERSION = "lmb-prospects-v1"
# Holdout read 2026-11-20 (cache/prospects/holdout.json, read once): both roles pass. Hitters (full) log loss
# .359 vs base rate .477 (interval -.158 to -.077), AUC .833; pitchers (baseline) .315 vs .342 (-.045 to -.010),
# AUC .749. Levels run high in the independent era: mean p .256 vs .147 observed (hitters), .205 vs .081
# (pitchers), so the output ranks well but overstates the chance: the board shows ranks only.
PUBLISHED_ROLES: tuple[str, ...] = ("H", "P")
COLUMNS = [
    "snapshot_season", "player_id", "name", "role", "age", "team_id", "team", "level", "mexican_born",
    "role_rank", "raw_score", "model_version", "scored_at",
]  # fmt: skip


def board_season(today: date) -> int:
    return today.year if today >= date(today.year, 10, 1) else today.year - 1


def _team_names(team_ids) -> dict[int, str]:
    ids = sorted({int(t) for t in team_ids if pd.notna(t)})
    names = {}
    for i in range(0, len(ids), 100):
        for team in get("/v1/teams", teamIds=",".join(map(str, ids[i : i + 100])))["teams"]:
            names[team["id"]] = team.get("shortName") or team["name"]
    return names


def score(today: date) -> pd.DataFrame:
    season = board_season(today)
    frame_lines, people, team_games, played = lines.load_inputs(today, refresh_open=True)
    snaps = snapshots.build(frame_lines, people, team_games, list(range(evaluate.DEVELOPMENT[0], season + 1)), played)
    board = []
    for role in PUBLISHED_ROLES:
        part = snaps[snaps.role == role]
        fitted = model.fit(part[part.success.notna()], evaluate.features_for(role, evaluate.CHOICE[role]))
        now = part[part.season == season].copy()
        now["raw_score"] = fitted.predict(now)
        board.append(now)
    if not board:
        raise RuntimeError("No prospect role passed the holdout rule; nothing to score")
    out = pd.concat(board, ignore_index=True)
    names = people.set_index("player").full_name
    teams = _team_names(out.team)
    out["role_rank"] = out.groupby("role").raw_score.rank(ascending=False, method="first").astype(int)
    scored_at = datetime.now(UTC)
    return pd.DataFrame(
        {
            "snapshot_season": season,
            "player_id": out.player.astype(int),
            "name": out.player.map(names),
            "role": out.role,
            "age": out.age.round(1),
            "team_id": out.team.astype("Int64"),
            "team": out.team.map(lambda t: teams.get(int(t)) if pd.notna(t) else None),
            "level": out.source_now,
            "mexican_born": out.mexican.astype(bool),
            "raw_score": out.raw_score.round(4),
            "role_rank": out.role_rank,
            "model_version": MODEL_VERSION,
            "scored_at": scored_at,
        }
    )[COLUMNS].sort_values(["role", "role_rank"])


def write(board: pd.DataFrame) -> None:
    """Replace the board season's rows in one transaction."""
    seasons = board.snapshot_season.unique()
    if len(seasons) != 1:
        raise ValueError(f"A board holds one snapshot season, got {seasons}")
    rows = board.astype(object).where(board.notna(), None).to_dict("records")
    with get_engine().begin() as conn:
        conn.execute(text("DELETE FROM lmb.prospects WHERE snapshot_season = :s"), {"s": int(seasons[0])})
        conn.execute(
            text(f"INSERT INTO lmb.prospects ({', '.join(COLUMNS)}) VALUES ({', '.join(':' + c for c in COLUMNS)})"),
            rows,
        )
