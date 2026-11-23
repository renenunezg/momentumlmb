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

MODEL_VERSION = "lmb-prospects-v2"
# Holdout read 2026-11-23 (cache/prospects/holdout_v2.json, read once): both roles pass. Hitters (full) log loss
# .254 vs base rate .356 (interval -.141 to -.064), AUC .880; pitchers (full) .285 vs .324 (-.075 to -.004),
# AUC .808 (vs baseline .750). Levels still run high in the independent era: mean p .160 vs .106 observed
# (hitters), .165 vs .088 (pitchers), so the board shows ranks only.
PUBLISHED_ROLES: tuple[str, ...] = ("H", "P")
COLUMNS = [
    "snapshot_season", "player_id", "name", "role", "age", "team_id", "team", "level", "mexican_born",
    "role_rank", "raw_score", "model_version", "scored_at", "details",
]  # fmt: skip
HIT_STAT_COLS = ("g", "pa", "ab", "h", "d2", "d3", "hr", "bb", "hbp", "so")
PIT_STAT_COLS = ("g", "gs", "bf", "outs", "h", "hr", "bb", "hbp", "so")
SOURCE_ORDER = ("lmb", "milb", "lmp")


def board_season(today: date) -> int:
    return today.year if today >= date(today.year, 10, 1) else today.year - 1


def _team_names(team_ids) -> dict[int, str]:
    ids = sorted({int(t) for t in team_ids if pd.notna(t)})
    names = {}
    for i in range(0, len(ids), 100):
        for team in get("/v1/teams", teamIds=",".join(map(str, ids[i : i + 100])))["teams"]:
            names[team["id"]] = team.get("shortName") or team["name"]
    return names


def _stat_dict(row: pd.Series, cols: tuple[str, ...]) -> dict[str, int]:
    return {col: int(row[col]) for col in cols}


def _player_details(
    board_role: pd.DataFrame,
    contribs: pd.DataFrame,
    frame_lines: pd.DataFrame,
    bio: pd.DataFrame,
    season: int,
    role: str,
) -> list[dict]:
    """Build the expandable card payload for one role's board rows."""
    group = "hitting" if role == "H" else "pitching"
    opp_col = "pa" if role == "H" else "bf"
    stat_cols = HIT_STAT_COLS if role == "H" else PIT_STAT_COLS

    # Center contributions against this role's current board mean so each shift
    # explains why a player sits above or below the average current prospect.
    centered = contribs - contribs.mean(axis=0)
    driver_groups: list[tuple[str, tuple[str, ...]]] = [
        ("age", ("age", "age_rel")),
        ("milb_level", ("milb_level",)),
        ("playing_time", ("log_window", "log_lmb", "log_lmp", "log_milb")),
        ("kbb_rel", ("kbb_rel",)),
    ]
    if role == "H":
        driver_groups.append(("woba_rel", ("woba_rel",)))

    # Within-role percentiles (0 to 100, oriented higher = better).
    pcts = {
        "age_rel": (board_role["age_rel"].rank(ascending=False, pct=True) * 100).round().astype(int),
        "kbb_rel": (board_role["kbb_rel"].rank(ascending=(role == "P"), pct=True) * 100).round().astype(int),
    }
    if role == "H":
        pcts["woba_rel"] = (board_role["woba_rel"].rank(ascending=True, pct=True) * 100).round().astype(int)

    ids = set(board_role["player"].astype(int))
    role_lines = frame_lines[(frame_lines["group"] == group) & frame_lines["player"].isin(ids)]
    win_opp = role_lines[role_lines["season"].between(season - 2, season)].groupby("player")[opp_col].sum().to_dict()
    cur_lines = role_lines[role_lines["season"] == season]
    cur_totals = cur_lines.groupby("player")[list(stat_cols)].sum()
    cur_splits = cur_lines.groupby(["player", "source"])[list(stat_cols)].sum()

    payloads: list[dict] = []
    for idx, row in board_role.iterrows():
        pid = int(row["player"])
        person = bio.loc[pid] if pid in bio.index else None
        pos = person["position"] if person is not None and pd.notna(person["position"]) else None
        throws = person["throws"] if person is not None and pd.notna(person["throws"]) else None

        c_row = centered.loc[idx]
        drivers = []
        for key, cols in driver_groups:
            present = [c for c in cols if c in c_row.index]
            if present:
                val = round(float(c_row[present].sum()), 2)
                if val != 0.0:
                    drivers.append([key, val])
        drivers.sort(key=lambda item: abs(item[1]), reverse=True)

        splits = []
        opp_by_source = {"lmb": 0, "lmp": 0, "milb": 0}
        if pid in cur_totals.index:
            tot = cur_totals.loc[pid]
            season_opp = int(tot[opp_col])
            season_stats = _stat_dict(tot, stat_cols) if season_opp > 0 else None
            for src in SOURCE_ORDER:
                if (pid, src) in cur_splits.index:
                    srow = cur_splits.loc[(pid, src)]
                    sopp = int(srow[opp_col])
                    opp_by_source[src] = sopp
                    if sopp > 0:
                        entry = {"source": src}
                        entry.update(_stat_dict(srow, stat_cols))
                        splits.append(entry)
        else:
            season_opp = 0
            season_stats = None

        payloads.append(
            {
                "position": str(pos) if pos is not None else None,
                "throws": str(throws) if throws is not None else None,
                "milb_level": int(row["milb_level"]),
                "age_rel": round(float(row["age_rel"]), 1),
                "window_opp": int(win_opp.get(pid, 0)),
                "season_opp": season_opp,
                "opp_by_source": opp_by_source,
                "drivers": drivers,
                "percentiles": {k: int(v.loc[idx]) for k, v in pcts.items()},
                "season_stats": season_stats,
                "splits": splits,
            }
        )
    return payloads


def score(today: date) -> pd.DataFrame:
    season = board_season(today)
    frame_lines, people, team_games, played = lines.load_inputs(today, refresh_open=True)
    snaps = snapshots.build(frame_lines, people, team_games, list(range(evaluate.DEVELOPMENT[0], season + 1)), played)
    bio = people.set_index("player")
    board = []
    for role in PUBLISHED_ROLES:
        part = snaps[snaps.role == role]
        fitted = model.fit(part[part.success.notna()], evaluate.features_for(role, evaluate.CHOICE[role]))
        now = part[part.season == season].copy()
        now["raw_score"] = fitted.predict(now)
        contribs = fitted.contributions(now)
        now["details"] = _player_details(now, contribs, frame_lines, bio, season, role)
        board.append(now)
    if not board:
        raise RuntimeError("No prospect role passed the holdout rule; nothing to score")
    out = pd.concat(board, ignore_index=True)
    names = bio.full_name
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
            "details": out.details,
        }
    )[COLUMNS].sort_values(["role", "role_rank"])


def write(board: pd.DataFrame) -> None:
    """Replace the board season's rows in one statement, so the statement-level
    site_revalidate trigger fires once instead of once per row."""
    seasons = board.snapshot_season.unique()
    if len(seasons) != 1:
        raise ValueError(f"A board holds one snapshot season, got {seasons}")
    cols = ", ".join(COLUMNS)
    payload = board[COLUMNS].to_json(orient="records", date_format="iso", force_ascii=False)
    with get_engine().begin() as conn:
        conn.execute(text("DELETE FROM lmb.prospects WHERE snapshot_season = :s"), {"s": int(seasons[0])})
        conn.execute(
            text(
                f"INSERT INTO lmb.prospects ({cols}) "
                f"SELECT {cols} FROM json_populate_recordset(NULL::lmb.prospects, CAST(:rows AS json))"
            ),
            {"rows": payload},
        )
