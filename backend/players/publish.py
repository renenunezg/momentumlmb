"""Rows for lmb.player_ratings and lmb.player_rating_model, and the publish.

The table holds the latest season only (about 1,000 rows, ~0.16 MB). Writes go
through backend.db, which refuses them locally unless MOMENTUMLMB_DB_WRITES=1."""

import json

import numpy as np
import pandas as pd

RATING_COLUMNS = [
    "player_id",
    "season",
    "name",
    "team_id",
    "pos",
    "age",
    "pa",
    "bf",
    "batting",
    "baserunning",
    "fielding",
    "positional",
    "pitching",
    "leverage",
    "replacement",
    "war",
    "grade_season",
    "grade",
    "grade_lo",
    "grade_hi",
    "proj_war",
]
# Bytes per row in Postgres (tuple header, fixed columns, a ~16 character
# name) plus the primary key index entry: an estimate for the size check.
EST_ROW_BYTES = 150


def rating_rows(war: pd.DataFrame, grades: pd.DataFrame, season: int) -> pd.DataFrame:
    p = war[war["season"] == season].copy()
    p["fielding"] = p["fld_zone"] + p["throwing"] + p["fld_p"]
    p["positional"] = p["pos_runs"] + p["league_adj"]
    p["pitching"] = p["pitch"]
    p["leverage"] = p["leverage_runs"]
    p = p.merge(
        grades[["player", "grade_season", "grade", "grade_lo", "grade_hi", "proj_war"]], on="player", how="left"
    )
    p["grade_season"] = season + 1
    p = p.rename(columns={"player": "player_id"})
    ledger = p[["batting", "baserunning", "fielding", "positional", "pitching", "leverage", "replacement"]].sum(axis=1)
    if not np.allclose(ledger / p["runs_per_win"], p["war"], atol=1e-6):
        raise ValueError("WAR ledger does not add up to WAR")
    if p["name"].isna().any():
        raise ValueError(f"{int(p['name'].isna().sum())} players have no name")
    out = p[RATING_COLUMNS].copy()
    for c in ("pa", "bf", "season", "grade_season"):
        out[c] = out[c].astype(int)
    for c in ("grade", "grade_lo", "grade_hi", "age"):
        out[c] = out[c].round().astype("Int64")
    for c in (
        "batting",
        "baserunning",
        "fielding",
        "positional",
        "pitching",
        "leverage",
        "replacement",
        "war",
        "proj_war",
    ):
        out[c] = out[c].astype(float).round(2)
    return out


def size_estimate(rows: pd.DataFrame, model_details: dict) -> dict:
    return {
        "rows": int(len(rows)),
        "player_ratings_kb": round(len(rows) * EST_ROW_BYTES / 1024, 1),
        "player_rating_model_kb": round(len(json.dumps(model_details)) / 1024, 1),
    }


def publish(rows: pd.DataFrame, season: int, model_version: str, details: dict) -> None:
    from sqlalchemy import text

    from backend.db import get_engine

    records = rows.astype(object).where(rows.notna(), None).to_dict("records")
    cols = ", ".join(RATING_COLUMNS)
    params = ", ".join(f":{c}" for c in RATING_COLUMNS)
    with get_engine().begin() as conn:
        conn.execute(text("DELETE FROM lmb.player_ratings"))
        conn.execute(text(f"INSERT INTO lmb.player_ratings ({cols}) VALUES ({params})"), records)
        conn.execute(
            text(
                "INSERT INTO lmb.player_rating_model (id, season, model_version, details, updated_at) "
                "VALUES (1, :season, :model_version, CAST(:details AS jsonb), now()) "
                "ON CONFLICT (id) DO UPDATE SET season = EXCLUDED.season, model_version = EXCLUDED.model_version, "
                "details = EXCLUDED.details, updated_at = now()"
            ),
            {"season": season, "model_version": model_version, "details": json.dumps(details)},
        )
