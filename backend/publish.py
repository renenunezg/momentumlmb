"""Writes to the lmb schema. Every statement names the schema. Writes keep
advancing updated_at so an open games page reloads."""

import json
from datetime import date, datetime, timedelta

import pandas as pd
from sqlalchemy import text

from backend.db import get_engine
from backend.markets.kalshi import Quote
from backend.statsapi.league import TEAMS

FORECAST_COLUMNS = [
    "game_pk", "team", "opponent", "is_home", "date", "start_time", "win_prob", "win_prob_p10", "win_prob_p90",
    "sim_win_prob", "market_prob", "kalshi_bid", "kalshi_ask", "expected_runs", "expected_runs_p10",
    "expected_runs_p50", "expected_runs_p90", "runs_hist", "lineup_source", "lineup_hash",
    "prediction_updated_at", "prediction_context",
]  # fmt: skip
# The refresh job runs every 15 minutes, so the last run before first pitch falls inside this window.
CLOSE_WINDOW = timedelta(minutes=16)


def _team_code(team_id: int) -> str:
    team = TEAMS.get(int(team_id))
    return team.code if team else str(team_id)


def upsert_games(games: pd.DataFrame) -> None:
    rows = [
        {
            "game_pk": int(g.game_pk),
            "game_date": g.official_date,
            "start_time": g.start_time_utc.to_pydatetime() if pd.notna(g.start_time_utc) else None,
            "season": int(g.season),
            "game_type": g.game_type,
            "home_team": _team_code(g.home_team_id),
            "away_team": _team_code(g.away_team_id),
            "home_team_id": int(g.home_team_id),
            "away_team_id": int(g.away_team_id),
            "venue_id": int(g.venue_id),
            "scheduled_innings": int(g.scheduled_innings),
            "status": g.status,
            "home_score": None if pd.isna(g.home_score) else int(g.home_score),
            "away_score": None if pd.isna(g.away_score) else int(g.away_score),
        }
        for g in games.itertuples(index=False)
    ]
    if not rows:
        return
    with get_engine().begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO lmb.games (game_pk, game_date, start_time, season, game_type, home_team, away_team,
                  home_team_id, away_team_id, venue_id, scheduled_innings, status, home_score, away_score, updated_at)
                VALUES (:game_pk, :game_date, :start_time, :season, :game_type, :home_team, :away_team,
                  :home_team_id, :away_team_id, :venue_id, :scheduled_innings, :status, :home_score, :away_score, now())
                ON CONFLICT (game_pk) DO UPDATE SET game_date = EXCLUDED.game_date, start_time = EXCLUDED.start_time,
                  status = EXCLUDED.status, home_score = EXCLUDED.home_score, away_score = EXCLUDED.away_score,
                  scheduled_innings = EXCLUDED.scheduled_innings, venue_id = EXCLUDED.venue_id, updated_at = now()
                WHERE (lmb.games.game_date, lmb.games.start_time, lmb.games.status, lmb.games.home_score,
                       lmb.games.away_score, lmb.games.scheduled_innings, lmb.games.venue_id)
                  IS DISTINCT FROM (EXCLUDED.game_date, EXCLUDED.start_time, EXCLUDED.status, EXCLUDED.home_score,
                       EXCLUDED.away_score, EXCLUDED.scheduled_innings, EXCLUDED.venue_id)
                """
            ),
            rows,
        )


def write_closes(quotes: list[tuple[int, Quote]], games: pd.DataFrame, now: datetime) -> int:
    """Write each game's closing quote once: the quotes seen by the last
    refresh before first pitch (games starting within CLOSE_WINDOW). The
    table's trigger refuses a post-start or late quote."""
    by_game: dict[int, dict[int, Quote]] = {}
    for game_pk, q in quotes:
        by_game.setdefault(game_pk, {})[q.team_id] = q
    rows = []
    for g in games.itertuples(index=False):
        start = g.start_time_utc.to_pydatetime()
        seen = by_game.get(int(g.game_pk), {})
        home, away = seen.get(int(g.home_team_id)), seen.get(int(g.away_team_id))
        if home is None or away is None or not (now < start <= now + CLOSE_WINDOW):
            continue
        rows.append(
            {"game_pk": int(g.game_pk), "season": int(g.season), "start_time": start,
             "fetched_at": min(home.fetched_at, away.fetched_at), "hb": home.yes_bid, "ha": home.yes_ask,
             "ab": away.yes_bid, "aa": away.yes_ask}
        )  # fmt: skip
    if rows:
        with get_engine().begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO lmb.closing_lines (game_pk, season, start_time, fetched_at, home_yes_bid, "
                    "home_yes_ask, away_yes_bid, away_yes_ask) "
                    "VALUES (:game_pk, :season, :start_time, :fetched_at, :hb, :ha, :ab, :aa) "
                    "ON CONFLICT (game_pk) DO NOTHING"
                ),
                rows,
            )
    return len(rows)


class DbSlates:
    """The pick slate store backed by lmb.pick_slates (write-once)."""

    def get(self, day: date) -> list[int] | None:
        with get_engine().begin() as conn:
            row = conn.execute(text("SELECT game_pks FROM lmb.pick_slates WHERE date = :d"), {"d": day}).first()
        return None if row is None else list(row.game_pks)

    def put(self, day: date, ranked_at: datetime, game_pks: list[int]) -> None:
        with get_engine().begin() as conn:
            conn.execute(
                text("INSERT INTO lmb.pick_slates VALUES (:d, :t, :pks) ON CONFLICT (date) DO NOTHING"),
                {"d": day, "t": ranked_at, "pks": game_pks},
            )


class DryRunSlates:
    """Reads the stored slate but never writes one (no-write replays)."""

    def __init__(self):
        self.rows: dict[date, list[int]] = {}

    def get(self, day: date) -> list[int] | None:
        return self.rows.get(day)

    def put(self, day: date, ranked_at: datetime, game_pks: list[int]) -> None:
        self.rows[day] = game_pks


COMPARED = ("date", "start_time", "win_prob", "sim_win_prob", "market_prob", "kalshi_bid", "kalshi_ask",
            "expected_runs", "lineup_source", "lineup_hash", "prediction_context")  # fmt: skip
# Rows for games starting within this margin are dropped before writing, so a
# game whose first pitch passes during the refresh is skipped instead of making
# the freeze trigger roll back every other game's write.
START_MARGIN = timedelta(seconds=60)


def _comparable(row: dict) -> tuple:
    out = []
    for c in COMPARED:
        v = row[c]
        if c == "prediction_context":
            v = json.dumps(v if isinstance(v, dict) else json.loads(v), sort_keys=True)
        elif c == "start_time":
            v = None if v is None else pd.Timestamp(v).tz_convert("UTC").isoformat()
        elif c == "date":
            v = str(v)
        elif v is not None and not isinstance(v, str):
            v = round(float(v), 4)
        out.append(v)
    return tuple(out)


def publish(rows: list[dict]) -> int:
    """Upsert the date's forecasts for unstarted games, writing only games whose
    forecast changed since the last run (keeps database writes to a minimum).
    Rows for games that started (or start within START_MARGIN) are dropped.
    Raises if a row's date is not its game's official date. Returns the number
    of rows written."""
    if not rows:
        return 0
    game_pks = sorted({r["game_pk"] for r in rows})
    day = rows[0]["date"]
    columns = ", ".join(FORECAST_COLUMNS)
    values = ", ".join(f":{c}" for c in FORECAST_COLUMNS)
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in FORECAST_COLUMNS if c not in ("game_pk", "team"))
    with get_engine().begin() as conn:
        games = conn.execute(
            text(
                "SELECT game_pk, game_date, (start_time IS NULL OR start_time <= clock_timestamp() + :margin "
                "OR status IN ('Live', 'In Progress', 'Final', 'Completed Early', 'Game Over')) AS frozen "
                "FROM lmb.games WHERE game_pk = ANY(:pks)"
            ),
            {"pks": game_pks, "margin": START_MARGIN},
        ).all()
        official = {g.game_pk: g.game_date for g in games}
        frozen = {g.game_pk for g in games if g.frozen}
        rows = [r for r in rows if r["game_pk"] not in frozen]
        for r in rows:
            if official.get(r["game_pk"]) != r["date"]:
                raise RuntimeError(f"Game {r['game_pk']} official date {official.get(r['game_pk'])} != {r['date']}")
        stored = {
            (m["game_pk"], m["team"]): _comparable(m)
            for m in conn.execute(
                text(f"SELECT {', '.join(('game_pk', 'team') + COMPARED)} FROM lmb.model_outputs_season "
                     "WHERE game_pk = ANY(:pks)"),
                {"pks": game_pks},
            ).mappings()
        }  # fmt: skip
        changed = {r["game_pk"] for r in rows if stored.get((r["game_pk"], r["team"])) != _comparable(r)}
        payload = []
        for r in rows:
            if r["game_pk"] in changed:
                item = {k: r[k] for k in FORECAST_COLUMNS}
                item["prediction_context"] = json.dumps(item["prediction_context"])
                payload.append(item)
        conn.execute(text("DELETE FROM lmb.model_outputs WHERE date <> :d"), {"d": day})
        if payload:
            for table in ("model_outputs", "model_outputs_season"):
                conn.execute(
                    text(
                        f"INSERT INTO lmb.{table} ({columns}, updated_at) VALUES ({values}, now()) "
                        f"ON CONFLICT (game_pk, team) DO UPDATE SET {updates}, updated_at = now()"
                    ),
                    payload,
                )
    return len(payload)


PLAYOFF_COLUMNS = [
    "season", "as_of", "team", "zone", "wins", "losses", "proj_wins", "proj_losses",
    "p_playoffs", "p_zone_series", "p_zone_final", "p_final", "p_champion",
]  # fmt: skip


def publish_playoff_odds(rows: list[dict]) -> int:
    """Upsert one date's playoff odds (one row per team). A rerun with the same
    results is identical and writes nothing. The rows go in one INSERT ...
    SELECT FROM json_populate_recordset statement, so the statement-level
    site_revalidate trigger fires once. Returns the number of rows written."""
    if not rows:
        return 0
    columns = ", ".join(PLAYOFF_COLUMNS)
    data = [c for c in PLAYOFF_COLUMNS if c not in ("season", "as_of", "team")]
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in data)
    stored = ", ".join(f"lmb.playoff_odds.{c}" for c in data)
    incoming = ", ".join(f"EXCLUDED.{c}" for c in data)
    with get_engine().begin() as conn:
        result = conn.execute(
            text(
                f"INSERT INTO lmb.playoff_odds ({columns}) "
                f"SELECT {columns} FROM json_populate_recordset(NULL::lmb.playoff_odds, CAST(:rows AS json)) "
                f"ON CONFLICT (season, as_of, team) DO UPDATE SET {updates} "
                f"WHERE ({stored}) IS DISTINCT FROM ({incoming})"
            ),
            {"rows": json.dumps([{c: r[c] for c in PLAYOFF_COLUMNS} for r in rows], default=str)},
        )
    return result.rowcount
