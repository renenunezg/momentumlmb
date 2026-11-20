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
CLOSE_WINDOW = timedelta(minutes=20)


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


def upsert_workload(workload: pd.DataFrame) -> None:
    rows = [
        {
            "game_pk": int(r.game_pk),
            "game_date": r.official_date,
            "team_id": int(r.team_id),
            "pitcher_id": int(r.pitcher_id),
            "is_starter": bool(r.is_starter),
            "appearance_order": int(r.order),
            "outs": int(r.outs),
            "pitches": int(r.pitches),
            "runs": int(r.runs),
        }
        for r in workload.itertuples(index=False)
    ]
    if not rows:
        return
    with get_engine().begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO lmb.pitcher_workload VALUES (:game_pk, :game_date, :team_id, :pitcher_id, :is_starter,
                  :appearance_order, :outs, :pitches, :runs)
                ON CONFLICT (game_pk, pitcher_id) DO NOTHING
                """
            ),
            rows,
        )


def record_quotes(quotes: list[tuple[int, Quote]]) -> None:
    """Append a quote only when its bid or ask changed since the ticker's last row."""
    if not quotes:
        return
    with get_engine().begin() as conn:
        last = {
            r.ticker: (r.yes_bid, r.yes_ask)
            for r in conn.execute(
                text(
                    """
                    SELECT DISTINCT ON (ticker) ticker, yes_bid, yes_ask FROM lmb.kalshi_quotes
                    WHERE ticker = ANY(:tickers) ORDER BY ticker, fetched_at DESC
                    """
                ),
                {"tickers": [q.ticker for _, q in quotes]},
            )
        }
        rows = []
        for game_pk, q in quotes:
            prev = last.get(q.ticker)
            current = (q.yes_bid, q.yes_ask)
            if prev is not None and tuple(None if v is None else float(v) for v in prev) == current:
                continue
            rows.append(
                {"game_pk": game_pk, "team_id": q.team_id, "ticker": q.ticker, "yes_bid": q.yes_bid,
                 "yes_ask": q.yes_ask, "fetched_at": q.fetched_at}
            )  # fmt: skip
        if rows:
            conn.execute(
                text(
                    "INSERT INTO lmb.kalshi_quotes VALUES "
                    "(:game_pk, :team_id, :ticker, :yes_bid, :yes_ask, :fetched_at)"
                ),
                rows,
            )


def capture_closes(now: datetime) -> int:
    """Write the closing quote (the last quote fetched before first pitch) for
    games starting within the next CLOSE_WINDOW or started in the last one,
    once each. The table's trigger refuses a late or post-start quote."""
    with get_engine().begin() as conn:
        result = conn.execute(
            text(
                """
                INSERT INTO lmb.closing_lines (game_pk, season, start_time, fetched_at, home_yes_bid, home_yes_ask,
                  away_yes_bid, away_yes_ask)
                SELECT g.game_pk, g.season, g.start_time, max(q.fetched_at),
                  (array_agg(q.yes_bid ORDER BY q.fetched_at DESC) FILTER (WHERE q.team_id = g.home_team_id))[1],
                  (array_agg(q.yes_ask ORDER BY q.fetched_at DESC) FILTER (WHERE q.team_id = g.home_team_id))[1],
                  (array_agg(q.yes_bid ORDER BY q.fetched_at DESC) FILTER (WHERE q.team_id = g.away_team_id))[1],
                  (array_agg(q.yes_ask ORDER BY q.fetched_at DESC) FILTER (WHERE q.team_id = g.away_team_id))[1]
                FROM lmb.games g
                JOIN lmb.kalshi_quotes q ON q.game_pk = g.game_pk AND q.fetched_at < g.start_time
                WHERE g.start_time BETWEEN :lo AND :hi
                  AND NOT EXISTS (SELECT 1 FROM lmb.closing_lines c WHERE c.game_pk = g.game_pk)
                GROUP BY g.game_pk, g.season, g.start_time
                """
            ),
            {"lo": now - CLOSE_WINDOW, "hi": now + CLOSE_WINDOW},
        )
        return result.rowcount


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


def publish(rows: list[dict]) -> None:
    """Replace today's rows and upsert the season rows of unstarted games.
    Raises if a row's date is not its game's official date."""
    if not rows:
        return
    payload = []
    for r in rows:
        item = {k: r[k] for k in FORECAST_COLUMNS}
        item["runs_hist"] = json.dumps(item["runs_hist"])
        item["prediction_context"] = json.dumps(item["prediction_context"])
        payload.append(item)
    game_pks = sorted({r["game_pk"] for r in rows})
    day = rows[0]["date"]
    columns = ", ".join(FORECAST_COLUMNS)
    values = ", ".join(f":{c}" for c in FORECAST_COLUMNS)
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in FORECAST_COLUMNS if c not in ("game_pk", "team"))
    with get_engine().begin() as conn:
        official = dict(
            conn.execute(text("SELECT game_pk, game_date FROM lmb.games WHERE game_pk = ANY(:pks)"), {"pks": game_pks})
            .tuples()
            .all()
        )
        for r in rows:
            if official.get(r["game_pk"]) != r["date"]:
                raise RuntimeError(f"Game {r['game_pk']} official date {official.get(r['game_pk'])} != {r['date']}")
        conn.execute(
            text("DELETE FROM lmb.model_outputs WHERE date <> :d OR game_pk = ANY(:pks)"), {"d": day, "pks": game_pks}
        )
        conn.execute(text(f"INSERT INTO lmb.model_outputs ({columns}, updated_at) VALUES ({values}, now())"), payload)
        conn.execute(
            text(
                f"INSERT INTO lmb.model_outputs_season ({columns}, updated_at) VALUES ({values}, now()) "
                f"ON CONFLICT (game_pk, team) DO UPDATE SET {updates}, updated_at = now()"
            ),
            payload,
        )
