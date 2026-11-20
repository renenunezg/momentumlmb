"""Production commands: train, score, nightly, and the no-write replay."""

import shutil
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from backend import strategy
from backend.config import ARTIFACTS_DIR
from backend.etl.games import build_games_cache, fetch_games
from backend.etl.pa_frame import build_pa_cache, load_pa
from backend.etl.workload import build_workload_cache
from backend.scoring import CURRENT_BUNDLE, score


def _seasons(day: date) -> list[int]:
    return [s for s in range(day.year - strategy.SEASONS_BACK + 1, day.year + 1) if s >= 2021 and s != 2020]


def update_caches(day: date) -> None:
    """Games, plate appearances and workloads for the seasons the model reads.
    Cached feeds are reused, so only new final games are fetched."""
    seasons = _seasons(day)
    build_games_cache(seasons)
    played = [s for s in seasons if s < day.year or day >= date(day.year, 4, 1)]
    build_pa_cache(played)
    build_workload_cache(played)


def train(day: date, out: Path = CURRENT_BUNDLE) -> Path:
    """Fit the skills and tables on everything before `day` into a fresh bundle,
    then swap it in for the current one."""
    from backend.sim.tables import build_tables
    from backend.skills.fit import fit_all

    update_caches(day)
    pa = load_pa(_seasons(day), missing_ok=True)
    staging = out.with_name(out.name + "_staging")
    shutil.rmtree(staging, ignore_errors=True)
    build_tables(pa, day, staging)
    fit_all(pa, day, staging, strategy.SEASONS_BACK)
    shutil.rmtree(out, ignore_errors=True)
    staging.rename(out)
    return out


def score_and_publish(day: date) -> None:
    from backend import publish

    games = fetch_games(day - timedelta(days=3), day + timedelta(days=1))
    publish.upsert_games(games)
    rows, quotes = score(day, slates=publish.DbSlates())
    written = publish.publish(rows)
    closes = publish.write_closes(quotes, games[games.official_date == day], datetime.now(UTC))
    print(f"{day}: {len(rows)} rows scored, {written} written, {len(quotes)} quotes seen, {closes} closes captured")


def nightly(day: date) -> None:
    from backend import publish
    from backend.evaluation.nightly import evaluate

    update_caches(day)
    publish.upsert_games(fetch_games(day - timedelta(days=3), day))
    evaluate(day - timedelta(days=1))


def replay(day: date, bundle: Path) -> list[dict]:
    """Score a past date as of 30 minutes before its first pitch, with no
    database writes and no stored slate (the write guard stays on)."""
    from backend.publish import DryRunSlates

    games = fetch_games(day, day)
    first = games.start_time_utc.min().to_pydatetime()
    rows, _ = score(day, freeze_started=False, now=first - timedelta(minutes=30), slates=DryRunSlates(), bundle=bundle)
    return rows


def bundle_for(day: date) -> Path:
    return ARTIFACTS_DIR / "replay" / day.isoformat()
