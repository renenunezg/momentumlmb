"""Score a date's LMB games: build each game's inputs as they stand, simulate,
anchor to Kalshi, price the pick, and return one row per team per game.

Started games are never rescored (freeze_started). Picks wait for posted
lineups, known starters and known bullpen usage; the slate is ranked once
per date (backend.picks)."""

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from backend import strategy
from backend.config import ARTIFACTS_DIR
from backend.etl.games import fetch_games
from backend.etl.pa_frame import load_pa
from backend.etl.workload import load_workload
from backend.markets import kalshi
from backend.markets.probs import anchor_home_prob
from backend.picks import SlateStore, choose_pick, resolve_slate
from backend.sim.baserunner import load_advancement_table, load_out_subtype_table
from backend.sim.bullpen import BullpenQueue, queue_from_workload
from backend.sim.game import GameInputs, simulate_draws
from backend.sim.gbq import load_gb_quartiles
from backend.sim.posteriors import load_posterior_draws, training_max_date
from backend.sim.rules import game_rules
from backend.sim.uncertainty import win_probability_uncertainty
from backend.statsapi import live
from backend.statsapi.league import TEAMS
from backend.weather import forecast_for_games, hr_logit_shift

CURRENT_BUNDLE = ARTIFACTS_DIR / "current"
N_DRAWS = 30
N_PER_DRAW = 333
PROJECTED_LINEUP_DAYS = 30
RUNS_HIST_BINS = 21
FINISHED = {"Final", "Completed Early", "Game Over", "In Progress", "Live", "Manager challenge", "Delayed"}


@dataclass
class Context:
    game: pd.Series
    inputs: GameInputs
    lineup_source: str
    lineup_hash: str
    ready: bool
    home: kalshi.Quote | None
    away: kalshi.Quote | None
    starters: dict[str, dict | None]  # side -> {"id", "name"} for the site's portraits


def games_to_score(games: pd.DataFrame, now: datetime, freeze_started: bool) -> pd.DataFrame:
    """Games still open for a forecast. With freeze_started, a game whose first
    pitch has passed, or whose status says it began, is left alone."""
    playable = games[~games["status"].isin({"Cancelled", "Postponed", "Forfeit"})]
    if not freeze_started:  # replays of past dates only
        return playable
    open_games = playable[~playable["status"].isin(FINISHED)]
    return open_games[open_games["start_time_utc"] > pd.Timestamp(now)]


def lineup_hash(lineups: dict[str, list[int]]) -> str:
    return hashlib.sha1(json.dumps(lineups, sort_keys=True).encode()).hexdigest()[:12]


def projected_lineup(pa: pd.DataFrame, team_id: int, day: date, hitters: list[int]) -> list[int] | None:
    """Top nine active hitters by plate appearances for the team over the last 30 days."""
    recent = pa[(pd.to_datetime(pa.official_date).dt.date < day)]
    recent = recent[pd.to_datetime(recent.official_date).dt.date >= day - timedelta(days=PROJECTED_LINEUP_DAYS)]
    batting = np.where(recent.inning_topbot == "top", recent.away_team_id, recent.home_team_id)
    team_pa = recent[(batting == team_id) & recent.batter.isin(hitters)]
    top = team_pa.batter.value_counts().head(9).index.tolist()
    return [int(x) for x in top] if len(top) == 9 else None


def _quotes_for(day: date, games: pd.DataFrame) -> dict[int, tuple[kalshi.Quote, kalshi.Quote]]:
    try:
        matched = kalshi.match_events(kalshi.events(day), games)
    except Exception as error:  # noqa: BLE001  (no Kalshi market means no pick, never a failed run)
        print(f"Kalshi unavailable: {error}")
        return {}
    out = {}
    for pk, kg in matched.items():
        g = games[games.game_pk == pk].iloc[0]
        try:
            out[pk] = kalshi.quotes(kg, int(g.home_team_id), int(g.away_team_id))
        except Exception as error:  # noqa: BLE001  (one delisted market costs that game its pick, not the run)
            print(f"game {pk}: Kalshi market unavailable: {error}")
    return out


def _weather_for(games: pd.DataFrame) -> dict[int, float]:
    try:
        return forecast_for_games(games).set_index("game_pk").air_density.to_dict()
    except Exception as error:  # noqa: BLE001  (no forecast means no weather shift, never a failed run)
        print(f"Weather unavailable: {error}")
        return {}


def build_contexts(day: date, games: pd.DataFrame, pa: pd.DataFrame, workload: pd.DataFrame) -> list[Context]:
    starters = live.probables(day)
    throws = pa.groupby("pitcher").p_throws.agg(lambda s: s.mode().iloc[0]).to_dict()
    weather = _weather_for(games) if strategy.WEATHER_HR_COEF else {}
    quotes = _quotes_for(day, games)
    contexts = []
    for _, g in games.iterrows():
        pk = int(g.game_pk)
        posted = live.posted_lineups(pk)
        lineups, sources, queues, known, probable_by_side = {}, [], {}, True, {}
        for side in ("home", "away"):
            team_id = int(g[f"{side}_team_id"])
            roster = live.active_roster(team_id, day)
            probable = (starters.get(pk) or {}).get(side)
            starter = probable["id"] if probable else None
            probable_by_side[side] = probable
            if posted[side] is not None:
                lineups[side], source = posted[side], "posted"
            else:
                lineups[side], source = projected_lineup(pa, team_id, day, roster["hitters"]), "projected"
            sources.append(source)
            if starter is None:
                known = False
                queues[side] = BullpenQueue(starter=0, relievers=[0])
            else:
                queues[side] = queue_from_workload(day, team_id, int(starter), workload, roster["pitchers"])
        if lineups["home"] is None or lineups["away"] is None:
            print(f"game {pk}: no lineup available, not scored")
            continue
        hr_shift = hr_logit_shift(int(g.venue_id), weather[pk], strategy.WEATHER_HR_COEF) if pk in weather else 0.0
        inputs = GameInputs(
            home_lineup=np.array(lineups["home"]),
            away_lineup=np.array(lineups["away"]),
            home_queue=queues["home"],
            away_queue=queues["away"],
            venue_id=int(g.venue_id),
            season=int(g.season),
            month=day.month,
            home_throws=throws,
            away_throws=throws,
            rules=game_rules(int(g.season), g.game_type, int(g.scheduled_innings)),
            hr_logit_shift=hr_shift,
        )
        ready = known and sources == ["posted", "posted"] and queues["home"].usage_known and queues["away"].usage_known
        home_q, away_q = quotes.get(pk, (None, None))
        source = "posted" if sources == ["posted", "posted"] else "projected"
        contexts.append(Context(g, inputs, source, lineup_hash(lineups), ready, home_q, away_q, probable_by_side))
    return contexts


def _runs_hist(runs: np.ndarray) -> list[float]:
    counts = np.bincount(np.clip(runs, 0, RUNS_HIST_BINS - 1), minlength=RUNS_HIST_BINS)
    return [round(float(c), 4) for c in counts / counts.sum()]


def _team_code(team_id: int) -> str:
    return TEAMS[team_id].code


def score(
    day: date,
    *,
    freeze_started: bool = True,
    now: datetime | None = None,
    slates: SlateStore,
    bundle: Path = CURRENT_BUNDLE,
    seed: int = 0,
) -> tuple[list[dict], list[tuple]]:
    """Rows to publish for `day` and the Kalshi quotes seen. Refuses a bundle
    trained on or after `day`."""
    now = now or datetime.now(UTC)
    cutoff = date.fromisoformat(training_max_date(bundle))
    if cutoff >= day:
        raise RuntimeError(f"Bundle trained through {cutoff}, not before {day}")
    games = fetch_games(day, day)
    games = games[games.official_date == day]
    todo = games_to_score(games, now, freeze_started)
    if todo.empty:
        return [], []
    pa = load_pa([day.year - 1, day.year], missing_ok=True)
    workload = load_workload([day.year - 1, day.year], missing_ok=True)
    contexts = build_contexts(day, todo, pa, workload)

    candidates = [{"game_pk": int(c.game.game_pk), "home": c.home, "away": c.away} for c in contexts]
    slate = resolve_slate(day, now, games.start_time_utc.dropna().tolist(), candidates, slates)

    draws = load_posterior_draws(bundle, N_DRAWS, np.random.default_rng(seed))
    adv, sub, gbq = load_advancement_table(bundle), load_out_subtype_table(bundle), load_gb_quartiles(bundle)
    rows, quotes_seen = [], []
    for c in contexts:
        # Same inputs, same simulation: an unchanged game produces an identical row, which publish() skips.
        starters = f"{c.inputs.home_queue.starter}|{c.inputs.away_queue.starter}"
        key = f"{seed}|{cutoff}|{int(c.game.game_pk)}|{c.lineup_hash}|{starters}"
        rng = np.random.default_rng(int(hashlib.sha1(key.encode()).hexdigest()[:15], 16))
        result = simulate_draws(rng, draws, adv, sub, gbq, c.inputs, N_PER_DRAW)
        home_win = result.home_runs > result.away_runs
        p_sim = float(home_win.mean())
        per_draw = [float(home_win[i * N_PER_DRAW : (i + 1) * N_PER_DRAW].mean()) for i in range(N_DRAWS)]
        band = win_probability_uncertainty(per_draw, N_PER_DRAW)
        p_market = kalshi.market_home_prob(c.home, c.away)
        p_home = anchor_home_prob(p_sim, p_market, strategy.MARKET_ANCHOR_W_MODEL, strategy.HOME_FIELD_LOGIT)
        pick = choose_pick(int(c.game.game_pk), c.home, c.away, p_market, p_sim, slate, ready=c.ready)
        home_code, away_code = _team_code(int(c.game.home_team_id)), _team_code(int(c.game.away_team_id))
        if pick is not None:
            pick["team"] = home_code if pick["side"] == "home" else away_code
        shared = {"pick": pick} if pick else {}
        shared["slate"] = bool(slate and int(c.game.game_pk) in slate)
        for side, code, opp, runs, p, q in (
            ("home", home_code, away_code, result.home_runs, p_home, c.home),
            ("away", away_code, home_code, result.away_runs, 1 - p_home, c.away),
        ):
            p10, p90 = band.get("p10"), band.get("p90")
            if side == "away" and p10 is not None:
                p10, p90 = 1 - p90, 1 - p10
            rows.append(
                {
                    "game_pk": int(c.game.game_pk),
                    "team": code,
                    "opponent": opp,
                    "is_home": side == "home",
                    "date": day,
                    "start_time": c.game.start_time_utc.to_pydatetime(),
                    "win_prob": round(p, 4),
                    "win_prob_p10": p10,
                    "win_prob_p90": p90,
                    "sim_win_prob": round(p_sim if side == "home" else 1 - p_sim, 4),
                    "market_prob": None if p_market is None else round(p_market if side == "home" else 1 - p_market, 4),
                    "kalshi_bid": None if q is None else q.yes_bid,
                    "kalshi_ask": None if q is None else q.yes_ask,
                    "expected_runs": round(float(runs.mean()), 3),
                    "expected_runs_p10": float(np.percentile(runs, 10)),
                    "expected_runs_p50": float(np.percentile(runs, 50)),
                    "expected_runs_p90": float(np.percentile(runs, 90)),
                    "runs_hist": _runs_hist(runs),
                    "lineup_source": c.lineup_source,
                    "lineup_hash": c.lineup_hash,
                    "prediction_updated_at": now,
                    "prediction_context": {**shared, "starter": c.starters.get(side)},
                }
            )
        for q in (c.home, c.away):
            if q is not None:
                quotes_seen.append((int(c.game.game_pk), q))
    return rows, quotes_seen
