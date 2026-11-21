"""The LMB run scale, fitted per regular season from the play-by-play: a
run-expectancy table over the 24 base-out states, the run value of every
event (RE24), linear weights per plate-appearance outcome, and runs per win."""

import numpy as np
import pandas as pd

from backend.etl.pa_frame import OUTCOMES


def _half_inning_keys(ev: pd.DataFrame) -> pd.Series:
    return ev["game_pk"].astype(str) + "-" + ev["inning"].astype(str) + ev["half"]


def add_runs_to_end(ev: pd.DataFrame) -> pd.DataFrame:
    """Runs scored from the start of each event to the end of its half-inning,
    and whether the half-inning counts for the run-expectancy table (complete,
    three outs, inside the scheduled innings, not a walk-off bottom half)."""
    ev = ev.sort_values(["game_pk", "at_bat_index", "play_index"]).copy()
    key = _half_inning_keys(ev)
    total = ev.groupby(key)["runs"].transform("sum")
    before = ev.groupby(key)["runs"].cumsum() - ev["runs"]
    ev["runs_to_end"] = total - before
    last_outs = ev.groupby(key)["outs_after"].transform("last")
    last_inning = ev["inning"] >= ev["scheduled_innings"]
    ev["re_ok"] = (
        (last_outs == 3) & (ev["inning"] <= ev["scheduled_innings"]) & ~(last_inning & (ev["half"] == "bottom"))
    )
    return ev


def re_table(ev: pd.DataFrame) -> pd.Series:
    """Mean runs to the end of the half-inning by (outs, bases) at the start of
    each event, one season's regular-season events."""
    rows = ev[ev["re_ok"] & (ev["outs_before"] < 3)]
    table = rows.groupby(["outs_before", "bases_before"])["runs_to_end"].mean()
    full = pd.MultiIndex.from_product([range(3), range(8)], names=["outs_before", "bases_before"])
    if not full.isin(table.index).all():
        raise ValueError("run-expectancy table is missing base-out states")
    return table.reindex(full)


def add_re24(ev: pd.DataFrame, re: pd.Series) -> pd.DataFrame:
    """Run value of each event: expectancy after minus before plus runs scored."""
    before = re.reindex(pd.MultiIndex.from_arrays([ev["outs_before"].clip(upper=2), ev["bases_before"]])).to_numpy()
    after = re.reindex(pd.MultiIndex.from_arrays([ev["outs_after"].clip(upper=2), ev["bases_after"]])).to_numpy()
    after = np.where(ev["outs_after"].to_numpy() >= 3, 0.0, after)
    return ev.assign(re24=after - before + ev["runs"].to_numpy())


def linear_weights(pa_events: pd.DataFrame) -> pd.Series:
    """Mean RE24 by outcome, in OUTCOMES order."""
    lw = pa_events.groupby("outcome")["re24"].mean()
    return lw.reindex(list(OUTCOMES))


def runs_per_win(runs_per_team_game: float) -> float:
    """Pythagenpat: the exponent is (runs per game, both teams)^0.287, and a
    win costs 4r / exponent runs at r runs per team per game."""
    exponent = (2.0 * runs_per_team_game) ** 0.287
    return 4.0 * runs_per_team_game / exponent


def season_scale(ev: pd.DataFrame) -> dict:
    """RE table, linear weights and runs per win for one regular season."""
    ev = add_runs_to_end(ev)
    re = re_table(ev)
    ev = add_re24(ev, re)
    pa = ev[ev["kind"] == "pa"]
    games = ev.groupby("game_pk")["runs"].sum()
    rpg = float(games.sum() / (2 * len(games)))
    return {"events": ev, "re": re, "lw": linear_weights(pa), "runs_per_game": rpg, "runs_per_win": runs_per_win(rpg)}
