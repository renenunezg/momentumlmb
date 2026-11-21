"""Leverage of relief appearances. Win expectancy is measured from LMB
play-by-play (home-team view, by innings left, half, outs, bases and score
difference), each event's swing is the change to the next event's state (or
to the final result), and leverage is a state's mean swing over the league's.
A reliever's game leverage is the leverage when he enters."""

import numpy as np
import pandas as pd

MIN_CELL = 50
DIFF_CAP = 8
INNINGS_CAP = 8


def _with_state(ev: pd.DataFrame) -> pd.DataFrame:
    ev = ev.sort_values(["game_pk", "at_bat_index", "play_index"]).copy()
    top = ev["half"] == "top"
    home_runs = np.where(top, 0, ev["runs"])
    away_runs = np.where(top, ev["runs"], 0)
    ev["home_before"] = pd.Series(home_runs, index=ev.index).groupby(ev["game_pk"]).cumsum() - home_runs
    ev["away_before"] = pd.Series(away_runs, index=ev.index).groupby(ev["game_pk"]).cumsum() - away_runs
    total_home = pd.Series(home_runs, index=ev.index).groupby(ev["game_pk"]).transform("sum")
    total_away = pd.Series(away_runs, index=ev.index).groupby(ev["game_pk"]).transform("sum")
    ev["home_won"] = (total_home > total_away).astype(float)
    ev["diff"] = (ev["home_before"] - ev["away_before"]).clip(-DIFF_CAP, DIFF_CAP)
    ev["inn_left"] = (ev["scheduled_innings"] - ev["inning"]).clip(0, INNINGS_CAP)
    ev["outs_s"] = ev["outs_before"].clip(upper=2)
    return ev


LEVELS = (
    ["inn_left", "half", "outs_s", "bases_before", "diff"],
    ["inn_left", "half", "outs_s", "diff"],
    ["inn_left", "half", "diff"],
    ["inn_left", "diff"],
)


def _smoothed(ev: pd.DataFrame, value: str) -> np.ndarray:
    """Cell mean of `value` at the finest level with MIN_CELL events."""
    out = np.full(len(ev), np.nan)
    for keys in LEVELS:
        g = ev.groupby(keys)[value]
        mean, n = g.transform("mean").to_numpy(), g.transform("size").to_numpy()
        fill = np.isnan(out) & (n >= MIN_CELL)
        out[fill] = mean[fill]
    out[np.isnan(out)] = ev[value].mean()
    return out


def leverage_table(ev: pd.DataFrame) -> pd.DataFrame:
    """Events with win expectancy `we` and leverage `li`."""
    ev = _with_state(ev)
    ev["we"] = _smoothed(ev, "home_won")
    nxt = ev.groupby("game_pk")["we"].shift(-1)
    ev["swing"] = (nxt.fillna(ev["home_won"]) - ev["we"]).abs()
    pa = ev["kind"] == "pa"
    li = _smoothed(ev, "swing")
    ev["li"] = li / li[pa.to_numpy()].mean()
    return ev


def relief_leverage(ev: pd.DataFrame) -> pd.DataFrame:
    """Game leverage of each relief appearance (the leverage of the first
    plate appearance he pitches), and batters faced in relief, by pitcher-season."""
    pa = ev[ev["kind"] == "pa"].sort_values(["game_pk", "at_bat_index"])
    side = np.where(pa["half"] == "top", "home", "away")
    pa = pa.assign(side=side)
    first = pa.groupby(["game_pk", "side"])["pitcher"].transform("first")
    relief = pa[pa["pitcher"] != first]
    entries = relief.groupby(["game_pk", "pitcher"]).first().reset_index()
    gm = entries.groupby(["season", "pitcher"])["li"].mean().rename("gmli")
    bf = relief.groupby(["season", "pitcher"]).size().rename("relief_bf")
    return pd.concat([gm, bf], axis=1).reset_index()
