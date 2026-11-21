"""Batting and pitching runs above average, park- and month-adjusted.

Each plate appearance is valued at its outcome's linear weight and compared
with the expected linear weight of the league-average matchup in that season,
month and park (`context.context_probs`). A Mexico City double is therefore
measured against what an average hitter does in Mexico City in that month.
Each season is centred so batting runs sum to zero.

Pitching has three candidate definitions, built side by side and chosen by
measurement (`validate.choose_pitching`):
- A: strikeouts, walks, hit batters and home runs at face value, balls in play
  at their expected value from trajectory, spray angle and park (the defense
  owns the rest, see `fielding`);
- B: the same, with every ball in play at the league's value for a ball in
  play in that context (FIP-style);
- C: runs charged to the pitcher against the context's expected runs, with
  the fielders' runs on his balls in play taken out."""

import numpy as np
import pandas as pd

from backend.etl.pa_frame import OUTCOMES

NON_BIP = ("K", "BB", "HBP", "HR")
BIP_OUTCOMES = ("1B", "2B", "3B", "OUT")


def context_value(probs: np.ndarray, lw: pd.Series, seasons: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Expected linear weight of each PA's context, overall and given a ball in
    play (not a home run)."""
    w = np.vstack([lw.loc[s].reindex(list(OUTCOMES)).to_numpy() for s in seasons.to_numpy()])
    overall = (probs * w).sum(axis=1)
    idx = [OUTCOMES.index(o) for o in BIP_OUTCOMES]
    bip = (probs[:, idx] * w[:, idx]).sum(axis=1) / probs[:, idx].sum(axis=1)
    return overall, bip


def pa_values(pa: pd.DataFrame, lw: pd.Series, probs: np.ndarray, bip_expected: pd.Series) -> pd.DataFrame:
    """Per-PA columns: `lw_value`, `ctx`, and the pitcher-side values `v_a`,
    `v_b` (lower is better for the pitcher)."""
    out = pa[["event_id", "season", "batter", "pitcher", "outcome"]].copy()
    out["lw_value"] = lw.reindex(pd.MultiIndex.from_arrays([pa["season"], pa["outcome"]])).to_numpy()
    ctx, ctx_bip = context_value(probs, lw, pa["season"])
    out["ctx"] = ctx
    exp_bip = bip_expected.reindex(pa["event_id"]).to_numpy()
    non_bip = pa["outcome"].isin(NON_BIP).to_numpy()
    out["v_a"] = np.where(non_bip, out["lw_value"], np.where(np.isnan(exp_bip), out["lw_value"], exp_bip))
    out["v_b"] = np.where(non_bip, out["lw_value"], ctx_bip)
    return out


def _centred(values: pd.Series, seasons: pd.Series) -> pd.Series:
    return values - values.groupby(seasons).transform("mean")


def batting_runs(v: pd.DataFrame) -> pd.DataFrame:
    runs = _centred(v["lw_value"] - v["ctx"], v["season"])
    return v.assign(runs=runs).groupby(["season", "batter"])["runs"].sum().rename("batting").reset_index()


def pitching_runs_ab(v: pd.DataFrame) -> pd.DataFrame:
    out = v.assign(
        pitch_a=_centred(v["ctx"] - v["v_a"], v["season"]),
        pitch_b=_centred(v["ctx"] - v["v_b"], v["season"]),
    )
    return out.groupby(["season", "pitcher"])[["pitch_a", "pitch_b"]].sum().reset_index()


def runs_charged(events: pd.DataFrame, moves: pd.DataFrame) -> pd.DataFrame:
    """Runs charged to the responsible pitcher (the pitcher who put the runner
    on), by season."""
    scored = moves[moves["end"] == "score"].merge(events[["event_id", "season", "pitcher"]], on="event_id")
    who = scored["responsible_pitcher"].fillna(scored["pitcher"]).astype("int64")
    return scored.assign(player=who).groupby(["season", "player"]).size().rename("charged").reset_index()


def pitching_runs_c(
    v: pd.DataFrame, charged: pd.DataFrame, fielding_on_bip: pd.Series, runs_per_pa: dict[int, float]
) -> pd.DataFrame:
    """Expected runs for the pitcher's batters faced (the season's runs per PA
    shifted by the context's expected linear weight) minus runs charged, minus
    the runs his fielders saved on his balls in play."""
    shift = v["ctx"] - v.groupby("season")["ctx"].transform("mean")
    expected = v["season"].map(runs_per_pa) + shift
    saved = fielding_on_bip.reindex(v["event_id"]).fillna(0.0).to_numpy()
    per = (
        v.assign(expected=expected.to_numpy(), saved=saved).groupby(["season", "pitcher"])[["expected", "saved"]].sum()
    )
    per = (
        per.reset_index()
        .merge(charged.rename(columns={"player": "pitcher"}), on=["season", "pitcher"], how="left")
        .fillna({"charged": 0})
    )
    raw = per["expected"] - per["charged"] - per["saved"]
    # Centre each season on zero in proportion to batters faced (expected runs).
    season = per["season"]
    per["pitch_c"] = raw - raw.groupby(season).transform("sum") * per["expected"] / per["expected"].groupby(
        season
    ).transform("sum")
    return per[["season", "pitcher", "pitch_c"]]
