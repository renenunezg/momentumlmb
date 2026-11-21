"""Baserunning runs: steals, caught stealing and pickoffs against the league
rate per opportunity, and advancement on balls in play against what the
average runner did from the same base, outs and kind of play."""

import numpy as np
import pandas as pd

BASES = ("1B", "2B", "3B")
STEAL_PATTERN = r"^(stolen_base|caught_stealing|pickoff_caught_stealing|pickoff)_(2b|3b|home|1b)$"


def runner_marginals(re: pd.Series) -> dict:
    """Run value of a runner on each base with 0, 1 or 2 outs (expectancy with
    him minus without, averaged over the other runners), and the cost of an
    out by outs."""
    marg = {}
    for o in range(3):
        for i, b in enumerate(BASES):
            bit = 1 << i
            diffs = [re.loc[(o, s)] - re.loc[(o, s & ~bit)] for s in range(8) if s & bit]
            marg[(b, o)] = float(np.mean(diffs))
    out_cost = {
        o: float(np.mean([re.loc[(o, s)] - (re.loc[(o + 1, s)] if o < 2 else 0.0) for s in range(8)])) for o in range(3)
    }
    return {"base": marg, "out": out_cost}


def runners_on(events: pd.DataFrame) -> pd.DataFrame:
    """One row per runner on base at the start of each event, with where he
    ended (a runner with no movement stayed on his base)."""
    frames = []
    for b in BASES:
        col = events[f"runner_{b}"]
        on = events.loc[col.notna()].assign(runner=col[col.notna()].astype("int64"), start=b)
        frames.append(on)
    return pd.concat(frames, ignore_index=True)


def _with_end(on: pd.DataFrame, moves: pd.DataFrame) -> pd.DataFrame:
    mv = moves[~moves["is_batter"]][["event_id", "runner", "end"]]
    out = on.merge(mv, on=["event_id", "runner"], how="left")
    out["end"] = out["end"].fillna(out["start"])
    return out


def steal_runs(br: pd.DataFrame, moves: pd.DataFrame, pa: pd.DataFrame) -> pd.DataFrame:
    """Steal-family run value by runner, net of the league's value per
    opportunity (a plate appearance begun on first with second open, or on
    second with third open)."""
    kind = br["event_type"].fillna("").str.match(STEAL_PATTERN)
    ev = br[kind][["event_id", "season", "re24"]]
    mv = moves[moves["event_id"].isin(ev["event_id"]) & ~moves["is_batter"] & moves["start"].isin(BASES)]
    mv = mv.merge(ev, on="event_id")
    mv["runs"] = mv["re24"] / mv.groupby("event_id")["runner"].transform("size")
    by_runner = mv.groupby(["season", "runner"])["runs"].sum()

    on = runners_on(pa)
    open_next = ((on["start"] == "1B") & (on["runner_2B"].isna())) | ((on["start"] == "2B") & (on["runner_3B"].isna()))
    opps = on[open_next].groupby(["season", "runner"]).size()
    out = pd.concat([by_runner.rename("steal_raw"), opps.rename("opps")], axis=1).fillna(0.0)
    league = out.groupby("season")[["steal_raw", "opps"]].transform("sum")
    out["steal_runs"] = out["steal_raw"] - out["opps"] * league["steal_raw"] / league["opps"]
    return out.reset_index()[["season", "runner", "steal_runs"]]


def advancement_runs(pa: pd.DataFrame, moves: pd.DataFrame, marginals_by_season: dict) -> pd.DataFrame:
    """Runner advancement on singles, doubles and batted-ball outs: the value of
    where each runner ended (scored, a base, or out) minus the mean for runners
    from the same base with the same outs on the same kind of play, ball hit
    to the same field. Runners forced on ground balls are left out."""
    plays = pa[pa["outcome"].isin(["1B", "2B", "OUT"]) & pa["bb_type"].notna()]
    mv = _with_end(runners_on(plays), moves)
    # A runner forced on a ground ball has no choice: leave those out.
    forced = (
        (mv["start"] == "1B")
        | ((mv["start"] == "2B") & mv["runner_1B"].notna())
        | ((mv["start"] == "3B") & mv["runner_1B"].notna() & mv["runner_2B"].notna())
    )
    mv = mv[~((mv["outcome"] == "OUT") & (mv["bb_type"] == "ground_ball") & forced)]
    field = np.where(mv["location"].isin(["7", "8", "9"]), mv["location"], "IF")
    mv["context"] = (
        mv["outcome"]
        + "|"
        + np.where(mv["outcome"] == "OUT", mv["bb_type"], "")
        + "|"
        + mv["start"]
        + "|"
        + mv["outs_before"].astype(str)
        + "|"
        + field
    )
    value = np.zeros(len(mv))
    for season, m in marginals_by_season.items():
        sel = (mv["season"] == season).to_numpy()
        sub = mv[sel]
        v = np.where(sub["end"] == "score", 1.0, 0.0)
        is_out = (sub["end"] == "out").to_numpy()
        v = np.where(is_out, -np.array([m["out"][o] for o in sub["outs_before"].clip(upper=2)]), v)
        on_base = sub["end"].isin(BASES).to_numpy() & (sub["outs_after"] < 3).to_numpy()
        base_val = [m["base"].get((e, min(o, 2)), 0.0) for e, o in zip(sub["end"], sub["outs_after"])]
        v = np.where(on_base, base_val, v)
        value[sel] = v
    mv["value"] = value
    mv["adv_runs"] = mv["value"] - mv.groupby(["season", "context"])["value"].transform("mean")
    return mv.groupby(["season", "runner"])["adv_runs"].sum().reset_index()
