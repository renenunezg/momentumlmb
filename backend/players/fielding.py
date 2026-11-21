"""Zone fielding from the scorer's batted-ball record, and catcher throwing
and blocking.

Every ball in play (not a home run) is valued at the run value of what
happened (a reached-on-error counts as a single) and at the value expected
from its trajectory, its spray angle and the park. The scorer's hardness
field is "medium" on 98% of balls, so it is not used. The gap between
expected and actual goes to the defense: on an out, to the fielder who made
the play; on a hit, spread over the positions that make that ball's outs, in
the shares they make them; on an error, to the fielder charged. Each
position-season is then centred on its own league average per batter faced,
so a player is measured against the average fielder at his position.

The pitcher keeps the expected value of his balls in play (pitching variant
A), so batting, pitching and fielding add up to the runs on the field."""

import numpy as np
import pandas as pd

from backend.players.events import FIELD_POSITIONS, spray_angle

ANGLE_EDGES = np.arange(-45.0, 46.0, 5.0)
VENUE_SHRINK_BIP = 300.0
POSITIONS_WITH_P = ("P", *FIELD_POSITIONS)
STEAL_TYPES = ("stolen_base", "caught_stealing")


def balls_in_play(pa: pd.DataFrame, lw: pd.Series) -> pd.DataFrame:
    """PA events with a batted ball that is not a home run, with their run
    value (`lw` indexed by season and outcome) and angle bin."""
    bip = pa[pa["bb_type"].notna() & (pa["outcome"] != "HR") & pa["coord_x"].notna()].copy()
    roe = (bip["event_type"] == "field_error") & bip["error_pos"].notna()
    outcome = np.where(roe, "1B", bip["outcome"])
    bip["roe"] = roe
    bip["value"] = lw.reindex(pd.MultiIndex.from_arrays([bip["season"], outcome])).to_numpy()
    angle = np.clip(spray_angle(bip["coord_x"], bip["coord_y"]), -44.99, 44.99)
    bip["angle_bin"] = np.digitize(angle, ANGLE_EDGES) - 1
    bip["is_out"] = (bip["outcome"] == "OUT") & ~roe
    return bip


def expected_value(bip: pd.DataFrame) -> pd.Series:
    """Expected run value of each ball in play: the season's mean for its
    trajectory and angle bin, plus a park term per trajectory shrunk toward
    zero (park dimensions, and each home scorer's habits)."""
    cell = bip.groupby(["season", "bb_type", "angle_bin"])["value"].transform("mean")
    resid = bip["value"] - cell
    grp = bip.assign(resid=resid).groupby(["venue_id", "bb_type"])["resid"]
    park = grp.transform("sum") / (grp.transform("size") + VENUE_SHRINK_BIP)
    return cell + park


def responsibility_shares(bip: pd.DataFrame) -> pd.DataFrame:
    """Share of outs each position makes, by trajectory and angle bin."""
    outs = bip[bip["is_out"] & bip["fielder_pos"].isin(POSITIONS_WITH_P)]
    counts = outs.groupby(["bb_type", "angle_bin", "fielder_pos"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=list(POSITIONS_WITH_P), fill_value=0)
    return counts.div(counts.sum(axis=1), axis=0)


def fielding_credits(bip: pd.DataFrame, shares: pd.DataFrame) -> pd.DataFrame:
    """One row per (event, position, player) with the runs the play saved
    (positive) or cost relative to its expected value. Pitcher credits use the
    batter-faced pitcher."""
    bip = bip.assign(saved=bip["expected"] - bip["value"])
    rows = []
    outs = bip[bip["is_out"] & bip["fielder_pos"].isin(POSITIONS_WITH_P)]
    rows.append(pd.DataFrame({"event_id": outs["event_id"], "pos": outs["fielder_pos"], "saved": outs["saved"]}))
    errs = bip[bip["roe"] & bip["error_pos"].isin(POSITIONS_WITH_P)]
    rows.append(pd.DataFrame({"event_id": errs["event_id"], "pos": errs["error_pos"], "saved": errs["saved"]}))
    hits = bip[~bip["is_out"] & ~bip["roe"]]
    share = shares.reindex(pd.MultiIndex.from_arrays([hits["bb_type"], hits["angle_bin"]])).fillna(0.0).to_numpy()
    for j, pos in enumerate(POSITIONS_WITH_P):
        w = share[:, j]
        keep = w > 0
        rows.append(
            pd.DataFrame(
                {
                    "event_id": hits["event_id"].to_numpy()[keep],
                    "pos": pos,
                    "saved": hits["saved"].to_numpy()[keep] * w[keep],
                }
            )
        )
    credits = pd.concat(rows, ignore_index=True)
    meta = bip.set_index("event_id")
    credits["season"] = meta.loc[credits["event_id"], "season"].to_numpy()
    player = np.full(len(credits), -1, dtype=np.int64)
    for pos in POSITIONS_WITH_P:
        m = (credits["pos"] == pos).to_numpy()
        col = "pitcher" if pos == "P" else f"def_{pos}"
        ids = meta.loc[credits.loc[m, "event_id"], col].astype("Int64").fillna(-1).to_numpy()
        player[m] = ids
    credits["player"] = player
    return credits[credits["player"] >= 0]


def defensive_pa(pa: pd.DataFrame) -> pd.DataFrame:
    """Plate appearances each player stood at each position for, by season
    (pitchers at P)."""
    frames = [pa[["season", "pitcher"]].rename(columns={"pitcher": "player"}).assign(pos="P")]
    for pos in FIELD_POSITIONS:
        col = pa[f"def_{pos}"]
        frames.append(pa.loc[col.notna(), ["season"]].assign(player=col[col.notna()].astype("int64"), pos=pos))
    return pd.concat(frames).groupby(["season", "player", "pos"]).size().rename("def_pa").reset_index()


def zone_fielding(credits: pd.DataFrame, def_pa: pd.DataFrame) -> pd.DataFrame:
    """Runs saved by player, season and position against the average fielder
    at the position (credits minus the position-season mean per batter faced)."""
    by = credits.groupby(["season", "player", "pos"])["saved"].sum().rename("raw").reset_index()
    out = def_pa.merge(by, on=["season", "player", "pos"], how="left").fillna({"raw": 0.0})
    rate = out.groupby(["season", "pos"])[["raw", "def_pa"]].transform("sum")
    out["fielding_runs"] = out["raw"] - out["def_pa"] * rate["raw"] / rate["def_pa"]
    return out


def catcher_runs(br: pd.DataFrame, pa: pd.DataFrame) -> pd.DataFrame:
    """Catcher throwing (steal attempts against the league's run value per
    attempt at that base) and blocking (passed balls against the league rate
    per batter caught), by season and catcher."""
    br = br[br["def_C"].notna()].copy()
    br["catcher"] = br["def_C"].astype("int64")
    family = br["event_type"].fillna("").str.extract(r"^(stolen_base|caught_stealing)_(2b|3b|home)$")
    steals = br[family[0].notna()].assign(base=family[1])
    steals["saved"] = -(steals["re24"] - steals.groupby(["season", "base"])["re24"].transform("mean"))
    throwing = steals.groupby(["season", "catcher"])["saved"].sum().rename("throwing")

    caught = pa[pa["def_C"].notna()].assign(catcher=lambda d: d["def_C"].astype("int64"))
    n_caught = caught.groupby(["season", "catcher"]).size().rename("caught")
    pb = br[br["event_type"] == "passed_ball"].groupby(["season", "catcher"])["re24"].sum().rename("pb_runs")
    out = pd.concat([n_caught, throwing, pb], axis=1).fillna(0.0)
    league = out.groupby("season")[["pb_runs", "caught"]].transform("sum")
    out["blocking"] = -(out["pb_runs"] - out["caught"] * league["pb_runs"] / league["caught"])
    return out[["caught", "throwing", "blocking"]].reset_index().rename(columns={"catcher": "player"})
