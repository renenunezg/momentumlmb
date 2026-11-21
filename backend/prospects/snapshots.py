"""Prospect snapshots: one row per player at the end of season S (the LMB and
minor league summer plus the LMP winter that starts that autumn), with
features from seasons S-2..S only and, when matured, the outcome over the next
four LMB seasons.

Who is a prospect at S: under 25 on July 1 of S, a line somewhere in S, LMB or
LMP time in S-2..S or born in Mexico (the LMB is the destination of most
Mexican-born minor leaguers: 282 of 406 first seen 2005-2019), and not yet an
LMB regular. A player is a hitter or a pitcher by his larger window sample.

Rates are level-relative: each count minus the player's opportunities times
his league-season mean, pooled over the window with weights 1, .8, .6 (the
mlbmodel player window, 5/4/3) and shrunk toward zero with 100 opportunities.

Success: an LMB regular season (2.5 plate appearances or 1.8 outs per team game,
250 PA or 60 IP in a 100-game season) in the next four LMB seasons played, or
an MLB debut in that span, so a player who leaves for MLB is not a failure."""

import numpy as np
import pandas as pd

WINDOW = {0: 1.0, 1: 0.8, 2: 0.6}
SHRINK = 100.0
MAX_AGE = 25.0
MIN_AGE = 15.0
HORIZON = 4
REGULAR_PA_PER_GAME = 2.5
REGULAR_OUTS_PER_GAME = 1.8
MILB_LEVEL = {16: 1, 15: 2, 14: 2, 13: 3, 12: 4, 11: 5}
PREMIUM_POSITIONS = ("SS", "2B", "CF")

RATES = {"hitting": ("so", "bb", "hr", "xbh", "hit"), "pitching": ("so", "bb", "hr", "hit")}
BASELINE_FEATURES = ("age", "log_window", "milb_level", "log_lmb", "log_lmp", "log_milb")
FULL_FEATURES = {
    "H": BASELINE_FEATURES + ("mexican",) + tuple(f"{r}_rel" for r in RATES["hitting"]) + ("catcher", "premium"),
    "P": BASELINE_FEATURES + ("mexican",) + tuple(f"{r}_rel" for r in RATES["pitching"]) + ("start_share", "lefty"),
}


def _prepare(lines: pd.DataFrame, people: pd.DataFrame) -> pd.DataFrame:
    df = lines.merge(people[["player", "birth_date", "birth_country"]], on="player", how="left")
    df["age"] = (pd.to_datetime(df.season.astype(str) + "-07-01") - pd.to_datetime(df.birth_date)).dt.days / 365.25
    df["opp"] = np.where(df.group == "hitting", df.pa, df.bf).astype(float)
    df["xbh"] = df.d2 + df.d3
    df["hit"] = df.h - df.hr
    df["level_key"] = df.source + df.league.astype("Int64").astype(str)
    df["milb_level"] = np.where(df.source == "milb", df.sport.map(MILB_LEVEL).fillna(0), 0)
    means = df.groupby(["group", "level_key", "season"])[["opp", "so", "bb", "hr", "xbh", "hit"]].sum()
    for rate in ("so", "bb", "hr", "xbh", "hit"):
        means[f"{rate}_m"] = means[rate] / means.opp.where(means.opp > 0)
    df = df.merge(
        means[[c for c in means.columns if c.endswith("_m")]].reset_index(), on=["group", "level_key", "season"]
    )
    for rate in ("so", "bb", "hr", "xbh", "hit"):
        df[f"{rate}_dev"] = df[rate] - df.opp * df[f"{rate}_m"].fillna(0)
    return df


def lmb_regulars(lines: pd.DataFrame, team_games: pd.Series) -> pd.DataFrame:
    """(player, season) pairs with an LMB regular season."""
    lmb = lines[lines.source == "lmb"]
    pa = lmb[lmb.group == "hitting"].groupby(["player", "season"]).pa.sum()
    outs = lmb[lmb.group == "pitching"].groupby(["player", "season"]).outs.sum()
    tot = pd.concat([pa, outs], axis=1).fillna(0).reset_index()
    games = tot.season.map(team_games)
    if games.isna().any():
        raise ValueError(f"No team games for LMB seasons {sorted(tot.season[games.isna()].unique())}")
    regular = (tot.pa >= REGULAR_PA_PER_GAME * games) | (tot.outs >= REGULAR_OUTS_PER_GAME * games)
    return tot.loc[regular, ["player", "season"]]


def build(
    lines: pd.DataFrame, people: pd.DataFrame, team_games: pd.Series, seasons: list[int], played_lmb: list[int]
) -> pd.DataFrame:
    """Snapshots for `seasons`. `played_lmb` lists every LMB season played so
    far (complete ones only), which bounds the outcome windows."""
    df = _prepare(lines, people)
    bio = people.set_index("player")
    regulars = lmb_regulars(lines, team_games)
    first_regular = regulars.groupby("player").season.min()
    debut_year = pd.to_datetime(bio.mlb_debut).dt.year
    rows = []
    for s in seasons:
        here = df[df.season == s]
        ages = here.groupby("player").age.first()
        cand = ages[(ages >= MIN_AGE) & (ages < MAX_AGE)].index
        win = df[(df.season <= s) & (df.season >= s - 2) & df.player.isin(cand)].copy()
        win["w"] = (s - win.season).map(WINDOW)
        pro = win[win.source.isin(["lmb", "lmp"])].player.unique()
        mexican = bio.birth_country.reindex(cand).eq("Mexico")
        eligible = set(pro) | set(mexican[mexican].index)
        eligible -= set(first_regular[first_regular <= s].index)
        win = win[win.player.isin(eligible)]
        if win.empty:
            continue
        # Team and level come from every line this season, in either role, so a player whose season was all
        # in his other role (a hitter converting to pitching) still has them.
        here_any = win[win.season == s]
        win["wopp"] = win.w * win.opp
        role_opp = win.pivot_table(index="player", columns="group", values="wopp", aggfunc="sum", fill_value=0)
        role_opp = role_opp.reindex(columns=["hitting", "pitching"], fill_value=0)
        roles = pd.Series(np.where(role_opp.hitting >= role_opp.pitching, "hitting", "pitching"), index=role_opp.index)
        win = win[win.group == win.player.map(roles)]
        cur = win[win.season == s]
        g = win.groupby("player")
        feat = pd.DataFrame(
            {
                "log_window": np.log1p(g.wopp.sum()),
                "milb_level": g.milb_level.max(),
                "start_share": g.apply(
                    lambda x: (x.w * x.gs).sum() / max((x.w * x.g).sum(), 1.0), include_groups=False
                ),
            }
        )
        for source in ("lmb", "lmp", "milb"):
            feat[f"log_{source}"] = (
                np.log1p(cur[cur.source == source].groupby("player").opp.sum()).reindex(feat.index).fillna(0)
            )
        denom = g.wopp.sum() + SHRINK
        for rate in ("so", "bb", "hr", "xbh", "hit"):
            feat[f"{rate}_rel"] = (win.w * win[f"{rate}_dev"]).groupby(win.player).sum() / denom
        feat["season"] = s
        feat["role"] = roles.reindex(feat.index).map({"hitting": "H", "pitching": "P"})
        feat["age"] = ages.reindex(feat.index)
        feat["mexican"] = bio.birth_country.reindex(feat.index).eq("Mexico").astype(float).to_numpy()
        pos = bio.position.reindex(feat.index)
        feat["catcher"] = pos.eq("C").astype(float).to_numpy()
        feat["premium"] = pos.isin(PREMIUM_POSITIONS).astype(float).to_numpy()
        feat["lefty"] = bio.throws.reindex(feat.index).eq("L").astype(float).to_numpy()
        feat["team"] = here_any.sort_values("opp").groupby("player").team.last().reindex(feat.index)
        feat["source_now"] = (
            here_any.groupby("player").source.agg(lambda x: "+".join(sorted(set(x)))).reindex(feat.index)
        )
        rows.append(feat.reset_index(names="player"))
    snaps = pd.concat(rows, ignore_index=True)
    return _attach_outcomes(snaps, regulars, debut_year, played_lmb)


def _attach_outcomes(
    snaps: pd.DataFrame, regulars: pd.DataFrame, debut_year: pd.Series, played_lmb: list[int]
) -> pd.DataFrame:
    reg_seasons = regulars.groupby("player").season.apply(set).to_dict()
    ends, ys = [], []
    for player, s in zip(snaps.player, snaps.season):
        window = [y for y in played_lmb if y > s][:HORIZON]
        if len(window) < HORIZON:
            ends.append(np.nan)
            ys.append(np.nan)
            continue
        hit = bool(reg_seasons.get(player, set()) & set(window))
        debut = debut_year.get(player)
        hit = hit or (pd.notna(debut) and s < debut <= window[-1])
        ends.append(window[-1])
        ys.append(float(hit))
    snaps["window_end"] = ends
    snaps["success"] = ys
    return snaps
