"""Prospect snapshots: one row per player at the end of season S (the LMB and
minor league summer plus the LMP winter that starts that autumn), with
features from seasons S-2..S only and, when matured, the outcome over the next
four LMB seasons.

Who is a prospect at S: age 15 to under 25 on July 1 of S, a line somewhere in
S, LMB time in S-2..S or born in Mexico (the LMB is the destination of most
Mexican-born minor leaguers: 282 of 406 first seen 2005-2019), and not yet
graduated (neither an LMB regular nor an MLB debutant in or before S). A player
is a hitter or a pitcher by his larger window sample.

Rates are level-relative: each count minus the player's opportunities times his
league-season mean, pooled over the window with weights 1, .8, .6 (the mlbmodel
player window, 5/4/3) and shrunk toward zero with 250 opportunities. Age is
also expressed relative to the opportunity-weighted mean age of the player's
league-seasons.

Success: an impact LMB season (at least 10 runs above replacement, roughly 1+
WAR: linear-weight batting plus FanGraphs positional adjustment plus replacement
for hitters, or FIP runs saved plus replacement for pitchers) in the next four
LMB seasons played, or an MLB debut in that span."""

import numpy as np
import pandas as pd

from backend.players.positions import FANGRAPHS_RUNS_PER_162

WINDOW = {0: 1.0, 1: 0.8, 2: 0.6}
SHRINK = 250.0
MAX_AGE = 25.0
MIN_AGE = 15.0
HORIZON = 4
REGULAR_PA_PER_GAME = 2.5
REGULAR_OUTS_PER_GAME = 1.8
IMPACT_RAR = 10.0
MEXICO_COUNTRIES = frozenset({"Mexico", "MEX"})
MILB_LEVEL = {16: 1, 15: 2, 14: 2, 13: 3, 12: 4, 11: 5}

BASELINE_FEATURES = ("age", "log_window", "milb_level", "log_lmb", "log_lmp", "log_milb")
FULL_FEATURES = {
    "H": BASELINE_FEATURES + ("age_rel", "woba_rel", "kbb_rel"),
    "P": BASELINE_FEATURES + ("age_rel", "kbb_rel"),
}


def _woba_num(frame: pd.DataFrame) -> pd.Series:
    singles = (frame.h - frame.d2 - frame.d3 - frame.hr).clip(lower=0)
    return 0.70 * (frame.bb + frame.hbp) + 0.88 * singles + 1.25 * frame.d2 + 1.60 * frame.d3 + 2.05 * frame.hr


def _prepare(lines: pd.DataFrame, people: pd.DataFrame) -> pd.DataFrame:
    df = lines.merge(people[["player", "birth_date", "birth_country"]], on="player", how="left")
    df["age"] = (pd.to_datetime(df.season.astype(str) + "-07-01") - pd.to_datetime(df.birth_date)).dt.days / 365.25
    df["opp"] = np.where(df.group == "hitting", df.pa, df.bf).astype(float)
    df["woba_num"] = _woba_num(df)
    df["kbb_num"] = df.so - df.bb - df.hbp
    df["age_opp"] = df.age * df.opp
    df["level_key"] = df.source + df.league.astype("Int64").astype(str)
    df["milb_level"] = np.where(df.source == "milb", df.sport.map(MILB_LEVEL).fillna(0), 0)
    means = df.groupby(["group", "level_key", "season"])[["opp", "woba_num", "kbb_num", "age_opp"]].sum()
    for col in ("woba_num", "kbb_num", "age_opp"):
        means[f"{col}_m"] = means[col] / means.opp.where(means.opp > 0)
    df = df.merge(
        means[[c for c in means.columns if c.endswith("_m")]].reset_index(), on=["group", "level_key", "season"]
    )
    for col in ("woba_num", "kbb_num"):
        df[f"{col}_dev"] = df[col] - df.opp * df[f"{col}_m"].fillna(0)
    df["age_dev"] = df.age - df.age_opp_m
    return df


def lmb_regulars(lines: pd.DataFrame, team_games: pd.Series) -> pd.DataFrame:
    """(player, season) pairs with an LMB regular season (graduation check)."""
    lmb = lines[lines.source == "lmb"]
    pa = lmb[lmb.group == "hitting"].groupby(["player", "season"]).pa.sum()
    outs = lmb[lmb.group == "pitching"].groupby(["player", "season"]).outs.sum()
    tot = pd.concat([pa, outs], axis=1).fillna(0).reset_index()
    games = tot.season.map(team_games)
    if games.isna().any():
        raise ValueError(f"No team games for LMB seasons {sorted(tot.season[games.isna()].unique())}")
    regular = (tot.pa >= REGULAR_PA_PER_GAME * games) | (tot.outs >= REGULAR_OUTS_PER_GAME * games)
    return tot.loc[regular, ["player", "season"]]


def lmb_impact(lines: pd.DataFrame, bio: pd.DataFrame) -> pd.DataFrame:
    """(player, season) pairs with an impact LMB season (>= IMPACT_RAR)."""
    lmb = lines[lines.source == "lmb"]
    lh = (
        lmb[lmb.group == "hitting"]
        .groupby(["player", "season"], as_index=False)[["pa", "h", "d2", "d3", "hr", "bb", "hbp"]]
        .sum()
    )
    lh["raw_w"] = _woba_num(lh)
    lg_h = lh.groupby("season")[["pa", "raw_w"]].sum()
    pos_adj = bio.position.map(FANGRAPHS_RUNS_PER_162).fillna(0.0)
    pos_runs = lh.player.map(pos_adj).fillna(0.0) * (lh.pa / 650.0)
    lh["rar"] = (lh.raw_w - lh.pa * lh.season.map(lg_h.raw_w / lg_h.pa)) / 1.15 + pos_runs + 0.025 * lh.pa

    lp = (
        lmb[lmb.group == "pitching"]
        .groupby(["player", "season"], as_index=False)[["outs", "hr", "bb", "hbp", "so"]]
        .sum()
    )
    lp["ip"] = lp.outs / 3.0
    lp["fip_num"] = 13.0 * lp.hr + 3.0 * (lp.bb + lp.hbp) - 2.0 * lp.so
    lg_p = lp.groupby("season")[["ip", "fip_num"]].sum()
    lp["rar"] = (lp.season.map(lg_p.fip_num / lg_p.ip) * lp.ip - lp.fip_num) / 9.0 + 0.10 * lp.ip

    both = pd.concat([lh[["player", "season", "rar"]], lp[["player", "season", "rar"]]], ignore_index=True)
    peak = both.groupby(["player", "season"], as_index=False).rar.max()
    return peak.loc[peak.rar >= IMPACT_RAR, ["player", "season"]]


def build(
    lines: pd.DataFrame, people: pd.DataFrame, team_games: pd.Series, seasons: list[int], played_lmb: list[int]
) -> pd.DataFrame:
    """Snapshots for `seasons`. `played_lmb` lists every LMB season played so
    far (complete ones only), which bounds the outcome windows."""
    df = _prepare(lines, people)
    bio = people.set_index("player")
    regulars = lmb_regulars(lines, team_games)
    first_regular = regulars.groupby("player").season.min()
    impacts = lmb_impact(lines, bio)
    debut_year = pd.to_datetime(bio.mlb_debut).dt.year
    is_mexican = bio.birth_country.isin(MEXICO_COUNTRIES)
    rows = []
    for s in seasons:
        here = df[df.season == s]
        ages = here.groupby("player").age.first()
        cand = ages[(ages >= MIN_AGE) & (ages < MAX_AGE)].index
        win = df[(df.season <= s) & (df.season >= s - 2) & df.player.isin(cand)].copy()
        win["w"] = (s - win.season).map(WINDOW)
        in_lmb = win[win.source == "lmb"].player.unique()
        mexican = is_mexican.reindex(cand).fillna(False)
        eligible = set(in_lmb) | set(mexican[mexican].index)
        eligible -= set(first_regular[first_regular <= s].index)
        eligible -= set(debut_year[debut_year <= s].index)
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
        wopp_sum = g.wopp.sum()
        denom = wopp_sum + SHRINK
        feat = pd.DataFrame(
            {
                "log_window": np.log1p(wopp_sum),
                "milb_level": g.milb_level.max(),
                "age_rel": ((win.wopp * win.age_dev).groupby(win.player).sum() / wopp_sum.clip(lower=1.0)).fillna(0.0),
                "woba_rel": (win.w * win.woba_num_dev).groupby(win.player).sum() / denom,
                "kbb_rel": (win.w * win.kbb_num_dev).groupby(win.player).sum() / denom,
            }
        )
        for source in ("lmb", "lmp", "milb"):
            feat[f"log_{source}"] = (
                np.log1p(cur[cur.source == source].groupby("player").opp.sum()).reindex(feat.index).fillna(0)
            )
        feat["season"] = s
        feat["role"] = roles.reindex(feat.index).map({"hitting": "H", "pitching": "P"})
        feat["age"] = ages.reindex(feat.index)
        feat["mexican"] = is_mexican.reindex(feat.index).fillna(False).astype(float).to_numpy()
        feat["team"] = here_any.sort_values("opp").groupby("player").team.last().reindex(feat.index)
        feat["source_now"] = (
            here_any.groupby("player").source.agg(lambda x: "+".join(sorted(set(x)))).reindex(feat.index)
        )
        rows.append(feat.reset_index(names="player"))
    snaps = pd.concat(rows, ignore_index=True)
    return _attach_outcomes(snaps, impacts, debut_year, played_lmb)


def _attach_outcomes(
    snaps: pd.DataFrame, impacts: pd.DataFrame, debut_year: pd.Series, played_lmb: list[int]
) -> pd.DataFrame:
    imp_seasons = impacts.groupby("player").season.apply(set).to_dict()
    ends, ys = [], []
    for player, s in zip(snaps.player, snaps.season):
        window = [y for y in played_lmb if y > s][:HORIZON]
        if len(window) < HORIZON:
            ends.append(np.nan)
            ys.append(np.nan)
            continue
        hit = bool(imp_seasons.get(player, set()) & set(window))
        debut = debut_year.get(player)
        hit = hit or (pd.notna(debut) and s < debut <= window[-1])
        ends.append(window[-1])
        ys.append(float(hit))
    snaps["window_end"] = ends
    snaps["success"] = ys
    return snaps
