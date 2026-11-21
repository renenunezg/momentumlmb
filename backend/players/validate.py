"""Validation of the WAR and the grade. Every choice is read on the
development seasons (2021-2025); the 2026 holdout is reported once and never
used to choose anything.

- `choose_pitching`: pitching variants A/B/C by how well last season's
  pitching-plus-fielding rates, on this season's batters faced, predict each
  team's park-adjusted runs allowed (pooled correlation over 2022-2025), with
  the variants' year-to-year repeatability beside it.
- `team_reconciliation`: team WAR plus replacement wins against actual wins
  (Pythagorean wins from actual runs as the ceiling).
- `forward_team_wins`: last season's WAR rates times this season's playing
  time against this season's wins.
- `repeatability`: year-to-year correlation of each component's rate.
- `park_check`: park-adjusted batting runs against home elevation by team.
- `grade_forward`: rank correlation of the grade's projection with the target
  season's actual rate, against the prior-season rate and a Marcel average."""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from backend.etl.games import load_games
from backend.etl.pa_frame import FINAL_STATES
from backend.venues import load_venues

PITCH_K = 300.0
FIELD_K = 1000.0


def _boot_r(x: np.ndarray, y: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    r = float(np.corrcoef(x, y)[0, 1])
    bs = []
    for _ in range(n):
        i = rng.integers(0, len(x), len(x))
        bs.append(np.corrcoef(x[i], y[i])[0, 1])
    lo, hi = np.percentile(bs, [5, 95])
    return round(r, 3), round(float(lo), 3), round(float(hi), 3)


def team_results(seasons: list[int]) -> pd.DataFrame:
    g = load_games(seasons)
    g = g[(g["game_type"] == "R") & g["status"].isin(FINAL_STATES)]
    rows = []
    for side, other in (("home", "away"), ("away", "home")):
        rows.append(
            pd.DataFrame(
                {
                    "season": g["season"],
                    "team_id": g[f"{side}_team_id"],
                    "rs": g[f"{side}_score"],
                    "ra": g[f"{other}_score"],
                    "win": (g[f"{side}_score"] > g[f"{other}_score"]).astype(int),
                }
            )
        )
    t = (
        pd.concat(rows)
        .groupby(["season", "team_id"])
        .agg(games=("win", "size"), wins=("win", "sum"), rs=("rs", "sum"), ra=("ra", "sum"))
    )
    t = t.reset_index()
    rpg = (t["rs"] + t["ra"]) / t["games"]
    x = rpg**0.287
    t["pyth_wins"] = t["games"] * t["rs"] ** x / (t["rs"] ** x + t["ra"] ** x)
    return t


def stint_shares(pa: pd.DataFrame) -> pd.DataFrame:
    """Share of each player-season's playing time (PA plus BF) with each team."""
    bat_team = np.where(pa["half"] == "top", pa["away_team_id"], pa["home_team_id"])
    fld_team = np.where(pa["half"] == "top", pa["home_team_id"], pa["away_team_id"])
    a = pd.DataFrame({"season": pa["season"], "player": pa["batter"], "team_id": bat_team})
    b = pd.DataFrame({"season": pa["season"], "player": pa["pitcher"], "team_id": fld_team})
    n = pd.concat([a, b]).groupby(["season", "player", "team_id"]).size().rename("n").reset_index()
    n["share"] = n["n"] / n.groupby(["season", "player"])["n"].transform("sum")
    return n


def team_reconciliation(players: pd.DataFrame, replacement: pd.DataFrame, pa: pd.DataFrame) -> pd.DataFrame:
    shares = stint_shares(pa)
    t = shares.merge(players[["season", "player", "war"]], on=["season", "player"])
    team_war = (
        t.assign(w=t["war"] * t["share"]).groupby(["season", "team_id"])["w"].sum().rename("team_war").reset_index()
    )
    res = team_results(sorted(players["season"].unique())).merge(team_war, on=["season", "team_id"])
    res["war_wins"] = res["team_war"] + res["season"].map(replacement["team"]) * res["games"]
    return res


def summarize_fit(res: pd.DataFrame, pred: str, actual: str = "wins") -> dict:
    err = res[pred] - res[actual]
    return {
        "teams": int(len(res)),
        "r": round(float(np.corrcoef(res[pred], res[actual])[0, 1]), 3),
        "rmse_wins": round(float(np.sqrt((err**2).mean())), 2),
        "mean_error": round(float(err.mean()), 2),
    }


def forward_team_wins(
    players: pd.DataFrame, replacement: pd.DataFrame, pa: pd.DataFrame, targets: list[int]
) -> pd.DataFrame:
    """Last season's WAR per unit of playing time (PA plus BF), regressed
    toward that season's league average rate (newcomers get the average),
    times this season's playing time with each team."""
    shares = stint_shares(pa)
    out = []
    for t in targets:
        prev = players[players["season"] == t - 1].set_index("player")
        cur = players[players["season"] == t]
        n_prev = (prev["pa"] + prev["bf"]).rename("n_prev")
        avg = prev["war"].sum() / n_prev.sum()
        rate = ((prev["war"] + 200.0 * avg) / (n_prev + 200.0)).rename("rate")
        c = (
            cur.assign(n=cur["pa"] + cur["bf"])
            .merge(rate, left_on="player", right_index=True, how="left")
            .fillna({"rate": avg})
        )
        c["proj"] = c["rate"] * c["n"]
        s = shares[shares["season"] == t].merge(c[["player", "proj"]], on="player")
        team = s.assign(w=s["proj"] * s["share"]).groupby("team_id")["w"].sum().rename("proj_war").reset_index()
        team["season"] = t
        out.append(team)
    res = team_results(targets).merge(pd.concat(out), on=["season", "team_id"])
    res["proj_wins"] = res["proj_war"] + res["season"].map(replacement["team"]) * res["games"]
    return res


def repeatability(players: pd.DataFrame, cols_n: dict[str, str], min_n: dict[str, int], seasons: list[int]) -> dict:
    p = players[players["season"].isin(seasons)]
    out = {}
    for col, n in cols_n.items():
        a = p[p[n] >= min_n[n]][["season", "player", col, n]]
        b = a.assign(season=a["season"] - 1)
        m = a.merge(b, on=["season", "player"], suffixes=("_0", "_1"))
        x = (m[f"{col}_0"] / m[f"{n}_0"]).to_numpy()
        y = (m[f"{col}_1"] / m[f"{n}_1"]).to_numpy()
        out[col] = {"pairs": int(len(m)), "r": _boot_r(x, y)}
    return out


def choose_pitching(players: pd.DataFrame, events_pa: pd.DataFrame, probs_ctx: pd.Series, targets: list[int]) -> dict:
    """Pooled correlation over teams and seasons between predicted and actual
    runs allowed above the park-and-month expectation, per variant."""
    fld_team = np.where(events_pa["half"] == "top", events_pa["home_team_id"], events_pa["away_team_id"])
    pa = events_pa.assign(team_id=fld_team, ctx=probs_ctx.to_numpy())
    runs_per_pa = pa.groupby("season")["runs"].sum() / pa.groupby("season").size()
    # Actual runs allowed above expectation per team-season (runs on PA events).
    pa["expected"] = pa["season"].map(runs_per_pa) + pa["ctx"] - pa.groupby("season")["ctx"].transform("mean")
    actual = pa.groupby(["season", "team_id"]).apply(
        lambda d: d["expected"].sum() - d["runs"].sum(), include_groups=False
    )
    actual = actual.rename("actual").reset_index()

    bf = pa.groupby(["season", "team_id", "pitcher"]).size().rename("bf_t").reset_index()
    from backend.players.events import FIELD_POSITIONS

    dpa = []
    for pos in FIELD_POSITIONS:
        col = pa[f"def_{pos}"]
        dpa.append(pa.loc[col.notna(), ["season", "team_id"]].assign(player=col[col.notna()].astype("int64")))
    dpa = pd.concat(dpa).groupby(["season", "team_id", "player"]).size().rename("dpa_t").reset_index()

    results = {}
    for variant in ("pitch_a", "pitch_b", "pitch_c"):
        preds = []
        for t in targets:
            prev = players[players["season"] == t - 1].set_index("player")
            prate = (prev[variant] + prev["fld_p"]) / (prev["bf"] + PITCH_K)
            frate = (prev["fld_zone"] + prev["throwing"]) / (prev["def_pa"] + FIELD_K)
            if variant == "pitch_c":
                frate = frate * 0.0  # C already carries the defense behind the pitcher
            pb = (
                bf[bf["season"] == t]
                .merge(prate.rename("r"), left_on="pitcher", right_index=True, how="left")
                .fillna({"r": 0.0})
            )
            fb = (
                dpa[dpa["season"] == t]
                .merge(frate.rename("r"), left_on="player", right_index=True, how="left")
                .fillna({"r": 0.0})
            )
            team = (
                pb.assign(x=pb["r"] * pb["bf_t"])
                .groupby("team_id")["x"]
                .sum()
                .add(fb.assign(x=fb["r"] * fb["dpa_t"]).groupby("team_id")["x"].sum(), fill_value=0.0)
            )
            preds.append(team.rename("pred").reset_index().assign(season=t))
        m = actual.merge(pd.concat(preds), on=["season", "team_id"])
        results[variant] = {"teams": int(len(m)), "r": _boot_r(m["pred"].to_numpy(), m["actual"].to_numpy())}
    return results


def park_check(players: pd.DataFrame, pa_values: pd.DataFrame, pa: pd.DataFrame) -> dict:
    """Correlation across team-seasons of batting runs per PA with the home
    park's elevation, park-adjusted and raw (linear weight minus the season's
    league mean). Home and road PAs both count, as a team's hitters' totals do."""
    venues = load_venues()
    bat_team = np.where(pa["half"] == "top", pa["away_team_id"], pa["home_team_id"])
    v = pa_values.assign(team_id=bat_team)
    v["adj"] = v["lw_value"] - v["ctx"]
    v["raw"] = v["lw_value"] - v.groupby("season")["lw_value"].transform("mean")
    home_venue = (
        pa[pa["half"] == "bottom"].groupby(["season", "home_team_id"])["venue_id"].agg(lambda s: s.mode().iloc[0])
    )
    elev = (
        home_venue.map(venues["elevation_m"])
        .rename("elevation")
        .reset_index()
        .rename(columns={"home_team_id": "team_id"})
    )
    t = v.groupby(["season", "team_id"])[["adj", "raw"]].mean().reset_index().merge(elev, on=["season", "team_id"])
    return {
        "team_seasons": int(len(t)),
        "adjusted_r": _boot_r(t["adj"].to_numpy(), t["elevation"].to_numpy()),
        "raw_r": _boot_r(t["raw"].to_numpy(), t["elevation"].to_numpy()),
    }


def spearman(a: pd.Series, b: pd.Series) -> float:
    return round(float(spearmanr(a, b).statistic), 3)


def _paired_boot(a: pd.DataFrame, col_x: str, col_y: str, target: str, n: int = 1000, seed: int = 0) -> list[float]:
    """90% interval of the pooled Spearman gain of col_x over col_y,
    resampling players within each target season."""
    rng = np.random.default_rng(seed)
    gains = []
    groups = [g for _, g in a.groupby("season")]
    for _ in range(n):
        gx, gy = [], []
        for g in groups:
            i = rng.integers(0, len(g), len(g))
            s = g.iloc[i]
            gx.append(spearmanr(s[col_x], s[target]).statistic)
            gy.append(spearmanr(s[col_y], s[target]).statistic)
        gains.append(np.mean(gx) - np.mean(gy))
    lo, hi = np.percentile(gains, [5, 95])
    return [round(float(lo), 3), round(float(hi), 3)]


def grade_forward(rows: pd.DataFrame, target: str, columns: list[str]) -> dict:
    """Spearman of each column with the target rate, per season and pooled
    (mean over seasons), with the first column's gain over the others."""
    out = {"n": int(len(rows)), "by_season": {}, "pooled": {}}
    for season, g in rows.groupby("season"):
        out["by_season"][int(season)] = {c: spearman(g[c], g[target]) for c in columns}
    for c in columns:
        out["pooled"][c] = round(float(np.mean([v[c] for v in out["by_season"].values()])), 3)
    out["gain_interval"] = {c: _paired_boot(rows, columns[0], c, target) for c in columns[1:]}
    return out
