"""Replacement level, measured from the players a team turns to when its
regulars are out: everyone outside his team's top 13 hitters by plate
appearances, or top 13 pitchers by batters faced. Starters and relievers get
separate levels (the pool's starts and its relief innings).

One season's pool is small and noisy, so the level is measured once over the
development seasons (2021-2025) as three numbers: the replacement team's
winning percentage, the hitters' share of the runs above it, and the ratio of
the reliever to the starter rate. Each season then gets rates that put its
replacement team at that winning percentage on its own run scale.

The FanGraphs convention (a .294 team, 57% of WAR to hitters, one pitcher
rate) is the fallback if the measured team falls outside .25-.35."""

import pandas as pd

TOP_HITTERS = 13
TOP_PITCHERS = 13
TEAM_RANGE = (0.25, 0.35)
FANGRAPHS_TEAM = 0.294
FANGRAPHS_HITTER_SHARE = 0.57


def _pool(stints: pd.DataFrame, size_col: str, top: int) -> pd.DataFrame:
    rank = stints.groupby(["season", "team_id"])[size_col].rank(ascending=False, method="first")
    return stints[rank > top]


def measured_rates(players: pd.DataFrame, bat_stints: pd.DataFrame, pit_stints: pd.DataFrame) -> pd.DataFrame:
    """Replacement runs per PA for hitters and per batter faced for starters
    and relievers, by season (negative: below average). `players` holds
    season, player, hit_raa, pa, pitch_raa, bf."""
    hit = players.assign(hit_rate=players["hit_raa"] / players["pa"].where(players["pa"] > 0))
    pit = players.assign(pit_rate=players["pitch_raa"] / players["bf"].where(players["bf"] > 0))
    hp = _pool(bat_stints, "pa", TOP_HITTERS).merge(hit[["season", "player", "hit_rate"]], on=["season", "player"])
    pp = _pool(pit_stints, "bf", TOP_PITCHERS).merge(pit[["season", "player", "pit_rate"]], on=["season", "player"])
    rows = []
    for season in sorted(players["season"].unique()):
        h = hp[hp["season"] == season]
        p = pp[pp["season"] == season]
        rows.append(
            {
                "season": season,
                "hitter": (h["hit_rate"] * h["pa"]).sum() / h["pa"].sum(),
                "starter": (p["pit_rate"] * p["sp_bf"]).sum() / p["sp_bf"].sum(),
                "reliever": (p["pit_rate"] * p["rp_bf"]).sum() / p["rp_bf"].sum(),
            }
        )
    return pd.DataFrame(rows).set_index("season")


def replacement_team(rates: pd.DataFrame, totals: pd.DataFrame, runs_per_win: dict, team_games: pd.Series) -> pd.Series:
    """Winning percentage of a team of replacement players by season. `totals`
    holds the season's league PA, starter BF and reliever BF; `team_games` is
    the games played by all teams together."""
    runs = -(rates["hitter"] * totals["pa"] + rates["starter"] * totals["sp_bf"] + rates["reliever"] * totals["rp_bf"])
    wins_above = runs / pd.Series(runs_per_win)
    return 0.5 - wins_above / team_games


def pooled_level(
    rates: pd.DataFrame, totals: pd.DataFrame, runs_per_win: dict, team_games: pd.Series, seasons: list[int]
) -> dict:
    """Measured team, hitter share and reliever/starter ratio over `seasons`
    (season rates weighted by their playing time)."""
    r, t = rates.loc[seasons], totals.loc[seasons]
    rpw = pd.Series(runs_per_win).loc[seasons]
    hit_runs = -(r["hitter"] * t["pa"])
    pit_runs = -(r["starter"] * t["sp_bf"] + r["reliever"] * t["rp_bf"])
    wins = ((hit_runs + pit_runs) / rpw).sum()
    team = 0.5 - wins / team_games.loc[seasons].sum()
    share = float((hit_runs / rpw).sum() / wins)
    ratio = float(
        (r["reliever"] * t["rp_bf"]).sum() / t["rp_bf"].sum() / ((r["starter"] * t["sp_bf"]).sum() / t["sp_bf"].sum())
    )
    level = {"team": float(team), "hitter_share": share, "reliever_over_starter": ratio, "source": "measured"}
    lo, hi = TEAM_RANGE
    if not lo <= team <= hi:
        level = {
            "team": FANGRAPHS_TEAM,
            "hitter_share": FANGRAPHS_HITTER_SHARE,
            "reliever_over_starter": 1.0,
            "source": "fangraphs",
        }
    level["measured_team"] = float(team)
    level["measured_hitter_share"] = share
    return level


def season_rates(level: dict, totals: pd.DataFrame, runs_per_win: dict, team_games: pd.Series) -> pd.DataFrame:
    """Per-season replacement rates (runs per PA or BF, negative) that put
    each season's replacement team at the level's winning percentage."""
    rows = []
    for season, t in totals.iterrows():
        runs = (0.5 - level["team"]) * team_games[season] * runs_per_win[season]
        hitter = -level["hitter_share"] * runs / t["pa"]
        pitcher_runs = (1 - level["hitter_share"]) * runs
        starter = -pitcher_runs / (t["sp_bf"] + level["reliever_over_starter"] * t["rp_bf"])
        rows.append(
            {
                "season": season,
                "hitter": hitter,
                "starter": starter,
                "reliever": starter * level["reliever_over_starter"],
                "team": level["team"],
            }
        )
    return pd.DataFrame(rows).set_index("season")
