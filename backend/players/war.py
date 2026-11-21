"""LMB WAR for every player-season: batting, baserunning, fielding and
positional runs for hitters, pitching runs (with relief leverage) for
pitchers, plus runs above the measured replacement level, over the season's
runs per win."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backend.config import DEVELOPMENT_SEASONS
from backend.players import baserunning, fielding, leverage, positions, replacement, value
from backend.players.context import context_probs
from backend.players.season import Seasons, load_seasons

PITCHING_VARIANTS = ("a", "b", "c")


@dataclass(frozen=True)
class WarBuild:
    players: pd.DataFrame  # one row per season and player
    seasons: Seasons
    replacement: pd.DataFrame  # by season: rates and team
    level: dict  # measured replacement level and its source
    pos_scale: dict[str, float]
    event_fielding: pd.Series  # event_id -> runs the defense saved on that ball in play


def _stints(pa: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    bat_team = np.where(pa["half"] == "top", pa["away_team_id"], pa["home_team_id"])
    fld_team = np.where(pa["half"] == "top", pa["home_team_id"], pa["away_team_id"])
    frame = pa.assign(bat_team=bat_team, team_id=fld_team)
    bat = frame.groupby(["season", "bat_team", "batter"]).size().rename("pa").reset_index()
    bat = bat.rename(columns={"bat_team": "team_id", "batter": "player"})
    ordered = frame.sort_values(["game_pk", "at_bat_index"])
    starter = ordered.groupby(["game_pk", "team_id"])["pitcher"].transform("first")
    frame = ordered.assign(is_start=(ordered["pitcher"] == starter))
    pit = (
        frame.groupby(["season", "team_id", "pitcher"])
        .agg(bf=("is_start", "size"), sp_bf=("is_start", "sum"))
        .reset_index()
    )
    pit["rp_bf"] = pit["bf"] - pit["sp_bf"]
    return bat, pit.rename(columns={"pitcher": "player"})


def build_war(seasons: list[int], context_through: int, variant: str = "a") -> WarBuild:
    if variant not in PITCHING_VARIANTS:
        raise ValueError(f"pitching variant must be one of {PITCHING_VARIANTS}")
    s = load_seasons(seasons)
    pa = s.pa.reset_index(drop=True)
    probs = context_probs(pa, context_through)

    bip = fielding.balls_in_play(pa, s.lw)
    bip["expected"] = fielding.expected_value(bip)
    credits = fielding.fielding_credits(bip, fielding.responsibility_shares(bip))
    def_pa = fielding.defensive_pa(pa)
    zone = fielding.zone_fielding(credits, def_pa)
    event_fielding = (bip["expected"] - bip["value"]).set_axis(bip["event_id"])

    v = value.pa_values(pa, s.lw, probs, bip.set_index("event_id")["expected"])
    batting = value.batting_runs(v).rename(columns={"batter": "player"})
    pitching = value.pitching_runs_ab(v).rename(columns={"pitcher": "player"})
    runs_per_pa = (s.events.groupby("season")["runs"].sum() / pa.groupby("season").size()).to_dict()
    pitch_c = value.pitching_runs_c(v, value.runs_charged(s.events, s.moves), event_fielding, runs_per_pa)
    pitching = pitching.merge(pitch_c.rename(columns={"pitcher": "player"}), on=["season", "player"], how="outer")
    pitching["pitch"] = pitching[f"pitch_{variant}"]

    catcher = fielding.catcher_runs(s.br, pa)[["season", "player", "throwing"]]
    steals = baserunning.steal_runs(s.br, s.moves, pa).rename(columns={"runner": "player"})
    advance = baserunning.advancement_runs(pa, s.moves, s.marginals).rename(columns={"runner": "player"})

    games = pa.groupby("season")["game_pk"].nunique()
    pa_per_team_game = (pa.groupby("season").size() / (2 * games)).to_dict()
    dh_pa = pa[pa["bat_pos"] == "DH"].groupby(["season", "batter"]).size().rename("dh_pa").reset_index()
    dh_pa = dh_pa.rename(columns={"batter": "player"})
    pos_scale = positions.runs_per_162(None)
    pos_runs = positions.positional_runs(def_pa, dh_pa, pos_scale, pa_per_team_game)
    primary = positions.primary_position(def_pa, dh_pa)

    fld_hit = zone[zone["pos"] != "P"].groupby(["season", "player"])["fielding_runs"].sum().rename("fld_zone")
    fld_p = zone[zone["pos"] == "P"].groupby(["season", "player"])["fielding_runs"].sum().rename("fld_p")
    field_pa = def_pa[def_pa["pos"] != "P"].groupby(["season", "player"])["def_pa"].sum()

    lev = leverage.relief_leverage(leverage.leverage_table(s.events)).rename(columns={"pitcher": "player"})
    bat_stints, pit_stints = _stints(pa)

    keys = ["season", "player"]
    counts = pd.concat(
        [
            bat_stints.groupby(keys)["pa"].sum(),
            pit_stints.groupby(keys)[["bf", "sp_bf", "rp_bf"]].sum(),
        ],
        axis=1,
    )
    p = counts.join(
        [
            batting.set_index(keys)["batting"],
            pitching.set_index(keys)[["pitch", "pitch_a", "pitch_b", "pitch_c"]],
            catcher.set_index(keys)["throwing"],
            steals.set_index(keys)["steal_runs"],
            advance.set_index(keys)["adv_runs"],
            pos_runs.set_index(keys)["pos_runs"],
            fld_hit,
            fld_p,
            field_pa,
            lev.set_index(keys)[["gmli", "relief_bf"]],
        ],
        how="left",
    ).reset_index()
    fill = [c for c in p.columns if c not in ("gmli",)]
    p[fill] = p[fill].fillna(0.0)
    p["baserunning"] = p["steal_runs"] + p["adv_runs"]
    p["fielding"] = p["fld_zone"] + p["throwing"] + p["fld_p"]
    p["hit_raa"] = p["batting"] + p["baserunning"] + p["fld_zone"] + p["throwing"] + p["pos_runs"]
    # League adjustment (as FanGraphs): hitters' runs sum to zero each season,
    # since the positional scale itself does not.
    per_pa = p.groupby("season")["hit_raa"].transform("sum") / p.groupby("season")["pa"].transform("sum")
    p["league_adj"] = -per_pa * p["pa"]
    p["hit_raa"] += p["league_adj"]
    p["pitch_raa"] = p["pitch"] + p["fld_p"]
    relief_share = (p["relief_bf"] / p["bf"].where(p["bf"] > 0)).fillna(0.0)
    p["leverage_runs"] = p["pitch"] * relief_share * ((1.0 + p["gmli"].fillna(1.0)) / 2.0 - 1.0)

    measured = replacement.measured_rates(p, bat_stints, pit_stints)
    totals = p.groupby("season")[["pa", "sp_bf", "rp_bf"]].sum()
    team_games = 2 * games  # games played by all teams together
    level_seasons = [x for x in measured.index if x in DEVELOPMENT_SEASONS] or list(measured.index)
    level = replacement.pooled_level(measured, totals, s.runs_per_win, team_games, level_seasons)
    rates = replacement.season_rates(level, totals, s.runs_per_win, team_games)

    r = rates.reindex(p["season"])
    p["replacement"] = -(
        r["hitter"].to_numpy() * p["pa"] + r["starter"].to_numpy() * p["sp_bf"] + r["reliever"].to_numpy() * p["rp_bf"]
    )
    p["runs_per_win"] = p["season"].map(s.runs_per_win)
    p["rar"] = p["hit_raa"] + p["pitch_raa"] + p["leverage_runs"] + p["replacement"]
    p["war"] = p["rar"] / p["runs_per_win"]

    team = pd.concat(
        [
            bat_stints.rename(columns={"pa": "n"}),
            pit_stints.rename(columns={"bf": "n"})[["season", "team_id", "player", "n"]],
        ]
    )
    team = team.groupby(["season", "player", "team_id"])["n"].sum().reset_index().sort_values("n", ascending=False)
    p = p.merge(team.drop_duplicates(keys)[["season", "player", "team_id"]], on=keys, how="left")
    p = p.merge(primary, on=keys, how="left")
    is_pitcher = p["bf"] > p["pa"]
    p.loc[is_pitcher, "pos"] = np.where(p.loc[is_pitcher, "sp_bf"] >= p.loc[is_pitcher, "rp_bf"], "SP", "RP")
    return WarBuild(p, s, rates, level, pos_scale, event_fielding)
