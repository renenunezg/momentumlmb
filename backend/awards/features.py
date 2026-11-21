"""One candidate pool per award and race, built from regular-season lines only.

Counting inputs are put on a 100-team-game scale, since LMB seasons ran from 61
to 120 games a club. Hitting runs and pitching runs saved are measured against
the league with no park adjustment: voters judge the raw line, and adjusting for
park (LMB parks run from sea level to 2,230 m) made every award's held-out log
loss worse or no better (2005-2025, 2026-11-20: POY 1.109 raw vs 1.392 adjusted,
REL 0.963 vs 1.025, ROY 3.078 vs 3.231, MVP 1.232 vs 1.224).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np
import pandas as pd

from backend.awards import data

# Linear weights for the hitting runs input; only relative values matter to the
# race, so MLB's weights serve (LMB's run scale is absorbed by the league rate).
WEIGHTS = {"ubb": 0.69, "hbp": 0.72, "single": 0.89, "doubles": 1.27, "triples": 1.62, "homeRuns": 2.10}
TEAM_GAMES = 100.0
MIN_MVP_PA = 250.0
MIN_POY_IP = 50.0
MIN_REL_GAMES = 25.0
MAX_REL_START_SHARE = 0.2
# A rookie has at most 130 earlier LMB plate appearances and 60 earlier innings (every
# winner since 2005 fits); history starts in 2005, so rookie races start in 2007.
MAX_ROOKIE_PRIOR_PA = 130.0
MAX_ROOKIE_PRIOR_OUTS = 180.0
MIN_ROOKIE_PA = 100.0
MIN_ROOKIE_IP = 30.0
FIRST_ROOKIE_YEAR = 2007
FIRST_MANAGER_YEAR = 2006

FEATURES = {
    "MVP": ["bat_runs", "hr", "rbi", "avg", "win_pct"],
    "POY": ["wins", "runs_saved", "k", "ip"],
    "REL": ["saves", "runs_saved", "k", "holds"],
    "ROY": ["value", "playing_time", "is_pitcher", "age", "mexico_born"],
    "MOY": ["win_pct", "win_pct_change"],
}
# The single input each award is compared against.
BASELINE = {"MVP": "bat_runs", "POY": "wins", "REL": "saves", "ROY": "value", "MOY": "win_pct"}
STAT_LINE = {
    "MVP": lambda r: f"{r.real_hr:.0f} HR, {r.real_rbi:.0f} RBI, .{round(r.avg * 1000):03d} AVG, {r.real_pa:.0f} PA",
    "POY": lambda r: f"{r.real_wins:.0f}-{r.real_losses:.0f}, {r.era:.2f} ERA, {r.real_k:.0f} K, {r.real_ip:.1f} IP",
    "REL": lambda r: f"{r.real_saves:.0f} SV, {r.real_holds:.0f} HLD, {r.era:.2f} ERA, {r.real_ip:.1f} IP",
    "ROY": lambda r: (
        f"{r.real_wins:.0f}-{r.real_losses:.0f}, {r.era:.2f} ERA, {r.real_ip:.1f} IP"
        if r.is_pitcher
        else f"{r.real_hr:.0f} HR, .{round(r.avg * 1000):03d} AVG, {r.real_pa:.0f} PA"
    ),
    "MOY": lambda r: f"{r.real_wins:.0f}-{r.real_losses:.0f} ({r.win_pct:.3f})",
}


@dataclass(frozen=True)
class Race:
    key: str  # API season ("2018.1") or year.
    year: int
    pool: pd.DataFrame
    x: np.ndarray
    y: np.ndarray  # Winner share, summing to one when a winner is in the pool, else all zero.


def _season_scale(year: int) -> float:
    s = data.standings(year)
    return TEAM_GAMES / float((s.wins + s.losses).mean())


def _hitting(frame: pd.DataFrame) -> pd.DataFrame:
    h = frame.copy()
    h["ubb"] = h.baseOnBalls - h.intentionalWalks
    h["hbp"] = h.hitByPitch
    h["single"] = h.hits - h.doubles - h.triples - h.homeRuns
    raw = sum(w * h[c] for c, w in WEIGHTS.items())
    league_rate = raw.sum() / h.plateAppearances.sum()
    h["bat_runs_raw"] = raw - league_rate * h.plateAppearances
    h["avg"] = (h.hits / h.atBats.where(h.atBats > 0)).fillna(0.0)
    return h


def _pitching(frame: pd.DataFrame) -> pd.DataFrame:
    p = frame.copy()
    league_era = 27.0 * p.earnedRuns.sum() / p.outs.sum()
    p["era"] = 27.0 * p.earnedRuns / p.outs.clip(lower=1)
    p["runs_saved_raw"] = (league_era * p.outs - 27.0 * p.earnedRuns) / 27.0
    return p


def season_lines(year: int, key: str | None = None) -> pd.DataFrame:
    """Hitting and pitching per player for a year (or one 2018 season), with the club's record."""
    if key is None:
        hit, pit = data.year_lines(year, "hitting"), data.year_lines(year, "pitching")
    else:
        hit, pit = data.lines(key, "hitting"), data.lines(key, "pitching")
    scale = _season_scale(year)
    if key is not None and key != str(year):
        scale *= 2.0  # Half of the year's games.
    h = _hitting(hit)
    p = _pitching(pit)
    h = pd.DataFrame(
        {
            "player_id": h.player_id,
            "player_name": h.player_name,
            "team_id": h.team_id,
            "team": h.team,
            "real_pa": h.plateAppearances,
            "real_hr": h.homeRuns,
            "real_rbi": h.rbi,
            "avg": h.avg,
            "bat_runs": h.bat_runs_raw * scale,
        }
    )
    p = pd.DataFrame(
        {
            "player_id": p.player_id,
            "p_name": p.player_name,
            "p_team_id": p.team_id,
            "p_team": p.team,
            "real_ip": p.outs / 3.0,
            "real_outs": p.outs,
            "real_k": p.strikeOuts,
            "real_wins": p.wins,
            "real_losses": p.losses,
            "real_saves": p.saves,
            "real_holds": p.holds,
            "real_games": p.gamesPitched,
            "real_starts": p.gamesStarted,
            "era": p.era,
            "runs_saved": p.runs_saved_raw * scale,
        }
    )
    f = h.merge(p, on="player_id", how="outer")
    f["player_name"] = f.player_name.fillna(f.p_name)
    f["team_id"] = f.team_id.fillna(f.p_team_id)
    f["team"] = f.team.fillna(f.p_team)
    f = f.drop(columns=["p_name", "p_team_id", "p_team"])
    zero = [
        "real_pa",
        "real_hr",
        "real_rbi",
        "avg",
        "bat_runs",
        "real_ip",
        "real_outs",
        "real_k",
        "real_wins",
        "real_losses",
        "real_saves",
        "real_holds",
        "real_games",
        "real_starts",
        "runs_saved",
    ]
    f[zero] = f[zero].fillna(0.0)
    f["era"] = f.era.fillna(0.0)
    for real, scaled in (
        ("real_pa", "pa"),
        ("real_hr", "hr"),
        ("real_rbi", "rbi"),
        ("real_ip", "ip"),
        ("real_k", "k"),
        ("real_wins", "wins"),
        ("real_saves", "saves"),
        ("real_holds", "holds"),
        ("real_games", "games"),
    ):
        f[scaled] = f[real] * scale
    record = data.standings(year).set_index("team_id").win_pct
    f["win_pct"] = f.team_id.map(record)
    if f.loc[f.pa.ge(MIN_MVP_PA) | f.ip.ge(MIN_POY_IP), "win_pct"].isna().any():
        raise ValueError(f"Club record missing for a candidate in {key or year}")
    f["team_id"] = f.team_id.astype(int)
    return f.assign(year=year)


@cache
def _career_before(year: int) -> pd.DataFrame:
    """Each player's LMB plate appearances and outs in all years before this one."""
    frames = [
        pd.concat(
            [
                data.year_lines(y, "hitting")[["player_id", "plateAppearances"]],
                data.year_lines(y, "pitching")[["player_id", "outs"]],
            ]
        )
        for y in range(data.FIRST_YEAR, year)
        if y != 2020
    ]
    if not frames:
        return pd.DataFrame(columns=["plateAppearances", "outs"])
    return pd.concat(frames).fillna(0.0).groupby("player_id")[["plateAppearances", "outs"]].sum()


def pool(award: str, year: int, key: str | None = None) -> pd.DataFrame:
    if award == "MOY":
        return _manager_pool(year)
    f = season_lines(year, key)
    if award == "MVP":
        f = f[f.pa >= MIN_MVP_PA]
    elif award == "POY":
        f = f[f.ip >= MIN_POY_IP]
    elif award == "REL":
        f = f[(f.games >= MIN_REL_GAMES) & (f.real_starts <= MAX_REL_START_SHARE * f.real_games)]
    elif award == "ROY":
        f = _rookies(f, year)
    else:
        raise ValueError(f"Unknown award {award}")
    if f.empty:
        raise ValueError(f"Empty {award} pool for {key or year}")
    return f.reset_index(drop=True)


def _rookies(f: pd.DataFrame, year: int) -> pd.DataFrame:
    if year < FIRST_ROOKIE_YEAR:
        raise ValueError(f"No rookie race before {FIRST_ROOKIE_YEAR}")
    before = _career_before(year).reindex(f.player_id).fillna(0.0)
    eligible = (before.plateAppearances.to_numpy() <= MAX_ROOKIE_PRIOR_PA) & (
        before.outs.to_numpy() <= MAX_ROOKIE_PRIOR_OUTS
    )
    f = f[eligible & ((f.pa >= MIN_ROOKIE_PA) | (f.ip >= MIN_ROOKIE_IP)).to_numpy()].copy()
    f["is_pitcher"] = (f.ip > f.pa / 4.0).astype(float)
    f["value"] = np.where(f.is_pitcher.eq(1.0), f.runs_saved, f.bat_runs)
    f["playing_time"] = np.where(f.is_pitcher.eq(1.0), f.ip / 150.0, f.pa / 450.0)
    people = data.people(f.player_id.astype(int).tolist()).set_index("player_id")
    born = pd.to_datetime(f.player_id.map(people.birth_date))
    f["age"] = (pd.Timestamp(f"{year}-07-01") - born).dt.days / 365.25
    if f.age.isna().any():
        f["age"] = f.age.fillna(f.age.median())
    f["mexico_born"] = f.player_id.map(people.birth_country).eq("Mexico").astype(float)
    return f


def _manager_pool(year: int) -> pd.DataFrame:
    if year < FIRST_MANAGER_YEAR:
        raise ValueError(f"No manager race before {FIRST_MANAGER_YEAR}")
    record = data.standings(year)
    last = data.standings(year - 1 if year != 2021 else 2019).set_index("team_id").win_pct
    f = data.managers(year).merge(record, on="team_id")
    f["win_pct_change"] = f.win_pct - f.team_id.map(last).fillna(0.5)
    f["real_wins"], f["real_losses"] = f.wins, f.losses
    f["team"] = f.team_id.map(data.year_lines(year, "hitting").drop_duplicates("team_id").set_index("team_id").team)
    # A manager listed by two clubs is judged on the better one.
    f = f.sort_values("win_pct", ascending=False).drop_duplicates("player_id")
    return f.assign(year=year).reset_index(drop=True)


def races(award: str, first_year: int, last_year: int) -> list[Race]:
    """Every race with a known winner from first_year to last_year (both included)."""
    winners = data.recipients(award)
    out = []
    for year in (y for y in range(first_year, last_year + 1) if y != 2020):
        keys = ["2018.1", "2018.2"] if award == "MVP" and year == 2018 else [str(year)]
        for key in keys:
            frame = pool(award, year, key if key != str(year) else None)
            won = winners.player_id[winners.key.eq(key)]
            y = frame.player_id.isin(won).to_numpy(float)
            out.append(
                Race(key, year, frame, frame[FEATURES[award]].to_numpy(float), y / y.sum() if y.sum() > 0 else y)
            )
    return out
