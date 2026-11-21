"""The season as it stands on the morning of a date: regular-season results
before the date, the regular-season games still to play, and the postseason
series played so far. Zones come from data_static/zones.csv (the Stats API
standings by season); teams.csv holds only the current league."""

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from backend.config import STATIC_DIR

# A forfeit is left out: its schedule score need not name the winner (2022 lists
# the forfeiting home team ahead 3-2, 2023 lists 0-0), so each forfeit leaves
# its two teams one game short of the official standings.
PLAYED = {"Final", "Completed Early"}
ZONES = ("Norte", "Sur")
ROUNDS = ("F", "D", "L", "W")  # first round, zone series, zone final, Serie del Rey


def zones(season: int) -> pd.Series:
    frame = pd.read_csv(STATIC_DIR / "zones.csv")
    frame = frame[frame.season == season]
    if frame.empty:
        raise ValueError(f"No zones for {season} in zones.csv: add the season's Norte/Sur teams (Stats API standings)")
    return frame.set_index("team_id").zone.sort_index()


def title_decided_on(games: pd.DataFrame, season: int) -> date | None:
    """Date of the game that won the Serie del Rey, or None while it is open."""
    final = games[(games.season == season) & (games.game_type == "W") & games.status.isin(PLAYED)]
    final = final.sort_values("start_time_utc")
    winners = np.where(final.home_score > final.away_score, final.home_team_id, final.away_team_id)
    clinched = pd.Series(winners).groupby(winners).cumcount().to_numpy() + 1 >= 4
    return final.official_date.iloc[clinched.argmax()] if clinched.any() else None


@dataclass(frozen=True)
class SeasonState:
    season: int
    as_of: date
    teams: np.ndarray  # team ids, ascending
    zone: np.ndarray  # index into ZONES per team
    wins: np.ndarray
    losses: np.ndarray
    run_diff: np.ndarray
    h2h: np.ndarray  # h2h[i, j]: regular-season wins of team i over team j
    remaining_home: np.ndarray  # team indexes of regular-season games still to play
    remaining_away: np.ndarray
    series: dict  # (round, i, j) -> wins of team i over team j in that postseason round so far
    series_run_diff: np.ndarray  # first-round run differential per team so far

    @property
    def regular_done(self) -> bool:
        return len(self.remaining_home) == 0

    @property
    def n(self) -> int:
        return len(self.teams)


def _winner_rows(games: pd.DataFrame) -> pd.DataFrame:
    games = games[games.status.isin(PLAYED) & (games.home_score != games.away_score)]
    return games.dropna(subset=["home_score", "away_score"])


def build_state(games: pd.DataFrame, season: int, as_of: date) -> SeasonState:
    """`games`: the season's schedule frame (backend.etl.games columns)."""
    zone_of = zones(season)
    teams = zone_of.index.to_numpy()
    idx = {int(t): i for i, t in enumerate(teams)}
    games = games[games.season == season]
    unknown = (set(games.home_team_id) | set(games.away_team_id)) - set(idx)
    if unknown:
        raise ValueError(f"Teams missing from zones.csv for {season}: {sorted(unknown)}")
    regular = games[games.game_type == "R"]
    before = regular[regular.official_date < as_of]
    played = _winner_rows(before)
    later = regular[(regular.official_date >= as_of)]
    postseason_started = (games[games.game_type != "R"].official_date < as_of).any()
    if postseason_started:
        later = later.iloc[0:0]

    n = len(teams)
    h, a = played.home_team_id.map(idx).to_numpy(), played.away_team_id.map(idx).to_numpy()
    home_won = (played.home_score > played.away_score).to_numpy()
    winner, loser = np.where(home_won, h, a), np.where(home_won, a, h)
    margin = (played.home_score - played.away_score).to_numpy()
    h2h = np.zeros((n, n), dtype=np.int32)
    np.add.at(h2h, (winner, loser), 1)
    run_diff = np.zeros(n)
    np.add.at(run_diff, h, margin)
    np.add.at(run_diff, a, -margin)

    series, series_rd = {}, np.zeros(n)
    post = _winner_rows(games[(games.game_type != "R") & (games.official_date < as_of)])
    for g in post.itertuples(index=False):
        hi, ai = idx[int(g.home_team_id)], idx[int(g.away_team_id)]
        w, l_ = (hi, ai) if g.home_score > g.away_score else (ai, hi)
        series[(g.game_type, w, l_)] = series.get((g.game_type, w, l_), 0) + 1
        series.setdefault((g.game_type, l_, w), 0)
        if g.game_type == "F":
            series_rd[hi] += g.home_score - g.away_score
            series_rd[ai] -= g.home_score - g.away_score

    return SeasonState(
        season=season,
        as_of=as_of,
        teams=teams,
        zone=np.array([ZONES.index(z) for z in zone_of.to_numpy()]),
        wins=np.bincount(winner, minlength=n),
        losses=np.bincount(loser, minlength=n),
        run_diff=run_diff,
        h2h=h2h,
        remaining_home=later.home_team_id.map(idx).to_numpy(),
        remaining_away=later.away_team_id.map(idx).to_numpy(),
        series=series,
        series_run_diff=series_rd,
    )
