"""Monte Carlo of the rest of an LMB season: the remaining regular-season
games, the seeding and the postseason bracket, from a SeasonState and team
Ratings. Each simulated season draws its own ratings, so the uncertainty in a
team's strength is shared across all of its remaining games."""

from dataclasses import dataclass

import numpy as np
from scipy.special import expit

from backend.playoffs import bracket
from backend.playoffs.state import ROUNDS, ZONES, SeasonState
from backend.playoffs.strength import Ratings

STAGES = ("playoffs", "zone_series", "zone_final", "final", "champion")
# Run differential only breaks ties left after percentage and head to head;
# a simulated net win is worth this many runs of differential.
RUNS_PER_NET_WIN = 3.0


@dataclass(frozen=True)
class Odds:
    teams: np.ndarray
    reach: np.ndarray  # (T, len(STAGES)) probability of reaching each stage
    expected_wins: np.ndarray  # (T,) mean final regular-season wins
    expected_losses: np.ndarray
    n_sims: int


def _counts(index: np.ndarray, n_sims: int, size: int) -> np.ndarray:
    return np.bincount(index.ravel(), minlength=n_sims * size).reshape(n_sims, size)


def _regular_season(state: SeasonState, r: np.ndarray, home: float, rng):
    n_sims, n = r.shape
    rh, ra = state.remaining_home, state.remaining_away
    home_won = rng.random((n_sims, len(rh))) < expit(home + r[:, rh] - r[:, ra])
    winner = np.where(home_won, rh, ra)
    loser = np.where(home_won, ra, rh)
    sims = np.arange(n_sims)[:, None]
    wins = state.wins + _counts(sims * n + winner, n_sims, n)
    losses = state.losses + _counts(sims * n + loser, n_sims, n)
    h2h = state.h2h + _counts(sims * n * n + winner * n + loser, n_sims, n * n).reshape(n_sims, n, n)
    run_diff = state.run_diff + RUNS_PER_NET_WIN * (wins - state.wins - (losses - state.losses))
    return wins, losses, h2h, run_diff


def _known(state: SeasonState, rnd: str, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    table = np.zeros((state.n, state.n), dtype=int)
    for (r, i, j), w in state.series.items():
        if r == rnd:
            table[i, j] = w
    return table[a, b], table[b, a]


def _series(state, rnd, host, visitor, r, home, rng):
    rows = np.arange(len(host))
    diff = r[rows, host] - r[rows, visitor]
    host_wins, visitor_wins = _known(state, rnd, host, visitor)
    host_won, loser_wins = bracket.play_series(rng, expit(home + diff), expit(diff - home), host_wins, visitor_wins)
    return np.where(host_won, host, visitor), np.where(host_won, visitor, host), loser_wins


def simulate(state: SeasonState, ratings: Ratings, n_sims: int, rng: np.random.Generator) -> Odds:
    r = ratings.draw(rng, n_sims)
    home = ratings.home
    wins, losses, h2h, run_diff = _regular_season(state, r, home, rng)
    pct = wins / np.maximum(wins + losses, 1)
    rows = np.arange(n_sims)
    reach = np.zeros((n_sims, state.n, len(STAGES)), dtype=bool)

    def mark(stage: str, teams: np.ndarray) -> None:
        reach[rows[:, None] if teams.ndim == 2 else rows, teams, STAGES.index(stage)] = True

    champions, champion_was_loser = [], []
    for z in range(len(ZONES)):
        members = np.flatnonzero(state.zone == z)
        order = bracket.seed_order(pct[:, members], h2h[:, members][:, :, members], run_diff[:, members], rng)
        seeds = members[order[:, : bracket.QUALIFIERS]]  # (N, 6) team indexes, best first
        mark("playoffs", seeds)

        winners, losers, loser_wins = [], [], []
        for k in range(3):
            w, l_, lw = _series(state, ROUNDS[0], seeds[:, k], seeds[:, 5 - k], r, home, rng)
            winners.append(w), losers.append(l_), loser_wins.append(lw)
        winners, losers, loser_wins = np.stack(winners, 1), np.stack(losers, 1), np.stack(loser_wins, 1)
        seed_pos = np.argsort(order, axis=1)  # position in zone -> seed rank
        rank = np.full((n_sims, state.n), 99)
        rank[rows[:, None], members[None, :]] = seed_pos
        winner_seed = rank[rows[:, None], winners]
        lucky = bracket.best_loser(
            loser_wins,
            pct[rows[:, None], losers],
            state.series_run_diff[losers] + rng.normal(0, 1e-3, losers.shape),
            rng,
        )
        loser_team = losers[rows, lucky]
        vs_loser, other = bracket.zone_series_pairs(winner_seed, lucky)
        mark("zone_series", winners)
        mark("zone_series", loser_team)

        # Zone series: a winner hosts the best loser; the other pair by seed.
        top = winners[rows, vs_loser]
        a, b = winners[rows, other[:, 0]], winners[rows, other[:, 1]]
        a_hosts = rank[rows, a] < rank[rows, b]
        w1, _, _ = _series(state, ROUNDS[1], top, loser_team, r, home, rng)
        w2, _, _ = _series(state, ROUNDS[1], np.where(a_hosts, a, b), np.where(a_hosts, b, a), r, home, rng)
        mark("zone_final", w1)
        mark("zone_final", w2)

        # Zone final: the best loser visits; otherwise the better seed hosts.
        w1_lucky = w1 == loser_team
        w1_hosts = ~w1_lucky & (rank[rows, w1] < rank[rows, w2])
        champ, _, _ = _series(state, ROUNDS[2], np.where(w1_hosts, w1, w2), np.where(w1_hosts, w2, w1), r, home, rng)
        mark("final", champ)
        champions.append(champ)
        champion_was_loser.append(champ == loser_team)

    norte, sur = champions
    norte_lucky, sur_lucky = champion_was_loser
    p_n, p_s = pct[rows, norte], pct[rows, sur]
    better = np.where(p_n == p_s, rng.random(n_sims) < 0.5, p_n > p_s)
    norte_hosts = np.where(norte_lucky == sur_lucky, better, sur_lucky)
    host, visitor = np.where(norte_hosts, norte, sur), np.where(norte_hosts, sur, norte)
    title, _, _ = _series(state, ROUNDS[3], host, visitor, r, home, rng)
    mark("champion", title)

    return Odds(
        teams=state.teams,
        reach=reach.mean(axis=0),
        expected_wins=wins.mean(axis=0),
        expected_losses=losses.mean(axis=0),
        n_sims=n_sims,
    )
