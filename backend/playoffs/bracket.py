"""LMB postseason rules, the same every season 2021-2026 (league releases on
milb.com, checked against every series in the games cache). All functions
work on N simulated seasons at once.

- The top 6 of each zone by win percentage qualify. Ties: head to head among
  the tied teams, then run differential.
- First round: 1 v 6, 2 v 5, 3 v 4.
- The best loser (mejor perdedor) also advances: the first-round loser with
  the most wins in its series; ties by win percentage, then first-round run
  differential.
- Zone series: the best loser is the fourth seed and plays the top remaining
  seed it did not face in the first round; the other two winners play.
- Every series is best of 7 on a 2-3-2 pattern. The better seed hosts games
  1, 2, 6 and 7; the best loser is the visitor in every later round.
- Serie del Rey: the better regular-season percentage hosts; a best loser is
  the visitor (between two best losers, the better percentage hosts).
"""

import numpy as np

QUALIFIERS = 6
WINS_NEEDED = 4
HOST_GAMES = np.array([True, True, False, False, False, True, True])


def seed_order(pct: np.ndarray, h2h: np.ndarray, run_diff: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Positions within a zone in seed order, best first, shape (N, Z).
    pct, run_diff: (N, Z); h2h: (N, Z, Z) wins of row team over column team."""
    tied = pct[:, :, None] == pct[:, None, :]
    wins_in = (h2h * tied).sum(-1)
    games_in = ((h2h + h2h.transpose(0, 2, 1)) * tied).sum(-1)
    mini = np.divide(wins_in, games_in, out=np.full(pct.shape, 0.5), where=games_in > 0)
    return np.lexsort((rng.random(pct.shape), -run_diff, -mini, -pct), axis=-1)


def best_loser(loser_wins: np.ndarray, loser_pct: np.ndarray, loser_run_diff: np.ndarray, rng) -> np.ndarray:
    """Index (0-2) of the best loser among the three first-round losers, shape (N,)."""
    order = np.lexsort((rng.random(loser_wins.shape), -loser_run_diff, -loser_pct, -loser_wins), axis=-1)
    return order[:, 0]


def zone_series_pairs(winner_seed: np.ndarray, best_loser_opponent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Zone-series pairings from the three first-round winners.
    winner_seed: (N, 3) seed positions (0 best) of the winners;
    best_loser_opponent: (N,) column (0-2) of the winner that beat the best loser.
    Returns (vs_loser, other_pair): (N,) column of the winner that plays the
    best loser, and (N, 2) columns of the two winners that play each other."""
    by_seed = np.argsort(winner_seed, axis=1)
    rematch = by_seed[:, 0] == best_loser_opponent
    vs_loser = np.where(rematch, by_seed[:, 1], by_seed[:, 0])
    other = np.where(rematch[:, None], by_seed[:, [0, 2]], by_seed[:, [1, 2]])
    return vs_loser, other


def play_series(
    rng: np.random.Generator,
    p_host_at_home: np.ndarray,
    p_host_away: np.ndarray,
    host_wins: np.ndarray,
    visitor_wins: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Finish best-of-7 series from their current wins.
    Returns (host_won, loser_wins), each (N,)."""
    n = len(p_host_at_home)
    played = host_wins + visitor_wins
    p = np.where(HOST_GAMES[None, :], p_host_at_home[:, None], p_host_away[:, None])
    host_game = rng.random((n, len(HOST_GAMES))) < p
    future = np.arange(len(HOST_GAMES))[None, :] >= played[:, None]
    host_total = host_wins[:, None] + np.cumsum(host_game & future, axis=1)
    visitor_total = visitor_wins[:, None] + np.cumsum(~host_game & future, axis=1)
    decided = (host_total >= WINS_NEEDED) | (visitor_total >= WINS_NEEDED)
    last = decided.argmax(axis=1)
    rows = np.arange(n)
    host_won = host_total[rows, last] >= WINS_NEEDED
    loser_wins = np.minimum(host_total, visitor_total)[rows, last]
    return host_won, loser_wins
