"""The best loser decides who meets whom in the zone series, and it is the
rule most easily gotten wrong: it advances on series wins, drops to the fourth
seed and must not replay the team that beat it. Cases are real brackets."""

import numpy as np

from backend.playoffs.bracket import best_loser, play_series, zone_series_pairs

RNG = np.random.default_rng(0)


def test_best_loser_reseeding_2025_norte_and_2026_sur():
    # Series columns are 1v6, 2v5, 3v4. 2025 Norte: Jalisco (6) beat Monterrey (1) 4-3,
    # Laguna (5) beat Tijuana (2) 4-2, Laredo (4) beat Monclova (3) 4-3. Monterrey and
    # Monclova both won 3; Monterrey's better percentage makes it the best loser, and it
    # plays the top remaining seed, Laredo, while Laguna plays Jalisco.
    loser_wins = np.array([[3, 2, 3]])
    lucky = best_loser(loser_wins, np.array([[0.598, 0.587, 0.567]]), np.zeros((1, 3)), RNG)
    assert lucky.tolist() == [0]
    vs_loser, other = zone_series_pairs(np.array([[5, 4, 3]]), lucky)
    assert vs_loser.tolist() == [2] and sorted(other[0].tolist()) == [0, 1]

    # 2026 Sur: Diablos (1) beat Oaxaca (6) 4-3, Tabasco (2) beat Leon (5) 4-2, Puebla (4)
    # beat Campeche (3) 4-1. Oaxaca is the best loser but already lost to Diablos, so it
    # plays Tabasco and Diablos plays Puebla.
    lucky = best_loser(np.array([[3, 2, 1]]), np.array([[0.516, 0.538, 0.559]]), np.zeros((1, 3)), RNG)
    assert lucky.tolist() == [0]
    vs_loser, other = zone_series_pairs(np.array([[0, 1, 3]]), lucky)
    assert vs_loser.tolist() == [1] and sorted(other[0].tolist()) == [0, 2]


def test_series_resumes_from_its_current_wins():
    certain, never = np.ones(2), np.zeros(2)
    host_won, loser_wins = play_series(RNG, certain, certain, np.array([3, 4]), np.array([1, 2]))
    assert host_won.tolist() == [True, True] and loser_wins.tolist() == [1, 2]
    host_won, loser_wins = play_series(RNG, never, never, np.array([3, 0]), np.array([0, 0]))
    assert host_won.tolist() == [False, False] and loser_wins.tolist() == [3, 0]
