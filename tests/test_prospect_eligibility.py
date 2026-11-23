"""Regression tests for prospect eligibility, graduation and impact outcomes."""

import pandas as pd

from backend.prospects import snapshots


def _line(player, season, source, group="hitting", **kw):
    base = {
        "player": player,
        "name": f"P{player}",
        "team": 1,
        "league": 125 if source == "lmb" else (132 if source == "lmp" else 112),
        "sport": 23 if source == "lmb" else (17 if source == "lmp" else 11),
        "group": group,
        "pa": 0,
        "ab": 0,
        "h": 0,
        "d2": 0,
        "d3": 0,
        "hr": 0,
        "bb": 0,
        "hbp": 0,
        "so": 0,
        "bf": 0,
        "outs": 0,
        "gs": 0,
        "g": 10,
        "source": source,
        "season": season,
    }
    base.update(kw)
    return base


def test_eligibility_and_graduation_boundaries():
    lines = pd.DataFrame(
        [
            # 1: Mexican-born MiLB player stored as "MEX" -> eligible
            _line(1, 2026, "milb", pa=200, ab=180, h=55, d2=10, hr=5, bb=20, so=35),
            # 2: Non-Mexican LMP-only rental (never played LMB) -> excluded
            _line(2, 2026, "lmp", pa=120, ab=105, h=30, bb=15, so=25),
            # 3: Non-Mexican who actually played LMB -> eligible
            _line(3, 2026, "lmb", pa=80, ab=70, h=20, bb=10, so=15),
            # 4: Mexican-born who already debuted in MLB in 2025 -> graduated (excluded)
            _line(4, 2026, "milb", pa=300, ab=270, h=80, d2=15, hr=10, bb=30, so=50),
            # 5: Mexican-born who already had an LMB regular season in 2025 -> graduated (excluded)
            _line(5, 2025, "lmb", pa=260, ab=230, h=60, bb=30, so=40),
            _line(5, 2026, "lmb", pa=100, ab=90, h=25, bb=10, so=20),
        ]
    )
    people = pd.DataFrame(
        [
            {
                "player": 1,
                "full_name": "Mex Short",
                "birth_date": "2004-03-01",
                "birth_country": "MEX",
                "position": "SS",
                "throws": "R",
                "mlb_debut": None,
            },
            {
                "player": 2,
                "full_name": "Lmp Rental",
                "birth_date": "2003-03-01",
                "birth_country": "USA",
                "position": "OF",
                "throws": "R",
                "mlb_debut": None,
            },
            {
                "player": 3,
                "full_name": "Lmb Foreign",
                "birth_date": "2003-03-01",
                "birth_country": "Dominican Republic",
                "position": "CF",
                "throws": "R",
                "mlb_debut": None,
            },
            {
                "player": 4,
                "full_name": "Mlb Grad",
                "birth_date": "2003-03-01",
                "birth_country": "Mexico",
                "position": "2B",
                "throws": "R",
                "mlb_debut": "2025-08-10",
            },
            {
                "player": 5,
                "full_name": "Lmb Grad",
                "birth_date": "2003-03-01",
                "birth_country": "Mexico",
                "position": "1B",
                "throws": "R",
                "mlb_debut": None,
            },
        ]
    )
    team_games = pd.Series({2025: 90.0, 2026: 90.0})
    snaps = snapshots.build(lines, people, team_games, [2026], [2025, 2026])
    assert set(snaps.player) == {1, 3}
    assert snaps.set_index("player").loc[1, "mexican"] == 1.0
    assert snaps.set_index("player").loc[3, "mexican"] == 0.0
