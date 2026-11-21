"""The Stats API files the 2022 Rookie and Comeback winners under season "2002" and
splits the 2018 MVP by half; a misplaced winner would train a board on the wrong race."""

from backend.awards import data


def test_recipients_place_misfiled_and_half_season_winners(monkeypatch):
    payload = {
        "awards": [
            {"season": "2002", "date": "2022-11-19", "player": {"id": 1, "nameFirstLast": "A"}},
            {
                "season": "2018",
                "date": "2018-11-20",
                "notes": "First Half MVP",
                "player": {"id": 2, "nameFirstLast": "B"},
            },
            {
                "season": "2018",
                "date": "2018-11-20",
                "notes": "Second Half MVP",
                "player": {"id": 3, "nameFirstLast": "C"},
            },
            {"season": "2025", "date": "2025-10-28", "player": {"id": 4, "nameFirstLast": "D"}},
        ]
    }
    monkeypatch.setattr(data, "_cached", lambda *args, **kwargs: payload)
    rows = data.recipients("MVP").set_index("player_id")
    assert rows.loc[1, "year"] == 2022 and rows.loc[1, "key"] == "2022"
    assert rows.loc[2, "key"] == "2018.1" and rows.loc[3, "key"] == "2018.2"
    assert rows.loc[4, "key"] == "2025"
