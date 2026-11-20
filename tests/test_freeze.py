"""A started game is never rescored: its frozen forecast and pick are what
is displayed and graded."""

from datetime import UTC, datetime, timedelta

import pandas as pd

from backend.scoring import games_to_score


def test_started_game_not_rescored():
    now = datetime(2027, 5, 2, 1, 0, tzinfo=UTC)
    games = pd.DataFrame(
        {
            "game_pk": [1, 2],
            "start_time_utc": [pd.Timestamp(now - timedelta(minutes=1)), pd.Timestamp(now + timedelta(hours=1))],
            "status": ["Scheduled", "Scheduled"],
        }
    )
    assert games_to_score(games, now, freeze_started=True).game_pk.tolist() == [2]
