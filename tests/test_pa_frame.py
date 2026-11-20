"""The plate-appearance frame is the only source of every skill and table
fit, so it must reproduce the official box score of a real game."""

import json
from pathlib import Path

from backend.etl.pa_frame import pa_rows

FIXTURE = Path(__file__).parent / "fixtures" / "feed_809876.json"


def test_pa_rows_match_boxscore():
    feed = json.loads(FIXTURE.read_text())
    pa = pa_rows(feed)
    box = feed["liveData"]["boxscore"]["teams"]
    runs = feed["liveData"]["linescore"]["teams"]
    for side, half in (("away", "top"), ("home", "bottom")):
        rows = pa[pa["inning_topbot"] == half]
        batting = box[side]["teamStats"]["batting"]
        counts = rows["outcome"].value_counts()
        assert counts.get("HR", 0) == batting["homeRuns"]
        assert counts.get("BB", 0) == batting["baseOnBalls"]
        assert counts.get("HBP", 0) == batting["hitByPitch"]
        assert counts.get("K", 0) == batting["strikeOuts"]
        hits = sum(counts.get(o, 0) for o in ("1B", "2B", "3B", "HR"))
        assert hits == batting["hits"]
        assert rows["post_bat_score"].max() == runs[side]["runs"]
