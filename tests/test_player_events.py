"""The valuation event parser rebuilds base-out state, runs and the defensive
alignment play by play; every WAR component rests on it, so a real game must
come out whole: every run accounted for, outs never going backward, and all
eight fielders behind the pitcher on every plate appearance."""

import gzip
import json
from pathlib import Path

from backend.players.events import FIELD_POSITIONS, game_events

FIXTURE = Path(__file__).parent / "fixtures" / "feed_809800.json.gz"


def test_game_events_account_for_runs_outs_and_fielders():
    feed = json.loads(gzip.decompress(FIXTURE.read_bytes()))
    events, _ = game_events(feed)
    runs = feed["liveData"]["linescore"]["teams"]
    for side, half in (("away", "top"), ("home", "bottom")):
        rows = [e for e in events if e["half"] == half]
        assert sum(e["runs"] for e in rows) == runs[side]["runs"]
    for e in events:
        assert e["outs_after"] >= e["outs_before"]
        if e["kind"] == "pa":
            fielders = [e[f"def_{p}"] for p in FIELD_POSITIONS]
            assert None not in fielders and len(set(fielders)) == len(FIELD_POSITIONS)
