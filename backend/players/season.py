"""Regular-season events with their run values, and the per-season run scale."""

from dataclasses import dataclass

import pandas as pd

from backend.players.baserunning import runner_marginals
from backend.players.events import load_events
from backend.players.runs import season_scale


@dataclass(frozen=True)
class Seasons:
    events: pd.DataFrame  # every regular-season event, with re24
    moves: pd.DataFrame
    lw: pd.Series  # (season, outcome) -> run value
    runs_per_win: dict[int, float]
    runs_per_game: dict[int, float]
    marginals: dict[int, dict]

    @property
    def pa(self) -> pd.DataFrame:
        return self.events[self.events["kind"] == "pa"]

    @property
    def br(self) -> pd.DataFrame:
        return self.events[self.events["kind"] == "br"]


def load_seasons(seasons: list[int]) -> Seasons:
    ev, mv = load_events(seasons)
    ev = ev[ev["game_type"] == "R"]
    frames, lws, rpw, rpg, marg = [], {}, {}, {}, {}
    for season, block in ev.groupby("season"):
        sc = season_scale(block)
        frames.append(sc["events"])
        lws[season] = sc["lw"]
        rpw[season] = sc["runs_per_win"]
        rpg[season] = sc["runs_per_game"]
        marg[season] = runner_marginals(sc["re"])
    events = pd.concat(frames, ignore_index=True)
    lw = pd.concat(lws, names=["season", "outcome"])
    moves = mv[mv["event_id"].isin(events["event_id"])]
    return Seasons(events, moves, lw, rpw, rpg, marg)
