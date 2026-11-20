"""Pregame pitching queues from recent workloads (ported from mlbmodel
v2/simulator/bullpen.py, without MLB's opener reports and playoff roster
plumbing; LMB postseason games use the regular-season rules)."""

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

# Rest eligibility: outs thrown over the last one and two days.
ELIG_OUTS_1D = 6
ELIG_OUTS_2D = 9
# Starter pull thresholds (checked against LMB starter outs in Task 6).
PULL_RUNS_HARD = 6
PULL_OUTS_RUNS = 12
PULL_RUNS_AT_OUTS = 4
PULL_PA_PROXY = 24
RELIEVER_PULL_RUNS = 3
STARTER_HISTORY = 8
WORKLOAD_LOOKBACK_DAYS = 60


@dataclass
class BullpenQueue:
    """Starter first, then relievers in order; 0 is a league-average fallback arm."""

    starter: int
    relievers: list[int]
    starter_role: int = 0
    workloads: dict[int, tuple[int, ...]] = field(default_factory=dict)
    usage_known: bool = False

    def outs_samples(self, pitcher_id: int) -> tuple[int, ...]:
        # An unknown workload never gets a starter's length.
        return self.workloads.get(pitcher_id, (3,))


def queue_from_workload(
    game_date: date, team_id: int, starter_id: int, workload: pd.DataFrame, roster: list[int]
) -> BullpenQueue:
    """Regular starters draw their workload from their last starts; proven
    relievers (relief in each of their last five appearances) who are rested
    form the queue, most-used first. `workload` has official_date, team_id,
    pitcher_id, is_starter, outs."""
    wl = workload[(workload.team_id == team_id) & (workload.official_date < game_date)]
    wl = wl[wl.official_date >= max(date(game_date.year, 1, 1), game_date - timedelta(days=WORKLOAD_LOOKBACK_DAYS))]
    wl = wl.sort_values("official_date")
    recent = wl[wl.official_date >= game_date - timedelta(days=2)]
    yesterday = recent[recent.official_date == game_date - timedelta(days=1)].groupby("pitcher_id").outs.sum()
    two_days = recent.groupby("pitcher_id").outs.sum()
    histories = {int(pid): g.tail(STARTER_HISTORY) for pid, g in wl.groupby("pitcher_id")}

    starter_history = histories.get(starter_id, wl.iloc[:0])
    starts = starter_history[starter_history.is_starter]
    regular_starter = len(starts) >= 2 and len(starts) >= len(starter_history) / 2
    workloads = {starter_id: tuple(int(np.clip(x, 3, 21)) for x in starts.outs) if regular_starter else (3,)}

    candidates = []
    for pid in dict.fromkeys(roster):
        if pid == starter_id:
            continue
        hist = histories.get(pid, wl.iloc[:0])
        regular_reliever = len(hist) >= 2 and (~hist.tail(5).is_starter).all()
        if not regular_reliever:
            continue
        if yesterday.get(pid, 0) >= ELIG_OUTS_1D or two_days.get(pid, 0) >= ELIG_OUTS_2D:
            continue
        candidates.append((pid, len(hist), int(two_days.get(pid, 0))))
        workloads[pid] = tuple(int(np.clip(x, 1, 6)) for x in hist.outs) + (3, 3, 3)
    candidates.sort(key=lambda c: (-c[1], c[2], c[0]))
    return BullpenQueue(
        starter=starter_id,
        relievers=[c[0] for c in candidates] + [0],
        starter_role=0 if regular_starter else 1,
        workloads=workloads,
        usage_known=regular_starter and len(candidates) >= 3,
    )
