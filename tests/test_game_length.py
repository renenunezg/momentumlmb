"""Seven-inning games (most doubleheaders since 2023, most Tuesday and
Wednesday games in 2022) must be simulated as seven innings, and no game may
end tied. Built on neutral baserunning and league-average players so it
needs no fitted artifacts."""

import numpy as np
import pandas as pd

from backend.etl.pa_frame import OUTCOMES
from backend.sim.baserunner import AdvancementTable, OutSubtypeTable, advancement_from_frame, out_subtype_from_frame
from backend.sim.bullpen import BullpenQueue
from backend.sim.game import GameInputs, simulate_game
from backend.sim.gbq import GBQuartiles
from backend.sim.posteriors import PosteriorDraw
from backend.sim.rules import Rules
from backend.sim.tables import build_advancement, build_out_subtype
from backend.skills.common import NON_REF_IDX, REF_IDX

RATES = {"K": 0.20, "BB": 0.09, "HBP": 0.01, "1B": 0.15, "2B": 0.05, "3B": 0.005, "HR": 0.03, "OUT": 0.465}


def _draw() -> PosteriorDraw:
    logp = np.log([RATES[o] for o in OUTCOMES])
    intercept = logp[NON_REF_IDX] - logp[REF_IDX]
    zeros = np.zeros((1, len(NON_REF_IDX)))
    return PosteriorDraw(
        intercept={2026: intercept},
        intercept_mu=intercept,
        batter_offset=zeros,
        platoon_offset=zeros,
        pitcher_offset=zeros,
        park={5340: np.zeros(len(NON_REF_IDX))},
        elevation_slope=np.zeros(len(NON_REF_IDX)),
        elevation_center_km=0.0,
        batter_ids=np.array([], dtype=np.int64),
        pitcher_ids=np.array([], dtype=np.int64),
    )


def _tables() -> tuple[AdvancementTable, OutSubtypeTable]:
    empty = pd.DataFrame(
        columns=["state", "outs", "outcome_idx", "subtype_key", "new_state", "runs", "outs_added", "b_q", "p_q"]
    ).astype({c: "int64" for c in ["state", "outs", "outcome_idx", "new_state", "runs", "outs_added", "b_q", "p_q"]})
    return advancement_from_frame(build_advancement(empty)), out_subtype_from_frame(build_out_subtype(empty))


def _inputs(innings: int) -> GameInputs:
    queue = BullpenQueue(starter=1, relievers=[2, 3, 0], workloads={1: (15,), 2: (3,), 3: (3,)})
    return GameInputs(
        home_lineup=np.arange(100, 109),
        away_lineup=np.arange(200, 209),
        home_queue=queue,
        away_queue=queue,
        venue_id=5340,
        season=2026,
        home_throws={},
        away_throws={},
        rules=Rules(innings=innings, extra_runner=True),
    )


def _run(innings: int):
    adv, sub = _tables()
    return simulate_game(np.random.default_rng(7), _draw(), adv, sub, GBQuartiles({}, {}), _inputs(innings), 2000)


def test_seven_inning_game_ends_after_seven():
    seven = _run(7)
    nine = _run(9)
    decided_in_regulation = seven.innings_played <= 7
    assert decided_in_regulation.mean() > 0.8
    assert seven.innings_played[decided_in_regulation].max() == 7
    ratio = (seven.home_runs + seven.away_runs).mean() / (nine.home_runs + nine.away_runs).mean()
    assert 0.65 < ratio < 0.85


def test_extra_innings_resolve():
    result = _run(7)
    assert (result.home_runs != result.away_runs).all()
