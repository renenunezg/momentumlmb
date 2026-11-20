"""Game Monte Carlo, vectorized across sims of one game (ported from mlbmodel
v2/simulator/game_sim.py). Changes for the LMB: the regulation length comes
from the game's scheduled innings (seven-inning doubleheaders), the
extra-innings runner follows the measured league rule, parks key on
venue_id, and weather enters as a single home-run logit shift.

Known approximations, as in MLB: mid-PA steals and wild pitches are not
modeled; pitcher workloads are sampled from pregame role evidence."""

from dataclasses import dataclass

import numpy as np

from backend.sim import bullpen as bp
from backend.sim.baserunner import AdvancementTable, OutSubtypeTable, sample_subtypes_for_outs
from backend.sim.bullpen import BullpenQueue
from backend.sim.gbq import GBQuartiles
from backend.sim.pa import HR_IDX, pa_logits_batch, sample_categorical, softmax
from backend.sim.posteriors import PosteriorDraw
from backend.sim.rules import Rules

GHOST_RUNNER_STATE = 2  # runner on second only
# Per-game form noise on logits, zero-sum across outcomes (mlbmodel's calibrated value).
FORM_SIGMA = 0.18
MAX_PAS = 1000


@dataclass
class GameInputs:
    home_lineup: np.ndarray  # (9,) batter ids in batting order
    away_lineup: np.ndarray
    home_queue: BullpenQueue  # pitches the top halves
    away_queue: BullpenQueue  # pitches the bottom halves
    venue_id: int
    season: int
    home_throws: dict[int, str]  # pitcher id -> "L" or "R"
    away_throws: dict[int, str]
    rules: Rules
    hr_logit_shift: float = 0.0


@dataclass
class SimResult:
    home_runs: np.ndarray
    away_runs: np.ndarray
    innings_played: np.ndarray


def _queue_arrays(q: BullpenQueue, throws: dict[int, str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = [q.starter] + q.relievers or [0]
    arr_ids = np.array(ids, dtype=np.int64)
    lhp = np.array([1 if throws.get(int(p), "R") == "L" else 0 for p in ids], dtype=np.int64)
    roles = np.array([q.starter_role if i == 0 else 1 for i in range(len(ids))], dtype=np.int64)
    return arr_ids, lhp, roles


def simulate_game(
    rng: np.random.Generator,
    draw: PosteriorDraw,
    adv: AdvancementTable,
    sub_table: OutSubtypeTable,
    gbq: GBQuartiles,
    inputs: GameInputs,
    n_sims: int,
    form_sigma: float = FORM_SIGMA,
) -> SimResult:
    regulation = inputs.rules.innings
    home_pids, home_lhp, home_role = _queue_arrays(inputs.home_queue, inputs.home_throws)
    away_pids, away_lhp, away_role = _queue_arrays(inputs.away_queue, inputs.away_throws)
    home_max_idx, away_max_idx = len(home_pids) - 1, len(away_pids) - 1
    home_limits = np.stack([rng.choice(inputs.home_queue.outs_samples(int(p)), n_sims) for p in home_pids], axis=1)
    away_limits = np.stack([rng.choice(inputs.away_queue.outs_samples(int(p)), n_sims) for p in away_pids], axis=1)
    sims = np.arange(n_sims)

    home_arr = np.asarray(inputs.home_lineup, dtype=np.int64)
    away_arr = np.asarray(inputs.away_lineup, dtype=np.int64)
    home_bq, away_bq = gbq.batter_q(home_arr), gbq.batter_q(away_arr)
    home_pq, away_pq = gbq.pitcher_q(home_pids), gbq.pitcher_q(away_pids)

    n_out = len(draw.intercept_mu) + 1
    if form_sigma > 0:
        away_form = rng.normal(0.0, form_sigma, size=(n_sims, n_out))
        away_form -= away_form.mean(axis=1, keepdims=True)
        home_form = rng.normal(0.0, form_sigma, size=(n_sims, n_out))
        home_form -= home_form.mean(axis=1, keepdims=True)
    else:
        away_form = home_form = np.zeros((n_sims, n_out))
    weather = np.zeros(n_out)
    weather[HR_IDX] = inputs.hr_logit_shift

    state = np.zeros(n_sims, dtype=np.int64)
    outs = np.zeros(n_sims, dtype=np.int64)
    inning = np.ones(n_sims, dtype=np.int64)
    is_top = np.ones(n_sims, dtype=bool)
    home_runs = np.zeros(n_sims, dtype=np.int64)
    away_runs = np.zeros(n_sims, dtype=np.int64)
    li_home = np.zeros(n_sims, dtype=np.int64)
    li_away = np.zeros(n_sims, dtype=np.int64)
    p_idx_home = np.zeros(n_sims, dtype=np.int64)
    p_idx_away = np.zeros(n_sims, dtype=np.int64)
    p_outs_h = np.zeros(n_sims, dtype=np.int64)
    p_outs_a = np.zeros(n_sims, dtype=np.int64)
    p_runs_h = np.zeros(n_sims, dtype=np.int64)
    p_runs_a = np.zeros(n_sims, dtype=np.int64)
    p_pa_h = np.zeros(n_sims, dtype=np.int64)
    p_pa_a = np.zeros(n_sims, dtype=np.int64)
    done = np.zeros(n_sims, dtype=bool)
    final_inning = np.zeros(n_sims, dtype=np.int64)

    def finish(mask: np.ndarray, at_inning: np.ndarray) -> None:
        nonlocal done, final_inning
        newly = mask & ~done
        final_inning = np.where(newly, at_inning, final_inning)
        done = done | newly

    def pull_mask(p_idx, p_outs, p_runs, p_pa, limits, roles):
        starter_in = roles[p_idx] == 0
        starter_pull = starter_in & (
            (p_outs >= limits[sims, p_idx])
            | (p_runs >= bp.PULL_RUNS_HARD)
            | ((p_outs >= bp.PULL_OUTS_RUNS) & (p_runs >= bp.PULL_RUNS_AT_OUTS))
            | (p_pa >= bp.PULL_PA_PROXY)
        )
        reliever_pull = ~starter_in & ((p_outs >= limits[sims, p_idx]) | (p_runs >= bp.RELIEVER_PULL_RUNS))
        return starter_pull | reliever_pull

    for _ in range(MAX_PAS):
        if done.all():
            break
        active = ~done
        batter_ids = np.where(is_top, away_arr[li_away % 9], home_arr[li_home % 9])
        pitcher_ids = np.where(is_top, home_pids[p_idx_home], away_pids[p_idx_away])
        vs_lhp = np.where(is_top, home_lhp[p_idx_home], away_lhp[p_idx_away]).astype(bool)

        logits = pa_logits_batch(draw, batter_ids, pitcher_ids, vs_lhp, inputs.season, inputs.venue_id)
        logits = logits + weather + np.where(is_top[:, None], away_form, home_form)
        outcomes = sample_categorical(rng, softmax(logits))
        batter_bq = np.where(is_top, away_bq[li_away % 9], home_bq[li_home % 9])
        pitcher_pq = np.where(is_top, home_pq[p_idx_home], away_pq[p_idx_away])
        subtypes = sample_subtypes_for_outs(rng, outcomes, state, outs, batter_bq, pitcher_pq, sub_table)
        new_state, runs_scored, outs_added = adv.sample(rng, state, outs, outcomes, subtypes)

        top, bot = active & is_top, active & ~is_top
        away_runs = np.where(top, away_runs + runs_scored, away_runs)
        home_runs = np.where(bot, home_runs + runs_scored, home_runs)
        p_outs_h = np.where(top, p_outs_h + outs_added, p_outs_h)
        p_outs_a = np.where(bot, p_outs_a + outs_added, p_outs_a)
        p_runs_h = np.where(top, p_runs_h + runs_scored, p_runs_h)
        p_runs_a = np.where(bot, p_runs_a + runs_scored, p_runs_a)
        p_pa_h = np.where(top, p_pa_h + 1, p_pa_h)
        p_pa_a = np.where(bot, p_pa_a + 1, p_pa_a)
        outs = np.where(active, outs + outs_added, outs)
        state = np.where(active, new_state, state)
        li_away = np.where(top, li_away + 1, li_away)
        li_home = np.where(bot, li_home + 1, li_home)

        finish(bot & (inning >= regulation) & (home_runs > away_runs), inning)  # walk-off

        inning_end = active & ~done & (outs >= 3)
        if inning_end.any():
            to_bottom = inning_end & is_top
            to_top = inning_end & ~is_top
            is_top = np.where(to_bottom, False, np.where(to_top, True, is_top))
            inning = np.where(to_top, inning + 1, inning)
            outs = np.where(inning_end, 0, outs)
            extras = inning_end & (inning > regulation) & inputs.rules.extra_runner
            state = np.where(extras, GHOST_RUNNER_STATE, np.where(inning_end, 0, state))
            finish(to_bottom & (inning >= regulation) & (home_runs > away_runs), inning)
            finish(to_top & (inning - 1 >= regulation) & (home_runs != away_runs), inning - 1)

        swap_h = (
            ~done
            & pull_mask(p_idx_home, p_outs_h, p_runs_h, p_pa_h, home_limits, home_role)
            & (p_idx_home < home_max_idx)
        )
        swap_a = (
            ~done
            & pull_mask(p_idx_away, p_outs_a, p_runs_a, p_pa_a, away_limits, away_role)
            & (p_idx_away < away_max_idx)
        )
        if swap_h.any():
            p_idx_home = np.where(swap_h, p_idx_home + 1, p_idx_home)
            p_outs_h, p_runs_h, p_pa_h = (np.where(swap_h, 0, x) for x in (p_outs_h, p_runs_h, p_pa_h))
        if swap_a.any():
            p_idx_away = np.where(swap_a, p_idx_away + 1, p_idx_away)
            p_outs_a, p_runs_a, p_pa_a = (np.where(swap_a, 0, x) for x in (p_outs_a, p_runs_a, p_pa_a))

    if not done.all():
        raise RuntimeError(f"{int((~done).sum())} simulations unfinished after {MAX_PAS} plate appearances")
    return SimResult(home_runs=home_runs, away_runs=away_runs, innings_played=final_inning)


def simulate_draws(
    rng: np.random.Generator,
    draws: list[PosteriorDraw],
    adv: AdvancementTable,
    sub_table: OutSubtypeTable,
    gbq: GBQuartiles,
    inputs: GameInputs,
    n_per_draw: int,
) -> SimResult:
    """Simulate n_per_draw games under each posterior draw and pool them."""
    parts = [simulate_game(rng, d, adv, sub_table, gbq, inputs, n_per_draw) for d in draws]
    return SimResult(
        home_runs=np.concatenate([p.home_runs for p in parts]),
        away_runs=np.concatenate([p.away_runs for p in parts]),
        innings_played=np.concatenate([p.innings_played for p in parts]),
    )
