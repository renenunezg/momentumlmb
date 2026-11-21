"""Opening-day skill fits the grade is scored from. Each is the game model's
own fit (three seasons back, walk-forward draw settings) on everything before
the season's first regular-season game, so a grade never sees its target
season. The 2024-2026 openings reuse the walk-forward bundles."""

from datetime import date
from pathlib import Path

from backend.config import ARTIFACTS_DIR
from backend.etl.games import load_games
from backend.etl.pa_frame import load_pa
from backend.skills.fit import fit_all

WALKFORWARD_SB3 = ARTIFACTS_DIR / "walkforward" / "sb3"
GRADE_BUNDLES = ARTIFACTS_DIR / "players" / "bundles"
SEASONS_BACK = 3
DRAWS = 500
TUNE = 500


def opening_day(season: int) -> date:
    games = load_games([season])
    return min(games[games.game_type == "R"].official_date)


def opening_bundle(season: int) -> Path:
    """The fit the grade for `season` is scored from. After the last cached
    season it is a fit through the end of the previous one."""
    walkforward = WALKFORWARD_SB3 / opening_day(season).isoformat() if season <= 2026 else None
    if walkforward is not None and walkforward.exists():
        return walkforward
    return GRADE_BUNDLES / str(season)


def fit_opening_bundle(season: int) -> Path:
    through = opening_day(season) if season <= 2026 else date(season, 1, 1)
    pa = load_pa(list(range(max(2021, season - SEASONS_BACK), season + 1)), missing_ok=True)
    return fit_all(pa, through, GRADE_BUNDLES / str(season), SEASONS_BACK, draws=DRAWS, tune=TUNE, enforce_gate=False)
