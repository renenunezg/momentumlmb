"""Positional adjustment and each player's position.

The scale is fitted on LMB players who played more than one fielding
position in a season (the defensive-spectrum method: a player's zone fielding
rate at each position, with a player effect, gives how much harder one
position is than another). Catcher and designated hitter cannot be fitted
that way (catchers rarely play elsewhere, a DH does not field), so they keep
FanGraphs' distance from first base. The FanGraphs scale is the fallback for
any season the fit fails its stability rule."""

import numpy as np
import pandas as pd

from backend.players.events import FIELD_POSITIONS

# FanGraphs positional adjustment, runs per 162 games (borrowed constants).
FANGRAPHS_RUNS_PER_162 = {
    "C": 12.5,
    "SS": 7.5,
    "2B": 2.5,
    "3B": 2.5,
    "CF": 2.5,
    "LF": -7.5,
    "RF": -7.5,
    "1B": -12.5,
    "DH": -17.5,
}
FITTED_POSITIONS = ("1B", "2B", "3B", "SS", "LF", "CF", "RF")
MIN_DEF_PA_EACH = 150
# The fit is used only if every fitted position's 90% bootstrap interval is
# narrower than this many runs per 162 games (fixed before the fit was read).
MAX_INTERVAL_RUNS = 10.0


def fit_spectrum(zone: pd.DataFrame, seed: int = 0, n_boot: int = 500) -> dict:
    """Relative difficulty of the seven non-catcher fielding positions in runs
    per 162 games (first base = 0), with bootstrap intervals over players."""
    z = zone[zone["pos"].isin(FITTED_POSITIONS) & (zone["def_pa"] >= MIN_DEF_PA_EACH)]
    multi = z.groupby(["season", "player"])["pos"].transform("nunique") >= 2
    z = z[multi].assign(unit=lambda d: d["season"].astype(str) + "-" + d["player"].astype(str))
    z = z.assign(rate=z["fielding_runs"] / z["def_pa"])
    pa_per_162 = 162 * 38.0  # batters faced per 162 games, a scale constant for the report

    def solve(frame: pd.DataFrame) -> np.ndarray:
        units = frame["unit"].astype("category")
        pos = pd.Categorical(frame["pos"], categories=FITTED_POSITIONS)
        n_u = len(units.cat.categories)
        x = np.zeros((len(frame), n_u + len(FITTED_POSITIONS) - 1))
        x[np.arange(len(frame)), units.cat.codes] = 1.0
        for j in range(1, len(FITTED_POSITIONS)):
            x[:, n_u + j - 1] = -(pos.codes == j).astype(float)
        w = np.sqrt(frame["def_pa"].to_numpy())
        beta, *_ = np.linalg.lstsq(x * w[:, None], frame["rate"].to_numpy() * w, rcond=None)
        return np.concatenate([[0.0], beta[n_u:]]) * pa_per_162

    point = solve(z)
    rng = np.random.default_rng(seed)
    units = z["unit"].unique()
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(units, size=len(units), replace=True)
        frame = pd.concat([z[z["unit"] == u].assign(unit=f"{u}#{i}") for i, u in enumerate(pick)])
        if frame["pos"].nunique() == len(FITTED_POSITIONS):
            boots.append(solve(frame))
    lo, hi = np.percentile(np.array(boots), [5, 95], axis=0)
    width = hi - lo
    return {
        "difficulty": dict(zip(FITTED_POSITIONS, point.round(2))),
        "interval": {p: [round(a, 2), round(b, 2)] for p, a, b in zip(FITTED_POSITIONS, lo, hi)},
        "n_player_seasons": int(z["unit"].nunique()),
        "stable": bool((width[1:] < MAX_INTERVAL_RUNS).all()),
    }


def runs_per_162(spectrum: dict | None) -> dict[str, float]:
    """Positional adjustment per 162 games. With a stable fit, the fitted
    positions take its relative difficulty, re-levelled so their mean matches
    FanGraphs' mean over the same positions; C and DH keep FanGraphs' gap to 1B."""
    if spectrum is None or not spectrum["stable"]:
        return dict(FANGRAPHS_RUNS_PER_162)
    d = spectrum["difficulty"]
    fg_mean = np.mean([FANGRAPHS_RUNS_PER_162[p] for p in FITTED_POSITIONS])
    shift = fg_mean - np.mean([d[p] for p in FITTED_POSITIONS])
    out = {p: float(d[p] + shift) for p in FITTED_POSITIONS}
    for p in ("C", "DH"):
        out[p] = out["1B"] + FANGRAPHS_RUNS_PER_162[p] - FANGRAPHS_RUNS_PER_162["1B"]
    return out


def positional_runs(
    def_pa: pd.DataFrame, dh_pa: pd.DataFrame, scale: dict[str, float], pa_per_team_game: dict
) -> pd.DataFrame:
    """Positional runs by season and player: fielders per batter faced at the
    position, designated hitters per plate appearance as DH (a lineup slot
    bats a ninth of the team's plate appearances)."""
    f = def_pa[def_pa["pos"].isin(FIELD_POSITIONS)].copy()
    per_game = f["season"].map(pa_per_team_game)
    f["pos_runs"] = f["pos"].map(scale) / 162.0 * f["def_pa"] / per_game
    d = dh_pa.copy()
    d["pos_runs"] = scale["DH"] / 162.0 * d["dh_pa"] / (d["season"].map(pa_per_team_game) / 9.0)
    both = pd.concat([f[["season", "player", "pos_runs"]], d[["season", "player", "pos_runs"]]])
    return both.groupby(["season", "player"])["pos_runs"].sum().reset_index()


def primary_position(def_pa: pd.DataFrame, dh_pa: pd.DataFrame) -> pd.DataFrame:
    """Each player's position for the season: the fielding position (or DH)
    where he spent the most plate appearances; pitchers are labelled later."""
    f = def_pa[def_pa["pos"].isin(FIELD_POSITIONS)][["season", "player", "pos", "def_pa"]].rename(
        columns={"def_pa": "n"}
    )
    d = dh_pa.rename(columns={"dh_pa": "n"}).assign(pos="DH")
    both = pd.concat([f, d[["season", "player", "pos", "n"]]])
    top = both.sort_values("n", ascending=False).drop_duplicates(["season", "player"])
    return top[["season", "player", "pos"]]
