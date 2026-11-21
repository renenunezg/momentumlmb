"""Forward grade: a projection of each player's runs above average per plate
appearance (hitters) or per batter faced (pitchers) for the season to be
played, from what was known on its opening day.

Hitters: the batting skill posterior (`skills.neutral_skills`) and a Marcel
average of past hitter runs above average per PA (batting, baserunning,
fielding and position; seasons weighted 5/4/3 and regressed toward average),
Pitchers: the pitching skill posterior plus the weighted average of past WAR
pitching runs. The blend weights are fitted on the development seasons by
least squares on the target season's actual rate. A delta-method age curve
was tried and left out: it lowered the hitters' out-of-sample rank
correlation and did not change the pitchers' (`age_curve_check`).

The grade is the projected runs above replacement over a full LMB season
(FULL_PA plate appearances, FULL_BF_SP or FULL_BF_RP batters faced) on the
20-80 scale within the position group; projected WAR uses projected playing
time instead."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backend.players.bundles import opening_bundle
from backend.players.skills import neutral_skills

MARCEL_WEIGHTS = (5.0, 4.0, 3.0)
MIN_PRIOR = 50  # prior PA (hitters) or BF (pitchers) needed for a grade
FULL_PA = 400
FULL_BF_SP = 450
FULL_BF_RP = 220
# Marcel playing time, scaled from MLB's 162 games to the LMB's ~90.
PT_WEIGHTS = (0.5, 0.1)
PT_BASE = {"hitter": 110.0, "SP": 120.0, "RP": 60.0}
HITTER_FEATURES = ("bat_post", "hit_marcel")
PITCHER_FEATURES = ("pitch_post", "pitch_marcel")


@dataclass(frozen=True)
class GradeModel:
    hitter_coef: dict[str, float]
    pitcher_coef: dict[str, float]
    shrink: dict[str, float]  # Marcel regression, PA or BF of average added
    talent_sd: dict[str, float]  # true-talent spread of the rate (year-to-year covariance)


def _marcel(history: pd.DataFrame, target: int, value: str, n: str, k: float) -> pd.DataFrame:
    """Weighted rate over the three seasons before `target`, regressed by k."""
    rows = history[(history["season"] < target) & (history["season"] >= target - 3)]
    w = rows["season"].map(
        {target - 1: MARCEL_WEIGHTS[0], target - 2: MARCEL_WEIGHTS[1], target - 3: MARCEL_WEIGHTS[2]}
    )
    agg = rows.assign(wv=w * rows[value], wn=w * rows[n]).groupby("player")[["wv", "wn"]].sum()
    return (agg["wv"] / (agg["wn"] + k)).rename(f"{value}_marcel").reset_index()


def _weighted_n(history: pd.DataFrame, target: int, n: str) -> pd.Series:
    """The Marcel weighted sample in units of the latest season (weights 5/4/3 over 5)."""
    rows = history[(history["season"] < target) & (history["season"] >= target - 3)]
    w = rows["season"].map(dict(zip((target - 1, target - 2, target - 3), MARCEL_WEIGHTS)))
    return (w * rows[n]).groupby(rows["player"]).sum() / MARCEL_WEIGHTS[0]


def talent_sd(history: pd.DataFrame, value: str, n: str, min_n: float = 200) -> float:
    """Spread of true talent in the rate: the covariance of a player's rates in
    consecutive seasons (noise does not carry over, talent does)."""
    a = history[history[n] >= min_n].assign(rate=lambda d: d[value] / d[n])[["season", "player", "rate"]]
    m = a.merge(a.assign(season=a["season"] - 1), on=["season", "player"], suffixes=("_0", "_1"))
    return float(np.sqrt(max(np.cov(m["rate_0"], m["rate_1"])[0, 1], 0.0)))


def _talent_band(sd: float, k: float, weighted_n: pd.Series) -> np.ndarray:
    """Standard deviation of true talent left after a Marcel average on
    `weighted_n` with regression k: sd * sqrt(k / (n + k))."""
    n = weighted_n.fillna(0.0).to_numpy(dtype=float)
    return sd * np.sqrt(k / (n + k))


def fit_shrink(history: pd.DataFrame, value: str, n: str, targets: list[int], min_n: float) -> float:
    """The Marcel regression k that best predicts the target seasons' rates
    (weighted least squares over a log grid refined around its minimum)."""

    def loss(k: float) -> float:
        err, wsum = 0.0, 0.0
        for t in targets:
            pred = _marcel(history, t, value, n, k)
            actual = history[(history["season"] == t) & (history[n] >= min_n)]
            m = actual.merge(pred, on="player")
            r = m[value] / m[n] - m[f"{value}_marcel"]
            err += float((m[n] * r**2).sum())
            wsum += float(m[n].sum())
        return err / wsum

    from scipy.optimize import minimize_scalar

    res = minimize_scalar(lambda lk: loss(np.exp(lk)), bounds=(np.log(10), np.log(20000)), method="bounded")
    return float(np.exp(res.x))


def fit_age_curve(
    history: pd.DataFrame, value: str, n: str, ages: pd.DataFrame, seasons: list[int]
) -> tuple[float, float, float]:
    """Delta method: change in rate between consecutive seasons by age,
    weighted by the harmonic mean of the two samples, fitted as a quadratic in
    the first season's age."""
    h = history[history["season"].isin(seasons)].merge(ages, on=["season", "player"])
    a = h.assign(rate=h[value] / h[n])
    b = a.assign(season=a["season"] - 1)
    m = a.merge(b, on=["season", "player"], suffixes=("_0", "_1"))
    m = m[(m[f"{n}_0"] >= 50) & (m[f"{n}_1"] >= 50)]
    w = 2.0 / (1.0 / m[f"{n}_0"] + 1.0 / m[f"{n}_1"])
    age = m["age_0"].astype(float)
    x = np.column_stack([np.ones(len(m)), age, age**2])
    sw = np.sqrt(w.to_numpy())
    coef, *_ = np.linalg.lstsq(x * sw[:, None], (m["rate_1"] - m["rate_0"]).to_numpy() * sw, rcond=None)
    return tuple(float(c) for c in coef)


def _age_delta(coef: tuple[float, float, float], age: pd.Series) -> np.ndarray:
    a = age.astype(float).fillna(30.0).to_numpy()
    return coef[0] + coef[1] * a + coef[2] * a**2


def features(
    target: int, history: pd.DataFrame, ages: pd.DataFrame, shrink: dict[str, float], lw: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hitter and pitcher feature rows for `target`, from opening day's
    information: skills from the opening bundle, rates from earlier seasons."""
    bat_post, pitch_post = neutral_skills(opening_bundle(target), lw)
    prior = history[(history["season"] < target) & (history["season"] >= target - 3)]
    hitters = (
        prior.groupby("player")["pa"]
        .sum()
        .rename("prior_pa")
        .reset_index()
        .query("prior_pa > 0")
        .merge(bat_post, on="player", how="left")
        .merge(
            _marcel(history, target, "hit_raa", "pa", shrink["hit"]).rename(columns={"hit_raa_marcel": "hit_marcel"}),
            on="player",
            how="left",
        )
    )
    latest = prior[prior["pa"] > 0].sort_values("season").drop_duplicates("player", keep="last")
    hitters = hitters.merge(
        latest[["player", "pos", "pos_runs", "pa"]].rename(columns={"pa": "last_pa", "pos_runs": "last_pos_runs"}),
        on="player",
        how="left",
    )
    pitchers = (
        prior.groupby("player")[["bf", "sp_bf", "rp_bf"]]
        .sum()
        .rename(columns={"bf": "prior_bf"})
        .reset_index()
        .query("prior_bf > 0")
        .merge(pitch_post, on="player", how="left")
        .merge(
            _marcel(history, target, "pitch_raa", "bf", shrink["pitch"]).rename(
                columns={"pitch_raa_marcel": "pitch_marcel"}
            ),
            on="player",
            how="left",
        )
    )
    hitters = hitters.merge(
        _weighted_n(history, target, "pa").rename("hit_wn"), left_on="player", right_index=True, how="left"
    )
    pitchers = pitchers.merge(
        _weighted_n(history, target, "bf").rename("pitch_wn"), left_on="player", right_index=True, how="left"
    )
    pitchers["role"] = np.where(pitchers["sp_bf"] >= pitchers["rp_bf"], "SP", "RP")
    age_t = ages[ages["season"] == target - 1][["player", "age"]].assign(age=lambda d: d["age"].astype("Float64"))
    hitters = hitters.merge(age_t, on="player", how="left")
    pitchers = pitchers.merge(age_t, on="player", how="left")
    for frame, cols in ((hitters, HITTER_FEATURES), (pitchers, PITCHER_FEATURES)):
        for c in cols:
            frame[c] = frame[c].fillna(0.0)
    hitters["season"] = target
    pitchers["season"] = target
    return hitters, pitchers


def project(hitters: pd.DataFrame, pitchers: pd.DataFrame, model: GradeModel) -> tuple[pd.DataFrame, pd.DataFrame]:
    h, p = hitters.copy(), pitchers.copy()
    h["proj_rate"] = sum(model.hitter_coef[c] * h[c] for c in HITTER_FEATURES)
    p["proj_rate"] = sum(model.pitcher_coef[c] * p[c] for c in PITCHER_FEATURES)
    return h, p


def fit_blend(rows: pd.DataFrame, target_col: str, n_col: str, cols: tuple[str, ...]) -> dict[str, float]:
    """Weighted least squares of the target rate on the features, no intercept."""
    x = rows[list(cols)].to_numpy(dtype=float)
    y = rows[target_col].to_numpy(dtype=float)
    w = np.sqrt(rows[n_col].to_numpy(dtype=float))
    coef, *_ = np.linalg.lstsq(x * w[:, None], y * w, rcond=None)
    return dict(zip(cols, coef.round(4)))


def hitter_target(p: pd.DataFrame) -> pd.Series:
    return p["hit_raa"] / p["pa"]


def pitcher_target(p: pd.DataFrame) -> pd.Series:
    return p["pitch_raa"] / p["bf"]


MIN_TARGET = 150  # target-season PA or BF for the forward evaluation


def target_rows(
    t: int,
    history: pd.DataFrame,
    ages: pd.DataFrame,
    shrink: dict[str, float],
    lw: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Features, baselines and the actual target-season rate for players with
    enough prior and target-season playing time."""
    h, p = features(t, history, ages, shrink, lw)
    actual = history[history["season"] == t]
    prev = history[history["season"] == t - 1].set_index("player")
    h = h.merge(actual[["player", "pa", "hit_raa"]].rename(columns={"pa": "target_pa"}), on="player")
    h = h[(h["target_pa"] >= MIN_TARGET) & (h["prior_pa"] >= MIN_PRIOR)]
    h["target"] = h["hit_raa"] / h["target_pa"]
    h["blend_target"] = h["target"]
    h["marcel_rate"] = h["hit_marcel"]
    h = h.merge(
        (prev["hit_raa"] / prev["pa"].where(prev["pa"] > 0)).rename("prior_rate"),
        left_on="player",
        right_index=True,
        how="left",
    )
    p = p.merge(actual[["player", "bf", "pitch_raa"]].rename(columns={"bf": "target_bf"}), on="player")
    p = p[(p["target_bf"] >= MIN_TARGET) & (p["prior_bf"] >= MIN_PRIOR)]
    p["target"] = p["pitch_raa"] / p["target_bf"]
    p["blend_target"] = p["target"]
    p["marcel_rate"] = p["pitch_marcel"]
    p = p.merge(
        (prev["pitch_raa"] / prev["bf"].where(prev["bf"] > 0)).rename("prior_rate"),
        left_on="player",
        right_index=True,
        how="left",
    )
    return h, p


def fit_model(
    history: pd.DataFrame, ages: pd.DataFrame, lw_by_season: pd.Series, targets: list[int]
) -> tuple[GradeModel, pd.DataFrame, pd.DataFrame]:
    """Fit shrinkage, age curves and blend weights on `targets` (development
    seasons). Returns the model and the per-target evaluation rows, whose
    `proj_rate` is predicted with blend weights fitted on the other targets."""
    shrink = {
        "hit": fit_shrink(history, "hit_raa", "pa", targets, MIN_TARGET),
        "pitch": fit_shrink(history, "pitch_raa", "bf", targets, MIN_TARGET),
    }
    hit_rows, pit_rows = [], []
    for t in targets:
        h, p = target_rows(t, history, ages, shrink, lw_by_season.loc[t - 1])
        hit_rows.append(h)
        pit_rows.append(p)
    hits, pits = pd.concat(hit_rows), pd.concat(pit_rows)

    # Leave-one-season-out blend predictions for the evaluation.
    for frame, cols, n_col in ((hits, HITTER_FEATURES, "target_pa"), (pits, PITCHER_FEATURES, "target_bf")):
        frame["proj_rate"] = np.nan
        for t in targets:
            coef = fit_blend(frame[frame["season"] != t], "blend_target", n_col, cols)
            m = frame["season"] == t
            frame.loc[m, "proj_rate"] = sum(coef[c] * frame.loc[m, c] for c in cols)

    model = GradeModel(
        hitter_coef=fit_blend(hits, "blend_target", "target_pa", HITTER_FEATURES),
        pitcher_coef=fit_blend(pits, "blend_target", "target_bf", PITCHER_FEATURES),
        shrink=shrink,
        talent_sd={"hit": talent_sd(history, "hit_raa", "pa"), "pitch": talent_sd(history, "pitch_raa", "bf")},
    )
    return model, hits, pits


GRADE_REFERENCE_MIN = 200  # prior PA or BF of the players the 20-80 scale is set on
BAND_Z = 1.2816  # 80% band


def _projected_time(history: pd.DataFrame, target: int, col: str, base: float) -> pd.Series:
    last = history[history["season"] == target - 1].set_index("player")[col]
    before = history[history["season"] == target - 2].set_index("player")[col]
    both = pd.concat([last.rename("a"), before.rename("b")], axis=1).fillna(0.0)
    return (PT_WEIGHTS[0] * both["a"] + PT_WEIGHTS[1] * both["b"] + base).rename(f"proj_{col}")


def score_grades(
    target: int,
    history: pd.DataFrame,
    ages: pd.DataFrame,
    model: GradeModel,
    lw: pd.Series,
    repl: pd.Series,
    runs_per_win: float,
) -> pd.DataFrame:
    """Grade, 80% band and projected WAR for `target`, one row per player
    with enough prior playing time. `repl` holds the latest season's
    replacement rates (hitter, starter, reliever)."""
    h, p = features(target, history, ages, model.shrink, lw)
    h, p = project(h, p, model)
    h = h[h["prior_pa"] >= MIN_PRIOR].copy()
    p = p[p["prior_bf"] >= MIN_PRIOR].copy()

    h["full"] = (h["proj_rate"] - repl["hitter"]) * FULL_PA
    h["full_sd"] = _talent_band(model.talent_sd["hit"], model.shrink["hit"], h["hit_wn"]) * FULL_PA
    h["group"] = h["pos"].fillna("DH")
    h = h.merge(
        _projected_time(history, target, "pa", PT_BASE["hitter"]), left_on="player", right_index=True, how="left"
    )
    h["proj_war"] = (h["proj_rate"] - repl["hitter"]) * h["proj_pa"] / runs_per_win
    h["reference"] = h["prior_pa"] >= GRADE_REFERENCE_MIN

    full_bf = np.where(p["role"] == "SP", FULL_BF_SP, FULL_BF_RP)
    repl_rate = np.where(p["role"] == "SP", repl["starter"], repl["reliever"])
    p["full"] = (p["proj_rate"] - repl_rate) * full_bf
    p["full_sd"] = _talent_band(model.talent_sd["pitch"], model.shrink["pitch"], p["pitch_wn"]) * full_bf
    p["group"] = p["role"]
    pt = pd.concat(
        [
            _projected_time(history, target, "sp_bf", 0.0),
            _projected_time(history, target, "rp_bf", 0.0),
        ],
        axis=1,
    )
    p = p.merge(pt, left_on="player", right_index=True, how="left")
    base = np.where(p["role"] == "SP", PT_BASE["SP"], PT_BASE["RP"])
    proj_bf = p["proj_sp_bf"].fillna(0.0) + p["proj_rp_bf"].fillna(0.0) + base
    p["proj_war"] = (p["proj_rate"] - repl_rate) * proj_bf / runs_per_win
    p["reference"] = p["prior_bf"] >= GRADE_REFERENCE_MIN

    rows = pd.concat(
        [
            h[["player", "group", "full", "full_sd", "proj_war", "proj_rate", "reference"]],
            p[["player", "group", "full", "full_sd", "proj_war", "proj_rate", "reference"]],
        ]
    )
    # A two-way player keeps his larger projection.
    rows = rows.sort_values("full", ascending=False).drop_duplicates("player")
    ref = rows[rows["reference"]].groupby("group")["full"].agg(["mean", "std"])
    stats = ref.reindex(rows["group"]).to_numpy()
    z = (rows["full"].to_numpy() - stats[:, 0]) / stats[:, 1]
    zs = rows["full_sd"].to_numpy() / stats[:, 1]
    rows["grade"] = np.clip(50 + 10 * z, 20, 80)
    rows["grade_lo"] = np.clip(50 + 10 * (z - BAND_Z * zs), 20, 80)
    rows["grade_hi"] = np.clip(50 + 10 * (z + BAND_Z * zs), 20, 80)
    rows["grade_season"] = target
    return rows.reset_index(drop=True)


def age_curve_check(history: pd.DataFrame, ages: pd.DataFrame, hits: pd.DataFrame, pits: pd.DataFrame) -> dict:
    """Pooled out-of-sample rank correlation of the projection with and
    without a delta-method age curve (fitted on the same seasons)."""
    from scipy.stats import spearmanr

    seasons = sorted(history["season"].unique())
    out = {}
    for name, frame, value, n in (("hitters", hits, "hit_raa", "pa"), ("pitchers", pits, "pitch_raa", "bf")):
        coef = fit_age_curve(history, value, n, ages, seasons)
        f = frame.dropna(subset=["proj_rate"]).copy()
        f["with_age"] = f["proj_rate"] + _age_delta(coef, f["age"])
        score = {
            col: round(float(np.mean([spearmanr(g[col], g["target"]).statistic for _, g in f.groupby("season")])), 3)
            for col in ("proj_rate", "with_age")
        }
        out[name] = {"without_age": score["proj_rate"], "with_age": score["with_age"], "curve": coef}
    return out
