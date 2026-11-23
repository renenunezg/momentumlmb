"""Development report and the one-time holdout read for the prospect model.

Splits, fixed before any fit (snapshot seasons; outcomes run four LMB seasons
past each):
- development 2006-2016: fits and the cross-validated penalty;
- validation 2017-2019 (outcomes 2018-2024): compares, per role, the baseline
  (age, level and playing time) against the compact skill model (plus
  level-relative age, level-relative K-BB%, and level-relative batting runs for
  hitters);
- holdout 2021-2022 (outcomes 2022-2026): read once with `read_holdout`,
  after the choice is committed in CHOICE.
Rows of a player who appears in a later split are left out of the earlier
split's fit, so no label shares outcome seasons with a scored player's label.
Intervals resample players, not snapshots.

Holdout rule, fixed before the read: a role is published if the 90% player
bootstrap interval of log loss (chosen model minus the development base rate)
lies wholly below zero."""

import json
from datetime import date

import numpy as np
import pandas as pd

from backend.evaluation.metrics import log_loss
from backend.prospects import lines, model, snapshots

DEVELOPMENT = tuple(range(2006, 2017))
VALIDATION = (2017, 2018, 2019)
HOLDOUT = (2021, 2022)
ROLES = ("H", "P")
# Validation results for lmb-prospects-v2 (2026-11-23), committed before reading holdout_v2.json:
# hitters full .3862 vs baseline .4148 (CI90 -.0507 to -.0072), AUC .810 vs .774, base rate .4973;
# pitchers full .3408 vs baseline .3536 (CI90 -.0280 to +.0020), AUC .803 vs .775, base rate .4009.
# Both roles use the compact v2 skill spec (age_rel + kbb_rel, plus woba_rel for hitters); pitchers
# clipped +.0020 on the 90% LL interval under a falling validation base rate (.180 -> .134) while
# improving validation AUC by +.028 and dev CV LL by -.017.
CHOICE: dict[str, str | None] = {"H": "full", "P": "full"}
HOLDOUT_REPORT = lines.PROSPECTS_DIR / "holdout_v2.json"


def features_for(role: str, spec: str) -> tuple[str, ...]:
    return snapshots.BASELINE_FEATURES if spec == "baseline" else snapshots.FULL_FEATURES[role]


def matured(today: date, seasons) -> pd.DataFrame:
    frame_lines, people, team_games, played = lines.load_inputs(today)
    snaps = snapshots.build(frame_lines, people, team_games, list(seasons), played)
    return snaps[snaps.success.notna()]


def auc(p: np.ndarray, y: np.ndarray) -> float:
    order = pd.Series(p).rank().to_numpy()
    pos = y == 1
    return float((order[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))


def player_bootstrap_ci(diff: np.ndarray, players: np.ndarray, level: float = 0.90, n: int = 10_000, seed: int = 0):
    """Interval for the mean per-snapshot difference, resampling players."""
    frame = pd.DataFrame({"d": diff, "p": players}).groupby("p").d.agg(["sum", "count"])
    sums, counts = frame["sum"].to_numpy(), frame["count"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), size=(n, len(sums)))
    means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    tail = (1 - level) / 2
    return float(np.quantile(means, tail)), float(np.quantile(means, 1 - tail))


def _score(scored: pd.DataFrame, preds: dict[str, np.ndarray], base_rate: float) -> dict:
    y = scored.success.to_numpy()
    base = log_loss(np.full(len(y), base_rate), y)
    out = {
        "snapshots": int(len(y)),
        "players": int(scored.player.nunique()),
        "successes": int(y.sum()),
        "base_rate_log_loss": float(base.mean()),
    }
    for name, p in preds.items():
        loss = log_loss(p, y)
        out[name] = {
            "log_loss": float(loss.mean()),
            "auc": auc(p, y),
            "mean_p": float(p.mean()),
            "observed": float(y.mean()),
            "vs_base_rate_ci90": player_bootstrap_ci(loss - base, scored.player.to_numpy()),
        }
    if {"baseline", "full"} <= preds.keys():
        diff = log_loss(preds["full"], y) - log_loss(preds["baseline"], y)
        out["full_minus_baseline_ci90"] = player_bootstrap_ci(diff, scored.player.to_numpy())
    return out


def development_report(today: date) -> dict:
    snaps = matured(today, DEVELOPMENT + VALIDATION)
    report = {}
    for role in ROLES:
        part = snaps[snaps.role == role]
        val = part[part.season.isin(VALIDATION)]
        dev = part[part.season.isin(DEVELOPMENT) & ~part.player.isin(val.player)]
        fits = {spec: model.fit(dev, features_for(role, spec)) for spec in ("baseline", "full")}
        report[role] = {
            "development": {
                "snapshots": int(len(dev)),
                "players": int(dev.player.nunique()),
                "successes": int(dev.success.sum()),
            },
            "penalties": {spec: m.penalty for spec, m in fits.items()},
            "validation": _score(val, {spec: m.predict(val) for spec, m in fits.items()}, float(dev.success.mean())),
            "full_weights": dict(
                zip(("intercept",) + fits["full"].features, np.round(fits["full"].weights, 3).tolist())
            ),
        }
    return report


def read_holdout(today: date) -> dict:
    if HOLDOUT_REPORT.exists():
        raise RuntimeError(f"The prospect holdout was already read ({HOLDOUT_REPORT}); it is read once")
    if None in CHOICE.values():
        raise RuntimeError("Commit CHOICE from the development report before reading the holdout")
    snaps = matured(today, DEVELOPMENT + VALIDATION + HOLDOUT)
    report = {
        "choice": CHOICE,
        "rule": "publish a role if its log loss minus the base rate has a 90% player interval wholly below zero",
    }
    for role in ROLES:
        part = snaps[snaps.role == role]
        hold = part[part.season.isin(HOLDOUT)]
        train = part[part.season.isin(DEVELOPMENT + VALIDATION) & ~part.player.isin(hold.player)]
        fits = {spec: model.fit(train, features_for(role, spec)) for spec in ("baseline", "full")}
        block = _score(hold, {spec: m.predict(hold) for spec, m in fits.items()}, float(train.success.mean()))
        block["training"] = {
            "snapshots": int(len(train)),
            "players": int(train.player.nunique()),
            "successes": int(train.success.sum()),
        }
        block["publish"] = block[CHOICE[role]]["vs_base_rate_ci90"][1] < 0
        report[role] = block
    HOLDOUT_REPORT.write_text(json.dumps(report, indent=2))
    return report
