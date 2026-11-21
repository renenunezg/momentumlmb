"""Player ratings commands.

    python -m backend.players.run develop   # choices and validation on 2021-2025
    python -m backend.players.run build     # 2021-2026 WAR, grades for 2027, the one 2026 holdout read
    python -m backend.players.run publish   # dry run; writes only with MOMENTUMLMB_DB_WRITES=1

`develop` writes artifacts/players/development.json; `build` writes
artifacts/players/war.parquet, grades.parquet and build.json."""

import argparse
import json

import pandas as pd

from backend.config import ARTIFACTS_DIR, DEVELOPMENT_SEASONS, HOLDOUT_SEASON
from backend.players import grade as G
from backend.players import positions, validate, value
from backend.players.context import context_probs
from backend.players.people import load_people, season_age
from backend.players.war import WarBuild, build_war

OUT = ARTIFACTS_DIR / "players"
DEV = list(DEVELOPMENT_SEASONS)
DEV_TARGETS = [s for s in DEV if s > DEV[0]]
ALL = [*DEV, HOLDOUT_SEASON]
MODEL_VERSION = "lmb-war-v1"


def _ages(players: pd.DataFrame) -> pd.DataFrame:
    people = load_people(players["player"].unique().tolist())
    a = players[["season", "player"]].merge(people[["player", "birth_date"]], on="player", how="left")
    a["age"] = season_age(a["birth_date"], a["season"])
    # The season after the last one, for grading the next season.
    nxt = a[a["season"] == a["season"].max()].assign(season=lambda d: d["season"] + 1)
    nxt["age"] = season_age(nxt["birth_date"], nxt["season"])
    return pd.concat([a, nxt])[["season", "player", "age"]]


def _war_validation(b: WarBuild, seasons: list[int]) -> dict:
    p, pa = b.players, b.seasons.pa.reset_index(drop=True)
    probs = context_probs(pa, max(seasons))
    ctx, _ = value.context_value(probs, b.seasons.lw, pa["season"])
    v = value.pa_values(pa, b.seasons.lw, probs, pd.Series(dtype=float))
    rec = validate.team_reconciliation(p, b.replacement, pa)
    targets = [s for s in seasons if s > seasons[0]]
    fw = validate.forward_team_wins(p, b.replacement, pa, targets)
    return {
        "choose_pitching": validate.choose_pitching(p, pa, pd.Series(ctx), targets),
        "team_reconciliation": {
            "war": validate.summarize_fit(rec, "war_wins"),
            "pythagorean_ceiling": validate.summarize_fit(rec, "pyth_wins"),
            "by_season": {int(s): validate.summarize_fit(g, "war_wins") for s, g in rec.groupby("season")},
        },
        "forward_team_wins": validate.summarize_fit(fw, "proj_wins"),
        "repeatability": validate.repeatability(
            p,
            {
                "batting": "pa",
                "baserunning": "pa",
                "fld_zone": "def_pa",
                "throwing": "def_pa",
                "pitch_a": "bf",
                "pitch_b": "bf",
                "pitch_c": "bf",
                "war": "pa",
            },
            {"pa": 200, "def_pa": 1000, "bf": 200},
            seasons,
        ),
        "park_check": validate.park_check(p, v, pa),
        "replacement": b.level,
        "runs_per_win": {int(k): round(v, 2) for k, v in b.seasons.runs_per_win.items()},
    }


def develop() -> dict:
    report = {}
    b = build_war(DEV, max(DEV), "a")
    report["war_variant_a"] = _war_validation(b, DEV)
    choice = report["war_variant_a"]["choose_pitching"]
    variant = max(choice, key=lambda k: choice[k]["r"][0]).split("_")[1]
    report["pitching_variant"] = variant
    if variant != "a":
        b = build_war(DEV, max(DEV), variant)
        report[f"war_variant_{variant}"] = _war_validation(b, DEV)
    zone = positions.fit_spectrum(_zone_frame(b), n_boot=200)
    report["positional_spectrum"] = zone

    ages = _ages(b.players)
    model, hits, pits = G.fit_model(b.players, ages, b.seasons.lw, DEV_TARGETS)
    report["grade_model"] = _model_dict(model)
    report["age_curve_check"] = G.age_curve_check(b.players, ages, hits, pits)
    report["grade_forward"] = {
        "hitters": validate.grade_forward(
            hits.dropna(subset=["prior_rate", "marcel_rate"]),
            "target",
            ["proj_rate", "marcel_rate", "prior_rate", "bat_post"],
        ),
        "pitchers": validate.grade_forward(
            pits.dropna(subset=["prior_rate", "marcel_rate"]),
            "target",
            ["proj_rate", "marcel_rate", "prior_rate", "pitch_post"],
        ),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "development.json").write_text(json.dumps(report, indent=1, default=float))
    return report


def _zone_frame(b: WarBuild) -> pd.DataFrame:
    from backend.players import fielding

    pa = b.seasons.pa.reset_index(drop=True)
    bip = fielding.balls_in_play(pa, b.seasons.lw)
    bip["expected"] = fielding.expected_value(bip)
    credits = fielding.fielding_credits(bip, fielding.responsibility_shares(bip))
    return fielding.zone_fielding(credits, fielding.defensive_pa(pa))


def _model_dict(m: G.GradeModel) -> dict:
    return {
        "hitter_coef": m.hitter_coef,
        "pitcher_coef": m.pitcher_coef,
        "shrink": {k: round(v, 1) for k, v in m.shrink.items()},
        "talent_sd": {k: round(v, 4) for k, v in m.talent_sd.items()},
    }


def _holdout_read(b_dev: WarBuild, b: WarBuild, ages: pd.DataFrame, model: G.GradeModel) -> dict:
    """The one read of the 2026 holdout: grades scored on 2026's opening day
    with the development model, against 2026's actual rates. The features come
    from the development build (park context fitted through 2025), so nothing
    from 2026 reaches them; only the 2026 target rates come from the full build."""
    t = HOLDOUT_SEASON
    history = pd.concat([b_dev.players, b.players[b.players["season"] == t]], ignore_index=True)
    h, p = G.target_rows(t, history, ages, model.shrink, b_dev.seasons.lw.loc[t - 1])
    h, p = G.project(h, p, model)
    cols_h = ["proj_rate", "marcel_rate", "prior_rate", "bat_post"]
    cols_p = ["proj_rate", "marcel_rate", "prior_rate", "pitch_post"]
    return {
        "hitters": validate.grade_forward(h.dropna(subset=["prior_rate", "marcel_rate"]), "target", cols_h),
        "pitchers": validate.grade_forward(p.dropna(subset=["prior_rate", "marcel_rate"]), "target", cols_p),
    }


def build() -> dict:
    dev = json.loads((OUT / "development.json").read_text())
    variant = dev["pitching_variant"]
    b_dev = build_war(DEV, max(DEV), variant)
    model, _, _ = G.fit_model(b_dev.players, _ages(b_dev.players), b_dev.seasons.lw, DEV_TARGETS)

    b = build_war(ALL, HOLDOUT_SEASON, variant)
    ages = _ages(b.players)
    report = {
        "model_version": MODEL_VERSION,
        "pitching_variant": variant,
        "grade_model": _model_dict(model),
        "holdout_2026_grade": _holdout_read(b_dev, b, ages, model),
        "war_2021_2026": _war_validation(b, ALL),
    }
    last = HOLDOUT_SEASON
    grades = G.score_grades(
        last + 1, b.players, ages, model, b.seasons.lw.loc[last], b.replacement.loc[last], b.seasons.runs_per_win[last]
    )
    people = load_people(b.players["player"].unique().tolist())
    war = b.players.merge(people[["player", "name"]], on="player", how="left").merge(
        ages, on=["season", "player"], how="left"
    )
    OUT.mkdir(parents=True, exist_ok=True)
    war.to_parquet(OUT / "war.parquet", index=False)
    grades.to_parquet(OUT / "grades.parquet", index=False)
    report["season_scale"] = {
        "linear_weights": {int(s): b.seasons.lw.loc[s].round(3).to_dict() for s in ALL},
        "replacement_rates": b.replacement.round(4).reset_index().to_dict("records"),
        "positional_runs_per_162": b.pos_scale,
    }
    (OUT / "build.json").write_text(json.dumps(report, indent=1, default=float))
    return report


def model_details(report: dict) -> dict:
    """What the site's info tips quote: the latest season's scale and the
    headline validation (`report` as read back from build.json)."""
    last = str(HOLDOUT_SEASON)
    val = report["war_2021_2026"]
    return {
        "linear_weights": report["season_scale"]["linear_weights"][last],
        "runs_per_win": val["runs_per_win"],
        "replacement": val["replacement"],
        "positional_runs_per_162": report["season_scale"]["positional_runs_per_162"],
        "pitching_variant": report["pitching_variant"],
        "grade_model": report["grade_model"],
        "validation": {
            "team_reconciliation": val["team_reconciliation"]["war"],
            "pythagorean_ceiling": val["team_reconciliation"]["pythagorean_ceiling"],
            "forward_team_wins": val["forward_team_wins"],
            "park_check": val["park_check"],
            "holdout_2026_grade": {k: v["pooled"] for k, v in report["holdout_2026_grade"].items()},
        },
    }


def publish_command(write: bool) -> dict:
    from backend.players import publish

    report = json.loads((OUT / "build.json").read_text())
    war = pd.read_parquet(OUT / "war.parquet")
    grades = pd.read_parquet(OUT / "grades.parquet")
    rows = publish.rating_rows(war, grades, HOLDOUT_SEASON)
    details = model_details(report)
    info = publish.size_estimate(rows, details)
    if write:
        publish.publish(rows, HOLDOUT_SEASON, MODEL_VERSION, details)
        info["published"] = True
    return info


def main() -> None:
    parser = argparse.ArgumentParser(prog="backend.players.run")
    parser.add_argument("command", choices=["develop", "build", "publish"])
    parser.add_argument("--write", action="store_true", help="publish to Supabase (needs MOMENTUMLMB_DB_WRITES=1)")
    args = parser.parse_args()
    if args.command == "develop":
        print(json.dumps(develop(), indent=1, default=float))
    elif args.command == "build":
        print(json.dumps(build(), indent=1, default=float))
    else:
        print(json.dumps(publish_command(args.write), indent=1, default=float))


if __name__ == "__main__":
    main()
