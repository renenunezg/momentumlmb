"""Ground-ball quartiles per batter and pitcher from the scorer's batted-ball
type (ported from mlbmodel v2/simulator/gb_quartiles.py). Quartile 0 is the
lowest ground-ball share; thin or unknown players get the median bin."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

BIP_TYPES = ("ground_ball", "fly_ball", "line_drive", "popup")
MIN_BIP = 100
MEDIAN_Q = 2
N_QUARTILES = 4


def _gb_rate_by(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    bip = df[df["bb_type"].isin(BIP_TYPES)]
    g = bip.groupby(id_col)
    out = pd.DataFrame({"bip": g.size(), "gb": g["bb_type"].apply(lambda s: (s == "ground_ball").sum())})
    out["gb_pct"] = out["gb"] / out["bip"]
    return out.reset_index().rename(columns={id_col: "player_id"})


def _assign_quartiles(rates: pd.DataFrame, role: str) -> pd.DataFrame:
    qualified = rates[rates["bip"] >= MIN_BIP].copy()
    qualified["gb_q"] = pd.qcut(qualified["gb_pct"], N_QUARTILES, labels=False, duplicates="drop").astype(np.int64)
    rates = rates.merge(qualified[["player_id", "gb_q"]], on="player_id", how="left")
    rates["gb_q"] = rates["gb_q"].fillna(MEDIAN_Q).astype(np.int64)
    rates["role"] = role
    return rates[["player_id", "role", "gb_q"]]


def build_gb_quartiles(pa: pd.DataFrame) -> pd.DataFrame:
    df = pa[pa["game_type"] == "R"]
    return pd.concat(
        [_assign_quartiles(_gb_rate_by(df, "batter"), "B"), _assign_quartiles(_gb_rate_by(df, "pitcher"), "P")],
        ignore_index=True,
    )


@dataclass
class GBQuartiles:
    batter: dict[int, int]
    pitcher: dict[int, int]

    def batter_q(self, ids: np.ndarray) -> np.ndarray:
        return np.array([self.batter.get(int(i), MEDIAN_Q) for i in ids], dtype=np.int64)

    def pitcher_q(self, ids: np.ndarray) -> np.ndarray:
        return np.array([self.pitcher.get(int(i), MEDIAN_Q) for i in ids], dtype=np.int64)


def load_gb_quartiles(tables_dir: Path) -> GBQuartiles:
    df = pd.read_parquet(tables_dir / "gb_quartiles.parquet")
    bat = df[df.role == "B"]
    pit = df[df.role == "P"]
    return GBQuartiles(
        batter=dict(zip(bat.player_id.astype(int), bat.gb_q.astype(int))),
        pitcher=dict(zip(pit.player_id.astype(int), pit.gb_q.astype(int))),
    )
