"""Names, birth dates and handedness from the Stats API people endpoint,
cached once per id set."""

import pandas as pd

from backend.config import CACHE_DIR
from backend.statsapi.client import get

PEOPLE_PATH = CACHE_DIR / "players" / "people.parquet"
BATCH = 100


def load_people(ids: list[int]) -> pd.DataFrame:
    cached = pd.read_parquet(PEOPLE_PATH) if PEOPLE_PATH.exists() else pd.DataFrame(columns=["player"])
    missing = sorted(set(int(i) for i in ids) - set(cached["player"].astype(int)))
    rows = []
    for start in range(0, len(missing), BATCH):
        chunk = missing[start : start + BATCH]
        try:
            payload = get("/v1/people", personIds=",".join(map(str, chunk)))
        except RuntimeError:
            if len(chunk) > 1:
                raise
            payload = {"people": []}  # an id the API no longer serves
        found = {p["id"] for p in payload.get("people", [])}
        rows.extend({"player": pid, "name": None} for pid in chunk if pid not in found)
        for p in payload.get("people", []):
            rows.append(
                {
                    "player": p["id"],
                    "name": p.get("fullName"),
                    "birth_date": p.get("birthDate"),
                    "bats": (p.get("batSide") or {}).get("code"),
                    "throws": (p.get("pitchHand") or {}).get("code"),
                    "primary_position": (p.get("primaryPosition") or {}).get("abbreviation"),
                }
            )
    if rows:
        cached = pd.concat([cached, pd.DataFrame(rows)], ignore_index=True)
        PEOPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
        cached.to_parquet(PEOPLE_PATH, index=False)
    return cached[cached["player"].isin(ids)]


def season_age(birth_date: pd.Series, season: pd.Series) -> pd.Series:
    """Age on June 30 of the season (the LMB's mid-season)."""
    born = pd.to_datetime(birth_date)
    mid = pd.to_datetime(season.astype(str) + "-06-30")
    return (
        mid.dt.year
        - born.dt.year
        - ((mid.dt.month < born.dt.month) | ((mid.dt.month == born.dt.month) & (mid.dt.day < born.dt.day)))
    ).astype("Int64")
