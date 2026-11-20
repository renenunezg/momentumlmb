"""Venue coordinates, elevation and time zone (the Stats API has no
elevation for any LMB park). Park effects key on venue_id."""

from functools import cache

import pandas as pd

from backend.config import STATIC_DIR


@cache
def load_venues() -> pd.DataFrame:
    return pd.read_csv(STATIC_DIR / "venues.csv").set_index("venue_id")


def venue(venue_id: int) -> pd.Series:
    venues = load_venues()
    if venue_id not in venues.index:
        raise KeyError(f"venue {venue_id} is not in backend/data_static/venues.csv; add it")
    return venues.loc[venue_id]
