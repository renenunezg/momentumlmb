"""Game-time weather from Open-Meteo: the historical archive for training and
the forecast API for pregame scoring. The Stats API's own weather is typed by
hand and unreliable, so it is never read."""

import pandas as pd
import requests

from backend.config import CACHE_DIR
from backend.venues import venue

WEATHER_DIR = CACHE_DIR / "weather"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HOURLY = "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m"
COLUMNS = ["game_pk", "temp_c", "rh", "pressure_hpa", "wind_kph", "wind_dir", "air_density"]


def air_density(temp_c: float, rh: float, pressure_hpa: float) -> float:
    """Moist air density in kg/m3 (Tetens saturation vapour pressure)."""
    saturation = 6.1078 * 10 ** (7.5 * temp_c / (temp_c + 237.3))
    vapour = rh / 100.0 * saturation
    dry = pressure_hpa - vapour
    kelvin = temp_c + 273.15
    return (dry * 100.0) / (287.058 * kelvin) + (vapour * 100.0) / (461.495 * kelvin)


def _hourly(url: str, lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    response = requests.get(
        url,
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": start,
            "end_date": end,
            "hourly": HOURLY,
            "timezone": "UTC",
            "wind_speed_unit": "kmh",
        },
        timeout=60,
    )
    response.raise_for_status()
    hourly = response.json()["hourly"]
    frame = pd.DataFrame(hourly)
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    return frame.set_index("time")


def _at_first_pitch(games: pd.DataFrame, url: str) -> pd.DataFrame:
    rows = []
    for venue_id, group in games.groupby("venue_id"):
        v = venue(int(venue_id))
        start = group["start_time_utc"].min().date().isoformat()
        end = (group["start_time_utc"].max() + pd.Timedelta(hours=1)).date().isoformat()
        hourly = _hourly(url, v.lat, v.lon, start, end)
        for g in group.itertuples(index=False):
            hour = g.start_time_utc.floor("h")
            if hour not in hourly.index:
                continue
            h = hourly.loc[hour]
            if any(pd.isna(h[c]) for c in ("temperature_2m", "relative_humidity_2m", "surface_pressure")):
                continue
            rows.append(
                {
                    "game_pk": g.game_pk,
                    "temp_c": h["temperature_2m"],
                    "rh": h["relative_humidity_2m"],
                    "pressure_hpa": h["surface_pressure"],
                    "wind_kph": h["wind_speed_10m"],
                    "wind_dir": h["wind_direction_10m"],
                    "air_density": air_density(h["temperature_2m"], h["relative_humidity_2m"], h["surface_pressure"]),
                }
            )
    return pd.DataFrame(rows, columns=COLUMNS)


def archive_for_games(games: pd.DataFrame) -> pd.DataFrame:
    """Weather at the hour of first pitch for finished games, one archive
    request per venue per season, cached per season."""
    WEATHER_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for season, group in games.groupby("season"):
        path = WEATHER_DIR / f"{season}.parquet"
        if path.exists():
            cached = pd.read_parquet(path)
            if set(group["game_pk"]) <= set(cached["game_pk"]):
                frames.append(cached[cached["game_pk"].isin(group["game_pk"])])
                continue
        fresh = _at_first_pitch(group, ARCHIVE_URL)
        fresh.to_parquet(path, index=False)
        frames.append(fresh)
    return pd.concat(frames, ignore_index=True)


def forecast_for_games(games: pd.DataFrame) -> pd.DataFrame:
    """Forecast weather at first pitch for upcoming games (up to 16 days out)."""
    return _at_first_pitch(games, FORECAST_URL)
