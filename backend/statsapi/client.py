"""Thin MLB Stats API client (free, unkeyed). Retries with backoff and paces
calls so bulk cache builds stay polite."""

import time

import requests

BASE_URL = "https://statsapi.mlb.com/api"
PACE_SECONDS = 0.15
RETRIES = 4

_session = requests.Session()
_last_call = 0.0


def get(path: str, **params) -> dict:
    """GET `BASE_URL + path` (path starts with /v1 or /v1.1). Raises after
    RETRIES failed attempts."""
    global _last_call
    url = BASE_URL + path
    last_error: Exception | None = None
    for attempt in range(RETRIES):
        wait = PACE_SECONDS - (time.monotonic() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.monotonic()
        try:
            response = _session.get(url, params=params, timeout=30)
            if response.status_code == 200:
                return response.json()
            last_error = RuntimeError(f"{response.status_code} for {response.url}")
            if 400 <= response.status_code < 500 and response.status_code != 429:
                break
        except requests.RequestException as error:
            last_error = error
        time.sleep(2**attempt)
    raise RuntimeError(f"Stats API request failed: {url} {params}") from last_error
