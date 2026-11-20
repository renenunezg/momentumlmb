"""One-time pull of LMB closing moneylines from BetExplorer (the same data as
OddsPortal). Its results pages list, per game, one home and away decimal
price pair: very likely the average close across the books it tracks (median
overround about 7%). The pages are plain HTML, so no browser is needed.

Used for fitting and evaluation only; the raw lines are never published."""

import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from backend.config import CACHE_DIR, STATIC_DIR

BASE = "https://www.betexplorer.com/baseball/mexico"
HTML_DIR = CACHE_DIR / "betexplorer" / "html"
CLOSES = CACHE_DIR / "market" / "betexplorer_closes.parquet"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/130.0 Safari/537.36"
}
PACE_SECONDS = 3.0
STAGES = ("Main", "Play Offs")
ROW = re.compile(r"<tr>(.*?)</tr>", re.S)
TEAMS = re.compile(r'class="in-match"><span>(?:<strong>)?([^<]+)(?:</strong>)?</span> - <span>(?:<strong>)?([^<]+)')
SCORE = re.compile(r">(\d+):(\d+)<")
ODDS = re.compile(r'data-odd="([\d.]+)"')
DATE = re.compile(r">(\d{2})\.(\d{2})\.(\d{4})?<")  # the current season omits the year


def _season_url(season: int, current: int) -> str:
    return f"{BASE}/lmb/results/" if season == current else f"{BASE}/lmb-{season}/results/"


def _fetch(url: str, path: Path) -> str:
    if path.exists():
        return path.read_text()
    time.sleep(PACE_SECONDS)
    response = requests.get(url, headers=HEADERS, timeout=60)
    response.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(response.text)
    return response.text


def parse_rows(html: str, season: int, stage: str) -> pd.DataFrame:
    rows = []
    for row in ROW.findall(html):
        teams, score, date = TEAMS.search(row), SCORE.search(row), DATE.search(row)
        odds = ODDS.findall(row)
        if not (teams and score and date and len(odds) == 2):
            continue
        dd, mm, yyyy = date.groups()
        rows.append(
            {
                "season": season,
                "stage": stage,
                "date": datetime(int(yyyy or season), int(mm), int(dd)).date(),
                "home_name": teams.group(1).strip(),
                "away_name": teams.group(2).strip(),
                "home_score": int(score.group(1)),
                "away_score": int(score.group(2)),
                "home_decimal": float(odds[0]),
                "away_decimal": float(odds[1]),
            }
        )
    return pd.DataFrame(rows)


def scrape(seasons: list[int], current: int) -> pd.DataFrame:
    """Every regular-season and playoff row with both prices, cached page by page."""
    frames = []
    for season in seasons:
        url = _season_url(season, current)
        landing = _fetch(url, HTML_DIR / f"{season}_landing.html")
        stages = {name: sid for sid, name in re.findall(r'href="\?stage=([^"&]+)"[^>]*>([^<]*)<', landing)}
        for stage in STAGES:
            if stage not in stages:
                continue
            html = _fetch(f"{url}?stage={stages[stage]}&month=all", HTML_DIR / f"{season}_{stages[stage]}.html")
            frames.append(parse_rows(html, season, stage))
    return pd.concat(frames, ignore_index=True)


def match(closes: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Attach gamePks. A row matches the final game with the same home and away
    teams and score on the listed date or a day either side (time zones);
    doubleheaders resolve by score. Rows that do not resolve to exactly one
    game are dropped and counted."""
    names = pd.read_csv(STATIC_DIR / "betexplorer_teams.csv")
    team = dict(zip(names.betexplorer_name, names.team_id))
    unknown = (set(closes.home_name) | set(closes.away_name)) - set(team)
    if unknown:
        raise ValueError(f"Unmapped BetExplorer team names: {sorted(unknown)}")
    final = games[games.status.isin(["Final", "Completed Early"])]
    by_key = final.groupby(["home_team_id", "away_team_id", "home_score", "away_score"])
    groups = {k: g for k, g in by_key}
    out, dropped = [], 0
    for r in closes.itertuples(index=False):
        key = (team[r.home_name], team[r.away_name], r.home_score, r.away_score)
        cand = groups.get(key)
        if cand is not None:
            near = (pd.to_datetime(cand.official_date) - pd.Timestamp(r.date)).abs() <= timedelta(days=1)
            cand = cand[near.to_numpy()]
        if cand is None or len(cand) != 1:
            dropped += 1
            continue
        out.append(
            {"game_pk": int(cand.game_pk.iloc[0]), "home_decimal": r.home_decimal, "away_decimal": r.away_decimal}
        )
    matched = pd.DataFrame(out).drop_duplicates("game_pk", keep=False)
    matched.attrs["dropped"] = dropped
    return matched


def build(seasons: list[int], current: int, games: pd.DataFrame) -> Path:
    closes = scrape(seasons, current)
    matched = match(closes, games)
    CLOSES.parent.mkdir(parents=True, exist_ok=True)
    matched.to_parquet(CLOSES, index=False)
    final = games[games.status.isin(["Final", "Completed Early"])]
    rate = final.assign(hit=final.game_pk.isin(matched.game_pk)).groupby("season").hit.mean().round(3)
    overround = (1 / matched.home_decimal + 1 / matched.away_decimal - 1).median()
    print(f"BetExplorer rows {len(closes)}, matched {len(matched)}, dropped {matched.attrs['dropped']}")
    print(f"match rate by season {rate.to_dict()}; median overround {overround:.3f}")
    return CLOSES
