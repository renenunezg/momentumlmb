"""Kalshi LMB game-winner markets (series KXLMBGAME): free public market data,
no key. One event per game with two yes markets, one per team.

Facts checked 2026-11-19:
- Event tickers encode the scheduled first pitch in US Eastern time
  (KXLMBGAME-26SEP152235... is Sep 15 2026 10:35 PM EDT, as the market rules
  text states).
- The series fee is quadratic with multiplier 1. Kalshi's fee page sits behind
  a bot checkpoint, so the formula is taken from Kalshi's published general
  schedule: 0.07 x price x (1 - price) per contract, model fee rounded up to
  $0.000001 per fill (docs.kalshi.com fee rounding). Cent rounding is not
  applied: a 1u pick is many contracts.
- Kalshi covered every LMB game from 2026-06-30 (419 games, 838 markets).
  Whether it lists the LMB in 2027 is unknown; no market means no pick."""

import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from backend.config import CACHE_DIR, STATIC_DIR

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
SERIES = "KXLMBGAME"
FEE_RATE = 0.07
EASTERN = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
TICKER_TIME = re.compile(r"^KXLMBGAME-(\d{2})([A-Z]{3})(\d{2})(\d{4})")
MONTHS = {
    m: i for i, m in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)
}
MATCH_TOLERANCE = timedelta(hours=3)
KALSHI_2026 = CACHE_DIR / "market" / "kalshi_2026.parquet"

_session = requests.Session()


@dataclass(frozen=True)
class KalshiGame:
    event_ticker: str
    game_pk: int
    home_ticker: str
    away_ticker: str


@dataclass(frozen=True)
class Quote:
    ticker: str
    team_id: int
    yes_bid: float | None
    yes_ask: float | None
    fetched_at: datetime


def _get(path: str, **params) -> dict:
    response = None
    for attempt in range(4):
        response = _session.get(BASE_URL + path, params=params, timeout=30)
        if response.status_code == 200:
            return response.json()
        time.sleep(2**attempt)
    response.raise_for_status()
    raise RuntimeError(f"Kalshi request failed: {path}")


def _team_ids() -> dict[str, int]:
    frame = pd.read_csv(STATIC_DIR / "kalshi_teams.csv")
    return dict(zip(frame.kalshi_name, frame.team_id))


def ticker_start(event_ticker: str) -> datetime:
    """Scheduled first pitch encoded in the event ticker, as UTC."""
    m = TICKER_TIME.match(event_ticker)
    if not m:
        raise ValueError(f"Unrecognized event ticker {event_ticker}")
    yy, mon, dd, hhmm = m.groups()
    local = datetime(2000 + int(yy), MONTHS[mon], int(dd), int(hhmm[:2]), int(hhmm[2:]), tzinfo=EASTERN)
    return local.astimezone(UTC)


def _price(value) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)) or value == "":
        return None
    price = float(value)
    return price if 0.0 < price < 1.0 else None


def events(day: date) -> list[dict]:
    """Open or upcoming LMB events whose ticker time falls on `day` (Eastern),
    each with its markets."""
    payload = _get("/events", series_ticker=SERIES, with_nested_markets="true", limit=200)
    return [e for e in payload.get("events", []) if ticker_start(e["event_ticker"]).astimezone(EASTERN).date() == day]


def match_events(event_list: list[dict], games: pd.DataFrame) -> dict[int, KalshiGame]:
    """Map Kalshi events to Stats API games: same team pair, same official
    date as the ticker's Eastern date, first pitch within three hours.
    Doubleheaders resolve by order: the pair's events that day, sorted by
    ticker time, go to game 1, game 2 (the Stats API lists game 2 five minutes
    after game 1, so the nearest start would pick the wrong game). The date
    check keeps a resumed suspended game (official date weeks earlier) from
    taking that day's regular game's market. Unmatched events are skipped,
    never guessed."""
    ids = _team_ids()
    pairs = games.apply(lambda g: frozenset((int(g.home_team_id), int(g.away_team_id))), axis=1)
    parsed = []
    for event in event_list:
        markets = {ids.get(m["yes_sub_title"]): m["ticker"] for m in event.get("markets", [])}
        if None in markets or len(markets) != 2:
            continue
        start = pd.Timestamp(ticker_start(event["event_ticker"]))
        parsed.append((event, markets, start, frozenset(markets), start.tz_convert(EASTERN).date()))
    order: dict[tuple, list[pd.Timestamp]] = {}
    for _, _, start, pair, day in parsed:
        order.setdefault((pair, day), []).append(start)

    out: dict[int, KalshiGame] = {}
    for event, markets, start, pair, day in parsed:
        gap = (games.start_time_utc - start).abs()
        candidates = games[(pairs == pair) & (games.official_date == day) & (gap <= MATCH_TOLERANCE)]
        if candidates.empty:
            continue
        rank = sorted(order[(pair, day)]).index(start) + 1
        if len(candidates) > 1:
            candidates = candidates[candidates.game_number == rank]
            if len(candidates) != 1:
                continue
        g = candidates.iloc[0]
        if int(g.game_pk) in out:
            continue
        out[int(g.game_pk)] = KalshiGame(
            event_ticker=event["event_ticker"],
            game_pk=int(g.game_pk),
            home_ticker=markets[int(g.home_team_id)],
            away_ticker=markets[int(g.away_team_id)],
        )
    return out


def quotes(game: KalshiGame, home_team_id: int, away_team_id: int) -> tuple[Quote, Quote]:
    fetched = datetime.now(UTC)
    out = []
    for ticker, team in ((game.home_ticker, home_team_id), (game.away_ticker, away_team_id)):
        market = _get(f"/markets/{ticker}")["market"]
        out.append(
            Quote(ticker, team, _price(market.get("yes_bid_dollars")), _price(market.get("yes_ask_dollars")), fetched)
        )
    return out[0], out[1]


def market_home_prob(home: Quote | None, away: Quote | None) -> float | None:
    """Midpoints of both yes markets, normalized to sum to one."""
    if home is None or away is None or None in (home.yes_bid, home.yes_ask, away.yes_bid, away.yes_ask):
        return None
    mid_home = (home.yes_bid + home.yes_ask) / 2
    mid_away = (away.yes_bid + away.yes_ask) / 2
    return mid_home / (mid_home + mid_away)


def trading_fee(price: float) -> float:
    return FEE_RATE * price * (1.0 - price)


def ev_per_dollar(p: float, ask: float) -> float:
    """Expected profit per dollar spent buying yes at `ask` plus the fee."""
    return p / (ask + trading_fee(ask)) - 1.0


def historical_markets() -> pd.DataFrame:
    rows, cursor = [], None
    while True:
        params = {"series_ticker": SERIES, "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        payload = _get("/historical/markets", **params)
        rows += payload.get("markets", [])
        cursor = payload.get("cursor")
        if not cursor or not payload.get("markets"):
            break
    return pd.DataFrame(rows)


def historical_candles(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    payload = _get(
        f"/historical/markets/{ticker}/candlesticks",
        start_ts=int(start.timestamp()),
        end_ts=int(end.timestamp()),
        period_interval=60,
    )
    return pd.json_normalize(payload.get("candlesticks", []))


def build_2026_cache(games: pd.DataFrame) -> Path:
    """For each 2026 Kalshi game, the last hourly bid and ask before first pitch
    for both teams. Holdout data: nothing may be tuned on it."""
    markets = historical_markets()
    event_list = [
        {"event_ticker": t, "markets": g[["ticker", "yes_sub_title"]].to_dict("records")}
        for t, g in markets.groupby("event_ticker")
    ]
    matched = match_events(event_list, games)
    rows = []
    for game_pk, kg in matched.items():
        first_pitch = games.loc[games.game_pk == game_pk, "start_time_utc"].iloc[0].to_pydatetime()
        row = {"game_pk": game_pk, "event_ticker": kg.event_ticker}
        for side, ticker in (("home", kg.home_ticker), ("away", kg.away_ticker)):
            candles = historical_candles(ticker, first_pitch - timedelta(days=4), first_pitch)
            if not candles.empty:
                ends = pd.to_datetime(candles["end_period_ts"], unit="s", utc=True)
                candles = candles[ends <= pd.Timestamp(first_pitch)]
            last = candles.iloc[-1] if len(candles) else None
            row[f"{side}_bid"] = _price(last.get("yes_bid.close")) if last is not None else None
            row[f"{side}_ask"] = _price(last.get("yes_ask.close")) if last is not None else None
            row[f"{side}_quote_ts"] = (
                pd.to_datetime(last["end_period_ts"], unit="s", utc=True) if last is not None else pd.NaT
            )
        rows.append(row)
    KALSHI_2026.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)
    out.to_parquet(KALSHI_2026, index=False)
    print(f"Kalshi 2026: {markets.event_ticker.nunique()} events, {len(matched)} matched to gamePks")
    return KALSHI_2026
