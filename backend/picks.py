"""Moneyline picks against Kalshi (policy lmb-picks-v1, the shape of mlbmodel's
mlb-picks-v3): at most one pick per game, the side with the highest expected
value at its executable Kalshi ask plus fee, priced from the Kalshi midpoint
plus a small lean toward the sim; tier `value` (EV >= 0) or `lean` (EV >=
PICK_MIN_EV), plus each date's minimum slate as tier `minimum`.

On a single exchange the ask sits above the midpoint, so market-only EV is
always negative and the slate is chosen mostly by spread width; the pick
record is expected to run slightly negative (accepted by Rene)."""

from datetime import date, datetime
from typing import Protocol

from backend import strategy
from backend.markets.kalshi import Quote, ev_per_dollar, market_home_prob, trading_fee
from backend.markets.probs import pick_prob


class SlateStore(Protocol):
    def get(self, day: date) -> list[int] | None: ...

    def put(self, day: date, ranked_at: datetime, game_pks: list[int]) -> None: ...


def slate_rank_time(start_times: list[datetime]) -> datetime:
    return min(start_times) - strategy.SLATE_LEAD


def _market_only_value(home: Quote | None, away: Quote | None) -> float | None:
    p_home = market_home_prob(home, away)
    if p_home is None:
        return None
    return max(ev_per_dollar(p_home, home.yes_ask), ev_per_dollar(1 - p_home, away.yes_ask))


def rank_slate(candidates: list[dict], n: int = strategy.PICK_DAILY_MINIMUM) -> list[int]:
    """Game pks with the best market-only EV at their asks; candidates are
    dicts with game_pk, home and away quotes."""
    scored = [(v, c["game_pk"]) for c in candidates if (v := _market_only_value(c["home"], c["away"])) is not None]
    return [pk for _, pk in sorted(scored, key=lambda x: (-x[0], x[1]))[:n]]


def resolve_slate(
    day: date, now: datetime, start_times: list[datetime], candidates: list[dict], store: SlateStore
) -> list[int] | None:
    """The date's slate: the stored one if it exists; otherwise ranked and
    stored once the ranking time has come; None before then."""
    stored = store.get(day)
    if stored is not None:
        return stored
    if not start_times or now < slate_rank_time(start_times):
        return None
    slate = rank_slate(candidates)
    store.put(day, now, slate)
    return slate


def choose_pick(
    game_pk: int,
    home: Quote | None,
    away: Quote | None,
    p_market_home: float | None,
    p_sim_home: float,
    slate: list[int] | None,
    *,
    ready: bool,
    picks_enabled: bool = strategy.PICKS_ENABLED,
) -> dict | None:
    """The game's pick, or None. `ready` means both starters are known, both
    lineups are posted and both bullpens' usage is known."""
    if not picks_enabled or not ready or p_market_home is None or home is None or away is None:
        return None
    if home.yes_ask is None or away.yes_ask is None:
        return None
    sides = []
    for side, quote, p_market, p_sim in (
        ("home", home, p_market_home, p_sim_home),
        ("away", away, 1 - p_market_home, 1 - p_sim_home),
    ):
        p = pick_prob(p_market, p_sim, strategy.PICK_MODEL_WEIGHT)
        sides.append((ev_per_dollar(p, quote.yes_ask), side, quote, p))
    ev, side, quote, p = max(sides, key=lambda s: s[0])
    if ev >= 0:
        tier = "value"
    elif ev >= strategy.PICK_MIN_EV:
        tier = "lean"
    elif slate is not None and game_pk in slate:
        tier = "minimum"
    else:
        return None
    fee = trading_fee(quote.yes_ask)
    return {
        "market": "ml",
        "side": side,
        "ticker": quote.ticker,
        "probability": round(p, 4),
        "price": quote.yes_ask,
        "fee": round(fee, 6),
        "expected_value": round(ev, 4),
        "edge": round(p - (quote.yes_ask + fee), 4),
        "tier": tier,
        "policy": strategy.PICK_POLICY,
    }
