"""The daily minimum slate decides almost every published pick, so it must be
ranked once and never re-ranked, and a slate game that loses its market gets
no pick and no stand-in."""

from datetime import UTC, date, datetime, timedelta

from backend.markets.kalshi import Quote
from backend.picks import choose_pick, resolve_slate

DAY = date(2027, 5, 1)
FIRST_PITCH = datetime(2027, 5, 2, 1, 0, tzinfo=UTC)


class MemorySlates:
    def __init__(self):
        self.rows: dict[date, list[int]] = {}

    def get(self, day: date) -> list[int] | None:
        return self.rows.get(day)

    def put(self, day: date, ranked_at: datetime, game_pks: list[int]) -> None:
        assert day not in self.rows
        self.rows[day] = game_pks


def _quote(team: int, bid: float, ask: float) -> Quote:
    return Quote(f"T{team}", team, bid, ask, FIRST_PITCH - timedelta(hours=2))


def _candidate(game_pk: int, home_mid: float, spread: float) -> dict:
    home = _quote(1, home_mid - spread / 2, home_mid + spread / 2)
    away = _quote(2, 1 - home_mid - spread / 2, 1 - home_mid + spread / 2)
    return {"game_pk": game_pk, "home": home, "away": away}


def test_slate_fixed_after_ranking():
    store = MemorySlates()
    starts = [FIRST_PITCH, FIRST_PITCH + timedelta(hours=1)]
    now = FIRST_PITCH - timedelta(minutes=30)
    first = [_candidate(pk, 0.5, 0.02 if pk <= 3 else 0.10) for pk in range(1, 7)]
    second = [_candidate(pk, 0.5, 0.10 if pk <= 3 else 0.02) for pk in range(1, 7)]
    slate = resolve_slate(DAY, now, starts, first, store)
    again = resolve_slate(DAY, now + timedelta(minutes=15), starts, second, store)
    assert sorted(slate) == [1, 2, 3]
    assert again == slate


def test_slate_waits_for_ranking_time():
    store = MemorySlates()
    early = FIRST_PITCH - timedelta(hours=3)
    assert resolve_slate(DAY, early, [FIRST_PITCH], [_candidate(1, 0.5, 0.02)], store) is None
    assert store.get(DAY) is None


def test_lost_market_gets_no_pick_and_no_replacement():
    slate = [1, 2, 3]
    lost = choose_pick(1, None, None, None, 0.6, slate, ready=True, picks_enabled=True)
    outsider = _candidate(4, 0.5, 0.06)
    other = choose_pick(4, outsider["home"], outsider["away"], 0.5, 0.5, slate, ready=True, picks_enabled=True)
    assert lost is None
    assert other is None


def test_slate_ranked_without_quotes_is_not_stored():
    store = MemorySlates()
    now = FIRST_PITCH - timedelta(minutes=30)
    unpriced = [{"game_pk": pk, "home": None, "away": None} for pk in range(1, 7)]
    assert resolve_slate(DAY, now, [FIRST_PITCH], unpriced, store) is None
    assert store.get(DAY) is None
    assert sorted(resolve_slate(DAY, now, [FIRST_PITCH], [_candidate(pk, 0.5, 0.02) for pk in range(1, 4)], store)) == [
        1,
        2,
        3,
    ]
