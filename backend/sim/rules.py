"""Game rules the simulator needs, per game.

Measured from 2021-2026 LMB play-by-play (cached feeds):
- Extra innings: no automatic runner. Of 449 extra-inning games, one (2026)
  carries a `runner_placed` event; every other extra half-inning starts empty.
- Designated hitter: universal. Pitchers take 0.004% to 0.034% of
  regular-season plate appearances a season (position players pitching).
- Regulation length is the game's `scheduledInnings` (seven-inning games are
  most doubleheaders since 2023 and most Tuesday and Wednesday games in 2022)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Rules:
    innings: int
    extra_runner: bool


def game_rules(season: int, game_type: str, scheduled_innings: int) -> Rules:
    if scheduled_innings not in (7, 9):
        raise ValueError(f"Unexpected scheduled innings {scheduled_innings}")
    return Rules(innings=scheduled_innings, extra_runner=False)
