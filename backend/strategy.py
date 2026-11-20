"""Forecast and pick policy constants. Every value here is either a recorded
product decision or fitted inside the 2021-2025 development seasons; the 2026
holdout is never used to choose them."""

from datetime import timedelta

# Published win probability: the home-field-shifted sim blended with the Kalshi
# midpoint on the logit scale. 0.5 is a product decision carried from MLB
# (MARKET_ANCHOR_W_MODEL there); do not change it without Rene's approval.
MARKET_ANCHOR_W_MODEL = 0.5

# Placeholders until the walk-forward fits in Tasks 6 and 7 replace them.
HOME_FIELD_LOGIT = 0.0
PICK_MODEL_WEIGHT = 0.07
SEASONS_BACK = 3
REFIT_EVERY_DAYS = 30
WEATHER_HR_COEF = 0.0

PICK_POLICY = "lmb-picks-v1"
PICK_MIN_EV = -0.0125
PICK_DAILY_MINIMUM = 3
SLATE_LEAD = timedelta(minutes=60)
# Set once by the 2026 holdout rule (Task 12): off if the sim is clearly worse
# than the team-strength baseline.
PICKS_ENABLED = False
