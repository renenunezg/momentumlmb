"""Forecast and pick policy constants. Every value here is either a recorded
product decision or fitted inside the 2021-2025 development seasons; the 2026
holdout is never used to choose them."""

from datetime import timedelta

# Published win probability: the home-field-shifted sim blended with the Kalshi
# midpoint on the logit scale. 0.5 is a product decision carried from MLB
# (MARKET_ANCHOR_W_MODEL there); do not change it without Rene's approval.
MARKET_ANCHOR_W_MODEL = 0.5

# Chosen 2026-11-19 by walk-forward on 2024-2025 (monthly refits, 1,959 decided games; fits use only
# earlier data). Training window: three seasons beat two, log loss 0.6873 vs 0.6895, difference
# 90% interval [0.0001, 0.0045]; an all-seasons window was not run (Rene, to save time).
SEASONS_BACK = 3
# Walk-forward and holdout refit cadence (production trains daily). 7 and 14 days were not tested.
REFIT_EVERY_DAYS = 30
# Air density moves home runs within a park (0.031 +- 0.009 HR logit per 0.01 kg/m3 on 2021-2025), but
# adding it changed walk-forward log loss by +0.0002 [-0.0007, +0.0012], so it is off.
WEATHER_HR_COEF = 0.0
# Home field on the sim's probability: 0.159 +- 0.046 (sim log loss 0.6873 -> 0.6842).
HOME_FIELD_LOGIT = 0.159
# The sim's coefficient beyond the de-vigged BetExplorer average close, 2024-2025 (1,878 games):
# 0.049 +- 0.143, so the sim adds nothing measurable beyond the market (as in MLB, 0.04 +- 0.30). The
# close's log loss is 0.6713 against the shifted sim's 0.6835. The log-loss-optimal anchor weight is
# 0.049; MARKET_ANCHOR_W_MODEL stays 0.5 (costs 0.0027 log loss on these games) pending Rene's call.
PICK_MODEL_WEIGHT = 0.049

PICK_POLICY = "lmb-picks-v1"
PICK_MIN_EV = -0.0125
PICK_DAILY_MINIMUM = 3
SLATE_LEAD = timedelta(minutes=60)
# Set once by the 2026 holdout rule (Task 12): off if the sim is clearly worse
# than the team-strength baseline.
PICKS_ENABLED = False
