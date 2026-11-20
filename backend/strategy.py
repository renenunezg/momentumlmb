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
# 90% interval [0.0001, 0.0045] (measured on lmb-sim-v2; not re-run after the park and month changes);
# an all-seasons window was not run (Rene, to save time).
SEASONS_BACK = 3
# Walk-forward and holdout refit cadence (production trains daily). 7 and 14 days were not tested.
REFIT_EVERY_DAYS = 30
# Results below are lmb-sim-v4 (per-outcome parks, month effect), walk-forward 2024-2025.
# Air density moves home runs within a park: 0.030 +- 0.009 HR logit per 0.01 kg/m3 thinner than the
# venue median, fitted on 2021-2025. In the walk-forward (each season's coefficient fitted on earlier
# seasons only) it cut total-runs squared error 30.35 -> 30.19, difference 90% interval
# [-0.245, -0.084], and left win log loss unchanged (+0.0005 [-0.0004, +0.0014]), so it is on.
WEATHER_HR_COEF = 0.030
# Home field on the sim's probability: 0.155 +- 0.046 (sim log loss 0.6869 -> 0.6840).
HOME_FIELD_LOGIT = 0.155
# The sim's coefficient beyond the de-vigged BetExplorer average close, 2024-2025 (1,878 games):
# 0.044 +- 0.145, so the sim adds nothing measurable beyond the market (as in MLB, 0.04 +- 0.30). The
# close's log loss is 0.6713 against the shifted sim's 0.6834. The log-loss-optimal anchor weight is
# 0.044; MARKET_ANCHOR_W_MODEL stays 0.5 pending Rene's call.
PICK_MODEL_WEIGHT = 0.044

PICK_POLICY = "lmb-picks-v1"
PICK_MIN_EV = -0.0125
PICK_DAILY_MINIMUM = 3
SLATE_LEAD = timedelta(minutes=60)
# Set once by the 2026 holdout rule, read 2026-11-20 (cache/holdout_2026.json): off if the sim is clearly
# worse than the team-strength baseline. Sim log loss 0.6866 vs team strength 0.6963 on 997 games,
# difference 90% interval [-0.0225, +0.0027], so picks are on.
PICKS_ENABLED = True
