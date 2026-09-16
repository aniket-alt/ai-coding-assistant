# CRISP-DM Report — Energy Demand Forecasting

**Business question.** How much electricity will this household use in each hour of tomorrow, and how wide should the uncertainty be?


## 1. Business Understanding

*Objective:* Issue a calibrated day-ahead forecast of hourly load.


A day-ahead forecast is issued once, at a fixed moment, for every hour of the following day. That timing defines what the model may know: nothing recorded after the forecast is issued.

The most common way to break that rule is a lag that is too short. The load one hour ago is an excellent predictor of the next hour, and it is simply not available when tomorrow evening is forecast this evening. This project enforces a minimum lag equal to the horizon, verifies it by recomputation, and measures separately how much a leaky model would have flattered the results.

For a single household the honest baselines are strong: tomorrow often looks like today, and like the same day last week. Every model is tested for whether it genuinely beats repeating yesterday.


| Finding | Value | Detail |
|---|---|---|
| Horizon | `24 hours ahead` |  |
| Primary metric | `Pinball loss` | P10, P50, P90 |
| Reference | `Same hour yesterday` |  |


*Code:* `ml/features.py:LAGS`, `ml/train.py:walk_forward`


> **Phase gate: PASSED.** Horizon, minimum lag, loss and reference baseline fixed before modelling.


## 2. Data Understanding

*Objective:* Aggregate minute readings to hours without inventing values for gaps.


2,075,259 minute readings from 2006-12-16 to 2010-11-26, of which 25,979 are missing. An hour is kept only if at least 45 of its minutes were recorded, leaving 34,587 hours with **436 missing**; the longest gap lasts 121 hours. Gaps are never interpolated, because a forecast scored against an interpolated value is scored against something that did not happen.

Mean load is 1.09 kW against a median of 0.80 kW, the skew of a household that is usually quiet and occasionally runs several appliances at once. Winter use is **1.95×** summer use.

Correlation with the same hour a day earlier is 0.43, and a week earlier 0.45. A single home is noisy, and no model will explain most of its hourly variation.


| Finding | Value | Detail |
|---|---|---|
| Hours | `34,587` |  |
| Missing hours | `436` |  |
| Longest gap | `121 h` |  |
| Winter / summer | `1.95×` |  |
| Same hour, one day earlier | `r = 0.43` |  |


*Code:* `ml/data.py:hourly`


> **Phase gate: PASSED.** 34,587 hours; gaps preserved as missing.


## 3. Data Preparation

*Objective:* Build features that exist when the forecast is issued, and prove it.


Every row is a target hour. Its features are lags of 24 to 672 hours, the same hour's four-week average, rolling statistics ending at the forecast origin, and the target hour's calendar, which is known in advance.

Two checks enforce the timing. The shortest lag must be at least the 24-hour horizon. And features for 30 random target hours are recomputed from a series truncated at the forecast origin: **600 values, 0 mismatches**.

Evaluation walks forward from December 2007: 18 refits, each trained on all earlier hours and forecasting the next sixty days.


| Finding | Value | Detail |
|---|---|---|
| Features | `20` |  |
| Shortest lag | `24 h` |  |
| Values re-verified | `600` |  |
| Walk-forward folds | `18` |  |


*Code:* `ml/features.py:build`, `ml/features.py:assert_no_lookahead`


> **Phase gate: PASSED.** Minimum-lag and recomputation checks passed.


## 4. Modeling

*Objective:* Compare learned forecasters with three seasonal baselines and a fat-tailed interval.


Three baselines repeat the past in different ways: yesterday, the same day last week, and the average of the last four weeks. Each is given an interval from its own training-window errors, so baselines compete on calibration as well as on the median.

Ridge and LightGBM use all features. LightGBM fits three boosters directly on the pinball loss, and its three quantiles are sorted per hour so that P10 never exceeds P90.


| Finding | Value | Detail |
|---|---|---|
| Forecasters | `6` |  |
| Winner | `LightGBM quantile` |  |
| Pinball loss | `0.1428` |  |
| 80% coverage | `77.5%` |  |


*Code:* `ml/train.py:enter`, `ml/train.py:residual_band`


> **Phase gate: PASSED.** 6 forecasters over 18 folds.


## 5. Evaluation

*Objective:* Decide whether the winner genuinely beats repeating the past, and measure the cost of a leak.


**LightGBM quantile** has the lowest pinball loss, 0.1428, a skill of 33.7% over repeating yesterday (0.2154). A paired bootstrap on hourly losses rates that difference as **better**. Its 80% interval covers 77.5% of outcomes, and its median misses by 41.7% of total load (WAPE).

**The leak.** The same LightGBM with one-, two- and three-hour lags added scores a pinball loss of 0.1104, **23% better** than the honest version. None of that improvement exists when the forecast is actually issued a day in advance. It is reported here and kept out of the leaderboard, because it is exactly the result a careless evaluation would have published.

No weather data is available, and winter use is nearly double summer use; temperature would likely be the most valuable missing input.


| Finding | Value | Detail |
|---|---|---|
| Skill vs yesterday | `33.7%` |  |
| Significance | `better` |  |
| Leaky model would claim | `23% better` |  |
| Critical findings | `0` |  |


*Code:* `ml/train.py:main`


> **Phase gate: PASSED.** No critical findings; leaky variant measured but excluded.


## 6. Deployment

*Objective:* Serve tomorrow's hourly band and the recent track record together.


`GET /api/forecast` returns the next 24 hours after the data ends from the final fold's quantile models, beside the last fourteen days of walk-forward forecasts against what actually happened, so the band can be judged by its history.


| Finding | Value | Detail |
|---|---|---|
| Served model | `LightGBM quantile` |  |
| Serving port | `8,012` |  |


*Code:* `server/main.py`


> **Phase gate: PASSED.** Final-fold quantile models persisted; forecasts served from artifacts.
