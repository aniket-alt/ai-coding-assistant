# Energy Demand Forecasting

*How much electricity will this home use in each hour of tomorrow?*

**Aniket Anil Naik** · Project 12 of the data science full-stack portfolio

## Abstract

**How much electricity will this household use in each hour of tomorrow, and how wide should the uncertainty be?**

We study this question on *Individual Household Electric Power Consumption (UCI)*: 2,075,259 rows as downloaded, used under Creative Commons Attribution 4.0 (CC BY 4.0). Following the CRISP-DM process, we compare 6 approaches, 3 of them baselines, on 25,458 evaluation rows and rank them by pinball loss (kw). Evaluation split: walk-forward, 18 refits every 60 days, day-ahead.

The best approach, **LightGBM quantile**, reaches 0.1428 against 0.1750 for the strongest baseline (Four-week hourly average).

4 of 4 leakage assertions passed while training ran, and static analysis of the source reported 0 critical and 0 warning findings.

**Keywords:** probabilistic day-ahead load forecasting, day-ahead quantile forecasting, minimum lag equal to the horizon, walk-forward with paired bootstrap, measured cost of a leaky lag


## 1. Introduction

*Objective.* Issue a calibrated day-ahead forecast of hourly load.

A day-ahead forecast is issued once, at a fixed moment, for every hour of the following day. That timing defines what the model may know: nothing recorded after the forecast is issued.

The most common way to break that rule is a lag that is too short. The load one hour ago is an excellent predictor of the next hour, and it is simply not available when tomorrow evening is forecast this evening. This project enforces a minimum lag equal to the horizon, verifies it by recomputation, and measures separately how much a leaky model would have flattered the results.

For a single household the honest baselines are strong: tomorrow often looks like today, and like the same day last week. Every model is tested for whether it genuinely beats repeating yesterday.

| Finding | Value | Note |
|---|---|---|
| Horizon | 24 hours ahead |  |
| Primary metric | Pinball loss | P10, P50, P90 |
| Reference | Same hour yesterday |  |

> Phase gate passed: Horizon, minimum lag, loss and reference baseline fixed before modelling.


## 2. Data

*Objective.* Aggregate minute readings to hours without inventing values for gaps.

2,075,259 minute readings from 2006-12-16 to 2010-11-26, of which 25,979 are missing. An hour is kept only if at least 45 of its minutes were recorded, leaving 34,587 hours with **436 missing**; the longest gap lasts 121 hours. Gaps are never interpolated, because a forecast scored against an interpolated value is scored against something that did not happen.

Mean load is 1.09 kW against a median of 0.80 kW, the skew of a household that is usually quiet and occasionally runs several appliances at once. Winter use is **1.95×** summer use.

Correlation with the same hour a day earlier is 0.43, and a week earlier 0.45. A single home is noisy, and no model will explain most of its hourly variation.

| Finding | Value | Note |
|---|---|---|
| Hours | 34,587 |  |
| Missing hours | 436 |  |
| Longest gap | 121 h |  |
| Winter / summer | 1.95× |  |
| Same hour, one day earlier | r = 0.43 |  |

### Provenance

✅ **Real, third-party data**

| Field | Value |
|---|---|
| Dataset | Individual Household Electric Power Consumption (UCI) |
| Source | https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip |
| Licence | Creative Commons Attribution 4.0 (CC BY 4.0) |
| Rows × columns (as downloaded) | 2,075,259 × 7 |
| SHA-256 | `9f84b46ade8a2d8e1286ec4b2b6c2987a45a755c59f263be3b3b3d10dfbda3ff` |
| Hours after aggregation | 34,587 |
| Hours missing | 436 |

> Phase gate passed: 34,587 hours; gaps preserved as missing.


## 3. Method

*Objective.* Build features that exist when the forecast is issued, and prove it.

Every row is a target hour. Its features are lags of 24 to 672 hours, the same hour's four-week average, rolling statistics ending at the forecast origin, and the target hour's calendar, which is known in advance.

Two checks enforce the timing. The shortest lag must be at least the 24-hour horizon. And features for 30 random target hours are recomputed from a series truncated at the forecast origin: **600 values, 0 mismatches**.

Evaluation walks forward from December 2007: 18 refits, each trained on all earlier hours and forecasting the next sixty days.

| Finding | Value | Note |
|---|---|---|
| Features | 20 |  |
| Shortest lag | 24 h |  |
| Values re-verified | 600 |  |
| Walk-forward folds | 18 |  |

> Phase gate passed: Minimum-lag and recomputation checks passed.


## 4. Models

*Objective.* Compare learned forecasters with three seasonal baselines and a fat-tailed interval.

Three baselines repeat the past in different ways: yesterday, the same day last week, and the average of the last four weeks. Each is given an interval from its own training-window errors, so baselines compete on calibration as well as on the median.

Ridge and LightGBM use all features. LightGBM fits three boosters directly on the pinball loss, and its three quantiles are sorted per hour so that P10 never exceeds P90.

| Finding | Value | Note |
|---|---|---|
| Forecasters | 6 |  |
| Winner | LightGBM quantile |  |
| Pinball loss | 0.1428 |  |
| 80% coverage | 77.5% |  |

**Table 1.** Tournament results.

| # | Forecaster | Pinball loss (kW) | Skill vs yesterday | 80% coverage | WAPE |
|---|---|---|---|---|---|
| 1 | **LightGBM quantile** 🏆 | 0.1428 | 33.7% | 77.5% | 41.7% |
| 2 | Ridge + residual quantiles | 0.1616 | 25.0% | 83.9% | 44.7% |
| 3 | Four-week hourly average *(baseline)* | 0.1750 | 18.7% | 84.0% | 47.1% |
| 4 | Four-week average + Student-t | 0.1830 | 15.0% | 87.7% | 47.4% |
| 5 | Same hour last week *(baseline)* | 0.2096 | 2.7% | 83.4% | 55.1% |
| 6 | Same hour yesterday *(baseline)* | 0.2154 | 0.0% | 84.0% | 56.3% |

Ranked by pinball loss (kw) (lower is better). Split: walk-forward, 18 refits every 60 days, day-ahead; 25,458 evaluation rows. Generated from `artifacts/metrics.json`.

> Phase gate passed: 6 forecasters over 18 folds.


## 5. Results

*Objective.* Decide whether the winner genuinely beats repeating the past, and measure the cost of a leak.

**LightGBM quantile** has the lowest pinball loss, 0.1428, a skill of 33.7% over repeating yesterday (0.2154). A paired bootstrap on hourly losses rates that difference as **better**. Its 80% interval covers 77.5% of outcomes, and its median misses by 41.7% of total load (WAPE).

**The leak.** The same LightGBM with one-, two- and three-hour lags added scores a pinball loss of 0.1104, **23% better** than the honest version. None of that improvement exists when the forecast is actually issued a day in advance. It is reported here and kept out of the leaderboard, because it is exactly the result a careless evaluation would have published.

No weather data is available, and winter use is nearly double summer use; temperature would likely be the most valuable missing input.

| Finding | Value | Note |
|---|---|---|
| Skill vs yesterday | 33.7% |  |
| Significance | better |  |
| Leaky model would claim | 23% better |  |
| Critical findings | 0 |  |

> Phase gate passed: No critical findings; leaky variant measured but excluded.

## 6. Threats to validity

### Assertions executed during training

| Assertion | Result | What it measured |
|---|---|---|
| `assert_lags_respect_horizon` | ✅ passed | 8 lags checked against a 24-hour horizon; shortest is 24 hours; 0 shorter than the horizon. |
| `assert_no_lookahead` | ✅ passed | 600 feature values across 30 target hours recomputed from data available at the forecast origin; 0 differed. |
| `assert_temporal_split` | ✅ passed | Last training timestamp 2007-12-16 17:00:00 precedes first test timestamp 2007-12-16 18:00:00 by 0 days 01:00:00; no embargo required. |
| `assert_no_target_leak` | ✅ passed | 20 numeric features correlated against the target; strongest was same_hour_4wk_mean at |r|=0.5275 (threshold 0.995). |

Recorded by the assertion ledger while training ran; generated from `artifacts/metrics.json`.

### Limitations

- No weather data. Heating and lighting follow temperature and daylight, which the model can only infer from the calendar.
- One household over four years; occupants' routines change and the model assumes the recent past resembles the near future.
- Hours with fewer than 45 recorded minutes are dropped, so the longest outages are simply not scored.


## 7. Deployment

*Objective.* Serve tomorrow's hourly band and the recent track record together.

`GET /api/forecast` returns the next 24 hours after the data ends from the final fold's quantile models, beside the last fourteen days of walk-forward forecasts against what actually happened, so the band can be judged by its history.

| Finding | Value | Note |
|---|---|---|
| Served model | LightGBM quantile |  |
| Serving port | 8,012 |  |

> Phase gate passed: Final-fold quantile models persisted; forecasts served from artifacts.

## 8. Conclusion

- A lag shorter than the forecast horizon is the easiest leak to introduce in time-series work, and here it would have flattered the result substantially.
- Averaging the same hour over several weeks is a far stronger baseline than repeating yesterday.
- Unlike daily stock returns, household load is genuinely forecastable a day ahead, and learned models beat the baselines by a significant margin.
- The best point forecaster was not the best-calibrated one; coverage has to be reported beside loss.

## Reproducibility

Trained 2026-09-16T23:48:26Z in 190.0 s at commit `39e3212` with Python 3.11.6, NumPy 2.4.6, pandas 3.0.5 and scikit-learn 1.9.1.

```bash
python ml/train.py
python -m uvicorn main:app --app-dir server --port 8012
npm run dev -w 12_energy_forecasting/client
```

## References

1. Hebrail, G., & Berard, A. (2006). Individual Household Electric Power Consumption [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C58K54
2. Hong, T., & Fan, S. (2016). Probabilistic electric load forecasting: A tutorial review. International Journal of Forecasting, 32(3), 914–938.
3. Koenker, R., & Bassett, G. (1978). Regression quantiles. Econometrica, 46(1), 33–50.
4. Hyndman, R. J., & Athanasopoulos, G. (2021). Forecasting: Principles and Practice (3rd ed.). OTexts.
5. Ke, G. et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. Advances in Neural Information Processing Systems 30.

---

*Generated from this project's artifacts by `scripts/generate_docs.py`.*
