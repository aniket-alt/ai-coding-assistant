# How much electricity will this home use in each hour of tomorrow?

*Energy Demand Forecasting — project 12 of a sixteen-project data science portfolio built with an AI coding assistant, on real data.*

![Energy Demand Forecasting](./screenshots/01_day_ahead_forecast.png)

## The question

How much electricity will this household use in each hour of tomorrow, and how wide should the uncertainty be?

A day-ahead forecast is issued once, at a fixed moment, for every hour of the following day. That timing defines what the model may know: nothing recorded after the forecast is issued.

The most common way to break that rule is a lag that is too short. The load one hour ago is an excellent predictor of the next hour, and it is simply not available when tomorrow evening is forecast this evening. This project enforces a minimum lag equal to the horizon, verifies it by recomputation, and measures separately how much a leaky model would have flattered the results.

For a single household the honest baselines are strong: tomorrow often looks like today, and like the same day last week. Every model is tested for whether it genuinely beats repeating yesterday.

## The data

2,075,259 minute readings from 2006-12-16 to 2010-11-26, of which 25,979 are missing. An hour is kept only if at least 45 of its minutes were recorded, leaving 34,587 hours with **436 missing**; the longest gap lasts 121 hours. Gaps are never interpolated, because a forecast scored against an interpolated value is scored against something that did not happen.

Mean load is 1.09 kW against a median of 0.80 kW, the skew of a household that is usually quiet and occasionally runs several appliances at once. Winter use is **1.95×** summer use.

Correlation with the same hour a day earlier is 0.43, and a week earlier 0.45. A single home is noisy, and no model will explain most of its hourly variation.

## What won

**LightGBM quantile**, at 0.1428 on pinball loss (kw), against 0.1750 for the best baseline, Four-week hourly average.

| # | Forecaster | Pinball loss (kW) | Skill vs yesterday | 80% coverage | WAPE |
|---|---|---|---|---|---|
| 1 | **LightGBM quantile** 🏆 | 0.1428 | 33.7% | 77.5% | 41.7% |
| 2 | Ridge + residual quantiles | 0.1616 | 25.0% | 83.9% | 44.7% |
| 3 | Four-week hourly average *(baseline)* | 0.1750 | 18.7% | 84.0% | 47.1% |
| 4 | Four-week average + Student-t | 0.1830 | 15.0% | 87.7% | 47.4% |
| 5 | Same hour last week *(baseline)* | 0.2096 | 2.7% | 83.4% | 55.1% |
| 6 | Same hour yesterday *(baseline)* | 0.2154 | 0.0% | 84.0% | 56.3% |

Ranked by pinball loss (kw) (lower is better). Split: walk-forward, 18 refits every 60 days, day-ahead; 25,458 evaluation rows. Generated from `artifacts/metrics.json`.

**LightGBM quantile** has the lowest pinball loss, 0.1428, a skill of 33.7% over repeating yesterday (0.2154). A paired bootstrap on hourly losses rates that difference as **better**. Its 80% interval covers 77.5% of outcomes, and its median misses by 41.7% of total load (WAPE).

**The leak.** The same LightGBM with one-, two- and three-hour lags added scores a pinball loss of 0.1104, **23% better** than the honest version. None of that improvement exists when the forecast is actually issued a day in advance. It is reported here and kept out of the leaderboard, because it is exactly the result a careless evaluation would have published.

No weather data is available, and winter use is nearly double summer use; temperature would likely be the most valuable missing input.

## What I took away

- A lag shorter than the forecast horizon is the easiest leak to introduce in time-series work, and here it would have flattered the result substantially.
- Averaging the same hour over several weeks is a far stronger baseline than repeating yesterday.
- Unlike daily stock returns, household load is genuinely forecastable a day ahead, and learned models beat the baselines by a significant margin.
- The best point forecaster was not the best-calibrated one; coverage has to be reported beside loss.

## How it is built

The backend is FastAPI on port 8012 and the front end is React on port 5112. Most of the interface — the metrics panel, tournament table, CRISP-DM walkthrough, audit panel and charts — is shared across the portfolio and driven entirely by the training run's artifacts, so nothing on screen is typed by hand.

Code: [github.com/aniket-alt/ai-coding-assistant](https://github.com/aniket-alt/ai-coding-assistant/tree/main/12_energy_forecasting)

---

*Figures in this article are generated from the project's artifacts.*
