# CRISP-DM Report — SPY Probabilistic Return Forecasting

**Business question.** How much can tomorrow's S&P 500 move actually be predicted, and how wide should the honest uncertainty band be?


## 1. Business Understanding

*Objective:* Produce a calibrated distribution for the next day's return, and establish honestly whether it beats doing nothing.


The useful output of a market model is not a number but a **distribution**. A point forecast of +0.03% is worthless without knowing whether the plausible range around it is ±0.5% or ±5%; the range is what determines position size and risk.

So this project forecasts the 10th, 50th and 90th percentiles and is scored on **pinball loss**, with interval **calibration** treated as a first-class result: an 80% band that contains 80% of outcomes is useful, and one that contains 55% is dangerous.

Forecasting a price level is the easiest way to produce an impressive-looking
and entirely worthless result.

Tomorrow's SPY close is within a fraction of a percent of today's. A model that
simply repeats today's price scores an R-squared above 0.999 against the price
series -- while containing no information whatsoever about what will happen. A
great deal of published "stock prediction" is exactly this, plotted as a
forecast line lying convincingly on top of the actual line.

This project forecasts the daily log return instead:

    r_t = ln(P_t / P_{t-1})

which is close to stationary and has no such free lunch. The consequence is
uncomfortable but correct: R-squared on returns will sit near zero, and the
random-walk baseline -- predicting exactly zero -- is genuinely hard to beat.
That is what the honest version of this problem looks like.



| Finding | Value | Detail |
|---|---|---|
| Primary metric | `Pinball loss` | lower is better |
| Calibration target | `80% interval covers 80%` |  |
| Horizon | `t+1 trading day` |  |


*Code:* `ml/data.py:WHY_RETURNS_NOT_PRICES`, `ml/train.py:pinball`


> **Phase gate: PASSED.** Target, loss function and calibration criterion fixed before modelling.


## 2. Data Understanding

*Objective:* Characterise the return series before assuming anything about it.


2,931 trading days from 2015-01-05 to 2026-08-31. SPY went from \$166.62 to \$767.05, a total return of +360% — +13.0% annualised against 17.6% annualised volatility.

**Buy-and-hold Sharpe over this window is 0.74.** That is the number any strategy has to beat, and it is the reference point that makes reported Sharpe ratios of 2 or 3 on daily equity data implausible on their face.

Returns are not normal. Excess kurtosis is **14.5** against 0 for a Gaussian, and the window contains a -11.6% day (2020-03-16) and a +10.0% day (2025-04-09). Any interval derived from a normal assumption will be too narrow in exactly the moments that matter. Maximum drawdown was -34%.

54.7% of days are positive — the thin edge that makes buy-and-hold work, and roughly the hit rate a directional model must exceed to add anything.


| Finding | Value | Detail |
|---|---|---|
| Trading days | `2,931` |  |
| Buy-hold Sharpe | `0.74` | the bar to beat |
| Annualised vol | `17.6%` |  |
| Excess kurtosis | `14.5` | normal = 0 |
| Max drawdown | `-34%` |  |
| Up days | `54.7%` |  |


*Code:* `ml/data.py:summarise`


> **Phase gate: PASSED.** Distributional properties established; normality rejected before modelling.


## 3. Data Preparation

*Objective:* Build features that use only the past, and prove it rather than assert it.


26 features: lagged returns, realised volatility over four windows, the Parkinson range estimator, momentum and trend, the VIX and its relation to realised volatility (the variance risk premium), the 10-year yield, relative volume, and calendar effects.

Every one is computed from data available at the close of day $t$, and the target is the return over $[t, t+h]$. That rule is easy to break by accident — a centred rolling window, a `shift()` in the wrong direction — so it is **verified empirically**: the feature block is recomputed on truncated history, as if standing at each of 30 randomly chosen dates with nothing after it, and every value must match what the full-history computation produced.

That check compared **780 values** and found 0 discrepancies. A feature that peeks ahead cannot survive it.

Evaluation is walk-forward across **32 quarterly refits** rather than one split, and each refit **purges** the last 0 training rows whose targets overlap the test block.


| Finding | Value | Detail |
|---|---|---|
| Features | `26` |  |
| Walk-forward folds | `32` |  |
| Burn-in | `750 days` |  |
| Values re-verified | `780` |  |
| Lookahead mismatches | `0` |  |


*Code:* `ml/features.py:assert_no_lookahead`, `ml/train.py:walk_forward`


> **Phase gate: PASSED.** Lookahead check passed over 780 recomputed values.


## 4. Modeling

*Objective:* Compare feature-driven models against baselines that assume returns are unpredictable.


The baselines here are not straw men. A random walk with a volatility-derived interval is close to the best available answer for daily equity returns, and beating it is the entire question.

Two families are represented: models that try to predict the **direction** (Ridge, LightGBM quantile) and models that concede direction is unpredictable and concentrate on getting the **width** right (random walk, EWMA with Student-$t$ quantiles). Given excess kurtosis of 14, the second family is more defensible than it first sounds.


| Finding | Value | Detail |
|---|---|---|
| Models | `6` |  |
| Winner | `Historical drift` |  |
| Pinball loss | `0.002660` |  |
| 80% coverage | `78.7%` |  |
| Hit rate | `55.3%` |  |


*Code:* `ml/train.py:walk_forward`, `ml/train.py:evaluate`


> **Phase gate: PASSED.** 6 models across 32 walk-forward folds.


## 5. Evaluation

*Objective:* State plainly how much signal there is, and resist the temptation to overclaim.


**Historical drift** achieves the lowest pinball loss (0.002660) across 32 walk-forward folds, with an 80% interval covering 78.7% of outcomes.

The directional hit rate is 55.3%, against 54.7% of days being positive — so a model that simply said *up* every day would score about the same. $R^2$ on returns is -0.0005.

**That is the finding, and it is not a disappointing one.** Daily equity returns are close to unpredictable in their first moment; the honest value is in the second moment, where volatility clusters and intervals can be calibrated. Models that concede the direction and concentrate on the width do well here for that reason.

Traded naively — long whenever the median forecast is positive, 2 bps round trip — the winner produces a Sharpe of **0.75** against **0.75** for simply holding SPY over the same period. Any published daily-equity strategy claiming a Sharpe above 2 should be read with this comparison in mind.


| Finding | Value | Detail |
|---|---|---|
| Pinball loss | `0.002660` |  |
| 80% coverage | `78.7%` | target 80% |
| Hit rate | `55.3%` |  |
| R² on returns | `-0.0005` |  |
| Strategy Sharpe | `0.75` |  |
| Buy-and-hold Sharpe | `0.75` |  |


*Code:* `ml/train.py:backtest`, `ml/train.py:evaluate`


> **Phase gate: PASSED.** No critical findings; every forecast is out-of-sample by construction.


## 6. Deployment

*Objective:* Serve a distribution, not a number.


`GET /api/forecast` returns the next-day quantile band from each model alongside the last close, so the band can be read in price terms. `GET /api/backtest` returns the walk-forward equity curve against buy-and-hold — displayed together deliberately, because a strategy curve shown alone is the oldest trick in quantitative marketing.


| Finding | Value | Detail |
|---|---|---|
| Served model | `Historical drift` |  |
| Serving port | `8,015` |  |
| Endpoints | `11` |  |


*Code:* `server/main.py`, `server/inference.py`


> **Phase gate: PASSED.** Forecasts, calibration and the buy-and-hold comparison all served.
