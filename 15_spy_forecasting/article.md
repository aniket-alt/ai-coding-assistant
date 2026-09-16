# How much of tomorrow's S&P 500 move is actually predictable?

*SPY Probabilistic Forecasting — project 15 of a sixteen-project data science portfolio built with an AI coding assistant, on real data.*

![SPY Probabilistic Forecasting](./screenshots/01_forecast_fan.png)

## The question

How much can tomorrow's S&P 500 move actually be predicted, and how wide should the honest uncertainty band be?

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

## The data

2,931 trading days from 2015-01-05 to 2026-08-31. SPY went from \$166.62 to \$767.05, a total return of +360% — +13.0% annualised against 17.6% annualised volatility.

**Buy-and-hold Sharpe over this window is 0.74.** That is the number any strategy has to beat, and it is the reference point that makes reported Sharpe ratios of 2 or 3 on daily equity data implausible on their face.

Returns are not normal. Excess kurtosis is **14.5** against 0 for a Gaussian, and the window contains a -11.6% day (2020-03-16) and a +10.0% day (2025-04-09). Any interval derived from a normal assumption will be too narrow in exactly the moments that matter. Maximum drawdown was -34%.

54.7% of days are positive — the thin edge that makes buy-and-hold work, and roughly the hit rate a directional model must exceed to add anything.

## What won

A baseline. **Historical drift** scored best on pinball loss (0.002660), ahead of every learned model in the tournament.

| # | Model | Pinball loss | 80% coverage | Hit rate | R² on returns | Sharpe |
|---|---|---|---|---|---|---|
| 1 | **Historical drift** 🏆 *(baseline)* | 0.002660 | 78.7% | 55.3% | −0.0005 | 0.75 |
| 2 | Random walk (zero) *(baseline)* | 0.002667 | 78.4% | 50.0% | −0.0022 | 0.00 |
| 3 | LightGBM quantile | 0.002697 | 72.8% | 51.9% | −0.0236 | 0.60 |
| 4 | EWMA vol + Student-t | 0.002698 | 86.4% | 50.0% | −0.0022 | 0.00 |
| 5 | Empirical quantiles *(baseline)* | 0.002804 | 78.5% | 54.7% | −0.0038 | 0.67 |
| 6 | Ridge + residual quantiles | 0.002895 | 73.9% | 51.7% | −0.0522 | 0.60 |

Ranked by pinball loss (lower is better). Split: walk-forward, 32 quarterly refits, purged; 1,981 evaluation rows. Generated from `artifacts/metrics.json`.

**Historical drift** achieves the lowest pinball loss (0.002660) across 32 walk-forward folds, with an 80% interval covering 78.7% of outcomes.

The directional hit rate is 55.3%, against 54.7% of days being positive — so a model that simply said *up* every day would score about the same. $R^2$ on returns is -0.0005.

**That is the finding, and it is not a disappointing one.** Daily equity returns are close to unpredictable in their first moment; the honest value is in the second moment, where volatility clusters and intervals can be calibrated. Models that concede the direction and concentrate on the width do well here for that reason.

Traded naively — long whenever the median forecast is positive, 2 bps round trip — the winner produces a Sharpe of **0.75** against **0.75** for simply holding SPY over the same period. Any published daily-equity strategy claiming a Sharpe above 2 should be read with this comparison in mind.

## What I took away

- Forecasting prices instead of returns produces an impressive-looking and worthless result.
- A random walk is a strong baseline for daily equity returns. Beating it is the whole question.
- A reported winner means little without a significance test, because the gaps between good forecasters are tiny.
- The predictable part of daily returns is their spread, not their direction.

## How it is built

The backend is FastAPI on port 8015 and the front end is React on port 5115. Most of the interface — the metrics panel, tournament table, CRISP-DM walkthrough, audit panel and charts — is shared across the portfolio and driven entirely by the training run's artifacts, so nothing on screen is typed by hand.

Code: [github.com/aniket-alt/ai-coding-assistant](https://github.com/aniket-alt/ai-coding-assistant/tree/main/15_spy_forecasting)

---

*Figures in this article are generated from the project's artifacts.*
