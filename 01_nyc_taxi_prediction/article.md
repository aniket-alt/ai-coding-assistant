# How long will this cab ride take? Estimated from booking-time information only.

*NYC Taxi Trip Duration — project 01 of a sixteen-project data science portfolio built with an AI coding assistant, on real data.*

![NYC Taxi Trip Duration](./screenshots/01_trip_estimator.png)

## The question

When a rider requests a yellow cab in New York City, how long will the trip take?

A duration estimate is only useful if it can be produced *before* the trip. That constraint, not model choice, is what makes this problem interesting.

The success criterion is set at the product level rather than the statistical one: the share of estimates landing within five minutes of the truth, since a rider notices a five-minute miss and does not notice a 0.01 change in $R^2$.

A rider requesting a taxi knows: where they are, where they are going, how many
passengers, and the current time. They do not know how far the meter will run,
what the fare will be, or when they will arrive.

The TLC file records `trip_distance`, `fare_amount` and `tpep_dropoff_datetime`,
but all three are measured *during or after* the trip. Training a
duration model on them produces a spectacular score and a useless product --
the classic form of target leakage, and the one this dataset invites.

This project therefore drops them from the feature set and reconstructs the
only distance signal a booking legitimately has: the historical median distance
for that pickup/dropoff zone pair, computed from the training window alone
(see `zone_pair_statistics` in features.py).

## The data

The January 2024 yellow-cab export holds **2,964,624 trips**. Seven filters reduce it to a modelling frame, each removing a specific, documented failure mode rather than trimming quantiles blindly.

| Rule | Removed | Share | Remaining |
|---|---|---|---|
| pickup timestamp outside January 2024 | 18 | 0.00% | 2,964,606 |
| duration outside [1, 180] minutes | 37,104 | 1.25% | 2,927,502 |
| metered distance outside [0.1, 100] miles | 38,590 | 1.30% | 2,888,912 |
| non-positive fare (voided record) | 31,753 | 1.07% | 2,857,159 |
| passenger count outside [1, 6] | 30,499 | 1.03% | 2,826,660 |
| unknown or out-of-city zone sentinel (264/265) | 26,487 | 0.89% | 2,800,173 |
| implied speed outside [1, 70] mph | 926 | 0.03% | 2,799,247 |
| imputed missing passenger_count / RatecodeID (indicator retained) | 0 | 0.00% | 2,799,247 |

The file is labelled January yet contains a handful of records timestamped outside it — a real quirk of TLC exports, and the kind of thing a generated dataset never reproduces.

## What won

**LightGBM**, at 5.345 on rmse (min), against 7.511 for the best baseline, Zone-pair lookup.

| # | Model | RMSE (min) | MAE (min) | R² | Within 5 min |
|---|---|---|---|---|---|
| 1 | **LightGBM** 🏆 | 5.345 | 3.329 | 0.778 | 81.4% |
| 2 | XGBoost | 5.349 | 3.340 | 0.777 | 81.2% |
| 3 | HistGradientBoosting | 5.542 | 3.450 | 0.761 | 80.1% |
| 4 | Random forest | 5.689 | 3.467 | 0.748 | 80.4% |
| 5 | Ridge regression | 6.392 | 4.086 | 0.682 | 73.3% |
| 6 | Zone-pair lookup *(baseline)* | 7.511 | 4.345 | 0.561 | 74.4% |
| 7 | Global median *(baseline)* | 11.772 | 7.530 | −0.078 | 48.9% |

Ranked by rmse (min) (lower is better). Split: chronological with 1-day embargo; 121,130 evaluation rows. Generated from `artifacts/metrics.json`.

**LightGBM** reaches an RMSE of 5.35 minutes on 121,130 held-out trips drawn from days the model never saw, placing 81.4% of estimates within five minutes.

Improvement over predicting a single global median: **54.6%**. Improvement over a historical zone-pair lookup table: **28.8%**. The second figure is the honest one — it is what the model adds over the obvious non-machine-learning solution.

The static audit reports 0 critical findings. Three runtime assertions executed during this run and would have halted training had temporal ordering, split disjointness, or target-correlation checks failed.

## What I took away

- The fields that would make this model look best are exactly the ones a booking cannot know. Excluding them costs score and buys a model that can run.
- A lookup table of historical route durations is the right baseline to beat. Beating a global median proves very little.
- Target encoding is only safe out-of-fold. Computed in-sample, every row contributes to its own feature.

## How it is built

The backend is FastAPI on port 8001 and the front end is React on port 5101. Most of the interface — the metrics panel, tournament table, CRISP-DM walkthrough, audit panel and charts — is shared across the portfolio and driven entirely by the training run's artifacts, so nothing on screen is typed by hand.

Code: [github.com/aniket-alt/ai-coding-assistant](https://github.com/aniket-alt/ai-coding-assistant/tree/main/01_nyc_taxi_prediction)

---

*Figures in this article are generated from the project's artifacts.*
