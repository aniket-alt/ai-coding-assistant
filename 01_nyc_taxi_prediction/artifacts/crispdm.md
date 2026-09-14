# CRISP-DM Report — NYC Taxi Trip Duration Prediction

**Business question.** When a rider requests a yellow cab in New York City, how long will the trip take?


## 1. Business Understanding

*Objective:* Give riders a trustworthy arrival estimate at the moment they request a cab, using only what is knowable at that moment.


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



| Finding | Value | Detail |
|---|---|---|
| Primary metric | `RMSE (minutes)` | lower is better |
| Product metric | `% within 5 min` | what a rider actually feels |
| Hard constraint | `booking-time features only` | no meter, no arrival time |


*Code:* `ml/data.py:BOOKING_TIME_NOTE`, `ml/train.py:evaluate`


> **Phase gate: PASSED.** Target, constraint and success metric fixed before any data was loaded.


## 2. Data Understanding

*Objective:* Establish what the raw TLC export actually contains, and what must be discarded.


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


| Finding | Value | Detail |
|---|---|---|
| Raw trips | `2,964,624` |  |
| After cleaning | `600,000` |  |
| Retained | `20.2%` |  |
| Median duration | `11.7 min` |  |


*Code:* `ml/data.py:clean`


> **Phase gate: PASSED.** 20.2% of raw records retained with a per-rule audit trail.


## 3. Data Preparation

*Objective:* Build features that exist at booking time, fitted on the training window alone.


The split is chronological with a **one-day embargo**: training ends 25 January, testing begins 26 January, leaving a 1 days 00:00:06 gap so no same-day correlation bleeds across the boundary. A random split would be easier and would silently let the model learn from the future.

Because the meter fields are unavailable at booking, distance is reconstructed as the historical median for the origin/destination pair, learned from training rows only. Two target encodings (zone pair, and origin zone × hour) are computed **out-of-fold**: within training, each row's encoding comes from folds excluding it, so no row informs its own feature.

$$\text{enc}(g) = \frac{\sum_{i \in g} y_i + \alpha\bar{y}}{n_g + \alpha}, \quad \alpha = 20$$

The shrinkage term $\alpha$ pulls rare zone pairs toward the global mean, so a pair seen twice does not get a confident encoding.


| Finding | Value | Detail |
|---|---|---|
| Train rows | `456,487` |  |
| Test rows | `121,130` |  |
| Features | `25` |  |
| Embargo gap | `1 days 00:00:06` |  |
| Excluded post-trip columns | `13` |  |


*Code:* `ml/features.py:OutOfFoldTargetEncoder`, `ml/train.py:TRAIN_END`


> **Phase gate: PASSED.** Temporal ordering, split disjointness and target-correlation checks all executed as runtime assertions during this run.


## 4. Modeling

*Objective:* Compare model families on identical features and an identical split.


Seven approaches were evaluated, from a single-number baseline up to gradient-boosted ensembles. Every model saw the same features and the same chronological split, so the ranking reflects the estimator rather than the preprocessing.

Two baselines are included deliberately. The global median shows what zero modelling achieves; the zone-pair lookup shows what a plain historical average achieves. A learned model has to beat the second one to justify existing at all.


| Finding | Value | Detail |
|---|---|---|
| Models evaluated | `7` |  |
| Winner | `LightGBM` |  |
| RMSE | `5.345 min` |  |
| Within 5 min | `81.4%` |  |


*Code:* `ml/train.py:run_model`


> **Phase gate: PASSED.** 7 models trained; winner LightGBM.


## 5. Evaluation

*Objective:* Decide whether the winner is good enough, and whether it was measured honestly.


**LightGBM** reaches an RMSE of 5.35 minutes on 121,130 held-out trips drawn from days the model never saw, placing 81.4% of estimates within five minutes.

Improvement over predicting a single global median: **54.6%**. Improvement over a historical zone-pair lookup table: **28.8%**. The second figure is the honest one — it is what the model adds over the obvious non-machine-learning solution.

The static audit reports 0 critical findings. Three runtime assertions executed during this run and would have halted training had temporal ordering, split disjointness, or target-correlation checks failed.


| Finding | Value | Detail |
|---|---|---|
| Best RMSE | `5.345 min` |  |
| Best MAE | `3.329 min` |  |
| R² | `0.7777` |  |
| Lift vs lookup table | `28.8%` |  |
| Critical audit findings | `0` |  |


*Code:* `ml/train.py:evaluate`, `platform/dsfs/leakage.py`


> **Phase gate: PASSED.** No critical leakage findings; metrics measured on a future window.


## 6. Deployment

*Objective:* Serve the estimate behind an API the front end can call.


The winning model is persisted with its fitted encoders so that inference reproduces training exactly — the zone statistics and target encodings travel with the model rather than being recomputed at request time.

`POST /api/predict` accepts a booking (origin zone, destination zone, timestamp, passengers) and returns an estimate with a prediction interval. The same server publishes this report, the leaderboard, the dataset receipt and the audit at `/api/crispdm`, `/api/metrics`, `/api/provenance` and `/api/audit`.


| Finding | Value | Detail |
|---|---|---|
| Artefact | `model.joblib` |  |
| Serving port | `8,001` |  |
| Endpoints | `10` |  |


*Code:* `server/main.py`, `server/inference.py`


> **Phase gate: PASSED.** Model, encoders and feature order persisted together.
