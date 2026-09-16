# NYC Taxi Trip Duration

*How long will this cab ride take? Estimated from booking-time information only.*

**Aniket Anil Naik** · Project 01 of the data science full-stack portfolio

## Abstract

**When a rider requests a yellow cab in New York City, how long will the trip take?**

We study this question on *NYC TLC Yellow Taxi Trip Records — January 2024*: 2,964,624 rows as downloaded, used under NYC Taxi & Limousine Commission open data (public domain). Following the CRISP-DM process, we compare 7 approaches, 2 of them baselines, on 121,130 evaluation rows and rank them by rmse (min). Evaluation split: chronological with 1-day embargo.

The best approach, **LightGBM**, reaches 5.345 against 7.511 for the strongest baseline (Zone-pair lookup).

3 of 3 leakage assertions passed while training ran, and static analysis of the source reported 0 critical and 0 warning findings.

**Keywords:** regression, chronological split with embargo, out-of-fold target encoding, gradient boosting tournament, leakage assertions


## 1. Introduction

*Objective.* Give riders a trustworthy arrival estimate at the moment they request a cab, using only what is knowable at that moment.

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

| Finding | Value | Note |
|---|---|---|
| Primary metric | RMSE (minutes) | lower is better |
| Product metric | % within 5 min | what a rider actually feels |
| Hard constraint | booking-time features only | no meter, no arrival time |

> Phase gate passed: Target, constraint and success metric fixed before any data was loaded.


## 2. Data

*Objective.* Establish what the raw TLC export actually contains, and what must be discarded.

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

| Finding | Value | Note |
|---|---|---|
| Raw trips | 2,964,624 |  |
| After cleaning | 600,000 |  |
| Retained | 20.2% |  |
| Median duration | 11.7 min |  |

### Provenance

✅ **Real, third-party data**

| Field | Value |
|---|---|
| Dataset | NYC TLC Yellow Taxi Trip Records — January 2024 |
| Source | https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet |
| Licence | NYC Taxi & Limousine Commission open data (public domain) |
| Rows × columns (as downloaded) | 2,964,624 × 19 |
| SHA-256 | `c4d59da7bbc8abaeeeb1727947ee93d9891a71acb42854bd80db1571b2030510` |
| Rows after cleaning | 2,799,247 |
| Rows modelled | 600,000 |
| Sampling | systematic, preserving chronology |

> Phase gate passed: 20.2% of raw records retained with a per-rule audit trail.


## 3. Method

*Objective.* Build features that exist at booking time, fitted on the training window alone.

The split is chronological with a **one-day embargo**: training ends 25 January, testing begins 26 January, leaving a 1 days 00:00:06 gap so no same-day correlation bleeds across the boundary. A random split would be easier and would silently let the model learn from the future.

Because the meter fields are unavailable at booking, distance is reconstructed as the historical median for the origin/destination pair, learned from training rows only. Two target encodings (zone pair, and origin zone × hour) are computed **out-of-fold**: within training, each row's encoding comes from folds excluding it, so no row informs its own feature.

$$\text{enc}(g) = \frac{\sum_{i \in g} y_i + \alpha\bar{y}}{n_g + \alpha}, \quad \alpha = 20$$

The shrinkage term $\alpha$ pulls rare zone pairs toward the global mean, so a pair seen twice does not get a confident encoding.

| Finding | Value | Note |
|---|---|---|
| Train rows | 456,487 |  |
| Test rows | 121,130 |  |
| Features | 25 |  |
| Embargo gap | 1 days 00:00:06 |  |
| Excluded post-trip columns | 13 |  |

> Phase gate passed: Temporal ordering, split disjointness and target-correlation checks all executed as runtime assertions during this run.


## 4. Models

*Objective.* Compare model families on identical features and an identical split.

Seven approaches were evaluated, from a single-number baseline up to gradient-boosted ensembles. Every model saw the same features and the same chronological split, so the ranking reflects the estimator rather than the preprocessing.

Two baselines are included deliberately. The global median shows what zero modelling achieves; the zone-pair lookup shows what a plain historical average achieves. A learned model has to beat the second one to justify existing at all.

| Finding | Value | Note |
|---|---|---|
| Models evaluated | 7 |  |
| Winner | LightGBM |  |
| RMSE | 5.345 min |  |
| Within 5 min | 81.4% |  |

**Table 1.** Tournament results.

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

> Phase gate passed: 7 models trained; winner LightGBM.


## 5. Results

*Objective.* Decide whether the winner is good enough, and whether it was measured honestly.

**LightGBM** reaches an RMSE of 5.35 minutes on 121,130 held-out trips drawn from days the model never saw, placing 81.4% of estimates within five minutes.

Improvement over predicting a single global median: **54.6%**. Improvement over a historical zone-pair lookup table: **28.8%**. The second figure is the honest one — it is what the model adds over the obvious non-machine-learning solution.

The static audit reports 0 critical findings. Three runtime assertions executed during this run and would have halted training had temporal ordering, split disjointness, or target-correlation checks failed.

| Finding | Value | Note |
|---|---|---|
| Best RMSE | 5.345 min |  |
| Best MAE | 3.329 min |  |
| R² | 0.7777 |  |
| Lift vs lookup table | 28.8% |  |
| Critical audit findings | 0 |  |

> Phase gate passed: No critical leakage findings; metrics measured on a future window.

## 6. Threats to validity

### Assertions executed during training

| Assertion | Result | What it measured |
|---|---|---|
| `assert_temporal_split` | ✅ passed | Last training timestamp 2024-01-24 23:59:55 precedes first test timestamp 2024-01-26 00:00:01 by 1 days 00:00:06; embargo of 12h required. |
| `assert_disjoint` | ✅ passed | 456,487 training and 121,130 test row indices compared; 0 appear in both. |
| `assert_no_target_leak` | ✅ passed | 22 numeric features correlated against the target; strongest was zp_median_distance at |r|=0.7821 (threshold 0.995). |

Recorded by the assertion ledger while training ran; generated from `artifacts/metrics.json`.

### Limitations

- One month of data. Seasonality across the year (summer traffic, holidays) is not represented.
- Distance is reconstructed per zone pair, so trips inside a large zone share one estimate of distance.
- The interval assumes roughly normal residuals; long delays such as accidents are heavier-tailed than that.


## 7. Deployment

*Objective.* Serve the estimate behind an API the front end can call.

The winning model is persisted with its fitted encoders so that inference reproduces training exactly — the zone statistics and target encodings travel with the model rather than being recomputed at request time.

`POST /api/predict` accepts a booking (origin zone, destination zone, timestamp, passengers) and returns an estimate with a prediction interval. The same server publishes this report, the leaderboard, the dataset receipt and the audit at `/api/crispdm`, `/api/metrics`, `/api/provenance` and `/api/audit`.

| Finding | Value | Note |
|---|---|---|
| Artefact | model.joblib |  |
| Serving port | 8,001 |  |
| Endpoints | 10 |  |

> Phase gate passed: Model, encoders and feature order persisted together.

## 8. Conclusion

- The fields that would make this model look best are exactly the ones a booking cannot know. Excluding them costs score and buys a model that can run.
- A lookup table of historical route durations is the right baseline to beat. Beating a global median proves very little.
- Target encoding is only safe out-of-fold. Computed in-sample, every row contributes to its own feature.

## Reproducibility

Trained 2026-09-14T06:46:26Z in 168.5 s at commit `e7b6737` with Python 3.11.6, NumPy 2.4.6, pandas 3.0.5 and scikit-learn 1.9.1.

```bash
python ml/data.py
python ml/build_zone_map.py
python ml/train.py
python -m uvicorn main:app --app-dir server --port 8001
npm run dev -w 01_nyc_taxi_prediction/client
```

## References

1. NYC Taxi & Limousine Commission (2024). TLC Trip Record Data. https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
2. Chapman, P., Clinton, J., Kerber, R., Khabaza, T., Reinartz, T., Shearer, C., & Wirth, R. (2000). CRISP-DM 1.0: Step-by-step data mining guide. SPSS Inc.
3. Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. Advances in Neural Information Processing Systems 30.
4. Micci-Barreca, D. (2001). A preprocessing scheme for high-cardinality categorical attributes in classification and prediction problems. ACM SIGKDD Explorations Newsletter, 3(1), 27–32.
5. Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830.

---

*Generated from this project's artifacts by `scripts/generate_docs.py`.*
