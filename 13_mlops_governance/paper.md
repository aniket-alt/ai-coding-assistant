# MLOps & Governance

*How fast does a deployed model go stale, and when is retraining worth it?*

**Aniket Anil Naik** · Project 13 of the data science full-stack portfolio

## Abstract

**Once project 01's taxi model is in service, how fast does it go stale, and which retraining policy keeps it accurate at the lowest cost?**

We study this question on *NYC TLC Yellow Taxi Trip Records — January to July 2024*: 23,408,996 rows as downloaded, used under NYC Taxi & Limousine Commission open data (public domain). Following the CRISP-DM process, we compare 6 approaches, 2 of them baselines, on 900,000 evaluation rows and rank them by mean rmse (min). Evaluation split: month-by-month, each policy trained only on finished months.

The best approach, **Retrain monthly on all history**, reaches 6.396 against 8.322 for the strongest baseline (Route median lookup, refreshed monthly).

3 of 3 leakage assertions passed while training ran, and static analysis of the source reported 0 critical and 0 warning findings.

**Keywords:** model monitoring and retraining policy, month-by-month retraining policies, population stability index, TreeSHAP contributions, measured load test


## 1. Introduction

*Objective.* Decide how often to retrain, using evidence from real months rather than a rule of thumb.

A model is trained once and then used on a world that keeps moving. Two questions follow: how quickly does accuracy decay, and does retraining earn its cost? The usual answers — retrain monthly, or retrain when a drift metric crosses a threshold — are conventions, not findings.

This project answers them with six real months. Project 01's model and features are taken unchanged, trained on January 2024, and then operated from February to July under several retraining policies. Each month, a policy may train only on months that have already finished; its model is then scored on the new month.

Around that experiment sit the pieces an auditor asks for: a model registry, per-prediction explanations, a measured load test and a model card.

| Finding | Value | Note |
|---|---|---|
| Training month | January 2024 |  |
| Monitored months | 6 |  |
| Drift alarm | PSI > 0.1 |  |

> Phase gate passed: Policies, drift threshold and evaluation fixed before any month was scored.


## 2. Data

*Objective.* Measure how much the city actually changed between January and July.

| Month | Trips | Median duration | Median distance |
|---|---:|---:|---:|
| 2024-01 | 2,964,624 | 11.73 min | 1.70 mi |
| 2024-02 | 3,007,526 | 12.08 min | 1.72 mi |
| 2024-03 | 3,582,628 | 12.57 min | 1.80 mi |
| 2024-04 | 3,514,289 | 12.93 min | 1.81 mi |
| 2024-05 | 3,723,833 | 13.45 min | 1.82 mi |
| 2024-06 | 3,539,193 | 13.08 min | 1.82 mi |
| 2024-07 | 3,076,903 | 12.92 min | 1.83 mi |

Median trip duration rises from 11.73 minutes in January to 13.45 in May, while median distance barely moves. Trips got slower rather than longer: the model's target shifted even where its inputs did not.

Drift against January is measured with the population stability index. By convention, values below 0.1 are stable and above 0.25 are a major shift:

| Month | duration_min | trip_distance | hour | passenger_count | PULocationID |
|---|---:|---:|---:|---:|---:|
| 2024-02 | 0.002 | 0.001 | 0.002 | 0.000 | 0.004 |
| 2024-03 | 0.009 | 0.003 | 0.006 | 0.002 | 0.009 |
| 2024-04 | 0.018 | 0.005 | 0.004 | 0.002 | 0.007 |
| 2024-05 | 0.038 | 0.006 | 0.004 | 0.002 | 0.008 |
| 2024-06 | 0.025 | 0.006 | 0.006 | 0.001 | 0.012 |
| 2024-07 | 0.019 | 0.007 | 0.002 | 0.000 | 0.016 |

| Finding | Value | Note |
|---|---|---|
| Trips across seven months | 23,408,996 |  |
| Median duration, January | 11.73 min |  |
| Peak median duration | 13.45 min |  |

### Provenance

✅ **Real, third-party data**

| Field | Value |
|---|---|
| Dataset | NYC TLC Yellow Taxi Trip Records — January to July 2024 |
| Source | https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet |
| Licence | NYC Taxi & Limousine Commission open data (public domain) |
| Rows × columns (as downloaded) | 23,408,996 × 19 |
| SHA-256 | `c4d59da7bbc8abaeeeb1727947ee93d9891a71acb42854bd80db1571b2030510` |
| Months | [{'month': '2024-01', 'rows': 2964624, 'sha256': 'c4d59da7bbc8abaeeeb1727947ee93d9891a71acb42854bd80db1571b2030510'}, {'month': '2024-02', 'rows': 3007526, 'sha256': 'c76c43c18c6c6664080dd920baab4928988d5786a6b65980792ca7cd796f9f20'}, {'month': '2024-03', 'rows': 3582628, 'sha256': '2d4cdc8fb96726cdd3803b13b02d2e61e71d45720aff0ebc693a8bdd1f249823'}, {'month': '2024-04', 'rows': 3514289, 'sha256': 'ce37f736c7cf9e92164c850ef2b502a8f7bb590fc1804b6edbcb80293da01b89'}, {'month': '2024-05', 'rows': 3723833, 'sha256': '1974230be2b5a3ea92e47b5eb77f32dbf1de896d81ad8df4ab6faba52749ad23'}, {'month': '2024-06', 'rows': 3539193, 'sha256': '677cf14c8347f745b583f012fbdba072334c6c4efa17bfd6a64369f2ba30329c'}, {'month': '2024-07', 'rows': 3076903, 'sha256': 'af12d6b04daeb78550799ae99933d495b27d715b6dbb0fad9bb750a4537d3070'}] |
| Rows modelled per month | 150,000 |

> Phase gate passed: Seven real months cleaned with project 01's rules; drift measured on every monitored month.


## 3. Method

*Objective.* Guarantee no policy ever trains on the month it is judged on.

Each policy chooses its training months at the start of a month, from months that have already finished. Every fit is written to the model registry with its training window, and an assertion checks all 14 fits: none may include the month it serves.

Features and hyperparameters are project 01's, unchanged, so the only variable is what the model was trained on and when.

| Finding | Value | Note |
|---|---|---|
| Model fits | 14 |  |
| Policies | 4 |  |
| Baselines | 2 |  |

> Phase gate passed: 14 registry entries verified against their serving month.


## 4. Models

*Objective.* Compare four retraining policies and two model-free baselines month by month.

- **Never retrain** serves January's model all year.
- **Retrain monthly on last month** follows the most recent month only.
- **Retrain monthly on all history** accumulates every finished month.
- **Retrain on drift alarm** retrains on all history only when some monitored column's PSI against the model's training data exceeds 0.1.

The two baselines refresh without learning: January's median duration, and the median duration of each pickup–dropoff pair over all finished months.

| Finding | Value | Note |
|---|---|---|
| Best policy | Retrain monthly on all history |  |
| Mean RMSE | 6.396 min |  |
| Retrains | 6 |  |

**Table 1.** Tournament results.

| # | Strategy | Mean RMSE (min) | Worst month | July | Models fitted | Within 5 min |
|---|---|---|---|---|---|---|
| 1 | **Retrain monthly on all history** 🏆 | 6.396 | 6.875 | 6.039 | 6 | 77.0% |
| 2 | Retrain monthly on last month | 6.667 | 7.070 | 6.838 | 6 | 75.3% |
| 3 | Never retrain | 7.054 | 7.858 | 6.878 | 1 | 75.4% |
| 4 | Retrain on drift alarm (PSI > 0.1) | 7.054 | 7.858 | 6.878 | 1 | 75.4% |
| 5 | Route median lookup, refreshed monthly *(baseline)* | 8.322 | 9.171 | 7.733 | 0 | 71.0% |
| 6 | January median, never refreshed *(baseline)* | 14.051 | 15.396 | 14.078 | 0 | 46.1% |

Ranked by mean rmse (min) (lower is better). Split: month-by-month, each policy trained only on finished months; 900,000 evaluation rows. Generated from `artifacts/metrics.json`.

> Phase gate passed: 6 strategies scored on 6 months.


## 5. Results

*Objective.* Decide which policy to run, and whether the drift alarm earns its place.

**Retrain monthly on all history** keeps average error lowest, at 6.396 minutes RMSE, against 7.054 for never retraining. Never retraining is worst in May, at 7.858.

The drift alarm fired in 0 of 5 monthly checks and retrained 1 times in total, reaching 7.054 minutes. PSI thresholds are conventions, and this is the evidence for how much this one is worth on this model.

Served, the model answers 25 requests per second at concurrency 8 on this machine, with a 99th-percentile latency of 428.4 ms. That is one uvicorn worker on a laptop, measured, not a production claim.

| Finding | Value | Note |
|---|---|---|
| Best mean RMSE | 6.396 min |  |
| Never retrain | 7.054 min |  |
| Alarm retrains | 1 |  |
| Peak throughput | 25 req/s |  |

> Phase gate passed: No critical findings; every policy judged only on months it had not seen.

## 6. Threats to validity

### Assertions executed during training

| Assertion | Result | What it measured |
|---|---|---|
| `assert_temporal_split` | ✅ passed | Last training timestamp 2024-01-30 23:07:53 precedes first test timestamp 2024-02-01 00:00:03 by 1 days 00:52:10; no embargo required. |
| `assert_no_target_leak` | ✅ passed | 22 numeric features correlated against the target; strongest was zp_median_distance at |r|=0.7821 (threshold 0.995). |
| `assert_policies_train_only_on_finished_months` | ✅ passed | 14 model fits checked; 0 used a month at or after the month they served. |

Recorded by the assertion ledger while training ran; generated from `artifacts/metrics.json`.

### Limitations

- Each month is a 150,000-trip sample, not the full month; per-month RMSE carries sampling noise.
- Six monitored months is a short history for judging a policy, and spring 2024 may not resemble other years.
- The load test ran on one development machine with one server worker, so its figures describe that setup only.
- TreeSHAP contributions explain this model's arithmetic, not the causes of a trip's duration.


## 7. Deployment

*Objective.* Serve the model with its lineage, explanations, load profile and model card.

The served model is the one trained on all history through June. `POST /api/predict` returns the estimate together with each feature's TreeSHAP contribution in minutes. `GET /api/registry`, `/api/drift`, `/api/load-test` and `/api/model-card` publish lineage, monitoring, capacity and governance information from the same run.

| Finding | Value | Note |
|---|---|---|
| Served policy | Retrain monthly on all history |  |
| Serving port | 8,013 |  |

> Phase gate passed: Model, registry, drift history, load profile and model card persisted together.

## 8. Conclusion

- Distribution monitoring can stay quiet while accuracy degrades: the drift alarm never fired, yet never retraining was clearly worse than retraining monthly.
- Retraining on all accumulated history beat retraining on only the latest month, so more data outweighed recency here.
- A retraining policy has to be evaluated like a model, month by month on data it has never seen, with baselines beside it.
- With a single worker, extra concurrency did not raise throughput; it only turned into queueing time.

## Reproducibility

Trained 2026-09-17T05:30:17Z in 281.6 s at commit `ada6490` with Python 3.11.6, NumPy 2.4.6, pandas 3.0.5 and scikit-learn 1.9.1.

```bash
python ml/train.py
python -m uvicorn main:app --app-dir server --port 8013
npm run dev -w 13_mlops_governance/client
```

## References

1. NYC Taxi & Limousine Commission (2024). TLC Trip Record Data. https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
2. Sculley, D. et al. (2015). Hidden technical debt in machine learning systems. Advances in Neural Information Processing Systems 28.
3. Mitchell, M. et al. (2019). Model cards for model reporting. Proceedings of FAT* 2019, 220–229.
4. Gama, J. et al. (2014). A survey on concept drift adaptation. ACM Computing Surveys, 46(4), 1–37.
5. Lundberg, S. M. et al. (2020). From local explanations to global understanding with explainable AI for trees. Nature Machine Intelligence, 2, 56–67.
6. Yurdakul, B. (2018). Statistical properties of population stability index. PhD dissertation, Western Michigan University.

---

*Generated from this project's artifacts by `scripts/generate_docs.py`.*
