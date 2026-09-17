# CRISP-DM Report — MLOps & Governance

**Business question.** Once project 01's taxi model is in service, how fast does it go stale, and which retraining policy keeps it accurate at the lowest cost?


## 1. Business Understanding

*Objective:* Decide how often to retrain, using evidence from real months rather than a rule of thumb.


A model is trained once and then used on a world that keeps moving. Two questions follow: how quickly does accuracy decay, and does retraining earn its cost? The usual answers — retrain monthly, or retrain when a drift metric crosses a threshold — are conventions, not findings.

This project answers them with six real months. Project 01's model and features are taken unchanged, trained on January 2024, and then operated from February to July under several retraining policies. Each month, a policy may train only on months that have already finished; its model is then scored on the new month.

Around that experiment sit the pieces an auditor asks for: a model registry, per-prediction explanations, a measured load test and a model card.


| Finding | Value | Detail |
|---|---|---|
| Training month | `January 2024` |  |
| Monitored months | `6` |  |
| Drift alarm | `PSI > 0.1` |  |


*Code:* `ml/train.py:main`


> **Phase gate: PASSED.** Policies, drift threshold and evaluation fixed before any month was scored.


## 2. Data Understanding

*Objective:* Measure how much the city actually changed between January and July.


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


| Finding | Value | Detail |
|---|---|---|
| Trips across seven months | `23,408,996` |  |
| Median duration, January | `11.73 min` |  |
| Peak median duration | `13.45 min` |  |


*Code:* `ml/data.py:load_month`, `ml/train.py:psi_numeric`


> **Phase gate: PASSED.** Seven real months cleaned with project 01's rules; drift measured on every monitored month.


## 3. Data Preparation

*Objective:* Guarantee no policy ever trains on the month it is judged on.


Each policy chooses its training months at the start of a month, from months that have already finished. Every fit is written to the model registry with its training window, and an assertion checks all 14 fits: none may include the month it serves.

Features and hyperparameters are project 01's, unchanged, so the only variable is what the model was trained on and when.


| Finding | Value | Detail |
|---|---|---|
| Model fits | `14` |  |
| Policies | `4` |  |
| Baselines | `2` |  |


*Code:* `ml/train.py:run_policy`, `ml/train.py:record_fit`


> **Phase gate: PASSED.** 14 registry entries verified against their serving month.


## 4. Modeling

*Objective:* Compare four retraining policies and two model-free baselines month by month.


- **Never retrain** serves January's model all year.
- **Retrain monthly on last month** follows the most recent month only.
- **Retrain monthly on all history** accumulates every finished month.
- **Retrain on drift alarm** retrains on all history only when some monitored column's PSI against the model's training data exceeds 0.1.

The two baselines refresh without learning: January's median duration, and the median duration of each pickup–dropoff pair over all finished months.


| Finding | Value | Detail |
|---|---|---|
| Best policy | `Retrain monthly on all history` |  |
| Mean RMSE | `6.396 min` |  |
| Retrains | `6` |  |


*Code:* `ml/train.py:on_alarm`, `ml/train.py:baseline`


> **Phase gate: PASSED.** 6 strategies scored on 6 months.


## 5. Evaluation

*Objective:* Decide which policy to run, and whether the drift alarm earns its place.


**Retrain monthly on all history** keeps average error lowest, at 6.396 minutes RMSE, against 7.054 for never retraining. Never retraining is worst in May, at 7.858.

The drift alarm fired in 0 of 5 monthly checks and retrained 1 times in total, reaching 7.054 minutes. PSI thresholds are conventions, and this is the evidence for how much this one is worth on this model.

Served, the model answers 25 requests per second at concurrency 8 on this machine, with a 99th-percentile latency of 428.4 ms. That is one uvicorn worker on a laptop, measured, not a production claim.


| Finding | Value | Detail |
|---|---|---|
| Best mean RMSE | `6.396 min` |  |
| Never retrain | `7.054 min` |  |
| Alarm retrains | `1` |  |
| Peak throughput | `25 req/s` |  |


*Code:* `ml/train.py:main`, `ml/load_test.py`


> **Phase gate: PASSED.** No critical findings; every policy judged only on months it had not seen.


## 6. Deployment

*Objective:* Serve the model with its lineage, explanations, load profile and model card.


The served model is the one trained on all history through June. `POST /api/predict` returns the estimate together with each feature's TreeSHAP contribution in minutes. `GET /api/registry`, `/api/drift`, `/api/load-test` and `/api/model-card` publish lineage, monitoring, capacity and governance information from the same run.


| Finding | Value | Detail |
|---|---|---|
| Served policy | `Retrain monthly on all history` |  |
| Serving port | `8,013` |  |


*Code:* `server/main.py`, `server/inference.py`


> **Phase gate: PASSED.** Model, registry, drift history, load profile and model card persisted together.
