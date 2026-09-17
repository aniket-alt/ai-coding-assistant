# How fast does a deployed model go stale, and when is retraining worth it?

*MLOps & Governance — project 13 of a sixteen-project data science portfolio built with an AI coding assistant, on real data.*

![MLOps & Governance](./screenshots/01_drift_retraining.png)

## The question

Once project 01's taxi model is in service, how fast does it go stale, and which retraining policy keeps it accurate at the lowest cost?

A model is trained once and then used on a world that keeps moving. Two questions follow: how quickly does accuracy decay, and does retraining earn its cost? The usual answers — retrain monthly, or retrain when a drift metric crosses a threshold — are conventions, not findings.

This project answers them with six real months. Project 01's model and features are taken unchanged, trained on January 2024, and then operated from February to July under several retraining policies. Each month, a policy may train only on months that have already finished; its model is then scored on the new month.

Around that experiment sit the pieces an auditor asks for: a model registry, per-prediction explanations, a measured load test and a model card.

## The data

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

## What won

**Retrain monthly on all history**, at 6.396 on mean rmse (min), against 8.322 for the best baseline, Route median lookup, refreshed monthly.

| # | Strategy | Mean RMSE (min) | Worst month | July | Models fitted | Within 5 min |
|---|---|---|---|---|---|---|
| 1 | **Retrain monthly on all history** 🏆 | 6.396 | 6.875 | 6.039 | 6 | 77.0% |
| 2 | Retrain monthly on last month | 6.667 | 7.070 | 6.838 | 6 | 75.3% |
| 3 | Never retrain | 7.054 | 7.858 | 6.878 | 1 | 75.4% |
| 4 | Retrain on drift alarm (PSI > 0.1) | 7.054 | 7.858 | 6.878 | 1 | 75.4% |
| 5 | Route median lookup, refreshed monthly *(baseline)* | 8.322 | 9.171 | 7.733 | 0 | 71.0% |
| 6 | January median, never refreshed *(baseline)* | 14.051 | 15.396 | 14.078 | 0 | 46.1% |

Ranked by mean rmse (min) (lower is better). Split: month-by-month, each policy trained only on finished months; 900,000 evaluation rows. Generated from `artifacts/metrics.json`.

**Retrain monthly on all history** keeps average error lowest, at 6.396 minutes RMSE, against 7.054 for never retraining. Never retraining is worst in May, at 7.858.

The drift alarm fired in 0 of 5 monthly checks and retrained 1 times in total, reaching 7.054 minutes. PSI thresholds are conventions, and this is the evidence for how much this one is worth on this model.

Served, the model answers 25 requests per second at concurrency 8 on this machine, with a 99th-percentile latency of 428.4 ms. That is one uvicorn worker on a laptop, measured, not a production claim.

## What I took away

- Distribution monitoring can stay quiet while accuracy degrades: the drift alarm never fired, yet never retraining was clearly worse than retraining monthly.
- Retraining on all accumulated history beat retraining on only the latest month, so more data outweighed recency here.
- A retraining policy has to be evaluated like a model, month by month on data it has never seen, with baselines beside it.
- With a single worker, extra concurrency did not raise throughput; it only turned into queueing time.

## How it is built

The backend is FastAPI on port 8013 and the front end is React on port 5113. Most of the interface — the metrics panel, tournament table, CRISP-DM walkthrough, audit panel and charts — is shared across the portfolio and driven entirely by the training run's artifacts, so nothing on screen is typed by hand.

Code: [github.com/aniket-alt/ai-coding-assistant](https://github.com/aniket-alt/ai-coding-assistant/tree/main/13_mlops_governance)

---

*Figures in this article are generated from the project's artifacts.*
