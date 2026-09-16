# CRISP-DM Report — Customer Segmentation

**Business question.** Which groups of customers behave differently enough to be treated differently, and do those groups still mean anything a year later?


## 1. Business Understanding

*Objective:* Find segments a retailer could act on, and prove they are more than an artefact of the algorithm.


Segmentation is unusually easy to fake. Every clustering algorithm returns clusters, including on random noise, and the usual quality scores measure how tidy the clusters look rather than whether they are useful.

A segment is useful if it predicts how customers will behave, and it is trustworthy if rebuilding it on a different sample gives the same answer. So segments here are built from customers' **first year** of purchases and judged on their **second year**, and every segmentation is rebuilt on repeated subsamples to measure its stability.

Two baselines keep the comparison honest: segments assigned at random, which should explain nothing, and the classic RFM scoring rule marketers have used for decades, which any learned method has to beat to be worth its complexity.


| Finding | Value | Detail |
|---|---|---|
| Primary metric | `ω² of future spend` | variance explained, corrected for k |
| Robustness | `Subsample stability` | adjusted Rand index |
| Baselines | `Random · RFM rule` |  |


*Code:* `ml/train.py:omega_squared`, `ml/train.py:stability`


> **Phase gate: PASSED.** Outcome, stability test and baselines fixed before clustering.


## 2. Data Understanding

*Objective:* Reduce two years of raw transaction lines to genuine product sales by known customers.


The export holds **1,067,371 lines** across two yearly sheets.

| Rule | Removed | Share | Remaining |
|---|---|---|---|
| exact duplicate lines (includes the 9-day overlap between sheets) | 34,335 | 3.22% | 1,033,036 |
| no customer ID | 235,151 | 22.03% | 797,885 |
| cancellation lines (kept aside for return rates) | 18,390 | 1.72% | 779,495 |
| non-product codes (postage, fees, adjustments, vouchers) | 2,858 | 0.27% | 776,637 |
| non-positive quantity or price | 60 | 0.01% | 776,577 |

The two sheets **overlap by nine days**, 1 to 9 December 2010, so a straightforward concatenation counts those orders twice. The duplicate rule removes them along with genuine repeated lines.

Cancellations are removed from sales but not discarded: how often a customer returns goods is behaviour, and it becomes a feature.


| Finding | Value | Detail |
|---|---|---|
| Raw lines | `1,067,371` |  |
| Product sales lines | `776,577` |  |
| First-year customers | `4,239` |  |


*Code:* `datasets/loaders/online_retail.py:clean`


> **Phase gate: PASSED.** 776,577 product lines retained, each removal rule recorded.


## 3. Data Preparation

*Objective:* Describe each customer by first-year behaviour only.


Features are computed at a snapshot on 01 December 2010 from first-year purchases alone: recency, order count, total spend, average basket, distinct products, tenure and return rate. Counts and amounts are log-transformed so a handful of wholesale buyers do not each form a cluster of their own.

The outcome — second-year revenue — is attached afterwards and used only for scoring. **64.2%** of first-year customers bought again in the second year.

Segmentations are scored on $\omega^2$ of log future revenue:

$$\omega^2 = \frac{SS_B - (k-1)\,MS_W}{SS_T + MS_W}$$

Plain $\eta^2$ grows with every extra segment, so a twenty-way split would always win. $\omega^2$ subtracts what $k$ groups explain by chance, which is why it can be negative.


| Finding | Value | Detail |
|---|---|---|
| Customers | `4,239` |  |
| Features | `7` |  |
| Bought again in year two | `64.2%` |  |


*Code:* `ml/features.py:customer_features`, `ml/train.py:omega_squared`


> **Phase gate: PASSED.** Features and outcome drawn from disjoint, ordered periods; assertions executed.


## 4. Modeling

*Objective:* Compare partitioning, probabilistic, hierarchical and density-based methods against two baselines.


Eleven segmentations were built on identical first-year features. The highest silhouette belongs to **k-means (k=4)** (0.301); the highest future-spend separation belongs to **RFM scoring rule** (ω² 0.3062). They are different methods, which is the reason silhouette is not the ranking metric.

The RFM rule reaches ω² 0.3062 with no learning at all.


| Finding | Value | Detail |
|---|---|---|
| Segmentations | `11` |  |
| Best on future spend | `RFM scoring rule` |  |
| Best silhouette | `k-means (k=4)` |  |
| RFM rule ω² | `0.3062` |  |


*Code:* `ml/train.py:evaluate`


> **Phase gate: PASSED.** 11 segmentations scored on outcome, stability and geometry.


## 5. Evaluation

*Objective:* Decide whether any learned segmentation earns its place over the marketing rule.


**RFM scoring rule** separates second-year spend best, at ω² 0.3062. The RFM rule itself won: no learned method beat a decades-old heuristic. Its segments rebuild with a stability of 0.970 (adjusted Rand index, where 1 is identical and 0 is chance), and retention differs by 58.3 percentage points between its most and least loyal segments.

Random segments score close to zero, as they should: that row is the check that the metric is not rewarding the act of splitting itself.

Stability deserves attention in the leaderboard. A method can separate future spend well and still produce different segments every time it is rebuilt, and a marketing team cannot act on a segment that changes on every rerun.


| Finding | Value | Detail |
|---|---|---|
| Best ω² | `0.3062` |  |
| Stability (ARI) | `0.970` |  |
| Retention spread | `58.3 pts` |  |
| Critical findings | `0` |  |


*Code:* `ml/train.py:omega_squared`, `ml/train.py:stability`


> **Phase gate: PASSED.** No critical findings; every segment scored on behaviour it never saw.


## 6. Deployment

*Objective:* Assign a new customer to a segment and show what that segment did next.


`POST /api/assign` takes a customer's recency, order count, spend and the other features, and returns the nearest segment of the winning method by distance to its centre in the scaled feature space, with that segment's second-year behaviour. Nearest-centre assignment works for every method in the tournament, including those with no native way to label a new point.


| Finding | Value | Detail |
|---|---|---|
| Served segmentation | `RFM scoring rule` |  |
| Serving port | `8,003` |  |


*Code:* `server/inference.py`


> **Phase gate: PASSED.** Scaler, segment centres and profiles persisted together.
