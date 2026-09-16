# Customer Segmentation

*Which customers behave differently, and do the groups still mean anything a year later?*

**Aniket Anil Naik** · Project 03 of the data science full-stack portfolio

## Abstract

**Which groups of customers behave differently enough to be treated differently, and do those groups still mean anything a year later?**

We study this question on *Online Retail II (UCI)*: 1,067,371 rows as downloaded, used under Creative Commons Attribution 4.0 (CC BY 4.0). Following the CRISP-DM process, we compare 11 approaches, 2 of them baselines, on 4,239 evaluation rows and rank them by ω² next-year spend. Evaluation split: features from year one, outcomes from year two.

The best-scoring approach is itself a baseline: **RFM scoring rule**, at 0.3062. No learned model improved on it.

3 of 3 leakage assertions passed while training ran, and static analysis of the source reported 0 critical and 0 warning findings.

**Keywords:** customer segmentation (clustering), segments scored on next-year spend, omega-squared corrected for segment count, subsample stability by adjusted Rand index, RFM rule as a baseline


## 1. Introduction

*Objective.* Find segments a retailer could act on, and prove they are more than an artefact of the algorithm.

Segmentation is unusually easy to fake. Every clustering algorithm returns clusters, including on random noise, and the usual quality scores measure how tidy the clusters look rather than whether they are useful.

A segment is useful if it predicts how customers will behave, and it is trustworthy if rebuilding it on a different sample gives the same answer. So segments here are built from customers' **first year** of purchases and judged on their **second year**, and every segmentation is rebuilt on repeated subsamples to measure its stability.

Two baselines keep the comparison honest: segments assigned at random, which should explain nothing, and the classic RFM scoring rule marketers have used for decades, which any learned method has to beat to be worth its complexity.

| Finding | Value | Note |
|---|---|---|
| Primary metric | ω² of future spend | variance explained, corrected for k |
| Robustness | Subsample stability | adjusted Rand index |
| Baselines | Random · RFM rule |  |

> Phase gate passed: Outcome, stability test and baselines fixed before clustering.


## 2. Data

*Objective.* Reduce two years of raw transaction lines to genuine product sales by known customers.

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

| Finding | Value | Note |
|---|---|---|
| Raw lines | 1,067,371 |  |
| Product sales lines | 776,577 |  |
| First-year customers | 4,239 |  |

### Provenance

✅ **Real, third-party data**

| Field | Value |
|---|---|
| Dataset | Online Retail II (UCI) |
| Source | https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip |
| Licence | Creative Commons Attribution 4.0 (CC BY 4.0) |
| Rows × columns (as downloaded) | 1,067,371 × 8 |
| SHA-256 | `572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb` |
| Customers in first year | 4,239 |
| Product lines after cleaning | 776,577 |

> Phase gate passed: 776,577 product lines retained, each removal rule recorded.


## 3. Method

*Objective.* Describe each customer by first-year behaviour only.

Features are computed at a snapshot on 01 December 2010 from first-year purchases alone: recency, order count, total spend, average basket, distinct products, tenure and return rate. Counts and amounts are log-transformed so a handful of wholesale buyers do not each form a cluster of their own.

The outcome — second-year revenue — is attached afterwards and used only for scoring. **64.2%** of first-year customers bought again in the second year.

Segmentations are scored on $\omega^2$ of log future revenue:

$$\omega^2 = \frac{SS_B - (k-1)\,MS_W}{SS_T + MS_W}$$

Plain $\eta^2$ grows with every extra segment, so a twenty-way split would always win. $\omega^2$ subtracts what $k$ groups explain by chance, which is why it can be negative.

| Finding | Value | Note |
|---|---|---|
| Customers | 4,239 |  |
| Features | 7 |  |
| Bought again in year two | 64.2% |  |

> Phase gate passed: Features and outcome drawn from disjoint, ordered periods; assertions executed.


## 4. Models

*Objective.* Compare partitioning, probabilistic, hierarchical and density-based methods against two baselines.

Eleven segmentations were built on identical first-year features. The highest silhouette belongs to **k-means (k=4)** (0.301); the highest future-spend separation belongs to **RFM scoring rule** (ω² 0.3062). They are different methods, which is the reason silhouette is not the ranking metric.

The RFM rule reaches ω² 0.3062 with no learning at all.

| Finding | Value | Note |
|---|---|---|
| Segmentations | 11 |  |
| Best on future spend | RFM scoring rule |  |
| Best silhouette | k-means (k=4) |  |
| RFM rule ω² | 0.3062 |  |

**Table 1.** Tournament results.

| # | Segmentation | ω² next-year spend | Stability (ARI) | Silhouette | Retention spread (pts) | k |
|---|---|---|---|---|---|---|
| 1 | **RFM scoring rule** 🏆 *(baseline)* | 0.3062 | 0.970 | 0.085 | 58.3 | 4 |
| 2 | k-means (k=8) | 0.2856 | 0.734 | 0.225 | 64.0 | 8 |
| 3 | k-means (k=5) | 0.2842 | 0.946 | 0.246 | 61.0 | 5 |
| 4 | k-means (k=6) | 0.2837 | 0.906 | 0.238 | 58.0 | 6 |
| 5 | Gaussian mixture (k=6) | 0.2765 | 0.811 | 0.059 | 55.8 | 6 |
| 6 | Gaussian mixture (k=4) | 0.2623 | 0.640 | 0.101 | 53.4 | 4 |
| 7 | Agglomerative (ward, k=5) | 0.2552 | 0.411 | 0.191 | 56.6 | 5 |
| 8 | k-means (k=3) | 0.2363 | 0.966 | 0.288 | 48.1 | 3 |
| 9 | k-means (k=4) | 0.2322 | 0.970 | 0.301 | 48.9 | 4 |
| 10 | HDBSCAN | 0.0367 | 0.855 | −0.212 | 51.7 | 4 |
| 11 | Random segments *(baseline)* | −0.0001 | 0.000 | −0.009 | 3.1 | 4 |

Ranked by ω² next-year spend (higher is better). Split: features from year one, outcomes from year two; 4,239 evaluation rows. Generated from `artifacts/metrics.json`.

> Phase gate passed: 11 segmentations scored on outcome, stability and geometry.


## 5. Results

*Objective.* Decide whether any learned segmentation earns its place over the marketing rule.

**RFM scoring rule** separates second-year spend best, at ω² 0.3062. The RFM rule itself won: no learned method beat a decades-old heuristic. Its segments rebuild with a stability of 0.970 (adjusted Rand index, where 1 is identical and 0 is chance), and retention differs by 58.3 percentage points between its most and least loyal segments.

Random segments score close to zero, as they should: that row is the check that the metric is not rewarding the act of splitting itself.

Stability deserves attention in the leaderboard. A method can separate future spend well and still produce different segments every time it is rebuilt, and a marketing team cannot act on a segment that changes on every rerun.

| Finding | Value | Note |
|---|---|---|
| Best ω² | 0.3062 |  |
| Stability (ARI) | 0.970 |  |
| Retention spread | 58.3 pts |  |
| Critical findings | 0 |  |

> Phase gate passed: No critical findings; every segment scored on behaviour it never saw.

## 6. Threats to validity

### Assertions executed during training

| Assertion | Result | What it measured |
|---|---|---|
| `assert_temporal_split` | ✅ passed | Last training timestamp 2010-11-30 19:35:00 precedes first test timestamp 2010-12-01 08:26:00 by 0 days 12:51:00; no embargo required. |
| `assert_disjoint` | ✅ passed | 385,427 training and 391,150 test row indices compared; 0 appear in both. |
| `assert_no_target_leak` | ✅ passed | 7 numeric features correlated against the target; strongest was recency_days at |r|=0.4046 (threshold 0.995). |

Recorded by the assertion ledger while training ran; generated from `artifacts/metrics.json`.

### Limitations

- One retailer, two years, mostly UK gift buyers. Segments may not transfer to other businesses.
- Second-year behaviour is observed for a single following year only.
- Customers without an ID, over a fifth of lines, are excluded, and they may differ systematically from those with one.


## 7. Deployment

*Objective.* Assign a new customer to a segment and show what that segment did next.

`POST /api/assign` takes a customer's recency, order count, spend and the other features, and returns the nearest segment of the winning method by distance to its centre in the scaled feature space, with that segment's second-year behaviour. Nearest-centre assignment works for every method in the tournament, including those with no native way to label a new point.

| Finding | Value | Note |
|---|---|---|
| Served segmentation | RFM scoring rule |  |
| Serving port | 8,003 |  |

> Phase gate passed: Scaler, segment centres and profiles persisted together.

## 8. Conclusion

- Clustering always returns clusters. A segmentation has to be judged on something that can fail, such as behaviour it never saw.
- The geometrically tidiest segmentation was not the most useful one, which is why silhouette is reported but not ranked on.
- A simple, stable marketing rule is a demanding baseline, and learned methods did not beat it here.
- Stability matters as much as separation: a segment that changes on every rebuild cannot be acted on.

## Reproducibility

Trained 2026-09-16T23:17:48Z in 30.1 s at commit `799695b` with Python 3.11.6, NumPy 2.4.6, pandas 3.0.5 and scikit-learn 1.9.1.

```bash
python ml/train.py
python -m uvicorn main:app --app-dir server --port 8003
npm run dev -w 03_customer_segmentation/client
```

## References

1. Chen, D. (2019). Online Retail II [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D
2. Hughes, A. M. (1994). Strategic Database Marketing. Probus Publishing.
3. Hubert, L., & Arabie, P. (1985). Comparing partitions. Journal of Classification, 2(1), 193–218.
4. Olejnik, S., & Algina, J. (2003). Generalized eta and omega squared statistics. Psychological Methods, 8(4), 434–447.
5. Campello, R. J. G. B., Moulavi, D., & Sander, J. (2013). Density-based clustering based on hierarchical density estimates. PAKDD.
6. Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. Journal of Computational and Applied Mathematics, 20, 53–65.

---

*Generated from this project's artifacts by `scripts/generate_docs.py`.*
