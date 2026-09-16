# Which customers behave differently, and do the groups still mean anything a year later?

*Customer Segmentation — project 03 of a sixteen-project data science portfolio built with an AI coding assistant, on real data.*

![Customer Segmentation](./screenshots/01_segments.png)

## The question

Which groups of customers behave differently enough to be treated differently, and do those groups still mean anything a year later?

Segmentation is unusually easy to fake. Every clustering algorithm returns clusters, including on random noise, and the usual quality scores measure how tidy the clusters look rather than whether they are useful.

A segment is useful if it predicts how customers will behave, and it is trustworthy if rebuilding it on a different sample gives the same answer. So segments here are built from customers' **first year** of purchases and judged on their **second year**, and every segmentation is rebuilt on repeated subsamples to measure its stability.

Two baselines keep the comparison honest: segments assigned at random, which should explain nothing, and the classic RFM scoring rule marketers have used for decades, which any learned method has to beat to be worth its complexity.

## The data

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

## What won

A baseline. **RFM scoring rule** scored best on ω² next-year spend (0.3062), ahead of every learned model in the tournament.

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

**RFM scoring rule** separates second-year spend best, at ω² 0.3062. The RFM rule itself won: no learned method beat a decades-old heuristic. Its segments rebuild with a stability of 0.970 (adjusted Rand index, where 1 is identical and 0 is chance), and retention differs by 58.3 percentage points between its most and least loyal segments.

Random segments score close to zero, as they should: that row is the check that the metric is not rewarding the act of splitting itself.

Stability deserves attention in the leaderboard. A method can separate future spend well and still produce different segments every time it is rebuilt, and a marketing team cannot act on a segment that changes on every rerun.

## What I took away

- Clustering always returns clusters. A segmentation has to be judged on something that can fail, such as behaviour it never saw.
- The geometrically tidiest segmentation was not the most useful one, which is why silhouette is reported but not ranked on.
- A simple, stable marketing rule is a demanding baseline, and learned methods did not beat it here.
- Stability matters as much as separation: a segment that changes on every rebuild cannot be acted on.

## How it is built

The backend is FastAPI on port 8003 and the front end is React on port 5103. Most of the interface — the metrics panel, tournament table, CRISP-DM walkthrough, audit panel and charts — is shared across the portfolio and driven entirely by the training run's artifacts, so nothing on screen is typed by hand.

Code: [github.com/aniket-alt/ai-coding-assistant](https://github.com/aniket-alt/ai-coding-assistant/tree/main/03_customer_segmentation)

---

*Figures in this article are generated from the project's artifacts.*
