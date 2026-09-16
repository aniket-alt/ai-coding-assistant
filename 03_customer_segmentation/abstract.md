# Abstract — Customer Segmentation

**Which groups of customers behave differently enough to be treated differently, and do those groups still mean anything a year later?**

We study this question on *Online Retail II (UCI)*: 1,067,371 rows as downloaded, used under Creative Commons Attribution 4.0 (CC BY 4.0). Following the CRISP-DM process, we compare 11 approaches, 2 of them baselines, on 4,239 evaluation rows and rank them by ω² next-year spend. Evaluation split: features from year one, outcomes from year two.

The best-scoring approach is itself a baseline: **RFM scoring rule**, at 0.3062. No learned model improved on it.

3 of 3 leakage assertions passed while training ran, and static analysis of the source reported 0 critical and 0 warning findings.

**Keywords:** customer segmentation (clustering), segments scored on next-year spend, omega-squared corrected for segment count, subsample stability by adjusted Rand index, RFM rule as a baseline

---

*Generated from this project's artifacts by `scripts/generate_docs.py`. Figures are not typed by hand.*
