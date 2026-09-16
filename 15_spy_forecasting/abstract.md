# Abstract — SPY Probabilistic Forecasting

**How much can tomorrow's S&P 500 move actually be predicted, and how wide should the honest uncertainty band be?**

We study this question on *SPY Daily Bars with VIX and 10-Year Yield*: 2,931 rows as downloaded, used under Yahoo Finance terms of use — personal and research use. Following the CRISP-DM process, we compare 6 approaches, 3 of them baselines, on 1,981 evaluation rows and rank them by pinball loss. Evaluation split: walk-forward, 32 quarterly refits, purged.

The best-scoring approach is itself a baseline: **Historical drift**, at 0.002660. No learned model improved on it.

2 of 2 leakage assertions passed while training ran, and static analysis of the source reported 0 critical and 0 warning findings.

**Keywords:** probabilistic time series forecasting, quantile forecasting on pinball loss, walk-forward with purging, empirical lookahead verification, paired bootstrap significance

---

*Generated from this project's artifacts by `scripts/generate_docs.py`. Figures are not typed by hand.*
