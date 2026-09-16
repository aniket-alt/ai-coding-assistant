# Abstract — Credit Card Fraud Detection

**Out of hundreds of thousands of card transactions, which handful should a fraud analyst look at today?**

We study this question on *Credit Card Fraud Detection (ULB)*: 284,807 rows as downloaded, used under Public (Open Database License) — OpenML dataset 1597. Following the CRISP-DM process, we compare 8 approaches, 2 of them baselines, on 100,915 evaluation rows and rank them by avg precision. Evaluation split: chronological with 1-hour embargo.

The best approach, **Logistic regression**, reaches 0.7547 against 0.0017 for the strongest baseline (Rank by amount).

3 of 3 leakage assertions passed while training ran, and static analysis of the source reported 0 critical and 0 warning findings.

**Keywords:** anomaly detection, average precision over accuracy, chronological split with embargo, unsupervised vs supervised comparison, review-budget evaluation

---

*Generated from this project's artifacts by `scripts/generate_docs.py`. Figures are not typed by hand.*
