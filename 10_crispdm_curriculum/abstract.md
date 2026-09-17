# Abstract — CRISP-DM Curriculum

**Who earns more than $50,000 a year in the 1994 US Census, and can a model predict it without treating men and women unequally?**

We study this question on *Adult Census Income (UCI, via OpenML 1590)*: 48,842 rows as downloaded, used under Public (OpenML); originally UCI Machine Learning Repository, CC BY 4.0. Following the CRISP-DM process, we compare 5 approaches, 1 of them baselines, on 12,211 evaluation rows and rank them by roc auc. Evaluation split: stratified random 75/25 (cross-sectional census).

The best approach, **LightGBM**, reaches 0.9300 against 0.5000 for the strongest baseline (Majority class).

3 of 3 leakage assertions passed while training ran, and static analysis of the source reported 0 critical and 0 warning findings.

**Keywords:** binary classification with a fairness audit, fairness through unawareness, tested, proxy detection, isolation forest and k-means, random-hyperplane LSH

---

*Generated from this project's artifacts by `scripts/generate_docs.py`. Figures are not typed by hand.*
