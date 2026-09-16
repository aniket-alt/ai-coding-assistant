# Abstract — Market Basket Mining

**Which products go together strongly enough to recommend, and do those associations still hold next season?**

We study this question on *Online Retail II (UCI)*: 1,067,371 rows as downloaded, used under Creative Commons Attribution 4.0 (CC BY 4.0). Following the CRISP-DM process, we compare 7 approaches, 2 of them baselines, on 6,548 evaluation rows and rank them by hit rate @ 5. Evaluation split: chronological with 7-day embargo, one item hidden per test basket.

The best approach, **Item co-occurrence**, reaches 18.6% against 1.9% for the strongest baseline (Bestsellers).

4 of 4 leakage assertions passed while training ran, and static analysis of the source reported 0 critical and 0 warning findings.

**Keywords:** association rule mining and recommendation, FP-Growth association rules, held-out item recommendation test, rule survival in a later season, item co-occurrence baseline

---

*Generated from this project's artifacts by `scripts/generate_docs.py`. Figures are not typed by hand.*
