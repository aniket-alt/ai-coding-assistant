# Abstract — Portfolio Audit

**How much should anyone trust this portfolio's leakage audit, and which leaks would it miss?**

We study this question on *Credit Card Fraud Detection (ULB)*: 284,807 rows as downloaded, used under Public (Open Database License) — OpenML dataset 1597. Following the CRISP-DM process, we compare 5 approaches, 2 of them baselines, on 14 evaluation rows and rank them by balanced accuracy. Evaluation split: planted-leak benchmark.

The best approach, **Static + runtime**, reaches 0.727 against 0.500 for the strongest baseline (Flag everything).

3 of 3 leakage assertions passed while training ran, and static analysis of the source reported 0 critical and 0 warning findings.

**Keywords:** leakage detector benchmark, planted-leak benchmark, balanced accuracy over F1, score inflation per leak, clean controls verified by score

---

*Generated from this project's artifacts by `scripts/generate_docs.py`. Figures are not typed by hand.*
