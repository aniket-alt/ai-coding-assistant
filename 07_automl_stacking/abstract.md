# Abstract — AutoML Stacking

**Which clients should a bank call about a term deposit, and does an automated stacked ensemble choose them better than one well-tuned model?**

We study this question on *Bank Marketing, with social and economic context (UCI)*: 41,188 rows as downloaded, used under Creative Commons Attribution 4.0 (CC BY 4.0). Following the CRISP-DM process, we compare 8 approaches, 2 of them baselines, on 2,058 evaluation rows and rank them by roc auc, 2010. Evaluation split: chronological: calls before 2010 train, calls in 2010 test.

The best approach, **LightGBM, tuned**, reaches 0.6969 against 0.6337 for the strongest baseline (Previous-campaign outcome rule).

3 of 3 leakage assertions passed while training ran, and static analysis of the source reported 0 critical and 0 warning findings.

**Keywords:** binary classification, ranking clients to call, AutoGluon best_quality, Caruana ensemble selection, built by hand, Optuna tuning, paired bootstrap on the deployment year

---

*Generated from this project's artifacts by `scripts/generate_docs.py`. Figures are not typed by hand.*
