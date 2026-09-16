# Credit Card Fraud Detection

*Which handful of transactions should an analyst look at today?*

**Aniket Anil Naik** · Project 06 of the data science full-stack portfolio

## Abstract

**Out of hundreds of thousands of card transactions, which handful should a fraud analyst look at today?**

We study this question on *Credit Card Fraud Detection (ULB)*: 284,807 rows as downloaded, used under Public (Open Database License) — OpenML dataset 1597. Following the CRISP-DM process, we compare 8 approaches, 2 of them baselines, on 100,915 evaluation rows and rank them by avg precision. Evaluation split: chronological with 1-hour embargo.

The best approach, **Logistic regression**, reaches 0.7547 against 0.0017 for the strongest baseline (Rank by amount).

3 of 3 leakage assertions passed while training ran, and static analysis of the source reported 0 critical and 0 warning findings.

**Keywords:** anomaly detection, average precision over accuracy, chronological split with embargo, unsupervised vs supervised comparison, review-budget evaluation


## 1. Introduction

*Objective.* Rank transactions by suspicion so a limited analyst team spends its review budget on the transactions most likely to be fraud.

Fraud detection is a ranking problem wearing a classification costume. Nobody blocks every transaction above a fixed probability; a team reviews as many alerts as it has hours for. So the operative question is not *is this fraud* but *is this among the most suspicious things I will see today*.

This dataset is 99.828% legitimate transactions.

A model that predicts "not fraud" for every single transaction therefore scores
99.828% accuracy -- higher than most of the models in the tournament below --
while catching zero fraud and being worth precisely nothing. Accuracy on a
problem this imbalanced is not a weak metric; it is an actively misleading one,
and it is the single most common way fraud results are overstated.

This project ranks on **average precision** (the area under the
precision-recall curve), which is sensitive to performance on the rare class
and ignores the vast pile of easy negatives. The all-legitimate baseline is
kept in the leaderboard specifically so its 99.8% accuracy and its ~0.002
average precision can sit on the same row.

The reported product metric is a review budget: an analyst team can only
examine so many flagged transactions a day, so what matters operationally is
how much fraud is caught inside that budget.

| Finding | Value | Note |
|---|---|---|
| Primary metric | Average precision | area under precision-recall |
| Product metric | Fraud caught in 100 alerts |  |
| Rejected metric | Accuracy | 99.83% by predicting nothing |

> Phase gate passed: Ranking metric and review budget fixed before any model was fitted.


## 2. Data

*Objective.* Understand the shape of the imbalance and where fraud actually sits.

284,807 transactions over 48 hours, of which **492 are fraudulent** — a rate of 0.173%, or one in 578.

The intuitive assumption is that fraud means large transactions. It does not. Median fraudulent amount is **€9.25** against **€22.00** for legitimate transactions — fraud here skews *small*, consistent with card-testing, where a stolen number is validated with a trivial charge before being used in earnest. 27 fraudulent transactions are for exactly €0.

Total value at stake across the window is €60,128.

Features V1–V28 are principal components published in place of the original fields for cardholder confidentiality. Only elapsed time and amount survive in their native units — which means no domain feature engineering is possible here, and the modelling question is purely how well each algorithm separates the classes.

| Finding | Value | Note |
|---|---|---|
| Transactions | 284,807 |  |
| Fraudulent | 492 | 0.173% |
| Imbalance | 1 : 578 |  |
| Median fraud amount | €9.25 | vs €22.00 legitimate |
| Value at stake | €60,128 |  |

### Provenance

✅ **Real, third-party data**

| Field | Value |
|---|---|
| Dataset | Credit Card Fraud Detection (ULB) |
| Source | https://openml.org/data/v1/download/1673544/creditcard.arff |
| Licence | Public (Open Database License) — OpenML dataset 1597 |
| Rows × columns (as downloaded) | 284,807 × 31 |
| SHA-256 | `fdaf12730dc1fc426f318b71349f24f5c5fd00aa1152940be7e7509ae3d89d2a` |

> Phase gate passed: Imbalance quantified; the accuracy trap documented before modelling.


## 3. Method

*Objective.* Split by time, and scale without letting the test window inform the scaler.

Fraud arrives in campaigns: a compromised batch of cards is exercised over hours. A random split would place the same campaign on both sides of the boundary and flatter every model. The split is therefore chronological — training on the first 34 hours, testing on the last, with the intervening hour embargoed.

That leaves **361 fraudulent transactions to learn from and 125 to be judged on**. The test window is genuinely unseen time.

Only `Time` and `Amount` are scaled, with a RobustScaler fitted inside the pipeline on training rows alone; V1–V28 are already PCA components on a common scale. Because the scaler lives in the pipeline rather than being applied to the frame up front, it cannot see the test window.

| Finding | Value | Note |
|---|---|---|
| Train rows | 175,582 |  |
| Test rows | 100,915 |  |
| Fraud in train | 361 |  |
| Fraud in test | 125 |  |
| Embargo gap | 0 days 01:00:03 |  |

> Phase gate passed: Temporal ordering, disjointness and target-correlation assertions all executed.


## 4. Models

*Objective.* Compare unsupervised detectors against supervised classifiers on identical data.

Two families compete here, and the comparison is not quite fair by design.

The **unsupervised** detectors — Isolation Forest, Elliptic Envelope, Local Outlier Factor — never see a label. They are asked only to find points that look unusual, and are then scored against labels they had no access to. That is the honest setting for a genuinely new fraud pattern nobody has labelled yet.

The **supervised** models do see the 361 labelled frauds in the training window. They should win, and the size of the gap is the value of having labels at all.

Best unsupervised detector: **Isolation Forest** at 0.0326 average precision, against **Logistic regression** at 0.7547.

### Two results worth not smoothing over

**Class reweighting made the gradient booster much worse.** Setting `scale_pos_weight=485` — the textbook remedy for 1:578 imbalance, and what most tutorials reach for first — takes LightGBM's average precision from 0.68 to 0.03. The reweighting optimises for a balanced decision threshold, while average precision measures *ranking*; inflating 361 positives by a factor of several hundred lets the booster memorise them, and it pays for that with confident false positives at the very top of the ranking, which is the only part of the ranking anyone reads. Both variants are in the table above.

**Local Outlier Factor scores below random** (ROC 0.17 — systematically inverted, not merely uninformative). LOF asks whether a point sits in a sparser neighbourhood than its neighbours do. In this PCA space the frauds resemble *each other*, forming their own dense pocket, so LOF reads them as comfortably normal. A density-based detector fails exactly when the anomalies are a cluster rather than scattered outliers, and this dataset is that case.

| Finding | Value | Note |
|---|---|---|
| Detectors | 8 |  |
| Winner | Logistic regression |  |
| Average precision | 0.7547 |  |
| Caught in budget | 88/125 |  |

**Table 1.** Tournament results.

| # | Detector | Avg precision | ROC-AUC | Fraud caught in 100 alerts | Precision in 100 alerts | Accuracy |
|---|---|---|---|---|---|---|
| 1 | **Logistic regression** 🏆 | 0.7547 | 0.9776 | 70.4% | 88.0% | 99.951% |
| 2 | LightGBM | 0.6826 | 0.9039 | 64.8% | 81.0% | 99.938% |
| 3 | LightGBM (class-reweighted) | 0.0337 | 0.9112 | 0.8% | 1.0% | 99.779% |
| 4 | Isolation Forest | 0.0326 | 0.9399 | 0.0% | 0.0% | 99.777% |
| 5 | Elliptic Envelope | 0.0190 | 0.9292 | 0.8% | 1.0% | 99.779% |
| 6 | Rank by amount *(baseline)* | 0.0017 | 0.4273 | 0.0% | 0.0% | 99.777% |
| 7 | Predict all legitimate *(baseline)* | 0.0012 | 0.5000 | 0.0% | 0.0% | 99.777% |
| 8 | Local Outlier Factor | 0.0007 | 0.1712 | 0.0% | 0.0% | 99.777% |

Ranked by avg precision (higher is better). Split: chronological with 1-hour embargo; 100,915 evaluation rows. Generated from `artifacts/metrics.json`.

> Phase gate passed: 8 detectors evaluated; winner Logistic regression.


## 5. Results

*Objective.* Decide whether the ranking is good enough to put in front of an analyst.

**Logistic regression** reaches an average precision of 0.7547 on the unseen window, against 0.0012 for the do-nothing baseline.

Operationally: reviewing the 100 most suspicious transactions catches **88 of 125 frauds** (70.4%), at a precision of 88.0% — meaning roughly 88 in every 100 alerts is genuine. Catching half of all fraud takes 64 alerts.

Note the accuracy column in the leaderboard. Every model sits above 99.8%, including the one that catches nothing. It separates none of them, which is exactly why it is not the ranking metric.

| Finding | Value | Note |
|---|---|---|
| Average precision | 0.7547 |  |
| ROC-AUC | 0.9776 |  |
| Caught in 100 alerts | 88/125 |  |
| Alerts for half | 64 |  |
| Critical findings | 0 |  |

> Phase gate passed: No critical findings; measured on a future time window.

## 6. Threats to validity

### Assertions executed during training

| Assertion | Result | What it measured |
|---|---|---|
| `assert_temporal_split` | ✅ passed | Last training timestamp 1 days 09:59:57 precedes first test timestamp 1 days 11:00:00 by 0 days 01:00:03; embargo of 30min required. |
| `assert_disjoint` | ✅ passed | 175,582 training and 100,915 test row indices compared; 0 appear in both. |
| `assert_no_target_leak` | ✅ passed | 30 numeric features correlated against the target; strongest was V17 at |r|=0.4096 (threshold 0.995). |

Recorded by the assertion ledger while training ran; generated from `artifacts/metrics.json`.

### Limitations

- Two days of data from 2013. Fraud patterns drift, and a model like this needs retraining on recent labels.
- The test window holds only a small number of frauds, so metrics carry wide uncertainty.
- Labels are confirmed fraud only. Undetected fraud in the legitimate class would make every detector look better than it is.


## 7. Deployment

*Objective.* Serve a suspicion score, and let an operator see the cost of their threshold.

`POST /api/score` accepts a transaction and returns its suspicion score plus where that score sits in the distribution of the unseen window — a score means little without knowing whether it is in the top ten or the top ten thousand.

`GET /api/budget-curve` returns precision and recall at each review budget, so an operator can see directly what doubling the analyst team would buy.

| Finding | Value | Note |
|---|---|---|
| Served model | Logistic regression |  |
| Serving port | 8,006 |  |
| Endpoints | 11 |  |

> Phase gate passed: Model and feature order persisted; budget curve precomputed from held-out scores.

## 8. Conclusion

- On a problem this imbalanced, accuracy cannot tell a useful detector from one that catches nothing.
- The textbook fix for imbalance, heavy positive-class reweighting, damaged the ranking that average precision measures.
- Density-based anomaly detection fails when the anomalies resemble each other and form their own cluster.
- Measuring against a fixed review budget turns a metric into a staffing question an operations team can act on.

## Reproducibility

Trained 2026-09-16T22:29:59Z in 63.5 s at commit `3d83606` with Python 3.11.6, NumPy 2.4.6, pandas 3.0.5 and scikit-learn 1.9.1.

```bash
python ml/data.py
python ml/train.py
python -m uvicorn main:app --app-dir server --port 8006
npm run dev -w 06_anomaly_detection/client
```

## References

1. Dal Pozzolo, A., Caelen, O., Johnson, R. A., & Bontempi, G. (2015). Calibrating probability with undersampling for unbalanced classification. IEEE Symposium Series on Computational Intelligence. Université Libre de Bruxelles, MLG.
2. Dal Pozzolo, A., Caelen, O., Johnson, R. A., & Bontempi, G. (2015). Calibrating probability with undersampling for unbalanced classification. IEEE Symposium Series on Computational Intelligence.
3. Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. PLoS ONE, 10(3), e0118432.
4. Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation forest. IEEE International Conference on Data Mining.
5. Breunig, M. M., Kriegel, H.-P., Ng, R. T., & Sander, J. (2000). LOF: Identifying density-based local outliers. ACM SIGMOD.
6. Chapman, P. et al. (2000). CRISP-DM 1.0: Step-by-step data mining guide. SPSS Inc.

---

*Generated from this project's artifacts by `scripts/generate_docs.py`.*
