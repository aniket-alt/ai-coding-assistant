# CRISP-DM Curriculum

*Seven lessons on the Adult census, and why leaving out sex does not make a model fair.*

**Aniket Anil Naik** · Project 10 of the data science full-stack portfolio

## Abstract

**Who earns more than $50,000 a year in the 1994 US Census, and can a model predict it without treating men and women unequally?**

We study this question on *Adult Census Income (UCI, via OpenML 1590)*: 48,842 rows as downloaded, used under Public (OpenML); originally UCI Machine Learning Repository, CC BY 4.0. Following the CRISP-DM process, we compare 5 approaches, 1 of them baselines, on 12,211 evaluation rows and rank them by roc auc. Evaluation split: stratified random 75/25 (cross-sectional census).

The best approach, **LightGBM**, reaches 0.9300 against 0.5000 for the strongest baseline (Majority class).

3 of 3 leakage assertions passed while training ran, and static analysis of the source reported 0 critical and 0 warning findings.

**Keywords:** binary classification with a fairness audit, fairness through unawareness, tested, proxy detection, isolation forest and k-means, random-hyperplane LSH


## 1. Introduction

*Objective.* Teach every CRISP-DM phase on one real dataset, with a fairness question at its centre.

This project is a course. Each lesson is a real analysis of the Adult census data, and each quiz question is answered from that analysis's own output, so what is taught cannot drift from what the data shows.

The dataset earns its place for one reason beyond familiarity. It records sex and race next to variables that act as stand-ins for them, which makes it a clean demonstration of a mistake that is still common: assuming a model is fair because the protected attribute was left out of its inputs.

| Finding | Value | Note |
|---|---|---|
| Target | income > $50,000 |  |
| Fairness criteria | four-fifths rule, equal opportunity |  |
| Lessons | 7 |  |

> Phase gate passed: Prediction target, fairness criteria and lessons fixed before modelling.


## 2. Data

*Objective.* Establish the base rates, gaps and quirks before any model.

48,842 people; **23.9%** earn above $50,000. The rate is **30.4%** for men and **10.9%** for women — a gap already present in the data, which any model trained on it will learn unless something stops it.

Three quirks shape preparation. Workclass, occupation and native country are missing for up to 5.8% of rows, marked `?` in the source. Capital gain is top-coded: 244 people report exactly $99,999. And `education` is a text copy of `education-num` — every education level maps to exactly one number, so only the ordinal version is modelled.

| Finding | Value | Note |
|---|---|---|
| People | 48,842 |  |
| Earn > $50K | 23.9% |  |
| Men / women | 30.4% / 10.9% |  |
| Top-coded capital gains | 244 |  |

### Provenance

✅ **Real, third-party data**

| Field | Value |
|---|---|
| Dataset | Adult Census Income (UCI, via OpenML 1590) |
| Source | https://openml.org/data/v1/download/1595261/adult.arff |
| Licence | Public (OpenML); originally UCI Machine Learning Repository, CC BY 4.0 |
| Rows × columns (as downloaded) | 48,842 × 16 |
| SHA-256 | `77aa1703717a29f0b5642e94c3ba1defd2486f0b34d4d8eccc1b37a5f7d226b0` |

> Phase gate passed: Base rates, missingness and duplicated columns documented.


## 3. Method

*Objective.* Prepare features inside pipelines so nothing is learned from the test set.

The census is a single cross-section with no time order, so a stratified random split is appropriate: 36,631 people for training and 12,211 for testing, each with the same 23.9% base rate.

Imputation, scaling and one-hot encoding all live inside a `ColumnTransformer` within each model's pipeline. They are fitted on the training rows alone, and rare categories (under 20 people) are pooled so the encoder does not memorise individuals.

| Finding | Value | Note |
|---|---|---|
| Training people | 36,631 |  |
| Test people | 12,211 |  |

> Phase gate passed: Disjoint split and target-correlation check executed.


## 4. Models

*Objective.* Fit models with and without sex and race, and find what still carries them.

Four real models were trained: logistic regression and LightGBM, each with and without sex and race, beside a majority-class baseline. Before the models, two unsupervised lessons run on the numeric columns.

**Clustering.** k-means with four clusters (silhouette 0.252) separates people largely by age, hours and capital income; the highest-income cluster reaches 62.2% above $50K and the lowest 5.5%.

**Outliers.** An isolation forest flags 367 people as the most unusual 1%. **83%** of them have a capital gain or loss, against 12% of everyone else, and 28% are 70 or older against 2%. Both capital columns are zero for almost everyone, so any non-zero value looks unusual: the detector is mostly reporting how those columns are distributed. Whether that makes someone interesting is a judgement the algorithm cannot make.

**Proxies.** A model given only the features left after removing sex and race predicts a person's sex with an ROC-AUC of **0.935**. Relationship status alone nearly does it: 100% of 'Husband' are men, 56% of 'Own-child' are men.

| Finding | Value | Note |
|---|---|---|
| Best ROC-AUC | 0.9300 |  |
| Sex recoverable from other features | AUC 0.935 |  |
| Cluster silhouette | 0.252 |  |
| Outliers with capital gain or loss | 83% |  |

**Table 1.** Tournament results.

| # | Model | ROC AUC | Balanced accuracy | Disparate impact (sex) | Equal-opportunity gap |
|---|---|---|---|---|---|
| 1 | **LightGBM** 🏆 | 0.9300 | 0.8004 | 0.3213 | 0.1002 |
| 2 | LightGBM, without sex or race | 0.9290 | 0.7998 | 0.3358 | 0.0769 |
| 3 | Logistic regression | 0.9055 | 0.7647 | 0.3000 | 0.1131 |
| 4 | Logistic regression, without sex or race | 0.9048 | 0.7643 | 0.3172 | 0.0823 |
| 5 | Majority class *(baseline)* | 0.5000 | 0.5000 | — | 0.0000 |

Ranked by roc auc (higher is better). Split: stratified random 75/25 (cross-sectional census); 12,211 evaluation rows. Generated from `artifacts/metrics.json`.

> Phase gate passed: 5 classifiers, including two fairness-through-unawareness variants.


## 5. Results

*Objective.* Judge the models on accuracy and on how they treat men and women.

**LightGBM** is the most accurate, with ROC-AUC 0.9300. At a 0.5 threshold, LightGBM predicts high income for 25.7% of men and 8.3% of women: a disparate impact ratio of **0.32**, far below the four-fifths threshold of 0.8.

**Removing sex and race does almost nothing.** The unaware LightGBM's ratio is **0.34**, and its equal-opportunity gap — the difference in how often truly high-earning men and women are identified — moves from +0.100 to +0.077. The information was never only in the `sex` column.

Two more lessons check themselves on held-out people. The strongest association rules for high income are recomputed on the test set, and locality-sensitive hashing finds 90% of each person's ten most similar people while scanning 7.7% of the index.

| Finding | Value | Note |
|---|---|---|
| Disparate impact, with sex | 0.32 |  |
| Disparate impact, without sex | 0.34 |  |
| LSH recall@10 | 90% |  |

> Phase gate passed: Fairness reported beside accuracy; unawareness shown not to remove disparity.

## 6. Threats to validity

### Assertions executed during training

| Assertion | Result | What it measured |
|---|---|---|
| `assert_disjoint` | ✅ passed | 36,631 training and 12,211 test row indices compared; 0 appear in both. |
| `assert_no_target_leak` | ✅ passed | 85 numeric features correlated against the target; strongest was cat__marital-status_Married-civ-spouse at |r|=0.4493 (threshold 0.995). |
| `assert_unaware_model_excludes_sensitive_attributes` | ✅ passed | 11 features in the unaware models; sensitive attributes present: []. |

Recorded by the assertion ledger while training ran; generated from `artifacts/metrics.json`.

### Limitations

- Sex is recorded as a binary field, and race in five coarse groups, as the 1994 census defined them.
- Fairness metrics are computed at a single 0.5 threshold; a different operating point changes them.
- Capital gains are top-coded at 99,999, so the largest incomes are indistinguishable.
- Association rules and clusters describe this sample, and the rules were rechecked on held-out people but not tested for significance.


## 7. Deployment

*Objective.* Publish the course: lessons, interactive results and quizzes answered by the data.

The course is served by endpoints for each lesson — `/api/eda`, `/api/unsupervised`, `/api/fairness`, `/api/rules`, `/api/lsh` — and `/api/quiz`, whose correct answers are computed from this run.

| Finding | Value | Note |
|---|---|---|
| Serving port | 8,010 |  |

> Phase gate passed: Every lesson and quiz answer served from the run record.

## 8. Conclusion

- Leaving sex and race out of the inputs barely changed disparate impact, because the remaining features predict sex almost perfectly.
- Relationship status alone largely encodes sex in this data, which makes it a proxy any model can use.
- Accuracy flatters a skewed problem: predicting that nobody earns above $50K is right about three times in four.
- An outlier detector finds what is unusual, not what is wrong; here that was mostly people with any capital gain or loss.
- Approximate similarity search keeps most true neighbours while scanning a small fraction of the index.

## Reproducibility

Trained 2026-09-17T05:37:32Z in 11.5 s at commit `ada6490` with Python 3.11.6, NumPy 2.4.6, pandas 3.0.5 and scikit-learn 1.9.1.

```bash
python ml/train.py
python -m uvicorn main:app --app-dir server --port 8010
npm run dev -w 10_crispdm_curriculum/client
```

## References

1. Becker, B., & Kohavi, R. (1996). Adult [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20
2. Kohavi, R. (1996). Scaling up the accuracy of naive-Bayes classifiers: A decision-tree hybrid. Proceedings of KDD-96, 202–207.
3. Chapman, P. et al. (2000). CRISP-DM 1.0: Step-by-step data mining guide. SPSS.
4. Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in supervised learning. Advances in Neural Information Processing Systems 29.
5. Barocas, S., Hardt, M., & Narayanan, A. (2023). Fairness and Machine Learning: Limitations and Opportunities. MIT Press.
6. Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation forest. Proceedings of ICDM 2008, 413–422.
7. Charikar, M. S. (2002). Similarity estimation techniques from rounding algorithms. Proceedings of STOC 2002, 380–388.
8. Han, J., Pei, J., & Yin, Y. (2000). Mining frequent patterns without candidate generation. Proceedings of SIGMOD 2000, 1–12.

---

*Generated from this project's artifacts by `scripts/generate_docs.py`.*
