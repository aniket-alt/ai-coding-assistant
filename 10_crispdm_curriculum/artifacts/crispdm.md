# CRISP-DM Report — CRISP-DM Curriculum

**Business question.** Who earns more than $50,000 a year in the 1994 US Census, and can a model predict it without treating men and women unequally?


## 1. Business Understanding

*Objective:* Teach every CRISP-DM phase on one real dataset, with a fairness question at its centre.


This project is a course. Each lesson is a real analysis of the Adult census data, and each quiz question is answered from that analysis's own output, so what is taught cannot drift from what the data shows.

The dataset earns its place for one reason beyond familiarity. It records sex and race next to variables that act as stand-ins for them, which makes it a clean demonstration of a mistake that is still common: assuming a model is fair because the protected attribute was left out of its inputs.


| Finding | Value | Detail |
|---|---|---|
| Target | `income > $50,000` |  |
| Fairness criteria | `four-fifths rule, equal opportunity` |  |
| Lessons | `7` |  |


*Code:* `ml/train.py:main`


> **Phase gate: PASSED.** Prediction target, fairness criteria and lessons fixed before modelling.


## 2. Data Understanding

*Objective:* Establish the base rates, gaps and quirks before any model.


48,842 people; **23.9%** earn above $50,000. The rate is **30.4%** for men and **10.9%** for women — a gap already present in the data, which any model trained on it will learn unless something stops it.

Three quirks shape preparation. Workclass, occupation and native country are missing for up to 5.8% of rows, marked `?` in the source. Capital gain is top-coded: 244 people report exactly $99,999. And `education` is a text copy of `education-num` — every education level maps to exactly one number, so only the ordinal version is modelled.


| Finding | Value | Detail |
|---|---|---|
| People | `48,842` |  |
| Earn > $50K | `23.9%` |  |
| Men / women | `30.4% / 10.9%` |  |
| Top-coded capital gains | `244` |  |


*Code:* `ml/data.py:load`


> **Phase gate: PASSED.** Base rates, missingness and duplicated columns documented.


## 3. Data Preparation

*Objective:* Prepare features inside pipelines so nothing is learned from the test set.


The census is a single cross-section with no time order, so a stratified random split is appropriate: 36,631 people for training and 12,211 for testing, each with the same 23.9% base rate.

Imputation, scaling and one-hot encoding all live inside a `ColumnTransformer` within each model's pipeline. They are fitted on the training rows alone, and rare categories (under 20 people) are pooled so the encoder does not memorise individuals.


| Finding | Value | Detail |
|---|---|---|
| Training people | `36,631` |  |
| Test people | `12,211` |  |


*Code:* `ml/train.py:preprocessor`


> **Phase gate: PASSED.** Disjoint split and target-correlation check executed.


## 4. Modeling

*Objective:* Fit models with and without sex and race, and find what still carries them.


Four real models were trained: logistic regression and LightGBM, each with and without sex and race, beside a majority-class baseline. Before the models, two unsupervised lessons run on the numeric columns.

**Clustering.** k-means with four clusters (silhouette 0.252) separates people largely by age, hours and capital income; the highest-income cluster reaches 62.2% above $50K and the lowest 5.5%.

**Outliers.** An isolation forest flags 367 people as the most unusual 1%. **83%** of them have a capital gain or loss, against 12% of everyone else, and 28% are 70 or older against 2%. Both capital columns are zero for almost everyone, so any non-zero value looks unusual: the detector is mostly reporting how those columns are distributed. Whether that makes someone interesting is a judgement the algorithm cannot make.

**Proxies.** A model given only the features left after removing sex and race predicts a person's sex with an ROC-AUC of **0.935**. Relationship status is one such proxy: 100% of 'Husband' are men, 56% of 'Own-child' are men.


| Finding | Value | Detail |
|---|---|---|
| Best ROC-AUC | `0.9300` |  |
| Sex recoverable from other features | `AUC 0.935` |  |
| Cluster silhouette | `0.252` |  |
| Outliers with capital gain or loss | `83%` |  |


*Code:* `ml/train.py:fairness`, `ml/train.py:enter`


> **Phase gate: PASSED.** 5 classifiers, including two fairness-through-unawareness variants.


## 5. Evaluation

*Objective:* Judge the models on accuracy and on how they treat men and women.


**LightGBM** is the most accurate, with ROC-AUC 0.9300. At a 0.5 threshold, LightGBM predicts high income for 25.7% of men and 8.3% of women: a disparate impact ratio of **0.32**, far below the four-fifths threshold of 0.8.

**Removing sex and race does almost nothing.** The unaware LightGBM's ratio is **0.34**, and its equal-opportunity gap — the difference in how often truly high-earning men and women are identified — moves from +0.100 to +0.077. The information was never only in the `sex` column.

Two more lessons check themselves on held-out people. The strongest association rules for high income are recomputed on the test set, and locality-sensitive hashing finds 90% of each person's ten most similar people while scanning 7.7% of the index.


| Finding | Value | Detail |
|---|---|---|
| Disparate impact, with sex | `0.32` |  |
| Disparate impact, without sex | `0.34` |  |
| LSH recall@10 | `90%` |  |


*Code:* `ml/train.py:fairness`, `ml/train.py:RandomHyperplaneLSH`


> **Phase gate: PASSED.** Fairness reported beside accuracy; unawareness shown not to remove disparity.


## 6. Deployment

*Objective:* Publish the course: lessons, interactive results and quizzes answered by the data.


The course is served by endpoints for each lesson — `/api/eda`, `/api/unsupervised`, `/api/fairness`, `/api/rules`, `/api/lsh` — and `/api/quiz`, whose correct answers are computed from this run.


| Finding | Value | Detail |
|---|---|---|
| Serving port | `8,010` |  |


*Code:* `server/main.py`


> **Phase gate: PASSED.** Every lesson and quiz answer served from the run record.
