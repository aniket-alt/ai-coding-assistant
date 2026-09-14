# CRISP-DM Report — Credit Card Fraud Detection

**Business question.** Out of hundreds of thousands of card transactions, which handful should a fraud analyst look at today?


## 1. Business Understanding

*Objective:* Rank transactions by suspicion so a limited analyst team spends its review budget on the transactions most likely to be fraud.


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



| Finding | Value | Detail |
|---|---|---|
| Primary metric | `Average precision` | area under precision-recall |
| Product metric | `Fraud caught in 100 alerts` |  |
| Rejected metric | `Accuracy` | 99.83% by predicting nothing |


*Code:* `ml/data.py:THE_ACCURACY_TRAP`, `ml/train.py:evaluate`


> **Phase gate: PASSED.** Ranking metric and review budget fixed before any model was fitted.


## 2. Data Understanding

*Objective:* Understand the shape of the imbalance and where fraud actually sits.


284,807 transactions over 48 hours, of which **492 are fraudulent** — a rate of 0.173%, or one in 578.

The intuitive assumption is that fraud means large transactions. It does not. Median fraudulent amount is **€9.25** against **€22.00** for legitimate transactions — fraud here skews *small*, consistent with card-testing, where a stolen number is validated with a trivial charge before being used in earnest. 27 fraudulent transactions are for exactly €0.

Total value at stake across the window is €60,128.

Features V1–V28 are principal components published in place of the original fields for cardholder confidentiality. Only elapsed time and amount survive in their native units — which means no domain feature engineering is possible here, and the modelling question is purely how well each algorithm separates the classes.


| Finding | Value | Detail |
|---|---|---|
| Transactions | `284,807` |  |
| Fraudulent | `492` | 0.173% |
| Imbalance | `1 : 578` |  |
| Median fraud amount | `€9.25` | vs €22.00 legitimate |
| Value at stake | `€60,128` |  |


*Code:* `ml/data.py:summarise`


> **Phase gate: PASSED.** Imbalance quantified; the accuracy trap documented before modelling.


## 3. Data Preparation

*Objective:* Split by time, and scale without letting the test window inform the scaler.


Fraud arrives in campaigns: a compromised batch of cards is exercised over hours. A random split would place the same campaign on both sides of the boundary and flatter every model. The split is therefore chronological — training on the first 34 hours, testing on the last, with the intervening hour embargoed.

That leaves **361 fraudulent transactions to learn from and 125 to be judged on**. The test window is genuinely unseen time.

Only `Time` and `Amount` are scaled, with a RobustScaler fitted inside the pipeline on training rows alone; V1–V28 are already PCA components on a common scale. Because the scaler lives in the pipeline rather than being applied to the frame up front, it cannot see the test window.


| Finding | Value | Detail |
|---|---|---|
| Train rows | `175,582` |  |
| Test rows | `100,915` |  |
| Fraud in train | `361` |  |
| Fraud in test | `125` |  |
| Embargo gap | `0 days 01:00:03` |  |


*Code:* `ml/train.py:TRAIN_END_S`, `ml/train.py:make_pipeline`


> **Phase gate: PASSED.** Temporal ordering, disjointness and target-correlation assertions all executed.


## 4. Modeling

*Objective:* Compare unsupervised detectors against supervised classifiers on identical data.


Two families compete here, and the comparison is not quite fair by design.

The **unsupervised** detectors — Isolation Forest, Elliptic Envelope, Local Outlier Factor — never see a label. They are asked only to find points that look unusual, and are then scored against labels they had no access to. That is the honest setting for a genuinely new fraud pattern nobody has labelled yet.

The **supervised** models do see the 361 labelled frauds in the training window. They should win, and the size of the gap is the value of having labels at all.

Best unsupervised detector: **Isolation Forest** at 0.0326 average precision, against **Logistic regression** at 0.7547.

### Two results worth not smoothing over

**Class reweighting made the gradient booster much worse.** Setting `scale_pos_weight=485` — the textbook remedy for 1:578 imbalance, and what most tutorials reach for first — takes LightGBM's average precision from 0.68 to 0.03. The reweighting optimises for a balanced decision threshold, while average precision measures *ranking*; inflating 361 positives by a factor of several hundred lets the booster memorise them, and it pays for that with confident false positives at the very top of the ranking, which is the only part of the ranking anyone reads. Both variants are in the table above.

**Local Outlier Factor scores below random** (ROC 0.17 — systematically inverted, not merely uninformative). LOF asks whether a point sits in a sparser neighbourhood than its neighbours do. In this PCA space the frauds resemble *each other*, forming their own dense pocket, so LOF reads them as comfortably normal. A density-based detector fails exactly when the anomalies are a cluster rather than scattered outliers, and this dataset is that case.


| Finding | Value | Detail |
|---|---|---|
| Detectors | `8` |  |
| Winner | `Logistic regression` |  |
| Average precision | `0.7547` |  |
| Caught in budget | `88/125` |  |


*Code:* `ml/train.py:run_model`


> **Phase gate: PASSED.** 8 detectors evaluated; winner Logistic regression.


## 5. Evaluation

*Objective:* Decide whether the ranking is good enough to put in front of an analyst.


**Logistic regression** reaches an average precision of 0.7547 on the unseen window, against 0.0012 for the do-nothing baseline.

Operationally: reviewing the 100 most suspicious transactions catches **88 of 125 frauds** (70.4%), at a precision of 88.0% — meaning roughly 88 in every 100 alerts is genuine. Catching half of all fraud takes 64 alerts.

Note the accuracy column in the leaderboard. Every model sits above 99.8%, including the one that catches nothing. It separates none of them, which is exactly why it is not the ranking metric.


| Finding | Value | Detail |
|---|---|---|
| Average precision | `0.7547` |  |
| ROC-AUC | `0.9776` |  |
| Caught in 100 alerts | `88/125` |  |
| Alerts for half | `64` |  |
| Critical findings | `0` |  |


*Code:* `ml/train.py:evaluate`


> **Phase gate: PASSED.** No critical findings; measured on a future time window.


## 6. Deployment

*Objective:* Serve a suspicion score, and let an operator see the cost of their threshold.


`POST /api/score` accepts a transaction and returns its suspicion score plus where that score sits in the distribution of the unseen window — a score means little without knowing whether it is in the top ten or the top ten thousand.

`GET /api/budget-curve` returns precision and recall at each review budget, so an operator can see directly what doubling the analyst team would buy.


| Finding | Value | Detail |
|---|---|---|
| Served model | `Logistic regression` |  |
| Serving port | `8,006` |  |
| Endpoints | `11` |  |


*Code:* `server/main.py`, `server/inference.py`


> **Phase gate: PASSED.** Model and feature order persisted; budget curve precomputed from held-out scores.
