# Which handful of transactions should an analyst look at today?

*Credit Card Fraud Detection — project 06 of a sixteen-project data science portfolio built with an AI coding assistant, on real data.*

![Credit Card Fraud Detection](./screenshots/01_transaction_scorer.png)

## The question

Out of hundreds of thousands of card transactions, which handful should a fraud analyst look at today?

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

## The data

284,807 transactions over 48 hours, of which **492 are fraudulent** — a rate of 0.173%, or one in 578.

The intuitive assumption is that fraud means large transactions. It does not. Median fraudulent amount is **€9.25** against **€22.00** for legitimate transactions — fraud here skews *small*, consistent with card-testing, where a stolen number is validated with a trivial charge before being used in earnest. 27 fraudulent transactions are for exactly €0.

Total value at stake across the window is €60,128.

Features V1–V28 are principal components published in place of the original fields for cardholder confidentiality. Only elapsed time and amount survive in their native units — which means no domain feature engineering is possible here, and the modelling question is purely how well each algorithm separates the classes.

## What won

**Logistic regression**, at 0.7547 on avg precision, against 0.0017 for the best baseline, Rank by amount.

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

**Logistic regression** reaches an average precision of 0.7547 on the unseen window, against 0.0012 for the do-nothing baseline.

Operationally: reviewing the 100 most suspicious transactions catches **88 of 125 frauds** (70.4%), at a precision of 88.0% — meaning roughly 88 in every 100 alerts is genuine. Catching half of all fraud takes 64 alerts.

Note the accuracy column in the leaderboard. Every model sits above 99.8%, including the one that catches nothing. It separates none of them, which is exactly why it is not the ranking metric.

## What I took away

- On a problem this imbalanced, accuracy cannot tell a useful detector from one that catches nothing.
- The textbook fix for imbalance, heavy positive-class reweighting, damaged the ranking that average precision measures.
- Density-based anomaly detection fails when the anomalies resemble each other and form their own cluster.
- Measuring against a fixed review budget turns a metric into a staffing question an operations team can act on.

## How it is built

The backend is FastAPI on port 8006 and the front end is React on port 5106. Most of the interface — the metrics panel, tournament table, CRISP-DM walkthrough, audit panel and charts — is shared across the portfolio and driven entirely by the training run's artifacts, so nothing on screen is typed by hand.

Code: [github.com/aniket-alt/ai-coding-assistant](https://github.com/aniket-alt/ai-coding-assistant/tree/main/06_anomaly_detection)

---

*Figures in this article are generated from the project's artifacts.*
