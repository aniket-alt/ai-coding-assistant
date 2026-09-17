# CRISP-DM Report — AutoML Stacking

**Business question.** Which clients should a bank call about a term deposit, and does an automated stacked ensemble choose them better than one well-tuned model?


## 1. Business Understanding

*Objective:* Rank clients by their chance of subscribing, using only what is known before the call.


A call centre has more clients than time. A model that ranks clients by their chance of subscribing lets the best prospects be called first, so ranking quality (ROC AUC) is the measure, with average precision beside it.

The second question is about tooling. AutoML systems such as AutoGluon win benchmarks by stacking many models. This project checks whether that holds on a deployment-shaped test, against an ensemble built by hand and a single tuned LightGBM, and whether each method's own validation told the truth about it.


| Finding | Value | Detail |
|---|---|---|
| Target | `client subscribes to a term deposit` |  |
| Primary metric | `ROC AUC` |  |
| Excluded by rule | `duration` |  |


*Code:* `ml/train.py:main`


> **Phase gate: PASSED.** Target, ranking metric and a before-the-call feature rule fixed before modelling.


## 2. Data Understanding

*Objective:* Find what the data's time order and documentation imply for evaluation.


41,188 calls from 2008-05 to 2010-11. The file is in date order but records only the month, so the year is reconstructed from the order.

**The success rate moves with the financial crisis.** 9.1% of calls before 2010 led to a subscription, against **52.1%** in 2010, while the three-month Euribor rate collapsed. A random split would mix these regimes and hide the change a deployed model actually faces.

**`duration` is a documented leak.** It is the length of the call, known only after the client has answered. 96.3% of calls have `pdays` = 999, meaning never contacted before.


| Finding | Value | Detail |
|---|---|---|
| Calls | `41,188` |  |
| Success rate before 2010 | `9.1%` |  |
| Success rate in 2010 | `52.1%` |  |


*Code:* `ml/data.py:load`


> **Phase gate: PASSED.** Time order, regime change and the leaking column documented.


## 3. Data Preparation

*Objective:* Split by time and keep every transformation inside the model pipelines.


Training uses the 39,130 calls from before 2010; the test set is all 2,058 calls made in 2010. `pdays` = 999 becomes an explicit 'previously contacted' indicator, because as a number it would read as a very long gap. Scaling and one-hot encoding are fitted inside each pipeline on training rows only, and every contestant, AutoGluon included, sees the same 19 columns.


| Finding | Value | Detail |
|---|---|---|
| Training calls | `39,130` |  |
| Test calls | `2,058` |  |


*Code:* `ml/features.py:preprocessor`, `ml/data.py:split`


> **Phase gate: PASSED.** Chronological split, leak exclusion and target-correlation checks executed.


## 4. Modeling

*Objective:* Build the same kind of ensemble AutoML builds, by hand, beside a tuned single model.


Five different learners are each bagged over five folds, giving out-of-fold predictions for every training call. Caruana selection then adds, 30 times, whichever model most improves the out-of-fold AUC of the running average; it settled on XGBoost (57%), Logistic regression (23%), LightGBM (20%). A logistic stacker trained on the same out-of-fold predictions is the second ensemble. LightGBM is tuned with 40 Optuna trials of five-fold cross-validation. AutoGluon 1.6.2 ran with `best_quality` for up to 900 seconds and trained 13 models.


| Finding | Value | Detail |
|---|---|---|
| Caruana weights | `XGBoost 57%, Logistic regression 23%, LightGBM 20%` |  |
| Tuning trials | `40` |  |


*Code:* `ml/train.py:bag`, `ml/train.py:caruana`, `ml/autogluon_run.py:main`


> **Phase gate: PASSED.** 8 contestants scored on the same 2010 calls.


## 5. Evaluation

*Objective:* Score every contestant on the deployment year, and check each against its own promise.


**LightGBM, tuned** ranks 2010 clients best, with ROC AUC 0.6969. The hand-built Caruana stack against tuned LightGBM: -0.0172 AUC, 95% interval [-0.0279, -0.0060], **worse**.

**Every method overestimated itself.** Validation drawn at random from 2008-2009 calls expected more than 2010 delivered; the smallest gap was +0.071 (LightGBM, tuned) and the largest +0.141 (Logistic regression). A time-ordered check, training on 2008 and validating on 2009, gave 0.5927 against a random-CV 0.7590; the actual 2010 score was 0.6786.

**The leak.** Adding `duration` lifts LightGBM's 2010 AUC from 0.6786 to 0.7820, a gain that would vanish the moment the model had to score a client before calling them.


| Finding | Value | Detail |
|---|---|---|
| Best ROC AUC | `0.6969` |  |
| Stack vs tuned LightGBM | `worse` |  |
| AUC with the duration leak | `0.7820` |  |


*Code:* `ml/train.py:bootstrap_auc`, `ml/train.py:enter`


> **Phase gate: PASSED.** Paired bootstrap on the test year; believed scores compared with actual.


## 6. Deployment

*Objective:* Serve the stronger of the two servable models, with the stack's parts visible beside it.


The API serves **LightGBM, tuned**, the better of the two contestants that run in this repository's environment on the 2010 calls. AutoGluon needs its own environment, so its results are served from its recorded run rather than live. Every prediction also returns the hand-built stack's answer with each member model's score and Caruana weight, so an ensemble's output can be traced to its parts.


| Finding | Value | Detail |
|---|---|---|
| Served model | `LightGBM, tuned` |  |
| Serving port | `8,007` |  |


*Code:* `server/inference.py:predict`, `server/main.py`


> **Phase gate: PASSED.** Served models persisted with the preprocessing they were trained with.
