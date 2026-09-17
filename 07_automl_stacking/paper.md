# AutoML Stacking

*Does automated stacking choose which clients to call better than one tuned model?*

**Aniket Anil Naik** · Project 07 of the data science full-stack portfolio

## Abstract

**Which clients should a bank call about a term deposit, and does an automated stacked ensemble choose them better than one well-tuned model?**

We study this question on *Bank Marketing, with social and economic context (UCI)*: 41,188 rows as downloaded, used under Creative Commons Attribution 4.0 (CC BY 4.0). Following the CRISP-DM process, we compare 8 approaches, 2 of them baselines, on 2,058 evaluation rows and rank them by roc auc, 2010. Evaluation split: chronological: calls before 2010 train, calls in 2010 test.

The best approach, **LightGBM, tuned**, reaches 0.6969 against 0.6337 for the strongest baseline (Previous-campaign outcome rule).

3 of 3 leakage assertions passed while training ran, and static analysis of the source reported 0 critical and 0 warning findings.

**Keywords:** binary classification, ranking clients to call, AutoGluon best_quality, Caruana ensemble selection, built by hand, Optuna tuning, paired bootstrap on the deployment year


## 1. Introduction

*Objective.* Rank clients by their chance of subscribing, using only what is known before the call.

A call centre has more clients than time. A model that ranks clients by their chance of subscribing lets the best prospects be called first, so ranking quality (ROC AUC) is the measure, with average precision beside it.

The second question is about tooling. AutoML systems such as AutoGluon win benchmarks by stacking many models. This project checks whether that holds on a deployment-shaped test, against an ensemble built by hand and a single tuned LightGBM, and whether each method's own validation told the truth about it.

| Finding | Value | Note |
|---|---|---|
| Target | client subscribes to a term deposit |  |
| Primary metric | ROC AUC |  |
| Excluded by rule | duration |  |

> Phase gate passed: Target, ranking metric and a before-the-call feature rule fixed before modelling.


## 2. Data

*Objective.* Find what the data's time order and documentation imply for evaluation.

41,188 calls from 2008-05 to 2010-11. The file is in date order but records only the month, so the year is reconstructed from the order.

**The success rate moves with the financial crisis.** 9.1% of calls before 2010 led to a subscription, against **52.1%** in 2010, while the three-month Euribor rate collapsed. A random split would mix these regimes and hide the change a deployed model actually faces.

**`duration` is a documented leak.** It is the length of the call, known only after the client has answered. 96.3% of calls have `pdays` = 999, meaning never contacted before.

| Finding | Value | Note |
|---|---|---|
| Calls | 41,188 |  |
| Success rate before 2010 | 9.1% |  |
| Success rate in 2010 | 52.1% |  |

### Provenance

✅ **Real, third-party data**

| Field | Value |
|---|---|
| Dataset | Bank Marketing, with social and economic context (UCI) |
| Source | https://archive.ics.uci.edu/static/public/222/bank+marketing.zip |
| Licence | Creative Commons Attribution 4.0 (CC BY 4.0) |
| Rows × columns (as downloaded) | 41,188 × 21 |
| SHA-256 | `e0bf5f5de5b846e2f18e9d90606637267d46dfa260e0f17bb12e605db5efbeb4` |

> Phase gate passed: Time order, regime change and the leaking column documented.


## 3. Method

*Objective.* Split by time and keep every transformation inside the model pipelines.

Training uses the 39,130 calls from before 2010; the test set is all 2,058 calls made in 2010. `pdays` = 999 becomes an explicit 'previously contacted' indicator, because as a number it would read as a very long gap. Scaling and one-hot encoding are fitted inside each pipeline on training rows only, and every contestant, AutoGluon included, sees the same 19 columns.

| Finding | Value | Note |
|---|---|---|
| Training calls | 39,130 |  |
| Test calls | 2,058 |  |

> Phase gate passed: Chronological split, leak exclusion and target-correlation checks executed.


## 4. Models

*Objective.* Build the same kind of ensemble AutoML builds, by hand, beside a tuned single model.

Five different learners are each bagged over five folds, giving out-of-fold predictions for every training call. Caruana selection then adds, 30 times, whichever model most improves the out-of-fold AUC of the running average; it settled on XGBoost (57%), Logistic regression (23%), LightGBM (20%). A logistic stacker trained on the same out-of-fold predictions is the second ensemble. LightGBM is tuned with 40 Optuna trials of five-fold cross-validation. AutoGluon 1.6.2 ran with `best_quality` for up to 900 seconds and trained 13 models.

| Finding | Value | Note |
|---|---|---|
| Caruana weights | XGBoost 57%, Logistic regression 23%, LightGBM 20% |  |
| Tuning trials | 40 |  |

**Table 1.** Tournament results.

| # | Contestant | ROC AUC, 2010 | Avg precision | Believed AUC | Overestimate |
|---|---|---|---|---|---|
| 1 | **LightGBM, tuned** 🏆 | 0.6969 | 0.7008 | 0.7679 | 0.071 |
| 2 | Hand-built stack, logistic stacker | 0.6837 | 0.6875 | 0.7623 | 0.079 |
| 3 | AutoGluon, best_quality | 0.6830 | 0.6989 | 0.7697 | 0.087 |
| 4 | Hand-built stack, Caruana selection | 0.6797 | 0.6889 | 0.7630 | 0.083 |
| 5 | LightGBM, defaults | 0.6786 | 0.6728 | 0.7590 | 0.080 |
| 6 | Previous-campaign outcome rule *(baseline)* | 0.6337 | 0.6463 | — | — |
| 7 | Logistic regression | 0.6100 | 0.6324 | 0.7508 | 0.141 |
| 8 | Constant score *(baseline)* | 0.5000 | 0.5214 | 0.5000 | 0.000 |

Ranked by roc auc, 2010 (higher is better). Split: chronological: calls before 2010 train, calls in 2010 test; 2,058 evaluation rows. Generated from `artifacts/metrics.json`.

> Phase gate passed: 8 contestants scored on the same 2010 calls.


## 5. Results

*Objective.* Score every contestant on the deployment year, and check each against its own promise.

**LightGBM, tuned** ranks 2010 clients best, with ROC AUC 0.6969. The hand-built Caruana stack against tuned LightGBM: -0.0172 AUC, 95% interval [-0.0279, -0.0060], **worse**.

**Every method overestimated itself.** Validation drawn at random from 2008-2009 calls expected more than 2010 delivered; the smallest gap was +0.071 (LightGBM, tuned) and the largest +0.141 (Logistic regression). A time-ordered check, training on 2008 and validating on 2009, gave 0.5927 against a random-CV 0.7590; the actual 2010 score was 0.6786.

**The leak.** Adding `duration` lifts LightGBM's 2010 AUC from 0.6786 to 0.7820, a gain that would vanish the moment the model had to score a client before calling them.

| Finding | Value | Note |
|---|---|---|
| Best ROC AUC | 0.6969 |  |
| Stack vs tuned LightGBM | worse |  |
| AUC with the duration leak | 0.7820 |  |

> Phase gate passed: Paired bootstrap on the test year; believed scores compared with actual.

## 6. Threats to validity

### Assertions executed during training

| Assertion | Result | What it measured |
|---|---|---|
| `assert_temporal_split` | ✅ passed | Last training timestamp 2009-12-01 00:00:00 precedes first test timestamp 2010-03-01 00:00:00 by 90 days 00:00:00; no embargo required. |
| `assert_duration_excluded` | ✅ passed | 19 model inputs; `duration` (known only after the call) absent. |
| `assert_no_target_leak` | ✅ passed | 63 numeric features correlated against the target; strongest was emp.var.rate at |r|=0.2634 (threshold 0.995). |

Recorded by the assertion ledger while training ran; generated from `artifacts/metrics.json`.

### Limitations

- The test year is 2,058 calls from a single year of unusual conditions; the ranking may not hold in another period.
- AutoGluon ran without its neural-network models, because PyTorch was not installed in its environment.
- Years are reconstructed from the file's date order, which the dataset's documentation states but does not encode.
- Each contestant was trained once with one seed; run-to-run variation of the tuned and AutoML models was not measured.


## 7. Deployment

*Objective.* Serve the stronger of the two servable models, with the stack's parts visible beside it.

The API serves **LightGBM, tuned**, the better of the two contestants that run in this repository's environment on the 2010 calls. AutoGluon needs its own environment, so its results are served from its recorded run rather than live. Every prediction also returns the hand-built stack's answer with each member model's score and Caruana weight, so an ensemble's output can be traced to its parts.

| Finding | Value | Note |
|---|---|---|
| Served model | LightGBM, tuned |  |
| Serving port | 8,007 |  |

> Phase gate passed: Served models persisted with the preprocessing they were trained with.

## 8. Conclusion

- On a test set drawn from a later period, a single tuned gradient-boosting model beat both AutoGluon and a hand-built stack of five models.
- Every method's own random validation overestimated how it would do on the following year; among the boosted and ensemble methods, AutoGluon's estimate was the most optimistic.
- Stacking chooses members by how well they fit the validation folds, which rewards what fitted the past rather than what transfers.
- A documented leak such as call duration can raise a cross-validated AUC dramatically while adding far less on the period that matters.

## Reproducibility

Trained 2026-09-17T15:48:06Z in 134.1 s at commit `1e3bb26` with Python 3.11.6, NumPy 2.4.6, pandas 3.0.5 and scikit-learn 1.9.1.

```bash
../.venv-automl/Scripts/python ml/autogluon_run.py
python ml/train.py
python -m uvicorn main:app --app-dir server --port 8007
npm run dev -w 07_automl_stacking/client
```

## References

1. Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. Decision Support Systems, 62, 22-31. https://doi.org/10.1016/j.dss.2014.03.001
2. Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. Decision Support Systems, 62, 22–31.
3. Caruana, R., Niculescu-Mizil, A., Crew, G., & Ksikes, A. (2004). Ensemble selection from libraries of models. Proceedings of ICML 2004.
4. Erickson, N. et al. (2020). AutoGluon-Tabular: Robust and accurate AutoML for structured data. arXiv:2003.06505.
5. Wolpert, D. H. (1992). Stacked generalization. Neural Networks, 5(2), 241–259.
6. Akiba, T. et al. (2019). Optuna: A next-generation hyperparameter optimization framework. Proceedings of KDD 2019.

---

*Generated from this project's artifacts by `scripts/generate_docs.py`.*
