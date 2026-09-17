# Does automated stacking choose which clients to call better than one tuned model?

*AutoML Stacking — project 07 of a sixteen-project data science portfolio built with an AI coding assistant, on real data.*

![AutoML Stacking](./screenshots/01_tournament.png)

## The question

Which clients should a bank call about a term deposit, and does an automated stacked ensemble choose them better than one well-tuned model?

A call centre has more clients than time. A model that ranks clients by their chance of subscribing lets the best prospects be called first, so ranking quality (ROC AUC) is the measure, with average precision beside it.

The second question is about tooling. AutoML systems such as AutoGluon win benchmarks by stacking many models. This project checks whether that holds on a deployment-shaped test, against an ensemble built by hand and a single tuned LightGBM, and whether each method's own validation told the truth about it.

## The data

41,188 calls from 2008-05 to 2010-11. The file is in date order but records only the month, so the year is reconstructed from the order.

**The success rate moves with the financial crisis.** 9.1% of calls before 2010 led to a subscription, against **52.1%** in 2010, while the three-month Euribor rate collapsed. A random split would mix these regimes and hide the change a deployed model actually faces.

**`duration` is a documented leak.** It is the length of the call, known only after the client has answered. 96.3% of calls have `pdays` = 999, meaning never contacted before.

## What won

**LightGBM, tuned**, at 0.6969 on roc auc, 2010, against 0.6337 for the best baseline, Previous-campaign outcome rule.

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

**LightGBM, tuned** ranks 2010 clients best, with ROC AUC 0.6969. The hand-built Caruana stack against tuned LightGBM: -0.0172 AUC, 95% interval [-0.0279, -0.0060], **worse**.

**Every method overestimated itself.** Validation drawn at random from 2008-2009 calls expected more than 2010 delivered; the smallest gap was +0.071 (LightGBM, tuned) and the largest +0.141 (Logistic regression). A time-ordered check, training on 2008 and validating on 2009, gave 0.5927 against a random-CV 0.7590; the actual 2010 score was 0.6786.

**The leak.** Adding `duration` lifts LightGBM's 2010 AUC from 0.6786 to 0.7820, a gain that would vanish the moment the model had to score a client before calling them.

## What I took away

- On a test set drawn from a later period, a single tuned gradient-boosting model beat both AutoGluon and a hand-built stack of five models.
- Every method's own random validation overestimated how it would do on the following year; among the boosted and ensemble methods, AutoGluon's estimate was the most optimistic.
- Stacking chooses members by how well they fit the validation folds, which rewards what fitted the past rather than what transfers.
- A documented leak such as call duration can raise a cross-validated AUC dramatically while adding far less on the period that matters.

## How it is built

The backend is FastAPI on port 8007 and the front end is React on port 5107. Most of the interface — the metrics panel, tournament table, CRISP-DM walkthrough, audit panel and charts — is shared across the portfolio and driven entirely by the training run's artifacts, so nothing on screen is typed by hand.

Code: [github.com/aniket-alt/ai-coding-assistant](https://github.com/aniket-alt/ai-coding-assistant/tree/main/07_automl_stacking)

---

*Figures in this article are generated from the project's artifacts.*
