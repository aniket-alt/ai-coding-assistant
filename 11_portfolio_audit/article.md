# How much should anyone trust this portfolio's leakage audit?

*Portfolio Audit — project 11 of a sixteen-project data science portfolio built with an AI coding assistant, on real data.*

![Portfolio Audit](./screenshots/01_leak_benchmark.png)

## The question

How much should anyone trust this portfolio's leakage audit, and which leaks would it miss?

Every project in this portfolio reports a passing leakage audit. That claim is only as good as the auditor, and an auditor that has never been tested against known leaks proves nothing by passing.

So this project builds a benchmark: 11 copies of a real training script, each with one deliberately planted leak, and 3 clean controls. Both detectors — the static code audit and the runtime assertions — are run against every copy and scored like any classifier.

Each copy is also run with the assertions switched off, to measure how much its leak inflates the reported score. A missed leak that changes nothing is a curiosity; a missed leak that doubles a headline number is a serious gap.

## The data

The leaks fall into four families, each a mistake that appears in real work:

| Variant | Family | What goes wrong |
|---|---|---|
| Clean template | control | Chronological split with a one-hour gap; scaler fitted on training rows. |
| Clean, scaler inside a pipeline | control | The same model with scaling wrapped in a Pipeline. |
| Clean, unshuffled train_test_split | control | train_test_split with shuffle=False on time-sorted rows is a legitimate chronological split. |
| Scaler fitted on every row | preprocessing | The scaler sees test rows before the split, so test statistics shape training. |
| Scaler fitted on every row, innocuous name | preprocessing | The same leak with the full frame stored under an unremarkable variable name. |
| Scaler fitted on the test set | preprocessing | Scaling parameters are learned from held-out rows. |
| Shuffled split on time-ordered data | split | A random split scatters later transactions into training. |
| Train and test windows overlap | split | The test window starts five hours before training ends. |
| Test rows copied into training | split | A third of the test rows, including their labels, are appended to training. |
| The label used as a feature | target | The fraud flag itself is left among the inputs. |
| A noisy copy of the label | target | A feature built from the label plus noise: correlated, but not perfectly. |
| Rolling fraud rate centred on each row | target | A centred window averages labels from transactions that happen later. |
| Target encoding computed on all rows | target | The fraud rate per amount band is computed before the split, test labels included. |
| Hyperparameter chosen on the test set | evaluation | Several models are scored on the test set and only the best score is reported. |

The controls matter as much as the leaks. A detector that fires on a legitimate unshuffled split or a scaler inside a pipeline would bury real warnings in false alarms.

Every variant runs on the real ULB credit card data (284,807 transactions). The leaks are constructed; the data is not.

## What won

**Static + runtime**, at 0.727 on balanced accuracy, against 0.500 for the best baseline, Flag everything.

| # | Detector | Balanced accuracy | Leaks caught | False alarms | Damage caught |
|---|---|---|---|---|---|
| 1 | **Static + runtime** 🏆 | 0.727 | 45.5% | 0.0% | 60.4% |
| 2 | Runtime assertions | 0.682 | 36.4% | 0.0% | 60.4% |
| 3 | Static code audit | 0.591 | 18.2% | 0.0% | 9.4% |
| 4 | Flag everything *(baseline)* | 0.500 | 100.0% | 100.0% | 100.0% |
| 5 | Flag nothing *(baseline)* | 0.500 | 0.0% | 0.0% | 0.0% |

Ranked by balanced accuracy (higher is better). Split: planted-leak benchmark; 14 evaluation rows. Generated from `artifacts/metrics.json`.

Together the two detectors catch **5 of 11 planted leaks** with 0% false alarms on clean scripts. Weighted by how much each leak inflates the score, they catch 60% of the damage.

**What they miss matters.** The costliest undetected leak is *a noisy copy of the label*, which moves average precision from 0.7017 to 0.8602. It passes because no feature is perfectly correlated with the label and the split itself is clean — the leak lives in how a feature was computed, which neither detector inspects.

The consequence for the rest of this portfolio is concrete: a passing audit rules out split mistakes and blatant label leaks, and says little about target encodings, label-derived features or tuning on the test set. Those are guarded in each project by design — out-of-fold encoders, walk-forward refits, baselines — not by the auditor, and should be read that way.

## What I took away

- An auditor that has never been tested against planted leaks proves nothing by passing.
- The costliest missed leak was a feature built from the label with noise added, which no correlation threshold near one can catch.
- Name-based static rules miss the same leak once a variable is renamed.
- Some leaks lower a score rather than raise it, so inflation alone is not a safe proxy for whether a leak exists.
- A passing audit in this portfolio rules out split mistakes and blatant label leaks; the subtler cases are guarded by design, not by the auditor.

## How it is built

The backend is FastAPI on port 8011 and the front end is React on port 5111. Most of the interface — the metrics panel, tournament table, CRISP-DM walkthrough, audit panel and charts — is shared across the portfolio and driven entirely by the training run's artifacts, so nothing on screen is typed by hand.

Code: [github.com/aniket-alt/ai-coding-assistant](https://github.com/aniket-alt/ai-coding-assistant/tree/main/11_portfolio_audit)

---

*Figures in this article are generated from the project's artifacts.*
