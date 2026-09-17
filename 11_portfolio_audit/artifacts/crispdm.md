# CRISP-DM Report — Portfolio Audit

**Business question.** How much should anyone trust this portfolio's leakage audit, and which leaks would it miss?


## 1. Business Understanding

*Objective:* Measure the auditor instead of trusting it.


Every project in this portfolio reports a passing leakage audit. That claim is only as good as the auditor, and an auditor that has never been tested against known leaks proves nothing by passing.

So this project builds a benchmark: 11 copies of a real training script, each with one deliberately planted leak, and 3 clean controls. Both detectors — the static code audit and the runtime assertions — are run against every copy and scored like any classifier.

Each copy is also run with the assertions switched off, to measure how much its leak inflates the reported score. A missed leak that changes nothing is a curiosity; a missed leak that doubles a headline number is a serious gap.


| Finding | Value | Detail |
|---|---|---|
| Planted leaks | `11` |  |
| Clean controls | `3` |  |
| Primary metric | `Balanced accuracy` |  |


*Code:* `ml/mutations.py:VARIANTS`, `ml/train.py:score_detector`


> **Phase gate: PASSED.** Leak catalogue, controls and scoring fixed before any detector was run.


## 2. Data Understanding

*Objective:* Build a catalogue of leaks that covers the ways real projects go wrong.


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


| Finding | Value | Detail |
|---|---|---|
| Variants | `14` |  |
| Leak families | `4` |  |
| Transactions per run | `284,807` |  |


*Code:* `ml/mutations.py:TEMPLATE`


> **Phase gate: PASSED.** 14 variants across 4 leak families plus controls.


## 3. Data Preparation

*Objective:* Establish a clean reference score and confirm the controls really are clean.


The clean template scores an average precision of **0.7017** on its held-out window. Each control stays within 0.0036 of it and trips no runtime check, which is the evidence that the controls are genuinely leak-free rather than assumed to be.

Every variant is executed twice. With the assertions live, a leak is detected if one of them raises. With the assertions disabled, the variant runs to completion and its score is recorded, so the inflation each leak causes can be measured against the template.

The variants' own assertions are kept out of this project's audit ledger, so the benchmark cannot be mistaken for evidence about this project.


| Finding | Value | Detail |
|---|---|---|
| Clean average precision | `0.7017` |  |
| Largest control drift | `0.0036` |  |


*Code:* `ml/train.py:execute`, `ml/train.py:LiveChecks`


> **Phase gate: PASSED.** Controls verified clean by score and by assertion.


## 4. Modeling

*Objective:* Score each detector, and both together, against two baselines.


Flagging everything finds every leak and condemns every clean script; flagging nothing does the reverse. Both score exactly 0.5 balanced accuracy, which is the floor a real detector has to clear.

The two detectors look for different things. The static audit reads code for patterns such as a fit on a variable named like test data. The runtime assertions inspect what the code actually produced: whether the splits overlap, whether time runs forward, whether any feature is nearly the label.


| Finding | Value | Detail |
|---|---|---|
| Best detector | `Static + runtime` |  |
| Leaks caught together | `5 of 11` |  |
| Damage caught together | `60%` |  |


*Code:* `ml/train.py:score_detector`


> **Phase gate: PASSED.** 5 detectors scored.


## 5. Evaluation

*Objective:* State plainly what the audit can and cannot see.


Together the two detectors catch **5 of 11 planted leaks** with 0% false alarms on clean scripts. Weighted by how much each leak inflates the score, they catch 60% of the damage.

**What they miss matters.** The costliest undetected leak is *a noisy copy of the label*, which moves average precision from 0.7017 to 0.8602. It passes because no feature is perfectly correlated with the label and the split itself is clean — the leak lives in how a feature was computed, which neither detector inspects.

The consequence for the rest of this portfolio is concrete: a passing audit rules out split mistakes and blatant label leaks, and says little about target encodings, label-derived features or tuning on the test set. Those are guarded in each project by design — out-of-fold encoders, walk-forward refits, baselines — not by the auditor, and should be read that way.


| Finding | Value | Detail |
|---|---|---|
| Leaks missed | `6` |  |
| Costliest miss | `A noisy copy of the label` |  |
| False alarms on controls | `0%` |  |


*Code:* `ml/mutations.py`, `platform/dsfs/leakage.py`


> **Phase gate: PASSED.** The auditor's blind spots are published with its results.


## 6. Deployment

*Objective:* Publish the benchmark beside the portfolio's audit results.


`GET /api/benchmark` returns every variant with both detectors' verdicts and its score inflation, `GET /api/corpus/{id}` returns a variant's source so any reader can check what was planted, and `GET /api/portfolio` returns every other project's audit state.


| Finding | Value | Detail |
|---|---|---|
| Serving port | `8,011` |  |


*Code:* `server/main.py`


> **Phase gate: PASSED.** Corpus committed so the benchmark is inspectable.
