# Portfolio Audit

*How much should anyone trust this portfolio's leakage audit?*

**Aniket Anil Naik** · Project 11 of the data science full-stack portfolio

## Abstract

**How much should anyone trust this portfolio's leakage audit, and which leaks would it miss?**

We study this question on *Credit Card Fraud Detection (ULB)*: 284,807 rows as downloaded, used under Public (Open Database License) — OpenML dataset 1597. Following the CRISP-DM process, we compare 5 approaches, 2 of them baselines, on 14 evaluation rows and rank them by balanced accuracy. Evaluation split: planted-leak benchmark.

The best approach, **Static + runtime**, reaches 0.727 against 0.500 for the strongest baseline (Flag everything).

3 of 3 leakage assertions passed while training ran, and static analysis of the source reported 0 critical and 0 warning findings.

**Keywords:** leakage detector benchmark, planted-leak benchmark, balanced accuracy over F1, score inflation per leak, clean controls verified by score


## 1. Introduction

*Objective.* Measure the auditor instead of trusting it.

Every project in this portfolio reports a passing leakage audit. That claim is only as good as the auditor, and an auditor that has never been tested against known leaks proves nothing by passing.

So this project builds a benchmark: 11 copies of a real training script, each with one deliberately planted leak, and 3 clean controls. Both detectors — the static code audit and the runtime assertions — are run against every copy and scored like any classifier.

Each copy is also run with the assertions switched off, to measure how much its leak inflates the reported score. A missed leak that changes nothing is a curiosity; a missed leak that doubles a headline number is a serious gap.

| Finding | Value | Note |
|---|---|---|
| Planted leaks | 11 |  |
| Clean controls | 3 |  |
| Primary metric | Balanced accuracy |  |

> Phase gate passed: Leak catalogue, controls and scoring fixed before any detector was run.


## 2. Data

*Objective.* Build a catalogue of leaks that covers the ways real projects go wrong.

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

| Finding | Value | Note |
|---|---|---|
| Variants | 14 |  |
| Leak families | 4 |  |
| Transactions per run | 284,807 |  |

### Provenance

✅ **Real, third-party data**

| Field | Value |
|---|---|
| Dataset | Credit Card Fraud Detection (ULB) |
| Source | https://openml.org/data/v1/download/1673544/creditcard.arff |
| Licence | Public (Open Database License) — OpenML dataset 1597 |
| Rows × columns (as downloaded) | 284,807 × 31 |
| SHA-256 | `fdaf12730dc1fc426f318b71349f24f5c5fd00aa1152940be7e7509ae3d89d2a` |
| Benchmark | leaks are planted deliberately by this project; the data itself is real |
| Variants | 14 |

> Phase gate passed: 14 variants across 4 leak families plus controls.


## 3. Method

*Objective.* Establish a clean reference score and confirm the controls really are clean.

The clean template scores an average precision of **0.7017** on its held-out window. Each control stays within 0.0036 of it and trips no runtime check, which is the evidence that the controls are genuinely leak-free rather than assumed to be.

Every variant is executed twice. With the assertions live, a leak is detected if one of them raises. With the assertions disabled, the variant runs to completion and its score is recorded, so the inflation each leak causes can be measured against the template.

The variants' own assertions are kept out of this project's audit ledger, so the benchmark cannot be mistaken for evidence about this project.

| Finding | Value | Note |
|---|---|---|
| Clean average precision | 0.7017 |  |
| Largest control drift | 0.0036 |  |

> Phase gate passed: Controls verified clean by score and by assertion.


## 4. Models

*Objective.* Score each detector, and both together, against two baselines.

Flagging everything finds every leak and condemns every clean script; flagging nothing does the reverse. Both score exactly 0.5 balanced accuracy, which is the floor a real detector has to clear.

The two detectors look for different things. The static audit reads code for patterns such as a fit on a variable named like test data. The runtime assertions inspect what the code actually produced: whether the splits overlap, whether time runs forward, whether any feature is nearly the label.

| Finding | Value | Note |
|---|---|---|
| Best detector | Static + runtime |  |
| Leaks caught together | 5 of 11 |  |
| Damage caught together | 60% |  |

**Table 1.** Tournament results.

| # | Detector | Balanced accuracy | Leaks caught | False alarms | Damage caught |
|---|---|---|---|---|---|
| 1 | **Static + runtime** 🏆 | 0.727 | 45.5% | 0.0% | 60.4% |
| 2 | Runtime assertions | 0.682 | 36.4% | 0.0% | 60.4% |
| 3 | Static code audit | 0.591 | 18.2% | 0.0% | 9.4% |
| 4 | Flag everything *(baseline)* | 0.500 | 100.0% | 100.0% | 100.0% |
| 5 | Flag nothing *(baseline)* | 0.500 | 0.0% | 0.0% | 0.0% |

Ranked by balanced accuracy (higher is better). Split: planted-leak benchmark; 14 evaluation rows. Generated from `artifacts/metrics.json`.

> Phase gate passed: 5 detectors scored.


## 5. Results

*Objective.* State plainly what the audit can and cannot see.

Together the two detectors catch **5 of 11 planted leaks** with 0% false alarms on clean scripts. Weighted by how much each leak inflates the score, they catch 60% of the damage.

**What they miss matters.** The costliest undetected leak is *a noisy copy of the label*, which moves average precision from 0.7017 to 0.8602. It passes because no feature is perfectly correlated with the label and the split itself is clean — the leak lives in how a feature was computed, which neither detector inspects.

The consequence for the rest of this portfolio is concrete: a passing audit rules out split mistakes and blatant label leaks, and says little about target encodings, label-derived features or tuning on the test set. Those are guarded in each project by design — out-of-fold encoders, walk-forward refits, baselines — not by the auditor, and should be read that way.

| Finding | Value | Note |
|---|---|---|
| Leaks missed | 6 |  |
| Costliest miss | A noisy copy of the label |  |
| False alarms on controls | 0% |  |

> Phase gate passed: The auditor's blind spots are published with its results.

## 6. Threats to validity

### Assertions executed during training

| Assertion | Result | What it measured |
|---|---|---|
| `assert_leaks_differ_from_template` | ✅ passed | 11 leak variants compared with the clean template; 0 identical. |
| `assert_all_variants_executed` | ✅ passed | 14 variants executed; 0 failed for reasons other than a leakage check. |
| `assert_controls_are_clean` | ✅ passed | 3 controls scored within 0.0036 average precision of the template and raised 0 runtime checks. |

Recorded by the assertion ledger while training ran; generated from `artifacts/metrics.json`.

### Limitations

- The corpus holds one example of each leak on one dataset and one model; detection could differ on other code shapes.
- Score inflation is measured on a single split, so small inflations are within run-to-run noise.
- The portfolio snapshot reflects artifacts at the moment this project was trained, not live state.


## 7. Deployment

*Objective.* Publish the benchmark beside the portfolio's audit results.

`GET /api/benchmark` returns every variant with both detectors' verdicts and its score inflation, `GET /api/corpus/{id}` returns a variant's source so any reader can check what was planted, and `GET /api/portfolio` returns every other project's audit state.

| Finding | Value | Note |
|---|---|---|
| Serving port | 8,011 |  |

> Phase gate passed: Corpus committed so the benchmark is inspectable.

## 8. Conclusion

- An auditor that has never been tested against planted leaks proves nothing by passing.
- The costliest missed leak was a feature built from the label with noise added, which no correlation threshold near one can catch.
- Name-based static rules miss the same leak once a variable is renamed.
- Some leaks lower a score rather than raise it, so inflation alone is not a safe proxy for whether a leak exists.
- A passing audit in this portfolio rules out split mistakes and blatant label leaks; the subtler cases are guarded by design, not by the auditor.

## Reproducibility

Trained 2026-09-17T05:11:28Z in 34.3 s at commit `b6a62b4` with Python 3.11.6, NumPy 2.4.6, pandas 3.0.5 and scikit-learn 1.9.1.

```bash
python ml/train.py
python -m uvicorn main:app --app-dir server --port 8011
npm run dev -w 11_portfolio_audit/client
```

## References

1. Dal Pozzolo, A., Caelen, O., Johnson, R. A., & Bontempi, G. (2015). Calibrating probability with undersampling for unbalanced classification. IEEE Symposium Series on Computational Intelligence. Université Libre de Bruxelles, MLG.
2. Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in data mining: Formulation, detection, and avoidance. ACM Transactions on Knowledge Discovery from Data, 6(4).
3. Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. Patterns, 4(9).
4. Jia, Y., & Harman, M. (2011). An analysis and survey of the development of mutation testing. IEEE Transactions on Software Engineering, 37(5), 649–678.
5. Brodersen, K. H., Ong, C. S., Stephan, K. E., & Buhmann, J. M. (2010). The balanced accuracy and its posterior distribution. International Conference on Pattern Recognition.

---

*Generated from this project's artifacts by `scripts/generate_docs.py`.*
