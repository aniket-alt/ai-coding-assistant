# Data Science Skills Lab

*A catalogue of techniques, each one measured the careless way and the careful way.*

**Aniket Anil Naik** · Project 05 of the data science full-stack portfolio

## Abstract

**For each technique in a working data scientist's catalogue, does doing it the careless way change the result -- and by how much, on real data?**

We study this question on *Titanic (OpenML 40945)*: 1,309 rows as downloaded, used under Public (OpenML). Following the CRISP-DM process, we compare 12 approaches, 0 of them baselines, and rank them by how much it flatters. Evaluation split: each skill repeats its own design over 25 seeds.

The best approach, **Reading accuracy on an imbalanced target**, reaches 0.2207.

3 of 3 leakage assertions passed while training ran, and static analysis of the source reported 0 critical and 0 warning findings.

**Keywords:** measured comparison of careless and careful practice, per skill, paired comparisons over many seeds, bootstrap intervals on the difference, leakage demonstrated rather than described, live re-runs of any entry


## 1. Introduction

*Objective.* Turn a catalogue of techniques into a catalogue of measured differences.

A skills catalogue usually lists techniques with advice about pitfalls. Advice is cheap: it cannot say whether a pitfall costs anything on the data in front of you. Each entry here runs the careless version and the careful version of the same analysis, over many seeds, and reports the paired difference with an interval. Entries whose interval covers zero say that plainly, which is how the catalogue stays worth reading.

| Finding | Value | Note |
|---|---|---|
| Skills | 12 |  |
| Seeds per skill | 25 |  |

> Phase gate passed: Every entry must execute on a real dataset and report a paired interval.


## 2. Data

*Objective.* Choose datasets small enough to re-run live and real enough to be worth measuring.

Four datasets from OpenML: **Titanic** (1,309 rows, 38% positive, 20% of rows with a missing value); **German Credit** (1,000 rows, 30% positive); **Pima Indians Diabetes** (768 rows, 35% positive); **Blood Transfusion Service Center** (748 rows, 24% positive). Each was chosen for a property a skill needs: missing values, categorical columns with rare levels, an imbalanced target, and passengers who travelled in families.

| Finding | Value | Note |
|---|---|---|
| Datasets | 4 |  |
| Smallest | 748 rows |  |

### Provenance

✅ **Real, third-party data**

| Field | Value |
|---|---|
| Dataset | Titanic (OpenML 40945) |
| Source | https://openml.org/data/v1/download/16826755/Titanic.arff |
| Licence | Public (OpenML) |
| Rows × columns (as downloaded) | 1,309 × 9 |
| SHA-256 | `c56e3785c96cd8aceac85f11bff21c2bf72841426f50903a6f27be8f334a7d03` |

> Phase gate passed: Provenance recorded for each; an assertion rejects synthetic data.


## 3. Method

*Objective.* Give every skill the same shape so they can be compared and browsed.

Each entry declares its dataset, its metric, what the careless and careful versions are, and a function returning both numbers for one seed. The catalogue runs that function over the seeds, pairs the results so that split-to-split variation cancels, and bootstraps the paired differences. Nothing in an entry's text is written by hand except its explanation and its pitfalls.

| Finding | Value | Note |
|---|---|---|
| Record fields | careless, careful, gap, interval, verdict |  |

> Phase gate passed: One record shape for every skill.


## 4. Models

*Objective.* Implement both versions of each technique faithfully.

The careless versions are written the way the mistake is usually made: a frame imputed or scaled or encoded before any split, a feature selector fitted once, classes balanced as a preparation step, a threshold swept against the held-out labels. The careful versions put the same work inside the fold. Two entries measure something else: whether accuracy or balanced accuracy is the honest reading of the same predictions, and whether calibration improves the probabilities.

| Finding | Value | Note |
|---|---|---|
| Categories | Evaluation design, Measurement, Preparation |  |

**Table 1.** Tournament results.

| # | Skill | How much it flatters | Careless | Careful | 95% from | to |
|---|---|---|---|---|---|---|
| 1 | **Reading accuracy on an imbalanced target** 🏆 | 0.2207 | 0.7741 | 0.5535 | 0.2155 | 0.2254 |
| 2 | Oversampling before the split | 0.1099 | 0.7792 | 0.6693 | 0.1051 | 0.1148 |
| 3 | Choosing the decision threshold on the test set | 0.0345 | 0.6986 | 0.6640 | 0.0238 | 0.0467 |
| 4 | Target encoding without out-of-fold means | 0.0222 | 0.7844 | 0.7623 | 0.0213 | 0.0230 |
| 5 | Feature selection before the split | 0.0095 | 0.7640 | 0.7545 | 0.0081 | 0.0109 |
| 6 | Reporting the best of several models on the test set | 0.0063 | 0.7866 | 0.7804 | 0.0039 | 0.0089 |
| 7 | Trusting one train/test split | 0.0024 | 0.8328 | 0.8305 | −0.0085 | 0.0131 |
| 8 | The same oversampling, with a linear model | 0.0011 | 0.7529 | 0.7518 | −0.0027 | 0.0044 |
| 9 | Imputation fitted before the split | 0.0000 | 0.8398 | 0.8398 | −0.0000 | 0.0000 |
| 10 | Scaling fitted before the split | −0.0000 | 0.8305 | 0.8305 | −0.0000 | 0.0000 |
| 11 | Splitting rows that belong to the same group | −0.0095 | 0.8342 | 0.8436 | −0.0241 | 0.0059 |
| 12 | Calibrating predicted probabilities | −0.0114 | 0.1776 | 0.1662 | −0.0144 | −0.0083 |

Ranked by how much it flatters (higher is better). Split: each skill repeats its own design over 25 seeds. Generated from `artifacts/metrics.json`.

> Phase gate passed: 12 skills implemented in both versions.


## 5. Results

*Objective.* Report which differences the experiment can resolve, and which it cannot.

6 of 12 skills show a careless version that flatters the result by more than the seed-to-seed noise. The largest is **reading accuracy on an imbalanced target**, worth +0.221 accuracy against balanced accuracy.

5 entries report no difference this experiment can resolve: imputation fitted before the split, scaling fitted before the split, the same oversampling, with a linear model, splitting rows that belong to the same group, trusting one train/test split. That is a result, not a gap in the catalogue: on these datasets, with these models, the careless version costs nothing measurable.

Every careless version here is a real leak, yet the repository's own static auditor reports 0 warnings and 0 critical findings against this project's source. Reading source cannot see these: each is an ordinary call in an ordinary order, and nothing in the text says which rows will later become a validation fold. Only running both versions tells them apart.

| Finding | Value | Note |
|---|---|---|
| Skills with a measurable effect | 6 of 12 |  |
| Largest flattery | +0.221 |  |

> Phase gate passed: Verdicts checked against the intervals by an executed assertion.

## 6. Threats to validity

### Assertions executed during training

| Assertion | Result | What it measured |
|---|---|---|
| `assert_every_dataset_is_real` | ✅ passed | 4 datasets downloaded and checksummed: titanic 1,309 rows, sha256 c56e3785; credit_g 1,000 rows, sha256 768be84e; diabetes 768 rows, sha256 4eddd5b2; blood 748 rows, sha256 ee1304ca. Marked synthetic: none. |
| `assert_every_skill_paired_on_the_same_seeds` | ✅ passed | 12 skills, each run on the same 25 seeds, careless and careful on every seed; seed counts seen: [25]. |
| `assert_claims_match_intervals` | ✅ passed | 7 of 12 skills have a bootstrap interval that excludes zero and may state an effect; entries claiming one without it: none. |

Recorded by the assertion ledger while training ran; generated from `artifacts/metrics.json`.

### Limitations

- Every result is specific to its dataset, its model and its metric; the same mistake can cost far more or far less elsewhere.
- Twenty-five seeds resolve differences of roughly a hundredth in AUC, and no more; smaller effects are reported as unresolved rather than absent.
- The datasets are small, so a few hundred rows decide each estimate.
- Titanic, German Credit and Pima Diabetes carry the biases of their collection, and German Credit in particular encodes judgements about applicants that should not be read as ground truth about people.


## 7. Deployment

*Objective.* Serve the catalogue, and let any entry be re-run live.

The API serves every record and will re-run a chosen skill on a chosen number of seeds, so a reader can watch the interval narrow as seeds are added, and can confirm that the same seeds give the same numbers.

| Finding | Value | Note |
|---|---|---|
| Serving port | 8,005 |  |

> Phase gate passed: Live re-runs use the same code path as this script.

## 8. Conclusion

_None recorded._

## Reproducibility

Trained 2026-10-24T19:55:59Z in 77.8 s at commit `4b4031a` with Python 3.11.6, NumPy 2.4.6, pandas 3.0.5 and scikit-learn 1.9.1.

```bash
python ml/train.py
python -m uvicorn main:app --app-dir server --port 8005
npm run dev -w 05_skills_lab/client
```

## References

1. Titanic [Dataset]. OpenML, dataset 40945. https://www.openml.org/d/40945
2. Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in data mining: Formulation, detection, and avoidance. ACM Transactions on Knowledge Discovery from Data, 6(4).
3. Varma, S., & Simon, R. (2006). Bias in error estimation when using cross-validation for model selection. BMC Bioinformatics, 7, 91.
4. Ambroise, C., & McLachlan, G. J. (2002). Selection bias in gene extraction on the basis of microarray gene-expression data. PNAS, 99(10), 6562-6566.
5. Micci-Barreca, D. (2001). A preprocessing scheme for high-cardinality categorical attributes in classification and prediction problems. ACM SIGKDD Explorations, 3(1), 27-32.
6. Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. Proceedings of ICML 2005.
7. Vanschoren, J., van Rijn, J. N., Bischl, B., & Torgo, L. (2013). OpenML: Networked science in machine learning. ACM SIGKDD Explorations, 15(2), 49-60.

---

*Generated from this project's artifacts by `scripts/generate_docs.py`.*
