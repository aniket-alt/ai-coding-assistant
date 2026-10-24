# A catalogue of techniques, each one measured the careless way and the careful way.

*Data Science Skills Lab — project 05 of a sixteen-project data science portfolio built with an AI coding assistant, on real data.*

![Data Science Skills Lab](./screenshots/01_catalogue.png)

## The question

For each technique in a working data scientist's catalogue, does doing it the careless way change the result -- and by how much, on real data?

A skills catalogue usually lists techniques with advice about pitfalls. Advice is cheap: it cannot say whether a pitfall costs anything on the data in front of you. Each entry here runs the careless version and the careful version of the same analysis, over many seeds, and reports the paired difference with an interval. Entries whose interval covers zero say that plainly, which is how the catalogue stays worth reading.

## The data

Four datasets from OpenML: **Titanic** (1,309 rows, 38% positive, 20% of rows with a missing value); **German Credit** (1,000 rows, 30% positive); **Pima Indians Diabetes** (768 rows, 35% positive); **Blood Transfusion Service Center** (748 rows, 24% positive). Each was chosen for a property a skill needs: missing values, categorical columns with rare levels, an imbalanced target, and passengers who travelled in families.

## What won


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

6 of 12 skills show a careless version that flatters the result by more than the seed-to-seed noise. The largest is **reading accuracy on an imbalanced target**, worth +0.221 accuracy against balanced accuracy.

5 entries report no difference this experiment can resolve: imputation fitted before the split, scaling fitted before the split, the same oversampling, with a linear model, splitting rows that belong to the same group, trusting one train/test split. That is a result, not a gap in the catalogue: on these datasets, with these models, the careless version costs nothing measurable.

Every careless version here is a real leak, yet the repository's own static auditor reports 0 warnings and 0 critical findings against this project's source. Reading source cannot see these: each is an ordinary call in an ordinary order, and nothing in the text says which rows will later become a validation fold. Only running both versions tells them apart.

## What I took away

_None recorded._

## How it is built

The backend is FastAPI on port 8005 and the front end is React on port 5105. Most of the interface — the metrics panel, tournament table, CRISP-DM walkthrough, audit panel and charts — is shared across the portfolio and driven entirely by the training run's artifacts, so nothing on screen is typed by hand.

Code: [github.com/aniket-alt/ai-coding-assistant](https://github.com/aniket-alt/ai-coding-assistant/tree/main/05_skills_lab)

---

*Figures in this article are generated from the project's artifacts.*
