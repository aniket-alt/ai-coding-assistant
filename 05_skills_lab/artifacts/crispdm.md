# CRISP-DM Report — Data Science Skills Lab

**Business question.** For each technique in a working data scientist's catalogue, does doing it the careless way change the result -- and by how much, on real data?


## 1. Business Understanding

*Objective:* Turn a catalogue of techniques into a catalogue of measured differences.


A skills catalogue usually lists techniques with advice about pitfalls. Advice is cheap: it cannot say whether a pitfall costs anything on the data in front of you. Each entry here runs the careless version and the careful version of the same analysis, over many seeds, and reports the paired difference with an interval. Entries whose interval covers zero say that plainly, which is how the catalogue stays worth reading.


| Finding | Value | Detail |
|---|---|---|
| Skills | `12` |  |
| Seeds per skill | `25` |  |


*Code:* `ml/skills.py:Skill.measure`


> **Phase gate: PASSED.** Every entry must execute on a real dataset and report a paired interval.


## 2. Data Understanding

*Objective:* Choose datasets small enough to re-run live and real enough to be worth measuring.


Four datasets from OpenML: **Titanic** (1,309 rows, 38% positive, 20% of rows with a missing value); **German Credit** (1,000 rows, 30% positive); **Pima Indians Diabetes** (768 rows, 35% positive); **Blood Transfusion Service Center** (748 rows, 24% positive). Each was chosen for a property a skill needs: missing values, categorical columns with rare levels, an imbalanced target, and passengers who travelled in families.


| Finding | Value | Detail |
|---|---|---|
| Datasets | `4` |  |
| Smallest | `748 rows` |  |


*Code:* `ml/data.py:load`


> **Phase gate: PASSED.** Provenance recorded for each; an assertion rejects synthetic data.


## 3. Data Preparation

*Objective:* Give every skill the same shape so they can be compared and browsed.


Each entry declares its dataset, its metric, what the careless and careful versions are, and a function returning both numbers for one seed. The catalogue runs that function over the seeds, pairs the results so that split-to-split variation cancels, and bootstraps the paired differences. Nothing in an entry's text is written by hand except its explanation and its pitfalls.


| Finding | Value | Detail |
|---|---|---|
| Record fields | `careless, careful, gap, interval, verdict` |  |


*Code:* `ml/skills.py:Skill`, `ml/skills.py:CATALOGUE`


> **Phase gate: PASSED.** One record shape for every skill.


## 4. Modeling

*Objective:* Implement both versions of each technique faithfully.


The careless versions are written the way the mistake is usually made: a frame imputed or scaled or encoded before any split, a feature selector fitted once, classes balanced as a preparation step, a threshold swept against the held-out labels. The careful versions put the same work inside the fold. Two entries measure something else: whether accuracy or balanced accuracy is the honest reading of the same predictions, and whether calibration improves the probabilities.


| Finding | Value | Detail |
|---|---|---|
| Categories | `Evaluation design, Measurement, Preparation` |  |


*Code:* `ml/skills.py:_target_encode`, `ml/skills.py:_oversample_before_split`


> **Phase gate: PASSED.** 12 skills implemented in both versions.


## 5. Evaluation

*Objective:* Report which differences the experiment can resolve, and which it cannot.


6 of 12 skills show a careless version that flatters the result by more than the seed-to-seed noise. The largest is **reading accuracy on an imbalanced target**, worth +0.221 accuracy against balanced accuracy.

5 entries report no difference this experiment can resolve: imputation fitted before the split, scaling fitted before the split, the same oversampling, with a linear model, splitting rows that belong to the same group, trusting one train/test split. That is a result, not a gap in the catalogue: on these datasets, with these models, the careless version costs nothing measurable.

Every careless version here is a real leak, yet the repository's own static auditor reports 0 warnings and 0 critical findings against this project's source. Reading source cannot see these: each is an ordinary call in an ordinary order, and nothing in the text says which rows will later become a validation fold. Only running both versions tells them apart.


| Finding | Value | Detail |
|---|---|---|
| Skills with a measurable effect | `6 of 12` |  |
| Largest flattery | `+0.221` |  |


*Code:* `ml/train.py:main`


> **Phase gate: PASSED.** Verdicts checked against the intervals by an executed assertion.


## 6. Deployment

*Objective:* Serve the catalogue, and let any entry be re-run live.


The API serves every record and will re-run a chosen skill on a chosen number of seeds, so a reader can watch the interval narrow as seeds are added, and can confirm that the same seeds give the same numbers.


| Finding | Value | Detail |
|---|---|---|
| Serving port | `8,005` |  |


*Code:* `server/main.py`


> **Phase gate: PASSED.** Live re-runs use the same code path as this script.
