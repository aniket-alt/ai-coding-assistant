# Implementation plans

The status table is generated from what is actually in the repository. The plans
below it are written by hand: for built projects they are a record of what was
done, and for the rest they are intentions, not claims.

<!-- dsfs:status -->
| # | Project | Status | Dataset (planned or used) |
|---|---|---|---|
| 00 | Task Workspace | ⬜ planned | none — full-stack application |
| 01 | [NYC Taxi Trip Duration](./01_nyc_taxi_prediction) | ✅ built | NYC TLC Yellow Taxi, January 2024 |
| 02 | Nano LLM Transformer | ⬜ planned | TinyStories (Hugging Face) |
| 03 | [Customer Segmentation](./03_customer_segmentation) | ✅ built | UCI Online Retail II |
| 04 | [Market Basket Mining](./04_market_basket) | ✅ built | UCI Online Retail II invoices |
| 05 | Data Science Skills Lab | ⬜ planned | several OpenML datasets |
| 06 | [Credit Card Fraud Detection](./06_anomaly_detection) | ✅ built | ULB credit card fraud (OpenML 1597) |
| 07 | AutoML Stacking | ⬜ planned | Adult census (OpenML 1590) |
| 08 | Data Science Visual Foundations | ⬜ planned | none — interactive simulations |
| 09 | FlowForge DAG Engine | ⬜ planned | none — TypeScript engine |
| 10 | CRISP-DM Curriculum | ⬜ planned | Adult census (OpenML 1590) |
| 11 | Portfolio Audit | ⬜ planned | this repository's own artifacts |
| 12 | [Energy Demand Forecasting](./12_energy_forecasting) | ✅ built | UCI Individual Household Electric Power Consumption |
| 13 | MLOps & Governance Platform | ⬜ planned | NYC TLC Yellow Taxi (shared cache with 01) |
| 14 | Multimodal AutoML | ⬜ planned | to be confirmed — needs a public image/text/tabular set |
| 15 | [SPY Probabilistic Forecasting](./15_spy_forecasting) | ✅ built | SPY, VIX, 10-year yield daily, Yahoo Finance |
<!-- /dsfs:status -->

---

## How every project is built

Each project follows the same path, and none is marked built until every step
passes.

1. **Data.** Fetch a real dataset through `dsfs.fetch`, which caches it and
   records a provenance receipt. If no suitable public dataset exists, say so on
   the project page instead of generating data.
2. **Business understanding first.** Write down the target, the metric, and the
   trap in this dataset before fitting any model.
3. **Training.** One `ml/train.py` that runs a tournament including baselines,
   calls the runtime leakage assertions, and writes `metrics.json`,
   `crispdm.json`, `audit.json` and `provenance.json`.
4. **Service.** `server/main.py` on the platform factory, with only the
   project's own endpoints.
5. **Interface.** A client built from `@dsfs/ui`, adding only the views
   specific to the project.
6. **Verification.** `scripts/capture_screenshots.py` must confirm every view
   rendered, and `scripts/generate_docs.py --check` must report no drift.

Built projects link to their [design documents](#built-projects), which are
generated from the code and artifacts and therefore describe what exists.

---

## Built projects

| # | Design document | Paper |
|---|---|---|
| 01 | [NYC Taxi Trip Duration](./01_nyc_taxi_prediction/DESIGN_DOC.md) | [paper](./01_nyc_taxi_prediction/paper.md) |
| 06 | [Credit Card Fraud Detection](./06_anomaly_detection/DESIGN_DOC.md) | [paper](./06_anomaly_detection/paper.md) |
| 15 | [SPY Probabilistic Forecasting](./15_spy_forecasting/DESIGN_DOC.md) | [paper](./15_spy_forecasting/paper.md) |

---

## Planned projects

### 00 · Task Workspace
A full-stack task manager: FastAPI with SQLite, a React client from the shared
design system, natural-language quick-add, and server-sent events for live
updates across tabs. No dataset. The data science angle is an analytics view of
the user's own completion history, with the audit panel omitted because there is
no model to audit.

### 02 · Nano LLM Transformer
A small decoder-only transformer trained from scratch on TinyStories, with
rotary position embeddings, RMSNorm and SwiGLU. The honest angle is the
comparison a demo usually hides: generation quality against a bigram baseline
and against model size, with validation loss curves from a held-out split. Size
is set by the GPU actually available, and the page states it.

### 03 · Customer Segmentation
RFM features from UCI Online Retail II, compared across k-means, Gaussian
mixtures and HDBSCAN. Segments are judged on stability under resampling as well
as silhouette, since a clustering that changes on every rerun is not a
segmentation. `ModelLeaderboard`'s `selectable` prop drives the segment view.

### 04 · Market Basket Mining
Association rules from Online Retail II invoices with FP-Growth. Rules are
evaluated on a later time window to show how many survive, because rules mined
and scored on the same baskets always look good. This is the project where the
shared metrics and leaderboard components do not fit, and it gets its own rule
table and graph.

### 05 · Data Science Skills Lab
A catalogue of techniques, each demonstrated live on a small OpenML dataset
through one shared result view rather than raw JSON. Every demonstration runs
through the same provenance and assertion machinery as the full projects.

### 07 · AutoML Stacking
AutoGluon on Adult census income, set against a hand-tuned LightGBM, with the
comparison made at equal time budgets. AutoGluon's Windows and Python 3.11
support needs confirming before this is started; if it does not install
cleanly, the project will use a different AutoML library and say why.

### 08 · Data Science Visual Foundations
Interactive simulations for Naive Bayes, the confusion matrix and cost-sensitive
thresholds, derivatives and gradient descent, and the chain rule and
backpropagation, with quizzes. Published as a static page. No dataset.

### 09 · FlowForge DAG Engine
A typed TypeScript workflow engine: topological ordering with cycle detection,
retries, and a visual editor. No dataset. The shared design system provides the
interface.

### 10 · CRISP-DM Curriculum
A guided course through all six phases on Adult census data: exploratory
analysis, clustering, outlier detection, classification with a fairness check
across groups, association rules, and locality-sensitive hashing for
nearest-neighbour search, each with a quiz.

### 11 · Portfolio Audit
An application over this repository's own artifacts: the static audit, the
runtime assertion ledgers and the provenance receipts of every built project,
plus the source-repository comparison already in
[`AUDIT_REPORT.md`](./AUDIT_REPORT.md).

### 12 · Energy Demand Forecasting
Household electricity demand from the UCI Individual Household Electric Power
Consumption dataset, forecast at several horizons with seasonal baselines,
gradient boosting and a quantile model, reusing the walk-forward and
significance machinery from project 15.

### 13 · MLOps & Governance Platform
Built on the NYC TLC cache shared with project 01: model versioning, a
prediction service under load testing, drift monitoring with population
stability, and SHAP explanations, presented for an auditor rather than an
analyst.

### 14 · Multimodal AutoML
A model combining tabular, text and image inputs. The dataset is not chosen yet:
it has to be publicly downloadable without an account and small enough to train
here. If no dataset meets both conditions, this project will be scoped down and
the page will explain why.
