# Data science skills

Each skill links to the line where it is implemented. Links are located by searching the source for the named symbol, so they do not go stale when code moves.

| Skill | Where | Why it matters here |
|---|---|---|
| **Dataset provenance receipts** | [`platform/dsfs/provenance.py`](./platform/dsfs/provenance.py#L22) | Every dataset records its source, licence, checksum and shape, and synthetic data has to declare itself. |
| **Content-addressed data cache** | [`platform/dsfs/datasets.py`](./platform/dsfs/datasets.py#L35) | Downloads once, checksums the bytes, and makes reruns reproducible offline. |
| **CRISP-DM as a data structure** | [`platform/dsfs/crispdm.py`](./platform/dsfs/crispdm.py#L75) | Training fills the report as it runs, so the write-up cannot drift from what happened. |
| **Runtime leakage assertions with a ledger** | [`platform/dsfs/leakage.py`](./platform/dsfs/leakage.py#L99) | Each check records what it measured; deleting a check deletes its evidence. |
| **Static leakage audit (AST)** | [`platform/dsfs/leakage.py`](./platform/dsfs/leakage.py#L196) | Flags fitting on held-out data, shuffled time splits and unseeded estimators. |
| **Temporal split with embargo** | [`platform/dsfs/leakage.py`](./platform/dsfs/leakage.py#L350) | Guarantees training ends before testing starts, with a gap against boundary bleed. |
| **Generated documentation with drift check** | [`platform/dsfs/docs.py`](./platform/dsfs/docs.py#L982) | Papers, abstracts and result tables are regenerated from artifacts; CI fails if they drift. |
| **Rendered-output verification** | [`scripts/capture_screenshots.py`](./scripts/capture_screenshots.py#L44) | Screenshots are only saved as evidence after the view is confirmed to show its expected content. |
| **Excluding post-event fields** | [`01_nyc_taxi_prediction/ml/data.py`](./01_nyc_taxi_prediction/ml/data.py#L71) | Meter distance, fare and arrival time are unknowable at booking, so the model may not use them. |
| **Data cleaning with an audit trail** | [`01_nyc_taxi_prediction/ml/data.py`](./01_nyc_taxi_prediction/ml/data.py#L111) | Each filter records how many rows it removed and why. |
| **Missingness as a signal** | [`01_nyc_taxi_prediction/ml/data.py`](./01_nyc_taxi_prediction/ml/data.py#L179) | Missing fields cluster by vendor, so the gap itself is kept as a feature. |
| **Out-of-fold smoothed target encoding** | [`01_nyc_taxi_prediction/ml/features.py`](./01_nyc_taxi_prediction/ml/features.py#L133) | No row contributes to its own encoding, and rare categories shrink toward the mean. |
| **Cyclical time features** | [`01_nyc_taxi_prediction/ml/features.py`](./01_nyc_taxi_prediction/ml/features.py#L31) | 23:00 and 00:00 end up as neighbours rather than opposite ends of a scale. |
| **Polygon simplification** | [`01_nyc_taxi_prediction/ml/build_zone_map.py`](./01_nyc_taxi_prediction/ml/build_zone_map.py#L43) | Ramer-Douglas-Peucker turns the official shapefile into a small SVG map. |
| **Evaluation under extreme imbalance** | [`06_anomaly_detection/ml/train.py`](./06_anomaly_detection/ml/train.py#L52) | Average precision and review-budget recall replace accuracy, which a do-nothing model maximises. |
| **Operational metric: review budget** | [`06_anomaly_detection/ml/train.py`](./06_anomaly_detection/ml/train.py#L49) | Reports how much fraud a fixed-size analyst team would actually catch. |
| **Unsupervised anomaly detection** | [`06_anomaly_detection/ml/train.py`](./06_anomaly_detection/ml/train.py#L286) | Detectors that never see labels are scored against labels, alongside supervised models. |
| **Serving the model that actually won** | [`06_anomaly_detection/ml/train.py`](./06_anomaly_detection/ml/train.py#L461) | The API loads the tournament winner rather than a hardcoded estimator. |
| **Forecasting returns, not prices** | [`15_spy_forecasting/ml/data.py`](./15_spy_forecasting/ml/data.py#L52) | Price levels give a near-perfect R² for free; returns remove the illusion. |
| **Empirical lookahead verification** | [`15_spy_forecasting/ml/features.py`](./15_spy_forecasting/ml/features.py#L98) | Features are recomputed on truncated history and must match, proving they never read the future. |
| **Range-based volatility estimation** | [`15_spy_forecasting/ml/features.py`](./15_spy_forecasting/ml/features.py#L48) | The Parkinson estimator uses the day's high and low, not just closes. |
| **Quantile regression and pinball loss** | [`platform/dsfs/forecasting.py`](./platform/dsfs/forecasting.py#L16) | Scores a whole predictive distribution rather than a single point. |
| **Walk-forward validation with purging** | [`15_spy_forecasting/ml/train.py`](./15_spy_forecasting/ml/train.py#L124) | Repeated out-of-sample refits, dropping training rows whose targets overlap the test block. |
| **Paired bootstrap significance testing** | [`platform/dsfs/forecasting.py`](./platform/dsfs/forecasting.py#L45) | Asks whether a winner beats a random walk or just got lucky. |
| **Backtesting with costs against buy-and-hold** | [`15_spy_forecasting/ml/train.py`](./15_spy_forecasting/ml/train.py#L85) | A strategy is always compared with simply holding, net of execution cost. |
| **Fat-tail-aware intervals** | [`15_spy_forecasting/ml/train.py`](./15_spy_forecasting/ml/train.py#L409) | Student-t quantiles over EWMA volatility, because returns have heavy tails. |
| **Shared, audited data cleaning** | [`datasets/loaders/online_retail.py`](./datasets/loaders/online_retail.py#L73) | One loader for two projects, recording what each rule removed, including a nine-day overlap between source sheets. |
| **Judging clusters on future behaviour** | [`03_customer_segmentation/ml/features.py`](./03_customer_segmentation/ml/features.py#L16) | Segments are built from one year and scored on the next, so they have to predict something they never saw. |
| **Omega-squared effect size** | [`03_customer_segmentation/ml/train.py`](./03_customer_segmentation/ml/train.py#L47) | Corrects variance explained for the number of segments, so more clusters are not rewarded by default. |
| **Clustering stability by subsampling** | [`03_customer_segmentation/ml/train.py`](./03_customer_segmentation/ml/train.py#L67) | Rebuilds each segmentation on overlapping subsamples and compares assignments with the adjusted Rand index. |
| **RFM scoring as a baseline** | [`03_customer_segmentation/ml/train.py`](./03_customer_segmentation/ml/train.py#L116) | The classic marketing heuristic every learned segmentation has to beat. |
| **Held-out item recommendation test** | [`04_market_basket/ml/baskets.py`](./04_market_basket/ml/baskets.py#L53) | Hides one product from each later basket so every method, rules included, is judged on the same task. |
| **FP-Growth association rules** | [`04_market_basket/ml/baskets.py`](./04_market_basket/ml/baskets.py#L144) | Frequent itemsets mined after pruning items below support, turned into single-consequent rules. |
| **Item-to-item co-occurrence** | [`04_market_basket/ml/baskets.py`](./04_market_basket/ml/baskets.py#L107) | A sparse co-occurrence matrix as a threshold-free baseline, which won the tournament. |
| **Rule survival across seasons** | [`04_market_basket/ml/baskets.py`](./04_market_basket/ml/baskets.py#L236) | Recomputes every mined rule on later baskets to see whether its association still holds. |
| **Custom checks in the audit trail** | [`platform/dsfs/leakage.py`](./platform/dsfs/leakage.py#L234) | Project-specific leakage rules recorded in the same ledger as the standard assertions. |
| **Deterministic tie-breaking** | [`04_market_basket/ml/baskets.py`](./04_market_basket/ml/baskets.py#L39) | Explicit ordering so results do not change with Python's randomised string hashing. |
| **Minimum lag equal to the forecast horizon** | [`12_energy_forecasting/ml/features.py`](./12_energy_forecasting/ml/features.py#L10) | Every feature must exist when a day-ahead forecast is issued; enforced by an assertion in training. |
| **Measuring the cost of a leak** | [`12_energy_forecasting/ml/train.py`](./12_energy_forecasting/ml/train.py#L292) | Scores a deliberately leaky model the same way and reports how much it would have flattered the result. |
| **Missing data without interpolation** | [`12_energy_forecasting/ml/data.py`](./12_energy_forecasting/ml/data.py#L48) | Hours with too few readings stay missing and are excluded from training and scoring. |
| **Interval calibration** | [`platform/dsfs/forecasting.py`](./platform/dsfs/forecasting.py#L32) | Checks whether an 80% band actually contains 80% of outcomes, shared by both forecasting projects. |
| **Benchmarking a leakage detector** | [`11_portfolio_audit/ml/mutations.py`](./11_portfolio_audit/ml/mutations.py#L89) | Plants known leaks in copies of a real script so the auditor can be measured rather than trusted. |
| **Balanced accuracy for a skewed benchmark** | [`11_portfolio_audit/ml/train.py`](./11_portfolio_audit/ml/train.py#L65) | Stops a detector that flags everything from looking good on a corpus with more leaks than controls. |
| **Measuring score inflation from leakage** | [`11_portfolio_audit/ml/train.py`](./11_portfolio_audit/ml/train.py#L48) | Runs each variant with assertions off to see how much its leak changes the published number. |
| **Population stability index for drift monitoring** | [`13_mlops_governance/ml/train.py`](./13_mlops_governance/ml/train.py#L40) | Bins a column on the reference data's quantiles and compares shares, with a floor so empty bins do not divide by zero. |
| **Evaluating a retraining policy on unseen months** | [`13_mlops_governance/ml/train.py`](./13_mlops_governance/ml/train.py#L197) | Lets each policy choose training months from those already finished, then scores it on the next month, as a deployed model would be. |
| **Drift-triggered retraining** | [`13_mlops_governance/ml/train.py`](./13_mlops_governance/ml/train.py#L227) | Retrains only when the latest finished month drifts past the PSI threshold, so the alarm's value can be measured against simpler policies. |
| **Load testing a FastAPI model server** | [`13_mlops_governance/ml/load_test.py`](./13_mlops_governance/ml/load_test.py#L69) | Starts the real API, warms it up and records throughput and latency percentiles at several concurrency levels. |
| **Per-prediction TreeSHAP contributions** | [`13_mlops_governance/server/inference.py`](./13_mlops_governance/server/inference.py#L56) | Uses LightGBM's pred_contrib so every served estimate comes with contributions that sum exactly to it. |
| **Group fairness metrics at a threshold** | [`10_crispdm_curriculum/ml/train.py`](./10_crispdm_curriculum/ml/train.py#L51) | Reports disparate impact, equal-opportunity and false-positive gaps and selection rates, so accuracy is never read alone. |
| **Testing fairness through unawareness** | [`10_crispdm_curriculum/ml/train.py`](./10_crispdm_curriculum/ml/train.py#L238) | Trains each model with and without sensitive attributes and asserts the unaware version truly excludes them. |
| **Random-hyperplane LSH** | [`10_crispdm_curriculum/ml/train.py`](./10_crispdm_curriculum/ml/train.py#L67) | Approximate nearest-neighbour search whose recall and scanned fraction are measured against exact search. |
| **Quiz answers checked on the server** | [`10_crispdm_curriculum/server/inference.py`](./10_crispdm_curriculum/server/inference.py#L68) | Questions are served without answers, and explanations quote figures from the training run. |
| **Caruana ensemble selection** | [`07_automl_stacking/ml/train.py`](./07_automl_stacking/ml/train.py#L88) | Greedy forward selection with replacement over out-of-fold predictions, with deterministic tie-breaking and a recorded trajectory. |
| **Bagged out-of-fold predictions for stacking** | [`07_automl_stacking/ml/train.py`](./07_automl_stacking/ml/train.py#L70) | Gives every training row a prediction from a model that never saw it, and averages the fold models for the test set. |
| **Paired bootstrap of an AUC difference** | [`07_automl_stacking/ml/train.py`](./07_automl_stacking/ml/train.py#L114) | Resamples the same test rows for both models, so a verdict reflects the models and not which rows were drawn. |
| **Running a conflicting library in its own environment** | [`07_automl_stacking/ml/autogluon_run.py`](./07_automl_stacking/ml/autogluon_run.py#L37) | AutoGluon hands over test probabilities as a file, checked against the shared test set, instead of changing the shared environment. |
| **Reconstructing years from date-ordered rows** | [`07_automl_stacking/ml/data.py`](./07_automl_stacking/ml/data.py#L59) | Recovers a year the file does not store, so a chronological split is possible. |
| **Naive Bayes written out and matched to a library** | [`08_visual_foundations/ml/lessons.py`](./08_visual_foundations/ml/lessons.py#L32) | Every word's evidence is visible, and training asserts its probabilities equal scikit-learn's. |
| **Largest stable learning rate for least squares** | [`08_visual_foundations/ml/lessons.py`](./08_visual_foundations/ml/lessons.py#L142) | Computes 2 / largest eigenvalue, the limit beyond which batch gradient descent diverges. |
| **Gradient check by finite differences** | [`08_visual_foundations/ml/lessons.py`](./08_visual_foundations/ml/lessons.py#L201) | Verifies every hand-derived gradient without trusting any framework. |
| **Measuring a duplicate leak across many splits** | [`08_visual_foundations/ml/train.py`](./08_visual_foundations/ml/train.py#L48) | Compares scores with and without deduplication over twenty seeds, so a small effect is not mistaken for noise or vice versa. |
| **Kahn's algorithm with a concrete cycle** | [`09_flowforge_dag/engine/src/graph.ts`](./09_flowforge_dag/engine/src/graph.ts#L58) | Linear-time ordering that, on failure, returns one closed path instead of just 'cycle detected'. |
| **Incremental discrete-event scheduler** | [`09_flowforge_dag/engine/src/executor.ts`](./09_flowforge_dag/engine/src/executor.ts#L41) | Counts open, usable and dead inputs per node so scheduling is O((V + E) log V), with retries, branches and join modes. |
| **Differential testing against a slow reference** | [`09_flowforge_dag/engine/src/properties.ts`](./09_flowforge_dag/engine/src/properties.ts#L172) | Keeps the simple implementation as an oracle; it found a real double-start bug in the optimised one. |
| **Mutation testing a test suite** | [`09_flowforge_dag/ml/train.py`](./09_flowforge_dag/ml/train.py#L81) | Plants realistic bugs in copies of the code and requires the tests to fail on every one. |
| **Compile-time tests with @ts-expect-error** | [`09_flowforge_dag/engine/type-tests/compile-time.ts`](./09_flowforge_dag/engine/type-tests/compile-time.ts#L28) | Branded ids and exhaustive switches are tested: a type change that lets a bad line compile fails the build. |
| **Optimistic concurrency with versions** | [`00_task_workspace/server/store.py`](./00_task_workspace/server/store.py) | An edit names the version it was based on; a stale edit is refused with the current task instead of overwriting it. |
| **Server-sent events with gap detection** | [`00_task_workspace/client/src/useTasks.ts`](./00_task_workspace/client/src/useTasks.ts#L27) | Every event carries a global sequence number, so a tab that misses one reloads instead of drifting out of sync. |
| **A small, testable quick-entry grammar** | [`00_task_workspace/server/parser.py`](./00_task_workspace/server/parser.py#L98) | Resolves relative dates against an explicit day and keeps any marker it cannot parse in the title. |
| **Generated round-trip tests for a parser** | [`00_task_workspace/ml/train.py`](./00_task_workspace/ml/train.py#L77) | Renders known fields into text in random order and requires every one to come back, independently of hand-written cases. |
| **Measuring live-update delivery** | [`00_task_workspace/ml/train.py`](./00_task_workspace/ml/train.py#L160) | Runs the real server with many subscribers and checks completeness, order and latency of every event. |
| **Masking label words to measure a text leak** | [`14_multimodal_automl/ml/data.py`](./14_multimodal_automl/ml/data.py#L77) | Removes every word of any class name, so a text model's score without the answer written in the input can be measured. |
| **Grouped split by a shared key** | [`14_multimodal_automl/ml/train.py`](./14_multimodal_automl/ml/train.py#L75) | Keeps products sharing a display name on one side, checked by an executed assertion across train, validation and test. |
| **Cached CLIP image and text embeddings** | [`14_multimodal_automl/ml/embed.py`](./14_multimodal_automl/ml/embed.py#L55) | Embeds 44,000 images and names once on the GPU and reuses them while the product ids match. |
| **Late fusion with validation-chosen weights** | [`14_multimodal_automl/ml/train.py`](./14_multimodal_automl/ml/train.py#L75) | Weighted geometric mean of per-modality probabilities, with weights searched on validation products only. |
| **Text-to-image search with CLIP** | [`14_multimodal_automl/server/inference.py`](./14_multimodal_automl/server/inference.py#L141) | Ranks product photos by similarity to a typed description, served live. |

## Reproduction skills

Each built project ships an agent skill at `skills/reproduce/SKILL.md` describing how to rebuild and verify it from a fresh clone:

- [`00_task_workspace/skills/reproduce/SKILL.md`](./00_task_workspace/skills/reproduce/SKILL.md)
- [`01_nyc_taxi_prediction/skills/reproduce/SKILL.md`](./01_nyc_taxi_prediction/skills/reproduce/SKILL.md)
- [`02_nano_llm/skills/reproduce/SKILL.md`](./02_nano_llm/skills/reproduce/SKILL.md)
- [`03_customer_segmentation/skills/reproduce/SKILL.md`](./03_customer_segmentation/skills/reproduce/SKILL.md)
- [`04_market_basket/skills/reproduce/SKILL.md`](./04_market_basket/skills/reproduce/SKILL.md)
- [`06_anomaly_detection/skills/reproduce/SKILL.md`](./06_anomaly_detection/skills/reproduce/SKILL.md)
- [`07_automl_stacking/skills/reproduce/SKILL.md`](./07_automl_stacking/skills/reproduce/SKILL.md)
- [`08_visual_foundations/skills/reproduce/SKILL.md`](./08_visual_foundations/skills/reproduce/SKILL.md)
- [`09_flowforge_dag/skills/reproduce/SKILL.md`](./09_flowforge_dag/skills/reproduce/SKILL.md)
- [`10_crispdm_curriculum/skills/reproduce/SKILL.md`](./10_crispdm_curriculum/skills/reproduce/SKILL.md)
- [`11_portfolio_audit/skills/reproduce/SKILL.md`](./11_portfolio_audit/skills/reproduce/SKILL.md)
- [`12_energy_forecasting/skills/reproduce/SKILL.md`](./12_energy_forecasting/skills/reproduce/SKILL.md)
- [`13_mlops_governance/skills/reproduce/SKILL.md`](./13_mlops_governance/skills/reproduce/SKILL.md)
- [`14_multimodal_automl/skills/reproduce/SKILL.md`](./14_multimodal_automl/skills/reproduce/SKILL.md)
- [`15_spy_forecasting/skills/reproduce/SKILL.md`](./15_spy_forecasting/skills/reproduce/SKILL.md)

---

*Generated by `scripts/generate_docs.py` from `docs/skills.json`.*
