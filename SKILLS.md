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
| **Missingness as a signal** | [`01_nyc_taxi_prediction/ml/data.py`](./01_nyc_taxi_prediction/ml/data.py#L177) | Missing fields cluster by vendor, so the gap itself is kept as a feature. |
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
| **Quantile regression and pinball loss** | [`15_spy_forecasting/ml/train.py`](./15_spy_forecasting/ml/train.py#L48) | Scores a whole predictive distribution rather than a single point. |
| **Walk-forward validation with purging** | [`15_spy_forecasting/ml/train.py`](./15_spy_forecasting/ml/train.py#L161) | Repeated out-of-sample refits, dropping training rows whose targets overlap the test block. |
| **Paired bootstrap significance testing** | [`15_spy_forecasting/ml/train.py`](./15_spy_forecasting/ml/train.py#L95) | Asks whether a winner beats a random walk or just got lucky. |
| **Backtesting with costs against buy-and-hold** | [`15_spy_forecasting/ml/train.py`](./15_spy_forecasting/ml/train.py#L122) | A strategy is always compared with simply holding, net of execution cost. |
| **Fat-tail-aware intervals** | [`15_spy_forecasting/ml/train.py`](./15_spy_forecasting/ml/train.py#L446) | Student-t quantiles over EWMA volatility, because returns have heavy tails. |
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

## Reproduction skills

Each built project ships an agent skill at `skills/reproduce/SKILL.md` describing how to rebuild and verify it from a fresh clone:

- [`01_nyc_taxi_prediction/skills/reproduce/SKILL.md`](./01_nyc_taxi_prediction/skills/reproduce/SKILL.md)
- [`03_customer_segmentation/skills/reproduce/SKILL.md`](./03_customer_segmentation/skills/reproduce/SKILL.md)
- [`04_market_basket/skills/reproduce/SKILL.md`](./04_market_basket/skills/reproduce/SKILL.md)
- [`06_anomaly_detection/skills/reproduce/SKILL.md`](./06_anomaly_detection/skills/reproduce/SKILL.md)
- [`15_spy_forecasting/skills/reproduce/SKILL.md`](./15_spy_forecasting/skills/reproduce/SKILL.md)

---

*Generated by `scripts/generate_docs.py` from `docs/skills.json`.*
