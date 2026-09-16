# Code walkthrough

A guided tour of the repository, generated from the source files themselves: each summary is the header comment of the file it describes.

## The platform

Shared by every project. A project imports it instead of re-implementing data sourcing, reporting, auditing or its interface.

| File | Lines | What it does |
|---|---:|---|
| [`platform/dsfs/__init__.py`](./platform/dsfs/__init__.py) | 25 | dsfs -- the shared kernel behind this 16-project data science portfolio. |
| [`platform/dsfs/app.py`](./platform/dsfs/app.py) | 165 | FastAPI application factory shared by all 16 project backends. |
| [`platform/dsfs/crispdm.py`](./platform/dsfs/crispdm.py) | 134 | CRISP-DM as a data structure, not a section heading. |
| [`platform/dsfs/datasets.py`](./platform/dsfs/datasets.py) | 102 | `fetch()` downloads a URL once into `datasets/cache/`, checksums it, and returns a stamped `Provenance`. Re-running is free and offline-safe. Every project's data step goes… |
| [`platform/dsfs/docs.py`](./platform/dsfs/docs.py) | 991 | Every figure in a generated document is read from the artifacts a training run wrote: metrics.json, crispdm.json, provenance.json and audit.json. The only hand-written inputs… |
| [`platform/dsfs/leakage.py`](./platform/dsfs/leakage.py) | 445 | Leakage auditor: static (AST) + runtime assertions. |
| [`platform/dsfs/metrics.py`](./platform/dsfs/metrics.py) | 191 | The single source of truth for every number this portfolio publishes. |
| [`platform/dsfs/provenance.py`](./platform/dsfs/provenance.py) | 100 | Thesis of this portfolio: every number in every README traces to a real, citable dataset. A `Provenance` is the receipt. Projects cannot register a dataset without one, and… |
| [`platform/web/src/components/AppShell.tsx`](./platform/web/src/components/AppShell.tsx) | 295 | AppShell -- the frame every project's client renders inside. |
| [`platform/web/src/components/AuditPanel.tsx`](./platform/web/src/components/AuditPanel.tsx) | 288 | AuditPanel -- the leakage audit, published in the running app. |
| [`platform/web/src/components/Chart.tsx`](./platform/web/src/components/Chart.tsx) | 346 | Chart -- a thin wrapper over Recharts, not a charting library. |
| [`platform/web/src/components/CrispDmWalkthrough.tsx`](./platform/web/src/components/CrispDmWalkthrough.tsx) | 246 | CrispDmWalkthrough -- the six phases, driven entirely by report data. |
| [`platform/web/src/components/MarkdownMath.tsx`](./platform/web/src/components/MarkdownMath.tsx) | 198 | MarkdownMath -- renders the narrative markdown the backend produces, including KaTeX maths. |
| [`platform/web/src/components/MetricsPanel.tsx`](./platform/web/src/components/MetricsPanel.tsx) | 175 | MetricsPanel -- the headline summary of a training run. |
| [`platform/web/src/components/ModelLeaderboard.tsx`](./platform/web/src/components/ModelLeaderboard.tsx) | 246 | ModelLeaderboard -- the tournament table, for any project. |
| [`platform/web/src/components/PredictionForm.tsx`](./platform/web/src/components/PredictionForm.tsx) | 410 | PredictionForm -- schema-driven inference input. |
| [`platform/web/src/components/ProvenanceBadge.tsx`](./platform/web/src/components/ProvenanceBadge.tsx) | 179 | ProvenanceBadge -- renders the dataset receipt. |
| [`platform/web/src/components/StatTile.tsx`](./platform/web/src/components/StatTile.tsx) | 175 | Per the dataviz form heuristic, one number is not a chart. Hero figures use proportional figures; only aligned columns get tabular-nums. Delta direction ships with an arrow… |
| [`platform/web/src/hooks/useApi.ts`](./platform/web/src/hooks/useApi.ts) | 114 | Typed data access for the platform endpoints every backend exposes. |
| [`platform/web/src/index.ts`](./platform/web/src/index.ts) | 47 | @dsfs/ui -- shared design system and data-science UI primitives. |
| [`platform/web/src/types.ts`](./platform/web/src/types.ts) | 152 | TypeScript mirrors of the `dsfs` Python dataclasses. |

### How one training run flows through it

```mermaid
sequenceDiagram
  participant T as ml/train.py
  participant D as dsfs.datasets
  participant L as dsfs.leakage
  participant R as RunRecord / CrispDmReport
  participant A as artifacts/
  participant S as server (dsfs.app)
  participant U as client (@dsfs/ui)
  T->>D: fetch(Provenance)
  D-->>T: cached file + SHA-256
  T->>L: assert_* during split and features
  L-->>L: ledger records what was measured
  T->>R: log models, phases, audit
  R->>A: metrics.json, crispdm.json, audit.json
  S->>A: read on request
  U->>S: /api/metrics, /api/crispdm, ...
```

## 01 · NYC Taxi Trip Duration

How long will this cab ride take? Estimated from booking-time information only. Best: **LightGBM** (rmse (min) 5.345). [README](./01_nyc_taxi_prediction/README.md) · [paper](./01_nyc_taxi_prediction/paper.md) · [design doc](./01_nyc_taxi_prediction/DESIGN_DOC.md)

| File | Lines | What it does |
|---|---:|---|
| [`ml/build_zone_map.py`](./01_nyc_taxi_prediction/ml/build_zone_map.py) | 163 | Build the taxi-zone SVG map from the official TLC shapefile. |
| [`ml/data.py`](./01_nyc_taxi_prediction/ml/data.py) | 254 | Project 01 -- NYC Taxi trip duration: data acquisition and cleaning. |
| [`ml/features.py`](./01_nyc_taxi_prediction/ml/features.py) | 268 | Every feature here must be computable at the moment a rider requests a cab. |
| [`ml/train.py`](./01_nyc_taxi_prediction/ml/train.py) | 428 | Project 01 -- model tournament, audit, and artefact generation. |
| [`server/inference.py`](./01_nyc_taxi_prediction/server/inference.py) | 153 | Loads the persisted model together with the encoders it was trained with, so a prediction at request time is assembled exactly as it was during training. The zone statistics… |
| [`server/main.py`](./01_nyc_taxi_prediction/server/main.py) | 81 | Thanks to the platform factory, this file contains only what is specific to taxi prediction. Health, metadata, CRISP-DM, metrics, provenance and audit endpoints all arrive from… |
| [`client/src/App.tsx`](./01_nyc_taxi_prediction/client/src/App.tsx) | 226 | Almost everything here comes from @dsfs/ui. The only code specific to this project is the zone map and the trip estimator; the rest -- metrics, the tournament table, the… |
| [`client/src/main.tsx`](./01_nyc_taxi_prediction/client/src/main.tsx) | 10 |  |
| [`client/src/TripEstimator.tsx`](./01_nyc_taxi_prediction/client/src/TripEstimator.tsx) | 159 | Composes the shared PredictionForm with this project's zone map, and renders the API's response. Every figure shown is a field of that response; nothing is recomputed here. |
| [`client/src/ZoneMap.tsx`](./01_nyc_taxi_prediction/client/src/ZoneMap.tsx) | 193 | Renders the 263 real TLC taxi zones as inline SVG. Geometry comes from the official TLC shapefile (EPSG:2263), simplified at build time by `ml/build_zone_map.py` -- so this is… |

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/predict` | Estimate duration for one booking, with an interval from measured error. |
| GET | `/api/zones` | The 263 taxi zones, for the origin and destination pickers. |
| GET | `/api/importance` | Which features the winning model actually leans on. |

![NYC Taxi Trip Duration](./01_nyc_taxi_prediction/screenshots/01_trip_estimator.png)

## 06 · Credit Card Fraud Detection

Which handful of transactions should an analyst look at today? Best: **Logistic regression** (avg precision 0.7547). [README](./06_anomaly_detection/README.md) · [paper](./06_anomaly_detection/paper.md) · [design doc](./06_anomaly_detection/DESIGN_DOC.md)

| File | Lines | What it does |
|---|---:|---|
| [`ml/data.py`](./06_anomaly_detection/ml/data.py) | 187 | Project 06 -- Credit card fraud detection: data acquisition. |
| [`ml/train.py`](./06_anomaly_detection/ml/train.py) | 531 | Project 06 -- fraud detection tournament, audit, and artefact generation. |
| [`server/inference.py`](./06_anomaly_detection/server/inference.py) | 140 | Scores a transaction and, more usefully, says where that score sits. A raw suspicion score is close to meaningless on its own: what an analyst needs to know is whether this… |
| [`server/main.py`](./06_anomaly_detection/server/main.py) | 84 | Only the fraud-specific endpoints live here; health, meta, metrics, CRISP-DM, provenance and audit arrive from the platform factory. |
| [`client/src/App.tsx`](./06_anomaly_detection/client/src/App.tsx) | 293 | The only project-specific view is TransactionScorer. Metrics, the detector tournament, the CRISP-DM walkthrough, the audit and every chart come from @dsfs/ui unchanged -- which… |
| [`client/src/main.tsx`](./06_anomaly_detection/client/src/main.tsx) | 10 |  |
| [`client/src/TransactionScorer.tsx`](./06_anomaly_detection/client/src/TransactionScorer.tsx) | 208 | Scores real held-out transactions. The true label travels with each sample, so the view can show when the model gets one wrong -- which it does, for about a third of the fraud.… |

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/score` | Score a transaction and locate it in the held-out distribution. |
| GET | `/api/samples` | Real held-out transactions the UI can load as presets. |
| GET | `/api/budget-curve` | Precision and recall at each analyst review budget. |
| GET | `/api/eda` | Fraud rate by hour and by transaction amount band. |

![Credit Card Fraud Detection](./06_anomaly_detection/screenshots/01_transaction_scorer.png)

## 15 · SPY Probabilistic Forecasting

How much of tomorrow's S&P 500 move is actually predictable? Best: **Historical drift** (pinball loss 0.002660). [README](./15_spy_forecasting/README.md) · [paper](./15_spy_forecasting/paper.md) · [design doc](./15_spy_forecasting/DESIGN_DOC.md)

| File | Lines | What it does |
|---|---:|---|
| [`ml/data.py`](./15_spy_forecasting/ml/data.py) | 175 | Project 15 -- SPY forecasting: market data acquisition. |
| [`ml/features.py`](./15_spy_forecasting/ml/features.py) | 139 | Every feature at row t is computed from information available at the close of day t, and the target is the return of day t+1. That single rule is the whole discipline of this… |
| [`ml/train.py`](./15_spy_forecasting/ml/train.py) | 624 | Project 15 -- walk-forward forecasting tournament. |
| [`server/inference.py`](./15_spy_forecasting/server/inference.py) | 90 | Everything served here was computed during the walk-forward run and stored in metrics.json. Nothing is recalculated at request time, so the forecast band a visitor sees is the… |
| [`server/main.py`](./15_spy_forecasting/server/main.py) | 74 | Project 15 -- FastAPI application. |
| [`client/src/App.tsx`](./15_spy_forecasting/client/src/App.tsx) | 354 | ForecastFan is the only project-specific view. This project is the hardest test of the shared components: the primary metric is lower-is-better, the "models" include pure… |
| [`client/src/ForecastFan.tsx`](./15_spy_forecasting/client/src/ForecastFan.tsx) | 180 | Shows recent SPY closes and the next-period quantile band, in price terms. |
| [`client/src/main.tsx`](./15_spy_forecasting/client/src/main.tsx) | 10 |  |

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/forecast` | Next-period quantile band in return and price terms. |
| GET | `/api/prices` | Recent closes, for the forecast chart. |
| GET | `/api/backtest` | Walk-forward equity curves against buy-and-hold. |
| GET | `/api/calibration` | Interval coverage and significance versus a random walk. |
| GET | `/api/dataset` | Distributional facts and the lookahead verification result. |

![SPY Probabilistic Forecasting](./15_spy_forecasting/screenshots/01_forecast_fan.png)

---

*Generated by `scripts/generate_docs.py`.*
