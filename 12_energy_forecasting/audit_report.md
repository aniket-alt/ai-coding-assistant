# Audit report — Energy Demand Forecasting

**Result: PASS** · 0 critical · 0 warnings · 4/4 runtime assertions passed

## Runtime assertions

These executed inside the training script and would have halted it on failure. Each entry is written by the assertion itself, so removing a check removes its evidence.

| Assertion | Result | What it measured |
|---|---|---|
| `assert_lags_respect_horizon` | ✅ passed | 8 lags checked against a 24-hour horizon; shortest is 24 hours; 0 shorter than the horizon. |
| `assert_no_lookahead` | ✅ passed | 600 feature values across 30 target hours recomputed from data available at the forecast origin; 0 differed. |
| `assert_temporal_split` | ✅ passed | Last training timestamp 2007-12-16 17:00:00 precedes first test timestamp 2007-12-16 18:00:00 by 0 days 01:00:00; no embargo required. |
| `assert_no_target_leak` | ✅ passed | 20 numeric features correlated against the target; strongest was same_hour_4wk_mean at |r|=0.5275 (threshold 0.995). |

Recorded by the assertion ledger while training ran; generated from `artifacts/metrics.json`.

## Static source analysis

The AST pass over the project source found no findings.

## Dataset provenance

✅ **Real, third-party data**

| Field | Value |
|---|---|
| Dataset | Individual Household Electric Power Consumption (UCI) |
| Source | https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip |
| Licence | Creative Commons Attribution 4.0 (CC BY 4.0) |
| Rows × columns (as downloaded) | 2,075,259 × 7 |
| SHA-256 | `9f84b46ade8a2d8e1286ec4b2b6c2987a45a755c59f263be3b3b3d10dfbda3ff` |
| Hours after aggregation | 34,587 |
| Hours missing | 436 |

## Limitations of this audit

- The static pass matches known leakage patterns; it cannot prove the absence of leakage it has no rule for.
- Runtime assertions check what they were written to check: temporal order, disjoint splits, and features near-perfectly correlated with the target. Subtler leakage, such as a feature that is only partly derived from the target, can pass them.

---

*Generated from `artifacts/metrics.json` and `artifacts/audit.json`.*
