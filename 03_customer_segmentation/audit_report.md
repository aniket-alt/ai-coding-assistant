# Audit report — Customer Segmentation

**Result: PASS** · 0 critical · 0 warnings · 3/3 runtime assertions passed

## Runtime assertions

These executed inside the training script and would have halted it on failure. Each entry is written by the assertion itself, so removing a check removes its evidence.

| Assertion | Result | What it measured |
|---|---|---|
| `assert_temporal_split` | ✅ passed | Last training timestamp 2010-11-30 19:35:00 precedes first test timestamp 2010-12-01 08:26:00 by 0 days 12:51:00; no embargo required. |
| `assert_disjoint` | ✅ passed | 385,427 training and 391,150 test row indices compared; 0 appear in both. |
| `assert_no_target_leak` | ✅ passed | 7 numeric features correlated against the target; strongest was recency_days at |r|=0.4046 (threshold 0.995). |

Recorded by the assertion ledger while training ran; generated from `artifacts/metrics.json`.

## Static source analysis

The AST pass over the project source found no findings.

## Dataset provenance

✅ **Real, third-party data**

| Field | Value |
|---|---|
| Dataset | Online Retail II (UCI) |
| Source | https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip |
| Licence | Creative Commons Attribution 4.0 (CC BY 4.0) |
| Rows × columns (as downloaded) | 1,067,371 × 8 |
| SHA-256 | `572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb` |
| Customers in first year | 4,239 |
| Product lines after cleaning | 776,577 |

## Limitations of this audit

- The static pass matches known leakage patterns; it cannot prove the absence of leakage it has no rule for.
- Runtime assertions check what they were written to check: temporal order, disjoint splits, and features near-perfectly correlated with the target. Subtler leakage, such as a feature that is only partly derived from the target, can pass them.

---

*Generated from `artifacts/metrics.json` and `artifacts/audit.json`.*
