# Audit report — Credit Card Fraud Detection

**Result: PASS** · 0 critical · 0 warnings · 3/3 runtime assertions passed

## Runtime assertions

These executed inside the training script and would have halted it on failure. Each entry is written by the assertion itself, so removing a check removes its evidence.

| Assertion | Result | What it measured |
|---|---|---|
| `assert_temporal_split` | ✅ passed | Last training timestamp 1 days 09:59:57 precedes first test timestamp 1 days 11:00:00 by 0 days 01:00:03; embargo of 30min required. |
| `assert_disjoint` | ✅ passed | 175,582 training and 100,915 test row indices compared; 0 appear in both. |
| `assert_no_target_leak` | ✅ passed | 30 numeric features correlated against the target; strongest was V17 at |r|=0.4096 (threshold 0.995). |

Recorded by the assertion ledger while training ran; generated from `artifacts/metrics.json`.

## Static source analysis

The AST pass over the project source found no findings.

## Dataset provenance

✅ **Real, third-party data**

| Field | Value |
|---|---|
| Dataset | Credit Card Fraud Detection (ULB) |
| Source | https://openml.org/data/v1/download/1673544/creditcard.arff |
| Licence | Public (Open Database License) — OpenML dataset 1597 |
| Rows × columns (as downloaded) | 284,807 × 31 |
| SHA-256 | `fdaf12730dc1fc426f318b71349f24f5c5fd00aa1152940be7e7509ae3d89d2a` |

## Limitations of this audit

- The static pass matches known leakage patterns; it cannot prove the absence of leakage it has no rule for.
- Runtime assertions check what they were written to check: temporal order, disjoint splits, and features near-perfectly correlated with the target. Subtler leakage, such as a feature that is only partly derived from the target, can pass them.

---

*Generated from `artifacts/metrics.json` and `artifacts/audit.json`.*
