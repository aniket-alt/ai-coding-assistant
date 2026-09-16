# Audit report — SPY Probabilistic Forecasting

**Result: PASS** · 0 critical · 0 warnings · 2/2 runtime assertions passed

## Runtime assertions

These executed inside the training script and would have halted it on failure. Each entry is written by the assertion itself, so removing a check removes its evidence.

| Assertion | Result | What it measured |
|---|---|---|
| `assert_no_lookahead` | ✅ passed | 780 feature values across 30 dates recomputed on truncated history; 0 differed from the full-history value. |
| `assert_no_target_leak` | ✅ passed | 26 numeric features correlated against the target; strongest was ret_lag2 at |r|=0.0909 (threshold 0.995). |

Recorded by the assertion ledger while training ran; generated from `artifacts/metrics.json`.

## Static source analysis

The AST pass over the project source found no findings.

## Dataset provenance

✅ **Real, third-party data**

| Field | Value |
|---|---|
| Dataset | SPY Daily Bars with VIX and 10-Year Yield |
| Source | https://finance.yahoo.com/quote/SPY/history |
| Licence | Yahoo Finance terms of use — personal and research use |
| Rows × columns (as downloaded) | 2,931 × 7 |
| SHA-256 | `1e37e3dd2c6da9f77b12695e298b45196868bb004833daa20cbf95679d3309ab` |

## Limitations of this audit

- The static pass matches known leakage patterns; it cannot prove the absence of leakage it has no rule for.
- Runtime assertions check what they were written to check: temporal order, disjoint splits, and features near-perfectly correlated with the target. Subtler leakage, such as a feature that is only partly derived from the target, can pass them.

---

*Generated from `artifacts/metrics.json` and `artifacts/audit.json`.*
