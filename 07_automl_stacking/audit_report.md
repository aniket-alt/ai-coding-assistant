# Audit report — AutoML Stacking

**Result: PASS** · 0 critical · 0 warnings · 3/3 runtime assertions passed

## Runtime assertions

These executed inside the training script and would have halted it on failure. Each entry is written by the assertion itself, so removing a check removes its evidence.

| Assertion | Result | What it measured |
|---|---|---|
| `assert_temporal_split` | ✅ passed | Last training timestamp 2009-12-01 00:00:00 precedes first test timestamp 2010-03-01 00:00:00 by 90 days 00:00:00; no embargo required. |
| `assert_duration_excluded` | ✅ passed | 19 model inputs; `duration` (known only after the call) absent. |
| `assert_no_target_leak` | ✅ passed | 63 numeric features correlated against the target; strongest was emp.var.rate at |r|=0.2634 (threshold 0.995). |

Recorded by the assertion ledger while training ran; generated from `artifacts/metrics.json`.

## Static source analysis

The AST pass over the project source found no findings.

## Dataset provenance

✅ **Real, third-party data**

| Field | Value |
|---|---|
| Dataset | Bank Marketing, with social and economic context (UCI) |
| Source | https://archive.ics.uci.edu/static/public/222/bank+marketing.zip |
| Licence | Creative Commons Attribution 4.0 (CC BY 4.0) |
| Rows × columns (as downloaded) | 41,188 × 21 |
| SHA-256 | `e0bf5f5de5b846e2f18e9d90606637267d46dfa260e0f17bb12e605db5efbeb4` |

## Limitations of this audit

- The static pass matches known leakage patterns; it cannot prove the absence of leakage it has no rule for.
- Runtime assertions check what they were written to check: temporal order, disjoint splits, and features near-perfectly correlated with the target. Subtler leakage, such as a feature that is only partly derived from the target, can pass them.

---

*Generated from `artifacts/metrics.json` and `artifacts/audit.json`.*
