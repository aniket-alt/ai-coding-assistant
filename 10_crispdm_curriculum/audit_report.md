# Audit report — CRISP-DM Curriculum

**Result: PASS** · 0 critical · 0 warnings · 3/3 runtime assertions passed

## Runtime assertions

These executed inside the training script and would have halted it on failure. Each entry is written by the assertion itself, so removing a check removes its evidence.

| Assertion | Result | What it measured |
|---|---|---|
| `assert_disjoint` | ✅ passed | 36,631 training and 12,211 test row indices compared; 0 appear in both. |
| `assert_no_target_leak` | ✅ passed | 85 numeric features correlated against the target; strongest was cat__marital-status_Married-civ-spouse at |r|=0.4493 (threshold 0.995). |
| `assert_unaware_model_excludes_sensitive_attributes` | ✅ passed | 11 features in the unaware models; sensitive attributes present: []. |

Recorded by the assertion ledger while training ran; generated from `artifacts/metrics.json`.

## Static source analysis

The AST pass over the project source found no findings.

## Dataset provenance

✅ **Real, third-party data**

| Field | Value |
|---|---|
| Dataset | Adult Census Income (UCI, via OpenML 1590) |
| Source | https://openml.org/data/v1/download/1595261/adult.arff |
| Licence | Public (OpenML); originally UCI Machine Learning Repository, CC BY 4.0 |
| Rows × columns (as downloaded) | 48,842 × 16 |
| SHA-256 | `77aa1703717a29f0b5642e94c3ba1defd2486f0b34d4d8eccc1b37a5f7d226b0` |

## Limitations of this audit

- The static pass matches known leakage patterns; it cannot prove the absence of leakage it has no rule for.
- Runtime assertions check what they were written to check: temporal order, disjoint splits, and features near-perfectly correlated with the target. Subtler leakage, such as a feature that is only partly derived from the target, can pass them.

---

*Generated from `artifacts/metrics.json` and `artifacts/audit.json`.*
