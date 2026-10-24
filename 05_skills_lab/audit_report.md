# Audit report — Data Science Skills Lab

**Result: PASS** · 0 critical · 0 warnings · 3/3 runtime assertions passed

## Runtime assertions

These executed inside the training script and would have halted it on failure. Each entry is written by the assertion itself, so removing a check removes its evidence.

| Assertion | Result | What it measured |
|---|---|---|
| `assert_every_dataset_is_real` | ✅ passed | 4 datasets downloaded and checksummed: titanic 1,309 rows, sha256 c56e3785; credit_g 1,000 rows, sha256 768be84e; diabetes 768 rows, sha256 4eddd5b2; blood 748 rows, sha256 ee1304ca. Marked synthetic: none. |
| `assert_every_skill_paired_on_the_same_seeds` | ✅ passed | 12 skills, each run on the same 25 seeds, careless and careful on every seed; seed counts seen: [25]. |
| `assert_claims_match_intervals` | ✅ passed | 7 of 12 skills have a bootstrap interval that excludes zero and may state an effect; entries claiming one without it: none. |

Recorded by the assertion ledger while training ran; generated from `artifacts/metrics.json`.

## Static source analysis

The AST pass over the project source found no findings.

## Dataset provenance

✅ **Real, third-party data**

| Field | Value |
|---|---|
| Dataset | Titanic (OpenML 40945) |
| Source | https://openml.org/data/v1/download/16826755/Titanic.arff |
| Licence | Public (OpenML) |
| Rows × columns (as downloaded) | 1,309 × 9 |
| SHA-256 | `c56e3785c96cd8aceac85f11bff21c2bf72841426f50903a6f27be8f334a7d03` |

## Limitations of this audit

- The static pass matches known leakage patterns; it cannot prove the absence of leakage it has no rule for.
- Runtime assertions check what they were written to check: temporal order, disjoint splits, and features near-perfectly correlated with the target. Subtler leakage, such as a feature that is only partly derived from the target, can pass them.

---

*Generated from `artifacts/metrics.json` and `artifacts/audit.json`.*
