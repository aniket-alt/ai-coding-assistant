# Audit report — Multimodal AutoML

**Result: PASS** · 0 critical · 1 warnings · 3/3 runtime assertions passed

## Runtime assertions

These executed inside the training script and would have halted it on failure. Each entry is written by the assertion itself, so removing a check removes its evidence.

| Assertion | Result | What it measured |
|---|---|---|
| `assert_no_name_in_two_splits` | ✅ passed | 28,510 training, 6,111 validation and 6,254 test products grouped by display name; 0 names appear in more than one split. |
| `assert_every_class_in_test` | ✅ passed | All 45 article types have test products; missing: []. |
| `assert_hierarchy_fields_excluded` | ✅ passed | Catalogue features ['gender', 'baseColour', 'season', 'year', 'usage']; the coarser category levels, which determine most of the article type, are excluded. |

Recorded by the assertion ledger while training ran; generated from `artifacts/metrics.json`.

## Static source analysis

| Severity | Rule | Location | Finding |
|---|---|---|---|
| WARNING | `fit-before-split` | `ml\train.py:57` | `.fit()` on line 57 precedes the train/test split on line 136; verify the transformer is not fitted on all rows. |

## Dataset provenance

✅ **Real, third-party data**

| Field | Value |
|---|---|
| Dataset | Fashion Product Images (Small) |
| Source | https://huggingface.co/datasets/ashraq/fashion-product-images-small/resolve/3859c76db2f6f3d3b9a3863345e3ccdbff75879d/data/train-00000-of-00002-6cff4c59f91661c3.parquet |
| Licence | MIT, as declared by the Kaggle dataset; product photographs originate from a retailer's catalogue |
| Rows × columns (as downloaded) | 44,072 × 10 |
| SHA-256 | `0f9b4042c6eea3c77aa0794630c956d1985de791c365abdb332f9d9cfa78d00a` |

## Limitations of this audit

- The static pass matches known leakage patterns; it cannot prove the absence of leakage it has no rule for.
- Runtime assertions check what they were written to check: temporal order, disjoint splits, and features near-perfectly correlated with the target. Subtler leakage, such as a feature that is only partly derived from the target, can pass them.

---

*Generated from `artifacts/metrics.json` and `artifacts/audit.json`.*
