# Audit report — MLOps & Governance

**Result: PASS** · 0 critical · 0 warnings · 3/3 runtime assertions passed

## Runtime assertions

These executed inside the training script and would have halted it on failure. Each entry is written by the assertion itself, so removing a check removes its evidence.

| Assertion | Result | What it measured |
|---|---|---|
| `assert_temporal_split` | ✅ passed | Last training timestamp 2024-01-30 23:07:53 precedes first test timestamp 2024-02-01 00:00:03 by 1 days 00:52:10; no embargo required. |
| `assert_no_target_leak` | ✅ passed | 22 numeric features correlated against the target; strongest was zp_median_distance at |r|=0.7821 (threshold 0.995). |
| `assert_policies_train_only_on_finished_months` | ✅ passed | 14 model fits checked; 0 used a month at or after the month they served. |

Recorded by the assertion ledger while training ran; generated from `artifacts/metrics.json`.

## Static source analysis

The AST pass over the project source found no findings.

## Dataset provenance

✅ **Real, third-party data**

| Field | Value |
|---|---|
| Dataset | NYC TLC Yellow Taxi Trip Records — January to July 2024 |
| Source | https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet |
| Licence | NYC Taxi & Limousine Commission open data (public domain) |
| Rows × columns (as downloaded) | 23,408,996 × 19 |
| SHA-256 | `c4d59da7bbc8abaeeeb1727947ee93d9891a71acb42854bd80db1571b2030510` |
| Months | [{'month': '2024-01', 'rows': 2964624, 'sha256': 'c4d59da7bbc8abaeeeb1727947ee93d9891a71acb42854bd80db1571b2030510'}, {'month': '2024-02', 'rows': 3007526, 'sha256': 'c76c43c18c6c6664080dd920baab4928988d5786a6b65980792ca7cd796f9f20'}, {'month': '2024-03', 'rows': 3582628, 'sha256': '2d4cdc8fb96726cdd3803b13b02d2e61e71d45720aff0ebc693a8bdd1f249823'}, {'month': '2024-04', 'rows': 3514289, 'sha256': 'ce37f736c7cf9e92164c850ef2b502a8f7bb590fc1804b6edbcb80293da01b89'}, {'month': '2024-05', 'rows': 3723833, 'sha256': '1974230be2b5a3ea92e47b5eb77f32dbf1de896d81ad8df4ab6faba52749ad23'}, {'month': '2024-06', 'rows': 3539193, 'sha256': '677cf14c8347f745b583f012fbdba072334c6c4efa17bfd6a64369f2ba30329c'}, {'month': '2024-07', 'rows': 3076903, 'sha256': 'af12d6b04daeb78550799ae99933d495b27d715b6dbb0fad9bb750a4537d3070'}] |
| Rows modelled per month | 150,000 |

## Limitations of this audit

- The static pass matches known leakage patterns; it cannot prove the absence of leakage it has no rule for.
- Runtime assertions check what they were written to check: temporal order, disjoint splits, and features near-perfectly correlated with the target. Subtler leakage, such as a feature that is only partly derived from the target, can pass them.

---

*Generated from `artifacts/metrics.json` and `artifacts/audit.json`.*
