# Audit report — Task Workspace

**Result: PASS** · 0 critical · 0 warnings · 3/3 runtime assertions passed

## Runtime assertions

These executed inside the training script and would have halted it on failure. Each entry is written by the assertion itself, so removing a check removes its evidence.

| Assertion | Result | What it measured |
|---|---|---|
| `assert_round_trips_recover_every_field` | ✅ passed | 5,000 generated tasks rendered as quick-entry text in random order; 0 did not parse back to exactly the fields they were built from. |
| `assert_every_tab_receives_every_change_in_order` | ✅ passed | 1 subscribers: 1 received all 200 changes in sequence; 10 subscribers: 10 received all 200 changes in sequence; 50 subscribers: 50 received all 200 changes in sequence. |
| `assert_exactly_one_concurrent_writer_wins` | ✅ passed | 50 rounds of 8 writers on the same version: 50 accepted, 350 refused with 409; final version 51 (expected 51). |

Recorded by the assertion ledger while training ran; generated from `artifacts/metrics.json`.

## Static source analysis

The AST pass over the project source found no findings.

## Dataset provenance

⚠️ **Synthetic data**

| Field | Value |
|---|---|
| Dataset | Hand-written quick-entry phrases and generated workspace traffic |
| Source | https://github.com/aniket-alt/ai-coding-assistant/blob/main/00_task_workspace/ml/parser_cases.json |
| Licence | MIT (written for this repository) |
| Rows × columns (as downloaded) | 60 × 0 |
| SHA-256 | `d958d955c4b15f69c36271b3e7b57d7945d527d9bc6ad74543645629102caa6e` |

## Limitations of this audit

- The static pass matches known leakage patterns; it cannot prove the absence of leakage it has no rule for.
- Runtime assertions check what they were written to check: temporal order, disjoint splits, and features near-perfectly correlated with the target. Subtler leakage, such as a feature that is only partly derived from the target, can pass them.

---

*Generated from `artifacts/metrics.json` and `artifacts/audit.json`.*
