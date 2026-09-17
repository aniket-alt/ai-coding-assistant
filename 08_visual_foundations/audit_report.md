# Audit report — DS Visual Foundations

**Result: PASS** · 0 critical · 0 warnings · 4/4 runtime assertions passed

## Runtime assertions

These executed inside the training script and would have halted it on failure. Each entry is written by the assertion itself, so removing a check removes its evidence.

| Assertion | Result | What it measured |
|---|---|---|
| `assert_no_message_in_both_splits` | ✅ passed | 4,136 training and 1,035 test messages after removing 403 duplicates; 0 message texts appear in both. |
| `assert_naive_bayes_matches_sklearn` | ✅ passed | Spam probabilities for 1,035 test messages from the written-out Naive Bayes and scikit-learn's MultinomialNB; largest difference 1.6e-14 (limit 1e-9). |
| `assert_gradient_descent_reaches_least_squares` | ✅ passed | Gradient descent (learning rate 0.1, 3,000 steps, scaled features) against the exact least-squares solution: largest coefficient difference 1.1e-16 relative (limit 1e-6). |
| `assert_backprop_matches_finite_differences` | ✅ passed | All 49 parameter gradients compared with central finite differences, before and after training; largest relative error 1.1e-07 (limit 1e-5). |

Recorded by the assertion ledger while training ran; generated from `artifacts/metrics.json`.

## Static source analysis

The AST pass over the project source found no findings.

## Dataset provenance

✅ **Real, third-party data**

| Field | Value |
|---|---|
| Dataset | SMS Spam Collection (UCI) |
| Source | https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip |
| Licence | Creative Commons Attribution 4.0 (CC BY 4.0) |
| Rows × columns (as downloaded) | 5,574 × 3 |
| SHA-256 | `1587ea43e58e82b14ff1f5425c88e17f8496bfcdb67a583dbff9eefaf9963ce3` |

## Limitations of this audit

- The static pass matches known leakage patterns; it cannot prove the absence of leakage it has no rule for.
- Runtime assertions check what they were written to check: temporal order, disjoint splits, and features near-perfectly correlated with the target. Subtler leakage, such as a feature that is only partly derived from the target, can pass them.

---

*Generated from `artifacts/metrics.json` and `artifacts/audit.json`.*
