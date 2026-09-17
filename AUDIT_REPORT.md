# Portfolio audit report

Every row below is read from a project's `artifacts/metrics.json`. Projects that have not been built yet do not appear.

| Project | Dataset | Real data | Critical | Warnings | Runtime assertions |
|---|---|:---:|---:|---:|---|
| [00 Task Workspace](./00_task_workspace/audit_report.md) | Hand-written quick-entry phrases and generated workspace traffic | no ⚠️ | 0 | 0 | 3/3 passed |
| [01 NYC Taxi Trip Duration](./01_nyc_taxi_prediction/audit_report.md) | NYC TLC Yellow Taxi Trip Records — January 2024 | yes | 0 | 0 | 3/3 passed |
| [02 Nano LLM Transformer](./02_nano_llm/audit_report.md) | TinyStories V2 (GPT-4 stories), training split | yes | 0 | 0 | 3/3 passed |
| [03 Customer Segmentation](./03_customer_segmentation/audit_report.md) | Online Retail II (UCI) | yes | 0 | 0 | 3/3 passed |
| [04 Market Basket Mining](./04_market_basket/audit_report.md) | Online Retail II (UCI) | yes | 0 | 0 | 4/4 passed |
| [06 Credit Card Fraud Detection](./06_anomaly_detection/audit_report.md) | Credit Card Fraud Detection (ULB) | yes | 0 | 0 | 3/3 passed |
| [07 AutoML Stacking](./07_automl_stacking/audit_report.md) | Bank Marketing, with social and economic context (UCI) | yes | 0 | 0 | 3/3 passed |
| [08 DS Visual Foundations](./08_visual_foundations/audit_report.md) | SMS Spam Collection (UCI) | yes | 0 | 0 | 4/4 passed |
| [09 FlowForge DAG Engine](./09_flowforge_dag/audit_report.md) | Seeded random workflow graphs (FlowForge generator) | no ⚠️ | 0 | 0 | 4/4 passed |
| [10 CRISP-DM Curriculum](./10_crispdm_curriculum/audit_report.md) | Adult Census Income (UCI, via OpenML 1590) | yes | 0 | 0 | 3/3 passed |
| [11 Portfolio Audit](./11_portfolio_audit/audit_report.md) | Credit Card Fraud Detection (ULB) | yes | 0 | 0 | 3/3 passed |
| [12 Energy Demand Forecasting](./12_energy_forecasting/audit_report.md) | Individual Household Electric Power Consumption (UCI) | yes | 0 | 0 | 4/4 passed |
| [13 MLOps & Governance](./13_mlops_governance/audit_report.md) | NYC TLC Yellow Taxi Trip Records — January to July 2024 | yes | 0 | 0 | 3/3 passed |
| [15 SPY Probabilistic Forecasting](./15_spy_forecasting/audit_report.md) | SPY Daily Bars with VIX and 10-Year Yield | yes | 0 | 0 | 2/2 passed |

## Runtime assertions by project

### 00 · Task Workspace

| Assertion | Result | What it measured |
|---|---|---|
| `assert_round_trips_recover_every_field` | ✅ passed | 5,000 generated tasks rendered as quick-entry text in random order; 0 did not parse back to exactly the fields they were built from. |
| `assert_every_tab_receives_every_change_in_order` | ✅ passed | 1 subscribers: 1 received all 200 changes in sequence; 10 subscribers: 10 received all 200 changes in sequence; 50 subscribers: 50 received all 200 changes in sequence. |
| `assert_exactly_one_concurrent_writer_wins` | ✅ passed | 50 rounds of 8 writers on the same version: 50 accepted, 350 refused with 409; final version 51 (expected 51). |

Recorded by the assertion ledger while training ran; generated from `artifacts/metrics.json`.

### 01 · NYC Taxi Trip Duration

| Assertion | Result | What it measured |
|---|---|---|
| `assert_temporal_split` | ✅ passed | Last training timestamp 2024-01-24 23:59:55 precedes first test timestamp 2024-01-26 00:00:01 by 1 days 00:00:06; embargo of 12h required. |
| `assert_disjoint` | ✅ passed | 456,487 training and 121,130 test row indices compared; 0 appear in both. |
| `assert_no_target_leak` | ✅ passed | 22 numeric features correlated against the target; strongest was zp_median_distance at |r|=0.7821 (threshold 0.995). |

Recorded by the assertion ledger while training ran; generated from `artifacts/metrics.json`.

### 02 · Nano LLM Transformer

| Assertion | Result | What it measured |
|---|---|---|
| `assert_validation_stories_unseen` | ✅ passed | 27,630 validation stories hashed against 2,717,495 training stories; 0 also appeared in training and were removed; 0 remain. |
| `assert_causal_mask` | ✅ passed | Tokens 40-63 of a 64-token sequence replaced; largest change in logits at positions 0-39 was 0.00e+00 (limit 1e-4). The model cannot see the future. |
| `assert_kv_cache_matches_recompute` | ✅ passed | Greedy 60-token continuations of 3 prompts generated with and without the KV cache; 3 of 3 identical. |

Recorded by the assertion ledger while training ran; generated from `artifacts/metrics.json`.

### 03 · Customer Segmentation

| Assertion | Result | What it measured |
|---|---|---|
| `assert_temporal_split` | ✅ passed | Last training timestamp 2010-11-30 19:35:00 precedes first test timestamp 2010-12-01 08:26:00 by 0 days 12:51:00; no embargo required. |
| `assert_disjoint` | ✅ passed | 385,427 training and 391,150 test row indices compared; 0 appear in both. |
| `assert_no_target_leak` | ✅ passed | 7 numeric features correlated against the target; strongest was recency_days at |r|=0.4046 (threshold 0.995). |

Recorded by the assertion ledger while training ran; generated from `artifacts/metrics.json`.

### 04 · Market Basket Mining

| Assertion | Result | What it measured |
|---|---|---|
| `assert_temporal_split` | ✅ passed | Last training timestamp 2011-08-31 17:30:00 precedes first test timestamp 2011-09-08 08:56:00 by 7 days 15:26:00; embargo of 6D required. |
| `assert_disjoint` | ✅ passed | 789,736 training and 203,765 test row indices compared; 0 appear in both. |
| `assert_no_shared_invoices` | ✅ passed | 32,126 training and 7,012 test invoices compared; 0 in both. |
| `assert_hidden_item_not_visible` | ✅ passed | 6,548 test baskets checked; the hidden item was visible to the model in 0. |

Recorded by the assertion ledger while training ran; generated from `artifacts/metrics.json`.

### 06 · Credit Card Fraud Detection

| Assertion | Result | What it measured |
|---|---|---|
| `assert_temporal_split` | ✅ passed | Last training timestamp 1 days 09:59:57 precedes first test timestamp 1 days 11:00:00 by 0 days 01:00:03; embargo of 30min required. |
| `assert_disjoint` | ✅ passed | 175,582 training and 100,915 test row indices compared; 0 appear in both. |
| `assert_no_target_leak` | ✅ passed | 30 numeric features correlated against the target; strongest was V17 at |r|=0.4096 (threshold 0.995). |

Recorded by the assertion ledger while training ran; generated from `artifacts/metrics.json`.

### 07 · AutoML Stacking

| Assertion | Result | What it measured |
|---|---|---|
| `assert_temporal_split` | ✅ passed | Last training timestamp 2009-12-01 00:00:00 precedes first test timestamp 2010-03-01 00:00:00 by 90 days 00:00:00; no embargo required. |
| `assert_duration_excluded` | ✅ passed | 19 model inputs; `duration` (known only after the call) absent. |
| `assert_no_target_leak` | ✅ passed | 63 numeric features correlated against the target; strongest was emp.var.rate at |r|=0.2634 (threshold 0.995). |

Recorded by the assertion ledger while training ran; generated from `artifacts/metrics.json`.

### 08 · DS Visual Foundations

| Assertion | Result | What it measured |
|---|---|---|
| `assert_no_message_in_both_splits` | ✅ passed | 4,136 training and 1,035 test messages after removing 403 duplicates; 0 message texts appear in both. |
| `assert_naive_bayes_matches_sklearn` | ✅ passed | Spam probabilities for 1,035 test messages from the written-out Naive Bayes and scikit-learn's MultinomialNB; largest difference 1.6e-14 (limit 1e-9). |
| `assert_gradient_descent_reaches_least_squares` | ✅ passed | Gradient descent (learning rate 0.1, 3,000 steps, scaled features) against the exact least-squares solution: largest coefficient difference 1.1e-16 relative (limit 1e-6). |
| `assert_backprop_matches_finite_differences` | ✅ passed | All 49 parameter gradients compared with central finite differences, before and after training; largest relative error 1.1e-07 (limit 1e-5). |

Recorded by the assertion ledger while training ran; generated from `artifacts/metrics.json`.

### 09 · FlowForge DAG Engine

| Assertion | Result | What it measured |
|---|---|---|
| `assert_engine_tests_pass` | ✅ passed | node --test ran 11 tests: 11 passed, 0 failed. |
| `assert_type_tests_pass` | ✅ passed | tsc over the engine, its tests and type-tests: 0 errors, including 0 @ts-expect-error lines that no longer fail. |
| `assert_properties_hold` | ✅ passed | 5 properties over 5,000 random graphs and workflows: 77,531 checks, 0 failures. |
| `assert_every_mutant_killed` | ✅ passed | 10 planted bugs; the test suites caught 10. |

Recorded by the assertion ledger while training ran; generated from `artifacts/metrics.json`.

### 10 · CRISP-DM Curriculum

| Assertion | Result | What it measured |
|---|---|---|
| `assert_disjoint` | ✅ passed | 36,631 training and 12,211 test row indices compared; 0 appear in both. |
| `assert_no_target_leak` | ✅ passed | 85 numeric features correlated against the target; strongest was cat__marital-status_Married-civ-spouse at |r|=0.4493 (threshold 0.995). |
| `assert_unaware_model_excludes_sensitive_attributes` | ✅ passed | 11 features in the unaware models; sensitive attributes present: []. |

Recorded by the assertion ledger while training ran; generated from `artifacts/metrics.json`.

### 11 · Portfolio Audit

| Assertion | Result | What it measured |
|---|---|---|
| `assert_leaks_differ_from_template` | ✅ passed | 11 leak variants compared with the clean template; 0 identical. |
| `assert_all_variants_executed` | ✅ passed | 14 variants executed; 0 failed for reasons other than a leakage check. |
| `assert_controls_are_clean` | ✅ passed | 3 controls scored within 0.0036 average precision of the template and raised 0 runtime checks. |

Recorded by the assertion ledger while training ran; generated from `artifacts/metrics.json`.

### 12 · Energy Demand Forecasting

| Assertion | Result | What it measured |
|---|---|---|
| `assert_lags_respect_horizon` | ✅ passed | 8 lags checked against a 24-hour horizon; shortest is 24 hours; 0 shorter than the horizon. |
| `assert_no_lookahead` | ✅ passed | 600 feature values across 30 target hours recomputed from data available at the forecast origin; 0 differed. |
| `assert_temporal_split` | ✅ passed | Last training timestamp 2007-12-16 17:00:00 precedes first test timestamp 2007-12-16 18:00:00 by 0 days 01:00:00; no embargo required. |
| `assert_no_target_leak` | ✅ passed | 20 numeric features correlated against the target; strongest was same_hour_4wk_mean at |r|=0.5275 (threshold 0.995). |

Recorded by the assertion ledger while training ran; generated from `artifacts/metrics.json`.

### 13 · MLOps & Governance

| Assertion | Result | What it measured |
|---|---|---|
| `assert_temporal_split` | ✅ passed | Last training timestamp 2024-01-30 23:07:53 precedes first test timestamp 2024-02-01 00:00:03 by 1 days 00:52:10; no embargo required. |
| `assert_no_target_leak` | ✅ passed | 22 numeric features correlated against the target; strongest was zp_median_distance at |r|=0.7821 (threshold 0.995). |
| `assert_policies_train_only_on_finished_months` | ✅ passed | 14 model fits checked; 0 used a month at or after the month they served. |

Recorded by the assertion ledger while training ran; generated from `artifacts/metrics.json`.

### 15 · SPY Probabilistic Forecasting

| Assertion | Result | What it measured |
|---|---|---|
| `assert_no_lookahead` | ✅ passed | 780 feature values across 30 dates recomputed on truncated history; 0 differed from the full-history value. |
| `assert_no_target_leak` | ✅ passed | 26 numeric features correlated against the target; strongest was ret_lag2 at |r|=0.0909 (threshold 0.995). |

Recorded by the assertion ledger while training ran; generated from `artifacts/metrics.json`.

## The same static auditor run on the source repository

For comparison, `dsfs.leakage.audit_source` was run over `dlmastery/data_science_examples`. That repository's own `AUDIT_REPORT.md` states "Critical Data Leakages Detected: 0" and gives each of its 13 systems the same sub-scores.

**1 critical · 4 warnings**

| Severity | Rule | Location | Finding |
|---|---|---|---|
| CRITICAL | `fit-on-test` | `03_customer_segmentation_clustering/ml/autoresearch_clustering.py:201` | .fit_transform() called on `df` -- a transformer or model is being fitted on held-out or full data. |
| WARNING | `unseeded` | `01_nyc_taxi_trip_prediction/ml/train.py:113` | XGBRegressor() has no random_state -- results are not reproducible. |
| WARNING | `unseeded` | `01_nyc_taxi_trip_prediction/server/inference.py:28` | XGBRegressor() has no random_state -- results are not reproducible. |
| WARNING | `fit-before-split` | `07_automl_autogluon/ml/autogluon_stacking.py:28` | `.fit()` on line 28 precedes the train/test split on line 64; verify the transformer is not fitted on all rows. |
| WARNING | `fit-before-split` | `10_crispdm_masters_curriculum/backend/main.py:92` | `.fit()` on line 92 precedes the train/test split on line 103; verify the transformer is not fitted on all rows. |

Static findings are leads, not verdicts: a `fit-on-test` match on a variable named `df` needs a human to confirm the frame really contains held-out rows. They are reported here as found.

## What these checks cannot catch

- Leakage with no static signature, such as a feature partly derived from the target through a join.
- Selection effects in the data itself, such as unlabelled fraud sitting in the legitimate class.
- Overfitting to the test set through repeated manual iteration. Every project here was tuned while its test metrics were visible, which is a real if modest bias.

---

*Generated by `scripts/generate_docs.py`.*
