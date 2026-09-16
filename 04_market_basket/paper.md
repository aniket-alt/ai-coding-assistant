# Market Basket Mining

*Which products go together, and do those associations hold next season?*

**Aniket Anil Naik** · Project 04 of the data science full-stack portfolio

## Abstract

**Which products go together strongly enough to recommend, and do those associations still hold next season?**

We study this question on *Online Retail II (UCI)*: 1,067,371 rows as downloaded, used under Creative Commons Attribution 4.0 (CC BY 4.0). Following the CRISP-DM process, we compare 7 approaches, 2 of them baselines, on 6,548 evaluation rows and rank them by hit rate @ 5. Evaluation split: chronological with 7-day embargo, one item hidden per test basket.

The best approach, **Item co-occurrence**, reaches 18.6% against 1.9% for the strongest baseline (Bestsellers).

4 of 4 leakage assertions passed while training ran, and static analysis of the source reported 0 critical and 0 warning findings.

**Keywords:** association rule mining and recommendation, FP-Growth association rules, held-out item recommendation test, rule survival in a later season, item co-occurrence baseline


## 1. Introduction

*Objective.* Recommend a next item to add to a basket, using associations that hold up over time.

Association rules are usually reported with support, confidence and lift computed on the same transactions they were mined from. Measured that way, every rule above its thresholds looks valid by construction.

The questions a retailer actually has are different: *if I recommend from these rules, will customers want what I suggest?* and *will the rules still be true next season?* So rules are mined from purchases before September 2011 and judged on the following Christmas season, in two ways:

- **Recommendation.** One item is hidden from every later basket, and each method recommends five items from the rest. A hit means the hidden item was among them.
- **Survival.** Every rule is recomputed on the later baskets to see whether its association still exists.

Rules compete against two baselines, a random guess and the current bestsellers, and against plain item co-occurrence, which learns conditional probabilities but needs no mining or thresholds.

| Finding | Value | Note |
|---|---|---|
| Primary metric | Hit rate @ 5 | hidden item recovered |
| Robustness | Rule survival | lift still above 1 next season |
| Baselines | Random · Bestsellers |  |
| Threshold-free rival | Item co-occurrence |  |

> Phase gate passed: Held-out test, survival definition and baselines fixed before mining.


## 2. Data

*Objective.* Turn transaction lines into baskets, keeping orders from customers without an ID.

The shared Online Retail II loader reduces 1,067,371 lines to genuine product sales. Unlike segmentation, market-basket analysis keeps orders with no customer ID: a basket is still a basket.

| Rule | Removed | Share | Remaining |
|---|---|---|---|
| exact duplicate lines (includes the 9-day overlap between sheets) | 34,335 | 3.22% | 1,033,036 |
| cancellation lines (kept aside for return rates) | 19,104 | 1.79% | 1,013,932 |
| non-product codes (postage, fees, adjustments, vouchers) | 4,791 | 0.45% | 1,009,141 |
| non-positive quantity or price | 5,928 | 0.56% | 1,003,213 |

Before September 2011 there are **32,126 baskets** over 4,604 products, with a median of 15 distinct products per basket. Purchases are very long-tailed: only 691 products appear in at least 1% of baskets.

| Finding | Value | Note |
|---|---|---|
| Training baskets | 32,126 |  |
| Test baskets | 7,012 |  |
| Products | 4,604 |  |
| Median basket size | 15 |  |

### Provenance

✅ **Real, third-party data**

| Field | Value |
|---|---|
| Dataset | Online Retail II (UCI) |
| Source | https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip |
| Licence | Creative Commons Attribution 4.0 (CC BY 4.0) |
| Rows × columns (as downloaded) | 1,067,371 × 8 |
| SHA-256 | `572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb` |
| Products in training | 4,604 |
| Test baskets | 7,012 |
| Training baskets | 32,126 |

> Phase gate passed: 32,126 training and 7,012 test baskets.


## 3. Method

*Objective.* Build an evaluation no method can pass by seeing the answer.

Training covers purchases before 01 September 2011; testing starts a week later, on 08 September 2011. From each of the **6,548 test baskets** with at least two products, one product is hidden at random and the rest are shown to the recommender.

**9.9%** of hidden products never appeared in training at all — new Christmas stock, mostly. No method can recommend a product it has never seen, so these count as misses for everyone. They are kept in rather than filtered out, because filtering them would flatter every method equally and hide a real limit of mining past purchases.

Two project-specific checks run alongside the standard ones: no invoice appears on both sides, and no hidden product is visible to the model.

| Finding | Value | Note |
|---|---|---|
| Test baskets scored | 6,548 |  |
| Hidden products never seen in training | 9.9% |  |
| Embargo | 7 days |  |

> Phase gate passed: Temporal, disjointness, invoice and hidden-item assertions all executed.


## 4. Models

*Objective.* Compare mined rules at three thresholds with baselines that need no mining.

FP-Growth was run at three support and confidence thresholds, from strict to permissive, producing between 107 and 11,617 single-consequent rules.

Stricter thresholds give fewer, stronger rules, but a rule can only recommend when its antecedent is in the basket, so strict rule sets stay silent for many baskets. **Coverage** in the leaderboard is the share of baskets where a method produced any recommendation of its own. The hybrid fills the silence with bestsellers, which separates the value of the rules from the value of always saying something.

| Finding | Value | Note |
|---|---|---|
| Recommenders | 7 |  |
| Winner | Item co-occurrence |  |
| Hit rate @ 5 | 18.63% |  |
| Most rules | 11,617 |  |

**Table 1.** Tournament results.

| # | Recommender | Hit rate @ 5 | MRR @ 5 | Coverage | Rules | Rules held next season |
|---|---|---|---|---|---|---|
| 1 | **Item co-occurrence** 🏆 | 18.6% | 0.1154 | 98.2% | — | — |
| 2 | Rules + bestseller fallback | 14.4% | 0.0913 | 92.7% | 11,617 | 99.4% |
| 3 | FP-Growth rules (s≥0.5%, c≥10%) | 14.3% | 0.0910 | 92.7% | 11,617 | 99.4% |
| 4 | FP-Growth rules (s≥1.0%, c≥20%) | 10.0% | 0.0680 | 85.2% | 1,147 | 99.8% |
| 5 | FP-Growth rules (s≥2.0%, c≥30%) | 3.3% | 0.0252 | 45.4% | 107 | 100.0% |
| 6 | Bestsellers *(baseline)* | 1.9% | 0.0096 | 100.0% | — | — |
| 7 | Random products *(baseline)* | 0.1% | 0.0006 | 100.0% | — | — |

Ranked by hit rate @ 5 (higher is better). Split: chronological with 7-day embargo, one item hidden per test basket; 6,548 evaluation rows. Generated from `artifacts/metrics.json`.

> Phase gate passed: 7 recommenders scored on 6,548 held-out baskets.


## 5. Results

*Objective.* Establish whether mined rules add anything a retailer could not get from a bestseller list.

**Item co-occurrence** recovers the hidden product in 18.63% of later baskets, against 1.88% for simply recommending the bestsellers.

Of the 11,617 rules in the most permissive set, 11,411 had antecedents common enough in the test season to judge. **99.4%** of those still showed a positive association (lift above 1), and their confidence moved by 0.067 on average between seasons. The median rule kept 101% of its training confidence.

That survival rate should not be read as a discovery. Lift above 1 is a low bar, and the confidence figures are the stronger evidence. More revealing still is what the strongest rules say: the top rule by lift is *HERB MARKER THYME + HERB MARKER PARSLEY* → *HERB MARKER CHIVES*. Many of the strongest associations link variants of one product line. Those are durable precisely because they are obvious, and a merchandiser would not need a mining algorithm to find them.

Read the absolute hit rates with the unseen-product share in mind: a hidden product that never existed before September counts as a miss for every method, so no method can approach 100%.

| Finding | Value | Note |
|---|---|---|
| Best hit rate @ 5 | 18.63% |  |
| Bestsellers hit rate | 1.88% |  |
| Rules surviving next season | 99.4% |  |
| Mean confidence drift | 0.067 |  |

> Phase gate passed: No critical findings; every recommendation judged on a later season.

## 6. Threats to validity

### Assertions executed during training

| Assertion | Result | What it measured |
|---|---|---|
| `assert_temporal_split` | ✅ passed | Last training timestamp 2011-08-31 17:30:00 precedes first test timestamp 2011-09-08 08:56:00 by 7 days 15:26:00; embargo of 6D required. |
| `assert_disjoint` | ✅ passed | 789,736 training and 203,765 test row indices compared; 0 appear in both. |
| `assert_no_shared_invoices` | ✅ passed | 32,126 training and 7,012 test invoices compared; 0 in both. |
| `assert_hidden_item_not_visible` | ✅ passed | 6,548 test baskets checked; the hidden item was visible to the model in 0. |

Recorded by the assertion ledger while training ran; generated from `artifacts/metrics.json`.

### Limitations

- The test season is Christmas, so seasonal products are over-represented among the hidden items.
- One hidden item per basket is chosen at random, which weights every product in a basket equally.
- Lift above one is a lenient definition of a rule surviving; confidence retained is the stronger evidence.


## 7. Deployment

*Objective.* Recommend for a basket, and show which rules justify each suggestion.

`POST /api/recommend` takes the products in a basket and returns five suggestions from the winning method. `GET /api/rules` returns the strongest rules with both their training statistics and their measured behaviour in the following season, so a merchandiser can see which associations are durable and which were seasonal.

| Finding | Value | Note |
|---|---|---|
| Served recommender | Item co-occurrence |  |
| Serving port | 8,004 |  |

> Phase gate passed: Winning recommender, bestseller fallback and product names persisted.

## 8. Conclusion

- Rules scored on the baskets they were mined from always look valid. Judging them on a later season is what makes the numbers mean something.
- Plain item co-occurrence, with no mining and no thresholds, recommended better than every mined rule set.
- Stricter rule thresholds produce stronger rules that stay silent for most baskets; coverage has to be reported beside accuracy.
- The most durable associations were often variants of a single product line, which are stable because they are obvious.
- Iterating a Python set follows randomised string hashing, so tie-breaking has to be explicit or results change between identical runs.

## Reproducibility

Trained 2026-09-16T23:40:24Z in 44.4 s at commit `3d7f8d5` with Python 3.11.6, NumPy 2.4.6, pandas 3.0.5 and scikit-learn 1.9.1.

```bash
python ml/train.py
python -m uvicorn main:app --app-dir server --port 8004
npm run dev -w 04_market_basket/client
```

## References

1. Chen, D. (2019). Online Retail II [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D
2. Agrawal, R., Imieliński, T., & Swami, A. (1993). Mining association rules between sets of items in large databases. ACM SIGMOD.
3. Han, J., Pei, J., & Yin, Y. (2000). Mining frequent patterns without candidate generation. ACM SIGMOD.
4. Linden, G., Smith, B., & York, J. (2003). Amazon.com recommendations: Item-to-item collaborative filtering. IEEE Internet Computing, 7(1), 76–80.
5. Raschka, S. (2018). MLxtend: Providing machine learning and data science utilities and extensions to Python's scientific computing stack. Journal of Open Source Software, 3(24), 638.

---

*Generated from this project's artifacts by `scripts/generate_docs.py`.*
