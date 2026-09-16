# CRISP-DM Report — Market Basket Mining

**Business question.** Which products go together strongly enough to recommend, and do those associations still hold next season?


## 1. Business Understanding

*Objective:* Recommend a next item to add to a basket, using associations that hold up over time.


Association rules are usually reported with support, confidence and lift computed on the same transactions they were mined from. Measured that way, every rule above its thresholds looks valid by construction.

The questions a retailer actually has are different: *if I recommend from these rules, will customers want what I suggest?* and *will the rules still be true next season?* So rules are mined from purchases before September 2011 and judged on the following Christmas season, in two ways:

- **Recommendation.** One item is hidden from every later basket, and each method recommends five items from the rest. A hit means the hidden item was among them.
- **Survival.** Every rule is recomputed on the later baskets to see whether its association still exists.

Rules compete against two baselines, a random guess and the current bestsellers, and against plain item co-occurrence, which learns conditional probabilities but needs no mining or thresholds.


| Finding | Value | Detail |
|---|---|---|
| Primary metric | `Hit rate @ 5` | hidden item recovered |
| Robustness | `Rule survival` | lift still above 1 next season |
| Baselines | `Random · Bestsellers` |  |
| Threshold-free rival | `Item co-occurrence` |  |


*Code:* `ml/baskets.py:score_recommender`, `ml/baskets.py:rule_survival`


> **Phase gate: PASSED.** Held-out test, survival definition and baselines fixed before mining.


## 2. Data Understanding

*Objective:* Turn transaction lines into baskets, keeping orders from customers without an ID.


The shared Online Retail II loader reduces 1,067,371 lines to genuine product sales. Unlike segmentation, market-basket analysis keeps orders with no customer ID: a basket is still a basket.

| Rule | Removed | Share | Remaining |
|---|---|---|---|
| exact duplicate lines (includes the 9-day overlap between sheets) | 34,335 | 3.22% | 1,033,036 |
| cancellation lines (kept aside for return rates) | 19,104 | 1.79% | 1,013,932 |
| non-product codes (postage, fees, adjustments, vouchers) | 4,791 | 0.45% | 1,009,141 |
| non-positive quantity or price | 5,928 | 0.56% | 1,003,213 |

Before September 2011 there are **32,126 baskets** over 4,604 products, with a median of 15 distinct products per basket. Purchases are very long-tailed: only 691 products appear in at least 1% of baskets.


| Finding | Value | Detail |
|---|---|---|
| Training baskets | `32,126` |  |
| Test baskets | `7,012` |  |
| Products | `4,604` |  |
| Median basket size | `15` |  |


*Code:* `datasets/loaders/online_retail.py:clean`, `ml/baskets.py:to_baskets`


> **Phase gate: PASSED.** 32,126 training and 7,012 test baskets.


## 3. Data Preparation

*Objective:* Build an evaluation no method can pass by seeing the answer.


Training covers purchases before 01 September 2011; testing starts a week later, on 08 September 2011. From each of the **6,548 test baskets** with at least two products, one product is hidden at random and the rest are shown to the recommender.

**9.9%** of hidden products never appeared in training at all — new Christmas stock, mostly. No method can recommend a product it has never seen, so these count as misses for everyone. They are kept in rather than filtered out, because filtering them would flatter every method equally and hide a real limit of mining past purchases.

Two project-specific checks run alongside the standard ones: no invoice appears on both sides, and no hidden product is visible to the model.


| Finding | Value | Detail |
|---|---|---|
| Test baskets scored | `6,548` |  |
| Hidden products never seen in training | `9.9%` |  |
| Embargo | `7 days` |  |


*Code:* `ml/baskets.py:hold_out`, `ml/train.py:TEST_START`


> **Phase gate: PASSED.** Temporal, disjointness, invoice and hidden-item assertions all executed.


## 4. Modeling

*Objective:* Compare mined rules at three thresholds with baselines that need no mining.


FP-Growth was run at three support and confidence thresholds, from strict to permissive, producing between 107 and 11,617 single-consequent rules.

Stricter thresholds give fewer, stronger rules, but a rule can only recommend when its antecedent is in the basket, so strict rule sets stay silent for many baskets. **Coverage** in the leaderboard is the share of baskets where a method produced any recommendation of its own. The hybrid fills the silence with bestsellers, which separates the value of the rules from the value of always saying something.


| Finding | Value | Detail |
|---|---|---|
| Recommenders | `7` |  |
| Winner | `Item co-occurrence` |  |
| Hit rate @ 5 | `18.63%` |  |
| Most rules | `11,617` |  |


*Code:* `ml/baskets.py:RuleSet`, `ml/baskets.py:CoOccurrence`


> **Phase gate: PASSED.** 7 recommenders scored on 6,548 held-out baskets.


## 5. Evaluation

*Objective:* Establish whether mined rules add anything a retailer could not get from a bestseller list.


**Item co-occurrence** recovers the hidden product in 18.63% of later baskets, against 1.88% for simply recommending the bestsellers.

Of the 11,617 rules in the most permissive set, 11,411 had antecedents common enough in the test season to judge. **99.4%** of those still showed a positive association (lift above 1), and their confidence moved by 0.067 on average between seasons. The median rule kept 101% of its training confidence.

That survival rate should not be read as a discovery. Lift above 1 is a low bar, and the confidence figures are the stronger evidence. More revealing still is what the strongest rules say: the top rule by lift is *HERB MARKER THYME + HERB MARKER PARSLEY* → *HERB MARKER CHIVES*. Many of the strongest associations link variants of one product line. Those are durable precisely because they are obvious, and a merchandiser would not need a mining algorithm to find them.

Read the absolute hit rates with the unseen-product share in mind: a hidden product that never existed before September counts as a miss for every method, so no method can approach 100%.


| Finding | Value | Detail |
|---|---|---|
| Best hit rate @ 5 | `18.63%` |  |
| Bestsellers hit rate | `1.88%` |  |
| Rules surviving next season | `99.4%` |  |
| Mean confidence drift | `0.067` |  |


*Code:* `ml/baskets.py:rule_survival`


> **Phase gate: PASSED.** No critical findings; every recommendation judged on a later season.


## 6. Deployment

*Objective:* Recommend for a basket, and show which rules justify each suggestion.


`POST /api/recommend` takes the products in a basket and returns five suggestions from the winning method. `GET /api/rules` returns the strongest rules with both their training statistics and their measured behaviour in the following season, so a merchandiser can see which associations are durable and which were seasonal.


| Finding | Value | Detail |
|---|---|---|
| Served recommender | `Item co-occurrence` |  |
| Serving port | `8,004` |  |


*Code:* `server/inference.py`


> **Phase gate: PASSED.** Winning recommender, bestseller fallback and product names persisted.
