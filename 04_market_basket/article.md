# Which products go together, and do those associations hold next season?

*Market Basket Mining — project 04 of a sixteen-project data science portfolio built with an AI coding assistant, on real data.*

![Market Basket Mining](./screenshots/01_build_basket.png)

## The question

Which products go together strongly enough to recommend, and do those associations still hold next season?

Association rules are usually reported with support, confidence and lift computed on the same transactions they were mined from. Measured that way, every rule above its thresholds looks valid by construction.

The questions a retailer actually has are different: *if I recommend from these rules, will customers want what I suggest?* and *will the rules still be true next season?* So rules are mined from purchases before September 2011 and judged on the following Christmas season, in two ways:

- **Recommendation.** One item is hidden from every later basket, and each method recommends five items from the rest. A hit means the hidden item was among them.
- **Survival.** Every rule is recomputed on the later baskets to see whether its association still exists.

Rules compete against two baselines, a random guess and the current bestsellers, and against plain item co-occurrence, which learns conditional probabilities but needs no mining or thresholds.

## The data

The shared Online Retail II loader reduces 1,067,371 lines to genuine product sales. Unlike segmentation, market-basket analysis keeps orders with no customer ID: a basket is still a basket.

| Rule | Removed | Share | Remaining |
|---|---|---|---|
| exact duplicate lines (includes the 9-day overlap between sheets) | 34,335 | 3.22% | 1,033,036 |
| cancellation lines (kept aside for return rates) | 19,104 | 1.79% | 1,013,932 |
| non-product codes (postage, fees, adjustments, vouchers) | 4,791 | 0.45% | 1,009,141 |
| non-positive quantity or price | 5,928 | 0.56% | 1,003,213 |

Before September 2011 there are **32,126 baskets** over 4,604 products, with a median of 15 distinct products per basket. Purchases are very long-tailed: only 691 products appear in at least 1% of baskets.

## What won

**Item co-occurrence**, at 18.6% on hit rate @ 5, against 1.9% for the best baseline, Bestsellers.

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

**Item co-occurrence** recovers the hidden product in 18.63% of later baskets, against 1.88% for simply recommending the bestsellers.

Of the 11,617 rules in the most permissive set, 11,411 had antecedents common enough in the test season to judge. **99.4%** of those still showed a positive association (lift above 1), and their confidence moved by 0.067 on average between seasons. The median rule kept 101% of its training confidence.

That survival rate should not be read as a discovery. Lift above 1 is a low bar, and the confidence figures are the stronger evidence. More revealing still is what the strongest rules say: the top rule by lift is *HERB MARKER THYME + HERB MARKER PARSLEY* → *HERB MARKER CHIVES*. Many of the strongest associations link variants of one product line. Those are durable precisely because they are obvious, and a merchandiser would not need a mining algorithm to find them.

Read the absolute hit rates with the unseen-product share in mind: a hidden product that never existed before September counts as a miss for every method, so no method can approach 100%.

## What I took away

- Rules scored on the baskets they were mined from always look valid. Judging them on a later season is what makes the numbers mean something.
- Plain item co-occurrence, with no mining and no thresholds, recommended better than every mined rule set.
- Stricter rule thresholds produce stronger rules that stay silent for most baskets; coverage has to be reported beside accuracy.
- The most durable associations were often variants of a single product line, which are stable because they are obvious.
- Iterating a Python set follows randomised string hashing, so tie-breaking has to be explicit or results change between identical runs.

## How it is built

The backend is FastAPI on port 8004 and the front end is React on port 5104. Most of the interface — the metrics panel, tournament table, CRISP-DM walkthrough, audit panel and charts — is shared across the portfolio and driven entirely by the training run's artifacts, so nothing on screen is typed by hand.

Code: [github.com/aniket-alt/ai-coding-assistant](https://github.com/aniket-alt/ai-coding-assistant/tree/main/04_market_basket)

---

*Figures in this article are generated from the project's artifacts.*
