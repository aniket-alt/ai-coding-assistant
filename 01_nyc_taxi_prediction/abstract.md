# Abstract — NYC Taxi Trip Duration

**When a rider requests a yellow cab in New York City, how long will the trip take?**

We study this question on *NYC TLC Yellow Taxi Trip Records — January 2024*: 2,964,624 rows as downloaded, used under NYC Taxi & Limousine Commission open data (public domain). Following the CRISP-DM process, we compare 7 approaches, 2 of them baselines, on 121,130 evaluation rows and rank them by rmse (min). Evaluation split: chronological with 1-day embargo.

The best approach, **LightGBM**, reaches 5.345 against 7.511 for the strongest baseline (Zone-pair lookup).

3 of 3 leakage assertions passed while training ran, and static analysis of the source reported 0 critical and 0 warning findings.

**Keywords:** regression, chronological split with embargo, out-of-fold target encoding, gradient boosting tournament, leakage assertions

---

*Generated from this project's artifacts by `scripts/generate_docs.py`. Figures are not typed by hand.*
