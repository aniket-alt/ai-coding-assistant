# Abstract — MLOps & Governance

**Once project 01's taxi model is in service, how fast does it go stale, and which retraining policy keeps it accurate at the lowest cost?**

We study this question on *NYC TLC Yellow Taxi Trip Records — January to July 2024*: 23,408,996 rows as downloaded, used under NYC Taxi & Limousine Commission open data (public domain). Following the CRISP-DM process, we compare 6 approaches, 2 of them baselines, on 900,000 evaluation rows and rank them by mean rmse (min). Evaluation split: month-by-month, each policy trained only on finished months.

The best approach, **Retrain monthly on all history**, reaches 6.396 against 8.322 for the strongest baseline (Route median lookup, refreshed monthly).

3 of 3 leakage assertions passed while training ran, and static analysis of the source reported 0 critical and 0 warning findings.

**Keywords:** model monitoring and retraining policy, month-by-month retraining policies, population stability index, TreeSHAP contributions, measured load test

---

*Generated from this project's artifacts by `scripts/generate_docs.py`. Figures are not typed by hand.*
