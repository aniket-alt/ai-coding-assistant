# Abstract — Data Science Skills Lab

**For each technique in a working data scientist's catalogue, does doing it the careless way change the result -- and by how much, on real data?**

We study this question on *Titanic (OpenML 40945)*: 1,309 rows as downloaded, used under Public (OpenML). Following the CRISP-DM process, we compare 12 approaches, 0 of them baselines, and rank them by how much it flatters. Evaluation split: each skill repeats its own design over 25 seeds.

The best approach, **Reading accuracy on an imbalanced target**, reaches 0.2207.

3 of 3 leakage assertions passed while training ran, and static analysis of the source reported 0 critical and 0 warning findings.

**Keywords:** measured comparison of careless and careful practice, per skill, paired comparisons over many seeds, bootstrap intervals on the difference, leakage demonstrated rather than described, live re-runs of any entry

---

*Generated from this project's artifacts by `scripts/generate_docs.py`. Figures are not typed by hand.*
