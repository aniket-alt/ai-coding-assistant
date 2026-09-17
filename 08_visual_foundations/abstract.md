# Abstract — DS Visual Foundations

**Can the foundations of machine learning -- probability, evaluation, gradient descent and backpropagation -- be taught on real data, with every mechanism checked?**

We study this question on *SMS Spam Collection (UCI)*: 5,574 rows as downloaded, used under Creative Commons Attribution 4.0 (CC BY 4.0). Following the CRISP-DM process, we compare 4 approaches, 2 of them baselines, on 1,035 evaluation rows and rank them by average precision. Evaluation split: deduplicated messages, stratified random 80/20.

The best approach, **Logistic regression, TF-IDF**, reaches 0.9729 against 0.8333 for the strongest baseline (Contains a 5+ digit number).

4 of 4 leakage assertions passed while training ran, and static analysis of the source reported 0 critical and 0 warning findings.

**Keywords:** SMS spam classification (lesson 1 and 2), Naive Bayes, written out, thresholds and precision-recall, gradient descent and its stability limit, backpropagation checked by finite differences

---

*Generated from this project's artifacts by `scripts/generate_docs.py`. Figures are not typed by hand.*
