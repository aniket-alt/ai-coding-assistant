# DS Visual Foundations

*Probability, evaluation, gradient descent and backpropagation, on real data and checked.*

**Aniket Anil Naik** · Project 08 of the data science full-stack portfolio

## Abstract

**Can the foundations of machine learning -- probability, evaluation, gradient descent and backpropagation -- be taught on real data, with every mechanism checked?**

We study this question on *SMS Spam Collection (UCI)*: 5,574 rows as downloaded, used under Creative Commons Attribution 4.0 (CC BY 4.0). Following the CRISP-DM process, we compare 4 approaches, 2 of them baselines, on 1,035 evaluation rows and rank them by average precision. Evaluation split: deduplicated messages, stratified random 80/20.

The best approach, **Logistic regression, TF-IDF**, reaches 0.9729 against 0.8333 for the strongest baseline (Contains a 5+ digit number).

4 of 4 leakage assertions passed while training ran, and static analysis of the source reported 0 critical and 0 warning findings.

**Keywords:** SMS spam classification (lesson 1 and 2), Naive Bayes, written out, thresholds and precision-recall, gradient descent and its stability limit, backpropagation checked by finite differences


## 1. Introduction

*Objective.* Teach four foundations interactively, each on a real dataset and verified against a reference.

Foundations are usually taught on toy numbers, so the lessons never meet the problems real data brings: duplicated records, features on wildly different scales, and thresholds that trade one error for another. Each lesson here runs on a real dataset, and each algorithm is written out and then checked against a trusted implementation, so what is taught is shown to be correct.

| Finding | Value | Note |
|---|---|---|
| Lessons | 4 |  |
| Datasets | SMS Spam Collection, Combined Cycle Power Plant |  |

> Phase gate passed: Four lessons, two datasets and three reference checks fixed before building.


## 2. Data

*Objective.* Find what in each dataset would mislead a naive lesson.

**SMS.** 5,574 messages, 13.4% spam. 403 are exact copies of an earlier message, and copies are disproportionately spam (26% of duplicated messages, against 13.4% overall): the same promotion sent to many people.

**Power plant.** 9,568 hourly readings. Ambient pressure averages 1,013 mbar while temperature averages 19.7 °C, so unscaled features differ by two orders of magnitude.

| Finding | Value | Note |
|---|---|---|
| Duplicate messages | 403 |  |
| Spam share | 13.4% |  |

### Provenance

✅ **Real, third-party data**

| Field | Value |
|---|---|
| Dataset | SMS Spam Collection (UCI) |
| Source | https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip |
| Licence | Creative Commons Attribution 4.0 (CC BY 4.0) |
| Rows × columns (as downloaded) | 5,574 × 3 |
| SHA-256 | `1587ea43e58e82b14ff1f5425c88e17f8496bfcdb67a583dbff9eefaf9963ce3` |

> Phase gate passed: Duplicates counted; feature scales recorded.


## 3. Method

*Objective.* Split so that no message is tested on after being trained on.

Messages are deduplicated before an 80/20 stratified split, leaving 4,136 to train and 1,035 to test, with no text in both.

Measured over 20 splits: when duplicates are kept, 10.5% of test messages already appear in training, and Naive Bayes recall is 92.8% against 91.7% when messages are deduplicated first.

| Finding | Value | Note |
|---|---|---|
| Test messages seen in training, duplicates kept | 10.5% |  |

> Phase gate passed: Executed assertion: no message text in both splits.


## 4. Models

*Objective.* Write out each algorithm and check it against a trusted implementation.

**Naive Bayes** is written out with add-one smoothing; its spam probabilities match scikit-learn's to within 2e-14. **Gradient descent** on scaled power-plant features reaches the least-squares coefficients to a relative difference of 1e-16. **Backpropagation** in a 4-8-1 tanh network matches finite differences to 1e-07 over all 49 parameters.

| Finding | Value | Note |
|---|---|---|
| Naive Bayes vs scikit-learn | 2e-14 |  |
| Backprop vs finite differences | 1e-07 |  |

**Table 1.** Tournament results.

| # | Spam filter | Average precision | F1 at 0.5 | Precision | Recall |
|---|---|---|---|---|---|
| 1 | **Logistic regression, TF-IDF** 🏆 | 0.9729 | 0.911 | 96.6% | 86.3% |
| 2 | Naive Bayes, written out | 0.9654 | 0.941 | 96.8% | 91.6% |
| 3 | Contains a 5+ digit number *(baseline)* | 0.8333 | 0.895 | 100.0% | 80.9% |
| 4 | Everything is ham *(baseline)* | 0.1266 | 0.000 | 0.0% | 0.0% |

Ranked by average precision (higher is better). Split: deduplicated messages, stratified random 80/20; 1,035 evaluation rows. Generated from `artifacts/metrics.json`.

> Phase gate passed: Three executed reference checks.


## 5. Results

*Objective.* Report what each lesson's analysis actually found.

**Spam.** Logistic regression, TF-IDF has the highest average precision, 0.973; Naive Bayes reaches 0.965 with recall 91.6% at a 0.5 threshold.

**Scaling.** Gradient descent is stable only below a learning rate of 0.410 on scaled features and 9.7e-07 on raw ones. After 500 steps, scaled features at learning rate 0.1 reach a training MSE of 20.89; raw features, at just under their own limit, reach 177.8. The exact minimum is 20.89.

**Networks.** The trained network's test RMSE is 4.13 MW against 4.50 MW for linear least squares and 17.03 MW for predicting the mean.

| Finding | Value | Note |
|---|---|---|
| Best average precision | 0.973 |  |
| Stable learning rate, scaled / raw | 0.410 / 9.7e-07 |  |

> Phase gate passed: Baselines beside every result.

## 6. Threats to validity

### Assertions executed during training

| Assertion | Result | What it measured |
|---|---|---|
| `assert_no_message_in_both_splits` | ✅ passed | 4,136 training and 1,035 test messages after removing 403 duplicates; 0 message texts appear in both. |
| `assert_naive_bayes_matches_sklearn` | ✅ passed | Spam probabilities for 1,035 test messages from the written-out Naive Bayes and scikit-learn's MultinomialNB; largest difference 1.6e-14 (limit 1e-9). |
| `assert_gradient_descent_reaches_least_squares` | ✅ passed | Gradient descent (learning rate 0.1, 3,000 steps, scaled features) against the exact least-squares solution: largest coefficient difference 1.1e-16 relative (limit 1e-6). |
| `assert_backprop_matches_finite_differences` | ✅ passed | All 49 parameter gradients compared with central finite differences, before and after training; largest relative error 1.1e-07 (limit 1e-5). |

Recorded by the assertion ledger while training ran; generated from `artifacts/metrics.json`.

### Limitations

- Duplicates are detected by exact text only; near-copies that differ by a word or a number remain on both sides of the split.
- The duplicate-leak experiment uses one model, Naive Bayes, over twenty splits; other models may be affected differently.
- The stable learning rate is exact for batch gradient descent on squared error only; stochastic or adaptive optimisers behave differently.
- The neural network is deliberately tiny, so its comparison with linear regression says little about larger networks.


## 7. Deployment

*Objective.* Serve the lessons live from the checked implementations.

The API loads the trained Naive Bayes and network and imports ml/lessons.py, so classifying a typed message, moving a threshold, rerunning gradient descent at a new learning rate, or stepping through backpropagation all run the checked code.

| Finding | Value | Note |
|---|---|---|
| Serving port | 8,008 |  |

> Phase gate passed: Interactive views call the same functions the checks covered.

## 8. Conclusion

- Writing an algorithm out and matching it against a trusted library to many decimal places is a cheap, strong test.
- Removing duplicated messages before splitting changed the measured scores only slightly on this data, but it is the only way to know that.
- Feature scaling decides whether gradient descent is practical: it moved the largest stable learning rate by several orders of magnitude.
- The learning-rate limit computed from the largest eigenvalue held in practice: gradient descent converged at 99% of it and diverged at 101%, with and without scaling.
- Finite differences verify backpropagation without trusting any framework.

## Reproducibility

Trained 2026-09-17T16:12:40Z in 13.6 s at commit `1e3bb26` with Python 3.11.6, NumPy 2.4.6, pandas 3.0.5 and scikit-learn 1.9.1.

```bash
python ml/train.py
python -m uvicorn main:app --app-dir server --port 8008
npm run dev -w 08_visual_foundations/client
```

## References

1. Almeida, T. A., Gómez Hidalgo, J. M., & Yamakami, A. (2011). Contributions to the study of SMS spam filtering: New collection and results. Proceedings of DocEng 2011, 259-262.
2. Almeida, T. A., Gómez Hidalgo, J. M., & Yamakami, A. (2011). Contributions to the study of SMS spam filtering. Proceedings of DocEng 2011.
3. Tüfekci, P. (2014). Prediction of full load electrical power output of a base load operated combined cycle power plant using machine learning methods. International Journal of Electrical Power & Energy Systems, 60, 126–140.
4. McCallum, A., & Nigam, K. (1998). A comparison of event models for Naive Bayes text classification. AAAI-98 Workshop on Learning for Text Categorization.
5. Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. Nature, 323, 533–536.
6. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning, chapters 4 and 6. MIT Press.
7. Davis, J., & Goadrich, M. (2006). The relationship between precision-recall and ROC curves. Proceedings of ICML 2006.

---

*Generated from this project's artifacts by `scripts/generate_docs.py`.*
