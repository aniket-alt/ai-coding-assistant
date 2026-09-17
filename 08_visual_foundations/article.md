# Probability, evaluation, gradient descent and backpropagation, on real data and checked.

*DS Visual Foundations — project 08 of a sixteen-project data science portfolio built with an AI coding assistant, on real data.*

![DS Visual Foundations](./screenshots/01_probability.png)

## The question

Can the foundations of machine learning -- probability, evaluation, gradient descent and backpropagation -- be taught on real data, with every mechanism checked?

Foundations are usually taught on toy numbers, so the lessons never meet the problems real data brings: duplicated records, features on wildly different scales, and thresholds that trade one error for another. Each lesson here runs on a real dataset, and each algorithm is written out and then checked against a trusted implementation, so what is taught is shown to be correct.

## The data

**SMS.** 5,574 messages, 13.4% spam. 403 are exact copies of an earlier message, and copies are disproportionately spam (26% of duplicated messages, against 13.4% overall): the same promotion sent to many people.

**Power plant.** 9,568 hourly readings. Ambient pressure averages 1,013 mbar while temperature averages 19.7 °C, so unscaled features differ by two orders of magnitude.

## What won

**Logistic regression, TF-IDF**, at 0.9729 on average precision, against 0.8333 for the best baseline, Contains a 5+ digit number.

| # | Spam filter | Average precision | F1 at 0.5 | Precision | Recall |
|---|---|---|---|---|---|
| 1 | **Logistic regression, TF-IDF** 🏆 | 0.9729 | 0.911 | 96.6% | 86.3% |
| 2 | Naive Bayes, written out | 0.9654 | 0.941 | 96.8% | 91.6% |
| 3 | Contains a 5+ digit number *(baseline)* | 0.8333 | 0.895 | 100.0% | 80.9% |
| 4 | Everything is ham *(baseline)* | 0.1266 | 0.000 | 0.0% | 0.0% |

Ranked by average precision (higher is better). Split: deduplicated messages, stratified random 80/20; 1,035 evaluation rows. Generated from `artifacts/metrics.json`.

**Spam.** Logistic regression, TF-IDF has the highest average precision, 0.973; Naive Bayes reaches 0.965 with recall 91.6% at a 0.5 threshold.

**Scaling.** Gradient descent is stable only below a learning rate of 0.410 on scaled features and 9.7e-07 on raw ones. After 500 steps, scaled features at learning rate 0.1 reach a training MSE of 20.89; raw features, at just under their own limit, reach 177.8. The exact minimum is 20.89.

**Networks.** The trained network's test RMSE is 4.13 MW against 4.50 MW for linear least squares and 17.03 MW for predicting the mean.

## What I took away

- Writing an algorithm out and matching it against a trusted library to many decimal places is a cheap, strong test.
- Removing duplicated messages before splitting changed the measured scores only slightly on this data, but it is the only way to know that.
- Feature scaling decides whether gradient descent is practical: it moved the largest stable learning rate by several orders of magnitude.
- The learning-rate limit computed from the largest eigenvalue held in practice: gradient descent converged at 99% of it and diverged at 101%, with and without scaling.
- Finite differences verify backpropagation without trusting any framework.

## How it is built

The backend is FastAPI on port 8008 and the front end is React on port 5108. Most of the interface — the metrics panel, tournament table, CRISP-DM walkthrough, audit panel and charts — is shared across the portfolio and driven entirely by the training run's artifacts, so nothing on screen is typed by hand.

Code: [github.com/aniket-alt/ai-coding-assistant](https://github.com/aniket-alt/ai-coding-assistant/tree/main/08_visual_foundations)

---

*Figures in this article are generated from the project's artifacts.*
