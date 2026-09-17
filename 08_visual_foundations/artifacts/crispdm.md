# CRISP-DM Report — DS Visual Foundations

**Business question.** Can the foundations of machine learning -- probability, evaluation, gradient descent and backpropagation -- be taught on real data, with every mechanism checked?


## 1. Business Understanding

*Objective:* Teach four foundations interactively, each on a real dataset and verified against a reference.


Foundations are usually taught on toy numbers, so the lessons never meet the problems real data brings: duplicated records, features on wildly different scales, and thresholds that trade one error for another. Each lesson here runs on a real dataset, and each algorithm is written out and then checked against a trusted implementation, so what is taught is shown to be correct.


| Finding | Value | Detail |
|---|---|---|
| Lessons | `4` |  |
| Datasets | `SMS Spam Collection, Combined Cycle Power Plant` |  |


*Code:* `ml/lessons.py`


> **Phase gate: PASSED.** Four lessons, two datasets and three reference checks fixed before building.


## 2. Data Understanding

*Objective:* Find what in each dataset would mislead a naive lesson.


**SMS.** 5,574 messages, 13.4% spam. 403 are exact copies of an earlier message, and copies are disproportionately spam (26% of duplicated messages, against 13.4% overall): the same promotion sent to many people.

**Power plant.** 9,568 hourly readings. Ambient pressure averages 1,013 mbar while temperature averages 19.7 °C, so unscaled features differ by two orders of magnitude.


| Finding | Value | Detail |
|---|---|---|
| Duplicate messages | `403` |  |
| Spam share | `13.4%` |  |


*Code:* `ml/data.py:load_sms`, `ml/data.py:load_power`


> **Phase gate: PASSED.** Duplicates counted; feature scales recorded.


## 3. Data Preparation

*Objective:* Split so that no message is tested on after being trained on.


Messages are deduplicated before an 80/20 stratified split, leaving 4,136 to train and 1,035 to test, with no text in both.

Measured over 20 splits: when duplicates are kept, 10.5% of test messages already appear in training, and Naive Bayes recall is 92.8% against 91.7% when messages are deduplicated first.


| Finding | Value | Detail |
|---|---|---|
| Test messages seen in training, duplicates kept | `10.5%` |  |


*Code:* `ml/train.py:main`


> **Phase gate: PASSED.** Executed assertion: no message text in both splits.


## 4. Modeling

*Objective:* Write out each algorithm and check it against a trusted implementation.


**Naive Bayes** is written out with add-one smoothing; its spam probabilities match scikit-learn's to within 2e-14. **Gradient descent** on scaled power-plant features reaches the least-squares coefficients to a relative difference of 1e-16. **Backpropagation** in a 4-8-1 tanh network matches finite differences to 1e-07 over all 49 parameters.


| Finding | Value | Detail |
|---|---|---|
| Naive Bayes vs scikit-learn | `2e-14` |  |
| Backprop vs finite differences | `1e-07` |  |


*Code:* `ml/lessons.py:NaiveBayes`, `ml/lessons.py:gradient_descent`, `ml/lessons.py:TinyMLP.backward`


> **Phase gate: PASSED.** Three executed reference checks.


## 5. Evaluation

*Objective:* Report what each lesson's analysis actually found.


**Spam.** Logistic regression, TF-IDF has the highest average precision, 0.973; Naive Bayes reaches 0.965 with recall 91.6% at a 0.5 threshold.

**Scaling.** Gradient descent is stable only below a learning rate of 0.410 on scaled features and 9.7e-07 on raw ones. After 500 steps, scaled features at learning rate 0.1 reach a training MSE of 20.89; raw features, at just under their own limit, reach 177.8. The exact minimum is 20.89.

**Networks.** The trained network's test RMSE is 4.13 MW against 4.50 MW for linear least squares and 17.03 MW for predicting the mean.


| Finding | Value | Detail |
|---|---|---|
| Best average precision | `0.973` |  |
| Stable learning rate, scaled / raw | `0.410 / 9.7e-07` |  |


*Code:* `ml/train.py:scores`


> **Phase gate: PASSED.** Baselines beside every result.


## 6. Deployment

*Objective:* Serve the lessons live from the checked implementations.


The API loads the trained Naive Bayes and network and imports ml/lessons.py, so classifying a typed message, moving a threshold, rerunning gradient descent at a new learning rate, or stepping through backpropagation all run the checked code.


| Finding | Value | Detail |
|---|---|---|
| Serving port | `8,008` |  |


*Code:* `server/inference.py`, `server/main.py`


> **Phase gate: PASSED.** Interactive views call the same functions the checks covered.
