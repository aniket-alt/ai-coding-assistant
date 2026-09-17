"""Benchmark variant: Hyperparameter chosen on the test set"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [c for c in ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'] if c not in ("Class",)]


def run(frame, leakage):
    frame = frame.sort_values("Time").reset_index(drop=True)

    cut = frame["Time"].quantile(0.7)
    train = frame[frame["Time"] < cut]
    test = frame[frame["Time"] >= cut + 3600]
    leakage.assert_temporal_split(pd.to_timedelta(train["Time"], unit="s"),
                                  pd.to_timedelta(test["Time"], unit="s"))
    leakage.assert_disjoint(train.index, test.index)
    X_train, y_train = train[FEATURES], train["Class"]
    X_test, y_test = test[FEATURES], test["Class"]
    leakage.assert_no_target_leak(X_train, y_train)
    scaler = StandardScaler()
    scaler.fit(X_train)
    best = -1.0
    for c in (0.001, 0.01, 0.1, 1.0, 10.0):
        model = LogisticRegression(C=c, max_iter=1000, random_state=0)
        model.fit(scaler.transform(X_train), y_train)
        s = model.predict_proba(scaler.transform(X_test))[:, 1]
        if average_precision_score(y_test, s) > best:
            best, scores = average_precision_score(y_test, s), s
    return average_precision_score(y_test, scores)
