"""
Project 08 -- serving the lessons from the checked implementations.

Classification, thresholds, gradient descent and backpropagation are computed
on request with the functions in ml/lessons.py and the models training saved.
Everything else is read from metrics.json.
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))
ARTIFACTS = ROOT / "artifacts"

from lessons import gradient_descent, largest_stable_rate, mse, ols  # noqa: E402

MAX_STEPS = 2000


class ModelNotTrained(RuntimeError):
    """Raised when the API starts before `python ml/train.py` has run."""


@lru_cache(maxsize=1)
def bundle() -> dict[str, Any]:
    p = ARTIFACTS / "model.joblib"
    if not p.exists():
        raise ModelNotTrained("artifacts/model.joblib not found -- run ml/train.py.")
    return joblib.load(p)


@lru_cache(maxsize=1)
def extra() -> dict[str, Any]:
    p = ARTIFACTS / "metrics.json"
    if not p.exists():
        raise ModelNotTrained("artifacts/metrics.json not found -- run ml/train.py.")
    return json.loads(p.read_text(encoding="utf-8"))["extra"]


def classify(text: str) -> dict[str, Any]:
    return bundle()["nb"].explain(text)


def spam_lesson() -> dict[str, Any]:
    e = extra()
    return {"sms": e["sms"], "leak": e["leak"], "leak_seeds": e["leak_seeds"], "pr_curve": e["pr_curve"]}


def threshold(t: float) -> dict[str, Any]:
    """Confusion matrix of the written-out Naive Bayes on the held-out messages at threshold t."""
    rows = extra()["test_scores"]
    tp = sum(1 for r in rows if r["p"] >= t and r["spam"])
    fp = sum(1 for r in rows if r["p"] >= t and not r["spam"])
    fn = sum(1 for r in rows if r["p"] < t and r["spam"])
    tn = len(rows) - tp - fp - fn
    return {"threshold": t, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn) if tp + fn else None,
            "messages": len(rows)}


def descend(lr: float, steps: int, scaled: bool) -> dict[str, Any]:
    b = bundle()
    X = b["Xtr_s"] if scaled else b["Xtr"]
    y = b["ytr"]
    steps = min(steps, MAX_STEPS)
    run = gradient_descent(X, y, lr=lr, steps=steps, record_every=max(1, steps // 200))
    exact = ols(X, y)
    return {**run, "lr": lr, "steps": steps, "scaled": scaled, "exact_mse": mse(exact, X, y),
            "exact_weights": exact.tolist(), "stable_rate": largest_stable_rate(X),
            "features": list(extra()["gd"]["features"])}


def gd_lesson() -> dict[str, Any]:
    g = extra()["gd"]
    return {k: g[k] for k in ("features", "target", "exact_mse", "exact_test_rmse", "mean_baseline_test_rmse",
                              "stable_rate_scaled", "stable_rate_raw", "condition_scaled", "condition_raw",
                              "scaled_after_500", "raw_after_500", "runs")}


def backprop(index: int) -> dict[str, Any]:
    """One training row through the trained network, with every chain-rule factor."""
    b = bundle()
    mlp = b["mlp"]
    if not 0 <= index < len(b["Xtr_s"]):
        raise IndexError(index)
    X = b["Xtr_s"][index:index + 1]
    y_mean, y_sd = b["y"]
    target = (b["ytr"][index:index + 1] - y_mean) / y_sd
    grads, values = mlp.backward(X, target)
    rounded = lambda a: np.round(np.asarray(a, dtype=float), 5).tolist()
    return {
        "row": index, "features": list(extra()["gd"]["features"]),
        "inputs": rounded(X[0]), "raw_inputs": rounded(b["Xtr"][index]),
        "z1": rounded(values["z1"][0]), "h": rounded(values["h"][0]),
        "output": float(values["out"][0]), "target": float(target[0]),
        "output_mw": float(values["out"][0] * y_sd + y_mean), "target_mw": float(b["ytr"][index]),
        "loss": float((values["out"][0] - target[0]) ** 2),
        "d_out": float(values["d_out"][0]), "d_h": rounded(values["d_h"][0]), "d_z1": rounded(values["d_z1"][0]),
        "W2": rounded(mlp.W2[0]), "grad_W2": rounded(grads["W2"][0]), "grad_b2": float(grads["b2"][0]),
        "grad_W1": rounded(grads["W1"]), "grad_b1": rounded(grads["b1"]),
        "training_rows": int(len(b["Xtr_s"])),
    }


def backprop_lesson() -> dict[str, Any]:
    return extra()["backprop"]


def quiz() -> list[dict[str, Any]]:
    return [{k: q[k] for k in ("id", "phase", "question", "options")} for q in extra()["quiz"]]


def check_answer(question_id: str, choice: int) -> dict[str, Any]:
    q = next((q for q in extra()["quiz"] if q["id"] == question_id), None)
    if q is None:
        raise KeyError(question_id)
    return {"correct": choice == q["answer"], "answer": q["answer"], "explanation": q["explanation"]}

