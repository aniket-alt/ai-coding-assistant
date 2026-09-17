"""
Project 07 -- serving the tournament's servable winner, with the stack beside it.

Tuned LightGBM and the hand-built Caruana stack both run in this environment;
the one that scored better on the 2010 calls is served. Every response also
carries the stack's answer, with each member's score (the average of its five
fold models) and weight, so the ensemble can be traced to its parts. Every
other view is read from the training run's metrics.json.
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))
ARTIFACTS = ROOT / "artifacts"


class ModelNotTrained(RuntimeError):
    """Raised when the API starts before `python ml/train.py` has run."""


class InvalidCall(ValueError):
    """Raised when a prediction request is missing or misstates a feature."""


@lru_cache(maxsize=1)
def bundle() -> dict[str, Any]:
    p = ARTIFACTS / "model.joblib"
    if not p.exists():
        raise ModelNotTrained("artifacts/model.joblib not found -- run ml/train.py.")
    import features  # noqa: F401  -- the pickled pipelines reference this module

    return joblib.load(p)


@lru_cache(maxsize=1)
def run_record() -> dict[str, Any]:
    p = ARTIFACTS / "metrics.json"
    if not p.exists():
        raise ModelNotTrained("artifacts/metrics.json not found -- run ml/train.py.")
    return json.loads(p.read_text(encoding="utf-8"))


def extra() -> dict[str, Any]:
    return run_record()["extra"]


def form() -> dict[str, Any]:
    return extra()["form"]


def _row(payload: dict[str, Any]) -> pd.DataFrame:
    b = bundle()
    choices = form()["choices"]
    row = {}
    for f in b["features"]:
        if f not in payload:
            raise InvalidCall(f"missing feature '{f}'")
        v = payload[f]
        if f in b["categorical"]:
            if str(v) not in choices[f]:
                raise InvalidCall(f"'{v}' is not a recorded value of {f}")
            row[f] = str(v)
        else:
            try:
                row[f] = float(v)
            except (TypeError, ValueError) as exc:
                raise InvalidCall(f"{f} must be a number") from exc
    return pd.DataFrame([row])


def predict(payload: dict[str, Any]) -> dict[str, Any]:
    b = bundle()
    X = _row(payload)
    members = []
    for name, weight in sorted(b["weights"].items(), key=lambda r: -r[1]):
        p = float(np.mean([m.predict_proba(X)[:, 1] for m in b["fold_models"][name]]))
        members.append({"model": name, "probability": p, "weight": weight, "contribution": weight * p})
    stack = sum(m["contribution"] for m in members)
    tuned = float(np.mean([m.predict_proba(X)[:, 1] for m in b["tuned_models"]]))
    served = b["served"]
    shift = extra()["shift"]
    return {
        "probability": tuned if served == "LightGBM, tuned" else stack,
        "tuned_lightgbm": tuned,
        "stack": stack,
        "members": members,
        "train_rate_pct": shift["train_rate_pct"],
        "test_rate_pct": shift["test_rate_pct"],
        "served": served,
    }


def tournament() -> dict[str, Any]:
    e = extra()
    return {"significance": e["significance"], "estimate_check": e["estimate_check"],
            "autogluon": e["autogluon"], "leak": e["leak"]}


def stacking() -> dict[str, Any]:
    e = extra()
    return {"level_one": e["level_one"], "trajectory": e["caruana_trajectory"], "weights": e["weights"],
            "tuning_trace": e["tuning_trace"], "tuned_params": e["tuned_params"]}


def shift() -> dict[str, Any]:
    e = extra()
    return {"shift": e["shift"], "leak": e["leak"], "estimate_check": e["estimate_check"],
            "importance": e["importance"]}
