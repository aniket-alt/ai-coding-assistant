"""
Project 10 -- serving the course.

Every lesson is read from the training run's metrics.json. Quiz questions are
sent without their answers; a choice is checked on the server, so the answer
cannot be read from the page before it is attempted.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"


class ModelNotTrained(RuntimeError):
    """Raised when the API starts before `python ml/train.py` has run."""


@lru_cache(maxsize=1)
def run_record() -> dict[str, Any]:
    p = ARTIFACTS / "metrics.json"
    if not p.exists():
        raise ModelNotTrained("artifacts/metrics.json not found -- run ml/train.py.")
    return json.loads(p.read_text(encoding="utf-8"))


def extra() -> dict[str, Any]:
    return run_record()["extra"]


def eda() -> dict[str, Any]:
    return extra()["eda"]


def unsupervised() -> dict[str, Any]:
    e = extra()
    return {"clusters": e["clusters"], "silhouette": e["cluster_silhouette"], "outliers": e["outliers"]}


def fairness() -> dict[str, Any]:
    rec = run_record()
    keys = ("roc_auc", "balanced_accuracy", "accuracy", "disparate_impact_sex", "equal_opportunity_gap_sex",
            "false_positive_gap_sex", "selection_rate_men", "selection_rate_women")
    return {
        "models": [{"model": m["name"], "family": m["family"], **{k: m["metrics"].get(k) for k in keys}}
                   for m in rec["models"] if m["family"] != "baseline"],
        "proxy_auc": rec["extra"]["proxy_auc"],
        "relationship_proxy": rec["extra"]["relationship_proxy"],
    }


def rules() -> list[dict[str, Any]]:
    return extra()["rules"]


def lsh() -> dict[str, Any]:
    return extra()["lsh"]


def quiz() -> list[dict[str, Any]]:
    return [{k: q[k] for k in ("id", "phase", "question", "options")} for q in extra()["quiz"]]


def check_answer(question_id: str, choice: int) -> dict[str, Any]:
    q = next((q for q in extra()["quiz"] if q["id"] == question_id), None)
    if q is None:
        raise KeyError(question_id)
    return {"correct": choice == q["answer"], "answer": q["answer"], "explanation": q["explanation"]}
