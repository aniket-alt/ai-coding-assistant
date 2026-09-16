"""
Project 04 -- serving.

Recommends products for a basket with the tournament's winning method, and
explains each suggestion by the basket item that most strongly implies it.
Rules and their next-season statistics come from metrics.json.
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
sys.path.insert(0, str(ROOT / "ml"))       # the pickled recommender's classes live here
ARTIFACTS = ROOT / "artifacts"

K = 5


class ModelNotTrained(RuntimeError):
    """Raised when the API starts before `python ml/train.py` has run."""


@lru_cache(maxsize=1)
def bundle() -> dict[str, Any]:
    p = ARTIFACTS / "model.joblib"
    if not p.exists():
        raise ModelNotTrained("artifacts/model.joblib not found -- run ml/train.py.")
    return joblib.load(p)


@lru_cache(maxsize=1)
def run_record() -> dict[str, Any]:
    p = ARTIFACTS / "metrics.json"
    if not p.exists():
        raise ModelNotTrained("artifacts/metrics.json not found -- run ml/train.py.")
    return json.loads(p.read_text(encoding="utf-8"))


def items() -> list[dict[str, Any]]:
    """The most frequently bought products, as options for a basket picker."""
    return [{"id": i["code"], "label": f"{i['name']} ({i['support_pct']}% of baskets)"}
            for i in run_record()["extra"]["popular_items"]]


def rules() -> list[dict[str, Any]]:
    return run_record()["extra"]["rules"]


def recommend(codes: list[str]) -> dict[str, Any]:
    b = bundle()
    rec, names = b["recommender"], b["names"]
    visible = frozenset(codes)

    own = rec.own_recommendations(visible, K)
    out = rec.recommend(visible, K)
    if len(out) < K:
        out += b["fallback"].recommend(visible | frozenset(out), K - len(out))

    suggestions = []
    for code in out:
        entry = {"code": code, "name": names.get(code, code),
                 "source": "model" if code in own else "bestseller fallback"}
        # Co-occurrence can say which basket item implied the suggestion.
        if hasattr(rec, "conf") and code in rec.index:
            rows = [(i, rec.conf[rec.index[i], rec.index[code]]) for i in visible if i in rec.index]
            if rows:
                because, p = max(rows, key=lambda r: (r[1], r[0]))
                entry["because"] = names.get(because, because)
                entry["probability"] = round(float(p), 4)
        suggestions.append(entry)

    return {
        "model": b["model_name"],
        "basket": [{"code": c, "name": names.get(c, c)} for c in sorted(visible)],
        "suggestions": suggestions,
    }
