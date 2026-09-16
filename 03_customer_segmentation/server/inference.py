"""
Project 03 -- serving.

Assigns a customer to the nearest segment of the winning segmentation and
returns what that segment went on to do. Every profile figure comes from the
training run's metrics.json; nothing is recalculated here.
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

from features import LOG_FEATURES  # noqa: E402


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


def segments(model: str | None = None) -> dict[str, Any]:
    """Segment profiles for one segmentation in the tournament."""
    rec = run_record()
    name = model or rec["best_model"]
    profiles = rec["extra"]["segments"].get(name)
    if profiles is None:
        raise KeyError(name)
    return {"model": name, "served": name == rec["best_model"], "segments": profiles}


def feature_ranges() -> dict[str, Any]:
    return run_record()["extra"]["feature_ranges"]


def assign(customer: dict[str, float]) -> dict[str, Any]:
    """Nearest segment centre in the scaled feature space of the winning method."""
    b = bundle()
    row = np.array([[float(customer[f]) for f in b["features"]]])
    for i, f in enumerate(b["features"]):
        if f in LOG_FEATURES:
            row[0, i] = np.log1p(max(row[0, i], 0.0))
    import pandas as pd
    z = b["scaler"].transform(pd.DataFrame(row, columns=b["features"]))[0]

    distances = {seg: float(np.linalg.norm(z - np.array(c))) for seg, c in b["centroids"].items()}
    nearest = min(distances, key=distances.get)
    prof = next(p for p in b["profiles"] if p["segment_id"] == nearest)
    return {
        "model": b["model_name"],
        "segment": prof,
        "distance": round(distances[nearest], 3),
        "distances": {
            next(p["label"] for p in b["profiles"] if p["segment_id"] == s): round(d, 3)
            for s, d in sorted(distances.items(), key=lambda kv: kv[1])
        },
    }
