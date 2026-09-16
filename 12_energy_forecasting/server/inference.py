"""
Project 12 -- serving.

Every response is read from the training run's metrics.json: tomorrow's band,
the recent track record, load profiles, calibration, significance and the
leakage demonstration. Nothing is recomputed per request.
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


def forecast() -> dict[str, Any]:
    e = run_record()["extra"]
    recent = e["recent_window"]
    coverage = {}
    for days in (7, 14):
        window = recent[-24 * days:]
        inside = sum(1 for r in window if r["q10"] <= r["actual"] <= r["q90"])
        coverage[str(days)] = {"hours": len(window), "inside": inside,
                               "pct": round(inside / len(window) * 100, 1) if window else None}
    return {"model": e["recent_model"], "recent": recent, "next_day": e["next_day"],
            "recent_coverage": coverage}


def patterns() -> dict[str, Any]:
    e = run_record()["extra"]
    return {**e["profiles"], "summary": e["summary"], "facts": e["facts"]}


def calibration() -> dict[str, Any]:
    rec = run_record()
    return {
        "models": [{"model": m["name"], "coverage_80": round(m["metrics"]["coverage_80"] * 100, 2),
                    "interval_width_kw": round(m["metrics"]["interval_width"], 3),
                    "pinball": m["metrics"]["pinball"]} for m in rec["models"]],
        "significance_vs_yesterday": rec["extra"]["significance_vs_yesterday"],
        "leakage_demo": rec["extra"]["leakage_demo"],
    }
