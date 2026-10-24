"""
Project 05 -- serving the catalogue.

Recorded results are read from the training run. A skill can also be re-run on
request, through the same code the training run used, so a reader can watch an
interval narrow as seeds are added and can check that the same seeds give the
same numbers.
"""

from __future__ import annotations

import json
import sys
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))
ARTIFACTS = ROOT / "artifacts"

MAX_SEEDS = 40


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


def catalogue() -> dict[str, Any]:
    e = extra()
    return {"skills": e["skills"], "datasets": e["datasets"], "categories": e["categories"],
            "seeds": len(e["seeds"]), "auditor_warnings": e["auditor_warnings"]}


def skill(skill_id: str) -> dict[str, Any]:
    for record in extra()["skills"]:
        if record["id"] == skill_id:
            return record
    raise KeyError(skill_id)


_frames: dict[str, Any] = {}
_lock = threading.Lock()


def _frame(name: str):
    from data import load

    with _lock:
        if name not in _frames:
            _frames[name] = load(name)[0]
        return _frames[name]


def rerun(skill_id: str, seeds: int) -> dict[str, Any]:
    """Re-run one skill on the first `seeds` seeds and return a fresh record."""
    from skills import BY_ID

    if skill_id not in BY_ID:
        raise KeyError(skill_id)
    seeds = max(2, min(int(seeds), MAX_SEEDS))
    entry = BY_ID[skill_id]
    record = entry.measure(_frame(entry.dataset), list(range(seeds)))
    recorded = skill(skill_id)
    record["recorded"] = {"seeds": recorded["seeds"], "gap_mean": recorded["gap"]["mean"],
                          "ci_low": recorded["gap"]["ci_low"], "ci_high": recorded["gap"]["ci_high"]}
    return record
