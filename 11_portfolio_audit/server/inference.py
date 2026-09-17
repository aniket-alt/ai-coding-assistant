"""
Project 11 -- serving.

The benchmark results and the portfolio snapshot are read from metrics.json.
Variant source code is read from the committed corpus, but only for ids the
benchmark itself recorded, so the endpoint cannot be used to read other files.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
CORPUS = ARTIFACTS / "leak_corpus"


class ModelNotTrained(RuntimeError):
    """Raised when the API starts before `python ml/train.py` has run."""


@lru_cache(maxsize=1)
def run_record() -> dict[str, Any]:
    p = ARTIFACTS / "metrics.json"
    if not p.exists():
        raise ModelNotTrained("artifacts/metrics.json not found -- run ml/train.py.")
    return json.loads(p.read_text(encoding="utf-8"))


def benchmark() -> dict[str, Any]:
    e = run_record()["extra"]
    return {"clean_average_precision": e["clean_average_precision"], "variants": e["variants"]}


def portfolio() -> list[dict[str, Any]]:
    return run_record()["extra"]["portfolio"]


def variant_source(variant_id: str) -> dict[str, Any]:
    known = {v["id"]: v for v in run_record()["extra"]["variants"]}
    if variant_id not in known:
        raise KeyError(variant_id)
    return {**known[variant_id], "source": (CORPUS / f"{variant_id}.py").read_text(encoding="utf-8")}
