"""
Project 15 -- serving.

Everything served here was computed during the walk-forward run and stored in
metrics.json. Nothing is recalculated at request time, so the forecast band a
visitor sees is the same band the backtest was scored on.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"


class ModelNotTrained(RuntimeError):
    """Raised when the API starts before `python ml/train.py` has run."""


@lru_cache(maxsize=1)
def run_record() -> dict[str, Any]:
    p = ARTIFACTS / "metrics.json"
    if not p.exists():
        raise ModelNotTrained("artifacts/metrics.json not found -- run ml/train.py.")
    return json.loads(p.read_text(encoding="utf-8"))


def _extra() -> dict[str, Any]:
    return run_record().get("extra", {})


def forecast() -> dict[str, Any]:
    """
    Next-period quantile band, expressed in both return and price terms.

    The band is a log-return interval; converting to price is exp(r) * close,
    which is why the price band is not symmetric about the last close.
    """
    e = _extra()
    close = float(e.get("last_close", 0.0))
    out = []
    for name, q in e.get("latest_forecasts", {}).items():
        out.append({
            "model": name,
            "q10": q["q10"], "q50": q["q50"], "q90": q["q90"],
            "q10_pct": round(q["q10"] * 100, 3),
            "q50_pct": round(q["q50"] * 100, 3),
            "q90_pct": round(q["q90"] * 100, 3),
            "price_low": round(close * float(np.exp(q["q10"])), 2),
            "price_mid": round(close * float(np.exp(q["q50"])), 2),
            "price_high": round(close * float(np.exp(q["q90"])), 2),
        })
    return {
        "last_close": close,
        "last_date": e.get("last_date"),
        "horizon": e.get("horizon", 1),
        "forecasts": out,
    }


def price_series() -> list[dict[str, Any]]:
    return list(_extra().get("price_history", []))


def equity_curves() -> dict[str, Any]:
    return {
        "curves": _extra().get("equity_curves", {}),
        "summary": _extra().get("backtests", {}),
    }


def calibration() -> dict[str, Any]:
    """Coverage against the nominal 80%, plus significance versus a random walk."""
    e = _extra()
    return {
        "calibration": e.get("calibration", []),
        "significance_vs_random_walk": e.get("significance_vs_random_walk", {}),
        "nominal_coverage": 80.0,
    }


def dataset_facts() -> dict[str, Any]:
    e = _extra()
    return {"dataset": e.get("dataset", {}), "lookahead_check": e.get("lookahead_check", {})}
