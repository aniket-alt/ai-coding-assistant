"""
Project 13 -- serving project 01's model in operation.

Predictions reproduce project 01's feature pipeline exactly, using the encoders
persisted with the served model, and return each feature's TreeSHAP
contribution in minutes. Monitoring, lineage and load figures are read from the
training run's metrics.json.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))
ARTIFACTS = ROOT / "artifacts"

from data import project01, zones as zone_names  # noqa: E402

features01 = project01("features")


class ModelNotTrained(RuntimeError):
    """Raised when the API starts before `python ml/train.py` has run."""


@lru_cache(maxsize=1)
def bundle() -> dict[str, Any]:
    p = ARTIFACTS / "model.joblib"
    if not p.exists():
        raise ModelNotTrained("artifacts/model.joblib not found -- run ml/train.py.")
    return joblib.load(p)


def run_record() -> dict[str, Any]:
    p = ARTIFACTS / "metrics.json"
    if not p.exists():
        raise ModelNotTrained("artifacts/metrics.json not found -- run ml/train.py.")
    return json.loads(p.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def zones() -> list[dict[str, Any]]:
    return [{"id": k, "label": v} for k, v in sorted(zone_names().items()) if 1 <= k <= 263]


def predict(pu: int, do: int, when: datetime, passengers: int = 1) -> dict[str, Any]:
    b = bundle()
    enc = b["encoders"]
    row = pd.DataFrame({"tpep_pickup_datetime": [pd.Timestamp(when)], "PULocationID": [pu],
                        "DOLocationID": [do], "passenger_count": [float(passengers)],
                        "RatecodeID": [1.0], "vendor_fields_missing": [0]})
    X = pd.concat([features01.calendar_features(row),
                   row[["PULocationID", "DOLocationID", "passenger_count", "RatecodeID", "vendor_fields_missing"]],
                   features01.apply_zone_statistics(row, enc["zone_stats"])], axis=1)
    X["te_zone_pair"] = enc["te_zone_pair"].transform(row)
    X["te_pu_hour"] = enc["te_pu_hour"].transform(row.assign(_hour=row["tpep_pickup_datetime"].dt.hour))
    X = X[b["feature_order"]]
    for c in b["categorical"]:
        X[c] = pd.Categorical(X[c].astype("int32"), categories=b["categories"][c])

    contrib = b["model"].predict(X, pred_contrib=True)[0]
    minutes = float(np.clip(contrib.sum(), 1.0, 180.0))
    parts = sorted(({"feature": f, "minutes": round(float(v), 3)}
                    for f, v in zip(b["feature_order"], contrib[:-1])), key=lambda r: -abs(r["minutes"]))
    return {"duration_minutes": round(minutes, 1), "baseline_minutes": round(float(contrib[-1]), 2),
            "contributions": parts[:10], "policy": b["policy"], "trained_on": b["trained_on"]}


def monitoring() -> dict[str, Any]:
    e = run_record()["extra"]
    return {"drift_vs_january": e["drift_vs_january"], "psi_alarm": e["psi_alarm"],
            "alarm_checks": e["alarm_checks"], "policy_months": e["policy_months"], "facts": e["facts"]}


def registry() -> dict[str, Any]:
    e = run_record()["extra"]
    return {"versions": e["registry"], "served": e["served"], "global_importance": e["global_importance"]}


def load_test() -> dict[str, Any]:
    return run_record()["extra"]["load_test"]


def model_card() -> dict[str, Any]:
    """A model card assembled from the run record; no figure is written by hand."""
    rec = run_record()
    e = rec["extra"]
    served = next(m for m in rec["models"] if m["name"] == e["served"]["policy"])
    peak = max(e["load_test"]["levels"], key=lambda r: r["throughput_rps"])
    return {
        "model": "LightGBM regressor with project 01's features",
        "intended_use": "Estimate a yellow-cab trip's duration at booking time from origin, destination, time and party size.",
        "not_for": "Fares, routing, or trips outside New York City's 263 TLC taxi zones.",
        "training_data": f"NYC TLC yellow-cab trips from {', '.join(e['served']['trained_on'])}; "
                         f"{rec['split']['n_train']:,} sampled trips per month.",
        "evaluation": f"Month by month on unseen months: mean RMSE {served['metrics']['mean_rmse']:.3f} min, "
                      f"worst month {served['metrics']['worst_month_rmse']:.3f} min.",
        "monitoring": f"Population stability index on five columns against the training data; alarm above {e['psi_alarm']}.",
        "capacity": f"{peak['throughput_rps']:.0f} requests per second at concurrency {peak['concurrency']}, "
                    f"p99 {peak['p99_ms']:.1f} ms, one worker on {e['load_test']['machine']}.",
        "limitations": [
            "Trained without weather, events or live traffic, all of which drive real journey times.",
            "Trip durations rose through spring 2024, so a model that is never retrained drifts with them.",
            "Explanations are TreeSHAP contributions of this model, not causal effects of the features.",
        ],
        "assertions": [f"{a['name']}: {'passed' if a['passed'] else 'FAILED'}"
                       for a in rec["audit"]["runtime_assertions"]],
    }
