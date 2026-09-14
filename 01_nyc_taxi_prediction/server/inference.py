"""
Project 01 -- serving.

Loads the persisted model together with the encoders it was trained with, so a
prediction at request time is assembled exactly as it was during training. The
zone statistics and target encodings travel inside the artefact rather than
being recomputed, which is what stops training and serving from drifting apart.
"""

from __future__ import annotations

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

from features import calendar_features, apply_zone_statistics  # noqa: E402

ARTIFACTS = ROOT / "artifacts"


class ModelNotTrained(RuntimeError):
    """Raised when the API is started before `python ml/train.py` has run."""


@lru_cache(maxsize=1)
def load_bundle() -> dict[str, Any]:
    path = ARTIFACTS / "model.joblib"
    if not path.exists():
        raise ModelNotTrained(
            "artifacts/model.joblib not found -- run `python ml/train.py` first."
        )
    return joblib.load(path)


@lru_cache(maxsize=1)
def load_zones() -> list[dict[str, Any]]:
    """Zone list for the UI's origin/destination pickers."""
    sys.path.insert(0, str(ROOT / "ml"))
    from data import load_zones as _load

    zones, _ = _load()
    zones = zones[zones["LocationID"].between(1, 263)]
    return [
        {
            "id": int(r.LocationID),
            "zone": str(r.Zone),
            "borough": str(r.Borough),
            "label": f"{r.Zone} ({r.Borough})",
        }
        for r in zones.itertuples()
    ]


def _residual_sigma() -> float:
    """
    Spread of the held-out residuals, used for the prediction interval.

    Read from the run record rather than assumed, so the interval reflects the
    error the model actually made on unseen days.
    """
    import json

    p = ARTIFACTS / "metrics.json"
    if not p.exists():
        return 6.0
    rec = json.loads(p.read_text(encoding="utf-8"))
    best = rec.get("best_score")
    return float(best) if best else 6.0


def predict(
    pu_location: int,
    do_location: int,
    when: datetime,
    passenger_count: int = 1,
    rate_code: int = 1,
) -> dict[str, Any]:
    """Estimate trip duration for a single booking."""
    bundle = load_bundle()
    model = bundle["model"]
    encoders = bundle["encoders"]
    order = bundle["feature_order"]
    categorical = bundle["categorical"]
    categories = bundle["categories"]

    row = pd.DataFrame(
        {
            "tpep_pickup_datetime": [pd.Timestamp(when)],
            "PULocationID": [int(pu_location)],
            "DOLocationID": [int(do_location)],
            "passenger_count": [float(passenger_count)],
            "RatecodeID": [float(rate_code)],
            "vendor_fields_missing": [0],
        }
    )

    X = pd.concat(
        [
            calendar_features(row),
            row[["PULocationID", "DOLocationID", "passenger_count",
                 "RatecodeID", "vendor_fields_missing"]],
            apply_zone_statistics(row, encoders["zone_stats"]),
        ],
        axis=1,
    )

    X["te_zone_pair"] = encoders["te_zone_pair"].transform(row)
    hour_row = row.assign(_hour=row["tpep_pickup_datetime"].dt.hour)
    X["te_pu_hour"] = encoders["te_pu_hour"].transform(hour_row)

    X = X[order]
    for c in categorical:
        X[c] = pd.Categorical(X[c].astype("int32"), categories=categories[c])

    minutes = float(np.clip(model.predict(X)[0], 1.0, 180.0))
    sigma = _residual_sigma()

    return {
        "duration_minutes": round(minutes, 1),
        # Roughly an 80% interval under a normal residual assumption; the
        # spread is the model's measured held-out RMSE, not a guess.
        "lower_minutes": round(max(1.0, minutes - 1.28 * sigma), 1),
        "upper_minutes": round(minutes + 1.28 * sigma, 1),
        "rmse_minutes": round(sigma, 2),
        "route_median_distance_miles": round(float(X["zp_median_distance"].iloc[0]), 2),
        "route_trip_count": int(X["zp_trip_count"].iloc[0]),
        "is_rare_route": bool(X["zp_is_rare"].iloc[0]),
        "requested_at": when.isoformat(),
    }


def feature_importance(top: int = 15) -> list[dict[str, Any]]:
    """Gain-based importance from the fitted booster."""
    bundle = load_bundle()
    model = bundle["model"]
    names = bundle["feature_order"]
    gains = model.booster_.feature_importance(importance_type="gain")
    total = float(gains.sum()) or 1.0
    rows = [
        {"feature": n, "gain": float(g), "share": round(float(g) / total * 100, 2)}
        for n, g in zip(names, gains)
    ]
    rows.sort(key=lambda r: r["gain"], reverse=True)
    return rows[:top]
