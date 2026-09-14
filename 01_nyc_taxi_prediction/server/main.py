"""
Project 01 -- FastAPI application.

Thanks to the platform factory, this file contains only what is specific to
taxi prediction. Health, metadata, CRISP-DM, metrics, provenance and audit
endpoints all arrive from `dsfs.app.create_app`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel, Field

from dsfs.app import ProjectMeta, create_app

from inference import (
    ModelNotTrained,
    feature_importance,
    load_zones,
    predict,
)

META = ProjectMeta(
    slug="01_nyc_taxi_prediction",
    number=1,
    title="NYC Taxi Trip Duration",
    tagline="How long will this cab ride take? Estimated from booking-time information only.",
    domain="Spatial regression · CRISP-DM",
    api_port=8001,
    web_port=5101,
    dataset="NYC TLC Yellow Taxi, January 2024 (2.96M trips)",
    techniques=(
        "chronological split with embargo",
        "out-of-fold target encoding",
        "gradient boosting tournament",
        "leakage assertions",
    ),
)

app = create_app(META)


class PredictRequest(BaseModel):
    pu_location: int = Field(..., ge=1, le=263, description="Pickup taxi zone LocationID")
    do_location: int = Field(..., ge=1, le=263, description="Dropoff taxi zone LocationID")
    when: datetime | None = Field(None, description="Pickup time; defaults to now")
    passenger_count: int = Field(1, ge=1, le=6)
    rate_code: int = Field(1, ge=1, le=6)


@app.post("/api/predict", tags=["prediction"])
def post_predict(req: PredictRequest) -> dict[str, Any]:
    """Estimate duration for one booking, with an interval from measured error."""
    try:
        return predict(
            pu_location=req.pu_location,
            do_location=req.do_location,
            when=req.when or datetime.now(),
            passenger_count=req.passenger_count,
            rate_code=req.rate_code,
        )
    except ModelNotTrained as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/zones", tags=["prediction"])
def get_zones() -> list[dict[str, Any]]:
    """The 263 taxi zones, for the origin and destination pickers."""
    return load_zones()


@app.get("/api/importance", tags=["prediction"])
def get_importance(top: int = 15) -> list[dict[str, Any]]:
    """Which features the winning model actually leans on."""
    try:
        return feature_importance(top=top)
    except ModelNotTrained as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
