"""Project 13 -- FastAPI application."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel, Field

from dsfs.app import ProjectMeta, create_app

from inference import ModelNotTrained, load_test, model_card, monitoring, predict, registry, zones

META = ProjectMeta(
    slug="13_mlops_governance",
    number=13,
    title="MLOps & Governance",
    tagline="How fast does a deployed model go stale, and when is retraining worth it?",
    domain="Monitoring · retraining policy · explainability",
    api_port=8013,
    web_port=5113,
    dataset="NYC TLC yellow-cab trips, January to July 2024",
    techniques=(
        "month-by-month retraining policies",
        "population stability index",
        "TreeSHAP contributions",
        "measured load test",
    ),
)

app = create_app(META)


class Booking(BaseModel):
    pu_location: int = Field(..., ge=1, le=263)
    do_location: int = Field(..., ge=1, le=263)
    when: datetime
    passenger_count: int = Field(1, ge=1, le=6)


def _guard(fn):
    try:
        return fn()
    except ModelNotTrained as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/predict", tags=["serving"])
def post_predict(b: Booking) -> dict[str, Any]:
    """Trip duration with each feature's TreeSHAP contribution in minutes."""
    return _guard(lambda: predict(b.pu_location, b.do_location, b.when, b.passenger_count))


@app.get("/api/zones", tags=["serving"])
def get_zones() -> list[dict[str, Any]]:
    """The 263 taxi zones."""
    return _guard(zones)


@app.get("/api/drift", tags=["monitoring"])
def get_drift() -> dict[str, Any]:
    """Monthly drift against January and every policy's month-by-month error."""
    return _guard(monitoring)


@app.get("/api/registry", tags=["governance"])
def get_registry() -> dict[str, Any]:
    """Every model version fitted, with its training window and first month in service."""
    return _guard(registry)


@app.get("/api/load-test", tags=["governance"])
def get_load_test() -> dict[str, Any]:
    """Latency and throughput measured against this API during training."""
    return _guard(load_test)


@app.get("/api/model-card", tags=["governance"])
def get_model_card() -> dict[str, Any]:
    """Intended use, data, evaluation, monitoring, capacity and limitations."""
    return _guard(model_card)
