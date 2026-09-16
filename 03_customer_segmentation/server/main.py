"""Project 03 -- FastAPI application."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel, Field

from dsfs.app import ProjectMeta, create_app

from inference import ModelNotTrained, assign, feature_ranges, segments

META = ProjectMeta(
    slug="03_customer_segmentation",
    number=3,
    title="Customer Segmentation",
    tagline="Which customers behave differently, and do the groups still mean anything a year later?",
    domain="Clustering · judged on future behaviour",
    api_port=8003,
    web_port=5103,
    dataset="Online Retail II, UCI, 2009-2011 (1,067,371 lines)",
    techniques=(
        "segments scored on next-year spend",
        "omega-squared corrected for segment count",
        "subsample stability by adjusted Rand index",
        "RFM rule as a baseline",
    ),
)

app = create_app(META)


class Customer(BaseModel):
    recency_days: float = Field(..., ge=0)
    frequency: float = Field(..., ge=1)
    monetary: float = Field(..., ge=0)
    avg_basket: float = Field(..., ge=0)
    n_products: float = Field(..., ge=1)
    tenure_days: float = Field(..., ge=0)
    return_rate: float = Field(0.0, ge=0, le=1)


def _guard(fn):
    try:
        return fn()
    except ModelNotTrained as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/segments", tags=["segments"])
def get_segments(model: str | None = None) -> dict[str, Any]:
    """Segment profiles for one segmentation, defaulting to the winner."""
    try:
        return _guard(lambda: segments(model))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"no segmentation named {exc}") from exc


@app.get("/api/feature-ranges", tags=["segments"])
def get_feature_ranges() -> dict[str, Any]:
    """5th percentile, median and 95th percentile of each feature, for input bounds."""
    return _guard(feature_ranges)


@app.post("/api/assign", tags=["segments"])
def post_assign(customer: Customer) -> dict[str, Any]:
    """Assign a customer to the nearest segment of the winning segmentation."""
    return _guard(lambda: assign(customer.model_dump()))
