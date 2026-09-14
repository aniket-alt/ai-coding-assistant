"""
Project 06 -- FastAPI application.

Only the fraud-specific endpoints live here; health, meta, metrics, CRISP-DM,
provenance and audit arrive from the platform factory.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel, Field

from dsfs.app import ProjectMeta, create_app

from inference import (
    ModelNotTrained,
    budget_curve,
    eda,
    sample_transactions,
    score,
)

META = ProjectMeta(
    slug="06_anomaly_detection",
    number=6,
    title="Credit Card Fraud Detection",
    tagline="Which handful of transactions should an analyst look at today?",
    domain="Anomaly detection · extreme class imbalance",
    api_port=8006,
    web_port=5106,
    dataset="ULB credit card fraud, 284,807 transactions (492 fraudulent)",
    techniques=(
        "average precision over accuracy",
        "chronological split with embargo",
        "unsupervised vs supervised comparison",
        "review-budget evaluation",
    ),
)

app = create_app(META)


class ScoreRequest(BaseModel):
    features: dict[str, float] = Field(
        ..., description="Transaction features: Time, Amount, V1..V28"
    )


@app.post("/api/score", tags=["fraud"])
def post_score(req: ScoreRequest) -> dict[str, Any]:
    """Score a transaction and locate it in the held-out distribution."""
    try:
        return score(req.features)
    except ModelNotTrained as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/samples", tags=["fraud"])
def get_samples(n: int = 12) -> list[dict[str, Any]]:
    """Real held-out transactions the UI can load as presets."""
    try:
        return sample_transactions(n=n)
    except ModelNotTrained as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/budget-curve", tags=["fraud"])
def get_budget_curve() -> list[dict[str, Any]]:
    """Precision and recall at each analyst review budget."""
    try:
        return budget_curve()
    except ModelNotTrained as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/eda", tags=["fraud"])
def get_eda() -> dict[str, Any]:
    """Fraud rate by hour and by transaction amount band."""
    try:
        return eda()
    except ModelNotTrained as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
