"""Project 07 -- FastAPI application."""

from __future__ import annotations

from typing import Any

from fastapi import Body, HTTPException

from dsfs.app import ProjectMeta, create_app

from inference import InvalidCall, ModelNotTrained, form, predict, shift, stacking, tournament

META = ProjectMeta(
    slug="07_automl_stacking",
    number=7,
    title="AutoML Stacking",
    tagline="Does automated stacking choose which clients to call better than one tuned model?",
    domain="Classification · AutoML · ensembles",
    api_port=8007,
    web_port=5107,
    dataset="UCI Bank Marketing calls, May 2008 to November 2010",
    techniques=(
        "AutoGluon best_quality",
        "Caruana ensemble selection, built by hand",
        "Optuna tuning",
        "paired bootstrap on the deployment year",
    ),
)

app = create_app(META)


def _guard(fn):
    try:
        return fn()
    except ModelNotTrained as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except InvalidCall as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/form", tags=["predict"])
def get_form() -> dict[str, Any]:
    """Recorded values of each categorical feature, numeric ranges, and a starting client."""
    return _guard(form)


@app.post("/api/predict", tags=["predict"])
def post_predict(call: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Chance a client subscribes, with every stacked model's score and weight."""
    return _guard(lambda: predict(call))


@app.get("/api/tournament", tags=["results"])
def get_tournament() -> dict[str, Any]:
    """Paired bootstrap verdicts, AutoGluon's own leaderboard and the estimate check."""
    return _guard(tournament)


@app.get("/api/stacking", tags=["results"])
def get_stacking() -> dict[str, Any]:
    """Level-one models, the Caruana selection trajectory and the tuning trace."""
    return _guard(stacking)


@app.get("/api/shift", tags=["results"])
def get_shift() -> dict[str, Any]:
    """Success rate over time, the duration leak and permutation importance."""
    return _guard(shift)
