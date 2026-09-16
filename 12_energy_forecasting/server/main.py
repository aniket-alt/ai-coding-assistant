"""Project 12 -- FastAPI application."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from dsfs.app import ProjectMeta, create_app

from inference import ModelNotTrained, calibration, forecast, patterns

META = ProjectMeta(
    slug="12_energy_forecasting",
    number=12,
    title="Energy Demand Forecasting",
    tagline="How much electricity will this home use in each hour of tomorrow?",
    domain="Probabilistic day-ahead forecasting",
    api_port=8012,
    web_port=5112,
    dataset="UCI household electric power, 2006-2010 (2,075,259 minute readings)",
    techniques=(
        "day-ahead quantile forecasting",
        "minimum lag equal to the horizon",
        "walk-forward with paired bootstrap",
        "measured cost of a leaky lag",
    ),
)

app = create_app(META)


def _guard(fn):
    try:
        return fn()
    except ModelNotTrained as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/forecast", tags=["forecast"])
def get_forecast() -> dict[str, Any]:
    """Tomorrow's hourly band, with the last fourteen days of forecasts against actual load."""
    return _guard(forecast)


@app.get("/api/patterns", tags=["forecast"])
def get_patterns() -> dict[str, Any]:
    """Average load by hour for weekdays and weekends, and by month."""
    return _guard(patterns)


@app.get("/api/calibration", tags=["forecast"])
def get_calibration() -> dict[str, Any]:
    """Interval coverage, significance against yesterday, and the leakage demonstration."""
    return _guard(calibration)
