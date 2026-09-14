"""Project 15 -- FastAPI application."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from dsfs.app import ProjectMeta, create_app

from inference import (
    ModelNotTrained,
    calibration,
    dataset_facts,
    equity_curves,
    forecast,
    price_series,
)

META = ProjectMeta(
    slug="15_spy_forecasting",
    number=15,
    title="SPY Probabilistic Forecasting",
    tagline="How much of tomorrow's S&P 500 move is actually predictable?",
    domain="Probabilistic time series · walk-forward backtesting",
    api_port=8015,
    web_port=5115,
    dataset="SPY daily bars with VIX and 10-year yield, 2015-2026 (2,931 days)",
    techniques=(
        "quantile forecasting on pinball loss",
        "walk-forward with purging",
        "empirical lookahead verification",
        "paired bootstrap significance",
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
    """Next-period quantile band in return and price terms."""
    return _guard(forecast)


@app.get("/api/prices", tags=["forecast"])
def get_prices() -> list[dict[str, Any]]:
    """Recent closes, for the forecast chart."""
    return _guard(price_series)


@app.get("/api/backtest", tags=["forecast"])
def get_backtest() -> dict[str, Any]:
    """Walk-forward equity curves against buy-and-hold."""
    return _guard(equity_curves)


@app.get("/api/calibration", tags=["forecast"])
def get_calibration() -> dict[str, Any]:
    """Interval coverage and significance versus a random walk."""
    return _guard(calibration)


@app.get("/api/dataset", tags=["forecast"])
def get_dataset() -> dict[str, Any]:
    """Distributional facts and the lookahead verification result."""
    return _guard(dataset_facts)
