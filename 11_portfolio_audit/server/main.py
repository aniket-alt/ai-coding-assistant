"""Project 11 -- FastAPI application."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from dsfs.app import ProjectMeta, create_app

from inference import ModelNotTrained, benchmark, portfolio, variant_source

META = ProjectMeta(
    slug="11_portfolio_audit",
    number=11,
    title="Portfolio Audit",
    tagline="How much should anyone trust this portfolio's leakage audit?",
    domain="Detector benchmark · governance",
    api_port=8011,
    web_port=5111,
    dataset="ULB credit card fraud with planted leaks, plus this portfolio's own artifacts",
    techniques=(
        "planted-leak benchmark",
        "balanced accuracy over F1",
        "score inflation per leak",
        "clean controls verified by score",
    ),
)

app = create_app(META)


def _guard(fn):
    try:
        return fn()
    except ModelNotTrained as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/benchmark", tags=["audit"])
def get_benchmark() -> dict[str, Any]:
    """Every planted leak and control, with both detectors' verdicts and the score inflation."""
    return _guard(benchmark)


@app.get("/api/corpus/{variant_id}", tags=["audit"])
def get_variant(variant_id: str) -> dict[str, Any]:
    """The source of one benchmark variant, so what was planted can be checked."""
    try:
        return _guard(lambda: variant_source(variant_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"no variant named {exc}") from exc


@app.get("/api/portfolio", tags=["audit"])
def get_portfolio() -> list[dict[str, Any]]:
    """Audit state of every other built project, read from its artifacts at training time."""
    return _guard(portfolio)
