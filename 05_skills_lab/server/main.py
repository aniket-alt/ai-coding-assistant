"""Project 05 -- FastAPI application."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Query

from dsfs.app import ProjectMeta, create_app

import inference as lab

META = ProjectMeta(
    slug="05_skills_lab",
    number=5,
    title="Data Science Skills Lab",
    tagline="A catalogue of techniques, each one measured the careless way and the careful way.",
    domain="Course · measured practice · leakage",
    api_port=8005,
    web_port=5105,
    dataset="Titanic, German Credit, Pima Diabetes and Blood Transfusion (OpenML)",
    techniques=(
        "paired comparisons over many seeds",
        "bootstrap intervals on the difference",
        "leakage demonstrated rather than described",
        "live re-runs of any entry",
    ),
)

app = create_app(META)


def _guard(fn):
    try:
        return fn()
    except lab.ModelNotTrained as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"no such skill: {exc}") from exc


@app.get("/api/catalogue", tags=["catalogue"])
def get_catalogue() -> dict[str, Any]:
    """Every skill's recorded comparison, the datasets behind them, and the auditor's warnings."""
    return _guard(lab.catalogue)


@app.get("/api/skills/{skill_id}", tags=["catalogue"])
def get_skill(skill_id: str) -> dict[str, Any]:
    """One skill's recorded comparison."""
    return _guard(lambda: lab.skill(skill_id))


@app.post("/api/skills/{skill_id}/rerun", tags=["catalogue"])
def post_rerun(skill_id: str, seeds: int = Query(10, ge=2, le=lab.MAX_SEEDS)) -> dict[str, Any]:
    """Run this skill again now, on a chosen number of seeds, beside what the training run recorded."""
    return _guard(lambda: lab.rerun(skill_id, seeds))
