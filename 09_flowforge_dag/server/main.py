"""Project 09 -- FastAPI application: the engine's recorded evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from dsfs.app import ProjectMeta, create_app

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
ENGINE = ROOT / "engine"
SOURCES = {
    "type-tests": ENGINE / "type-tests" / "compile-time.ts",
    "types": ENGINE / "src" / "types.ts",
    "kahn": ENGINE / "src" / "graph.ts",
}

META = ProjectMeta(
    slug="09_flowforge_dag",
    number=9,
    title="FlowForge DAG Engine",
    tagline="A typed workflow engine, with evidence that it is correct and scales.",
    domain="TypeScript · graph algorithms · verification",
    api_port=8009,
    web_port=5109,
    dataset="Seeded random workflow graphs (generated)",
    techniques=(
        "Kahn's algorithm with cycle paths",
        "incremental discrete-event scheduler",
        "property and differential testing",
        "mutation testing",
    ),
)

app = create_app(META)


def _extra() -> dict[str, Any]:
    p = ARTIFACTS / "metrics.json"
    if not p.exists():
        raise HTTPException(status_code=503, detail="artifacts/metrics.json not found -- run ml/train.py.")
    return json.loads(p.read_text(encoding="utf-8"))["extra"]


@app.get("/api/evidence", tags=["results"])
def evidence() -> dict[str, Any]:
    """Unit, property, type and mutation test results from the recorded run."""
    e = _extra()
    return {"tests": e["tests"], "types": e["types"], "mutants": e["mutants"],
            "properties": e["bench"]["properties"], "property_checks": e["property_checks"],
            "property_failures": e["property_failures"]}


@app.get("/api/benchmarks", tags=["results"])
def benchmarks() -> dict[str, Any]:
    """Ordering and executor timings from the recorded run."""
    b = _extra()["bench"]
    return {k: b[k] for k in ("node", "repeats", "edges_per_node", "scaling", "head_to_head", "executors",
                              "million_node_levels")}


@app.get("/api/source/{name}", tags=["source"])
def source(name: str) -> dict[str, str]:
    """The engine's own source for the type-design view, read from disk."""
    if name not in SOURCES:
        raise HTTPException(status_code=404, detail=f"unknown source '{name}'")
    path = SOURCES[name]
    return {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "text": path.read_text(encoding="utf-8")}
