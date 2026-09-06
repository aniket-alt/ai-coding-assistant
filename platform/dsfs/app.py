"""
FastAPI application factory shared by all 16 project backends.

Every project calls `create_app(...)` and gets, for free:

* `/api/health`            -- liveness + version + git commit
* `/api/meta`              -- project identity, ports, links
* `/api/crispdm`           -- the CRISP-DM report the training run produced
* `/api/metrics`           -- artifacts/metrics.json (the leaderboard)
* `/api/audit`             -- the leakage audit findings
* `/api/provenance`        -- the dataset receipt
* CORS for the local Vite port, timing middleware, uniform error shape

so each project's own `main.py` only contains what is genuinely unique to it.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import __version__


class ProjectMeta:
    """Identity card for one project in the portfolio."""

    def __init__(
        self,
        slug: str,
        number: int,
        title: str,
        tagline: str,
        domain: str,
        api_port: int,
        web_port: int,
        dataset: str = "",
        techniques: tuple[str, ...] = (),
    ) -> None:
        self.slug = slug
        self.number = number
        self.title = title
        self.tagline = tagline
        self.domain = domain
        self.api_port = api_port
        self.web_port = web_port
        self.dataset = dataset
        self.techniques = techniques

    def to_dict(self) -> dict[str, Any]:
        return {
            "slug": self.slug,
            "number": self.number,
            "title": self.title,
            "tagline": self.tagline,
            "domain": self.domain,
            "api_port": self.api_port,
            "web_port": self.web_port,
            "dataset": self.dataset,
            "techniques": list(self.techniques),
            "dsfs_version": __version__,
        }


def create_app(
    meta: ProjectMeta,
    *,
    artifacts_dir: str | Path = "artifacts",
    on_startup: Callable[[], None] | None = None,
    extra_origins: tuple[str, ...] = (),
) -> FastAPI:
    """Build a FastAPI app preloaded with the portfolio's standard endpoints."""

    art = Path(artifacts_dir)

    app = FastAPI(
        title=meta.title,
        description=meta.tagline,
        version=__version__,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )
    app.state.meta = meta
    app.state.artifacts_dir = art

    origins = [
        f"http://localhost:{meta.web_port}",
        f"http://127.0.0.1:{meta.web_port}",
        *extra_origins,
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def _timing(request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Process-Time-ms"] = f"{(time.perf_counter() - t0) * 1000:.2f}"
        return response

    @app.exception_handler(Exception)
    async def _errors(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={"error": type(exc).__name__, "detail": str(exc),
                     "path": str(request.url.path)},
        )

    # -- standard endpoints -------------------------------------------

    @app.get("/api/health", tags=["platform"])
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "project": meta.slug,
            "version": __version__,
            "artifacts_present": art.exists(),
        }

    @app.get("/api/meta", tags=["platform"])
    def get_meta() -> dict[str, Any]:
        return meta.to_dict()

    @app.get("/api/metrics", tags=["platform"])
    def metrics() -> Any:
        return _read_json(art / "metrics.json")

    @app.get("/api/crispdm", tags=["platform"])
    def crispdm() -> Any:
        return _read_json(art / "crispdm.json")

    @app.get("/api/audit", tags=["platform"])
    def audit() -> Any:
        return _read_json(art / "audit.json")

    @app.get("/api/provenance", tags=["platform"])
    def provenance() -> Any:
        return _read_json(art / "provenance.json")

    if on_startup is not None:
        app.add_event_handler("startup", on_startup)

    return app


def _read_json(path: Path) -> Any:
    """Return parsed JSON, or a structured 'not built yet' marker."""
    if not path.exists():
        return {
            "available": False,
            "detail": f"{path.name} not found -- run the project's training "
                      f"script to generate it.",
        }
    return json.loads(path.read_text(encoding="utf-8"))
