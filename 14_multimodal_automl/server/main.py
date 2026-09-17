"""Project 14 -- FastAPI application."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Query, Response
from pydantic import BaseModel, Field

from dsfs.app import ProjectMeta, create_app

import inference as mm

META = ProjectMeta(
    slug="14_multimodal_automl",
    number=14,
    title="Multimodal AutoML",
    tagline="What do a product's photo, name and catalogue fields each add, and does fusing them pay?",
    domain="Multimodal classification · CLIP · fusion",
    api_port=8014,
    web_port=5114,
    dataset="Fashion Product Images (Small), 44,072 products",
    techniques=(
        "CLIP image and text embeddings",
        "late and early fusion, and AutoGluon",
        "label-word masking to measure a text leak",
        "text-to-image search",
    ),
)

app = create_app(META)


class Name(BaseModel):
    name: str = Field(..., min_length=1, max_length=300)


def _guard(fn):
    try:
        return fn()
    except mm.ModelNotTrained as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"not found: {exc}") from exc


@app.get("/api/results", tags=["results"])
def get_results() -> dict[str, Any]:
    """Leak measurements, bootstrap verdicts, per-class gains, confusions and search quality."""
    return _guard(mm.results)


@app.get("/api/gallery", tags=["products"])
def get_gallery(article_type: str | None = None, page: int = Query(0, ge=0)) -> dict[str, Any]:
    """Test products, with the image model's and the fusion model's predictions."""
    return _guard(lambda: mm.gallery(article_type, page))


@app.get("/api/products/{product_id}", tags=["products"])
def get_product(product_id: int) -> dict[str, Any]:
    """One test product with the top three predictions of each model."""
    return _guard(lambda: mm.product(product_id))


@app.get("/api/images/{product_id}", tags=["products"])
def get_image(product_id: int) -> Response:
    return Response(content=_guard(lambda: mm.image(product_id)), media_type="image/jpeg",
                    headers={"Cache-Control": "public, max-age=86400"})


@app.post("/api/classify-name", tags=["products"])
def post_classify(body: Name) -> dict[str, Any]:
    """Classify a typed product name with both text models, with and without label words."""
    return _guard(lambda: mm.classify_name(body.name))


@app.get("/api/search", tags=["products"])
def get_search(q: str = Query(..., min_length=2, max_length=200)) -> dict[str, Any]:
    """Find test products whose photos best match a description, using CLIP."""
    return _guard(lambda: mm.search(q))
