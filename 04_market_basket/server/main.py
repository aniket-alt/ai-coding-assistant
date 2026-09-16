"""Project 04 -- FastAPI application."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel, Field

from dsfs.app import ProjectMeta, create_app

from inference import ModelNotTrained, items, recommend, rules

META = ProjectMeta(
    slug="04_market_basket",
    number=4,
    title="Market Basket Mining",
    tagline="Which products go together, and do those associations hold next season?",
    domain="Association rules · recommendation",
    api_port=8004,
    web_port=5104,
    dataset="Online Retail II, UCI, 2009-2011 (1,067,371 lines)",
    techniques=(
        "FP-Growth association rules",
        "held-out item recommendation test",
        "rule survival in a later season",
        "item co-occurrence baseline",
    ),
)

app = create_app(META)


class Basket(BaseModel):
    items: list[str] = Field(..., min_length=1, max_length=50)


def _guard(fn):
    try:
        return fn()
    except ModelNotTrained as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/items", tags=["baskets"])
def get_items() -> list[dict[str, Any]]:
    """The 300 most frequently bought products, for building a basket."""
    return _guard(items)


@app.get("/api/rules", tags=["baskets"])
def get_rules() -> list[dict[str, Any]]:
    """Strongest rules, with their statistics in training and in the following season."""
    return _guard(rules)


@app.post("/api/recommend", tags=["baskets"])
def post_recommend(basket: Basket) -> dict[str, Any]:
    """Five product suggestions for a basket, each with the reason it was suggested."""
    return _guard(lambda: recommend(basket.items))
