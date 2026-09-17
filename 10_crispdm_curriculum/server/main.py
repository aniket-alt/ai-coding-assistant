"""Project 10 -- FastAPI application."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel, Field

from dsfs.app import ProjectMeta, create_app

from inference import ModelNotTrained, check_answer, eda, fairness, lsh, quiz, rules, unsupervised

META = ProjectMeta(
    slug="10_crispdm_curriculum",
    number=10,
    title="CRISP-DM Curriculum",
    tagline="Seven lessons on the Adult census, and why leaving out sex does not make a model fair.",
    domain="Course · classification · fairness",
    api_port=8010,
    web_port=5110,
    dataset="Adult census income, 1994 US Census (48,842 people)",
    techniques=(
        "fairness through unawareness, tested",
        "proxy detection",
        "isolation forest and k-means",
        "random-hyperplane LSH",
    ),
)

app = create_app(META)


class Answer(BaseModel):
    choice: int = Field(..., ge=0, le=9)


def _guard(fn):
    try:
        return fn()
    except ModelNotTrained as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/eda", tags=["lessons"])
def get_eda() -> dict[str, Any]:
    """Base rates by sex, race and age, missing values and data quirks."""
    return _guard(eda)


@app.get("/api/unsupervised", tags=["lessons"])
def get_unsupervised() -> dict[str, Any]:
    """k-means cluster profiles and what the isolation forest flagged."""
    return _guard(unsupervised)


@app.get("/api/fairness", tags=["lessons"])
def get_fairness() -> dict[str, Any]:
    """Accuracy and fairness of each model, with and without sex and race."""
    return _guard(fairness)


@app.get("/api/rules", tags=["lessons"])
def get_rules() -> list[dict[str, Any]]:
    """Strongest association rules for high income, rechecked on test people."""
    return _guard(rules)


@app.get("/api/lsh", tags=["lessons"])
def get_lsh() -> dict[str, Any]:
    """Recall and speed of locality-sensitive hashing against exact search."""
    return _guard(lsh)


@app.get("/api/quiz", tags=["quiz"])
def get_quiz() -> list[dict[str, Any]]:
    """Quiz questions, without their answers."""
    return _guard(quiz)


@app.post("/api/quiz/{question_id}", tags=["quiz"])
def post_answer(question_id: str, answer: Answer) -> dict[str, Any]:
    """Check one answer; the explanation is computed from this run."""
    try:
        return _guard(lambda: check_answer(question_id, answer.choice))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"no question {exc}") from exc
