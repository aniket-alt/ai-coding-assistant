"""Project 08 -- FastAPI application."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Query
from pydantic import BaseModel, Field

from dsfs.app import ProjectMeta, create_app

import inference as lessons

META = ProjectMeta(
    slug="08_visual_foundations",
    number=8,
    title="DS Visual Foundations",
    tagline="Probability, evaluation, gradient descent and backpropagation, on real data and checked.",
    domain="Course · interactive simulators",
    api_port=8008,
    web_port=5108,
    dataset="UCI SMS Spam Collection and Combined Cycle Power Plant",
    techniques=(
        "Naive Bayes, written out",
        "thresholds and precision-recall",
        "gradient descent and its stability limit",
        "backpropagation checked by finite differences",
    ),
)

app = create_app(META)


class Message(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)


class Descent(BaseModel):
    lr: float = Field(..., gt=0, le=10)
    steps: int = Field(300, ge=1, le=lessons.MAX_STEPS)
    scaled: bool = True


class Answer(BaseModel):
    choice: int = Field(..., ge=0, le=9)


def _guard(fn):
    try:
        return fn()
    except lessons.ModelNotTrained as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/classify", tags=["probability"])
def post_classify(msg: Message) -> dict[str, Any]:
    """Spam probability of a message, with every word's contribution to the log-odds."""
    return _guard(lambda: lessons.classify(msg.text))


@app.get("/api/spam", tags=["evaluation"])
def get_spam() -> dict[str, Any]:
    """Dataset facts, the duplicate-leak experiment and the precision-recall curve."""
    return _guard(lessons.spam_lesson)


@app.get("/api/threshold", tags=["evaluation"])
def get_threshold(t: float = Query(0.5, ge=0.0, le=1.0)) -> dict[str, Any]:
    """Confusion matrix on the held-out messages at a threshold."""
    return _guard(lambda: lessons.threshold(t))


@app.get("/api/gradient-descent", tags=["gradient descent"])
def get_gd() -> dict[str, Any]:
    """Recorded runs, stability limits and condition numbers."""
    return _guard(lessons.gd_lesson)


@app.post("/api/gradient-descent", tags=["gradient descent"])
def post_gd(req: Descent) -> dict[str, Any]:
    """Run gradient descent now, at any learning rate, with or without feature scaling."""
    return _guard(lambda: lessons.descend(req.lr, req.steps, req.scaled))


@app.get("/api/backprop", tags=["backpropagation"])
def get_backprop(row: int = Query(0, ge=0)) -> dict[str, Any]:
    """One training row through the trained network, with every chain-rule factor."""
    try:
        return _guard(lambda: lessons.backprop(row))
    except IndexError as exc:
        raise HTTPException(status_code=404, detail="row out of range") from exc


@app.get("/api/backprop/training", tags=["backpropagation"])
def get_backprop_training() -> dict[str, Any]:
    """Loss curve, gradient checks and test error of the network."""
    return _guard(lessons.backprop_lesson)


@app.get("/api/quiz", tags=["quiz"])
def get_quiz() -> list[dict[str, Any]]:
    return _guard(lessons.quiz)


@app.post("/api/quiz/{question_id}", tags=["quiz"])
def post_answer(question_id: str, answer: Answer) -> dict[str, Any]:
    try:
        return _guard(lambda: lessons.check_answer(question_id, answer.choice))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"no question {exc}") from exc
