"""Project 02 -- FastAPI application."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel, Field

from dsfs.app import ProjectMeta, create_app

from inference import ModelNotTrained, ablations, attention, checks, generate, tokenize, training

META = ProjectMeta(
    slug="02_nano_llm",
    number=2,
    title="Nano LLM Transformer",
    tagline="A small language model written from scratch, and which of its parts earn their place.",
    domain="Language modelling · PyTorch · ablations",
    api_port=8002,
    web_port=5102,
    dataset="TinyStories V2 (GPT-4 stories)",
    techniques=(
        "RoPE, SwiGLU and RMSNorm from scratch",
        "byte-level BPE tokenizer",
        "KV-cache generation, verified",
        "ablations against seed noise",
    ),
)

app = create_app(META)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1000)
    max_new: int = Field(120, ge=1, le=240)
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    top_p: float = Field(0.9, gt=0.0, le=1.0)
    repetition_penalty: float = Field(1.1, ge=1.0, le=2.0)
    seed: int = Field(0, ge=0)
    use_cache: bool = True


class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)


def _guard(fn):
    try:
        return fn()
    except ModelNotTrained as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/generate", tags=["model"])
def post_generate(req: GenerateRequest) -> dict[str, Any]:
    """Continue a prompt with nucleus sampling."""
    return _guard(lambda: generate(req.prompt, req.max_new, req.temperature, req.top_p,
                                   req.repetition_penalty, req.seed, req.use_cache))


@app.post("/api/tokenize", tags=["model"])
def post_tokenize(req: TextRequest) -> dict[str, Any]:
    """Split text into the model's BPE tokens."""
    return _guard(lambda: tokenize(req.text))


@app.post("/api/attention", tags=["model"])
def post_attention(req: TextRequest) -> dict[str, Any]:
    """Attention weights of every layer and head for a short prompt."""
    return _guard(lambda: attention(req.text))


@app.get("/api/training", tags=["results"])
def get_training() -> dict[str, Any]:
    """Loss curves, baselines and the deduplication counts."""
    return _guard(training)


@app.get("/api/ablations", tags=["results"])
def get_ablations() -> dict[str, Any]:
    """Each component swapped for its older equivalent, beside a second seed of the reference."""
    return _guard(ablations)


@app.get("/api/checks", tags=["results"])
def get_checks() -> dict[str, Any]:
    """Executed mechanics checks, generation timings and fixed-seed samples."""
    return _guard(checks)
