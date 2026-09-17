"""
Project 02 -- serving the trained language model.

Generation, tokenisation and attention maps come from the checkpoint and
tokenizer that training saved together. Training curves, ablations and
samples are read from metrics.json.
"""

from __future__ import annotations

import json
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))
ARTIFACTS = ROOT / "artifacts"

MAX_ATTENTION_TOKENS = 48


class ModelNotTrained(RuntimeError):
    """Raised when the API starts before `python ml/train.py` has run."""


@lru_cache(maxsize=1)
def run_record() -> dict[str, Any]:
    p = ARTIFACTS / "metrics.json"
    if not p.exists():
        raise ModelNotTrained("artifacts/metrics.json not found -- run ml/train.py.")
    return json.loads(p.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def loaded() -> tuple[Any, Any, str]:
    ckpt, tok = ARTIFACTS / "model.pt", ARTIFACTS / "tokenizer.json"
    if not ckpt.exists() or not tok.exists():
        raise ModelNotTrained("artifacts/model.pt or tokenizer.json not found -- run ml/train.py.")
    import torch
    from tokenizers import Tokenizer

    from model import Config, NanoLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(ckpt, map_location=device, weights_only=True)
    model = NanoLM(Config(**state["config"])).to(device)
    model.load_state_dict(state["state"])
    model.eval()
    return model, Tokenizer.from_file(str(tok)), device


def _pieces(tokenizer, ids: list[int]) -> list[dict[str, Any]]:
    return [{"id": i, "text": tokenizer.decode([i])} for i in ids]


def tokenize(text: str) -> dict[str, Any]:
    _, tokenizer, _ = loaded()
    ids = tokenizer.encode(text).ids
    return {"tokens": _pieces(tokenizer, ids), "characters": len(text), "count": len(ids),
            "vocab_size": tokenizer.get_vocab_size()}


def generate(prompt: str, max_new: int, temperature: float, top_p: float, repetition_penalty: float,
             seed: int, use_cache: bool) -> dict[str, Any]:
    import torch

    model, tokenizer, device = loaded()
    ids = tokenizer.encode(prompt).ids
    if not ids:
        raise ValueError("prompt is empty after tokenisation")
    eot = tokenizer.token_to_id("<|endoftext|>")
    if device == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    out = model.generate(ids, max_new, temperature=temperature, top_p=top_p,
                         repetition_penalty=repetition_penalty, use_cache=use_cache, stop_id=eot, seed=seed)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t
    text_ids = [i for i in out if i != eot]
    return {"prompt": prompt, "continuation": tokenizer.decode(text_ids), "tokens": _pieces(tokenizer, text_ids),
            "prompt_tokens": len(ids), "generated_tokens": len(out), "stopped_at_end_of_story": bool(out and out[-1] == eot),
            "ms_per_token": elapsed / max(len(out), 1) * 1000, "device": device, "kv_cache": use_cache}


def attention(prompt: str) -> dict[str, Any]:
    import torch

    model, tokenizer, device = loaded()
    ids = tokenizer.encode(prompt).ids[:MAX_ATTENTION_TOKENS]
    if not ids:
        raise ValueError("prompt is empty after tokenisation")
    with torch.no_grad():
        _, _, maps = model(torch.tensor([ids], device=device), return_weights=True)
    return {"tokens": _pieces(tokenizer, ids), "truncated_to": MAX_ATTENTION_TOKENS,
            "layers": [[[[round(float(w), 4) for w in row] for row in head] for head in layer[0].float().cpu()]
                       for layer in maps]}


def training() -> dict[str, Any]:
    e = run_record()["extra"]
    return {"main": e["main"], "ngram": e["ngram"], "dedup": e["dedup"], "chars_per_token": e["chars_per_token"],
            "hardware": e["hardware"], "torch": e["torch"]}


def ablations() -> dict[str, Any]:
    e = run_record()["extra"]
    return {"ablations": e["ablations"], "seed_gap": e["seed_gap"], "steps": e["ablation_steps"]}


def checks() -> dict[str, Any]:
    e = run_record()["extra"]
    return {"checks": e["checks"], "samples": e["samples"]}
