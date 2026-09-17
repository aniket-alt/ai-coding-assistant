"""
Project 14 -- CLIP embeddings for every product image and name.

CLIP (Radford et al., 2021) maps images and text into one space, so the same
model gives image features, text features, zero-shot classification and text
search over images. Embeddings are computed once on the GPU and cached; the
file records the model and revision it came from.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from data import decode

MODEL = "openai/clip-vit-base-patch32"
WORK = Path(__file__).resolve().parents[1] / "artifacts" / "work"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_clip():
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(MODEL).to(DEVICE).eval()
    return model, CLIPProcessor.from_pretrained(MODEL)


@torch.no_grad()
def embed_images(model, processor, images: list[bytes], batch: int = 256) -> np.ndarray:
    out = []
    for i in range(0, len(images), batch):
        pixels = processor(images=[decode(b) for b in images[i:i + batch]], return_tensors="pt")["pixel_values"].to(DEVICE)
        with torch.autocast(DEVICE, dtype=torch.float16, enabled=DEVICE == "cuda"):
            f = model.get_image_features(pixel_values=pixels)
        f = f.pooler_output if hasattr(f, "pooler_output") else f
        out.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def embed_texts(model, processor, texts: list[str], batch: int = 512) -> np.ndarray:
    out = []
    for i in range(0, len(texts), batch):
        tok = processor(text=texts[i:i + batch], return_tensors="pt", padding=True, truncation=True, max_length=77)
        f = model.get_text_features(input_ids=tok["input_ids"].to(DEVICE), attention_mask=tok["attention_mask"].to(DEVICE))
        f = f.pooler_output if hasattr(f, "pooler_output") else f
        out.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy())
    return np.concatenate(out)


def cached(ids: np.ndarray, images: list[bytes], names: list[str]) -> dict:
    """Image and name embeddings for every id, computed once and reused while the ids match."""
    WORK.mkdir(parents=True, exist_ok=True)
    path = WORK / "clip.npz"
    if path.exists():
        z = np.load(path, allow_pickle=False)
        if np.array_equal(z["ids"], ids) and str(z["model"]) == MODEL:
            return {"image": z["image"], "text": z["text"], "seconds": float(z["seconds"]), "model": MODEL}
    model, processor = load_clip()
    t = time.perf_counter()
    image = embed_images(model, processor, images)
    text = embed_texts(model, processor, names)
    seconds = time.perf_counter() - t
    np.savez(path, ids=ids, image=image, text=text, seconds=seconds, model=MODEL)
    return {"image": image, "text": text, "seconds": seconds, "model": MODEL}
