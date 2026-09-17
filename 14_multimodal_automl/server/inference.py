"""
Project 14 -- serving the multimodal results.

Test products are served with every contestant's recorded prediction, product
photos are read from the dataset cache, typed names are classified live by the
two text models, and descriptions search the test photos through CLIP.
"""

from __future__ import annotations

import json
import sys
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))
ARTIFACTS = ROOT / "artifacts"

SHOWN = ["Image only (CLIP + logistic)", "Name only, label words removed (TF-IDF)", "Catalogue fields only (LightGBM)",
         "Late fusion: image + masked name + fields", "Late fusion: image + name + fields"]


class ModelNotTrained(RuntimeError):
    """Raised when the API starts before `python ml/train.py` has run."""


@lru_cache(maxsize=1)
def run_record() -> dict[str, Any]:
    p = ARTIFACTS / "metrics.json"
    if not p.exists():
        raise ModelNotTrained("artifacts/metrics.json not found -- run ml/train.py.")
    return json.loads(p.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def served() -> dict[str, Any]:
    p, b = ARTIFACTS / "work" / "served.npz", ARTIFACTS / "model.joblib"
    if not p.exists() or not b.exists():
        raise ModelNotTrained("served predictions not found -- run ml/train.py.")
    z = np.load(p)
    bundle = joblib.load(b)
    names = bundle["model_names"]
    return {"ids": z["ids"], "labels": z["labels"], "image_emb": z["image_emb"], "classes": bundle["classes"],
            "text_models": bundle["text_models"], "vocab": bundle["vocab"],
            "proba": {n: z[f"p{i}"] for i, n in enumerate(names)},
            "row": {int(i): k for k, i in enumerate(z["ids"])}}


@lru_cache(maxsize=1)
def catalogue() -> dict[int, dict[str, Any]]:
    from data import TABULAR, load

    df, _ = load()
    wanted = set(int(i) for i in served()["ids"])
    df = df[df["id"].isin(wanted)]
    return {int(r["id"]): {"name": r["productDisplayName"], "image": r["image_bytes"],
                           **{c: _field(r[c]) for c in TABULAR}}
            for _, r in df.iterrows()}


def _field(v: Any) -> str | None:
    """Catalogue values as text; years are stored as floats, so 2012.0 is shown as 2012."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)


_clip_lock = threading.Lock()


@lru_cache(maxsize=1)
def clip():
    from embed import load_clip

    with _clip_lock:
        return load_clip()


def _top(p: np.ndarray, classes: list[str], k: int) -> list[dict[str, Any]]:
    order = np.argsort(-p)[:k]
    return [{"article_type": classes[i], "probability": float(p[i])} for i in order]


def gallery(article_type: str | None, page: int, per_page: int = 24) -> dict[str, Any]:
    s = served()
    classes = s["classes"]
    idx = np.arange(len(s["ids"]))
    if article_type:
        if article_type not in classes:
            raise KeyError(article_type)
        idx = idx[s["labels"] == classes.index(article_type)]
    fusion = s["proba"]["Late fusion: image + masked name + fields"]
    image = s["proba"]["Image only (CLIP + logistic)"]
    chunk = idx[page * per_page:(page + 1) * per_page]
    cat = catalogue()
    return {"total": int(len(idx)), "page": page, "per_page": per_page, "classes": classes,
            "items": [{"id": int(s["ids"][k]), "name": cat[int(s["ids"][k])]["name"], "actual": classes[s["labels"][k]],
                       "image_prediction": classes[int(image[k].argmax())],
                       "fusion_prediction": classes[int(fusion[k].argmax())]} for k in chunk]}


def product(product_id: int) -> dict[str, Any]:
    s = served()
    if product_id not in s["row"]:
        raise KeyError(product_id)
    k = s["row"][product_id]
    item = catalogue()[product_id]
    from data import mask_label_words

    return {"id": product_id, "name": item["name"], "masked_name": mask_label_words(item["name"], s["vocab"]),
            "fields": {c: v for c, v in item.items() if c not in ("name", "image")},
            "actual": s["classes"][s["labels"][k]],
            "predictions": [{"model": n, "top": _top(s["proba"][n][k], s["classes"], 3)} for n in SHOWN]}


def image(product_id: int) -> bytes:
    item = catalogue().get(product_id)
    if item is None:
        raise KeyError(product_id)
    return item["image"]


def classify_name(name: str) -> dict[str, Any]:
    s = served()
    from data import mask_label_words

    out = {"name": name, "masked_name": mask_label_words(name, s["vocab"]), "models": []}
    for label, (vec, model) in s["text_models"].items():
        text = out["masked_name"] if "removed" in label else name
        p = model.predict_proba(vec.transform([text]))[0]
        out["models"].append({"model": f"{label} (TF-IDF)", "top": _top(p, s["classes"], 5)})
    return out


def search(query: str, k: int = 24) -> dict[str, Any]:
    from embed import embed_texts

    s = served()
    model, processor = clip()
    q = embed_texts(model, processor, [query])[0]
    sims = s["image_emb"] @ q
    order = np.argsort(-sims)[:k]
    cat = catalogue()
    return {"query": query, "gallery": int(len(s["ids"])),
            "results": [{"id": int(s["ids"][i]), "name": cat[int(s["ids"][i])]["name"], "actual": s["classes"][s["labels"][i]],
                         "similarity": float(sims[i])} for i in order]}


def results() -> dict[str, Any]:
    e = run_record()["extra"]
    return {k: e[k] for k in ("classes", "class_counts", "leaks", "significance", "best_single", "per_class",
                              "image_confusions", "search", "autogluon", "clip", "fusion_reported")}
