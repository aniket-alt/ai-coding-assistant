"""
Project 14 -- fashion products: images, names and catalogue attributes.

44,072 products from a fashion retailer's catalogue, published on Kaggle
by Param Aggarwal as "Fashion Product Images (Small)" under the MIT licence, and
read here from a Hugging Face parquet mirror pinned to one revision. Every
product has a small photograph, a display name and catalogue fields.

The task is to predict a product's article type (shirt, watch, sports shoe...)
from each kind of information and from all of them together. Two properties of
the catalogue decide how that has to be evaluated:

* The display name often contains the answer: "Turtle Check Men Navy Blue
  Shirt" is labelled Shirts. How much of the text model's success is that one
  word is measured by masking it.
* Many products share a display name. A random split puts such siblings on
  both sides, so the split keeps every product
  with the same name together.
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import numpy as np
import pandas as pd

from dsfs import Provenance, fetch, stamp_frame

REVISION = "3859c76db2f6f3d3b9a3863345e3ccdbff75879d"
BASE = f"https://huggingface.co/datasets/ashraq/fashion-product-images-small/resolve/{REVISION}/data"
SHARDS = ["train-00000-of-00002-6cff4c59f91661c3.parquet", "train-00001-of-00002-bb459e5ac5f01e71.parquet"]

FASHION = Provenance(
    slug="fashion-product-images-small",
    name="Fashion Product Images (Small)",
    source_url=f"{BASE}/{SHARDS[0]}",
    license="MIT, as declared by the Kaggle dataset; product photographs originate from a retailer's catalogue",
    citation=("Aggarwal, P. (2019). Fashion Product Images (Small) [Dataset]. Kaggle. "
              "https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small"),
    notes=f"Hugging Face mirror ashraq/fashion-product-images-small at revision {REVISION}, two parquet shards.",
)

TARGET = "articleType"
TABULAR = ["gender", "baseColour", "season", "year", "usage"]
# masterCategory and subCategory are coarser levels of the same hierarchy as
# articleType, so they would hand the model most of the answer.
HIERARCHY = ["masterCategory", "subCategory"]
MIN_PER_CLASS = 150


def load() -> tuple[pd.DataFrame, Provenance]:
    frames, prov = [], None
    for shard in SHARDS:
        spec = FASHION if shard == SHARDS[0] else Provenance(**{**FASHION.__dict__, "source_url": f"{BASE}/{shard}"})
        path, p = fetch(spec, filename=shard, timeout=1800)
        prov = prov or p
        frames.append(pd.read_parquet(path))
    df = pd.concat(frames, ignore_index=True)
    prov = stamp_frame(prov, df.drop(columns=["image"]))
    df["image_bytes"] = df["image"].map(lambda d: d["bytes"])
    df = df.drop(columns=["image"])
    for c in ["gender", "baseColour", "season", "usage", "productDisplayName", TARGET] + HIERARCHY:
        df[c] = df[c].astype(object).where(df[c].notna(), None)
    df["name_key"] = df["productDisplayName"].fillna("").str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    return df, prov


def label_words(label: str) -> set[str]:
    """Words of an article type and their singular forms: 'Sports Shoes' -> sports, shoes, sport, shoe."""
    words = set(re.findall(r"[a-z]+", label.lower()))
    return words | {w[:-1] for w in words if w.endswith("s") and len(w) > 3} | {w[:-2] for w in words if w.endswith("es") and len(w) > 4}


def mask_label_words(name: str, vocabulary: set[str]) -> str:
    """Remove every word that appears in any article-type name, so the label cannot be read off the text."""
    return " ".join(w for w in re.findall(r"[a-z0-9]+", (name or "").lower()) if w not in vocabulary)


def decode(image_bytes: bytes):
    from PIL import Image

    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


if __name__ == "__main__":
    df, prov = load()
    print(df.shape, prov.sha256[:12])
    counts = df[TARGET].value_counts()
    print(len(counts), "article types;", (counts >= MIN_PER_CLASS).sum(), f"with at least {MIN_PER_CLASS} products")
    kept = df[df[TARGET].isin(counts[counts >= MIN_PER_CLASS].index)]
    contains = kept.apply(lambda r: bool(label_words(r[TARGET]) & set(re.findall(r"[a-z]+", (r["productDisplayName"] or "").lower()))), axis=1)
    print(f"{len(kept):,} products kept; name contains its own label word: {contains.mean():.1%}")
    print("duplicate names:", kept["name_key"].duplicated().sum())
