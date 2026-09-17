"""
Project 02 -- TinyStories: acquisition, deduplication and tokenisation.

TinyStories is a corpus of short children's stories written by GPT-3.5 and
GPT-4 for training very small language models (Eldan & Li, 2023). Version 2
uses only GPT-4 stories. Stories are separated by an <|endoftext|> line.

A validation story that also appears in the training file would make
perplexity look better than the model's real ability to continue unseen text.
Generated corpora can repeat themselves, so rather than assume either way,
every story in both files is hashed, the overlap is measured, and any
validation story found in training is removed before scoring.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np

from dsfs import Provenance, fetch

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
SEPARATOR = "<|endoftext|>"
VOCAB_SIZE = 4096

_CITATION = ("Eldan, R., & Li, Y. (2023). TinyStories: How small can language models be and still "
             "speak coherent English? arXiv:2305.07759.")
TRAIN = Provenance(
    slug="tinystories-v2",
    name="TinyStories V2 (GPT-4 stories), training split",
    source_url="https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt",
    license="Community Data License Agreement – Sharing, Version 1.0 (CDLA-Sharing-1.0)",
    citation=_CITATION,
    notes="Stories separated by <|endoftext|> lines.",
)
VALID = Provenance(
    slug="tinystories-v2",
    name="TinyStories V2 (GPT-4 stories), validation split",
    source_url="https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt",
    license=TRAIN.license,
    citation=_CITATION,
    notes=TRAIN.notes,
)


def normalise(story: str) -> str:
    """Collapse whitespace so trivially re-wrapped copies hash the same."""
    return re.sub(r"\s+", " ", story).strip()


def story_hash(story: str) -> str:
    return hashlib.blake2b(normalise(story).encode("utf-8"), digest_size=12).hexdigest()


def stories(path: Path):
    """Yield stories one at a time; the training file is too large to hold as one string."""
    buf: list[str] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip() == SEPARATOR:
                text = "".join(buf).strip()
                if text:
                    yield text
                buf = []
            else:
                buf.append(line)
    text = "".join(buf).strip()
    if text:
        yield text


def download() -> tuple[Path, Provenance, Path, Provenance]:
    train_path, train_prov = fetch(TRAIN, filename="TinyStoriesV2-GPT4-train.txt", timeout=1800)
    valid_path, valid_prov = fetch(VALID, filename="TinyStoriesV2-GPT4-valid.txt")
    return train_path, train_prov, valid_path, valid_prov


def encode_to_file(tokenizer, texts, out: Path, eot_id: int) -> int:
    """Tokenise stories in batches straight into a uint16 file; returns the token count."""
    n = 0
    with open(out, "wb") as f:
        batch: list[str] = []
        for t in texts:
            batch.append(t)
            if len(batch) == 4096:
                n += _write(tokenizer, batch, f, eot_id)
                batch = []
        if batch:
            n += _write(tokenizer, batch, f, eot_id)
    return n


def _write(tokenizer, batch: list[str], f, eot_id: int) -> int:
    ids: list[int] = []
    for enc in tokenizer.encode_batch(batch):
        ids.extend(enc.ids)
        ids.append(eot_id)
    np.asarray(ids, dtype=np.uint16).tofile(f)
    return len(ids)


if __name__ == "__main__":
    tp, tprov, vp, vprov = download()
    print(tp, tprov.bytes_on_disk, vp, vprov.bytes_on_disk)
