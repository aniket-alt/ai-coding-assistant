"""
Dataset provenance records.

Thesis of this portfolio: every number in every README traces to a real,
citable dataset. A `Provenance` is the receipt. Projects cannot register a
dataset without one, and `dsfs.artifacts` refuses to emit a paper for a
dataset whose provenance is marked synthetic unless the caller opts in
explicitly -- which then stamps "SYNTHETIC" across the generated docs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Provenance:
    """Where a dataset actually came from."""

    slug: str                      # stable id, e.g. "nyc-tlc-yellow-2024-01"
    name: str                      # human title
    source_url: str                # the canonical, citable URL
    license: str                   # e.g. "NYC TLC Open Data / public domain"
    citation: str                  # how to cite it in paper.md
    retrieved_at: str = ""         # ISO8601 UTC, filled on download
    sha256: str = ""               # checksum of the cached bytes
    n_rows: int = 0
    n_cols: int = 0
    bytes_on_disk: int = 0
    synthetic: bool = False        # True only for deliberately generated data
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def stamped(self, **kwargs: Any) -> "Provenance":
        """Return a copy with download-time facts filled in."""
        merged = {**asdict(self), **kwargs}
        if not merged.get("retrieved_at"):
            merged["retrieved_at"] = _utcnow()
        return Provenance(**merged)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        """Provenance block for README / paper.md."""
        flag = "⚠️ **SYNTHETIC**" if self.synthetic else "✅ **Real, third-party data**"
        rows = f"{self.n_rows:,}" if self.n_rows else "—"
        cols = f"{self.n_cols:,}" if self.n_cols else "—"
        size = _human_bytes(self.bytes_on_disk) if self.bytes_on_disk else "—"
        return (
            f"### Dataset provenance\n\n"
            f"{flag}\n\n"
            f"| Field | Value |\n|---|---|\n"
            f"| Dataset | {self.name} |\n"
            f"| Slug | `{self.slug}` |\n"
            f"| Source | {self.source_url} |\n"
            f"| License | {self.license} |\n"
            f"| Retrieved | {self.retrieved_at or '—'} |\n"
            f"| SHA-256 | `{self.sha256[:16] + '…' if self.sha256 else '—'}` |\n"
            f"| Rows × Cols | {rows} × {cols} |\n"
            f"| Size on disk | {size} |\n\n"
            f"> {self.citation}\n"
        )

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> "Provenance":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _human_bytes(n: int) -> str:
    step = 1024.0
    val = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if val < step:
            return f"{val:.1f} {unit}"
        val /= step
    return f"{val:.1f} TB"
