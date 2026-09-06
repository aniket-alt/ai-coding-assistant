"""
Content-addressed dataset cache.

`fetch()` downloads a URL once into `datasets/cache/`, checksums it, and
returns a stamped `Provenance`. Re-running is free and offline-safe. Every
project's data step goes through here, so "where did this number come from"
always has an answer.
"""

from __future__ import annotations

import os
import shutil
import urllib.request
from pathlib import Path
from typing import Callable

from .provenance import Provenance, sha256_file

_UA = "ds-fullstack-ai/1.0 (academic portfolio; +https://github.com/)"


def cache_dir() -> Path:
    """Repo-level dataset cache. Override with DSFS_CACHE."""
    env = os.environ.get("DSFS_CACHE")
    if env:
        d = Path(env)
    else:
        # platform/dsfs/datasets.py -> repo root
        d = Path(__file__).resolve().parents[2] / "datasets" / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def fetch(
    spec: Provenance,
    filename: str | None = None,
    *,
    force: bool = False,
    timeout: int = 300,
    progress: bool = True,
) -> tuple[Path, Provenance]:
    """
    Download `spec.source_url` into the cache (once) and return (path, provenance).

    The returned provenance carries retrieved_at, sha256 and bytes_on_disk.
    Row/col counts are filled later by `stamp_frame`.
    """
    name = filename or spec.source_url.split("/")[-1].split("?")[0] or spec.slug
    dest = cache_dir() / spec.slug / name

    if dest.exists() and not force:
        return dest, spec.stamped(
            sha256=sha256_file(dest),
            bytes_on_disk=dest.stat().st_size,
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    if progress:
        print(f"[dsfs] downloading {spec.name}\n       {spec.source_url}")

    req = urllib.request.Request(spec.source_url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as out:
        shutil.copyfileobj(resp, out, length=1 << 20)
    tmp.replace(dest)

    size = dest.stat().st_size
    if progress:
        print(f"[dsfs] cached -> {dest}  ({size / 1e6:.1f} MB)")

    return dest, spec.stamped(sha256=sha256_file(dest), bytes_on_disk=size)


def fetch_via(
    spec: Provenance,
    loader: Callable[[Path], None],
    filename: str,
    *,
    force: bool = False,
) -> tuple[Path, Provenance]:
    """
    Cache data that arrives through a library rather than a plain URL
    (yfinance, kagglehub, HuggingFace). `loader` receives the destination
    path and must write the file; it is called only on a cache miss.
    """
    dest = cache_dir() / spec.slug / filename
    if not dest.exists() or force:
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"[dsfs] materializing {spec.name} via loader -> {dest}")
        loader(dest)
    if not dest.exists():
        raise RuntimeError(f"loader did not produce {dest}")
    return dest, spec.stamped(
        sha256=sha256_file(dest),
        bytes_on_disk=dest.stat().st_size,
    )


def stamp_frame(spec: Provenance, df) -> Provenance:
    """Attach shape facts from a loaded DataFrame."""
    return spec.stamped(n_rows=int(df.shape[0]), n_cols=int(df.shape[1]))
