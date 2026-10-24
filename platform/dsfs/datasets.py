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


def read_arff(raw: bytes) -> "pd.DataFrame":
    """
    Parse an ARFF file, the format OpenML serves, into a DataFrame.

    Only what OpenML's files actually use is handled: an `@attribute` line per
    column, then `@data` and comma-separated rows. Missing values are written
    as `?`.

    Files differ in how they quote. OpenML's Adult wraps values in single
    quotes, its Titanic in double quotes, and a passenger name there contains a
    comma -- so a fixed quote character silently splits one row into two
    columns. Both are tried, and the one that yields the declared number of
    columns is used; any quotes the other convention left behind are stripped.
    """
    import io

    import pandas as pd

    text = raw.decode("utf-8", errors="replace").splitlines()
    columns, start = [], 0
    for i, line in enumerate(text):
        low = line.strip().lower()
        if low.startswith("@attribute"):
            columns.append(line.split()[1].strip("'\""))
        elif low.startswith("@data"):
            start = i + 1
            break
    if not columns:
        raise ValueError("no @attribute lines found; is this an ARFF file?")

    body = "\n".join(text[start:])
    frame = None
    for quote in ('"', "'"):
        try:
            candidate = pd.read_csv(io.StringIO(body), header=None, names=columns, quotechar=quote,
                                    skipinitialspace=True, na_values=["?"])
        except pd.errors.ParserError:
            continue
        if candidate.shape[1] == len(columns):
            frame = candidate
            break
    if frame is None:
        raise ValueError(f"could not parse the @data rows into {len(columns)} columns")

    # Tested with is_numeric_dtype rather than `== object`: pandas 3 reads text
    # into a string dtype, which is not object, so an object test would skip
    # every column that needs stripping.
    for column in frame.columns:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            frame[column] = frame[column].map(
                lambda v: v[1:-1] if isinstance(v, str) and len(v) > 1 and v[0] == v[-1] and v[0] in "\"'" else v)
    return frame
