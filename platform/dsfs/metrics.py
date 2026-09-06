"""
The single source of truth for every number this portfolio publishes.

A training run opens a `RunRecord`, logs model scores into it, and saves
`artifacts/metrics.json`. The API serves that file, the README badge
generator reads it, and `dsfs.artifacts` writes paper.md from it. Nothing
in this repo hand-types a metric, so a README can never claim a score the
code did not produce.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .provenance import Provenance


@dataclass
class ModelScore:
    """One model's evaluation on the held-out set."""
    name: str
    family: str = ""                       # "gradient boosting", "linear", ...
    metrics: dict[str, float] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    fit_seconds: float = 0.0
    predict_ms_per_1k: float = 0.0
    notes: str = ""

    def get(self, key: str, default: float = float("nan")) -> float:
        return self.metrics.get(key, default)


@dataclass
class SplitInfo:
    strategy: str                          # "random", "chronological", "grouped"
    n_train: int = 0
    n_test: int = 0
    n_val: int = 0
    train_end: str = ""
    test_start: str = ""
    embargo: str = ""
    notes: str = ""


@dataclass
class RunRecord:
    """Everything one training run produced."""
    project_slug: str
    project_title: str
    task: str                               # "regression" | "classification" | ...
    primary_metric: str                     # the metric the leaderboard sorts by
    higher_is_better: bool = True
    provenance: dict[str, Any] = field(default_factory=dict)
    split: dict[str, Any] = field(default_factory=dict)
    models: list[ModelScore] = field(default_factory=list)
    baseline: dict[str, float] = field(default_factory=dict)
    audit: dict[str, Any] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    created_at: str = ""
    duration_seconds: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    # -- construction -------------------------------------------------

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if not self.env:
            self.env = capture_env()

    def set_provenance(self, prov: Provenance) -> "RunRecord":
        self.provenance = prov.to_dict()
        return self

    def set_split(self, split: SplitInfo) -> "RunRecord":
        self.split = asdict(split)
        return self

    def add_model(self, score: ModelScore) -> "RunRecord":
        self.models.append(score)
        return self

    def log(self, name: str, metrics: dict[str, float], **kw: Any) -> ModelScore:
        """Convenience: build and register a ModelScore in one call."""
        s = ModelScore(name=name, metrics={k: float(v) for k, v in metrics.items()}, **kw)
        self.models.append(s)
        return s

    # -- reading ------------------------------------------------------

    @property
    def best(self) -> ModelScore | None:
        scored = [m for m in self.models if self.primary_metric in m.metrics]
        if not scored:
            return None
        return (max if self.higher_is_better else min)(
            scored, key=lambda m: m.metrics[self.primary_metric]
        )

    def leaderboard(self) -> list[ModelScore]:
        scored = [m for m in self.models if self.primary_metric in m.metrics]
        return sorted(
            scored,
            key=lambda m: m.metrics[self.primary_metric],
            reverse=self.higher_is_better,
        )

    def lift_over_baseline(self) -> float | None:
        """Relative improvement of the best model over the naive baseline."""
        b = self.best
        base = self.baseline.get(self.primary_metric)
        if b is None or base in (None, 0):
            return None
        v = b.metrics[self.primary_metric]
        return (v - base) / abs(base) * (1.0 if self.higher_is_better else -1.0)

    # -- persistence --------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        b = self.best
        d["best_model"] = b.name if b else None
        d["best_score"] = b.metrics.get(self.primary_metric) if b else None
        lift = self.lift_over_baseline()
        d["lift_over_baseline"] = lift
        return d

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")
        return p

    @staticmethod
    def load(path: str | Path) -> dict[str, Any]:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    # -- display ------------------------------------------------------

    def leaderboard_markdown(self, extra_metrics: tuple[str, ...] = ()) -> str:
        rows = self.leaderboard()
        if not rows:
            return "_No scored models._\n"
        cols = [self.primary_metric, *extra_metrics]
        head = "| # | Model | " + " | ".join(cols) + " | Fit (s) |"
        sep = "|---|---|" + "---|" * (len(cols) + 1)
        out = [head, sep]
        for i, m in enumerate(rows, 1):
            vals = " | ".join(_fmt(m.metrics.get(c)) for c in cols)
            star = " 🏆" if m is rows[0] else ""
            out.append(f"| {i} | **{m.name}**{star} | {vals} | {m.fit_seconds:.1f} |")
        return "\n".join(out) + "\n"


def capture_env() -> dict[str, str]:
    """Record the machine so a reviewer can reproduce the numbers."""
    env = {
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
    }
    for mod in ("numpy", "pandas", "sklearn", "torch"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            pass
    try:
        env["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True, timeout=5,
        ).strip()
    except Exception:
        pass
    return env


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        if v != v:  # NaN
            return "—"
        return f"{v:,.4f}" if abs(v) < 1e4 else f"{v:,.1f}"
    return str(v)
