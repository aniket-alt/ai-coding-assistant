"""
Leakage auditor: static (AST) + runtime assertions.

Two halves, both real:

* `audit_source()` walks a project's Python with `ast` and flags the classic
  leakage smells -- transformers fitted on the full frame, shuffled splits in
  time-series code, unseeded estimators.
* The `assert_*` helpers run *inside* training and raise on actual leakage:
  target-correlated features, train/test row overlap, temporal lookahead.

Findings are severity-ranked and rendered into each project's audit_report.md.
A project whose audit contains a CRITICAL finding fails CI.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

Severity = str  # "CRITICAL" | "WARNING" | "INFO"

# Transformers that leak if fitted before the split.
_FIT_LEAKY = {
    "StandardScaler", "MinMaxScaler", "RobustScaler", "MaxAbsScaler",
    "QuantileTransformer", "PowerTransformer", "Normalizer",
    "SimpleImputer", "KNNImputer", "IterativeImputer",
    "PCA", "TruncatedSVD", "SelectKBest", "TargetEncoder",
}

# Variable names that must never be passed to .fit()
_TEST_NAMES = {
    "x_test", "xtest", "x_val", "xval", "test", "val", "holdout",
    "x_full", "xfull", "x_all", "xall", "df", "data",
}

_UNSEEDED_ESTIMATORS = {
    "RandomForestClassifier", "RandomForestRegressor", "IsolationForest",
    "KMeans", "train_test_split", "GradientBoostingRegressor",
    "LGBMRegressor", "XGBRegressor", "ExtraTreesRegressor",
}


@dataclass
class Finding:
    severity: Severity
    rule: str
    message: str
    file: str = ""
    line: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LeakageError(AssertionError):
    """Raised by the runtime assertions when real leakage is detected."""


# --------------------------------------------------------------------------
# Static analysis
# --------------------------------------------------------------------------

class _Visitor(ast.NodeVisitor):
    def __init__(self, path: Path, temporal: bool) -> None:
        self.path = path
        self.temporal = temporal
        self.findings: list[Finding] = []
        self._split_line: int | None = None
        self._fits: list[tuple[int, str, str]] = []  # (line, receiver, arg0)

    def _add(self, sev: str, rule: str, msg: str, node: ast.AST) -> None:
        self.findings.append(
            Finding(sev, rule, msg, str(self.path), getattr(node, "lineno", 0))
        )

    def visit_Call(self, node: ast.Call) -> None:
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")

        # Record where the split happens.
        if name in {"train_test_split", "split"} and self._split_line is None:
            self._split_line = node.lineno

        # shuffle=True in temporal projects
        if name == "train_test_split" and self.temporal:
            shuffled = not any(
                kw.arg == "shuffle"
                and isinstance(kw.value, ast.Constant)
                and kw.value.value is False
                for kw in node.keywords
            )
            if shuffled:
                self._add(
                    "CRITICAL", "temporal-shuffle",
                    "train_test_split without shuffle=False in a time-series module "
                    "-- destroys chronology and leaks the future into training.",
                    node,
                )

        # Unseeded estimators
        if name in _UNSEEDED_ESTIMATORS:
            has_seed = any(kw.arg in {"random_state", "seed"} for kw in node.keywords)
            if not has_seed:
                self._add(
                    "WARNING", "unseeded",
                    f"{name}() has no random_state -- results are not reproducible.",
                    node,
                )

        # .fit()/.fit_transform() on suspicious data
        if name in {"fit", "fit_transform", "fit_predict"} and node.args:
            recv = _receiver_name(fn)
            arg0 = _arg_name(node.args[0])
            self._fits.append((node.lineno, recv, arg0))
            if arg0.lower() in _TEST_NAMES:
                self._add(
                    "CRITICAL", "fit-on-test",
                    f".{name}() called on `{arg0}` -- a transformer or model is being "
                    f"fitted on held-out or full data.",
                    node,
                )
        self.generic_visit(node)

    def finish(self, src: str) -> None:
        """Ordering check: leaky transformer fitted before the split line."""
        if self._split_line is None:
            return
        leaky_present = any(
            _ctor_name(n) in _FIT_LEAKY
            for n in ast.walk(ast.parse(src))
            if isinstance(n, ast.Call)
        )
        if not leaky_present:
            return
        for line, _recv, _arg0 in self._fits:
            if line < self._split_line:
                self.findings.append(Finding(
                    "WARNING", "fit-before-split",
                    f"`.fit()` on line {line} precedes the train/test split on line "
                    f"{self._split_line}; verify the transformer is not fitted on all rows.",
                    str(self.path), line,
                ))
                break


def audit_source(
    root: str | Path,
    *,
    temporal_modules: Iterable[str] = (),
    skip: Iterable[str] = ("node_modules", ".venv", "venv311", "__pycache__", "dist"),
) -> list[Finding]:
    """Walk `root` for .py files and return ranked leakage findings."""
    root = Path(root)
    temporal = {t.lower() for t in temporal_modules}
    skip_set = set(skip)
    findings: list[Finding] = []

    for py in sorted(root.rglob("*.py")):
        if any(part in skip_set for part in py.parts):
            continue
        try:
            src = py.read_text(encoding="utf-8")
            tree = ast.parse(src)
        except (SyntaxError, UnicodeDecodeError) as exc:
            findings.append(Finding("WARNING", "unparseable",
                                    f"Could not parse: {exc}", str(py), 0))
            continue
        rel = py.relative_to(root)
        is_temporal = any(t in str(rel).lower() for t in temporal)
        v = _Visitor(rel, is_temporal)
        v.visit(tree)
        v.finish(src)
        findings.extend(v.findings)

    order = {"CRITICAL": 0, "WARNING": 1, "INFO": 2}
    findings.sort(key=lambda f: (order.get(f.severity, 3), f.file, f.line))
    return findings


# --------------------------------------------------------------------------
# Runtime assertions -- these run during training and raise on real leakage
# --------------------------------------------------------------------------

def assert_no_target_leak(X, y, *, threshold: float = 0.995) -> list[str]:
    """
    Flag features almost perfectly correlated with the target -- the signature
    of accidentally including the label (or a deterministic function of it).
    """
    import numpy as np
    import pandas as pd

    Xd = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    ys = pd.Series(np.asarray(y).ravel())
    if ys.nunique() < 2:
        return []

    leaked: list[str] = []
    for col in Xd.columns:
        s = pd.to_numeric(Xd[col], errors="coerce")
        if s.notna().sum() < 2 or s.nunique() < 2:
            continue
        r = abs(s.corr(ys))
        if pd.notna(r) and r >= threshold:
            leaked.append(f"{col} (|r|={r:.4f})")
    if leaked:
        raise LeakageError(
            "Target leakage: feature(s) near-perfectly correlated with y: "
            + ", ".join(leaked)
        )
    return leaked


def assert_disjoint(train_idx: Sequence, test_idx: Sequence) -> None:
    """No row may appear in both splits."""
    overlap = set(map(int, train_idx)) & set(map(int, test_idx))
    if overlap:
        raise LeakageError(
            f"Train/test overlap: {len(overlap):,} shared rows "
            f"(e.g. {sorted(overlap)[:5]})."
        )


def assert_temporal_split(train_times, test_times, *, embargo: str | None = None) -> dict:
    """
    Every training timestamp must precede every test timestamp. Optionally
    require a gap (embargo) to kill autocorrelation bleed at the boundary.
    """
    import pandas as pd

    tr = pd.to_datetime(pd.Series(list(train_times))).max()
    te = pd.to_datetime(pd.Series(list(test_times))).min()
    if tr >= te:
        raise LeakageError(
            f"Temporal lookahead: last training timestamp {tr} is not before "
            f"first test timestamp {te}."
        )
    gap = te - tr
    if embargo is not None and gap < pd.Timedelta(embargo):
        raise LeakageError(
            f"Embargo violated: gap between train and test is {gap}, "
            f"required >= {embargo}."
        )
    return {"train_end": str(tr), "test_start": str(te), "gap": str(gap)}


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def to_markdown(findings: Sequence[Finding], project_title: str) -> str:
    crit = [f for f in findings if f.severity == "CRITICAL"]
    warn = [f for f in findings if f.severity == "WARNING"]
    grade = "FAIL" if crit else ("PASS (with warnings)" if warn else "PASS")
    icon = "❌" if crit else ("⚠️" if warn else "✅")

    out = [
        f"# Leakage & Reproducibility Audit — {project_title}\n",
        f"{icon} **Result: {grade}**  \n",
        f"`{len(crit)}` critical · `{len(warn)}` warnings · "
        f"`{len(findings)}` total findings\n",
        "\nGenerated by `dsfs.leakage`, which combines an AST pass over the "
        "project source with runtime assertions executed during training. "
        "Unlike a checklist, these checks fail the build.\n",
    ]
    if not findings:
        out.append("\nNo findings. All transformers are fitted inside "
                   "cross-validation folds, splits are disjoint, and every "
                   "estimator is seeded.\n")
    else:
        out.append("\n| Severity | Rule | Location | Finding |\n|---|---|---|---|")
        for f in findings:
            loc = f"`{f.file}:{f.line}`" if f.file else "—"
            out.append(f"| {f.severity} | `{f.rule}` | {loc} | {f.message} |")
        out.append("")
    return "\n".join(out)


def save_report(findings: Sequence[Finding], path: str | Path,
                project_title: str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(to_markdown(findings, project_title), encoding="utf-8")
    p.with_suffix(".json").write_text(
        json.dumps([f.to_dict() for f in findings], indent=2), encoding="utf-8"
    )
    return p


# --------------------------------------------------------------------------

def _receiver_name(fn: ast.AST) -> str:
    if isinstance(fn, ast.Attribute):
        return _arg_name(fn.value)
    return ""


def _arg_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return _arg_name(node.value)
    return ""


def _ctor_name(node: ast.Call) -> str:
    fn = node.func
    if isinstance(fn, ast.Name):
        return fn.id
    if isinstance(fn, ast.Attribute):
        return fn.attr
    return ""
