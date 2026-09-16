"""
Document generation from run artifacts.

Every figure in a generated document is read from the artifacts a training
run wrote: metrics.json, crispdm.json, provenance.json and audit.json. The
only hand-written inputs are a project's docs.json -- metric labels, formats,
references and qualitative notes -- which by convention contain no results.

Output is deterministic (no timestamps of its own, sorted file walks), so
`--check` can regenerate everything and fail if a committed document has
drifted from the artifacts it claims to describe.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PLATFORM_ROUTES = [
    ("GET", "/api/health", "Liveness, version and whether artifacts are present."),
    ("GET", "/api/meta", "Project identity, ports and techniques."),
    ("GET", "/api/metrics", "The full run record: models, split, audit, environment."),
    ("GET", "/api/crispdm", "The six-phase CRISP-DM report written during training."),
    ("GET", "/api/audit", "Static leakage findings."),
    ("GET", "/api/provenance", "The dataset receipt."),
    ("GET", "/api/docs", "Interactive OpenAPI documentation."),
]

PHASE_TITLES = {
    "business_understanding": "Business understanding",
    "data_understanding": "Data understanding",
    "data_preparation": "Data preparation",
    "modeling": "Modeling",
    "evaluation": "Evaluation",
    "deployment": "Deployment",
}

SOURCE_DIRS = ("ml", "server", "client/src")
SOURCE_EXT = {".py", ".ts", ".tsx"}


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

@dataclass
class Component:
    path: str
    lines: int
    summary: str


@dataclass
class Route:
    method: str
    path: str
    summary: str


@dataclass
class Project:
    root: Path
    slug: str
    meta: dict[str, Any]
    metrics: dict[str, Any]
    crispdm: dict[str, Any]
    provenance: dict[str, Any]
    audit: list[dict[str, Any]]
    config: dict[str, Any]
    screenshots: list[str] = field(default_factory=list)
    routes: list[Route] = field(default_factory=list)
    components: list[Component] = field(default_factory=list)

    @property
    def number(self) -> str:
        return f"{int(self.meta.get('number', 0)):02d}"

    @property
    def title(self) -> str:
        return self.meta.get("title") or self.metrics.get("project_title", self.slug)

    def phase(self, name: str) -> dict[str, Any]:
        for p in self.crispdm.get("phases", []):
            if p["name"] == name:
                return p
        return {}


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def load_project(root: Path) -> Project:
    art = root / "artifacts"
    metrics = _read_json(art / "metrics.json", None)
    if metrics is None:
        raise FileNotFoundError(f"{root.name}: artifacts/metrics.json missing -- train first")
    shots = sorted(p.name for p in (root / "screenshots").glob("*.png")) \
        if (root / "screenshots").exists() else []
    return Project(
        root=root,
        slug=root.name,
        meta=parse_meta(root / "server" / "main.py"),
        metrics=metrics,
        crispdm=_read_json(art / "crispdm.json", {}),
        provenance=_read_json(art / "provenance.json", {}),
        audit=_read_json(art / "audit.json", []),
        config=_read_json(root / "docs.json", {}),
        screenshots=shots,
        routes=parse_routes(root / "server" / "main.py"),
        components=scan_components(root, SOURCE_DIRS),
    )


def parse_meta(main_py: Path) -> dict[str, Any]:
    """Read the ProjectMeta(...) literal without importing the server."""
    if not main_py.exists():
        return {}
    tree = ast.parse(main_py.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "ProjectMeta"):
            out = {}
            for kw in node.keywords:
                try:
                    out[kw.arg] = ast.literal_eval(kw.value)
                except ValueError:
                    pass
            return out
    return {}


def parse_routes(main_py: Path) -> list[Route]:
    """Project-specific routes, from decorators and docstrings."""
    if not main_py.exists():
        return []
    tree = ast.parse(main_py.read_text(encoding="utf-8"))
    routes = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        for dec in node.decorator_list:
            if (isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute)
                    and dec.func.attr in {"get", "post", "put", "delete"}
                    and dec.args and isinstance(dec.args[0], ast.Constant)):
                doc = (ast.get_docstring(node) or "").strip().split("\n")[0]
                routes.append(Route(dec.func.attr.upper(), dec.args[0].value, doc))
    return routes


def scan_components(root: Path, dirs: tuple[str, ...]) -> list[Component]:
    """Every source file with its line count and a one-line summary from its header."""
    out = []
    for d in dirs:
        base = root / d
        if not base.exists():
            continue
        # Sort on the POSIX string, not the Path: Windows compares paths
        # case-insensitively and Linux does not, which reordered these tables
        # between a local run and CI.
        for f in sorted(base.rglob("*"), key=lambda q: q.as_posix()):
            if (f.suffix not in SOURCE_EXT or "node_modules" in f.parts
                    or "__pycache__" in f.parts):
                continue
            text = f.read_text(encoding="utf-8")
            out.append(Component(
                path=f.relative_to(root).as_posix(),
                lines=text.count("\n") + (0 if text.endswith("\n") else 1),
                summary=_header_summary(f, text),
            ))
    return out


def _header_summary(path: Path, text: str) -> str:
    if path.suffix == ".py":
        try:
            doc = ast.get_docstring(ast.parse(text)) or ""
        except SyntaxError:
            doc = ""
    else:
        m = re.match(r"\s*/\*\*(.*?)\*/", text, re.S)
        doc = "\n".join(
            re.sub(r"^\s*\*\s?", "", line) for line in (m.group(1) if m else "").splitlines()
        )
    paras = [" ".join(p.split()) for p in re.split(r"\n\s*\n", doc.strip()) if p.strip()]
    if not paras:
        return ""
    # A title line like "ZoneMap -- project-01-specific." says little; prefer
    # the next paragraph when the first is that short.
    pick = paras[1] if len(paras[0]) < 45 and len(paras) > 1 else paras[0]
    # A paragraph that introduces a list ends mid-thought; keep its first sentence.
    if pick.endswith(":") and ". " in pick:
        pick = pick.split(". ")[0] + "."
    return pick if len(pick) <= 180 else pick[:177].rsplit(" ", 1)[0] + "…"


# --------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------

def fmt(value: Any, spec: str = "f4") -> str:
    if value is None or (isinstance(value, float) and value != value):
        return "—"
    v = float(value)
    if spec == "pct":
        return f"{v * 100:.1f}%"
    if spec == "pct3":
        return f"{v * 100:.3f}%"
    if spec == "pct_raw":
        return f"{v:.1f}%"
    if spec == "signed4":
        return f"{v:+.4f}".replace("-", "−")
    if spec == "int":
        return f"{int(round(v)):,}"
    if spec.startswith("f") and spec[1:].isdigit():
        return f"{v:.{int(spec[1:])}f}".replace("-", "−")
    return str(value)


def columns(p: Project) -> list[dict[str, str]]:
    cols = p.config.get("metric_columns")
    if cols:
        return cols
    return [{"key": p.metrics["primary_metric"], "label": p.metrics["primary_metric"], "fmt": "f4"}]


def ranked_models(p: Project) -> list[dict[str, Any]]:
    key = p.metrics["primary_metric"]
    rows = [m for m in p.metrics["models"] if key in m["metrics"]]
    return sorted(rows, key=lambda m: m["metrics"][key], reverse=p.metrics["higher_is_better"])


def is_baseline(m: dict[str, Any]) -> bool:
    return m.get("family", "").lower() == "baseline"


def primary(p: Project) -> dict[str, str]:
    cols = columns(p)
    key = p.metrics["primary_metric"]
    return next((c for c in cols if c["key"] == key), cols[0])


def score(p: Project, model: dict[str, Any], key: str | None = None) -> str:
    col = primary(p) if key is None else next(
        (c for c in columns(p) if c["key"] == key), {"key": key, "fmt": "f4"})
    return fmt(model["metrics"].get(col["key"]), col.get("fmt", "f4"))


def slugify_heading(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


# --------------------------------------------------------------------------
# Shared blocks
# --------------------------------------------------------------------------

def leaderboard_md(p: Project) -> str:
    cols = columns(p)
    entity = p.config.get("entity", "Model")
    best = p.metrics.get("best_model")
    head = f"| # | {entity} | " + " | ".join(c["label"] for c in cols) + " |"
    sep = "|---|---|" + "---|" * len(cols)
    out = [head, sep]
    for i, m in enumerate(ranked_models(p), 1):
        name = f"**{m['name']}**" if m["name"] == best else m["name"]
        if m["name"] == best:
            name += " 🏆"
        if is_baseline(m):
            name += " *(baseline)*"
        vals = " | ".join(fmt(m["metrics"].get(c["key"]), c.get("fmt", "f4")) for c in cols)
        out.append(f"| {i} | {name} | {vals} |")
    split = p.metrics.get("split", {})
    direction = "higher" if p.metrics["higher_is_better"] else "lower"
    out.append("")
    out.append(
        f"Ranked by {primary(p)['label'].lower()} ({direction} is better). "
        f"Split: {split.get('strategy', '—')}; {split.get('n_test', 0):,} evaluation rows. "
        f"Generated from `artifacts/metrics.json`."
    )
    return "\n".join(out)


def assertions_md(p: Project) -> str:
    rows = p.metrics.get("audit", {}).get("runtime_assertions", [])
    if not rows:
        return "_No runtime assertions were recorded for this run._"
    out = ["| Assertion | Result | What it measured |", "|---|---|---|"]
    for a in rows:
        mark = "✅ passed" if a["passed"] else "❌ failed"
        out.append(f"| `{a['name']}` | {mark} | {a['detail']} |")
    out.append("")
    out.append("Recorded by the assertion ledger while training ran; generated from "
               "`artifacts/metrics.json`.")
    return "\n".join(out)


def provenance_md(p: Project) -> str:
    v = p.provenance
    if not v:
        return "_No provenance recorded._"
    kind = "⚠️ **Synthetic data**" if v.get("synthetic") else "✅ **Real, third-party data**"
    rows = [
        ("Dataset", v.get("name", "—")),
        ("Source", v.get("source_url", "—")),
        ("Licence", v.get("license", "—")),
        ("Rows × columns (as downloaded)",
         f"{v.get('n_rows', 0):,} × {v.get('n_cols', 0):,}"),
        ("SHA-256", f"`{v.get('sha256', '')}`" if v.get("sha256") else "—"),
    ]
    for k, val in sorted((v.get("extra") or {}).items()):
        rows.append((k.replace("_", " ").capitalize(),
                     f"{val:,}" if isinstance(val, int) else str(val)))
    out = [kind, "", "| Field | Value |", "|---|---|"]
    out += [f"| {k} | {val} |" for k, val in rows]
    return "\n".join(out)


def findings_table(phase: dict[str, Any]) -> str:
    fs = phase.get("findings") or []
    if not fs:
        return ""
    out = ["| Finding | Value | Note |", "|---|---|---|"]
    out += [f"| {f['label']} | {f['value']} | {f.get('detail', '')} |" for f in fs]
    return "\n".join(out)


def audit_summary(p: Project) -> dict[str, int]:
    a = p.metrics.get("audit", {})
    return {
        "critical": int(a.get("critical", 0)),
        "warnings": int(a.get("warnings", 0)),
        "assertions_run": int(a.get("assertions_run", 0)),
        "assertions_passed": int(a.get("assertions_passed", 0)),
    }


def strongest_baseline(p: Project) -> dict[str, Any] | None:
    base = [m for m in ranked_models(p) if is_baseline(m)]
    return base[0] if base else None


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {i}" for i in items) if items else "_None recorded._"


# --------------------------------------------------------------------------
# Per-project documents
# --------------------------------------------------------------------------

def render_abstract(p: Project) -> str:
    m, prov, split = p.metrics, p.provenance, p.metrics.get("split", {})
    models = ranked_models(p)
    best = models[0]
    base = strongest_baseline(p)
    a = audit_summary(p)
    n_base = sum(1 for x in models if is_baseline(x))
    pcol = primary(p)

    s = [
        f"# Abstract — {p.title}",
        "",
        f"**{p.crispdm.get('business_question', '')}**",
        "",
        f"We study this question on *{prov.get('name', 'the dataset')}*: "
        f"{prov.get('n_rows', 0):,} rows as downloaded, used under "
        f"{prov.get('license', 'an unrecorded licence')}. Following the CRISP-DM "
        f"process, we compare {len(models)} approaches, {n_base} of them baselines, "
        f"on {split.get('n_test', 0):,} evaluation rows and rank them by "
        f"{pcol['label'].lower()}. Evaluation split: {split.get('strategy', 'held-out')}.",
        "",
    ]
    if is_baseline(best):
        s.append(
            f"The best-scoring approach is itself a baseline: **{best['name']}**, at "
            f"{score(p, best)}. No learned model improved on it."
        )
    elif base:
        s.append(
            f"The best approach, **{best['name']}**, reaches {score(p, best)} against "
            f"{score(p, base)} for the strongest baseline ({base['name']})."
        )
    else:
        s.append(f"The best approach, **{best['name']}**, reaches {score(p, best)}.")
    s += [
        "",
        f"{a['assertions_passed']} of {a['assertions_run']} leakage assertions passed "
        f"while training ran, and static analysis of the source reported "
        f"{a['critical']} critical and {a['warnings']} warning findings.",
        "",
        "**Keywords:** " + ", ".join([m.get("task", "")] + list(p.meta.get("techniques", []))),
        "",
        "---",
        "",
        "*Generated from this project's artifacts by `scripts/generate_docs.py`. "
        "Figures are not typed by hand.*",
        "",
    ]
    return "\n".join(s)


def render_paper(p: Project) -> str:
    models = ranked_models(p)
    best = models[0]
    env = p.metrics.get("env", {})
    refs = [p.provenance.get("citation", "")] + p.config.get("references", [])
    sections = [
        ("1. Introduction", "business_understanding"),
        ("2. Data", "data_understanding"),
        ("3. Method", "data_preparation"),
        ("4. Models", "modeling"),
        ("5. Results", "evaluation"),
        ("7. Deployment", "deployment"),
    ]
    out = [
        f"# {p.title}",
        "",
        f"*{p.meta.get('tagline', '')}*",
        "",
        f"**Aniket Anil Naik** · Project {p.number} of the data science full-stack portfolio",
        "",
        "## Abstract",
        "",
    ]
    # Reuse the abstract body without its heading or footer.
    abstract = render_abstract(p).split("\n")
    body = abstract[2:abstract.index("---")]
    out += body
    for title, phase_name in sections:
        ph = p.phase(phase_name)
        out += ["", f"## {title}", ""]
        if ph.get("objective"):
            out += [f"*Objective.* {ph['objective']}", ""]
        if ph.get("narrative"):
            out += [ph["narrative"].strip(), ""]
        table = findings_table(ph)
        if table:
            out += [table, ""]
        if phase_name == "data_understanding":
            out += ["### Provenance", "", provenance_md(p), ""]
        if phase_name == "modeling":
            out += [f"**Table 1.** Tournament results.", "", leaderboard_md(p), ""]
        if ph.get("gate_notes"):
            gate = "passed" if ph.get("gate_passed", True) else "failed"
            out += [f"> Phase gate {gate}: {ph['gate_notes']}", ""]
        if phase_name == "evaluation":
            out += [
                "## 6. Threats to validity",
                "",
                "### Assertions executed during training",
                "",
                assertions_md(p),
                "",
                "### Limitations",
                "",
                _bullets(p.config.get("risks", [])),
                "",
            ]
    out += [
        "## 8. Conclusion",
        "",
        _bullets(p.config.get("lessons", [])),
        "",
        "## Reproducibility",
        "",
        f"Trained {p.metrics.get('created_at', '—')} in {p.metrics.get('duration_seconds', '—')} s "
        f"at commit `{env.get('git_commit', '—')}` with Python {env.get('python', '—')}, "
        f"NumPy {env.get('numpy', '—')}, pandas {env.get('pandas', '—')} and "
        f"scikit-learn {env.get('sklearn', '—')}.",
        "",
        "```bash",
        *p.config.get("reproduce", []),
        "```",
        "",
        "## References",
        "",
        *[f"{i}. {r}" for i, r in enumerate([r for r in refs if r], 1)],
        "",
        "---",
        "",
        "*Generated from this project's artifacts by `scripts/generate_docs.py`.*",
        "",
    ]
    return "\n".join(out)


def render_article(p: Project) -> str:
    models = ranked_models(p)
    best = models[0]
    base = strongest_baseline(p)
    shots = [s for s in p.screenshots if not s.endswith("_dark.png")]
    out = [
        f"# {p.meta.get('tagline', p.title)}",
        "",
        f"*{p.title} — project {p.number} of a sixteen-project data science portfolio "
        f"built with an AI coding assistant, on real data.*",
        "",
    ]
    if shots:
        out += [f"![{p.title}](./screenshots/{shots[0]})", ""]
    out += [
        "## The question",
        "",
        p.crispdm.get("business_question", ""),
        "",
        p.phase("business_understanding").get("narrative", "").strip(),
        "",
        "## The data",
        "",
        p.phase("data_understanding").get("narrative", "").strip(),
        "",
        "## What won",
        "",
    ]
    if is_baseline(best):
        out.append(
            f"A baseline. **{best['name']}** scored best on "
            f"{primary(p)['label'].lower()} ({score(p, best)}), ahead of every "
            f"learned model in the tournament."
        )
    elif base:
        out.append(
            f"**{best['name']}**, at {score(p, best)} on {primary(p)['label'].lower()}, "
            f"against {score(p, base)} for the best baseline, {base['name']}."
        )
    out += [
        "",
        leaderboard_md(p),
        "",
        p.phase("evaluation").get("narrative", "").strip(),
        "",
        "## What I took away",
        "",
        _bullets(p.config.get("lessons", [])),
        "",
        "## How it is built",
        "",
        f"The backend is FastAPI on port {p.meta.get('api_port', '—')} and the front end "
        f"is React on port {p.meta.get('web_port', '—')}. Most of the interface — the "
        f"metrics panel, tournament table, CRISP-DM walkthrough, audit panel and charts — "
        f"is shared across the portfolio and driven entirely by the training run's "
        f"artifacts, so nothing on screen is typed by hand.",
        "",
        "Code: [github.com/aniket-alt/ai-coding-assistant]"
        f"(https://github.com/aniket-alt/ai-coding-assistant/tree/main/{p.slug})",
        "",
        "---",
        "",
        "*Figures in this article are generated from the project's artifacts.*",
        "",
    ]
    return "\n".join(out)


def render_design_doc(p: Project) -> str:
    env = p.metrics.get("env", {})
    a = audit_summary(p)
    out = [
        f"# Design document — {p.title}",
        "",
        f"| | |",
        f"|---|---|",
        f"| Project | {p.number} · `{p.slug}` |",
        f"| Domain | {p.meta.get('domain', '—')} |",
        f"| Task | {p.metrics.get('task', '—')} |",
        f"| Ports | API {p.meta.get('api_port', '—')} · web {p.meta.get('web_port', '—')} |",
        f"| Dataset | {p.meta.get('dataset', '—')} |",
        "",
        "## 1. Problem",
        "",
        p.crispdm.get("business_question", ""),
        "",
        p.phase("business_understanding").get("objective", ""),
        "",
        "### Goals",
        "",
        _bullets([ph["objective"] for ph in p.crispdm.get("phases", []) if ph.get("objective")]),
        "",
        "### Non-goals",
        "",
        _bullets(p.config.get("non_goals", [])),
        "",
        "## 2. Architecture",
        "",
        "```mermaid",
        "flowchart LR",
        f"  SRC[(\"{p.provenance.get('name', 'Dataset')}\")] --> DATA[ml/data.py]",
        "  DATA --> TRAIN[ml/train.py]",
        "  TRAIN --> ART[(artifacts/<br/>metrics · crispdm · audit · provenance · model)]",
        "  ART --> API[server/main.py<br/>FastAPI]",
        "  API --> UI[client<br/>React + @dsfs/ui]",
        "  LEAK[dsfs.leakage<br/>assertion ledger] -.-> TRAIN",
        "```",
        "",
        "Training is the only writer of results. The API serves the artifacts it wrote, "
        "and the client renders what the API returns, so a figure on screen always "
        "traces back to one run.",
        "",
        "## 3. Components",
        "",
        "| File | Lines | Responsibility |",
        "|---|---:|---|",
        *[f"| `{c.path}` | {c.lines} | {c.summary or '—'} |" for c in p.components],
        "",
        "Everything else — the FastAPI factory, dataset cache, provenance, CRISP-DM "
        "report, metrics record, leakage auditor and the six shared UI components — "
        "comes from `platform/`.",
        "",
        "## 4. API",
        "",
        "| Method | Path | Purpose |",
        "|---|---|---|",
        *[f"| {r.method} | `{r.path}` | {r.summary} |" for r in p.routes],
        *[f"| {m} | `{path}` | {s} (platform) |" for m, path, s in PLATFORM_ROUTES],
        "",
        "## 5. Data and evaluation",
        "",
        provenance_md(p),
        "",
        f"**Split.** {p.metrics.get('split', {}).get('strategy', '—')}. "
        f"{p.metrics.get('split', {}).get('notes', '')}",
        "",
        "## 6. Key decisions",
        "",
    ]
    for ph in p.crispdm.get("phases", []):
        if ph.get("gate_notes"):
            out.append(f"- **{PHASE_TITLES.get(ph['name'], ph['name'])}.** {ph['gate_notes']}")
    out += [
        "",
        "## 7. Verification",
        "",
        f"- {a['assertions_passed']} of {a['assertions_run']} runtime leakage assertions passed.",
        f"- Static source audit: {a['critical']} critical, {a['warnings']} warnings.",
        f"- {len(p.screenshots)} screenshots captured by `scripts/capture_screenshots.py`, "
        f"which fails if any view does not render its expected content.",
        "",
        assertions_md(p),
        "",
        "## 8. Risks",
        "",
        _bullets(p.config.get("risks", [])),
        "",
        "## 9. Reproducibility",
        "",
        f"Last trained at commit `{env.get('git_commit', '—')}` on Python "
        f"{env.get('python', '—')}.",
        "",
        "```bash",
        *p.config.get("reproduce", []),
        "```",
        "",
        "---",
        "",
        "*Generated by `scripts/generate_docs.py`.*",
        "",
    ]
    return "\n".join(out)


def render_audit_report(p: Project) -> str:
    a = audit_summary(p)
    verdict = "FAIL" if a["critical"] else "PASS"
    out = [
        f"# Audit report — {p.title}",
        "",
        f"**Result: {verdict}** · {a['critical']} critical · {a['warnings']} warnings · "
        f"{a['assertions_passed']}/{a['assertions_run']} runtime assertions passed",
        "",
        "## Runtime assertions",
        "",
        "These executed inside the training script and would have halted it on "
        "failure. Each entry is written by the assertion itself, so removing a "
        "check removes its evidence.",
        "",
        assertions_md(p),
        "",
        "## Static source analysis",
        "",
    ]
    if not p.audit:
        out.append("The AST pass over the project source found no findings.")
    else:
        out += ["| Severity | Rule | Location | Finding |", "|---|---|---|---|"]
        out += [f"| {f['severity']} | `{f['rule']}` | `{f['file']}:{f['line']}` | {f['message']} |"
                for f in p.audit]
    out += [
        "",
        "## Dataset provenance",
        "",
        provenance_md(p),
        "",
        "## Limitations of this audit",
        "",
        "- The static pass matches known leakage patterns; it cannot prove the absence "
        "of leakage it has no rule for.",
        "- Runtime assertions check what they were written to check: temporal order, "
        "disjoint splits, and features near-perfectly correlated with the target. "
        "Subtler leakage, such as a feature that is only partly derived from the "
        "target, can pass them.",
        "",
        "---",
        "",
        "*Generated from `artifacts/metrics.json` and `artifacts/audit.json`.*",
        "",
    ]
    return "\n".join(out)


def render_skill(p: Project) -> str:
    """An agent skill that reproduces and verifies this project."""
    name = f"reproduce-{p.slug.replace('_', '-')}"
    cmds = p.config.get("reproduce", [])
    a = audit_summary(p)
    out = [
        "---",
        f"name: {name}",
        f"description: Reproduce and verify project {p.number}, {p.title}, from a fresh "
        f"clone. Use when asked to rebuild, retrain or check this project.",
        "---",
        "",
        f"# Reproduce {p.title}",
        "",
        "## Steps",
        "",
        "From the repository root, install the kernel once: `pip install -e platform` "
        "and `npm install`. Then, inside the project directory:",
        "",
        *[f"{i}. `{c}`" for i, c in enumerate(cmds, 1)],
        "",
        "## Verify",
        "",
        f"- Training must finish with all runtime assertions passing (the last run "
        f"recorded {a['assertions_passed']} of {a['assertions_run']}).",
        f"- The winner and its {primary(p)['label'].lower()} should match "
        f"`artifacts/metrics.json`. Small differences are expected if the upstream "
        f"dataset changed; compare the SHA-256 in `artifacts/provenance.json` first.",
        f"- `python ../scripts/capture_screenshots.py {p.slug}` must report every view rendered.",
        "- `python ../scripts/generate_docs.py --check` must report no drift.",
        "",
        "## Do not",
        "",
        "- Hand-edit figures in README tables, `paper.md`, `abstract.md` or `article.md`; "
        "regenerate them.",
        "- Replace the dataset with generated data. If that ever becomes necessary, set "
        "`synthetic=True` on the Provenance so the UI discloses it.",
        "",
    ]
    return "\n".join(out)


def replace_block(text: str, name: str, content: str) -> str:
    """Fill a `<!-- dsfs:name -->` ... `<!-- /dsfs:name -->` region."""
    pattern = re.compile(
        rf"(<!-- dsfs:{name} -->)(.*?)(<!-- /dsfs:{name} -->)", re.S)
    if not pattern.search(text):
        return text
    return pattern.sub(lambda m: f"{m.group(1)}\n{content}\n{m.group(3)}", text)


def project_outputs(p: Project) -> dict[Path, str]:
    out = {
        p.root / "abstract.md": render_abstract(p),
        p.root / "paper.md": render_paper(p),
        p.root / "article.md": render_article(p),
        p.root / "DESIGN_DOC.md": render_design_doc(p),
        p.root / "audit_report.md": render_audit_report(p),
        p.root / "skills" / "reproduce" / "SKILL.md": render_skill(p),
    }
    readme = p.root / "README.md"
    if readme.exists():
        text = readme.read_text(encoding="utf-8")
        text = replace_block(text, "leaderboard", leaderboard_md(p))
        text = replace_block(text, "assertions", assertions_md(p))
        out[readme] = text
    return out


# --------------------------------------------------------------------------
# Portfolio-level documents
# --------------------------------------------------------------------------

def render_portfolio_audit(projects: list[Project], source: dict[str, Any] | None) -> str:
    out = [
        "# Portfolio audit report",
        "",
        "Every row below is read from a project's `artifacts/metrics.json`. Projects "
        "that have not been built yet do not appear.",
        "",
        "| Project | Dataset | Real data | Critical | Warnings | Runtime assertions |",
        "|---|---|:---:|---:|---:|---|",
    ]
    for p in projects:
        a = audit_summary(p)
        real = "no ⚠️" if p.provenance.get("synthetic") else "yes"
        out.append(
            f"| [{p.number} {p.title}](./{p.slug}/audit_report.md) | "
            f"{p.provenance.get('name', '—')} | {real} | {a['critical']} | {a['warnings']} | "
            f"{a['assertions_passed']}/{a['assertions_run']} passed |"
        )
    out += ["", "## Runtime assertions by project", ""]
    for p in projects:
        out += [f"### {p.number} · {p.title}", "", assertions_md(p), ""]

    out += ["## The same static auditor run on the source repository", ""]
    if not source:
        out.append("_Not recorded. Run `scripts/generate_docs.py --source-repo PATH`._")
    else:
        out += [
            f"For comparison, `dsfs.leakage.audit_source` was run over "
            f"`{source['label']}`. That repository's own `AUDIT_REPORT.md` states "
            f"\"Critical Data Leakages Detected: 0\" and gives each of its 13 "
            f"systems the same sub-scores.",
            "",
            f"**{source['critical']} critical · {source['warnings']} warnings**",
            "",
            "| Severity | Rule | Location | Finding |",
            "|---|---|---|---|",
            *[f"| {f['severity']} | `{f['rule']}` | `{f['file']}:{f['line']}` | {f['message']} |"
              for f in source["findings"]],
            "",
            "Static findings are leads, not verdicts: a `fit-on-test` match on a "
            "variable named `df` needs a human to confirm the frame really contains "
            "held-out rows. They are reported here as found.",
        ]
    out += [
        "",
        "## What these checks cannot catch",
        "",
        "- Leakage with no static signature, such as a feature partly derived from "
        "the target through a join.",
        "- Selection effects in the data itself, such as unlabelled fraud sitting in "
        "the legitimate class.",
        "- Overfitting to the test set through repeated manual iteration. Every "
        "project here was tuned while its test metrics were visible, which is a "
        "real if modest bias.",
        "",
        "---",
        "",
        "*Generated by `scripts/generate_docs.py`.*",
        "",
    ]
    return "\n".join(out)


def render_walkthrough(projects: list[Project], platform: list[Component]) -> str:
    out = [
        "# Code walkthrough",
        "",
        "A guided tour of the repository, generated from the source files themselves: "
        "each summary is the header comment of the file it describes.",
        "",
        "## The platform",
        "",
        "Shared by every project. A project imports it instead of re-implementing "
        "data sourcing, reporting, auditing or its interface.",
        "",
        "| File | Lines | What it does |",
        "|---|---:|---|",
        *[f"| [`{c.path}`](./{c.path}) | {c.lines} | {c.summary} |" for c in platform],
        "",
        "### How one training run flows through it",
        "",
        "```mermaid",
        "sequenceDiagram",
        "  participant T as ml/train.py",
        "  participant D as dsfs.datasets",
        "  participant L as dsfs.leakage",
        "  participant R as RunRecord / CrispDmReport",
        "  participant A as artifacts/",
        "  participant S as server (dsfs.app)",
        "  participant U as client (@dsfs/ui)",
        "  T->>D: fetch(Provenance)",
        "  D-->>T: cached file + SHA-256",
        "  T->>L: assert_* during split and features",
        "  L-->>L: ledger records what was measured",
        "  T->>R: log models, phases, audit",
        "  R->>A: metrics.json, crispdm.json, audit.json",
        "  S->>A: read on request",
        "  U->>S: /api/metrics, /api/crispdm, ...",
        "```",
        "",
    ]
    for p in projects:
        best = ranked_models(p)[0]
        out += [
            f"## {p.number} · {p.title}",
            "",
            f"{p.meta.get('tagline', '')} Best: **{best['name']}** "
            f"({primary(p)['label'].lower()} {score(p, best)}). "
            f"[README](./{p.slug}/README.md) · [paper](./{p.slug}/paper.md) · "
            f"[design doc](./{p.slug}/DESIGN_DOC.md)",
            "",
            "| File | Lines | What it does |",
            "|---|---:|---|",
            *[f"| [`{c.path}`](./{p.slug}/{c.path}) | {c.lines} | {c.summary} |"
              for c in p.components],
            "",
            "| Method | Endpoint | Purpose |",
            "|---|---|---|",
            *[f"| {r.method} | `{r.path}` | {r.summary} |" for r in p.routes],
            "",
        ]
        light = [s for s in p.screenshots if not s.endswith("_dark.png")]
        if light:
            out += [f"![{p.title}](./{p.slug}/screenshots/{light[0]})", ""]
    out += ["---", "", "*Generated by `scripts/generate_docs.py`.*", ""]
    return "\n".join(out)


def locate(root: Path, rel_file: str, symbol: str) -> str:
    """`path:line` of a symbol's definition, or of its first mention."""
    f = root / rel_file
    if not f.exists():
        return rel_file
    lines = f.read_text(encoding="utf-8").splitlines()
    pat_def = re.compile(rf"^\s*(def|class|function|export function|const)\s+{re.escape(symbol)}\b")
    for i, line in enumerate(lines, 1):
        if pat_def.search(line):
            return f"{rel_file}#L{i}"
    for i, line in enumerate(lines, 1):
        if symbol in line:
            return f"{rel_file}#L{i}"
    return rel_file


def render_skills(repo: Path, catalogue: list[dict[str, Any]], built: set[str]) -> str:
    out = [
        "# Data science skills",
        "",
        "Each skill links to the line where it is implemented. Links are located "
        "by searching the source for the named symbol, so they do not go stale when "
        "code moves.",
        "",
        "| Skill | Where | Why it matters here |",
        "|---|---|---|",
    ]
    for s in catalogue:
        proj = s["file"].split("/")[0]
        if proj[:2].isdigit() and proj not in built:
            continue
        loc = locate(repo, s["file"], s["symbol"])
        out.append(f"| **{s['skill']}** | [`{loc.split('#')[0]}`](./{loc}) | {s['why']} |")
    out += [
        "",
        "## Reproduction skills",
        "",
        "Each built project ships an agent skill at `skills/reproduce/SKILL.md` "
        "describing how to rebuild and verify it from a fresh clone:",
        "",
        *[f"- [`{b}/skills/reproduce/SKILL.md`](./{b}/skills/reproduce/SKILL.md)"
          for b in sorted(built)],
        "",
        "---",
        "",
        "*Generated by `scripts/generate_docs.py` from `docs/skills.json`.*",
        "",
    ]
    return "\n".join(out)


def render_status_table(repo: Path, plans: list[dict[str, Any]]) -> str:
    out = ["| # | Project | Status | Dataset (planned or used) |", "|---|---|---|---|"]
    for pl in plans:
        slug = pl["slug"]
        built = (repo / slug / "artifacts" / "metrics.json").exists()
        name = f"[{pl['title']}](./{slug})" if built else pl["title"]
        out.append(f"| {pl['number']} | {name} | {'✅ built' if built else '⬜ planned'} | "
                   f"{pl['dataset']} |")
    return "\n".join(out)


def normalise(text: str) -> str:
    return text.replace("\r\n", "\n")


def write_outputs(outputs: dict[Path, str], check: bool) -> list[Path]:
    """Write files, or in check mode return those that would change."""
    drifted = []
    for path, content in sorted(outputs.items()):
        content = normalise(content)
        current = normalise(path.read_text(encoding="utf-8")) if path.exists() else None
        if current == content:
            continue
        drifted.append(path)
        if not check:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8", newline="\n")
    return drifted
