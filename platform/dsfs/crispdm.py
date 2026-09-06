"""
CRISP-DM as a data structure, not a section heading.

Each project builds a `CrispDmReport` while it runs. The API serves it, the
frontend renders it phase-by-phase, and `dsfs.artifacts` turns it into
paper.md / abstract.md / article.md. Because the report is populated by the
training code itself, the documentation cannot drift from what actually ran.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Literal

PhaseName = Literal[
    "business_understanding",
    "data_understanding",
    "data_preparation",
    "modeling",
    "evaluation",
    "deployment",
]

PHASE_ORDER: tuple[PhaseName, ...] = (
    "business_understanding",
    "data_understanding",
    "data_preparation",
    "modeling",
    "evaluation",
    "deployment",
)

PHASE_TITLES: dict[str, str] = {
    "business_understanding": "1. Business Understanding",
    "data_understanding": "2. Data Understanding",
    "data_preparation": "3. Data Preparation",
    "modeling": "4. Modeling",
    "evaluation": "5. Evaluation",
    "deployment": "6. Deployment",
}


@dataclass
class Finding:
    """One concrete, quotable result inside a phase."""
    label: str
    value: str
    detail: str = ""


@dataclass
class Phase:
    name: PhaseName
    title: str = ""
    objective: str = ""
    narrative: str = ""                       # markdown, may contain $KaTeX$
    findings: list[Finding] = field(default_factory=list)
    figures: list[str] = field(default_factory=list)   # relative image paths
    code_refs: list[str] = field(default_factory=list) # "ml/train.py:42"
    gate_passed: bool = True                  # phase-exit review outcome
    gate_notes: str = ""

    def __post_init__(self) -> None:
        if not self.title:
            self.title = PHASE_TITLES.get(self.name, self.name)

    def add(self, label: str, value: Any, detail: str = "") -> "Phase":
        self.findings.append(Finding(label, _fmt(value), detail))
        return self


@dataclass
class CrispDmReport:
    project_slug: str
    project_title: str
    business_question: str = ""
    phases: dict[str, Phase] = field(default_factory=dict)

    def phase(self, name: PhaseName, **kwargs: Any) -> Phase:
        """Get-or-create a phase, updating any provided fields."""
        ph = self.phases.get(name) or Phase(name=name)
        for k, v in kwargs.items():
            if v is not None and hasattr(ph, k):
                setattr(ph, k, v)
        self.phases[name] = ph
        return ph

    def ordered(self) -> list[Phase]:
        return [self.phases[p] for p in PHASE_ORDER if p in self.phases]

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_slug": self.project_slug,
            "project_title": self.project_title,
            "business_question": self.business_question,
            "phases": [asdict(p) for p in self.ordered()],
        }

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return p

    def to_markdown(self) -> str:
        out = [f"# CRISP-DM Report — {self.project_title}\n"]
        if self.business_question:
            out.append(f"**Business question.** {self.business_question}\n")
        for ph in self.ordered():
            out.append(f"\n## {ph.title}\n")
            if ph.objective:
                out.append(f"*Objective:* {ph.objective}\n")
            if ph.narrative:
                out.append(f"\n{ph.narrative}\n")
            if ph.findings:
                out.append("\n| Finding | Value | Detail |\n|---|---|---|")
                for f in ph.findings:
                    out.append(f"| {f.label} | `{f.value}` | {f.detail} |")
                out.append("")
            if ph.code_refs:
                out.append("\n*Code:* " + ", ".join(f"`{c}`" for c in ph.code_refs) + "\n")
            gate = "PASSED" if ph.gate_passed else "FAILED"
            out.append(f"\n> **Phase gate: {gate}.** {ph.gate_notes}\n")
        return "\n".join(out)


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:,.4f}".rstrip("0").rstrip(".") if abs(v) < 1e6 else f"{v:.4g}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)
