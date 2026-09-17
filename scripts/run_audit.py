"""
Static leakage audit across every project in the repository.

    python scripts/run_audit.py

Exits non-zero if any project has a CRITICAL finding, so CI fails on it.
Runtime assertions are not re-run here: they execute during training, which
needs the full datasets.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "platform"))

from dsfs import leakage  # noqa: E402


def main() -> int:
    projects = sorted((p for p in REPO.iterdir() if p.is_dir() and p.name[:2].isdigit()),
                      key=lambda q: q.name)
    total_critical = 0
    for proj in projects:
        # leak_corpus holds project 11's deliberately leaky benchmark scripts;
        # auditing them here would fail CI on leaks planted on purpose.
        findings = leakage.audit_source(
            proj, temporal_modules=("train", "data", "features"),
            skip=("node_modules", ".venv", "__pycache__", "dist", "leak_corpus"))
        crit = [f for f in findings if f.severity == "CRITICAL"]
        warn = [f for f in findings if f.severity == "WARNING"]
        total_critical += len(crit)
        status = "FAIL" if crit else "ok"
        print(f"{status:<4} {proj.name:<28} {len(crit)} critical, {len(warn)} warnings")
        for f in crit + warn:
            print(f"       {f.severity:<8} {f.file}:{f.line}  [{f.rule}] {f.message}")
    print(f"\n{len(projects)} projects audited, {total_critical} critical findings")
    return 1 if total_critical else 0


if __name__ == "__main__":
    raise SystemExit(main())
