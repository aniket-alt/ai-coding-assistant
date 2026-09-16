"""
Generate every derived document in the repository from run artifacts.

    python scripts/generate_docs.py                    # write
    python scripts/generate_docs.py --check            # fail if anything drifted
    python scripts/generate_docs.py --source-repo PATH # also re-audit the source portfolio

Per built project: abstract.md, paper.md, article.md, DESIGN_DOC.md,
audit_report.md, skills/reproduce/SKILL.md, and the marked result blocks in
README.md. At the root: AUDIT_REPORT.md, WALKTHROUGH.md, SKILLS.md, and the
status table in IMPLEMENTATION_PLANS.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "platform"))

from dsfs import docs, leakage  # noqa: E402

SOURCE_AUDIT = REPO / "docs" / "source_repo_audit.json"


def discover() -> list[docs.Project]:
    roots = sorted(
        (p for p in REPO.iterdir()
        if p.is_dir() and p.name[:2].isdigit() and (p / "artifacts" / "metrics.json").exists()),
        key=lambda q: q.name,
    )
    return [docs.load_project(r) for r in roots]


def source_audit(path: str | None) -> dict | None:
    """Re-audit the source portfolio if given, otherwise use the recorded result."""
    if path:
        root = Path(path)
        findings = leakage.audit_source(root, temporal_modules=("timeseries", "spy", "forecast"))
        record = {
            "label": "dlmastery/data_science_examples",
            "critical": sum(1 for f in findings if f.severity == "CRITICAL"),
            "warnings": sum(1 for f in findings if f.severity == "WARNING"),
            "findings": [
                {**f.to_dict(), "file": Path(f.file).as_posix()} for f in findings
            ],
        }
        SOURCE_AUDIT.parent.mkdir(parents=True, exist_ok=True)
        SOURCE_AUDIT.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8",
                                newline="\n")
        return record
    if SOURCE_AUDIT.exists():
        return json.loads(SOURCE_AUDIT.read_text(encoding="utf-8"))
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if any generated document is out of date")
    ap.add_argument("--source-repo", help="path to a checkout of the source portfolio")
    args = ap.parse_args()

    projects = discover()
    if not projects:
        print("no trained projects found")
        return 1

    outputs: dict[Path, str] = {}
    for p in projects:
        outputs.update(docs.project_outputs(p))

    platform = docs.scan_components(REPO, ("platform/dsfs", "platform/web/src"))
    outputs[REPO / "AUDIT_REPORT.md"] = docs.render_portfolio_audit(
        projects, source_audit(args.source_repo))
    outputs[REPO / "WALKTHROUGH.md"] = docs.render_walkthrough(projects, platform)

    catalogue = json.loads((REPO / "docs" / "skills.json").read_text(encoding="utf-8"))
    outputs[REPO / "SKILLS.md"] = docs.render_skills(
        REPO, catalogue, {p.slug for p in projects})

    plans_md = REPO / "IMPLEMENTATION_PLANS.md"
    if plans_md.exists():
        plans = json.loads((REPO / "docs" / "plans.json").read_text(encoding="utf-8"))
        outputs[plans_md] = docs.replace_block(
            plans_md.read_text(encoding="utf-8"), "status",
            docs.render_status_table(REPO, plans))

    drifted = docs.write_outputs(outputs, check=args.check)
    rel = [str(p.relative_to(REPO)).replace("\\", "/") for p in drifted]

    if args.check:
        if drifted:
            print("generated documents are out of date:")
            for r in rel:
                print(f"  {r}")
            print("\nrun: python scripts/generate_docs.py")
            return 1
        print(f"all {len(outputs)} generated documents are current")
        return 0

    print(f"{len(projects)} projects · {len(outputs)} documents · {len(drifted)} written")
    for r in rel:
        print(f"  {r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
