"""
Project 11 -- benchmark the portfolio's leakage auditor, then audit the portfolio.

    python ml/train.py

Plants known leaks in copies of a real training script, runs both detectors
against every copy, and scores them like any classifier -- including how much
each missed leak would have inflated a published result.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "06_anomaly_detection" / "ml"))   # shares project 06's loader

from dsfs import CrispDmReport, ModelScore, RunRecord, SplitInfo, leakage  # noqa: E402
import data as fraud_data  # noqa: E402

from mutations import VARIANTS, columns_for  # noqa: E402

ARTIFACTS = ROOT / "artifacts"
CORPUS = ARTIFACTS / "leak_corpus"


class LiveChecks:
    """The real runtime assertions, kept out of this project's own audit ledger."""

    def __getattr__(self, name):
        fn = getattr(leakage, name)

        def call(*args, **kwargs):
            before = len(leakage._LEDGER)
            try:
                return fn(*args, **kwargs)
            finally:
                del leakage._LEDGER[before:]
        return call


class NoChecks:
    """Assertions switched off, to measure what a leak does to the score."""

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def execute(source: str, frame, checks) -> tuple[float | None, str | None]:
    """Run a variant. Returns (average precision, name of the check that fired)."""
    namespace: dict = {}
    exec(compile(source, "<variant>", "exec"), namespace)
    try:
        return float(namespace["run"](frame.copy(), checks)), None
    except leakage.LeakageError as exc:
        return None, str(exc).split(":")[0]


def score_detector(flags: dict[str, bool], variants, inflation: dict[str, float]) -> dict[str, float]:
    leaks = [v.id for v in variants if v.kind == "leak"]
    controls = [v.id for v in variants if v.kind == "control"]
    tp = sum(flags[i] for i in leaks)
    fp = sum(flags[i] for i in controls)
    tpr = tp / len(leaks)
    tnr = 1 - fp / len(controls)
    precision = tp / (tp + fp) if tp + fp else 0.0
    total_damage = sum(max(inflation[i], 0) for i in leaks)
    caught_damage = sum(max(inflation[i], 0) for i in leaks if flags[i])
    return {
        # Balanced accuracy, not F1: the corpus holds more leaks than controls,
        # so F1 would reward a detector that simply flags everything.
        "balanced_accuracy": (tpr + tnr) / 2,
        "recall_pct": tpr * 100,
        "false_positive_rate_pct": (1 - tnr) * 100,
        "precision_pct": precision * 100,
        "damage_caught_pct": caught_damage / total_damage * 100 if total_damage else 0.0,
        "leaks_caught": float(tp),
    }


def portfolio_snapshot() -> list[dict]:
    """Every other built project's audit state, read from its artifacts."""
    rows = []
    for proj in sorted((p for p in REPO.iterdir() if p.is_dir() and p.name[:2].isdigit()),
                       key=lambda p: p.name):
        path = proj / "artifacts" / "metrics.json"
        if proj == ROOT or not path.exists():
            continue
        m = json.loads(path.read_text(encoding="utf-8"))
        a = m.get("audit", {})
        rows.append({
            "project": proj.name,
            "title": m["project_title"],
            "task": m["task"],
            "dataset": m["provenance"].get("name", ""),
            "rows": m["provenance"].get("n_rows", 0),
            "real_data": not m["provenance"].get("synthetic", False),
            "models": len(m["models"]),
            "best_model": m["best_model"],
            "assertions_passed": a.get("assertions_passed", 0),
            "assertions_run": a.get("assertions_run", 0),
            "critical": a.get("critical", 0),
            "warnings": a.get("warnings", 0),
            "assertions": [x["name"] for x in a.get("runtime_assertions", [])],
            "trained_at": m["created_at"],
        })
    return rows


def main() -> None:
    t0 = time.perf_counter()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    CORPUS.mkdir(parents=True, exist_ok=True)
    leakage.reset_ledger()

    report = CrispDmReport(
        project_slug="11_portfolio_audit",
        project_title="Portfolio Audit",
        business_question=(
            "How much should anyone trust this portfolio's leakage audit, and "
            "which leaks would it miss?"
        ),
    )
    leaks = [v for v in VARIANTS if v.kind == "leak"]
    controls = [v for v in VARIANTS if v.kind == "control"]
    report.phase(
        "business_understanding",
        objective="Measure the auditor instead of trusting it.",
        narrative=(
            "Every project in this portfolio reports a passing leakage audit. "
            "That claim is only as good as the auditor, and an auditor that has "
            "never been tested against known leaks proves nothing by passing.\n\n"
            f"So this project builds a benchmark: {len(leaks)} copies of a real "
            f"training script, each with one deliberately planted leak, and "
            f"{len(controls)} clean controls. Both detectors — the static code "
            "audit and the runtime assertions — are run against every copy and "
            "scored like any classifier.\n\n"
            "Each copy is also run with the assertions switched off, to measure "
            "how much its leak inflates the reported score. A missed leak that "
            "changes nothing is a curiosity; a missed leak that doubles a "
            "headline number is a serious gap."
        ),
        code_refs=["ml/mutations.py:VARIANTS", "ml/train.py:score_detector"],
        gate_notes="Leak catalogue, controls and scoring fixed before any detector was run.",
    ).add("Planted leaks", len(leaks)) \
     .add("Clean controls", len(controls)) \
     .add("Primary metric", "Balanced accuracy")

    print("[1/6] loading the ULB fraud data (shared with project 06) ...")
    frame, prov = fraud_data.load()
    prov = prov.stamped(extra={
        "benchmark": "leaks are planted deliberately by this project; the data itself is real",
        "variants": len(VARIANTS),
    })
    prov.save(ARTIFACTS / "provenance.json")
    base_columns = [c for c in frame.columns if c != "Class"]

    categories = sorted({v.category for v in leaks})
    report.phase(
        "data_understanding",
        objective="Build a catalogue of leaks that covers the ways real projects go wrong.",
        narrative=(
            "The leaks fall into four families, each a mistake that appears in "
            "real work:\n\n"
            "| Variant | Family | What goes wrong |\n|---|---|---|\n"
            + "\n".join(f"| {v.title} | {v.category} | {v.explanation} |" for v in VARIANTS)
            + "\n\nThe controls matter as much as the leaks. A detector that fires "
            "on a legitimate unshuffled split or a scaler inside a pipeline would "
            "bury real warnings in false alarms.\n\n"
            f"Every variant runs on the real ULB credit card data "
            f"({prov.n_rows:,} transactions). The leaks are constructed; the data is not."
        ),
        code_refs=["ml/mutations.py:TEMPLATE"],
        gate_notes=f"{len(VARIANTS)} variants across {len(categories)} leak families plus controls.",
    ).add("Variants", len(VARIANTS)).add("Leak families", len(categories)) \
     .add("Transactions per run", prov.n_rows)

    print("[2/6] writing the corpus and running the static audit ...")
    sources = {}
    for v in VARIANTS:
        src = v.source(columns_for(v, base_columns))
        sources[v.id] = src
        (CORPUS / f"{v.id}.py").write_text(src, encoding="utf-8", newline="\n")
    template_src = sources["control_template"]
    identical = [v.id for v in leaks if sources[v.id] == template_src]
    leakage.require(
        "assert_leaks_differ_from_template", not identical,
        f"{len(leaks)} leak variants compared with the clean template; {len(identical)} identical.",
        leak_variants=len(leaks), identical=identical,
    )

    # The whole corpus is time-ordered transaction data, so every file is
    # audited as a temporal module.
    findings = leakage.audit_source(CORPUS, temporal_modules=("",))
    static_flags = {v.id: any(f.severity == "CRITICAL" and Path(f.file).stem == v.id for f in findings)
                    for v in VARIANTS}
    static_rules = {v.id: sorted({f.rule for f in findings
                                  if f.severity == "CRITICAL" and Path(f.file).stem == v.id})
                    for v in VARIANTS}

    print("[3/6] executing every variant with and without runtime assertions ...")
    runtime_flags, fired, scores, errors = {}, {}, {}, []
    for v in VARIANTS:
        t = time.perf_counter()
        try:
            _, check = execute(sources[v.id], frame, LiveChecks())
            ap, _ = execute(sources[v.id], frame, NoChecks())
        except Exception as exc:                       # a variant that cannot run is a bug
            errors.append(f"{v.id}: {type(exc).__name__}: {exc}")
            continue
        runtime_flags[v.id] = check is not None
        fired[v.id] = check
        scores[v.id] = ap
        print(f"      {v.id:<30} AP {ap:.4f}  static {'FLAG' if static_flags[v.id] else '-':<4}  "
              f"runtime {check or '-'}  ({time.perf_counter() - t:.0f}s)")

    leakage.require(
        "assert_all_variants_executed", not errors,
        f"{len(VARIANTS)} variants executed; {len(errors)} failed for reasons other than a leakage check.",
        variants=len(VARIANTS), failures=errors,
    )
    clean_ap = scores["control_template"]
    drift = {v.id: abs(scores[v.id] - clean_ap) for v in controls}
    leakage.require(
        "assert_controls_are_clean", max(drift.values()) < 0.02 and not any(runtime_flags[c.id] for c in controls),
        f"{len(controls)} controls scored within {max(drift.values()):.4f} average precision of the template "
        f"and raised {sum(runtime_flags[c.id] for c in controls)} runtime checks.",
        max_ap_difference=round(max(drift.values()), 6),
        runtime_flags_on_controls=sum(runtime_flags[c.id] for c in controls),
    )

    inflation = {v.id: scores[v.id] - clean_ap for v in VARIANTS}
    report.phase(
        "data_preparation",
        objective="Establish a clean reference score and confirm the controls really are clean.",
        narrative=(
            f"The clean template scores an average precision of **{clean_ap:.4f}** on "
            f"its held-out window. Each control stays within "
            f"{max(drift.values()):.4f} of it and trips no runtime check, which is "
            f"the evidence that the controls are genuinely leak-free rather than "
            f"assumed to be.\n\n"
            "Every variant is executed twice. With the assertions live, a leak is "
            "detected if one of them raises. With the assertions disabled, the "
            "variant runs to completion and its score is recorded, so the "
            "inflation each leak causes can be measured against the template.\n\n"
            "The variants' own assertions are kept out of this project's audit "
            "ledger, so the benchmark cannot be mistaken for evidence about this "
            "project."
        ),
        code_refs=["ml/train.py:execute", "ml/train.py:LiveChecks"],
        gate_notes="Controls verified clean by score and by assertion.",
    ).add("Clean average precision", f"{clean_ap:.4f}") \
     .add("Largest control drift", f"{max(drift.values()):.4f}")

    print("[4/6] scoring the detectors ...")
    run = RunRecord(project_slug="11_portfolio_audit", project_title="Portfolio Audit",
                    task="leakage detector benchmark", primary_metric="balanced_accuracy",
                    higher_is_better=True)
    run.set_provenance(prov)
    run.set_split(SplitInfo(strategy="planted-leak benchmark", n_train=0, n_test=len(VARIANTS),
                            notes=f"{len(leaks)} leaks and {len(controls)} clean controls."))

    detectors = {
        "Flag everything": ("baseline", {v.id: True for v in VARIANTS}),
        "Flag nothing": ("baseline", {v.id: False for v in VARIANTS}),
        "Static code audit": ("static", static_flags),
        "Runtime assertions": ("runtime", runtime_flags),
        "Static + runtime": ("combined", {i: static_flags[i] or runtime_flags[i] for i in static_flags}),
    }
    for name, (family, flags) in detectors.items():
        run.add_model(ModelScore(name=name, family=family, metrics=score_detector(flags, VARIANTS, inflation)))
        m = run.models[-1].metrics
        print(f"      {name:<22} balanced acc {m['balanced_accuracy']:.3f}  recall {m['recall_pct']:5.1f}%  "
              f"FPR {m['false_positive_rate_pct']:5.1f}%  damage caught {m['damage_caught_pct']:5.1f}%")
    best = run.best
    combined = run.models[-1].metrics

    missed = sorted((v for v in leaks if not (static_flags[v.id] or runtime_flags[v.id])),
                    key=lambda v: -inflation[v.id])
    report.phase(
        "modeling",
        objective="Score each detector, and both together, against two baselines.",
        narrative=(
            "Flagging everything finds every leak and condemns every clean script; "
            "flagging nothing does the reverse. Both score exactly 0.5 balanced "
            "accuracy, which is the floor a real detector has to clear.\n\n"
            "The two detectors look for different things. The static audit reads "
            "code for patterns such as a fit on a variable named like test data. "
            "The runtime assertions inspect what the code actually produced: "
            "whether the splits overlap, whether time runs forward, whether any "
            "feature is nearly the label."
        ),
        code_refs=["ml/train.py:score_detector"],
        gate_notes=f"{len(detectors)} detectors scored.",
    ).add("Best detector", best.name) \
     .add("Leaks caught together", f"{int(combined['leaks_caught'])} of {len(leaks)}") \
     .add("Damage caught together", f"{combined['damage_caught_pct']:.0f}%")

    print("[5/6] audit and portfolio snapshot ...")
    own = leakage.audit_source(ROOT, temporal_modules=("train",), skip=("leak_corpus", "__pycache__", "node_modules", "dist"))
    leakage.save_report(own, ARTIFACTS / "audit.md", "Portfolio Audit")
    crit = [f for f in own if f.severity == "CRITICAL"]
    executed = leakage.ledger()
    run.audit = {"critical": len(crit), "warnings": sum(1 for f in own if f.severity == "WARNING"),
                 "total": len(own), "runtime_assertions": executed,
                 "assertions_passed": sum(1 for a in executed if a["passed"]),
                 "assertions_run": len(executed)}

    worst = missed[0] if missed else None
    report.phase(
        "evaluation",
        objective="State plainly what the audit can and cannot see.",
        narrative=(
            f"Together the two detectors catch **{int(combined['leaks_caught'])} of "
            f"{len(leaks)} planted leaks** with {combined['false_positive_rate_pct']:.0f}% "
            f"false alarms on clean scripts. Weighted by how much each leak inflates "
            f"the score, they catch {combined['damage_caught_pct']:.0f}% of the damage.\n\n"
            + (f"**What they miss matters.** The costliest undetected leak is "
               f"*{worst.title.lower()}*, which moves average precision from "
               f"{clean_ap:.4f} to {scores[worst.id]:.4f}. It passes because no "
               f"feature is perfectly correlated with the label and the split itself "
               f"is clean — the leak lives in how a feature was computed, which "
               f"neither detector inspects.\n\n" if worst else "")
            + "The consequence for the rest of this portfolio is concrete: a passing "
            "audit rules out split mistakes and blatant label leaks, and says "
            "little about target encodings, label-derived features or tuning on "
            "the test set. Those are guarded in each project by design — "
            "out-of-fold encoders, walk-forward refits, baselines — not by the "
            "auditor, and should be read that way."
        ),
        code_refs=["ml/mutations.py", "platform/dsfs/leakage.py"],
        gate_passed=len(crit) == 0,
        gate_notes="The auditor's blind spots are published with its results.",
    ).add("Leaks missed", len(missed)) \
     .add("Costliest miss", worst.title if worst else "—") \
     .add("False alarms on controls", f"{combined['false_positive_rate_pct']:.0f}%")

    report.phase(
        "deployment",
        objective="Publish the benchmark beside the portfolio's audit results.",
        narrative=(
            "`GET /api/benchmark` returns every variant with both detectors' "
            "verdicts and its score inflation, `GET /api/corpus/{id}` returns a "
            "variant's source so any reader can check what was planted, and "
            "`GET /api/portfolio` returns every other project's audit state."
        ),
        code_refs=["server/main.py"],
        gate_notes="Corpus committed so the benchmark is inspectable.",
    ).add("Serving port", 8011)

    run.extra = {
        "clean_average_precision": clean_ap,
        "variants": [{
            "id": v.id, "title": v.title, "kind": v.kind, "category": v.category,
            "explanation": v.explanation, "average_precision": round(scores[v.id], 4),
            "inflation": round(inflation[v.id], 4),
            "static_detected": static_flags[v.id], "static_rules": static_rules[v.id],
            "runtime_detected": runtime_flags[v.id], "runtime_check": fired[v.id],
        } for v in VARIANTS],
        "portfolio": portfolio_snapshot(),
    }
    run.duration_seconds = round(time.perf_counter() - t0, 1)
    run.save(ARTIFACTS / "metrics.json")
    report.save(ARTIFACTS / "crispdm.json")
    (ARTIFACTS / "crispdm.md").write_text(report.to_markdown(), encoding="utf-8")

    print("[6/6] done.\n")
    print(run.leaderboard_markdown(("recall_pct", "false_positive_rate_pct", "damage_caught_pct")))
    print("missed:", [(v.id, round(inflation[v.id], 3)) for v in missed])
    print(f"critical: {len(crit)}  runtime: {run.duration_seconds}s")


if __name__ == "__main__":
    main()
