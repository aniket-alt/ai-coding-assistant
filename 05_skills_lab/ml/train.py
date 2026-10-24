"""
Project 05 -- run the skills catalogue and record what it measured.

    python ml/train.py

Every skill runs its careless and its careful version over the same seeds on a
real dataset, and the paired difference gets a bootstrap interval. A skill may
only claim an effect when that interval excludes zero; an assertion checks that
the catalogue's wording matches its own numbers before anything is published.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from dsfs import CrispDmReport, ModelScore, RunRecord, SplitInfo, leakage

from data import SOURCES, columns, load
from skills import CATALOGUE

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
SEEDS = list(range(25))


def flattery(record: dict) -> float:
    """How much better the careless version looks, in the metric's own units."""
    gap = record["gap"]["mean"]
    return gap if record["higher_is_better"] else -gap


def main() -> None:
    t0 = time.perf_counter()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    leakage.reset_ledger()
    report = CrispDmReport(
        project_slug="05_skills_lab", project_title="Data Science Skills Lab",
        business_question=("For each technique in a working data scientist's catalogue, does doing it the "
                           "careless way change the result -- and by how much, on real data?"),
    )
    report.phase(
        "business_understanding",
        objective="Turn a catalogue of techniques into a catalogue of measured differences.",
        narrative=(
            "A skills catalogue usually lists techniques with advice about pitfalls. Advice is cheap: it cannot "
            "say whether a pitfall costs anything on the data in front of you. Each entry here runs the careless "
            "version and the careful version of the same analysis, over many seeds, and reports the paired "
            "difference with an interval. Entries whose interval covers zero say that plainly, which is how the "
            "catalogue stays worth reading."
        ),
        code_refs=["ml/skills.py:Skill.measure"],
        gate_notes="Every entry must execute on a real dataset and report a paired interval.",
    ).add("Skills", len(CATALOGUE)).add("Seeds per skill", len(SEEDS))

    print("[1/5] loading datasets ...")
    frames, provenances = {}, {}
    for name in SOURCES:
        frames[name], provenances[name] = load(name)
    main_prov = provenances["titanic"]
    main_prov.save(ARTIFACTS / "provenance.json")
    synthetic = [n for n, p in provenances.items() if p.synthetic]
    leakage.require(
        "assert_every_dataset_is_real", not synthetic,
        f"{len(provenances)} datasets downloaded and checksummed: "
        + "; ".join(f"{n} {p.n_rows:,} rows, sha256 {p.sha256[:8]}" for n, p in provenances.items())
        + f". Marked synthetic: {synthetic or 'none'}.",
        datasets=len(provenances), synthetic=len(synthetic),
    )
    facts = {n: {"rows": int(len(f)), "numeric": len(columns(f)[0]), "categorical": len(columns(f)[1]),
                 "positive_rate": float(f["target"].mean()),
                 "rows_with_missing": float(f.isna().any(axis=1).mean()),
                 "name": provenances[n].name, "source_url": provenances[n].source_url,
                 "license": provenances[n].license, "sha256": provenances[n].sha256}
             for n, f in frames.items()}
    report.phase(
        "data_understanding",
        objective="Choose datasets small enough to re-run live and real enough to be worth measuring.",
        narrative=(
            "Four datasets from OpenML: "
            + "; ".join(f"**{v['name'].split(' (')[0]}** ({v['rows']:,} rows, {v['positive_rate'] * 100:.0f}% positive"
                        + (f", {v['rows_with_missing'] * 100:.0f}% of rows with a missing value" if v["rows_with_missing"] else "")
                        + ")" for v in facts.values())
            + ". Each was chosen for a property a skill needs: missing values, categorical columns with rare levels, "
            "an imbalanced target, and passengers who travelled in families."
        ),
        code_refs=["ml/data.py:load"],
        gate_notes="Provenance recorded for each; an assertion rejects synthetic data.",
    ).add("Datasets", len(facts)).add("Smallest", f"{min(v['rows'] for v in facts.values()):,} rows")

    print(f"[2/5] running {len(CATALOGUE)} skills over {len(SEEDS)} seeds ...")
    results = []
    for skill in CATALOGUE:
        record = skill.measure(frames[skill.dataset], SEEDS)
        results.append(record)
        print(f"      {record['id']:26} {record['careless']['mean']:.4f} vs {record['careful']['mean']:.4f}  "
              f"gap {record['gap']['mean']:+.4f} [{record['gap']['ci_low']:+.4f}, {record['gap']['ci_high']:+.4f}]  "
              f"{record['verdict']}")

    pairs = {r["id"]: r["seeds"] for r in results}
    leakage.require(
        "assert_every_skill_paired_on_the_same_seeds", set(pairs.values()) == {len(SEEDS)},
        f"{len(results)} skills, each run on the same {len(SEEDS)} seeds, careless and careful on every seed; "
        f"seed counts seen: {sorted(set(pairs.values()))}.",
        skills=len(results), seeds=len(SEEDS),
    )
    overclaims = [r["id"] for r in results if r["gap"]["flatters"] and not r["gap"]["decisive"]]
    leakage.require(
        "assert_claims_match_intervals", not overclaims,
        f"{sum(1 for r in results if r['gap']['decisive'])} of {len(results)} skills have a bootstrap interval that "
        f"excludes zero and may state an effect; entries claiming one without it: {overclaims or 'none'}.",
        skills=len(results), decisive=sum(1 for r in results if r["gap"]["decisive"]),
    )

    print("[3/5] recording ...")
    run = RunRecord(project_slug="05_skills_lab", project_title="Data Science Skills Lab",
                    task="measured comparison of careless and careful practice, per skill",
                    primary_metric="flattery", higher_is_better=True)
    run.set_provenance(main_prov)
    run.set_split(SplitInfo(strategy=f"each skill repeats its own design over {len(SEEDS)} seeds",
                            notes="Splits differ by skill; the pairing is by seed within a skill."))
    for r in results:
        # The interval is turned the same way round as `flattery`, so a row whose
        # metric improves downwards does not show a negative difference beside a
        # positive interval.
        orient = 1 if r["higher_is_better"] else -1
        low, high = sorted([r["gap"]["ci_low"] * orient, r["gap"]["ci_high"] * orient])
        run.add_model(ModelScore(
            name=r["name"], family=r["category"],
            metrics={"flattery": flattery(r), "careless": r["careless"]["mean"], "careful": r["careful"]["mean"],
                     "ci_low": low, "ci_high": high},
            fit_seconds=r["seconds"], notes=r["question"],
            params={"dataset": r["dataset"], "metric": r["metric"], "decisive": r["gap"]["decisive"]}))

    print("[4/5] audit ...")
    own = leakage.audit_source(ROOT, temporal_modules=())
    leakage.save_report(own, ARTIFACTS / "audit.md", "Data Science Skills Lab")
    crit = [f for f in own if f.severity == "CRITICAL"]
    warnings = [f for f in own if f.severity == "WARNING"]
    executed = leakage.ledger()
    run.audit = {"critical": len(crit), "warnings": len(warnings), "total": len(own),
                 "runtime_assertions": executed, "assertions_passed": sum(1 for a in executed if a["passed"]),
                 "assertions_run": len(executed)}

    decisive = [r for r in results if r["gap"]["decisive"] and r["gap"]["flatters"]]
    quiet = [r for r in results if not r["gap"]["decisive"]]
    biggest = max(results, key=flattery)
    report.phase(
        "data_preparation",
        objective="Give every skill the same shape so they can be compared and browsed.",
        narrative=(
            "Each entry declares its dataset, its metric, what the careless and careful versions are, and a function "
            "returning both numbers for one seed. The catalogue runs that function over the seeds, pairs the results "
            "so that split-to-split variation cancels, and bootstraps the paired differences. Nothing in an entry's "
            "text is written by hand except its explanation and its pitfalls."
        ),
        code_refs=["ml/skills.py:Skill", "ml/skills.py:CATALOGUE"],
        gate_notes="One record shape for every skill.",
    ).add("Record fields", "careless, careful, gap, interval, verdict")
    report.phase(
        "modeling",
        objective="Implement both versions of each technique faithfully.",
        narrative=(
            "The careless versions are written the way the mistake is usually made: a frame imputed or scaled or "
            "encoded before any split, a feature selector fitted once, classes balanced as a preparation step, a "
            "threshold swept against the held-out labels. The careful versions put the same work inside the fold. "
            "Two entries measure something else: whether accuracy or balanced accuracy is the honest reading of the "
            "same predictions, and whether calibration improves the probabilities."
        ),
        code_refs=["ml/skills.py:_target_encode", "ml/skills.py:_oversample_before_split"],
        gate_notes=f"{len(CATALOGUE)} skills implemented in both versions.",
    ).add("Categories", ", ".join(sorted({r['category'] for r in results})))
    report.phase(
        "evaluation",
        objective="Report which differences the experiment can resolve, and which it cannot.",
        narrative=(
            f"{len(decisive)} of {len(results)} skills show a careless version that flatters the result by more than "
            f"the seed-to-seed noise. The largest is **{biggest['name'].lower()}**, worth "
            f"{flattery(biggest):+.3f} {biggest['metric'].split(' (')[0]}.\n\n"
            + (f"{len(quiet)} entries report no difference this experiment can resolve: "
               + ", ".join(r["name"].lower() for r in quiet) + ". That is a result, not a gap in the catalogue: on "
               "these datasets, with these models, the careless version costs nothing measurable."
               if quiet else "Every entry resolved a difference.")
            + (f"\n\nEvery careless version here is a real leak, yet the repository's own static auditor reports "
               f"{len(warnings)} warnings and {len(crit)} critical findings against this project's source. Reading "
               f"source cannot see these: each is an ordinary call in an ordinary order, and nothing in the text "
               f"says which rows will later become a validation fold. Only running both versions tells them apart."
               if not warnings else
               f"\n\nThe repository's own static auditor reports {len(warnings)} warnings against this project's "
               f"source, on the careless implementations, which is what they are for.")
        ),
        code_refs=["ml/train.py:main"],
        gate_passed=len(crit) == 0 and all(a["passed"] for a in executed),
        gate_notes="Verdicts checked against the intervals by an executed assertion.",
    ).add("Skills with a measurable effect", f"{len(decisive)} of {len(results)}") \
     .add("Largest flattery", f"{flattery(biggest):+.3f}")
    report.phase(
        "deployment",
        objective="Serve the catalogue, and let any entry be re-run live.",
        narrative=(
            "The API serves every record and will re-run a chosen skill on a chosen number of seeds, so a reader can "
            "watch the interval narrow as seeds are added, and can confirm that the same seeds give the same numbers."
        ),
        code_refs=["server/main.py"],
        gate_notes="Live re-runs use the same code path as this script.",
    ).add("Serving port", 8005)

    run.extra = {"skills": results, "datasets": facts, "seeds": SEEDS,
                 "categories": sorted({r["category"] for r in results}),
                 "auditor_warnings": [{"file": str(f.file), "line": f.line, "rule": f.rule, "message": f.message}
                                      for f in warnings]}
    run.duration_seconds = round(time.perf_counter() - t0, 1)
    run.save(ARTIFACTS / "metrics.json")
    report.save(ARTIFACTS / "crispdm.json")
    (ARTIFACTS / "crispdm.md").write_text(report.to_markdown(), encoding="utf-8")

    print("[5/5] done.")
    print(run.leaderboard_markdown(("careless", "careful")))
    print(f"decisive {len(decisive)}/{len(results)} | warnings {len(warnings)} | critical {len(crit)} | "
          f"{run.duration_seconds}s")


if __name__ == "__main__":
    main()
