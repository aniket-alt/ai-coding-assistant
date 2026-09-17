"""
Project 09 -- verify and measure the FlowForge engine.

    python ml/train.py

The engine is TypeScript, run directly by Node. This script is its evidence:

1. the unit and property tests (node --test);
2. the compile-time tests (tsc, where every @ts-expect-error must still fail);
3. mutation testing: known bugs are planted in copies of the engine, and each
   must make the tests fail -- tests that pass on broken code prove nothing;
4. benchmarks of the orderings and of both executors.

Results are recorded in the same run record, audit trail and CRISP-DM report
as every other project.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from dsfs import CrispDmReport, ModelScore, Provenance, RunRecord, SplitInfo, leakage

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
ENGINE = ROOT / "engine"
ARTIFACTS = ROOT / "artifacts"
NODE = shutil.which("node") or "node"
TSC = REPO / "node_modules" / "typescript" / "bin" / "tsc"

# Each mutant is one realistic bug. The tests must catch every one.
MUTANTS = [
    ("Kahn ignores leftover nodes (misses cycles)", "src/graph.ts",
     "if (tail === g.n) return ok(queue);", "return ok(queue);"),
    ("Depth-first search ignores back edges", "src/graph.ts",
     "if (state[v] === 1) return null;", ""),
    ("Levels count two steps per edge", "src/graph.ts",
     "if (level[u] + 1 > level[v]) level[v] = level[u] + 1;", "if (level[u] + 1 > level[v]) level[v] = level[u] + 2;"),
    ("Concurrency limit off by one", "src/executor.ts",
     "while (active < opts.concurrency && ready.size > 0) {", "while (active <= opts.concurrency && ready.size > 0) {"),
    ("Condition branches ignored", "src/executor.ts",
     "(branch === undefined || String(outcome.get(u)) === branch)", "true"),
    ("Failed inputs treated as usable", "src/executor.ts",
     "const usable = state[u] === 'succeeded' &&", "const usable = state[u] !== 'running' &&"),
    ("Retries start before their backoff ends", "src/executor.ts",
     "if (state[v] !== 'pending' || queued[v] || backingOff[v]) return;", "if (state[v] !== 'pending' || queued[v]) return;"),
    ("Join-any waits for every input", "src/executor.ts",
     "if (live[v] > 0 || inputs[v].length === 0) enqueue(v);", "if (open[v] === 0 && live[v] > 0) enqueue(v);"),
]
TYPE_MUTANTS = [
    ("Node ids no longer branded", "src/types.ts",
     "export type NodeId = Brand<string, 'NodeId'>;", "export type NodeId = string;"),
    ("Condition threshold made optional", "src/types.ts",
     "| { kind: 'condition'; metric: string; atLeast: number }", "| { kind: 'condition'; metric: string; atLeast?: number }"),
]


def run(cmd: list[str], cwd: Path, timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", timeout=timeout)


def node_tests(cwd: Path) -> dict:
    out = run([NODE, "--test", "test/*.test.ts"], cwd)
    counts = {k: int(v) for k, v in re.findall(r"^ℹ (tests|pass|fail) (\d+)", out.stdout, re.M)}
    return {"exit": out.returncode, "tests": counts.get("tests", 0), "pass": counts.get("pass", 0), "fail": counts.get("fail", 0)}


def typecheck(cwd: Path) -> dict:
    out = run([NODE, str(TSC), "--noEmit", "-p", "."], cwd)
    errors = re.findall(r"error (TS\d+)", out.stdout)
    return {"exit": out.returncode, "errors": len(errors), "unused_expectations": errors.count("TS2578")}


def mutate(name: str, file: str, before: str, after: str, check) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "engine"
        shutil.copytree(ENGINE, copy, ignore=shutil.ignore_patterns("node_modules"))
        # tsc resolves @types/node from the repository, so link the copy's config back to it.
        tsconfig = json.loads((copy / "tsconfig.json").read_text(encoding="utf-8"))
        tsconfig["compilerOptions"]["typeRoots"] = [str(REPO / "node_modules" / "@types")]
        (copy / "tsconfig.json").write_text(json.dumps(tsconfig), encoding="utf-8")
        target = copy / file
        source = target.read_text(encoding="utf-8")
        if before not in source:
            raise RuntimeError(f"mutant '{name}' no longer matches {file}; update the mutant list")
        target.write_text(source.replace(before, after, 1), encoding="utf-8")
        result = check(copy)
    return {"mutant": name, "file": file, **result}


def main() -> None:
    t0 = time.perf_counter()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    leakage.reset_ledger()
    report = CrispDmReport(
        project_slug="09_flowforge_dag", project_title="FlowForge DAG Engine",
        business_question=(
            "Can a workflow engine's ordering and scheduling be shown correct and fast, "
            "rather than assumed to be?"
        ),
    )
    report.phase(
        "business_understanding",
        objective="Build a typed workflow engine and produce evidence that it is correct and scales.",
        narrative=(
            "Data pipelines are directed acyclic graphs: load, transform, train, evaluate, "
            "decide, deploy. An engine that runs them must order steps so none runs before "
            "its inputs, detect cycles, respect a concurrency limit, retry failures and skip "
            "the branch a condition did not take.\n\n"
            "Engines like this are usually trusted because they seem to work. This project "
            "asks for evidence instead: properties checked against independent references on "
            "thousands of random graphs, mutation testing to show those checks can fail, "
            "compile-time tests of the type design, and measured scaling."
        ),
        code_refs=["engine/src/graph.ts:kahn", "engine/src/executor.ts:simulate"],
        gate_notes="Correctness properties and scaling measures fixed before implementation.",
    ).add("Node kinds", 7).add("Evidence", "properties, mutants, type tests, benchmarks")

    generator = ENGINE / "src" / "random.ts"
    prov = Provenance(
        slug="flowforge-random-graphs",
        name="Seeded random workflow graphs (FlowForge generator)",
        source_url="https://github.com/aniket-alt/ai-coding-assistant/blob/main/09_flowforge_dag/engine/src/random.ts",
        license="MIT (generated by this repository)",
        citation="Generated by engine/src/random.ts with the Mulberry32 generator and fixed seeds.",
        synthetic=True,
        notes="No external dataset: the engine is measured on generated graphs, which are reproducible from their seeds.",
    ).stamped(sha256=hashlib.sha256(generator.read_bytes()).hexdigest(), bytes_on_disk=generator.stat().st_size)
    prov.save(ARTIFACTS / "provenance.json")

    print("[1/5] unit and property tests ...")
    tests = node_tests(ENGINE)
    leakage.require("assert_engine_tests_pass", tests["exit"] == 0 and tests["fail"] == 0 and tests["tests"] > 0,
                    f"node --test ran {tests['tests']} tests: {tests['pass']} passed, {tests['fail']} failed.", **tests)

    print("[2/5] compile-time tests ...")
    types = typecheck(ENGINE)
    leakage.require("assert_type_tests_pass", types["exit"] == 0,
                    f"tsc over the engine, its tests and type-tests: {types['errors']} errors, including "
                    f"{types['unused_expectations']} @ts-expect-error lines that no longer fail.", **types)

    print("[3/5] benchmarks and property counts ...")
    bench_run = run([NODE, "bench.ts"], ENGINE, timeout=1800)
    if bench_run.returncode != 0:
        raise RuntimeError(bench_run.stderr)
    bench = json.loads(bench_run.stdout)
    total_checks = sum(p["checks"] for p in bench["properties"])
    failures = sum(p["failure_count"] for p in bench["properties"])
    leakage.require("assert_properties_hold", failures == 0,
                    f"{len(bench['properties'])} properties over {sum(p['cases'] for p in bench['properties']):,} "
                    f"random graphs and workflows: {total_checks:,} checks, {failures} failures.",
                    properties=len(bench["properties"]), checks=total_checks, failures=failures)

    print(f"[4/5] mutation testing: {len(MUTANTS)} behaviour mutants, {len(TYPE_MUTANTS)} type mutants ...")
    killed = []
    for name, file, before, after in MUTANTS:
        r = mutate(name, file, before, after, node_tests)
        r["killed"] = r["fail"] > 0 or r["exit"] != 0
        killed.append(r)
        print(f"      {'killed ' if r['killed'] else 'SURVIVED'} {name}: {r['fail']} of {r['tests']} tests failed")
    for name, file, before, after in TYPE_MUTANTS:
        r = mutate(name, file, before, after, typecheck)
        r["killed"] = r["exit"] != 0
        r["tests"], r["fail"] = 0, r["errors"]
        killed.append(r)
        print(f"      {'killed ' if r['killed'] else 'SURVIVED'} {name}: {r['errors']} type errors")
    n_killed = sum(m["killed"] for m in killed)
    leakage.require("assert_every_mutant_killed", n_killed == len(killed),
                    f"{len(killed)} planted bugs; the test suites caught {n_killed}.",
                    mutants=len(killed), killed=n_killed)

    print("[5/5] record, report and audit ...")
    run_record = RunRecord(project_slug="09_flowforge_dag", project_title="FlowForge DAG Engine",
                           task="graph ordering: time to order a random DAG", primary_metric="ms_at_8000_nodes",
                           higher_is_better=False)
    run_record.set_provenance(prov)
    run_record.set_split(SplitInfo(strategy="seeded random graphs; every method timed on the same graphs",
                                   notes=f"Median of {bench['repeats']} runs after a warm-up; Node {bench['node']}."))
    head = {h["algorithm"]: h["ms"] for h in bench["head_to_head"]}
    for s in bench["scaling"]:
        largest = s["points"][-1]
        run_record.add_model(ModelScore(
            name=s["algorithm"], family="baseline" if s["algorithm"].startswith("Naive") else "linear-time",
            metrics={"ms_at_8000_nodes": head[s["algorithm"]], "scaling_exponent": s["slope"],
                     "largest_nodes": largest["n"], "ms_at_largest": largest["ms"]},
            notes="Scaling exponent is the slope of log time against log size: about 1 is linear, 2 quadratic."))

    own = leakage.audit_source(ROOT, temporal_modules=())
    leakage.save_report(own, ARTIFACTS / "audit.md", "FlowForge DAG Engine")
    crit = [f for f in own if f.severity == "CRITICAL"]
    executed = leakage.ledger()
    run_record.audit = {"critical": len(crit), "warnings": sum(1 for f in own if f.severity == "WARNING"),
                        "total": len(own), "runtime_assertions": executed,
                        "assertions_passed": sum(1 for a in executed if a["passed"]), "assertions_run": len(executed)}

    kahn = next(s for s in bench["scaling"] if s["algorithm"].startswith("Kahn"))
    naive = next(s for s in bench["scaling"] if s["algorithm"].startswith("Naive"))
    fast = next(e for e in bench["executors"] if e["executor"].startswith("Incremental"))
    slow = next(e for e in bench["executors"] if e["executor"].startswith("Rescanning"))
    shared = max(p["n"] for p in slow["points"])
    fast_at = next(p["ms"] for p in fast["points"] if p["n"] == shared)
    slow_at = next(p["ms"] for p in slow["points"] if p["n"] == shared)

    report.phase(
        "data_understanding",
        objective="Define the graphs the engine is tested and measured on.",
        narrative=(
            "There is no external dataset. Graphs are generated from fixed seeds: nodes are "
            "shuffled and edges only run forward in the hidden shuffle, so every graph is "
            "acyclic without the algorithms being able to read that from node numbers. A "
            "cyclic variant adds one back edge along an existing edge. Random workflows "
            "assign node kinds, durations, failure rates, retry budgets and condition branches."
        ),
        code_refs=["engine/src/random.ts:randomDag", "engine/src/properties.ts:randomWorkflow"],
        gate_notes="Generator seeded and checksummed; marked synthetic in provenance.",
    ).add("Random cases per property", bench["properties"][0]["cases"])
    report.phase(
        "data_preparation",
        objective="Represent graphs so the algorithms run in linear time.",
        narrative=(
            "Workflows are validated first: duplicate ids, self-loops and edges to unknown "
            "nodes are rejected as typed errors, never exceptions. Valid graphs are converted "
            "to compressed adjacency arrays, so each ordering touches every node and edge a "
            "constant number of times."
        ),
        code_refs=["engine/src/graph.ts:validate", "engine/src/graph.ts:fromEdges"],
        gate_notes="Validation returns a Result; no exception paths.",
    )
    report.phase(
        "modeling",
        objective="Implement ordering, cycle reporting, levels and an incremental scheduler.",
        narrative=(
            "Kahn's algorithm orders the graph and, when it cannot finish, reports one concrete "
            "cycle as a closed path. Levels give the longest path from any source, which is "
            "what can run in parallel. The first executor rescanned every node after every "
            "event; it is kept as a reference. The current one keeps, for each node, counts of "
            f"open, usable and dead inputs and a heap of ready nodes. On {shared:,}-node workflows "
            f"it takes {fast_at:.1f} ms against the reference's {slow_at:,.0f} ms.\n\n"
            "The property that compares the two executors found a real bug in the rewrite: a "
            "node waiting out a retry backoff could be started early by an unrelated event, "
            "then again by its timer. The fix is covered by a mutant below."
        ),
        code_refs=["engine/src/graph.ts:kahn", "engine/src/executor.ts:simulate", "engine/src/reference-executor.ts:simulateByRescan"],
        gate_notes="Both executors present; the reference is the oracle for the fast one.",
    ).add("Incremental executor exponent", f"{fast['slope']:.2f}").add("Reference executor exponent", f"{slow['slope']:.2f}")
    report.phase(
        "evaluation",
        objective="Show the engine is correct, that the tests could have caught it being wrong, and how it scales.",
        narrative=(
            f"**Correctness.** {total_checks:,} property checks over "
            f"{sum(p['cases'] for p in bench['properties']):,} random graphs and workflows, with "
            f"{failures} failures, plus {tests['tests']} tests in all.\n\n"
            f"**The tests can fail.** {n_killed} of {len(killed)} planted bugs were caught, "
            f"{len(MUTANTS)} in behaviour and {len(TYPE_MUTANTS)} in the types.\n\n"
            f"**Scaling.** Kahn's algorithm orders a {kahn['points'][-1]['n']:,}-node graph in "
            f"{kahn['points'][-1]['ms']:.0f} ms with a scaling exponent of {kahn['slope']:.2f}; the "
            f"naive repeated scan's exponent is {naive['slope']:.2f}, and at 8,000 nodes it takes "
            f"{head[naive['algorithm']]:.0f} ms against {head[kahn['algorithm']]:.2f} ms."
        ),
        code_refs=["engine/src/properties.ts", "ml/train.py:mutate", "engine/bench.ts"],
        gate_passed=len(crit) == 0 and all(a["passed"] for a in executed),
        gate_notes="Tests, type tests, properties and mutation testing all executed and recorded.",
    ).add("Property checks", total_checks).add("Mutants caught", f"{n_killed} of {len(killed)}")
    report.phase(
        "deployment",
        objective="Run the same engine in the browser.",
        narrative=(
            "The client imports the engine's TypeScript directly, so the canvas, the run "
            "replay and the cycle demonstration use exactly the code the tests and benchmarks "
            "measured. The API serves the recorded evidence."
        ),
        code_refs=["client/src/App.tsx", "server/main.py"],
        gate_notes="One engine source shared by tests, benchmarks and the UI.",
    ).add("Serving port", 8009)

    run_record.extra = {"tests": tests, "types": types, "bench": bench, "mutants": killed,
                        "property_checks": total_checks, "property_failures": failures}
    run_record.duration_seconds = round(time.perf_counter() - t0, 1)
    run_record.save(ARTIFACTS / "metrics.json")
    report.save(ARTIFACTS / "crispdm.json")
    (ARTIFACTS / "crispdm.md").write_text(report.to_markdown(), encoding="utf-8")
    print(run_record.leaderboard_markdown(("scaling_exponent", "ms_at_largest")))
    print(f"tests {tests} | types {types} | mutants {n_killed}/{len(killed)} | critical {len(crit)} | {run_record.duration_seconds}s")


if __name__ == "__main__":
    main()
