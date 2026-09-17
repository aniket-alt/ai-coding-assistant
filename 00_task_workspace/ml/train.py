"""
Project 00 -- verify the workspace's three promises.

    python ml/train.py

1. Quick entry understands what was typed: hand-written labelled phrases, plus
   thousands of generated round trips in which known fields are rendered into
   text and must be recovered exactly.
2. Every open tab sees every change, in order, quickly: a real server, many
   subscribers to the live event stream, and hundreds of changes.
3. No edit silently overwrites another: concurrent writers based on the same
   version, where exactly one may win.

There is no dataset to model; results go into the same run record, audit trail
and CRISP-DM report as every other project.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import date, timedelta
from pathlib import Path

import httpx

from dsfs import CrispDmReport, ModelScore, Provenance, RunRecord, SplitInfo, leakage

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
sys.path.insert(0, str(ROOT / "server"))

from parser import PRIORITY_NAMES, WEEKDAYS, parse  # noqa: E402

FIELDS = ("title", "priority", "tags", "due", "estimate_min")
ROUND_TRIPS = 5000
SUBSCRIBER_LEVELS = (1, 10, 50)
CHANGES = 200
CONFLICT_ROUNDS, WRITERS = 50, 8
SEED = 17


def score(parser, cases: list[dict], today: date) -> dict:
    per_field = {f: 0 for f in FIELDS}
    exact, failures = 0, []
    for c in cases:
        got = parser(c["text"], today)
        right = [f for f in FIELDS if got[f] == c[f]]
        for f in right:
            per_field[f] += 1
        if len(right) == len(FIELDS):
            exact += 1
        else:
            failures.append({"text": c["text"], "wrong": {f: {"got": got[f], "expected": c[f]} for f in FIELDS if f not in right}})
    n = len(cases)
    return {"exact_match_pct": exact / n * 100, "fields_correct_pct": sum(per_field.values()) / (n * len(FIELDS)) * 100,
            **{f"{f}_pct": v / n * 100 for f, v in per_field.items()}, "failures": failures}


def title_only(text: str, today: date) -> dict:
    """The baseline: no quick-entry syntax at all, so everything typed becomes the title."""
    return {"title": " ".join(text.split()), "priority": 3, "tags": [], "due": None, "estimate_min": None}


def full_parser(text: str, today: date) -> dict:
    return parse(text, today).to_dict()


def round_trips(n: int, today: date, rng: random.Random) -> dict:
    """Render known fields into quick-entry text in random order and check they come back exactly."""
    words = ["review", "draft", "plan", "email", "fix", "the", "budget", "report", "garden", "api", "notes", "call",
             "team", "invoice", "3pm", "v2", "q3", "c#", "50%", "bob@example.com", "issue", "#3", "!!", "and"]
    failures, checked = [], 0
    for i in range(n):
        title = " ".join(rng.choice(words) for _ in range(rng.randint(1, 6)))
        # A title word that is itself valid marker syntax would be parsed; keep generated titles unambiguous.
        if parse(title, today).to_dict()["title"] != title:
            continue
        priority = rng.choice([1, 2, 3, 4])
        tags = rng.sample(["work", "home", "deep-work", "q3_plan", "errands", "x1"], rng.randint(0, 3))
        due_kind = rng.choice(["none", "today", "tomorrow", "weekday", "in-days", "iso", "month"])
        due, marker = None, None
        if due_kind == "today":
            due, marker = today, "@today"
        elif due_kind == "tomorrow":
            due, marker = today + timedelta(days=1), "@tomorrow"
        elif due_kind == "weekday":
            k = rng.randrange(7)
            due, marker = today + timedelta(days=(k - today.weekday()) % 7 or 7), f"@{rng.choice([WEEKDAYS[k], WEEKDAYS[k][:3]])}"
        elif due_kind == "in-days":
            k = rng.randint(1, 60)
            due, marker = today + timedelta(days=k), f"@in-{k}-days"
        elif due_kind == "iso":
            due = today + timedelta(days=rng.randint(-30, 400))
            marker = f"@{due.isoformat()}"
        elif due_kind == "month":
            due = today + timedelta(days=rng.randint(0, 360))
            marker = f"@{due.strftime('%b').lower()}-{due.day}"
        estimate = rng.choice([None, 5, 25, 60, 90, 135, 240])
        parts = title.split()
        markers = [f"!{rng.choice([PRIORITY_NAMES[priority], str(priority)])}"] + [f"#{t}" for t in tags]
        if marker:
            markers.append(marker)
        if estimate:
            h, m = divmod(estimate, 60)
            markers.append("~" + rng.choice([f"{estimate}m", str(estimate)] + ([f"{h}h{m}m" if m else f"{h}h"] if h else [])))
        for mk in markers:
            parts.insert(rng.randint(0, len(parts)), mk)
        text = " ".join(parts)
        got = parse(text, today).to_dict()
        typed_order = [p[1:] for p in parts if p.startswith("#") and p[1:] in tags]  # tags are kept in typed order
        expected = {"title": title, "priority": priority, "tags": typed_order, "due": due.isoformat() if due else None,
                    "estimate_min": estimate}
        checked += 1
        if any(got[f] != expected[f] for f in FIELDS):
            failures.append({"text": text, "got": {f: got[f] for f in FIELDS}, "expected": expected})
    return {"generated": n, "checked": checked, "failures": len(failures), "examples": failures[:5]}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Server:
    """The real API in a subprocess, on a throwaway database."""

    def __enter__(self):
        # SQLite can hold the file briefly after the server exits on Windows; a leftover temp file is harmless.
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.port = free_port()
        env = {**os.environ, "TASKS_DB": str(Path(self.tmp.name) / "bench.db"), "TASKS_SEED": "0"}
        self.proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "main:app", "--app-dir", str(ROOT / "server"),
                                      "--port", str(self.port), "--log-level", "warning"],
                                     cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.url = f"http://127.0.0.1:{self.port}"
        for _ in range(200):
            try:
                if httpx.get(f"{self.url}/api/health", timeout=1).status_code == 200:
                    return self
            except httpx.HTTPError:
                time.sleep(0.1)
        raise RuntimeError("server did not start")

    def __exit__(self, *exc):
        self.proc.terminate()
        self.proc.wait(timeout=10)
        self.tmp.cleanup()


async def live_updates(url: str, subscribers: int, changes: int) -> dict:
    received: list[list[tuple[int, float, float]]] = [[] for _ in range(subscribers)]
    ready = [asyncio.Event() for _ in range(subscribers)]
    limits = httpx.Limits(max_connections=subscribers + 20)

    async with httpx.AsyncClient(timeout=None, limits=limits) as client:
        start_seq = (await client.get(f"{url}/api/tasks")).json()["seq"]

        async def listen(i: int):
            async with client.stream("GET", f"{url}/api/events") as resp:
                async for line in resp.aiter_lines():
                    if line.startswith("event: hello"):
                        ready[i].set()
                    elif line.startswith("data: ") and ready[i].is_set():
                        ev = json.loads(line[6:])
                        if "type" in ev:
                            received[i].append((ev["seq"], ev["at"], time.time() * 1000))
                            if ev["seq"] >= start_seq + changes:
                                return

        tasks = [asyncio.create_task(listen(i)) for i in range(subscribers)]
        await asyncio.wait_for(asyncio.gather(*(r.wait() for r in ready)), timeout=30)
        task = (await client.post(f"{url}/api/tasks", json={"title": "benchmark"})).json()
        version = task["version"]
        t0 = time.perf_counter()
        for k in range(changes - 1):
            r = await client.patch(f"{url}/api/tasks/{task['id']}", json={"version": version, "fields": {"notes": f"edit {k}"}})
            version = r.json()["version"]
        write_seconds = time.perf_counter() - t0
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=60)

    expected = list(range(start_seq + 1, start_seq + changes + 1))
    latencies = [recv - at for sub in received for _, at, recv in sub]
    complete = sum(1 for sub in received if [s for s, _, _ in sub] == expected)
    q = statistics.quantiles(latencies, n=100)
    return {"subscribers": subscribers, "changes": changes, "deliveries": len(latencies),
            "subscribers_complete_in_order": complete, "p50_ms": q[49], "p95_ms": q[94], "p99_ms": q[98],
            "max_ms": max(latencies), "writes_per_second": (changes - 1) / write_seconds}


async def conflicts(url: str, rounds: int, writers: int) -> dict:
    async with httpx.AsyncClient(timeout=30) as client:
        task = (await client.post(f"{url}/api/tasks", json={"title": "contended"})).json()
        wins = conflicts_seen = bad_rounds = 0
        for r in range(rounds):
            base = (await client.get(f"{url}/api/tasks")).json()["tasks"]
            version = next(t for t in base if t["id"] == task["id"])["version"]
            results = await asyncio.gather(*(client.patch(f"{url}/api/tasks/{task['id']}",
                                                          json={"version": version, "fields": {"notes": f"round {r} writer {w}"}})
                                             for w in range(writers)))
            codes = [x.status_code for x in results]
            ok, clash = codes.count(200), codes.count(409)
            wins += ok
            conflicts_seen += clash
            if ok != 1 or clash != writers - 1:
                bad_rounds += 1
        final = (await client.get(f"{url}/api/tasks")).json()["tasks"]
        final_version = next(t for t in final if t["id"] == task["id"])["version"]
    return {"rounds": rounds, "writers": writers, "accepted": wins, "refused_409": conflicts_seen,
            "rounds_not_exactly_one_winner": bad_rounds, "final_version": final_version,
            "expected_final_version": 1 + rounds}


def main() -> None:
    t0 = time.perf_counter()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    leakage.reset_ledger()
    report = CrispDmReport(
        project_slug="00_task_workspace", project_title="Task Workspace",
        business_question="Does the workspace understand quick entry, keep every tab in sync, and never lose an edit?",
    )
    report.phase(
        "business_understanding",
        objective="State the workspace's promises as things that can be measured.",
        narrative=(
            "A task app earns trust through three behaviours users rarely see directly. What they "
            "type is understood. A change made in one tab appears in every other. And when two "
            "people change the same task at once, neither change vanishes without a trace. Each is "
            "tested here against the running application."
        ),
        code_refs=["server/parser.py", "server/main.py", "server/store.py"],
        gate_notes="Three measurable promises fixed before building the checks.",
    ).add("Promises", 3)

    cases_file = ROOT / "ml" / "parser_cases.json"
    cases_doc = json.loads(cases_file.read_text(encoding="utf-8"))
    today = date.fromisoformat(cases_doc["today"])
    prov = Provenance(
        slug="task-workspace-cases",
        name="Hand-written quick-entry phrases and generated workspace traffic",
        source_url="https://github.com/aniket-alt/ai-coding-assistant/blob/main/00_task_workspace/ml/parser_cases.json",
        license="MIT (written for this repository)",
        citation="Labelled phrases in ml/parser_cases.json; round trips and traffic generated by ml/train.py with fixed seeds.",
        synthetic=True,
        notes="No external dataset. The labelled phrases were written by the parser's author.",
    ).stamped(sha256=__import__("hashlib").sha256(cases_file.read_bytes()).hexdigest(), n_rows=len(cases_doc["cases"]),
              bytes_on_disk=cases_file.stat().st_size)
    prov.save(ARTIFACTS / "provenance.json")

    print("[1/4] quick-entry parser ...")
    full = score(full_parser, cases_doc["cases"], today)
    base = score(title_only, cases_doc["cases"], today)
    trips = round_trips(ROUND_TRIPS, today, random.Random(SEED))
    leakage.require("assert_round_trips_recover_every_field", trips["failures"] == 0,
                    f"{trips['checked']:,} generated tasks rendered as quick-entry text in random order; "
                    f"{trips['failures']} did not parse back to exactly the fields they were built from.",
                    checked=trips["checked"], failures=trips["failures"])
    print(f"      labelled: exact {full['exact_match_pct']:.1f}% vs baseline {base['exact_match_pct']:.1f}% | round trips {trips['checked']} with {trips['failures']} failures")

    print(f"[2/4] live updates: {CHANGES} changes to {SUBSCRIBER_LEVELS} subscribers ...")
    with Server() as srv:
        live = [asyncio.run(live_updates(srv.url, n, CHANGES)) for n in SUBSCRIBER_LEVELS]
        for row in live:
            print(f"      {row['subscribers']:>3} subscribers: {row['subscribers_complete_in_order']} complete, p50 {row['p50_ms']:.2f} ms, p99 {row['p99_ms']:.2f} ms")
        print(f"[3/4] conflicts: {CONFLICT_ROUNDS} rounds of {WRITERS} simultaneous writers ...")
        clash = asyncio.run(conflicts(srv.url, CONFLICT_ROUNDS, WRITERS))
    complete = all(r["subscribers_complete_in_order"] == r["subscribers"] for r in live)
    leakage.require("assert_every_tab_receives_every_change_in_order", complete,
                    "; ".join(f"{r['subscribers']} subscribers: {r['subscribers_complete_in_order']} received all "
                              f"{r['changes']} changes in sequence" for r in live) + ".",
                    levels=len(live), changes=CHANGES)
    leakage.require("assert_exactly_one_concurrent_writer_wins", clash["rounds_not_exactly_one_winner"] == 0
                    and clash["final_version"] == clash["expected_final_version"],
                    f"{clash['rounds']} rounds of {clash['writers']} writers on the same version: {clash['accepted']} "
                    f"accepted, {clash['refused_409']} refused with 409; final version {clash['final_version']} "
                    f"(expected {clash['expected_final_version']}).",
                    **{k: clash[k] for k in ("rounds", "writers", "accepted", "refused_409", "rounds_not_exactly_one_winner")})

    print("[4/4] record, report and audit ...")
    run = RunRecord(project_slug="00_task_workspace", project_title="Task Workspace",
                    task="quick-entry parsing: recover fields from typed text", primary_metric="exact_match_pct",
                    higher_is_better=True)
    run.set_provenance(prov)
    run.set_split(SplitInfo(strategy="hand-written labelled phrases", n_test=len(cases_doc["cases"]),
                            notes="Written by the parser's author; generated round trips are the independent check."))
    run.add_model(ModelScore(name="Quick-entry parser", family="rule grammar",
                             metrics={k: v for k, v in full.items() if k != "failures"},
                             notes="server/parser.py"))
    run.add_model(ModelScore(name="Title only (no syntax)", family="baseline",
                             metrics={k: v for k, v in base.items() if k != "failures"},
                             notes="Everything typed becomes the title."))
    own = leakage.audit_source(ROOT, temporal_modules=())
    leakage.save_report(own, ARTIFACTS / "audit.md", "Task Workspace")
    crit = [f for f in own if f.severity == "CRITICAL"]
    executed = leakage.ledger()
    run.audit = {"critical": len(crit), "warnings": sum(1 for f in own if f.severity == "WARNING"),
                 "total": len(own), "runtime_assertions": executed,
                 "assertions_passed": sum(1 for a in executed if a["passed"]), "assertions_run": len(executed)}

    worst = max(live, key=lambda r: r["subscribers"])
    report.phase("data_understanding", objective="Decide what to test the parser and the server with.",
                 narrative=(f"{len(cases_doc['cases'])} phrases were written by hand against the documented grammar, "
                            f"including awkward ones: email addresses, issue numbers such as #3, impossible dates and "
                            f"markers with trailing punctuation. Because the same author wrote the parser and the "
                            f"phrases, {trips['checked']:,} generated round trips provide a check that no phrase was "
                            f"chosen to pass."),
                 code_refs=["ml/parser_cases.json", "ml/train.py:round_trips"],
                 gate_notes="Test inputs fixed; their authorship stated.").add("Labelled phrases", len(cases_doc["cases"]))
    report.phase("data_preparation", objective="Run every check against the real server on a throwaway database.",
                 narrative="The API is started as a subprocess on a free port with an empty temporary SQLite file, "
                           "so the checks exercise the same code users run, and nothing persists between runs.",
                 code_refs=["ml/train.py:Server"], gate_notes="Isolated database per run.")
    report.phase("modeling", objective="Build the grammar, the event stream and the version check.",
                 narrative=("Quick entry is a small grammar resolved against an explicit date. Changes are written to "
                            "SQLite and broadcast as server-sent events with a global sequence number, so a tab can "
                            "tell whether it missed anything. Every update names the version it was based on; the "
                            "store checks and writes under one lock."),
                 code_refs=["server/parser.py:parse", "server/main.py:Broker", "server/store.py:Store.update"],
                 gate_notes="Three mechanisms, each with a test.")
    report.phase("evaluation", objective="Report what the checks measured.",
                 narrative=(f"**Parsing.** {full['exact_match_pct']:.1f}% of labelled phrases were parsed exactly, "
                            f"against {base['exact_match_pct']:.1f}% for treating everything as the title; "
                            f"{trips['failures']} of {trips['checked']:,} round trips failed.\n\n"
                            f"**Live updates.** With {worst['subscribers']} tabs open, every tab received all "
                            f"{worst['changes']} changes in order: {'yes' if complete else 'no'}. Delivery took "
                            f"{worst['p50_ms']:.1f} ms at the median and {worst['p99_ms']:.1f} ms at the 99th percentile.\n\n"
                            f"**Conflicts.** In {clash['rounds']} rounds of {clash['writers']} simultaneous edits, "
                            f"{clash['accepted']} were accepted and {clash['refused_409']} refused; without the version "
                            f"check, each refused edit would have silently replaced another."),
                 code_refs=["ml/train.py:live_updates", "ml/train.py:conflicts"],
                 gate_passed=len(crit) == 0 and all(a["passed"] for a in executed),
                 gate_notes="All promises checked against the running server.") \
        .add("Exact parses", f"{full['exact_match_pct']:.1f}%").add(f"p99 delivery, {worst['subscribers']} tabs", f"{worst['p99_ms']:.1f} ms")
    report.phase("deployment", objective="Ship the workspace.",
                 narrative="FastAPI serves the tasks, analytics and event stream on port 8000; the React client keeps "
                           "every view in sync from the stream and shows a conflict when an edit is refused.",
                 code_refs=["server/main.py", "client/src/App.tsx"], gate_notes="One server, one stream, every view.") \
        .add("Serving port", 8000)

    run.extra = {"parser": {"cases": len(cases_doc["cases"]), "full": full, "baseline": {k: v for k, v in base.items() if k != "failures"},
                            "round_trips": trips, "today": cases_doc["today"]},
                 "live": live, "conflicts": clash, "machine": f"{os.cpu_count()} logical CPUs, one uvicorn worker"}
    run.duration_seconds = round(time.perf_counter() - t0, 1)
    run.save(ARTIFACTS / "metrics.json")
    report.save(ARTIFACTS / "crispdm.json")
    (ARTIFACTS / "crispdm.md").write_text(report.to_markdown(), encoding="utf-8")
    print(json.dumps({"conflicts": clash, "live": live}, indent=1))
    print(f"critical {len(crit)} | {run.duration_seconds}s")


if __name__ == "__main__":
    main()
