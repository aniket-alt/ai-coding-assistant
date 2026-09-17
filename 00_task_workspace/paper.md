# Task Workspace

*A live task workspace: every tab stays in sync, and no edit silently overwrites another.*

**Aniket Anil Naik** · Project 00 of the data science full-stack portfolio

## Abstract

**Does the workspace understand quick entry, keep every tab in sync, and never lose an edit?**

We study this question on *Hand-written quick-entry phrases and generated workspace traffic*: 60 rows as downloaded, used under MIT (written for this repository). Following the CRISP-DM process, we compare 2 approaches, 1 of them baselines, on 60 evaluation rows and rank them by exact parses. Evaluation split: hand-written labelled phrases.

The best approach, **Quick-entry parser**, reaches 100.0% against 28.3% for the strongest baseline (Title only (no syntax)).

3 of 3 leakage assertions passed while training ran, and static analysis of the source reported 0 critical and 0 warning findings.

**Keywords:** quick-entry parsing: recover fields from typed text, server-sent events with sequence numbers, optimistic concurrency with versions, quick-entry parser with a labelled test set, Kanban, Eisenhower matrix, calendar, focus timer


## 1. Introduction

*Objective.* State the workspace's promises as things that can be measured.

A task app earns trust through three behaviours users rarely see directly. What they type is understood. A change made in one tab appears in every other. And when two people change the same task at once, neither change vanishes without a trace. Each is tested here against the running application.

| Finding | Value | Note |
|---|---|---|
| Promises | 3 |  |

> Phase gate passed: Three measurable promises fixed before building the checks.


## 2. Data

*Objective.* Decide what to test the parser and the server with.

60 phrases were written by hand against the documented grammar, including awkward ones: email addresses, issue numbers such as #3, impossible dates and markers with trailing punctuation. Because the same author wrote the parser and the phrases, 5,000 generated round trips provide a check that no phrase was chosen to pass.

| Finding | Value | Note |
|---|---|---|
| Labelled phrases | 60 |  |

### Provenance

⚠️ **Synthetic data**

| Field | Value |
|---|---|
| Dataset | Hand-written quick-entry phrases and generated workspace traffic |
| Source | https://github.com/aniket-alt/ai-coding-assistant/blob/main/00_task_workspace/ml/parser_cases.json |
| Licence | MIT (written for this repository) |
| Rows × columns (as downloaded) | 60 × 0 |
| SHA-256 | `d958d955c4b15f69c36271b3e7b57d7945d527d9bc6ad74543645629102caa6e` |

> Phase gate passed: Test inputs fixed; their authorship stated.


## 3. Method

*Objective.* Run every check against the real server on a throwaway database.

The API is started as a subprocess on a free port with an empty temporary SQLite file, so the checks exercise the same code users run, and nothing persists between runs.

> Phase gate passed: Isolated database per run.


## 4. Models

*Objective.* Build the grammar, the event stream and the version check.

Quick entry is a small grammar resolved against an explicit date. Changes are written to SQLite and broadcast as server-sent events with a global sequence number, so a tab can tell whether it missed anything. Every update names the version it was based on; the store checks and writes under one lock.

**Table 1.** Tournament results.

| # | Parser | Exact parses | Fields correct | Due date | Tags |
|---|---|---|---|---|---|
| 1 | **Quick-entry parser** 🏆 | 100.0% | 100.0% | 100.0% | 100.0% |
| 2 | Title only (no syntax) *(baseline)* | 28.3% | 66.7% | 60.0% | 75.0% |

Ranked by exact parses (higher is better). Split: hand-written labelled phrases; 60 evaluation rows. Generated from `artifacts/metrics.json`.

> Phase gate passed: Three mechanisms, each with a test.


## 5. Results

*Objective.* Report what the checks measured.

**Parsing.** 100.0% of labelled phrases were parsed exactly, against 28.3% for treating everything as the title; 0 of 5,000 round trips failed.

**Live updates.** With 50 tabs open, every tab received all 200 changes in order: yes. Delivery took 4.0 ms at the median and 14.8 ms at the 99th percentile.

**Conflicts.** In 50 rounds of 8 simultaneous edits, 50 were accepted and 350 refused; without the version check, each refused edit would have silently replaced another.

| Finding | Value | Note |
|---|---|---|
| Exact parses | 100.0% |  |
| p99 delivery, 50 tabs | 14.8 ms |  |

> Phase gate passed: All promises checked against the running server.

## 6. Threats to validity

### Assertions executed during training

| Assertion | Result | What it measured |
|---|---|---|
| `assert_round_trips_recover_every_field` | ✅ passed | 5,000 generated tasks rendered as quick-entry text in random order; 0 did not parse back to exactly the fields they were built from. |
| `assert_every_tab_receives_every_change_in_order` | ✅ passed | 1 subscribers: 1 received all 200 changes in sequence; 10 subscribers: 10 received all 200 changes in sequence; 50 subscribers: 50 received all 200 changes in sequence. |
| `assert_exactly_one_concurrent_writer_wins` | ✅ passed | 50 rounds of 8 writers on the same version: 50 accepted, 350 refused with 409; final version 51 (expected 51). |

Recorded by the assertion ledger while training ran; generated from `artifacts/metrics.json`.

### Limitations

- The labelled phrases were written by the parser's author, so their pass rate is weak evidence; the generated round trips only cover the grammar they were generated from.
- Latency was measured with server and clients on one machine, which excludes network time, and Windows timestamps have roughly millisecond resolution.
- One uvicorn worker holds the event subscribers in memory; several workers would need a shared broker.
- The demonstration tasks and their history are generated, so the analytics show how the views work, not how anyone works.


## 7. Deployment

*Objective.* Ship the workspace.

FastAPI serves the tasks, analytics and event stream on port 8000; the React client keeps every view in sync from the stream and shows a conflict when an edit is refused.

| Finding | Value | Note |
|---|---|---|
| Serving port | 8,000 |  |

> Phase gate passed: One server, one stream, every view.

## 8. Conclusion

- Sequence numbers on live events turn 'did this tab miss something?' into a check the client can make and recover from.
- Sending the version an edit was based on is enough to make lost updates visible instead of silent.
- A generated round-trip test caught a mistake in its own expectations, which is why generated and hand-written tests are both worth having.
- A page that holds a live connection never becomes network-idle, so tooling that waits for idleness needs another signal.

## Reproducibility

Trained 2026-09-17T16:19:14Z in 9.4 s at commit `1e3bb26` with Python 3.11.6, NumPy 2.4.6, pandas 3.0.5 and scikit-learn 1.9.1.

```bash
python ml/train.py
python -m uvicorn main:app --app-dir server --port 8000
npm run dev -w 00_task_workspace/client
```

## References

1. Labelled phrases in ml/parser_cases.json; round trips and traffic generated by ml/train.py with fixed seeds.
2. WHATWG. HTML Living Standard, section 9.2: Server-sent events.
3. Kung, H. T., & Robinson, J. T. (1981). On optimistic methods for concurrency control. ACM Transactions on Database Systems, 6(2), 213–226.
4. Fielding, R., & Reschke, J. (2014). RFC 7232: Hypertext Transfer Protocol (HTTP/1.1): Conditional Requests.
5. Cirillo, F. (2006). The Pomodoro Technique.
6. Covey, S. R. (1989). The 7 Habits of Highly Effective People (the urgent–important matrix).

---

*Generated from this project's artifacts by `scripts/generate_docs.py`.*
