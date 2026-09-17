# Abstract — Task Workspace

**Does the workspace understand quick entry, keep every tab in sync, and never lose an edit?**

We study this question on *Hand-written quick-entry phrases and generated workspace traffic*: 60 rows as downloaded, used under MIT (written for this repository). Following the CRISP-DM process, we compare 2 approaches, 1 of them baselines, on 60 evaluation rows and rank them by exact parses. Evaluation split: hand-written labelled phrases.

The best approach, **Quick-entry parser**, reaches 100.0% against 28.3% for the strongest baseline (Title only (no syntax)).

3 of 3 leakage assertions passed while training ran, and static analysis of the source reported 0 critical and 0 warning findings.

**Keywords:** quick-entry parsing: recover fields from typed text, server-sent events with sequence numbers, optimistic concurrency with versions, quick-entry parser with a labelled test set, Kanban, Eisenhower matrix, calendar, focus timer

---

*Generated from this project's artifacts by `scripts/generate_docs.py`. Figures are not typed by hand.*
