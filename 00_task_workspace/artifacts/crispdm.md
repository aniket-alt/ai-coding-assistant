# CRISP-DM Report — Task Workspace

**Business question.** Does the workspace understand quick entry, keep every tab in sync, and never lose an edit?


## 1. Business Understanding

*Objective:* State the workspace's promises as things that can be measured.


A task app earns trust through three behaviours users rarely see directly. What they type is understood. A change made in one tab appears in every other. And when two people change the same task at once, neither change vanishes without a trace. Each is tested here against the running application.


| Finding | Value | Detail |
|---|---|---|
| Promises | `3` |  |


*Code:* `server/parser.py`, `server/main.py`, `server/store.py`


> **Phase gate: PASSED.** Three measurable promises fixed before building the checks.


## 2. Data Understanding

*Objective:* Decide what to test the parser and the server with.


60 phrases were written by hand against the documented grammar, including awkward ones: email addresses, issue numbers such as #3, impossible dates and markers with trailing punctuation. Because the same author wrote the parser and the phrases, 5,000 generated round trips provide a check that no phrase was chosen to pass.


| Finding | Value | Detail |
|---|---|---|
| Labelled phrases | `60` |  |


*Code:* `ml/parser_cases.json`, `ml/train.py:round_trips`


> **Phase gate: PASSED.** Test inputs fixed; their authorship stated.


## 3. Data Preparation

*Objective:* Run every check against the real server on a throwaway database.


The API is started as a subprocess on a free port with an empty temporary SQLite file, so the checks exercise the same code users run, and nothing persists between runs.


*Code:* `ml/train.py:Server`


> **Phase gate: PASSED.** Isolated database per run.


## 4. Modeling

*Objective:* Build the grammar, the event stream and the version check.


Quick entry is a small grammar resolved against an explicit date. Changes are written to SQLite and broadcast as server-sent events with a global sequence number, so a tab can tell whether it missed anything. Every update names the version it was based on; the store checks and writes under one lock.


*Code:* `server/parser.py:parse`, `server/main.py:Broker`, `server/store.py:Store.update`


> **Phase gate: PASSED.** Three mechanisms, each with a test.


## 5. Evaluation

*Objective:* Report what the checks measured.


**Parsing.** 100.0% of labelled phrases were parsed exactly, against 28.3% for treating everything as the title; 0 of 5,000 round trips failed.

**Live updates.** With 50 tabs open, every tab received all 200 changes in order: yes. Delivery took 4.0 ms at the median and 14.8 ms at the 99th percentile.

**Conflicts.** In 50 rounds of 8 simultaneous edits, 50 were accepted and 350 refused; without the version check, each refused edit would have silently replaced another.


| Finding | Value | Detail |
|---|---|---|
| Exact parses | `100.0%` |  |
| p99 delivery, 50 tabs | `14.8 ms` |  |


*Code:* `ml/train.py:live_updates`, `ml/train.py:conflicts`


> **Phase gate: PASSED.** All promises checked against the running server.


## 6. Deployment

*Objective:* Ship the workspace.


FastAPI serves the tasks, analytics and event stream on port 8000; the React client keeps every view in sync from the stream and shows a conflict when an edit is refused.


| Finding | Value | Detail |
|---|---|---|
| Serving port | `8,000` |  |


*Code:* `server/main.py`, `client/src/App.tsx`


> **Phase gate: PASSED.** One server, one stream, every view.
