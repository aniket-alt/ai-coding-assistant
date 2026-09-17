# CRISP-DM Report — FlowForge DAG Engine

**Business question.** Can a workflow engine's ordering and scheduling be shown correct and fast, rather than assumed to be?


## 1. Business Understanding

*Objective:* Build a typed workflow engine and produce evidence that it is correct and scales.


Data pipelines are directed acyclic graphs: load, transform, train, evaluate, decide, deploy. An engine that runs them must order steps so none runs before its inputs, detect cycles, respect a concurrency limit, retry failures and skip the branch a condition did not take.

Engines like this are usually trusted because they seem to work. This project asks for evidence instead: properties checked against independent references on thousands of random graphs, mutation testing to show those checks can fail, compile-time tests of the type design, and measured scaling.


| Finding | Value | Detail |
|---|---|---|
| Node kinds | `7` |  |
| Evidence | `properties, mutants, type tests, benchmarks` |  |


*Code:* `engine/src/graph.ts:kahn`, `engine/src/executor.ts:simulate`


> **Phase gate: PASSED.** Correctness properties and scaling measures fixed before implementation.


## 2. Data Understanding

*Objective:* Define the graphs the engine is tested and measured on.


There is no external dataset. Graphs are generated from fixed seeds: nodes are shuffled and edges only run forward in the hidden shuffle, so every graph is acyclic without the algorithms being able to read that from node numbers. A cyclic variant adds one back edge along an existing edge. Random workflows assign node kinds, durations, failure rates, retry budgets and condition branches.


| Finding | Value | Detail |
|---|---|---|
| Random cases per property | `1,000` |  |


*Code:* `engine/src/random.ts:randomDag`, `engine/src/properties.ts:randomWorkflow`


> **Phase gate: PASSED.** Generator seeded and checksummed; marked synthetic in provenance.


## 3. Data Preparation

*Objective:* Represent graphs so the algorithms run in linear time.


Workflows are validated first: duplicate ids, self-loops and edges to unknown nodes are rejected as typed errors, never exceptions. Valid graphs are converted to compressed adjacency arrays, so each ordering touches every node and edge a constant number of times.


*Code:* `engine/src/graph.ts:validate`, `engine/src/graph.ts:fromEdges`


> **Phase gate: PASSED.** Validation returns a Result; no exception paths.


## 4. Modeling

*Objective:* Implement ordering, cycle reporting, levels and an incremental scheduler.


Kahn's algorithm orders the graph and, when it cannot finish, reports one concrete cycle as a closed path. Levels give the longest path from any source, which is what can run in parallel. The first executor rescanned every node after every event; it is kept as a reference. The current one keeps, for each node, counts of open, usable and dead inputs and a heap of ready nodes. On 3,000-node workflows it takes 2.9 ms against the reference's 4,319 ms.

The property that compares the two executors found a real bug in the rewrite: a node waiting out a retry backoff could be started early by an unrelated event, then again by its timer. The fix is covered by a mutant below.


| Finding | Value | Detail |
|---|---|---|
| Incremental executor exponent | `0.96` |  |
| Reference executor exponent | `2.67` |  |


*Code:* `engine/src/graph.ts:kahn`, `engine/src/executor.ts:simulate`, `engine/src/reference-executor.ts:simulateByRescan`


> **Phase gate: PASSED.** Both executors present; the reference is the oracle for the fast one.


## 5. Evaluation

*Objective:* Show the engine is correct, that the tests could have caught it being wrong, and how it scales.


**Correctness.** 77,531 property checks over 5,000 random graphs and workflows, with 0 failures, plus 11 tests in all.

**The tests can fail.** 10 of 10 planted bugs were caught, 8 in behaviour and 2 in the types.

**Scaling.** Kahn's algorithm orders a 1,000,000-node graph in 37 ms with a scaling exponent of 1.06; the naive repeated scan's exponent is 2.10, and at 8,000 nodes it takes 172 ms against 0.18 ms.


| Finding | Value | Detail |
|---|---|---|
| Property checks | `77,531` |  |
| Mutants caught | `10 of 10` |  |


*Code:* `engine/src/properties.ts`, `ml/train.py:mutate`, `engine/bench.ts`


> **Phase gate: PASSED.** Tests, type tests, properties and mutation testing all executed and recorded.


## 6. Deployment

*Objective:* Run the same engine in the browser.


The client imports the engine's TypeScript directly, so the canvas, the run replay and the cycle demonstration use exactly the code the tests and benchmarks measured. The API serves the recorded evidence.


| Finding | Value | Detail |
|---|---|---|
| Serving port | `8,009` |  |


*Code:* `client/src/App.tsx`, `server/main.py`


> **Phase gate: PASSED.** One engine source shared by tests, benchmarks and the UI.
