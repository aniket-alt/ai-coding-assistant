/**
 * Benchmarks and property counts, printed as JSON for ml/train.py to record.
 *
 *     node 09_flowforge_dag/engine/bench.ts
 *
 * Each timing is the median of several runs on the same random graphs, after a
 * warm-up, so the first call's compilation is not counted.
 */

import { simulate } from './src/executor.ts';
import { simulateByRescan } from './src/reference-executor.ts';
import { dfsOrder, kahn, levels, naiveOrder } from './src/graph.ts';
import { allProperties, randomWorkflow } from './src/properties.ts';
import { mulberry32, randomDag } from './src/random.ts';
import { runId } from './src/types.ts';

const EDGES_PER_NODE = 2;
const REPEATS = 5;

function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
}

function time(fn: () => unknown): number {
  const runs: number[] = [];
  fn();
  for (let i = 0; i < REPEATS; i++) {
    const t = performance.now();
    fn();
    runs.push(performance.now() - t);
  }
  return median(runs);
}

/** Least-squares slope of log(time) on log(size): about 1 for linear growth, 2 for quadratic. */
function slope(points: { n: number; ms: number }[]): number {
  const xs = points.map((p) => Math.log(p.n));
  const ys = points.map((p) => Math.log(p.ms));
  const mx = xs.reduce((a, b) => a + b, 0) / xs.length;
  const my = ys.reduce((a, b) => a + b, 0) / ys.length;
  const num = xs.reduce((a, x, i) => a + (x - mx) * (ys[i] - my), 0);
  const den = xs.reduce((a, x) => a + (x - mx) ** 2, 0);
  return num / den;
}

const algorithms = [
  { name: "Kahn's algorithm", sizes: [10_000, 30_000, 100_000, 300_000, 1_000_000], run: (g: ReturnType<typeof randomDag>) => kahn(g) },
  { name: 'Depth-first search', sizes: [10_000, 30_000, 100_000, 300_000, 1_000_000], run: (g: ReturnType<typeof randomDag>) => dfsOrder(g) },
  { name: 'Naive repeated scan', sizes: [1_000, 2_000, 4_000, 8_000], run: (g: ReturnType<typeof randomDag>) => naiveOrder(g) },
];

const scaling = algorithms.map((alg) => {
  const points = alg.sizes.map((n) => {
    const g = randomDag(n, EDGES_PER_NODE, mulberry32(n));
    const ms = time(() => alg.run(g));
    return { n, edges: g.targets.length, ms, ms_per_1k_nodes: (ms / n) * 1000 };
  });
  return { algorithm: alg.name, points, slope: slope(points) };
});

// The three methods at one size they can all reach, for a like-for-like comparison.
const common = randomDag(8_000, EDGES_PER_NODE, mulberry32(8_000));
const headToHead = algorithms.map((alg) => ({ algorithm: alg.name, n: 8_000, ms: time(() => alg.run(common)) }));

/** Both executors on the same random workflows: the incremental one to 30,000 nodes, the rescanning reference to 3,000. */
const executorSizes = [300, 1_000, 3_000, 10_000, 30_000];
const workflows = new Map(executorSizes.map((n) => {
  const rand = mulberry32(n + 7);
  return [n, randomWorkflow(randomDag(n, 1.5, rand), rand)] as const;
}));
const executors = [
  { name: 'Incremental executor', sizes: executorSizes, run: simulate },
  { name: 'Rescanning reference', sizes: [300, 1_000, 3_000], run: simulateByRescan },
].map((ex) => {
  const points = ex.sizes.map((n) => {
    const wf = workflows.get(n)!;
    let events = 0;
    const ms = time(() => {
      const r = ex.run(wf, { run: runId('bench'), concurrency: 8, seed: 1 });
      if (r.ok) events = r.value.events.length;
    });
    return { n, events, ms, ms_per_1k_nodes: (ms / n) * 1000 };
  });
  return { executor: ex.name, points, slope: slope(points) };
});

const big = randomDag(1_000_000, EDGES_PER_NODE, mulberry32(1));
const order = kahn(big);
const depth = order.ok ? levels(big, order.value).reduce((a, b) => Math.max(a, b), 0) + 1 : 0;

const t = performance.now();
const properties = allProperties(20260917, 1000);
const propertySeconds = (performance.now() - t) / 1000;

console.log(JSON.stringify({
  node: process.version, edges_per_node: EDGES_PER_NODE, repeats: REPEATS,
  scaling, head_to_head: headToHead, executors, million_node_levels: depth,
  properties: properties.map((p) => ({ name: p.name, cases: p.cases, checks: p.checks, failures: p.failures.slice(0, 5), failure_count: p.failures.length })),
  property_seconds: propertySeconds,
}));
