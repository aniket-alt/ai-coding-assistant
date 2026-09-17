/**
 * Property checks: the engine against independent references, on many random graphs.
 *
 * The same functions back the unit tests (which require zero failures) and the
 * training run (which records how many graphs and events were checked), so the
 * numbers reported are the checks that actually ran.
 */

import { simulate } from './executor.ts';
import type { IndexedGraph } from './graph.ts';
import { dfsOrder, kahn, levels, naiveOrder, respectsEdges } from './graph.ts';
import { mulberry32, randomCyclic, randomDag } from './random.ts';
import { simulateByRescan } from './reference-executor.ts';
import type { Edge, NodeConfig, RunEvent, Workflow, WorkflowNode } from './types.ts';
import { nodeId, runId, workflowId } from './types.ts';

export interface PropertyReport {
  name: string;
  cases: number;
  checks: number;
  failures: string[];
}

const hasEdge = (g: IndexedGraph, u: number, v: number): boolean => {
  for (let k = g.offsets[u]; k < g.offsets[u + 1]; k++) if (g.targets[k] === v) return true;
  return false;
};

export function orderingProperties(seed: number, cases: number): PropertyReport {
  const rand = mulberry32(seed);
  const report: PropertyReport = { name: 'orderings respect every edge', cases, checks: 0, failures: [] };
  for (let c = 0; c < cases; c++) {
    const g = randomDag(1 + Math.floor(rand() * 80), rand() * 3, rand);
    const k = kahn(g);
    const d = dfsOrder(g);
    const nv = naiveOrder(g);
    report.checks += 3;
    if (!k.ok || !respectsEdges(g, k.value)) report.failures.push(`case ${c}: Kahn order invalid`);
    if (!d || !respectsEdges(g, d)) report.failures.push(`case ${c}: DFS order invalid`);
    if (!nv || !respectsEdges(g, nv)) report.failures.push(`case ${c}: naive order invalid`);
  }
  return report;
}

export function cycleProperties(seed: number, cases: number): PropertyReport {
  const rand = mulberry32(seed);
  const report: PropertyReport = { name: 'every cycle is detected and a real cycle is reported', cases, checks: 0, failures: [] };
  for (let c = 0; c < cases; c++) {
    const g = randomCyclic(2 + Math.floor(rand() * 80), rand() * 3, rand);
    const k = kahn(g);
    report.checks += 3;
    if (k.ok) {
      report.failures.push(`case ${c}: Kahn missed a cycle`);
    } else {
      const path = k.error;
      const closed = path.length >= 3 && path[0] === path[path.length - 1];
      const interior = new Set(path.slice(0, -1));
      const edgesExist = path.slice(0, -1).every((u, i) => hasEdge(g, u, path[i + 1]));
      if (!closed || interior.size !== path.length - 1 || !edgesExist) report.failures.push(`case ${c}: reported path ${path.join('>')} is not a cycle`);
    }
    if (dfsOrder(g) !== null) report.failures.push(`case ${c}: DFS missed a cycle`);
    if (naiveOrder(g) !== null) report.failures.push(`case ${c}: naive method missed a cycle`);
  }
  return report;
}

export function levelProperties(seed: number, cases: number): PropertyReport {
  const rand = mulberry32(seed);
  const report: PropertyReport = { name: 'levels equal the longest path from a source', cases, checks: 0, failures: [] };
  for (let c = 0; c < cases; c++) {
    const g = randomDag(1 + Math.floor(rand() * 60), rand() * 3, rand);
    const k = kahn(g);
    if (!k.ok) { report.failures.push(`case ${c}: unexpected cycle`); continue; }
    const lv = levels(g, k.value);
    // Reference: memoised recursion over parents, sharing no code with levels().
    const parents: number[][] = Array.from({ length: g.n }, () => []);
    for (let u = 0; u < g.n; u++) for (let e = g.offsets[u]; e < g.offsets[u + 1]; e++) parents[g.targets[e]].push(u);
    const memo = new Map<number, number>();
    const depth = (v: number): number => {
      if (!memo.has(v)) memo.set(v, parents[v].length ? 1 + Math.max(...parents[v].map(depth)) : 0);
      return memo.get(v)!;
    };
    for (let v = 0; v < g.n; v++) {
      report.checks++;
      if (lv[v] !== depth(v)) report.failures.push(`case ${c}: node ${v} level ${lv[v]}, longest path ${depth(v)}`);
    }
  }
  return report;
}

const KINDS: NodeConfig[] = [
  { kind: 'transform', operation: 'scale' }, { kind: 'train', model: 'LightGBM' },
  { kind: 'evaluate', metric: 'AUC' }, { kind: 'join', mode: 'all' }, { kind: 'join', mode: 'any' },
  { kind: 'action', action: 'report' },
];

export function randomWorkflow(g: IndexedGraph, rand: () => number): Workflow {
  const nodes: WorkflowNode[] = Array.from({ length: g.n }, (_, i) => ({
    id: nodeId(`n${i}`), title: `Node ${i}`,
    config: rand() < 0.15 ? { kind: 'condition', metric: 'AUC', atLeast: 0.75 } : KINDS[Math.floor(rand() * KINDS.length)],
    durationMs: 50 + Math.floor(rand() * 20) * 50, failureRate: rand() < 0.3 ? rand() * 0.5 : 0,
    retries: Math.floor(rand() * 3),
  }));
  const edges: Edge[] = [];
  for (let u = 0; u < g.n; u++) {
    for (let k = g.offsets[u]; k < g.offsets[u + 1]; k++) {
      const branch = nodes[u].config.kind === 'condition' ? (rand() < 0.5 ? 'true' : 'false') : undefined;
      edges.push({ from: nodes[u].id, to: nodes[g.targets[k]].id, branch });
    }
  }
  return { id: workflowId('random'), name: 'random', nodes, edges };
}

export function executorProperties(seed: number, cases: number): PropertyReport {
  const rand = mulberry32(seed);
  const report: PropertyReport = { name: 'executor honours dependencies, branches, limits and determinism', cases, checks: 0, failures: [] };
  for (let c = 0; c < cases; c++) {
    const g = randomDag(1 + Math.floor(rand() * 40), rand() * 2.5, rand);
    const wf = randomWorkflow(g, rand);
    const concurrency = 1 + Math.floor(rand() * 4);
    const opts = { run: runId(`r${c}`), concurrency, seed: c };
    const a = simulate(wf, opts);
    const b = simulate(wf, opts);
    if (!a.ok || !b.ok) { report.failures.push(`case ${c}: valid workflow rejected`); continue; }
    report.checks++;
    if (JSON.stringify(a.value.events) !== JSON.stringify(b.value.events)) report.failures.push(`case ${c}: same seed, different run`);

    const byId = new Map(wf.nodes.map((n) => [n.id, n]));
    const finished = new Map<string, { state: string; at: number; output?: number }>();
    let running = 0;
    for (const ev of a.value.events) {
      if (ev.type === 'node-started') {
        running++;
        report.checks += 2;
        if (running > concurrency) report.failures.push(`case ${c}: ${running} running with limit ${concurrency}`);
        const node = byId.get(ev.node)!;
        const ins = wf.edges.filter((e) => e.to === ev.node);
        const live = ins.map((e) => {
          const f = finished.get(e.from);
          if (!f || f.at > ev.at || f.state !== 'succeeded') return false;
          if (e.branch === undefined) return true;
          const cond = byId.get(e.from)!.config;
          const outcome = cond.kind === 'condition' && f.output !== undefined && f.output >= cond.atLeast;
          return String(outcome) === e.branch;
        });
        const allowed = node.config.kind === 'join' && node.config.mode === 'any'
          ? ins.length === 0 || live.some(Boolean)
          : live.every(Boolean);
        if (!allowed) report.failures.push(`case ${c}: ${ev.node} started without its inputs`);
      } else if (ev.type === 'node-succeeded') {
        running--;
        finished.set(ev.node, { state: 'succeeded', at: ev.at, output: ev.output });
      } else if (ev.type === 'node-failed' || ev.type === 'node-retrying') {
        running--;
        if (ev.type === 'node-failed') finished.set(ev.node, { state: 'failed', at: ev.at });
      }
    }
    report.checks += 2;
    if (Object.values(a.value.states).some((s) => s === 'pending' || s === 'running')) report.failures.push(`case ${c}: run ended with unfinished nodes`);
    if (a.value.peakConcurrency > concurrency) report.failures.push(`case ${c}: peak concurrency above limit`);
  }
  return report;
}

/**
 * The incremental executor against the rescanning reference: the same starts,
 * finishes and retries at the same virtual times, the same final states and
 * outputs. Skip events may be emitted in a different order within one instant,
 * so they are compared as a set of (node, time); when several inputs die at the
 * same instant, which one the skip message names is arbitrary and not compared.
 */
export function differentialProperties(seed: number, cases: number): PropertyReport {
  const rand = mulberry32(seed);
  const report: PropertyReport = { name: 'incremental executor matches the rescanning reference', cases, checks: 0, failures: [] };
  const canonical = (events: RunEvent[]) => ({
    timeline: events.filter((e) => e.type !== 'node-skipped').map((e) => JSON.stringify(e)),
    skipped: events.filter((e) => e.type === 'node-skipped')
      .map((e) => JSON.stringify({ ...e, reason: undefined })).sort(),
  });
  for (let c = 0; c < cases; c++) {
    const g = randomDag(1 + Math.floor(rand() * 40), rand() * 2.5, rand);
    const wf = randomWorkflow(g, rand);
    const opts = { run: runId(`d${c}`), concurrency: 1 + Math.floor(rand() * 4), seed: c };
    const fast = simulate(wf, opts);
    const ref = simulateByRescan(wf, opts);
    report.checks += 3;
    if (!fast.ok || !ref.ok) { report.failures.push(`case ${c}: rejected`); continue; }
    if (JSON.stringify(fast.value.states) !== JSON.stringify(ref.value.states)) report.failures.push(`case ${c}: final states differ`);
    if (fast.value.makespanMs !== ref.value.makespanMs) report.failures.push(`case ${c}: makespan ${fast.value.makespanMs} vs ${ref.value.makespanMs}`);
    const a = canonical(fast.value.events);
    const b = canonical(ref.value.events);
    if (JSON.stringify(a) !== JSON.stringify(b)) report.failures.push(`case ${c}: event streams differ`);
  }
  return report;
}

export function allProperties(seed = 20260917, cases = 1000): PropertyReport[] {
  return [
    orderingProperties(seed, cases),
    cycleProperties(seed + 1, cases),
    levelProperties(seed + 2, cases),
    executorProperties(seed + 3, cases),
    differentialProperties(seed + 4, cases),
  ];
}
