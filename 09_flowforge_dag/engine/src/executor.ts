/**
 * The FlowForge executor, as a discrete-event simulation.
 *
 * Work is simulated -- each node declares a duration and a failure rate -- but
 * the scheduling is real: nodes start only when their inputs allow, no more
 * than `concurrency` run at once, failures are retried with exponential
 * backoff, and a node whose inputs failed, were skipped or took the other
 * branch of a condition is skipped. Time is virtual, so a run is deterministic
 * for a given seed and can be replayed at any speed.
 *
 * Scheduling is incremental. Each node keeps counts of its inputs that are
 * still open, usable or dead, updated only when one of those inputs finishes,
 * and ready nodes wait in a heap ordered by topological position. A run costs
 * O((V + E) log V) rather than rescanning every node after every event.
 */

import { kahn, validate } from './graph.ts';
import { MinHeap } from './heap.ts';
import { mulberry32 } from './random.ts';
import type { GraphError, NodeId, NodeState, Result, RunEvent, RunId, Workflow, WorkflowNode } from './types.ts';
import { assertNever, err, ok } from './types.ts';

export interface RunOptions {
  run: RunId;
  concurrency: number;
  seed: number;
  backoffMs?: number;
}

export interface RunSummary {
  events: RunEvent[];
  states: Record<string, NodeState>;
  outputs: Record<string, number>;
  makespanMs: number;
  serialMs: number;
  peakConcurrency: number;
}

interface Running { index: number; attempt: number; endsAt: number; fails: boolean }

export function simulate(wf: Workflow, opts: RunOptions): Result<RunSummary, GraphError> {
  const valid = validate(wf);
  if (!valid.ok) return err(valid.error);
  const g = valid.value;
  const sorted = kahn(g);
  if (!sorted.ok) throw new Error('validate() accepted a graph that kahn() rejected');

  const n = wf.nodes.length;
  const rand = mulberry32(opts.seed);
  const backoff = opts.backoffMs ?? 200;
  const index = new Map(wf.nodes.map((node, i) => [node.id, i]));
  const rank = new Int32Array(n);
  sorted.value.forEach((v, pos) => { rank[v] = pos; });

  const inputs: { from: number; branch?: 'true' | 'false' }[][] = Array.from({ length: n }, () => []);
  const children: { to: number; branch?: 'true' | 'false' }[][] = Array.from({ length: n }, () => []);
  for (const e of wf.edges) {
    const u = index.get(e.from)!;
    const v = index.get(e.to)!;
    inputs[v].push({ from: u, branch: e.branch });
    children[u].push({ to: v, branch: e.branch });
  }

  const state: NodeState[] = new Array(n).fill('pending');
  const open = Int32Array.from(inputs, (ins) => ins.length);
  const live = new Int32Array(n);
  const deadReason: (string | null)[] = new Array(n).fill(null);
  const output = new Map<number, number>();
  const outcome = new Map<number, boolean>();
  const attempts = new Int32Array(n);
  const isAny = (v: number) => wf.nodes[v].config.kind === 'join' && (wf.nodes[v].config as { mode: string }).mode === 'any';

  const ready = new MinHeap<number>((a, b) => rank[a] < rank[b]);
  const running = new MinHeap<Running>((a, b) => a.endsAt < b.endsAt || (a.endsAt === b.endsAt && a.index < b.index));
  const retries = new MinHeap<{ at: number; index: number }>((a, b) => a.at < b.at || (a.at === b.at && rank[a.index] < rank[b.index]));
  const queued = new Uint8Array(n);
  // Set while a failed node waits out its backoff; only the retry timer may start it again.
  const backingOff = new Uint8Array(n);
  const events: RunEvent[] = [];
  let clock = 0;
  let active = 0;
  let peak = 0;
  let serial = 0;

  const id = (v: number): NodeId => wf.nodes[v].id;
  events.push({ type: 'run-started', run: opts.run, at: 0, order: Array.from(sorted.value, (v) => id(v)) });

  const enqueue = (v: number) => {
    if (!queued[v]) { queued[v] = 1; ready.push(v); }
  };

  /** Decide a pending node once its input counts change: ready, skipped, or still waiting. */
  const review = (v: number) => {
    if (state[v] !== 'pending' || queued[v] || backingOff[v]) return;
    if (isAny(v)) {
      if (live[v] > 0 || inputs[v].length === 0) enqueue(v);
      else if (open[v] === 0) skip(v, 'no input succeeded');
      return;
    }
    if (deadReason[v] !== null) skip(v, deadReason[v]!);
    else if (open[v] === 0) enqueue(v);
  };

  /** Tell a finished node's children whether the edge to them carries a usable result. */
  const finish = (u: number) => {
    for (const { to, branch } of children[u]) {
      open[to]--;
      const usable = state[u] === 'succeeded' && (branch === undefined || String(outcome.get(u)) === branch);
      if (usable) live[to]++;
      else if (deadReason[to] === null) deadReason[to] = `input ${id(u)} did not succeed${branch ? ' or took the other branch' : ''}`;
      review(to);
    }
  };

  function skip(v: number, reason: string) {
    state[v] = 'skipped';
    events.push({ type: 'node-skipped', run: opts.run, at: clock, node: id(v), reason });
    finish(v);
  }

  const produce = (v: number): number | undefined => {
    const c = wf.nodes[v].config;
    switch (c.kind) {
      case 'evaluate': return Math.round((0.6 + 0.35 * rand()) * 1000) / 1000;
      case 'condition': {
        let best = Number.NEGATIVE_INFINITY;
        for (const { from } of inputs[v]) if (output.has(from)) best = Math.max(best, output.get(from)!);
        const value = best === Number.NEGATIVE_INFINITY ? Number.NaN : best;
        outcome.set(v, value >= c.atLeast);
        return value;
      }
      case 'source': case 'transform': case 'train': case 'join': case 'action': return undefined;
      default: return assertNever(c);
    }
  };

  const startReady = () => {
    while (active < opts.concurrency && ready.size > 0) {
      const v = ready.pop()!;
      queued[v] = 0;
      const node: WorkflowNode = wf.nodes[v];
      const attempt = ++attempts[v];
      state[v] = 'running';
      active++;
      running.push({ index: v, attempt, endsAt: clock + node.durationMs, fails: rand() < (node.failureRate ?? 0) });
      serial += node.durationMs;
      peak = Math.max(peak, active);
      events.push({ type: 'node-started', run: opts.run, at: clock, node: node.id, attempt, running: active });
    }
  };

  for (let v = 0; v < n; v++) review(v);
  startReady();
  while (running.size > 0 || retries.size > 0) {
    const nextFinish = running.peek()?.endsAt ?? Infinity;
    const nextRetry = retries.peek()?.at ?? Infinity;
    clock = Math.min(nextFinish, nextRetry);
    while (running.size > 0 && running.peek()!.endsAt === clock) {
      const r = running.pop()!;
      active--;
      const node = wf.nodes[r.index];
      if (!r.fails) {
        state[r.index] = 'succeeded';
        const out = produce(r.index);
        if (out !== undefined) output.set(r.index, out);
        events.push({ type: 'node-succeeded', run: opts.run, at: clock, node: node.id, attempt: r.attempt, output: out });
        finish(r.index);
      } else if (r.attempt <= (node.retries ?? 0)) {
        const delay = backoff * 2 ** (r.attempt - 1);
        state[r.index] = 'pending';
        backingOff[r.index] = 1;
        retries.push({ at: clock + delay, index: r.index });
        events.push({ type: 'node-retrying', run: opts.run, at: clock, node: node.id, attempt: r.attempt, delayMs: delay });
      } else {
        state[r.index] = 'failed';
        events.push({ type: 'node-failed', run: opts.run, at: clock, node: node.id, attempts: r.attempt });
        finish(r.index);
      }
    }
    while (retries.size > 0 && retries.peek()!.at <= clock) {
      const v = retries.pop()!.index;
      backingOff[v] = 0;
      enqueue(v);
    }
    startReady();
  }

  const count = (s: NodeState) => state.filter((x) => x === s).length;
  events.push({ type: 'run-finished', run: opts.run, at: clock, succeeded: count('succeeded'),
                failed: count('failed'), skipped: count('skipped') });
  return ok({
    events,
    states: Object.fromEntries(state.map((s, v) => [id(v), s])),
    outputs: Object.fromEntries([...output].map(([v, o]) => [id(v), o])),
    makespanMs: clock, serialMs: serial, peakConcurrency: peak,
  });
}
