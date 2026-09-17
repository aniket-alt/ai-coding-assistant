/**
 * The first executor, kept as a reference.
 *
 * After every event it rescans every node to find what can start or must be
 * skipped. That is easy to read and easy to trust, and quadratic in the number
 * of nodes. The incremental executor in executor.ts must produce exactly the
 * same event stream; the property tests compare them on random workflows, and
 * the benchmark measures what the rewrite bought.
 */

import { kahn, validate } from './graph.ts';
import { mulberry32 } from './random.ts';
import type { RunOptions, RunSummary } from './executor.ts';
import type { GraphError, NodeId, NodeState, Result, RunEvent, Workflow, WorkflowNode } from './types.ts';
import { assertNever, err, ok } from './types.ts';

interface Running { node: WorkflowNode; index: number; attempt: number; endsAt: number; fails: boolean }

export function simulateByRescan(wf: Workflow, opts: RunOptions): Result<RunSummary, GraphError> {
  const valid = validate(wf);
  if (!valid.ok) return err(valid.error);
  const g = valid.value;
  const order = kahn(g);
  if (!order.ok) throw new Error('validate() accepted a graph that kahn() rejected');

  const rand = mulberry32(opts.seed);
  const backoff = opts.backoffMs ?? 200;
  const indexOf = new Map(wf.nodes.map((n, i) => [n.id, i]));
  const inputs = new Map<NodeId, { from: NodeId; branch?: 'true' | 'false' }[]>(wf.nodes.map((n) => [n.id, []]));
  for (const e of wf.edges) inputs.get(e.to)!.push({ from: e.from, branch: e.branch });

  const states = new Map<NodeId, NodeState>(wf.nodes.map((n) => [n.id, 'pending']));
  const outputs = new Map<NodeId, number>();
  const conditionOutcome = new Map<NodeId, boolean>();
  const attempts = new Map<NodeId, number>();
  const retryAt = new Map<NodeId, number>();
  const firstDead = new Map<NodeId, string>();
  const running: Running[] = [];
  const events: RunEvent[] = [];
  let clock = 0;
  let peak = 0;
  let serial = 0;

  events.push({ type: 'run-started', run: opts.run, at: 0, order: Array.from(order.value, (i) => g.ids[i]) });

  const readiness = (node: WorkflowNode): 'ready' | 'wait' | { skip: string } => {
    const ins = inputs.get(node.id)!;
    let open = 0;
    let live = 0;
    for (const { from, branch } of ins) {
      const s = states.get(from)!;
      if (s === 'pending' || s === 'running') open++;
      else if (s === 'succeeded' && (branch === undefined || String(conditionOutcome.get(from)) === branch)) live++;
    }
    if (node.config.kind === 'join' && node.config.mode === 'any') {
      if (live > 0 || ins.length === 0) return 'ready';
      return open === 0 ? { skip: 'no input succeeded' } : 'wait';
    }
    const dead = firstDead.get(node.id);
    if (dead !== undefined) return { skip: dead };
    return open === 0 ? 'ready' : 'wait';
  };

  /** Record, in finishing order, the first input of each child that became unusable. */
  const noteFinished = (id: NodeId) => {
    for (const e of wf.edges) {
      if (e.from !== id || firstDead.has(e.to)) continue;
      const s = states.get(id)!;
      const usable = s === 'succeeded' && (e.branch === undefined || String(conditionOutcome.get(id)) === e.branch);
      if (!usable) firstDead.set(e.to, `input ${id} did not succeed${e.branch ? ' or took the other branch' : ''}`);
    }
  };

  const outputOf = (node: WorkflowNode): number | undefined => {
    const c = node.config;
    switch (c.kind) {
      case 'evaluate': return Math.round((0.6 + 0.35 * rand()) * 1000) / 1000;
      case 'condition': {
        let best = Number.NEGATIVE_INFINITY;
        for (const i of inputs.get(node.id)!) if (outputs.has(i.from)) best = Math.max(best, outputs.get(i.from)!);
        const value = best === Number.NEGATIVE_INFINITY ? Number.NaN : best;
        conditionOutcome.set(node.id, value >= c.atLeast);
        return value;
      }
      case 'source': case 'transform': case 'train': case 'join': case 'action': return undefined;
      default: return assertNever(c);
    }
  };

  const settleSkips = () => {
    let changed = true;
    while (changed) {
      changed = false;
      for (const n of wf.nodes) {
        if (states.get(n.id) !== 'pending' || retryAt.has(n.id)) continue;
        const r = readiness(n);
        if (typeof r === 'object') {
          states.set(n.id, 'skipped');
          events.push({ type: 'node-skipped', run: opts.run, at: clock, node: n.id, reason: r.skip });
          noteFinished(n.id);
          changed = true;
        }
      }
    }
  };

  const startReady = () => {
    const candidates = wf.nodes
      .filter((n) => states.get(n.id) === 'pending' && readiness(n) === 'ready' && (retryAt.get(n.id) ?? -1) <= clock)
      .sort((a, b) => order.value.indexOf(indexOf.get(a.id)!) - order.value.indexOf(indexOf.get(b.id)!));
    for (const n of candidates) {
      if (running.length >= opts.concurrency) return;
      retryAt.delete(n.id);
      const attempt = (attempts.get(n.id) ?? 0) + 1;
      attempts.set(n.id, attempt);
      states.set(n.id, 'running');
      running.push({ node: n, index: indexOf.get(n.id)!, attempt, endsAt: clock + n.durationMs, fails: rand() < (n.failureRate ?? 0) });
      serial += n.durationMs;
      peak = Math.max(peak, running.length);
      events.push({ type: 'node-started', run: opts.run, at: clock, node: n.id, attempt, running: running.length });
    }
  };

  settleSkips();
  startReady();
  while (running.length > 0 || [...retryAt.values()].some((t) => t > clock)) {
    const nextFinish = running.length ? Math.min(...running.map((r) => r.endsAt)) : Infinity;
    // Only retries still in the future; a due retry waiting for a free slot must not pull the clock back.
    const nextRetry = Math.min(...[...retryAt.values()].filter((t) => t > clock), Infinity);
    clock = Math.min(nextFinish, nextRetry);
    const done = running.filter((r) => r.endsAt === clock).sort((a, b) => a.index - b.index);
    for (const r of done) {
      running.splice(running.indexOf(r), 1);
      if (!r.fails) {
        states.set(r.node.id, 'succeeded');
        const out = outputOf(r.node);
        if (out !== undefined) outputs.set(r.node.id, out);
        events.push({ type: 'node-succeeded', run: opts.run, at: clock, node: r.node.id, attempt: r.attempt, output: out });
        noteFinished(r.node.id);
      } else if (r.attempt <= (r.node.retries ?? 0)) {
        const delay = backoff * 2 ** (r.attempt - 1);
        states.set(r.node.id, 'pending');
        retryAt.set(r.node.id, clock + delay);
        events.push({ type: 'node-retrying', run: opts.run, at: clock, node: r.node.id, attempt: r.attempt, delayMs: delay });
      } else {
        states.set(r.node.id, 'failed');
        events.push({ type: 'node-failed', run: opts.run, at: clock, node: r.node.id, attempts: r.attempt });
        noteFinished(r.node.id);
      }
    }
    settleSkips();
    startReady();
  }

  const count = (s: NodeState) => [...states.values()].filter((v) => v === s).length;
  events.push({ type: 'run-finished', run: opts.run, at: clock, succeeded: count('succeeded'),
                failed: count('failed'), skipped: count('skipped') });
  return ok({ events, states: Object.fromEntries(states), outputs: Object.fromEntries(outputs),
              makespanMs: clock, serialMs: serial, peakConcurrency: peak });
}
