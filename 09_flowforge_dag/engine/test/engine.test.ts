/**
 * FlowForge engine tests. Run with: node --test 09_flowforge_dag/engine/test
 */

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { simulate } from '../src/executor.ts';
import { validate } from '../src/graph.ts';
import { allProperties } from '../src/properties.ts';
import type { Workflow, WorkflowNode } from '../src/types.ts';
import { describe as describeNode, nodeId, runId, workflowId } from '../src/types.ts';

const node = (id: string, config: WorkflowNode['config'], durationMs = 100, extra: Partial<WorkflowNode> = {}): WorkflowNode =>
  ({ id: nodeId(id), title: id, config, durationMs, ...extra });

for (const report of allProperties(20260917, 300)) {
  test(`property: ${report.name}`, () => {
    assert.ok(report.checks > 0);
    assert.deepEqual(report.failures, []);
  });
}

test('validate rejects duplicates, self-loops and unknown nodes', () => {
  const base = { id: workflowId('w'), name: 'w' };
  const a = node('a', { kind: 'train', model: 'm' });
  assert.equal((validate({ ...base, nodes: [a, a], edges: [] }) as { ok: false; error: { kind: string } }).error.kind, 'duplicate-node');
  assert.equal((validate({ ...base, nodes: [a], edges: [{ from: a.id, to: a.id }] }) as { ok: false; error: { kind: string } }).error.kind, 'self-loop');
  assert.equal((validate({ ...base, nodes: [a], edges: [{ from: a.id, to: nodeId('ghost') }] }) as { ok: false; error: { kind: string } }).error.kind, 'unknown-node');
});

test('a cycle is reported as a concrete path', () => {
  const [a, b, c] = ['a', 'b', 'c'].map((id) => node(id, { kind: 'transform', operation: 'scale' }));
  const wf: Workflow = { id: workflowId('w'), name: 'w', nodes: [a, b, c],
    edges: [{ from: a.id, to: b.id }, { from: b.id, to: c.id }, { from: c.id, to: a.id }] };
  const r = validate(wf);
  assert.equal(r.ok, false);
  if (!r.ok && r.error.kind === 'cycle') {
    assert.equal(r.error.path[0], r.error.path[r.error.path.length - 1]);
    assert.equal(r.error.path.length, 4);
  } else {
    assert.fail('expected a cycle error');
  }
});

test('a condition skips the branch it did not take, and join-any still runs', () => {
  const wf: Workflow = {
    id: workflowId('w'), name: 'w',
    nodes: [
      node('eval', { kind: 'evaluate', metric: 'AUC' }),
      node('gate', { kind: 'condition', metric: 'AUC', atLeast: 0 }),
      node('deploy', { kind: 'action', action: 'deploy' }),
      node('alert', { kind: 'action', action: 'alert' }),
      node('report', { kind: 'join', mode: 'any' }),
    ],
    edges: [
      { from: nodeId('eval'), to: nodeId('gate') },
      { from: nodeId('gate'), to: nodeId('deploy'), branch: 'true' },
      { from: nodeId('gate'), to: nodeId('alert'), branch: 'false' },
      { from: nodeId('deploy'), to: nodeId('report') },
      { from: nodeId('alert'), to: nodeId('report') },
    ],
  };
  const r = simulate(wf, { run: runId('r'), concurrency: 2, seed: 1 });
  assert.ok(r.ok);
  if (r.ok) {
    assert.equal(r.value.states.deploy, 'succeeded'); // any AUC is at least 0
    assert.equal(r.value.states.alert, 'skipped');
    assert.equal(r.value.states.report, 'succeeded');
  }
});

test('failures are retried with exponential backoff, then fail downstream', () => {
  const wf: Workflow = {
    id: workflowId('w'), name: 'w',
    nodes: [node('flaky', { kind: 'train', model: 'm' }, 100, { failureRate: 1, retries: 2 }),
            node('after', { kind: 'evaluate', metric: 'AUC' })],
    edges: [{ from: nodeId('flaky'), to: nodeId('after') }],
  };
  const r = simulate(wf, { run: runId('r'), concurrency: 1, seed: 3, backoffMs: 100 });
  assert.ok(r.ok);
  if (r.ok) {
    const retries = r.value.events.filter((e) => e.type === 'node-retrying');
    assert.deepEqual(retries.map((e) => (e.type === 'node-retrying' ? e.delayMs : 0)), [100, 200]);
    assert.equal(r.value.states.flaky, 'failed');
    assert.equal(r.value.states.after, 'skipped');
  }
});

test('parallel levels shorten the run against running one at a time', () => {
  const parts = ['a', 'b', 'c', 'd'].map((id) => node(id, { kind: 'transform', operation: 'encode' }, 400));
  const wf: Workflow = { id: workflowId('w'), name: 'w', nodes: [...parts, node('join', { kind: 'join', mode: 'all' }, 100)],
    edges: parts.map((p) => ({ from: p.id, to: nodeId('join') })) };
  const one = simulate(wf, { run: runId('1'), concurrency: 1, seed: 0 });
  const four = simulate(wf, { run: runId('4'), concurrency: 4, seed: 0 });
  assert.ok(one.ok && four.ok);
  if (one.ok && four.ok) {
    assert.equal(one.value.makespanMs, 1700);
    assert.equal(four.value.makespanMs, 500);
  }
});

test('every node kind has a description', () => {
  assert.match(describeNode({ kind: 'condition', metric: 'AUC', atLeast: 0.8 }), /AUC ≥ 0.8/);
});
