/**
 * The workflow canvas and run replay -- project-09-specific.
 *
 * Everything shown is computed by the engine itself, imported from
 * ../../engine/src: validation, ordering, levels and the simulated run. The
 * canvas only lays nodes out by the levels the engine returns and colours them
 * by the states its events record.
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import { StatRow, StatTile } from '@dsfs/ui';
import { simulate } from '../../engine/src/executor.ts';
import { kahn, levels, validate } from '../../engine/src/graph.ts';
import type { Edge, NodeId, NodeState, RunEvent, Workflow } from '../../engine/src/types.ts';
import { describe, nodeId, runId, workflowId } from '../../engine/src/types.ts';

const n = nodeId;

export function pipeline(flaky: boolean, threshold: number): Workflow {
  return {
    id: workflowId('churn-retraining'),
    name: 'Nightly churn model retraining',
    nodes: [
      { id: n('load'), title: 'Load calls', config: { kind: 'source', dataset: 'bank-marketing', rows: 41188 }, durationMs: 600 },
      { id: n('impute'), title: 'Impute', config: { kind: 'transform', operation: 'impute' }, durationMs: 400 },
      { id: n('encode'), title: 'Encode', config: { kind: 'transform', operation: 'encode' }, durationMs: 500 },
      { id: n('features'), title: 'Join features', config: { kind: 'transform', operation: 'join-features' }, durationMs: 300 },
      { id: n('train-lgbm'), title: 'Train LightGBM', config: { kind: 'train', model: 'LightGBM' }, durationMs: 1400,
        failureRate: flaky ? 0.9 : 0, retries: 2 },
      { id: n('train-lr'), title: 'Train logistic', config: { kind: 'train', model: 'Logistic regression' }, durationMs: 700 },
      { id: n('eval-lgbm'), title: 'Evaluate LightGBM', config: { kind: 'evaluate', metric: 'AUC' }, durationMs: 300 },
      { id: n('eval-lr'), title: 'Evaluate logistic', config: { kind: 'evaluate', metric: 'AUC' }, durationMs: 300 },
      { id: n('gate'), title: 'Quality gate', config: { kind: 'condition', metric: 'AUC', atLeast: threshold }, durationMs: 100 },
      { id: n('deploy'), title: 'Deploy', config: { kind: 'action', action: 'deploy' }, durationMs: 500 },
      { id: n('alert'), title: 'Alert on-call', config: { kind: 'action', action: 'alert' }, durationMs: 200 },
      { id: n('report'), title: 'Write report', config: { kind: 'join', mode: 'any' }, durationMs: 300 },
    ],
    edges: [
      { from: n('load'), to: n('impute') }, { from: n('load'), to: n('encode') },
      { from: n('impute'), to: n('features') }, { from: n('encode'), to: n('features') },
      { from: n('features'), to: n('train-lgbm') }, { from: n('features'), to: n('train-lr') },
      { from: n('train-lgbm'), to: n('eval-lgbm') }, { from: n('train-lr'), to: n('eval-lr') },
      { from: n('eval-lgbm'), to: n('gate') }, { from: n('eval-lr'), to: n('gate') },
      { from: n('gate'), to: n('deploy'), branch: 'true' }, { from: n('gate'), to: n('alert'), branch: 'false' },
      { from: n('deploy'), to: n('report') }, { from: n('alert'), to: n('report') },
    ],
  };
}

const STATE_COLOUR: Record<NodeState, string> = {
  pending: 'var(--surface-2)',
  running: 'var(--series-4)',
  succeeded: 'var(--status-good)',
  failed: 'var(--status-critical)',
  skipped: 'var(--text-muted)',
};

const COL_W = 190;
const ROW_H = 78;
const NODE_W = 150;
const NODE_H = 48;

function statesAt(wf: Workflow, events: RunEvent[], t: number): Map<NodeId, NodeState> {
  const s = new Map<NodeId, NodeState>(wf.nodes.map((x) => [x.id, 'pending']));
  for (const e of events) {
    if (e.at > t) break;
    switch (e.type) {
      case 'node-started': s.set(e.node, 'running'); break;
      case 'node-retrying': s.set(e.node, 'pending'); break;
      case 'node-succeeded': s.set(e.node, 'succeeded'); break;
      case 'node-failed': s.set(e.node, 'failed'); break;
      case 'node-skipped': s.set(e.node, 'skipped'); break;
      default: break;
    }
  }
  return s;
}

export function WorkflowGraph({ wf, states, highlight, layoutFrom }: {
  wf: Workflow; states?: Map<NodeId, NodeState>; highlight?: Set<string>;
  /** An acyclic workflow with the same nodes to lay out by, when `wf` itself has a cycle. */
  layoutFrom?: Workflow;
}) {
  const layout = useMemo(() => {
    const g = validate(layoutFrom ?? wf);
    if (!g.ok) return null;
    const order = kahn(g.value);
    if (!order.ok) return null;
    const lv = levels(g.value, order.value);
    const rows = new Map<number, number>();
    const pos = new Map<string, { x: number; y: number }>();
    wf.nodes.forEach((node, i) => {
      const r = rows.get(lv[i]) ?? 0;
      rows.set(lv[i], r + 1);
      pos.set(node.id, { x: 16 + lv[i] * COL_W, y: 16 + r * ROW_H });
    });
    const depth = Math.max(...lv) + 1;
    const height = Math.max(...rows.values());
    return { pos, width: depth * COL_W + 16, height: height * ROW_H + 16 };
  }, [wf, layoutFrom]);
  if (!layout) return null;

  const edgePath = (e: Edge) => {
    const a = layout.pos.get(e.from)!;
    const b = layout.pos.get(e.to)!;
    const x1 = a.x + NODE_W, y1 = a.y + NODE_H / 2, x2 = b.x, y2 = b.y + NODE_H / 2;
    if (x2 <= x1) {
      // A back edge: route it below the graph so the cycle is visible.
      const low = layout.height + 34;
      return `M ${x1 - NODE_W / 2} ${a.y + NODE_H} C ${x1 - NODE_W / 2} ${low}, ${x2 + NODE_W / 2} ${low}, ${x2 + NODE_W / 2} ${b.y + NODE_H}`;
    }
    const mid = (x1 + x2) / 2;
    return `M ${x1} ${y1} C ${mid} ${y1}, ${mid} ${y2}, ${x2} ${y2}`;
  };

  return (
    <div>
      <svg viewBox={`0 0 ${layout.width} ${layout.height + 48}`} width="100%" style={{ maxWidth: layout.width }} role="img" aria-label={`Workflow graph: ${wf.name}`}>
        <defs>
          <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--axis)" />
          </marker>
        </defs>
        {wf.edges.map((e) => {
          const hot = highlight?.has(`${e.from}>${e.to}`);
          return (
            <g key={`${e.from}>${e.to}`}>
              <path d={edgePath(e)} fill="none" markerEnd="url(#arrow)"
                    stroke={hot ? 'var(--status-critical)' : 'var(--axis)'} strokeWidth={hot ? 2.5 : 1.2}
                    strokeDasharray={e.branch === 'false' ? '5 4' : undefined} />
              {e.branch && (() => {
                const b = layout.pos.get(e.to)!;
                return <text x={b.x - 30} y={b.y + NODE_H / 2 - 6} fontSize={10} fill="var(--text-muted)">{e.branch}</text>;
              })()}
            </g>
          );
        })}
        {wf.nodes.map((node) => {
          const p = layout.pos.get(node.id)!;
          const state = states?.get(node.id) ?? 'pending';
          const hot = highlight && [...highlight].some((k) => k.split('>').includes(node.id));
          return (
            <g key={node.id} transform={`translate(${p.x}, ${p.y})`}>
              <title>{describe(node.config)}</title>
              <rect width={NODE_W} height={NODE_H} rx={8} fill="var(--surface-1)"
                    stroke={hot ? 'var(--status-critical)' : 'var(--border)'} strokeWidth={hot ? 2 : 1} />
              <rect width={6} height={NODE_H} rx={3} fill={STATE_COLOUR[state]} />
              <text x={14} y={20} fontSize={12} fontWeight={600} fill="var(--text-primary)">{node.title}</text>
              <text x={14} y={36} fontSize={10} fill="var(--text-muted)">{node.config.kind} · {state}</text>
            </g>
          );
        })}
      </svg>
      <div style={{ display: 'flex', gap: 'var(--space-4)', flexWrap: 'wrap', fontSize: 'var(--text-xs)', color: 'var(--text-secondary)' }}>
        {(Object.keys(STATE_COLOUR) as NodeState[]).map((s) => (
          <span key={s}><span style={{ display: 'inline-block', width: 10, height: 10, borderRadius: 2, background: STATE_COLOUR[s], marginRight: 4, border: '1px solid var(--border)' }} />{s}</span>
        ))}
        <span>Columns are the engine's levels: nodes in one column can run in parallel. Dashed edges are a condition's false branch.</span>
      </div>
    </div>
  );
}

export function RunReplay() {
  const [concurrency, setConcurrency] = useState(2);
  const [seed, setSeed] = useState(7);
  const [flaky, setFlaky] = useState(true);
  const [threshold, setThreshold] = useState(0.75);
  const wf = useMemo(() => pipeline(flaky, threshold), [flaky, threshold]);
  const result = useMemo(() => simulate(wf, { run: runId('demo'), concurrency, seed, backoffMs: 400 }), [wf, concurrency, seed]);
  const summary = result.ok ? result.value : null;
  const [t, setT] = useState(Number.POSITIVE_INFINITY);
  const frame = useRef<number | null>(null);

  useEffect(() => { setT(Number.POSITIVE_INFINITY); }, [summary]);
  useEffect(() => () => { if (frame.current) cancelAnimationFrame(frame.current); }, []);

  const replay = () => {
    if (!summary) return;
    if (frame.current) cancelAnimationFrame(frame.current);
    const start = performance.now();
    const tick = (now: number) => {
      const virtual = now - start;
      setT(virtual);
      if (virtual < summary.makespanMs) frame.current = requestAnimationFrame(tick);
    };
    frame.current = requestAnimationFrame(tick);
  };

  if (!summary) return <p>The workflow failed validation.</p>;
  const shown = Math.min(t, summary.makespanMs);
  const states = statesAt(wf, summary.events, shown);
  const log = summary.events.filter((e) => e.at <= shown && e.type !== 'run-started');
  const finished = summary.events[summary.events.length - 1];
  const control: React.CSSProperties = { display: 'block', fontSize: 'var(--text-xs)', color: 'var(--text-muted)', fontWeight: 600 };

  return (
    <div style={{ display: 'grid', gap: 'var(--space-4)' }}>
      <div style={{ display: 'grid', gap: 'var(--space-4)', gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))', alignItems: 'end' }}>
        <label style={control}>CONCURRENCY LIMIT: {concurrency}
          <input type="range" min={1} max={6} value={concurrency} onChange={(e) => setConcurrency(Number(e.target.value))} style={{ width: '100%' }} />
        </label>
        <label style={control}>SEED: {seed}
          <input type="range" min={0} max={40} value={seed} onChange={(e) => setSeed(Number(e.target.value))} style={{ width: '100%' }} />
        </label>
        <label style={control}>QUALITY GATE: AUC ≥ {threshold.toFixed(2)}
          <input type="range" min={0.6} max={0.95} step={0.01} value={threshold} onChange={(e) => setThreshold(Number(e.target.value))} style={{ width: '100%' }} />
        </label>
        <label style={{ ...control, fontWeight: 400, fontSize: 'var(--text-sm)', color: 'var(--text-primary)' }}>
          <input type="checkbox" checked={flaky} onChange={(e) => setFlaky(e.target.checked)} /> LightGBM training is flaky (90% failure, 2 retries)
        </label>
        <button type="button" onClick={replay} style={{ padding: '9px 16px', font: 'inherit', fontWeight: 600, borderRadius: 'var(--radius)',
          border: '1px solid var(--series-1)', background: 'var(--series-1)', color: '#fff', cursor: 'pointer' }}>Replay run</button>
      </div>
      <StatRow>
        <StatTile label="Wall time" value={(summary.makespanMs / 1000).toFixed(1)} unit="s (simulated)"
                  hint={`${(summary.serialMs / 1000).toFixed(1)} s of work in total`} />
        <StatTile label="Peak concurrency" value={summary.peakConcurrency} hint={`limit ${concurrency}`} />
        {finished.type === 'run-finished' && (
          <StatTile label="Outcome" value={`${finished.succeeded} / ${finished.failed} / ${finished.skipped}`}
                    hint="succeeded / failed / skipped" />
        )}
      </StatRow>
      <WorkflowGraph wf={wf} states={states} />
      <div style={{ maxHeight: 260, overflowY: 'auto', fontFamily: 'var(--font-mono, monospace)', fontSize: 12,
        background: 'var(--surface-1)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', padding: 'var(--space-3)' }}>
        {log.map((e, i) => (
          <div key={i} style={{ color: e.type === 'node-failed' ? 'var(--status-critical)' : 'var(--text-secondary)' }}>
            {String(e.at).padStart(5)} ms  {e.type.padEnd(15)} {'node' in e ? e.node : ''}
            {e.type === 'node-started' ? ` attempt ${e.attempt}, ${e.running} running` : ''}
            {e.type === 'node-retrying' ? ` retry in ${e.delayMs} ms` : ''}
            {e.type === 'node-succeeded' && e.output !== undefined ? ` → ${Number.isNaN(e.output) ? 'no input' : e.output}` : ''}
            {e.type === 'node-skipped' ? ` (${e.reason})` : ''}
          </div>
        ))}
      </div>
    </div>
  );
}

export function CycleDemo() {
  const [cyclic, setCyclic] = useState(true);
  const base = useMemo(() => pipeline(false, 0.75), []);
  const back: Edge = { from: n('eval-lgbm'), to: n('features') };
  const wf = useMemo(() => (cyclic ? { ...base, edges: [...base.edges, back] } : base), [cyclic, base]); // eslint-disable-line react-hooks/exhaustive-deps
  const result = validate(wf);
  const path = !result.ok && result.error.kind === 'cycle' ? result.error.path : [];
  const highlight = new Set(path.slice(0, -1).map((id, i) => `${id}>${path[i + 1]}`));

  return (
    <div style={{ display: 'grid', gap: 'var(--space-4)' }}>
      <label style={{ fontSize: 'var(--text-sm)' }}>
        <input type="checkbox" checked={cyclic} onChange={(e) => setCyclic(e.target.checked)} /> Add an edge from “Evaluate LightGBM” back to “Join features”
      </label>
      <div style={{ padding: 'var(--space-3)', borderRadius: 'var(--radius)', border: '1px solid var(--border)',
        background: 'var(--surface-1)', fontSize: 'var(--text-sm)' }}>
        {result.ok
          ? <span><strong style={{ color: 'var(--delta-up)' }}>Valid.</strong> The engine found an order for all {wf.nodes.length} nodes.</span>
          : result.error.kind === 'cycle'
            ? <span><strong style={{ color: 'var(--status-critical)' }}>Rejected: cycle.</strong> {path.join(' → ')}</span>
            : <span><strong style={{ color: 'var(--status-critical)' }}>Rejected:</strong> {result.error.kind}</span>}
      </div>
      <WorkflowGraph wf={wf} highlight={highlight} layoutFrom={base} />
    </div>
  );
}
