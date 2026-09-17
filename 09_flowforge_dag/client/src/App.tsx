/**
 * Project 09 client -- the engine and its evidence.
 *
 * The canvas, replay and cycle views run the engine's own TypeScript in the
 * browser. The evidence and benchmark views read the recorded training run.
 */

import { useState } from 'react';
import {
  AppShell, AuditPanel, Chart, CrispDmWalkthrough, ModelLeaderboard, Panel, ProvenanceBadge,
  StatRow, StatTile, useApi, useAudit, useCrispDm, useMeta, useMetrics, useProvenance,
} from '@dsfs/ui';
import type { Tab } from '@dsfs/ui';
import { CycleDemo, RunReplay } from './Canvas';

const TABS: Tab[] = [
  { id: 'run', label: 'Run a workflow' },
  { id: 'cycle', label: 'Cycle detection' },
  { id: 'evidence', label: 'Evidence' },
  { id: 'benchmarks', label: 'Benchmarks' },
  { id: 'types', label: 'Type design' },
  { id: 'crispdm', label: 'CRISP-DM' },
  { id: 'audit', label: 'Audit' },
];

interface Evidence {
  tests: { tests: number; pass: number; fail: number };
  types: { errors: number; unused_expectations: number };
  mutants: { mutant: string; file: string; killed: boolean; tests: number; fail: number }[];
  properties: { name: string; cases: number; checks: number; failure_count: number }[];
  property_checks: number; property_failures: number;
}
interface Point { n: number; ms: number; ms_per_1k_nodes: number; events?: number; edges?: number }
interface Benchmarks {
  node: string; repeats: number; edges_per_node: number; million_node_levels: number;
  scaling: { algorithm: string; points: Point[]; slope: number }[];
  head_to_head: { algorithm: string; n: number; ms: number }[];
  executors: { executor: string; points: Point[]; slope: number }[];
}
interface Source { path: string; text: string }

export function App() {
  const [tab, setTab] = useState('run');
  const meta = useMeta();
  const metrics = useMetrics();
  const crispdm = useCrispDm();
  const audit = useAudit();
  const provenance = useProvenance();
  const ev = useApi<Evidence>('/api/evidence');
  const bench = useApi<Benchmarks>('/api/benchmarks');
  const typeTests = useApi<Source>('/api/source/type-tests');
  const run = metrics.data;

  const leaderboard = run && (
    <ModelLeaderboard run={run} entityLabel="Ordering method" showTiming={false} columns={[
      { key: 'ms_at_8000_nodes', label: 'ms at 8,000 nodes', goodWhenUp: false, format: (v) => v.toFixed(2) },
      { key: 'scaling_exponent', label: 'Scaling exponent', goodWhenUp: false, format: (v) => v.toFixed(2) },
      { key: 'largest_nodes', label: 'Largest graph', format: (v) => v.toLocaleString() },
      { key: 'ms_at_largest', label: 'ms at largest', goodWhenUp: false, format: (v) => v.toFixed(1) },
    ]} />
  );

  return (
    <AppShell meta={meta.data} provenance={provenance.data} tabs={TABS} active={tab}
              onTabChange={setTab} repoUrl="https://github.com/aniket-alt/ai-coding-assistant">
      {tab === 'run' && (
        <Panel title="Run a workflow"
               description="A nightly retraining pipeline, scheduled by the engine running in your browser. Durations and failures are simulated; ordering, limits, retries and skips are the engine's real decisions.">
          <RunReplay />
        </Panel>
      )}

      {tab === 'cycle' && (
        <Panel title="Cycle detection" description="Validation runs Kahn's algorithm; when it cannot order every node, it returns one concrete cycle.">
          <CycleDemo />
        </Panel>
      )}

      {tab === 'evidence' && ev.data && (
        <>
          <Panel title="Is the engine correct?" description="Recorded when ml/train.py ran the engine's test suites.">
            <StatRow>
              <StatTile label="Tests" value={`${ev.data.tests.pass} / ${ev.data.tests.tests}`} hint="passed, with node --test" />
              <StatTile label="Property checks" value={ev.data.property_checks.toLocaleString()} hint={`${ev.data.property_failures} failures`} />
              <StatTile label="Planted bugs caught" value={`${ev.data.mutants.filter((m) => m.killed).length} / ${ev.data.mutants.length}`} hint="mutation testing" />
              <StatTile label="Type errors" value={ev.data.types.errors} hint="tsc, including compile-time tests" />
            </StatRow>
          </Panel>
          <Panel title="Properties" description="Each checks the engine against an independent reference on random graphs from fixed seeds.">
            <Table head={['Property', 'Random cases', 'Checks', 'Failures']}
                   rows={ev.data.properties.map((p) => [p.name, p.cases.toLocaleString(), p.checks.toLocaleString(), String(p.failure_count)])} />
          </Panel>
          <Panel title="Mutation testing"
                 description="Each row is a realistic bug planted in a copy of the engine. Tests that still pass on broken code prove nothing, so every mutant must be caught.">
            <Table head={['Planted bug', 'File', 'Caught', 'Failing tests or type errors']}
                   rows={ev.data.mutants.map((m) => [m.mutant, m.file, m.killed ? 'yes' : 'NO', String(m.fail)])} />
          </Panel>
        </>
      )}

      {tab === 'benchmarks' && bench.data && (
        <>
          <Panel title="Ordering a graph" description={`Median of ${bench.data.repeats} runs after a warm-up, Node ${bench.data.node}, ${bench.data.edges_per_node} edges per node.`}>
            {leaderboard}
            <div style={{ display: 'grid', gap: 'var(--space-4)', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', marginTop: 'var(--space-4)' }}>
              {bench.data.scaling.map((s) => (
                <Chart key={s.algorithm} kind="bar" height={220} data={s.points} x="n" xLabel={`${s.algorithm}: nodes`}
                       xFormat={(v) => Number(v).toLocaleString()} series={[{ key: 'ms_per_1k_nodes', label: 'ms per 1,000 nodes' }]}
                       yFormat={(v) => v.toFixed(3)}
                       caption={`Time per 1,000 nodes stays level when growth is linear and rises when it is not. Scaling exponent ${s.slope.toFixed(2)}.`} />
              ))}
            </div>
          </Panel>
          <Panel title="Scheduling a run" description="Both executors on the same random workflows with a concurrency limit of 8. The rescanning reference is only run to 3,000 nodes.">
            <Table head={['Executor', 'Nodes', 'Events', 'ms', 'ms per 1,000 nodes']}
                   rows={bench.data.executors.flatMap((e) => e.points.map((p) => [e.executor, p.n.toLocaleString(), (p.events ?? 0).toLocaleString(), p.ms.toFixed(1), p.ms_per_1k_nodes.toFixed(2)]))} />
            <StatRow>
              {bench.data.executors.map((e) => (
                <StatTile key={e.executor} label={e.executor} value={e.slope.toFixed(2)} hint="scaling exponent" />
              ))}
              <StatTile label="Levels in a 1,000,000-node graph" value={bench.data.million_node_levels} />
            </StatRow>
          </Panel>
        </>
      )}

      {tab === 'types' && typeTests.data && (
        <Panel title="Compile-time tests"
               description={`${typeTests.data.path}: every @ts-expect-error line must fail to compile. If a type change lets one compile, tsc reports the unused directive and the build fails.`}>
          <pre style={{ overflowX: 'auto', fontSize: 12, lineHeight: 1.5, background: 'var(--surface-1)', border: '1px solid var(--border)',
            borderRadius: 'var(--radius)', padding: 'var(--space-4)', margin: 0 }}>
            {typeTests.data.text.split('\n').map((line, i) => (
              <div key={i} style={{ color: line.includes('@ts-expect-error') ? 'var(--status-critical)'
                : line.trim().startsWith('//') || line.trim().startsWith('*') || line.trim().startsWith('/**') ? 'var(--text-muted)' : 'var(--text-primary)' }}>
                {line || ' '}
              </div>
            ))}
          </pre>
        </Panel>
      )}

      {tab === 'crispdm' && crispdm.data && (
        <CrispDmWalkthrough report={crispdm.data} slots={{ evaluation: leaderboard || undefined }} />
      )}

      {tab === 'audit' && (
        <>
          <AuditPanel findings={audit.data ?? []}
                      runtimeAssertions={(run?.audit?.runtime_assertions as never) ?? []}
                      summary={{ critical: run?.audit?.critical as number | undefined }} />
          {provenance.data && <ProvenanceBadge provenance={provenance.data} />}
        </>
      )}
    </AppShell>
  );
}

function Table({ head, rows }: { head: string[]; rows: string[][] }) {
  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', minWidth: 600, borderCollapse: 'collapse', fontSize: 'var(--text-sm)' }}>
        <thead><tr style={{ borderBottom: '2px solid var(--axis)' }}>
          {head.map((h, i) => <th key={h} style={{ ...th, textAlign: i ? 'right' : 'left' }}>{h}</th>)}
        </tr></thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} style={{ borderBottom: '1px solid var(--gridline)' }}>
              {r.map((c, j) => <td key={j} style={{ ...td, textAlign: j ? 'right' : 'left',
                color: c === 'NO' ? 'var(--status-critical)' : 'var(--text-primary)' }}>{c}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const th: React.CSSProperties = {
  padding: '8px 10px', color: 'var(--text-muted)', fontWeight: 600, fontSize: 'var(--text-xs)',
  textTransform: 'uppercase', letterSpacing: 0.3, whiteSpace: 'nowrap',
};
const td: React.CSSProperties = { padding: '9px 10px', color: 'var(--text-primary)', whiteSpace: 'nowrap' };
