/**
 * Project 11 client.
 *
 * LeakBenchmark is the project-specific view. The detector tournament uses the
 * shared leaderboard, and the portfolio table renders each project's audit
 * state exactly as its artifacts recorded it.
 */

import { useState } from 'react';
import {
  AppShell, AuditPanel, CrispDmWalkthrough, MetricsPanel, ModelLeaderboard, Panel,
  ProvenanceBadge, useApi, useAudit, useCrispDm, useMeta, useMetrics, useProvenance,
} from '@dsfs/ui';
import type { Tab } from '@dsfs/ui';
import { LeakBenchmark } from './LeakBenchmark';
import type { Variant } from './LeakBenchmark';

const TABS: Tab[] = [
  { id: 'benchmark', label: 'Leak benchmark' },
  { id: 'detectors', label: 'Detector tournament' },
  { id: 'portfolio', label: 'Portfolio' },
  { id: 'crispdm', label: 'CRISP-DM' },
  { id: 'audit', label: 'Audit' },
];

interface ProjectAudit {
  project: string; title: string; task: string; dataset: string; rows: number; real_data: boolean;
  models: number; best_model: string; assertions_passed: number; assertions_run: number;
  critical: number; warnings: number; assertions: string[]; trained_at: string;
}

export function App() {
  const [tab, setTab] = useState('benchmark');
  const meta = useMeta();
  const metrics = useMetrics();
  const crispdm = useCrispDm();
  const audit = useAudit();
  const provenance = useProvenance();
  const bench = useApi<{ clean_average_precision: number; variants: Variant[] }>('/api/benchmark');
  const portfolio = useApi<ProjectAudit[]>('/api/portfolio');
  const run = metrics.data;

  const leaderboard = run && (
    <ModelLeaderboard run={run} entityLabel="Detector" showTiming={false} columns={[
      { key: 'balanced_accuracy', label: 'Balanced accuracy', format: (v) => v.toFixed(3) },
      { key: 'recall_pct', label: 'Leaks caught', format: (v) => `${v.toFixed(0)}%` },
      { key: 'false_positive_rate_pct', label: 'False alarms', goodWhenUp: false, format: (v) => `${v.toFixed(0)}%` },
      { key: 'damage_caught_pct', label: 'Damage caught', format: (v) => `${v.toFixed(0)}%` },
    ]} />
  );

  return (
    <AppShell meta={meta.data} provenance={provenance.data} tabs={TABS} active={tab}
              onTabChange={setTab} repoUrl="https://github.com/aniket-alt/ai-coding-assistant">
      {tab === 'benchmark' && (
        <Panel title="Planted leaks, and what caught them"
               description="Each variant is a real training script with one leak planted on purpose. Select a row to read its source.">
          {bench.data ? <LeakBenchmark data={bench.data} /> : <p>{bench.error ?? 'Loading benchmark…'}</p>}
        </Panel>
      )}

      {tab === 'detectors' && run && (
        <>
          <Panel title="Detector tournament"
                 description="Balanced accuracy, because flagging everything would otherwise look good on a corpus with more leaks than controls.">
            {leaderboard}
          </Panel>
          <Panel title="Both detectors together" description="The auditor every other project in this portfolio relies on.">
            <MetricsPanel run={run} showBaselineLift={false} showSplit={false} highlight={[
              { key: 'balanced_accuracy', label: 'Balanced accuracy', format: (v) => v.toFixed(3),
                hint: 'Average of the share of leaks caught and the share of clean scripts left alone.' },
              { key: 'leaks_caught', label: 'Leaks caught', format: (v) => v.toFixed(0),
                unit: bench.data ? `of ${bench.data.variants.filter((x) => x.kind === 'leak').length}` : undefined },
              { key: 'damage_caught_pct', label: 'Damage caught', unit: '%', format: (v) => v.toFixed(0),
                hint: 'Share of total score inflation belonging to leaks that were caught.' },
              { key: 'false_positive_rate_pct', label: 'False alarms', unit: '%', format: (v) => v.toFixed(0) },
            ]} />
          </Panel>
        </>
      )}

      {tab === 'portfolio' && (
        <Panel title="Every built project's audit state"
               description="Read from each project's artifacts when this benchmark last ran. Read it alongside the benchmark: a pass rules out what the detectors can see.">
          {portfolio.data ? (
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', minWidth: 900, borderCollapse: 'collapse', fontSize: 'var(--text-sm)' }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid var(--axis)' }}>
                    {['Project', 'Dataset', 'Rows', 'Models', 'Assertions', 'Static findings', 'Checks that ran'].map((h, i) => (
                      <th key={h} style={{ padding: '8px 10px', color: 'var(--text-muted)', fontSize: 'var(--text-xs)',
                        textTransform: 'uppercase', textAlign: i === 2 || i === 3 ? 'right' : 'left', whiteSpace: 'nowrap' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {portfolio.data.map((p) => (
                    <tr key={p.project} style={{ borderBottom: '1px solid var(--gridline)', verticalAlign: 'top' }}>
                      <td style={cell}><strong>{p.title}</strong><div style={small}>{p.task}</div></td>
                      <td style={cell}>{p.dataset}<div style={small}>{p.real_data ? '✓ real data' : '⚠ synthetic'}</div></td>
                      <td style={{ ...cell, textAlign: 'right' }}>{p.rows.toLocaleString()}</td>
                      <td style={{ ...cell, textAlign: 'right' }}>{p.models}</td>
                      <td style={{ ...cell, fontWeight: 700,
                        color: p.assertions_passed === p.assertions_run ? 'var(--delta-up)' : 'var(--status-critical)' }}>
                        {p.assertions_passed}/{p.assertions_run} passed
                      </td>
                      <td style={cell}>{p.critical} critical · {p.warnings} warnings</td>
                      <td style={{ ...cell, whiteSpace: 'normal' }}><span style={small}>{p.assertions.join(', ')}</span></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : <p>{portfolio.error ?? 'Loading portfolio…'}</p>}
        </Panel>
      )}

      {tab === 'crispdm' && crispdm.data && (
        <CrispDmWalkthrough report={crispdm.data} slots={{ modeling: leaderboard || undefined }} />
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

const cell: React.CSSProperties = { padding: '9px 10px', color: 'var(--text-primary)', whiteSpace: 'nowrap' };
const small: React.CSSProperties = { color: 'var(--text-muted)', fontSize: 'var(--text-xs)' };
