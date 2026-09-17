/**
 * Project 13 client.
 *
 * Operations views for project 01's model: drift and retraining policy,
 * explainable predictions, the model registry, load test and model card.
 * Every figure is read from the training run through the API.
 */

import { useMemo, useState } from 'react';
import {
  AppShell, AuditPanel, Chart, CrispDmWalkthrough, ModelLeaderboard, modelColors, Panel, PredictionForm,
  ProvenanceBadge, StatRow, StatTile, useApi, useAudit, useCrispDm, useMeta, useMetrics, useProvenance,
} from '@dsfs/ui';
import type { RunRecord } from '@dsfs/ui';
import type { Tab } from '@dsfs/ui';

const TABS: Tab[] = [
  { id: 'drift', label: 'Drift & retraining' },
  { id: 'explain', label: 'Explain a prediction' },
  { id: 'registry', label: 'Model registry' },
  { id: 'capacity', label: 'Load test & model card' },
  { id: 'crispdm', label: 'CRISP-DM' },
  { id: 'audit', label: 'Audit' },
];

interface MonthScore { month: string; rmse: number; mae: number; within_5min: number; retrained: boolean; trained_on: string[] }
interface Drift {
  drift_vs_january: Record<string, number | string>[];
  psi_alarm: number;
  alarm_checks: { decided_for: string; compared: string; worst_column: string; psi: number; retrained: boolean }[];
  policy_months: Record<string, MonthScore[]>;
}
interface Registry {
  versions: { version: number; policy: string; trained_on: string[]; rows: number; fit_seconds: number;
              first_used_for: string; rmse_on_first_month: number }[];
  served: { policy: string; trained_on: string[] };
  global_importance: { feature: string; mean_abs_minutes: number }[];
}
interface LoadTest {
  machine: string; server: string;
  levels: { concurrency: number; requests: number; errors: number; throughput_rps: number;
            p50_ms: number; p95_ms: number; p99_ms: number; max_ms: number }[];
}
interface ModelCard {
  model: string; intended_use: string; not_for: string; training_data: string; evaluation: string;
  monitoring: string; capacity: string; limitations: string[]; assertions: string[];
}
interface Explanation {
  duration_minutes: number; baseline_minutes: number; policy: string; trained_on: string[];
  contributions: { feature: string; minutes: number }[];
}

const DRIFT_COLUMNS = ['duration_min', 'trip_distance', 'hour', 'passenger_count', 'PULocationID'];

export function App() {
  const [tab, setTab] = useState('drift');
  const meta = useMeta();
  const metrics = useMetrics();
  const crispdm = useCrispDm();
  const audit = useAudit();
  const provenance = useProvenance();
  const drift = useApi<Drift>('/api/drift');
  const reg = useApi<Registry>('/api/registry');
  const load = useApi<LoadTest>('/api/load-test');
  const card = useApi<ModelCard>('/api/model-card');
  const run = metrics.data;

  // One row per month with a column per strategy, as recorded by training.
  const monthly = useMemo(() => {
    if (!drift.data) return { rows: [], names: [] as string[] };
    const names = Object.keys(drift.data.policy_months);
    const months = drift.data.policy_months[names[0]].map((r) => r.month);
    const rows = months.map((m, i) => {
      const row: Record<string, number | string> = { month: m };
      names.forEach((n) => { row[n] = drift.data!.policy_months[n][i].rmse; });
      return row;
    });
    return { rows, names };
  }, [drift.data]);

  const leaderboard = run && (
    <ModelLeaderboard run={run} entityLabel="Strategy" showTiming={false} columns={[
      { key: 'mean_rmse', label: 'Mean RMSE (min)', goodWhenUp: false, format: (v) => v.toFixed(3) },
      { key: 'worst_month_rmse', label: 'Worst month', goodWhenUp: false, format: (v) => v.toFixed(3) },
      { key: 'july_rmse', label: 'July', goodWhenUp: false, format: (v) => v.toFixed(3) },
      { key: 'retrains', label: 'Models fitted', goodWhenUp: false, format: (v) => v.toFixed(0) },
      { key: 'mean_within_5min', label: 'Within 5 min', format: (v) => `${v.toFixed(1)}%` },
    ]} />
  );

  return (
    <AppShell meta={meta.data} provenance={provenance.data} tabs={TABS} active={tab}
              onTabChange={setTab} repoUrl="https://github.com/aniket-alt/ai-coding-assistant">
      {tab === 'drift' && (
        <>
          <Panel title="Which retraining policy keeps the model accurate?"
                 description="Each month, a strategy may train only on months that have already finished, then is scored on the new month.">
            {leaderboard}
          </Panel>
          {drift.data && (
            <>
              <Panel title="Error month by month" description="RMSE in minutes on each month's real trips.">
                <Chart kind="line" height={320} data={monthly.rows} x="month" xLabel="Month served"
                       series={seriesFor(run, monthly.names)}
                       directLabels={false} yFormat={(v) => v.toFixed(2)}
                       caption="The same model, features and hyperparameters throughout; only what each strategy was trained on changes." />
              </Panel>
              <Panel title="How much did the city drift from January?"
                     description={`Population stability index against January. Conventionally, below 0.1 is stable and above 0.25 is a major shift. The retraining alarm fires above ${drift.data.psi_alarm}.`}>
                <div style={{ overflowX: 'auto' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 'var(--text-sm)' }}>
                    <thead><tr style={{ borderBottom: '2px solid var(--axis)' }}>
                      {['Month', ...DRIFT_COLUMNS].map((h, i) => (
                        <th key={h} style={{ ...th, textAlign: i ? 'right' : 'left' }}>{h}</th>))}
                    </tr></thead>
                    <tbody>
                      {drift.data.drift_vs_january.map((r) => (
                        <tr key={String(r.month)} style={{ borderBottom: '1px solid var(--gridline)' }}>
                          <td style={td}>{String(r.month)}</td>
                          {DRIFT_COLUMNS.map((c) => {
                            const v = Number(r[c]);
                            const level = v > 0.25 ? 'var(--status-critical)' : v > 0.1 ? 'var(--status-serious)' : 'var(--text-primary)';
                            return (
                              <td key={c} style={{ ...td, textAlign: 'right', color: level, fontWeight: v > 0.1 ? 700 : 400 }}>
                                {v.toFixed(3)}{v > 0.25 ? ' ▲▲' : v > 0.1 ? ' ▲' : ''}
                              </td>
                            );
                          })}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  <p style={{ color: 'var(--text-muted)', fontSize: 'var(--text-xs)' }}>▲ above 0.1 · ▲▲ above 0.25</p>
                </div>
                <h3 style={{ fontSize: 'var(--text-sm)', margin: 'var(--space-4) 0 var(--space-2)' }}>Alarm decisions</h3>
                <ul style={{ margin: 0, paddingLeft: 'var(--space-5)', fontSize: 'var(--text-sm)' }}>
                  {drift.data.alarm_checks.map((a) => (
                    <li key={a.decided_for}>
                      For {a.decided_for}: {a.compared} against the model's training data, largest PSI {a.psi.toFixed(3)} on{' '}
                      <code>{a.worst_column}</code> — <strong>{a.retrained ? 'retrained' : 'kept the model'}</strong>
                    </li>
                  ))}
                </ul>
              </Panel>
            </>
          )}
        </>
      )}

      {tab === 'explain' && (
        <Panel title="Why did the model say that?"
               description="TreeSHAP contributions of the served model, in minutes. They add up to the estimate.">
          <PredictionForm
            fields={[
              { kind: 'select', name: 'pu_location', label: 'Pickup zone', options: '/api/zones' },
              { kind: 'select', name: 'do_location', label: 'Dropoff zone', options: '/api/zones' },
              { kind: 'datetime', name: 'when', label: 'Pickup time' },
              { kind: 'slider', name: 'passenger_count', label: 'Passengers', min: 1, max: 6 },
            ]}
            initialValues={{ pu_location: 132, do_location: 161, when: '2024-07-19T17:00', passenger_count: 1 }}
            autoSubmit
            renderResult={(r, { loading }) => {
              const e = r as Explanation | null;
              if (!e) return <p style={{ color: 'var(--text-muted)' }}>{loading ? 'Explaining…' : ''}</p>;
              return (
                <div style={{ opacity: loading ? 0.55 : 1 }}>
                  <StatRow>
                    <StatTile label="Estimated duration" value={e.duration_minutes} unit="min" size="hero" accent="var(--series-1)" />
                    <StatTile label="Starting point" value={e.baseline_minutes} unit="min"
                              hint="The model's average prediction before any feature is considered." />
                    <StatTile label="Served model" value={e.trained_on.length} unit="months"
                              hint={`${e.policy}: ${e.trained_on.join(', ')}`} />
                  </StatRow>
                  <div style={{ marginTop: 'var(--space-4)' }}>
                    <Chart kind="bar" horizontal height={320} data={e.contributions} x="feature"
                           series={[{ key: 'minutes', label: 'Contribution (minutes)' }]}
                           yFormat={(v) => `${v >= 0 ? '+' : ''}${v.toFixed(1)}`}
                           caption="The ten largest contributions. Positive values lengthen the estimate; negative values shorten it." />
                  </div>
                </div>
              );
            }}
          />
        </Panel>
      )}

      {tab === 'registry' && reg.data && (
        <>
          <Panel title="Model registry" description={`Every model fitted during the experiment. Served: ${reg.data.served.policy}, trained on ${reg.data.served.trained_on.join(', ')}.`}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', minWidth: 760, borderCollapse: 'collapse', fontSize: 'var(--text-sm)' }}>
                <thead><tr style={{ borderBottom: '2px solid var(--axis)' }}>
                  {['Version', 'Policy', 'Trained on', 'Rows', 'Fit (s)', 'First served', 'RMSE that month'].map((h, i) => (
                    <th key={h} style={{ ...th, textAlign: [0, 3, 4, 6].includes(i) ? 'right' : 'left' }}>{h}</th>))}
                </tr></thead>
                <tbody>
                  {reg.data.versions.map((v) => (
                    <tr key={v.version} style={{ borderBottom: '1px solid var(--gridline)' }}>
                      <td style={{ ...td, textAlign: 'right' }}>v{v.version}</td>
                      <td style={td}>{v.policy}</td>
                      <td style={{ ...td, whiteSpace: 'normal' }}>{v.trained_on.join(', ')}</td>
                      <td style={{ ...td, textAlign: 'right' }}>{v.rows.toLocaleString()}</td>
                      <td style={{ ...td, textAlign: 'right' }}>{v.fit_seconds.toFixed(1)}</td>
                      <td style={td}>{v.first_used_for}</td>
                      <td style={{ ...td, textAlign: 'right' }}>{v.rmse_on_first_month.toFixed(3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>
          <Panel title="What the served model relies on" description="Mean absolute TreeSHAP contribution across July trips, in minutes.">
            <Chart kind="bar" horizontal height={380} data={reg.data.global_importance.slice(0, 12)} x="feature"
                   series={[{ key: 'mean_abs_minutes', label: 'Mean |contribution| (minutes)' }]}
                   yFormat={(v) => v.toFixed(2)} />
          </Panel>
        </>
      )}

      {tab === 'capacity' && (
        <>
          {load.data && (
            <Panel title="Load test" description={`${load.data.server} on ${load.data.machine}. Measured during training, not a production claim.`}>
              <StatRow>
                {load.data.levels.map((l) => (
                  <StatTile key={l.concurrency} label={`Concurrency ${l.concurrency}`} value={l.throughput_rps.toFixed(0)}
                            unit="req/s" hint={`p50 ${l.p50_ms.toFixed(1)} ms · p99 ${l.p99_ms.toFixed(1)} ms · ${l.errors} errors`} />
                ))}
              </StatRow>
              <div style={{ marginTop: 'var(--space-4)' }}>
                <Chart kind="bar" height={260}
                       data={load.data.levels.map((l) => ({ level: `${l.concurrency} concurrent`, p50: l.p50_ms, p95: l.p95_ms, p99: l.p99_ms }))}
                       x="level" series={[{ key: 'p50', label: 'p50 (ms)' }, { key: 'p95', label: 'p95 (ms)' }, { key: 'p99', label: 'p99 (ms)' }]}
                       yFormat={(v) => `${v.toFixed(0)} ms`}
                       caption="Latency percentiles per request as concurrency rises. With one worker, extra concurrency mostly adds queueing time." />
              </div>
            </Panel>
          )}
          {card.data && (
            <Panel title="Model card" description="Assembled from the run record.">
              <dl style={{ display: 'grid', gridTemplateColumns: 'minmax(140px, 200px) 1fr', gap: '8px 16px', margin: 0, fontSize: 'var(--text-sm)' }}>
                {([['Model', card.data.model], ['Intended use', card.data.intended_use], ['Not for', card.data.not_for],
                   ['Training data', card.data.training_data], ['Evaluation', card.data.evaluation],
                   ['Monitoring', card.data.monitoring], ['Capacity', card.data.capacity]] as const).map(([k, v]) => (
                  <div key={k} style={{ display: 'contents' }}>
                    <dt style={{ color: 'var(--text-muted)', fontWeight: 600 }}>{k}</dt><dd style={{ margin: 0 }}>{v}</dd>
                  </div>
                ))}
                <dt style={{ color: 'var(--text-muted)', fontWeight: 600 }}>Limitations</dt>
                <dd style={{ margin: 0 }}><ul style={{ margin: 0, paddingLeft: 18 }}>{card.data.limitations.map((l) => <li key={l}>{l}</li>)}</ul></dd>
                <dt style={{ color: 'var(--text-muted)', fontWeight: 600 }}>Checks that ran</dt>
                <dd style={{ margin: 0 }}>{card.data.assertions.join(' · ')}</dd>
              </dl>
            </Panel>
          )}
        </>
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

const th: React.CSSProperties = {
  padding: '8px 10px', color: 'var(--text-muted)', fontWeight: 600, fontSize: 'var(--text-xs)',
  textTransform: 'uppercase', letterSpacing: 0.3, whiteSpace: 'nowrap',
};
const td: React.CSSProperties = { padding: '9px 10px', color: 'var(--text-primary)', whiteSpace: 'nowrap' };

// Line colours match the leaderboard swatches; baselines are muted, told apart by dash.
function seriesFor(run: RunRecord | null | undefined, names: string[]) {
  const colors = run ? modelColors(run) : {};
  let dashes = 0;
  return names.map((n) => {
    const muted = colors[n] === 'var(--text-muted)';
    return { key: n, label: n, color: colors[n], dash: muted ? ['6 4', '2 3'][dashes++ % 2] : undefined };
  });
}
