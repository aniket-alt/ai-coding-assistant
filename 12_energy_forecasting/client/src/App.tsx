/**
 * Project 12 client.
 *
 * LoadFan is the only project-specific view. Patterns, calibration, the
 * tournament and the leakage demonstration are built from shared components
 * reading the training artifacts.
 */

import { useState } from 'react';
import {
  AppShell, AuditPanel, Chart, CrispDmWalkthrough, MetricsPanel, ModelLeaderboard, Panel,
  ProvenanceBadge, StatRow, StatTile, useApi, useAudit, useCrispDm, useMeta, useMetrics, useProvenance,
} from '@dsfs/ui';
import type { Tab } from '@dsfs/ui';
import { LoadFan } from './LoadFan';
import type { Coverage, NextPoint, RecentPoint } from './LoadFan';

const TABS: Tab[] = [
  { id: 'forecast', label: 'Forecast' },
  { id: 'patterns', label: 'Load patterns' },
  { id: 'leak', label: 'The leak' },
  { id: 'tournament', label: 'Forecaster tournament' },
  { id: 'crispdm', label: 'CRISP-DM' },
  { id: 'audit', label: 'Audit' },
];

interface ForecastResponse {
  model: string; recent: RecentPoint[]; next_day: NextPoint[];
  recent_coverage: Record<string, Coverage>;
}
interface Patterns {
  by_hour: { hour: number; weekday_kw: number; weekend_kw: number }[];
  by_month: { month: string; kw: number }[];
}
interface Significance { mean_diff: number; ci_low: number; ci_high: number; verdict: string }
interface Calibration {
  models: { model: string; coverage_80: number; interval_width_kw: number; pinball: number }[];
  significance_vs_yesterday: Record<string, Significance>;
  leakage_demo: { honest_pinball: number; leaky_pinball: number; flattery_pct: number;
                  leaky_lags: number[]; leaky_coverage_80: number };
}

export function App() {
  const [tab, setTab] = useState('forecast');
  const [days, setDays] = useState(7);
  const meta = useMeta();
  const metrics = useMetrics();
  const crispdm = useCrispDm();
  const audit = useAudit();
  const provenance = useProvenance();
  const fc = useApi<ForecastResponse>('/api/forecast');
  const pat = useApi<Patterns>('/api/patterns');
  const cal = useApi<Calibration>('/api/calibration');
  const run = metrics.data;

  const leaderboard = run && (
    <ModelLeaderboard run={run} entityLabel="Forecaster" showParams columns={[
      { key: 'pinball', label: 'Pinball loss', goodWhenUp: false, format: (v) => v.toFixed(4) },
      { key: 'skill_vs_yesterday_pct', label: 'Skill vs yesterday', format: (v) => `${v.toFixed(1)}%` },
      { key: 'coverage_80', label: '80% coverage', format: (v) => `${(v * 100).toFixed(1)}%` },
      { key: 'wape_pct', label: 'WAPE', goodWhenUp: false, format: (v) => `${v.toFixed(1)}%` },
    ]} />
  );

  return (
    <AppShell meta={meta.data} provenance={provenance.data} tabs={TABS} active={tab}
              onTabChange={setTab} repoUrl="https://github.com/aniket-alt/ai-coding-assistant">
      {tab === 'forecast' && (
        <>
          <Panel title="Day-ahead forecast"
                 description="Every band below was issued a full day before the hour it describes."
                 actions={
                   <div style={{ display: 'flex', gap: 4 }}>
                     {[7, 14].map((d) => (
                       <button key={d} type="button" onClick={() => setDays(d)} aria-pressed={days === d}
                               style={{ padding: '4px 10px', borderRadius: 'var(--radius-sm)',
                                 border: '1px solid var(--border)', cursor: 'pointer', fontSize: 'var(--text-xs)',
                                 background: days === d ? 'var(--surface-2)' : 'transparent',
                                 color: days === d ? 'var(--text-primary)' : 'var(--text-muted)', fontWeight: 600 }}>
                         Last {d} days
                       </button>
                     ))}
                   </div>
                 }>
            {fc.data ? (
              <LoadFan recent={fc.data.recent} nextDay={fc.data.next_day} days={days}
                       coverage={fc.data.recent_coverage[String(days)]} />
            ) : <p style={{ color: 'var(--text-muted)' }}>{fc.error ?? 'Loading forecast…'}</p>}
          </Panel>
          {run && (
            <Panel title="Walk-forward performance" description="Across every day-ahead forecast issued since the first refit.">
              <MetricsPanel run={run} showBaselineLift={false} highlight={[
                { key: 'pinball', label: 'Pinball loss', format: (v) => v.toFixed(4),
                  hint: 'Quantile loss over P10, P50 and P90, in kW. Lower is better.' },
                { key: 'skill_vs_yesterday_pct', label: 'Skill vs yesterday', unit: '%', format: (v) => v.toFixed(1),
                  hint: 'Reduction in pinball loss against repeating the same hour yesterday.' },
                { key: 'coverage_80', label: '80% coverage', unit: '%', format: (v) => (v * 100).toFixed(1),
                  hint: 'Should be 80.' },
                { key: 'wape_pct', label: 'WAPE', unit: '%', format: (v) => v.toFixed(1),
                  hint: 'Median error as a share of total load.' },
              ]} />
            </Panel>
          )}
        </>
      )}

      {tab === 'patterns' && pat.data && (
        <>
          <Panel title="A household's day" description="Average load by hour, weekdays against weekends, across four years.">
            <Chart kind="line" height={300} data={pat.data.by_hour} x="hour" xLabel="Hour of day"
                   series={[{ key: 'weekday_kw', label: 'Weekday (kW)' }, { key: 'weekend_kw', label: 'Weekend (kW)' }]}
                   yFormat={(v) => v.toFixed(2)}
                   caption="The evening peak arrives on both, but weekends are busier through the middle of the day." />
          </Panel>
          <Panel title="A household's year" description="Average load by calendar month.">
            <Chart kind="bar" height={260} data={pat.data.by_month} x="month"
                   series={[{ key: 'kw', label: 'Mean load (kW)' }]} yFormat={(v) => v.toFixed(2)}
                   caption="Winter use is close to double summer use. No temperature data is available, so the models learn season only from the calendar." />
          </Panel>
        </>
      )}

      {tab === 'leak' && cal.data && (
        <Panel title="What a too-recent lag would have claimed"
               description="The same LightGBM, given the load one to three hours earlier, which is not available when forecasting a day ahead.">
          <StatRow>
            <StatTile label="Honest pinball loss" value={cal.data.leakage_demo.honest_pinball.toFixed(4)}
                      size="hero" accent="var(--series-1)" hint="Every lag at least 24 hours." />
            <StatTile label="Leaky pinball loss" value={cal.data.leakage_demo.leaky_pinball.toFixed(4)}
                      hint={`Adds lags of ${cal.data.leakage_demo.leaky_lags.join(', ')} hours.`} />
            <StatTile label="Improvement that does not exist" value={cal.data.leakage_demo.flattery_pct.toFixed(0)} unit="%"
                      hint="How much better the leaky model looks than anything that could run in production." />
          </StatRow>
          <p style={{ marginTop: 'var(--space-4)', color: 'var(--text-secondary)', maxWidth: 760, lineHeight: 1.6 }}>
            The leaky model is scored on exactly the same hours, with the same walk-forward refits. It is measured
            here and kept out of the tournament: it is the result a careless evaluation would have published, and
            the minimum-lag check in training exists to make it impossible to publish by accident.
          </p>
        </Panel>
      )}

      {tab === 'tournament' && run && cal.data && (
        <>
          <Panel title="Forecaster tournament" description="Three seasonal baselines, two learned models and a fat-tailed interval.">
            {leaderboard}
          </Panel>
          <Panel title="Better than repeating yesterday?" description="Paired bootstrap on hourly pinball loss, 5,000 resamples.">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 'var(--text-sm)' }}>
                <thead><tr style={{ borderBottom: '2px solid var(--axis)' }}>
                  {['Forecaster', 'Loss difference (kW)', '95% interval', 'Verdict'].map((h) => (
                    <th key={h} style={{ textAlign: 'left', padding: '8px 10px', color: 'var(--text-muted)',
                      fontSize: 'var(--text-xs)', textTransform: 'uppercase' }}>{h}</th>))}
                </tr></thead>
                <tbody>
                  {Object.entries(cal.data.significance_vs_yesterday).map(([name, s]) => (
                    <tr key={name} style={{ borderBottom: '1px solid var(--gridline)' }}>
                      <td style={{ padding: '9px 10px' }}>{name}</td>
                      <td style={{ padding: '9px 10px' }}>{s.mean_diff.toFixed(4)}</td>
                      <td style={{ padding: '9px 10px', color: 'var(--text-muted)' }}>[{s.ci_low.toFixed(4)}, {s.ci_high.toFixed(4)}]</td>
                      <td style={{ padding: '9px 10px', fontWeight: 700, color: s.verdict === 'better'
                        ? 'var(--delta-up)' : s.verdict === 'worse' ? 'var(--status-critical)' : 'var(--text-muted)' }}>
                        {s.verdict.toUpperCase()}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>
          <Panel title="Interval calibration" description="Share of hours inside each forecaster's 80% band.">
            <Chart kind="bar" horizontal height={260} data={cal.data.models} x="model"
                   series={[{ key: 'coverage_80', label: 'Actual coverage of the 80% band (%)' }]}
                   yFormat={(v) => `${v.toFixed(1)}%`}
                   caption="Nominal coverage is 80%. Below it a band is overconfident; above it, wider than it needs to be." />
          </Panel>
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
