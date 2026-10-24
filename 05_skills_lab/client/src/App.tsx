/**
 * Project 05 client -- the skills catalogue.
 *
 * Browse the entries, open one to see its measured comparison, and run any of
 * them again. Nothing here computes a result: every number is returned by the
 * API, from the training run or from a fresh run of the same code.
 */

import { useEffect, useState } from 'react';
import {
  AppShell, AuditPanel, Chart, CrispDmWalkthrough, ModelLeaderboard, Panel, ProvenanceBadge,
  StatRow, StatTile, useApi, useAudit, useCrispDm, useMeta, useMetrics, useProvenance,
} from '@dsfs/ui';
import type { Tab } from '@dsfs/ui';
import type { SkillRecord } from './Catalogue';
import { CatalogueList, show, signed, SkillDetail } from './Catalogue';

const TABS: Tab[] = [
  { id: 'catalogue', label: 'Catalogue' },
  { id: 'ranked', label: 'What flatters most' },
  { id: 'datasets', label: 'Datasets' },
  { id: 'crispdm', label: 'CRISP-DM' },
  { id: 'audit', label: 'Audit' },
];

interface DatasetFacts {
  rows: number; numeric: number; categorical: number; positive_rate: number; rows_with_missing: number;
  name: string; source_url: string; license: string; sha256: string;
}
interface Catalogue {
  skills: SkillRecord[]; datasets: Record<string, DatasetFacts>; categories: string[]; seeds: number;
  auditor_warnings: { file: string; line: number; rule: string; message: string }[];
}

export function App() {
  const [tab, setTab] = useState('catalogue');
  const [selected, setSelected] = useState('');
  const meta = useMeta();
  const metrics = useMetrics();
  const crispdm = useCrispDm();
  const audit = useAudit();
  const provenance = useProvenance();
  const cat = useApi<Catalogue>('/api/catalogue');
  const run = metrics.data;

  useEffect(() => { if (!selected && cat.data) setSelected(cat.data.skills[0].id); }, [cat.data, selected]);
  const current = cat.data?.skills.find((s) => s.id === selected);
  const decisive = cat.data?.skills.filter((s) => s.gap.decisive && s.gap.flatters) ?? [];
  const quiet = cat.data?.skills.filter((s) => !s.gap.decisive) ?? [];

  const leaderboard = run && (
    <ModelLeaderboard run={run} entityLabel="Skill" columns={[
      { key: 'flattery', label: 'How much it flatters', format: (v) => signed(v) },
      { key: 'careless', label: 'Careless', format: (v) => show(v) },
      { key: 'careful', label: 'Careful', format: (v) => show(v) },
      { key: 'ci_low', label: '95% from', format: (v) => signed(v) },
      { key: 'ci_high', label: 'to', format: (v) => signed(v) },
    ]} />
  );

  return (
    <AppShell meta={meta.data} provenance={provenance.data} tabs={TABS} active={tab}
              onTabChange={setTab} repoUrl="https://github.com/aniket-alt/ai-coding-assistant">
      {tab === 'catalogue' && cat.data && (
        <>
          <Panel title="The catalogue"
                 description={`${cat.data.skills.length} techniques, each run the careless way and the careful way on a real dataset over ${cat.data.seeds} seeds. Pick one to see what the difference was.`}>
            <CatalogueList skills={cat.data.skills} selected={selected} onSelect={setSelected} />
          </Panel>
          {current && (
            <Panel title={current.name} description={`${current.category} · ${current.dataset} · ${current.metric}`}>
              <SkillDetail skill={current} />
            </Panel>
          )}
        </>
      )}

      {tab === 'ranked' && cat.data && (
        <>
          <Panel title="Which mistakes flatter the result most?"
                 description="The paired difference between the careless and the careful version, in each entry's own metric. Positive means the careless version looks better than it is.">
            {leaderboard}
          </Panel>
          <Panel title="Measured differences, with their intervals"
                 description="A bar is only evidence of an effect when its interval excludes zero.">
            <Chart kind="bar" horizontal height={cat.data.skills.length * 34 + 60}
                   data={[...cat.data.skills].sort((a, b) => (b.gap.mean * (b.higher_is_better ? 1 : -1)) - (a.gap.mean * (a.higher_is_better ? 1 : -1)))
                     .map((s) => ({ skill: s.name, flattery: s.gap.mean * (s.higher_is_better ? 1 : -1) }))}
                   x="skill" series={[{ key: 'flattery', label: 'Careless minus careful, in the metric’s own units' }]}
                   yFormat={(v) => show(v)}
                   caption="Metrics differ between entries, so the bars are not on one scale; each is read against its own entry." />
          </Panel>
          <Panel title="What the experiment could not resolve"
                 description="Entries whose interval covers zero. On these datasets, with these models, the careless version cost nothing measurable.">
            <ul style={{ margin: 0, paddingLeft: 20, lineHeight: 1.7 }}>
              {quiet.map((s) => (
                <li key={s.id}><strong>{s.name}</strong> — {signed(s.gap.mean)} [{signed(s.gap.ci_low)}, {signed(s.gap.ci_high)}] {s.metric}</li>
              ))}
            </ul>
            {quiet.length === 0 && <p style={{ margin: 0 }}>Every entry resolved a difference.</p>}
          </Panel>
        </>
      )}

      {tab === 'datasets' && cat.data && (
        <>
          <Panel title="The four datasets"
                 description="Small enough to re-run an entry in the browser, and real: each was downloaded from OpenML and checksummed.">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', minWidth: 720, borderCollapse: 'collapse', fontSize: 'var(--text-sm)' }}>
                <thead><tr style={{ borderBottom: '2px solid var(--axis)' }}>
                  {['Dataset', 'Rows', 'Numeric', 'Categorical', 'Positive', 'Rows with a gap', 'Licence'].map((h, i) => (
                    <th key={h} style={{ ...th, textAlign: i ? 'right' : 'left' }}>{h}</th>))}
                </tr></thead>
                <tbody>
                  {Object.entries(cat.data.datasets).map(([key, d]) => (
                    <tr key={key} style={{ borderBottom: '1px solid var(--gridline)' }}>
                      <td style={td}><a href={d.source_url} style={{ color: 'var(--series-1)' }}>{d.name}</a></td>
                      <td style={{ ...td, textAlign: 'right' }}>{d.rows.toLocaleString()}</td>
                      <td style={{ ...td, textAlign: 'right' }}>{d.numeric}</td>
                      <td style={{ ...td, textAlign: 'right' }}>{d.categorical}</td>
                      <td style={{ ...td, textAlign: 'right' }}>{(d.positive_rate * 100).toFixed(1)}%</td>
                      <td style={{ ...td, textAlign: 'right' }}>{(d.rows_with_missing * 100).toFixed(1)}%</td>
                      <td style={{ ...td, textAlign: 'right' }}>{d.license.split(';')[0]}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>
          <Panel title="The repository's auditor, pointed at this project"
                 description="These warnings are the careless implementations being recognised for what they are. The static audit reports no critical findings, because nothing here fits a model on data named as a test set.">
            <StatRow>
              <StatTile label="Warnings" value={cat.data.auditor_warnings.length} />
              <StatTile label="Entries with a measurable effect" value={`${decisive.length} of ${cat.data.skills.length}`} />
            </StatRow>
            <ul style={{ margin: 'var(--space-3) 0 0', paddingLeft: 20, lineHeight: 1.6, fontSize: 'var(--text-sm)' }}>
              {cat.data.auditor_warnings.slice(0, 8).map((w) => (
                <li key={`${w.file}${w.line}`}><code>{String(w.file).split(/[\\/]/).pop()}:{w.line}</code> — {w.message}</li>
              ))}
            </ul>
          </Panel>
        </>
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

const th: React.CSSProperties = {
  padding: '8px 10px', color: 'var(--text-muted)', fontWeight: 600, fontSize: 'var(--text-xs)',
  textTransform: 'uppercase', letterSpacing: 0.3, whiteSpace: 'nowrap',
};
const td: React.CSSProperties = { padding: '9px 10px', color: 'var(--text-primary)', whiteSpace: 'nowrap' };
