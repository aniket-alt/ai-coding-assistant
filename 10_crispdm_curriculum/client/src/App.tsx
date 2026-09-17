/**
 * Project 10 client -- the course.
 *
 * One tab per lesson, built from shared components reading the training run,
 * plus the quiz. The fairness lesson is the centre of the course.
 */

import { useState } from 'react';
import {
  AppShell, AuditPanel, Chart, CrispDmWalkthrough, ModelLeaderboard, Panel, ProvenanceBadge,
  StatRow, StatTile, useApi, useAudit, useCrispDm, useMeta, useMetrics, useProvenance,
} from '@dsfs/ui';
import type { Tab } from '@dsfs/ui';
import { Quiz } from './Quiz';

const TABS: Tab[] = [
  { id: 'crispdm', label: '1 · CRISP-DM' },
  { id: 'eda', label: '2 · Exploration' },
  { id: 'unsupervised', label: '3 · Clusters & outliers' },
  { id: 'fairness', label: '4 · Models & fairness' },
  { id: 'rules', label: '5 · Rules' },
  { id: 'lsh', label: '6 · Similarity search' },
  { id: 'quiz', label: '7 · Quiz' },
  { id: 'audit', label: 'Audit' },
];

interface Eda {
  rows: number; high_income_pct: number;
  by_sex: { group: string; high_income_pct: number; people: number }[];
  by_race: { group: string; high_income_pct: number; people: number }[];
  missing_pct: Record<string, number>;
  age_hist: { age_band: string; people: number; high_income_pct: number }[];
  capital_gain_capped: number;
}
interface Unsupervised {
  silhouette: number;
  clusters: { cluster: string; people: number; median_age: number; median_hours: number;
              median_education_years: number; with_capital_gain_pct: number; high_income_pct: number }[];
  outliers: { flagged: number; any_capital_pct: number; any_capital_rest_pct: number; age70_pct: number;
              age70_rest_pct: number; top_coded_share_pct: number; high_income_pct: number };
}
interface Fairness {
  models: { model: string; family: string; roc_auc: number; disparate_impact_sex: number;
            equal_opportunity_gap_sex: number; selection_rate_men: number; selection_rate_women: number }[];
  proxy_auc: number;
  relationship_proxy: { relationship: string; men_pct: number; people: number }[];
}
interface Rule { if: string; support_pct: number; confidence: number; lift: number; test_people: number; test_confidence: number | null }
interface Lsh {
  indexed: number; queries: number; dimensions: number; bits: number; tables: number; recall_at_10: number;
  scanned_pct: number; exact_ms_per_query: number; lsh_ms_per_query: number;
}

export function App() {
  const [tab, setTab] = useState('crispdm');
  const meta = useMeta();
  const metrics = useMetrics();
  const crispdm = useCrispDm();
  const audit = useAudit();
  const provenance = useProvenance();
  const eda = useApi<Eda>('/api/eda');
  const uns = useApi<Unsupervised>('/api/unsupervised');
  const fair = useApi<Fairness>('/api/fairness');
  const rules = useApi<Rule[]>('/api/rules');
  const lsh = useApi<Lsh>('/api/lsh');
  const run = metrics.data;

  const leaderboard = run && (
    <ModelLeaderboard run={run} entityLabel="Model" columns={[
      { key: 'roc_auc', label: 'ROC-AUC', format: (v) => v.toFixed(4) },
      { key: 'balanced_accuracy', label: 'Balanced accuracy', format: (v) => v.toFixed(3) },
      { key: 'disparate_impact_sex', label: 'Disparate impact (women/men)', format: (v) => v.toFixed(2) },
      { key: 'equal_opportunity_gap_sex', label: 'Equal-opportunity gap', goodWhenUp: false, format: (v) => v.toFixed(3) },
    ]} />
  );

  return (
    <AppShell meta={meta.data} provenance={provenance.data} tabs={TABS} active={tab}
              onTabChange={setTab} repoUrl="https://github.com/aniket-alt/ai-coding-assistant">
      {tab === 'crispdm' && crispdm.data && (
        <CrispDmWalkthrough report={crispdm.data} slots={{ evaluation: leaderboard || undefined }} />
      )}

      {tab === 'eda' && eda.data && (
        <>
          <Panel title="The gap before any model" description="Share of people earning more than $50,000.">
            <StatRow>
              <StatTile label="Everyone" value={eda.data.high_income_pct.toFixed(1)} unit="%" size="hero"
                        accent="var(--series-1)" hint={`${eda.data.rows.toLocaleString()} people`} />
              {eda.data.by_sex.map((s) => (
                <StatTile key={s.group} label={s.group === 'Male' ? 'Men' : 'Women'} value={s.high_income_pct.toFixed(1)} unit="%"
                          hint={`${s.people.toLocaleString()} people`} />
              ))}
              <StatTile label="Top-coded capital gains" value={eda.data.capital_gain_capped}
                        hint="People reporting exactly $99,999, the recording ceiling." />
            </StatRow>
          </Panel>
          <Panel title="Income by age" description="Share earning above $50K in each ten-year band.">
            <Chart kind="bar" height={260} data={eda.data.age_hist} x="age_band" xLabel="Age"
                   series={[{ key: 'high_income_pct', label: 'Earning > $50K (%)' }]} yFormat={(v) => `${v.toFixed(0)}%`}
                   caption="Earnings peak in middle age; the youngest group almost never crosses the threshold." />
          </Panel>
          <Panel title="Income by race" description="As recorded in the 1994 census categories.">
            <Chart kind="bar" horizontal height={240} data={eda.data.by_race} x="group"
                   series={[{ key: 'high_income_pct', label: 'Earning > $50K (%)' }]} yFormat={(v) => `${v.toFixed(1)}%`}
                   caption="Group sizes differ by orders of magnitude; the table view shows how many people each bar represents." />
          </Panel>
        </>
      )}

      {tab === 'unsupervised' && uns.data && (
        <>
          <Panel title="Four clusters" description={`k-means on the numeric columns, silhouette ${uns.data.silhouette.toFixed(3)}. Income was not used to form them.`}>
            <div style={{ overflowX: 'auto' }}>
              <table style={tableStyle}>
                <thead><tr style={headRow}>{['Cluster', 'People', 'Median age', 'Median hours', 'Education (years)', 'With capital gain', 'Earning > $50K']
                  .map((h, i) => <th key={h} style={{ ...th, textAlign: i ? 'right' : 'left' }}>{h}</th>)}</tr></thead>
                <tbody>{uns.data.clusters.map((c) => (
                  <tr key={c.cluster} style={bodyRow}>
                    <td style={td}>{c.cluster}</td><td style={num}>{c.people.toLocaleString()}</td><td style={num}>{c.median_age}</td>
                    <td style={num}>{c.median_hours}</td><td style={num}>{c.median_education_years}</td>
                    <td style={num}>{c.with_capital_gain_pct}%</td><td style={{ ...num, fontWeight: 700 }}>{c.high_income_pct}%</td>
                  </tr>))}
                </tbody>
              </table>
            </div>
          </Panel>
          <Panel title="What the outlier detector found" description={`Isolation forest, most unusual 1% of training people (${uns.data.outliers.flagged}).`}>
            <StatRow>
              <StatTile label="Have a capital gain or loss" value={uns.data.outliers.any_capital_pct} unit="%" size="hero"
                        accent="var(--series-1)" hint={`Against ${uns.data.outliers.any_capital_rest_pct}% of everyone else.`} />
              <StatTile label="Aged 70 or older" value={uns.data.outliers.age70_pct} unit="%"
                        hint={`Against ${uns.data.outliers.age70_rest_pct}% of everyone else.`} />
              <StatTile label="Top-coded $99,999 gains" value={uns.data.outliers.top_coded_share_pct} unit="%" />
            </StatRow>
            <p style={lesson}>Both capital columns are zero for almost everyone, so any non-zero value looks unusual. The detector
              is mostly describing how those columns are distributed. Deciding whether that makes a person interesting is a
              human judgement.</p>
          </Panel>
        </>
      )}

      {tab === 'fairness' && fair.data && run && (
        <>
          <Panel title="Accurate, and unequal" description="Each model at a 0.5 threshold, with and without sex and race as inputs.">
            {leaderboard}
          </Panel>
          <Panel title="Removing sex from the inputs" description="Disparate impact: women's selection rate divided by men's. Below 0.8 fails the four-fifths rule.">
            <Chart kind="bar" horizontal height={240} data={fair.data.models} x="model"
                   series={[{ key: 'disparate_impact_sex', label: 'Disparate impact ratio' }]} yFormat={(v) => v.toFixed(2)}
                   caption="Models without sex or race among their inputs are almost as unequal as those with them." />
            <StatRow>
              <StatTile label="Sex predicted from the other features" value={fair.data.proxy_auc.toFixed(3)} unit="ROC-AUC"
                        size="hero" accent="var(--series-1)" hint="Using only the inputs the unaware models keep." />
              {fair.data.relationship_proxy.slice(0, 2).map((p) => (
                <StatTile key={p.relationship} label={`“${p.relationship}”`} value={p.men_pct.toFixed(1)} unit="% men"
                          hint={`${p.people.toLocaleString()} people`} />
              ))}
            </StatRow>
            <p style={lesson}>Leaving a protected attribute out of a model does not remove it from the model. Relationship status
              records it almost exactly. Fairness has to be measured on the outputs, not assumed from the inputs.</p>
          </Panel>
        </>
      )}

      {tab === 'rules' && rules.data && (
        <Panel title="What goes with high income?" description="Strongest association rules on training people, rechecked on test people.">
          <div style={{ overflowX: 'auto' }}>
            <table style={tableStyle}>
              <thead><tr style={headRow}>{['If', 'Support', 'Confidence', 'Lift', 'Test people', 'Test confidence']
                .map((h, i) => <th key={h} style={{ ...th, textAlign: i ? 'right' : 'left' }}>{h}</th>)}</tr></thead>
              <tbody>{rules.data.map((r) => (
                <tr key={r.if} style={bodyRow}>
                  <td style={{ ...td, whiteSpace: 'normal' }}>{r.if} → <strong>income &gt; $50K</strong></td>
                  <td style={num}>{r.support_pct}%</td><td style={num}>{r.confidence.toFixed(2)}</td>
                  <td style={num}>{r.lift.toFixed(2)}</td><td style={num}>{r.test_people.toLocaleString()}</td>
                  <td style={{ ...num, fontWeight: 700 }}>{r.test_confidence === null ? '—' : r.test_confidence.toFixed(2)}</td>
                </tr>))}
              </tbody>
            </table>
          </div>
        </Panel>
      )}

      {tab === 'lsh' && lsh.data && (
        <Panel title="Finding similar people fast" description={`Random-hyperplane LSH: ${lsh.data.tables} tables of ${lsh.data.bits}-bit hashes over ${lsh.data.dimensions} dimensions.`}>
          <StatRow>
            <StatTile label="Recall @ 10" value={(lsh.data.recall_at_10 * 100).toFixed(1)} unit="%" size="hero" accent="var(--series-1)"
                      hint={`Share of each person's ten true nearest neighbours found, over ${lsh.data.queries} test queries.`} />
            <StatTile label="People scanned" value={lsh.data.scanned_pct.toFixed(1)} unit="%" hint={`Of ${lsh.data.indexed.toLocaleString()} indexed.`} />
            <StatTile label="LSH query" value={lsh.data.lsh_ms_per_query.toFixed(2)} unit="ms" />
            <StatTile label="Exact query" value={lsh.data.exact_ms_per_query.toFixed(2)} unit="ms" />
          </StatRow>
          <p style={lesson}>Hashing trades a little recall for scanning a small fraction of the data. At this size exact search is
            already fast; the advantage grows with the index, which is why LSH matters for millions of records rather than
            thousands. Timings are from the machine that ran training.</p>
        </Panel>
      )}

      {tab === 'quiz' && (
        <Panel title="Check your understanding" description="Every answer is explained with this run's own numbers.">
          <Quiz />
        </Panel>
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

const tableStyle: React.CSSProperties = { width: '100%', minWidth: 680, borderCollapse: 'collapse', fontSize: 'var(--text-sm)' };
const headRow: React.CSSProperties = { borderBottom: '2px solid var(--axis)' };
const bodyRow: React.CSSProperties = { borderBottom: '1px solid var(--gridline)' };
const th: React.CSSProperties = { padding: '8px 10px', color: 'var(--text-muted)', fontWeight: 600, fontSize: 'var(--text-xs)',
  textTransform: 'uppercase', letterSpacing: 0.3, whiteSpace: 'nowrap' };
const td: React.CSSProperties = { padding: '9px 10px', color: 'var(--text-primary)', whiteSpace: 'nowrap' };
const num: React.CSSProperties = { ...td, textAlign: 'right' };
const lesson: React.CSSProperties = { margin: 'var(--space-4) 0 0', color: 'var(--text-secondary)', maxWidth: 780, lineHeight: 1.6 };
