/**
 * Project 04 client.
 *
 * RuleTable is the only project-specific view. The basket builder is the
 * shared PredictionForm with a multiselect fed by /api/items, and the
 * tournament is the shared ModelLeaderboard -- which fits here because every
 * method, rules included, is judged by the same held-out-item test.
 */

import { useState } from 'react';
import {
  AppShell, AuditPanel, Chart, CrispDmWalkthrough, MetricsPanel, ModelLeaderboard, Panel,
  PredictionForm, ProvenanceBadge, useApi, useAudit, useCrispDm, useMeta, useMetrics, useProvenance,
} from '@dsfs/ui';
import type { Tab } from '@dsfs/ui';
import { RuleTable } from './RuleTable';
import type { Rule } from './RuleTable';

const TABS: Tab[] = [
  { id: 'recommend', label: 'Build a basket' },
  { id: 'rules', label: 'Rules' },
  { id: 'tournament', label: 'Recommender tournament' },
  { id: 'crispdm', label: 'CRISP-DM' },
  { id: 'audit', label: 'Audit' },
];

interface Suggestion {
  code: string; name: string; source: string; because?: string; probability?: number;
}
interface Recommendation {
  model: string;
  basket: { code: string; name: string }[];
  suggestions: Suggestion[];
}
interface Item { id: string; label: string }

const pct = (v: number) => `${(v * 100).toFixed(2)}%`;

export function App() {
  const [tab, setTab] = useState('recommend');
  const meta = useMeta();
  const metrics = useMetrics();
  const crispdm = useCrispDm();
  const audit = useAudit();
  const provenance = useProvenance();
  const rules = useApi<Rule[]>('/api/rules');
  const items = useApi<Item[]>('/api/items');
  const run = metrics.data;

  const leaderboard = run && (
    <ModelLeaderboard
      run={run}
      entityLabel="Recommender"
      columns={[
        { key: 'hit_rate_at_5', label: 'Hit rate @ 5', format: pct },
        { key: 'mrr_at_5', label: 'MRR @ 5', format: (v) => v.toFixed(4) },
        { key: 'coverage_pct', label: 'Coverage', format: (v) => `${v.toFixed(1)}%` },
        { key: 'n_rules', label: 'Rules', format: (v) => v.toLocaleString() },
        { key: 'rule_survival_pct', label: 'Rules held', format: (v) => `${v.toFixed(1)}%` },
      ]}
      showParams
    />
  );

  return (
    <AppShell meta={meta.data} provenance={provenance.data} tabs={TABS} active={tab}
              onTabChange={setTab} repoUrl="https://github.com/aniket-alt/ai-coding-assistant">
      {tab === 'recommend' && (
        <>
          <Panel title="What else belongs in this basket?"
                 description="Pick products (Ctrl or Cmd to select several). Suggestions come from the tournament winner.">
            {items.data ? (
              <PredictionForm
                fields={[{ kind: 'multiselect', name: 'items', label: 'Basket', options: '/api/items', max: 20 }]}
                endpoint="/api/recommend"
                initialValues={{ items: [items.data[0].id, items.data[3].id] }}
                autoSubmit
                renderResult={(r, { loading }) => {
                  const rec = r as Recommendation | null;
                  if (!rec) return <p style={{ color: 'var(--text-muted)' }}>{loading ? 'Recommending…' : ''}</p>;
                  return (
                    <div style={{ opacity: loading ? 0.55 : 1 }}>
                      <h3 style={{ margin: '0 0 var(--space-2)', fontSize: 'var(--text-lg)' }}>Suggested additions</h3>
                      <ol style={{ margin: 0, paddingLeft: 'var(--space-5)' }}>
                        {rec.suggestions.map((s) => (
                          <li key={s.code} style={{ padding: '6px 0', borderBottom: '1px solid var(--gridline)' }}>
                            <strong>{s.name}</strong>
                            {s.because && (
                              <span style={{ color: 'var(--text-secondary)', fontSize: 'var(--text-sm)' }}>
                                {' '}— bought in {(s.probability! * 100).toFixed(0)}% of baskets containing {s.because}
                              </span>
                            )}
                            {s.source !== 'model' && (
                              <span style={{ color: 'var(--text-muted)', fontSize: 'var(--text-xs)' }}> ({s.source})</span>
                            )}
                          </li>
                        ))}
                      </ol>
                      <p style={{ marginTop: 'var(--space-2)', color: 'var(--text-muted)', fontSize: 'var(--text-xs)' }}>
                        Served recommender: {rec.model}.
                      </p>
                    </div>
                  );
                }}
              />
            ) : (
              <p style={{ color: 'var(--text-muted)' }}>{items.error ?? 'Loading products…'}</p>
            )}
          </Panel>
          {run && (
            <Panel title="How well it recommends" description="Hidden-item test on the following Christmas season.">
              <MetricsPanel run={run} showBaselineLift={false} highlight={[
                { key: 'hit_rate_at_5', label: 'Hit rate @ 5', format: (v) => (v * 100).toFixed(2), unit: '%',
                  hint: 'How often the hidden product was among five suggestions.' },
                { key: 'mrr_at_5', label: 'MRR @ 5', format: (v) => v.toFixed(4),
                  hint: 'Rewards putting the right product nearer the top.' },
                { key: 'coverage_pct', label: 'Coverage', unit: '%', format: (v) => v.toFixed(1),
                  hint: 'Share of baskets where the method suggested anything itself.' },
                { key: 'catalogue_coverage_pct', label: 'Catalogue reach', unit: '%', format: (v) => v.toFixed(1),
                  hint: 'Share of all products it ever recommended.' },
              ]} />
            </Panel>
          )}
        </>
      )}

      {tab === 'rules' && (
        <Panel title="Rules, and whether they held"
               description="Strongest rules from the most permissive rule set, recomputed on the following season.">
          {rules.data ? <RuleTable rules={rules.data} /> : <p>{rules.error ?? 'Loading rules…'}</p>}
        </Panel>
      )}

      {tab === 'tournament' && run && (
        <>
          <Panel title="Recommender tournament"
                 description="Every method recovers a hidden product from later baskets. Coverage shows how often a method said anything itself.">
            {leaderboard}
          </Panel>
          <Panel title="Hit rate by method" description="Baselines included.">
            <Chart kind="bar" horizontal height={300}
                   data={run.models.map((m) => ({ model: m.name, hit: m.metrics.hit_rate_at_5 * 100 }))}
                   x="model" series={[{ key: 'hit', label: 'Hit rate @ 5 (%)' }]}
                   highlightKey={run.best_model ?? undefined}
                   yFormat={(v) => `${v.toFixed(1)}%`}
                   caption="Share of held-out products recovered within five suggestions. The winner is highlighted." />
          </Panel>
        </>
      )}

      {tab === 'crispdm' && crispdm.data && (
        <CrispDmWalkthrough report={crispdm.data} slots={{
          modeling: leaderboard || undefined,
          evaluation: rules.data ? <RuleTable rules={rules.data.slice(0, 10)} /> : undefined,
        }} />
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
