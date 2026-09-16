/**
 * Project 03 client.
 *
 * SegmentExplorer is the only project-specific view. The customer assignment
 * form is the shared schema-driven PredictionForm, with slider bounds taken
 * from the training data's own percentiles.
 */

import { useState } from 'react';
import {
  AppShell, AuditPanel, CrispDmWalkthrough, MetricsPanel, Panel, PredictionForm,
  ProvenanceBadge, StatRow, StatTile, useApi, useAudit, useCrispDm, useMeta,
  useMetrics, useProvenance,
} from '@dsfs/ui';
import type { FieldSpec, Tab } from '@dsfs/ui';
import { SegmentExplorer } from './SegmentExplorer';
import type { SegmentProfile } from './SegmentExplorer';

const TABS: Tab[] = [
  { id: 'segments', label: 'Segments' },
  { id: 'assign', label: 'Assign a customer' },
  { id: 'crispdm', label: 'CRISP-DM' },
  { id: 'audit', label: 'Audit' },
];

type Ranges = Record<string, { p5: number; median: number; p95: number }>;

interface Assignment {
  model: string;
  segment: SegmentProfile;
  distance: number;
  distances: Record<string, number>;
}

const FIELD_LABELS: Record<string, [string, string?]> = {
  recency_days: ['Days since last order', 'days'],
  frequency: ['Orders in the year'],
  monetary: ['Total spend (£)'],
  avg_basket: ['Average basket (£)'],
  n_products: ['Distinct products'],
  tenure_days: ['Days since first order', 'days'],
  return_rate: ['Return rate'],
};

export function App() {
  const [tab, setTab] = useState('segments');
  const meta = useMeta();
  const metrics = useMetrics();
  const crispdm = useCrispDm();
  const audit = useAudit();
  const provenance = useProvenance();
  const ranges = useApi<Ranges>('/api/feature-ranges');
  const run = metrics.data;

  const fields: FieldSpec[] = ranges.data
    ? Object.entries(FIELD_LABELS).map(([key, [label, unit]]) => {
        const r = ranges.data![key];
        if (key === 'return_rate') {
          return { kind: 'slider', name: key, label, min: 0, max: 0.5, step: 0.01 };
        }
        return {
          kind: 'slider', name: key, label, unit,
          min: Math.max(key === 'frequency' || key === 'n_products' ? 1 : 0, Math.floor(r.p5)),
          max: Math.ceil(r.p95),
          step: r.p95 > 500 ? 10 : 1,
        };
      })
    : [];

  const initial = ranges.data
    ? Object.fromEntries(Object.keys(FIELD_LABELS).map((k) => [k, Math.round(ranges.data![k].median * 100) / 100]))
    : {};

  return (
    <AppShell meta={meta.data} provenance={provenance.data} tabs={TABS} active={tab}
              onTabChange={setTab} repoUrl="https://github.com/aniket-alt/ai-coding-assistant">
      {tab === 'segments' && run && (
        <>
          <Panel title="Segmentations compared"
                 description="Select a row to see its segments. Ranked by how much next-year spend the segments explain.">
            <SegmentExplorer run={run} />
          </Panel>
          <Panel title="Winning segmentation"
                 description="Judged on behaviour in a year it never saw.">
            <MetricsPanel run={run} showBaselineLift={false} highlight={[
              { key: 'omega2_future', label: 'ω² future spend', format: (v) => v.toFixed(4),
                hint: 'Variance in next-year spend explained, corrected for the number of segments.' },
              { key: 'stability_ari', label: 'Stability', format: (v) => v.toFixed(3),
                hint: 'Agreement when rebuilt on different subsamples (1 = identical).' },
              { key: 'retention_spread', label: 'Retention spread', unit: 'pts', format: (v) => v.toFixed(1),
                hint: 'Gap in repeat-purchase rate between the most and least loyal segment.' },
              { key: 'silhouette', label: 'Silhouette', format: (v) => v.toFixed(3),
                hint: 'Geometric tidiness. Reported, but not used for ranking.' },
            ]} />
          </Panel>
        </>
      )}

      {tab === 'assign' && (
        <Panel title="Which segment is this customer in?"
               description="Assigned to the nearest segment of the winning segmentation. Bounds span the 5th to 95th percentile of real customers.">
          {ranges.data ? (
            <PredictionForm
              key="assign"
              fields={fields}
              endpoint="/api/assign"
              initialValues={initial}
              autoSubmit
              renderResult={(r, { loading }) => {
                const a = r as Assignment | null;
                if (!a) return <p style={{ color: 'var(--text-muted)' }}>{loading ? 'Assigning…' : ''}</p>;
                const s = a.segment;
                return (
                  <div style={{ opacity: loading ? 0.55 : 1 }}>
                    <StatRow>
                      <StatTile label="Segment" value={s.label} size="hero" accent="var(--series-1)" hint={s.tag} />
                      <StatTile label="Bought again next year" value={s.retained_pct.toFixed(1)} unit="%"
                                hint={`Across the ${s.customers.toLocaleString()} customers in this segment.`} />
                      <StatTile label="Mean next-year revenue"
                                value={`£${s.mean_future_revenue.toLocaleString(undefined, { maximumFractionDigits: 0 })}`} />
                      <StatTile label="Distance to centre" value={a.distance.toFixed(2)}
                                hint="In standardised units. Larger means a less typical member." />
                    </StatRow>
                    <p style={{ marginTop: 'var(--space-3)', color: 'var(--text-muted)', fontSize: 'var(--text-xs)' }}>
                      Served segmentation: {a.model}. Distances to every segment:{' '}
                      {Object.entries(a.distances).map(([k, v]) => `${k} ${v.toFixed(2)}`).join(' · ')}
                    </p>
                  </div>
                );
              }}
            />
          ) : (
            <p style={{ color: 'var(--text-muted)' }}>{ranges.error ?? 'Loading feature ranges…'}</p>
          )}
        </Panel>
      )}

      {tab === 'crispdm' && crispdm.data && (
        <CrispDmWalkthrough report={crispdm.data} slots={{
          modeling: run ? <SegmentExplorer run={run} /> : undefined,
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
