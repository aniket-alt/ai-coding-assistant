/**
 * SegmentExplorer -- project-03-specific.
 *
 * Choosing a row in the shared leaderboard loads that segmentation's profiles,
 * so the comparison is between whole segmentations rather than one winner.
 * Every figure is a field of the /api/segments response.
 */

import { useEffect, useState } from 'react';
import { Chart, ModelLeaderboard, apiGet } from '@dsfs/ui';
import type { ModelScore, RunRecord } from '@dsfs/ui';

export interface SegmentProfile {
  segment_id: number;
  label: string;
  tag: string;
  customers: number;
  share_pct: number;
  median_recency_days: number;
  median_orders: number;
  median_spend: number;
  median_basket: number;
  return_rate_pct: number;
  retained_pct: number;
  median_future_revenue: number;
  mean_future_revenue: number;
}

interface SegmentsResponse {
  model: string;
  served: boolean;
  segments: SegmentProfile[];
}

const money = (v: number) => `£${v.toLocaleString(undefined, { maximumFractionDigits: 0 })}`;

export function SegmentExplorer({ run }: { run: RunRecord }) {
  const [model, setModel] = useState<string>(run.best_model ?? '');
  const [data, setData] = useState<SegmentsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!model) return;
    setError(null);
    apiGet<SegmentsResponse>(`/api/segments?model=${encodeURIComponent(model)}`)
      .then(setData)
      .catch((e: Error) => setError(e.message));
  }, [model]);

  const rows = data?.segments ?? [];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-5)' }}>
      <ModelLeaderboard
        run={run}
        entityLabel="Segmentation"
        selectable
        selectedModel={model}
        onSelect={(m: ModelScore) => setModel(m.name)}
        columns={[
          { key: 'omega2_future', label: 'ω² future spend', format: (v) => v.toFixed(4) },
          { key: 'stability_ari', label: 'Stability', format: (v) => v.toFixed(3) },
          { key: 'silhouette', label: 'Silhouette', format: (v) => v.toFixed(3) },
          { key: 'retention_spread', label: 'Retention spread', format: (v) => `${v.toFixed(1)} pts` },
          { key: 'n_segments', label: 'k', format: (v) => v.toFixed(0) },
        ]}
      />

      {error && <p role="alert" style={{ color: 'var(--status-critical)' }}>{error}</p>}

      {data && (
        <>
          <h3 style={{ margin: 0, fontSize: 'var(--text-lg)' }}>
            {data.model}
            {data.served && (
              <span style={{
                marginLeft: 8, fontSize: 'var(--text-xs)', color: 'var(--delta-up)',
                border: '1px solid currentColor', borderRadius: 'var(--radius-sm)', padding: '0 5px',
              }}>
                SERVED
              </span>
            )}
          </h3>

          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', minWidth: 760, borderCollapse: 'collapse', fontSize: 'var(--text-sm)' }}>
              <thead>
                <tr style={{ borderBottom: '2px solid var(--axis)' }}>
                  {['Segment', 'Profile', 'Customers', 'Recency (days)', 'Orders', 'Spend',
                    'Basket', 'Returns', 'Bought again', 'Next-year revenue (mean)'].map((h, i) => (
                    <th key={h} style={{ ...th, textAlign: i < 2 ? 'left' : 'right' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((s) => (
                  <tr key={s.segment_id} style={{ borderBottom: '1px solid var(--gridline)' }}>
                    <td style={{ ...td, fontWeight: 600 }}>{s.label}</td>
                    <td style={{ ...td, color: 'var(--text-secondary)' }}>{s.tag}</td>
                    <td style={num}>{s.customers.toLocaleString()} <span style={muted}>({s.share_pct}%)</span></td>
                    <td style={num}>{s.median_recency_days.toFixed(0)}</td>
                    <td style={num}>{s.median_orders.toFixed(0)}</td>
                    <td style={num}>{money(s.median_spend)}</td>
                    <td style={num}>{money(s.median_basket)}</td>
                    <td style={num}>{s.return_rate_pct.toFixed(1)}%</td>
                    <td style={{ ...num, fontWeight: 600 }}>{s.retained_pct.toFixed(1)}%</td>
                    <td style={num}>{money(s.mean_future_revenue)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p style={{ ...muted, marginTop: 'var(--space-2)' }}>
              Medians of first-year behaviour. The last two columns are what each segment did in the
              following year, which played no part in building it.
            </p>
          </div>

          <div style={{
            display: 'grid', gap: 'var(--space-5)',
            gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))',
          }}>
            <Chart
              kind="bar" height={240} data={rows} x="label"
              series={[{ key: 'retained_pct', label: 'Bought again next year (%)' }]}
              yFormat={(v) => `${v.toFixed(0)}%`}
              caption="Share of each segment that made at least one purchase in the following year."
            />
            <Chart
              kind="bar" height={240} data={rows} x="label"
              series={[{ key: 'mean_future_revenue', label: 'Mean next-year revenue (£)' }]}
              yFormat={(v) => money(v)}
              caption="Average revenue per customer in the following year, including those who never returned."
            />
          </div>
        </>
      )}
    </div>
  );
}

const th: React.CSSProperties = {
  padding: '8px 10px', color: 'var(--text-muted)', fontWeight: 600,
  fontSize: 'var(--text-xs)', textTransform: 'uppercase', letterSpacing: 0.3, whiteSpace: 'nowrap',
};
const td: React.CSSProperties = { padding: '9px 10px', color: 'var(--text-primary)', whiteSpace: 'nowrap' };
const num: React.CSSProperties = { ...td, textAlign: 'right' };
const muted: React.CSSProperties = { color: 'var(--text-muted)', fontSize: 'var(--text-xs)' };
