/**
 * Leaderboard -- the model tournament table.
 *
 * Reads straight from the RunRecord the training script wrote, so the ranking
 * on screen is the ranking that was measured. Doubles as the *table view* that
 * the light-mode contrast relief rule requires for the accompanying charts.
 */

import { useMemo, useState } from 'react';
import type { ModelScore, RunRecord } from '../types';

const SERIES = [
  'var(--series-1)', 'var(--series-2)', 'var(--series-3)', 'var(--series-4)',
  'var(--series-5)', 'var(--series-6)', 'var(--series-7)', 'var(--series-8)',
];

/** Fixed slot order, never cycled: past 8 models the swatch is neutral. */
export function seriesColor(i: number): string {
  return i < SERIES.length ? SERIES[i] : 'var(--text-muted)';
}

function fmt(v: number | undefined): string {
  if (v === undefined || v === null || Number.isNaN(v)) return '—';
  const a = Math.abs(v);
  if (a !== 0 && a < 0.001) return v.toExponential(2);
  if (a >= 1e4) return v.toLocaleString(undefined, { maximumFractionDigits: 0 });
  return v.toFixed(4).replace(/\.?0+$/, '');
}

export interface LeaderboardProps {
  run: RunRecord;
  /** Extra metric columns beyond the primary one. */
  columns?: string[];
  showTiming?: boolean;
  maxRows?: number;
}

export function Leaderboard({
  run,
  columns,
  showTiming = true,
  maxRows,
}: LeaderboardProps) {
  const [sortBy, setSortBy] = useState<string>(run.primary_metric);

  const metricCols = useMemo(() => {
    if (columns?.length) return [run.primary_metric, ...columns.filter((c) => c !== run.primary_metric)];
    const seen = new Set<string>([run.primary_metric]);
    run.models.forEach((m) => Object.keys(m.metrics).forEach((k) => seen.add(k)));
    return [...seen].slice(0, 5);
  }, [run, columns]);

  const rows = useMemo(() => {
    const scored = run.models.filter((m) => sortBy in m.metrics);
    // Only the primary metric knows its own direction; secondary sorts
    // default to descending.
    const desc = sortBy === run.primary_metric ? run.higher_is_better : true;
    const out = [...scored].sort((a, b) =>
      desc ? b.metrics[sortBy] - a.metrics[sortBy] : a.metrics[sortBy] - b.metrics[sortBy],
    );
    return maxRows ? out.slice(0, maxRows) : out;
  }, [run, sortBy, maxRows]);

  const bestName = run.best_model;

  return (
    <div style={{ overflowX: 'auto' }}>
      <table
        style={{
          width: '100%',
          minWidth: 520,
          borderCollapse: 'collapse',
          fontSize: 'var(--text-sm)',
        }}
      >
        <caption
          style={{
            captionSide: 'top',
            textAlign: 'left',
            color: 'var(--text-secondary)',
            fontSize: 'var(--text-xs)',
            paddingBottom: 'var(--space-2)',
          }}
        >
          {run.models.length} models evaluated on {run.split.n_test.toLocaleString()} held-out
          rows · sorted by <strong>{sortBy}</strong>
          {sortBy === run.primary_metric &&
            ` (${run.higher_is_better ? 'higher' : 'lower'} is better)`}
        </caption>
        <thead>
          <tr style={{ borderBottom: `2px solid var(--axis)` }}>
            <th style={{ ...th, width: 34 }}>#</th>
            <th style={th}>Model</th>
            {metricCols.map((c) => (
              <th key={c} style={{ ...th, textAlign: 'right' }}>
                <button
                  type="button"
                  onClick={() => setSortBy(c)}
                  aria-label={`Sort by ${c}`}
                  style={{
                    background: 'none',
                    border: 'none',
                    padding: 0,
                    cursor: 'pointer',
                    font: 'inherit',
                    color: c === sortBy ? 'var(--text-primary)' : 'var(--text-muted)',
                    fontWeight: c === sortBy ? 700 : 600,
                  }}
                >
                  {c}
                  {c === sortBy ? ' ▾' : ''}
                </button>
              </th>
            ))}
            {showTiming && <th style={{ ...th, textAlign: 'right' }}>fit&nbsp;(s)</th>}
          </tr>
        </thead>
        <tbody>
          {rows.map((m: ModelScore, i) => {
            const isBest = m.name === bestName;
            return (
              <tr
                key={m.name}
                style={{
                  borderBottom: '1px solid var(--gridline)',
                  background: isBest ? 'var(--surface-2)' : 'transparent',
                }}
              >
                <td style={{ ...td, color: 'var(--text-muted)' }}>{i + 1}</td>
                <td style={td}>
                  <span style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
                    <span
                      aria-hidden
                      style={{
                        width: 8,
                        height: 8,
                        borderRadius: 2,
                        background: seriesColor(i),
                        flexShrink: 0,
                      }}
                    />
                    <span style={{ fontWeight: isBest ? 700 : 500 }}>{m.name}</span>
                    {isBest && (
                      <span
                        style={{
                          fontSize: 'var(--text-xs)',
                          fontWeight: 700,
                          color: 'var(--delta-up)',
                          border: '1px solid currentColor',
                          borderRadius: 'var(--radius-sm)',
                          padding: '0 5px',
                        }}
                      >
                        BEST
                      </span>
                    )}
                    {m.family && (
                      <span style={{ color: 'var(--text-muted)', fontSize: 'var(--text-xs)' }}>
                        {m.family}
                      </span>
                    )}
                  </span>
                </td>
                {metricCols.map((c) => (
                  <td key={c} style={{ ...td, textAlign: 'right' }}>
                    {fmt(m.metrics[c])}
                  </td>
                ))}
                {showTiming && (
                  <td style={{ ...td, textAlign: 'right', color: 'var(--text-muted)' }}>
                    {m.fit_seconds ? m.fit_seconds.toFixed(1) : '—'}
                  </td>
                )}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

const th: React.CSSProperties = {
  textAlign: 'left',
  padding: '8px 10px',
  color: 'var(--text-muted)',
  fontWeight: 600,
  fontSize: 'var(--text-xs)',
  textTransform: 'uppercase',
  letterSpacing: 0.3,
  whiteSpace: 'nowrap',
};

const td: React.CSSProperties = {
  padding: '9px 10px',
  color: 'var(--text-primary)',
  whiteSpace: 'nowrap',
};
