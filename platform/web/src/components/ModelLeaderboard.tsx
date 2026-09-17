/**
 * ModelLeaderboard -- the tournament table, for any project.
 *
 * This component owns baselines: they are rows here, ranked alongside every
 * learned model, so a reader can see exactly what the model had to beat.
 * MetricsPanel reports the already-computed lift and does not re-derive it,
 * which keeps the two from disagreeing about what the baseline was.
 *
 * Every figure comes from the run record. Sorting reorders rows; it computes
 * nothing. `selectable` exists because clustering needs "click k=5 to see that
 * segmentation" and forecasting needs "click a backbone to see its forecast".
 */

import { useMemo, useState } from 'react';
import type { ModelScore, RunRecord } from '../types';

/**
 * One colour per model for the whole page, keyed by name.
 *
 * Slots follow the order training recorded the models in, not the current
 * sort, so a model keeps its colour when the table is re-sorted and a chart
 * of the same models can reuse the map. Baselines are muted.
 */
export function modelColors(run: RunRecord, baselineFamilies: string[] = ['baseline']): Record<string, string> {
  const out: Record<string, string> = {};
  let slot = 0;
  for (const m of run.models) {
    out[m.name] = baselineFamilies.includes((m.family ?? '').toLowerCase())
      ? 'var(--text-muted)' : seriesColor(slot++);
  }
  return out;
}
import { seriesColor } from './Chart';

export interface ColumnSpec {
  key: string;
  label?: string;
  unit?: string;
  format?: (v: number) => string;
  goodWhenUp?: boolean;
}

export interface ModelLeaderboardProps {
  run: RunRecord;
  columns?: (string | ColumnSpec)[];
  /** 'Model' | 'Algorithm' | 'Configuration' -- header wording. */
  entityLabel?: string;
  selectable?: boolean;
  selectedModel?: string | null;
  onSelect?: (m: ModelScore) => void;
  showParams?: boolean;
  showTiming?: boolean;
  maxRows?: number;
  /** Families rendered as baselines rather than competitors. */
  baselineFamilies?: string[];
}

const defaultFormat = (v: number | undefined): string => {
  if (v === undefined || v === null || Number.isNaN(v)) return '—';
  const a = Math.abs(v);
  if (a !== 0 && a < 0.001) return v.toExponential(2);
  if (a >= 1e4) return v.toLocaleString(undefined, { maximumFractionDigits: 0 });
  return v.toFixed(4).replace(/\.?0+$/, '');
};

function normalise(c: string | ColumnSpec): ColumnSpec {
  return typeof c === 'string' ? { key: c } : c;
}

export function ModelLeaderboard({
  run,
  columns,
  entityLabel = 'Model',
  selectable = false,
  selectedModel = null,
  onSelect,
  showParams = false,
  showTiming = true,
  maxRows,
  baselineFamilies = ['baseline'],
}: ModelLeaderboardProps) {
  const [sortBy, setSortBy] = useState<string>(run.primary_metric);
  const [expanded, setExpanded] = useState<string | null>(null);

  const cols: ColumnSpec[] = useMemo(() => {
    if (columns?.length) {
      const specs = columns.map(normalise);
      return specs.some((c) => c.key === run.primary_metric)
        ? specs
        : [{ key: run.primary_metric }, ...specs];
    }
    const seen = new Set<string>([run.primary_metric]);
    run.models.forEach((m) => Object.keys(m.metrics).forEach((k) => seen.add(k)));
    return [...seen].slice(0, 5).map((key) => ({ key }));
  }, [run, columns]);

  const rows = useMemo(() => {
    const scored = run.models.filter((m) => sortBy in m.metrics);
    const spec = cols.find((c) => c.key === sortBy);
    const desc = sortBy === run.primary_metric
      ? run.higher_is_better
      : spec?.goodWhenUp ?? true;
    const out = [...scored].sort((a, b) =>
      desc ? b.metrics[sortBy] - a.metrics[sortBy] : a.metrics[sortBy] - b.metrics[sortBy],
    );
    return maxRows ? out.slice(0, maxRows) : out;
  }, [run, sortBy, cols, maxRows]);

  const isBaseline = (m: ModelScore) =>
    baselineFamilies.includes((m.family ?? '').toLowerCase());

  const nBaseline = run.models.filter(isBaseline).length;
  const colors = useMemo(() => modelColors(run, baselineFamilies), [run, baselineFamilies]);

  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{
        width: '100%', minWidth: 560, borderCollapse: 'collapse',
        fontSize: 'var(--text-sm)',
      }}>
        <caption style={{
          captionSide: 'top', textAlign: 'left', color: 'var(--text-secondary)',
          fontSize: 'var(--text-xs)', paddingBottom: 'var(--space-2)',
        }}>
          {run.models.length} evaluated
          {run.split?.n_test ? ` on ${run.split.n_test.toLocaleString()} held-out rows` : ''}
          {nBaseline > 0 && `, including ${nBaseline} baseline${nBaseline > 1 ? 's' : ''}`}
          {' · sorted by '}<strong>{sortBy}</strong>
          {sortBy === run.primary_metric &&
            ` (${run.higher_is_better ? 'higher' : 'lower'} is better)`}
        </caption>
        <thead>
          <tr style={{ borderBottom: '2px solid var(--axis)' }}>
            <th style={{ ...th, width: 34 }}>#</th>
            <th style={th}>{entityLabel}</th>
            {cols.map((c) => (
              <th key={c.key} style={{ ...th, textAlign: 'right' }}>
                <button
                  type="button"
                  onClick={() => setSortBy(c.key)}
                  aria-label={`Sort by ${c.label ?? c.key}`}
                  style={{
                    background: 'none', border: 'none', padding: 0, font: 'inherit',
                    cursor: 'pointer',
                    color: c.key === sortBy ? 'var(--text-primary)' : 'var(--text-muted)',
                    fontWeight: c.key === sortBy ? 700 : 600,
                  }}
                >
                  {c.label ?? c.key}{c.unit ? ` (${c.unit})` : ''}{c.key === sortBy ? ' ▾' : ''}
                </button>
              </th>
            ))}
            {showTiming && <th style={{ ...th, textAlign: 'right' }}>fit&nbsp;(s)</th>}
          </tr>
        </thead>
        <tbody>
          {rows.map((m, i) => {
            const best = m.name === run.best_model;
            const base = isBaseline(m);
            const selected = m.name === selectedModel;
            const open = expanded === m.name;
            return (
              <>
                <tr
                  key={m.name}
                  onClick={selectable ? () => onSelect?.(m) : undefined}
                  aria-selected={selectable ? selected : undefined}
                  style={{
                    borderBottom: '1px solid var(--gridline)',
                    background: selected
                      ? 'var(--surface-2)'
                      : best ? 'var(--surface-2)' : 'transparent',
                    borderLeft: selected
                      ? '3px solid var(--series-1)' : '3px solid transparent',
                    cursor: selectable ? 'pointer' : 'default',
                    opacity: base ? 0.82 : 1,
                  }}
                >
                  <td style={{ ...td, color: 'var(--text-muted)' }}>{i + 1}</td>
                  <td style={td}>
                    <span style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
                      <span aria-hidden style={{
                        width: 8, height: 8, borderRadius: 2, flexShrink: 0,
                        background: colors[m.name],
                      }} />
                      <span style={{ fontWeight: best ? 700 : 500 }}>{m.name}</span>
                      {best && <Tag color="var(--delta-up)">BEST</Tag>}
                      {base && <Tag color="var(--text-muted)">BASELINE</Tag>}
                      {m.family && !base && (
                        <span style={{ color: 'var(--text-muted)', fontSize: 'var(--text-xs)' }}>
                          {m.family}
                        </span>
                      )}
                      {showParams && Object.keys(m.params ?? {}).length > 0 && (
                        <button
                          type="button"
                          onClick={(e) => { e.stopPropagation(); setExpanded(open ? null : m.name); }}
                          aria-expanded={open}
                          style={{
                            marginLeft: 4, background: 'none', border: 'none',
                            color: 'var(--text-muted)', cursor: 'pointer',
                            fontSize: 'var(--text-xs)', padding: 0,
                          }}
                        >
                          {open ? '▾ params' : '▸ params'}
                        </button>
                      )}
                    </span>
                  </td>
                  {cols.map((c) => (
                    <td key={c.key} style={{ ...td, textAlign: 'right' }}>
                      {/* A model may not have every metric (baselines have no rule count);
                          custom formatters are only called on real numbers. */}
                      {Number.isFinite(m.metrics[c.key])
                        ? (c.format ?? defaultFormat)(m.metrics[c.key])
                        : '—'}
                    </td>
                  ))}
                  {showTiming && (
                    <td style={{ ...td, textAlign: 'right', color: 'var(--text-muted)' }}>
                      {m.fit_seconds ? m.fit_seconds.toFixed(1) : '—'}
                    </td>
                  )}
                </tr>
                {open && (
                  <tr key={`${m.name}-params`}>
                    <td colSpan={cols.length + (showTiming ? 3 : 2)}
                        style={{ ...td, background: 'var(--surface-2)', whiteSpace: 'normal' }}>
                      <code style={{ fontSize: 'var(--text-xs)', color: 'var(--text-secondary)' }}>
                        {Object.entries(m.params).map(([k, v]) => `${k}=${String(v)}`).join('  ·  ')}
                      </code>
                      {m.notes && (
                        <p style={{ margin: '4px 0 0', color: 'var(--text-secondary)', fontSize: 'var(--text-xs)' }}>
                          {m.notes}
                        </p>
                      )}
                    </td>
                  </tr>
                )}
              </>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function Tag({ children, color }: { children: React.ReactNode; color: string }) {
  return (
    <span style={{
      fontSize: 'var(--text-xs)', fontWeight: 700, color,
      border: '1px solid currentColor', borderRadius: 'var(--radius-sm)',
      padding: '0 5px', lineHeight: 1.5,
    }}>
      {children}
    </span>
  );
}

const th: React.CSSProperties = {
  textAlign: 'left', padding: '8px 10px', color: 'var(--text-muted)',
  fontWeight: 600, fontSize: 'var(--text-xs)', textTransform: 'uppercase',
  letterSpacing: 0.3, whiteSpace: 'nowrap',
};

const td: React.CSSProperties = {
  padding: '9px 10px', color: 'var(--text-primary)', whiteSpace: 'nowrap',
};
