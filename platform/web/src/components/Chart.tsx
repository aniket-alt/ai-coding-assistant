/**
 * Chart -- a thin wrapper over Recharts, not a charting library.
 *
 * Its whole job is to make the portfolio's visualisation rules automatic:
 *
 *  - series colours come from the validated palette in fixed slot order and
 *    are never cycled; past eight series the caller must fold or facet
 *  - a legend is present for two or more series; one series needs none
 *  - direct labels on the leading value, so identity never rests on colour
 *  - a table view accompanies every chart (the light-mode contrast relief
 *    rule obliges it, and it doubles as the accessible fallback)
 *  - one y-axis, always -- dual-axis charts are not expressible here
 *
 * It renders the rows it is given and computes no aggregate of its own. Every
 * value plotted must already exist in an artifact or an API response.
 */

import { useId, useMemo, useState, type ReactNode } from 'react';
import {
  Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, Legend, Line, LineChart,
  ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis,
} from 'recharts';

const SERIES_VARS = [
  '--series-1', '--series-2', '--series-3', '--series-4',
  '--series-5', '--series-6', '--series-7', '--series-8',
];

/** Fixed slot order, never cycled. */
export function seriesColor(i: number): string {
  return i < SERIES_VARS.length
    ? `var(${SERIES_VARS[i]})`
    : 'var(--text-muted)';
}

export interface ChartSeries {
  key: string;
  label: string;
  /** Override the slot colour only when the entity has a fixed identity. */
  color?: string;
}

export interface ChartProps {
  kind: 'line' | 'bar' | 'area' | 'scatter';
  /**
   * Rows exactly as they arrived from the API. Never transformed here.
   *
   * Typed loosely on purpose: callers pass their own declared interfaces, and
   * a plain `Record<string, unknown>[]` would reject every one of them for
   * lacking an index signature.
   */
  data: Record<string, any>[];
  x: string;
  series: ChartSeries[];
  xLabel?: string;
  yLabel?: string;
  xFormat?: (v: unknown) => string;
  yFormat?: (v: number) => string;
  height?: number;
  /** Label the final point of each series directly. Default true for <= 4. */
  directLabels?: boolean;
  /** Offer the table view toggle. Default true; required by the relief rule. */
  tableView?: boolean;
  caption?: string;
  /** Horizontal bars, for long category names. */
  horizontal?: boolean;
  /** Highlight one category in a bar chart (e.g. the winning model). */
  highlightKey?: string;
  emptyMessage?: string;
  children?: ReactNode;
}

const fmtDefault = (v: number) =>
  Math.abs(v) >= 1000 ? v.toLocaleString(undefined, { maximumFractionDigits: 0 })
    : Number.isInteger(v) ? String(v)
      : v.toFixed(2);

export function Chart({
  kind,
  data,
  x,
  series,
  xLabel,
  yLabel,
  xFormat,
  yFormat = fmtDefault,
  height = 280,
  directLabels,
  tableView = true,
  caption,
  horizontal = false,
  highlightKey,
  emptyMessage = 'No data to plot.',
}: ChartProps) {
  const [showTable, setShowTable] = useState(false);
  const titleId = useId();

  const colors = useMemo(
    () => series.map((s, i) => s.color ?? seriesColor(i)),
    [series],
  );

  const showLegend = series.length >= 2;
  const labelPoints = directLabels ?? series.length <= 4;

  if (!data?.length) {
    return (
      <p style={{
        margin: 0, padding: 'var(--space-5)', textAlign: 'center',
        color: 'var(--text-muted)', background: 'var(--surface-2)',
        borderRadius: 'var(--radius)',
      }}>
        {emptyMessage}
      </p>
    );
  }

  const axisProps = {
    stroke: 'var(--axis)',
    tick: { fill: 'var(--text-muted)', fontSize: 11 },
    tickLine: false,
  };

  const tooltip = (
    <Tooltip
      cursor={{ fill: 'var(--surface-2)', stroke: 'var(--axis)' }}
      contentStyle={{
        background: 'var(--surface-1)',
        border: '1px solid var(--border)',
        borderRadius: 'var(--radius)',
        fontSize: 'var(--text-xs)',
        color: 'var(--text-primary)',
        boxShadow: 'var(--shadow)',
      }}
      labelStyle={{ color: 'var(--text-secondary)', fontWeight: 600 }}
      formatter={(v: number, name: string) => [yFormat(v), name]}
    />
  );

  const legend = showLegend ? (
    <Legend
      wrapperStyle={{ fontSize: 'var(--text-xs)', color: 'var(--text-secondary)' }}
      iconType="square"
      iconSize={8}
    />
  ) : null;

  const grid = <CartesianGrid stroke="var(--gridline)" strokeDasharray="0" vertical={false} />;

  // Recharts needs the label as a descriptor, and room at the bottom for it.
  const xAxisLabel = xLabel
    ? { value: xLabel, position: 'insideBottom' as const, offset: -4,
        fill: 'var(--text-muted)', fontSize: 11 }
    : undefined;
  const bottomMargin = xLabel ? 26 : 8;

  function body() {
    if (kind === 'bar') {
      return (
        <BarChart data={data} layout={horizontal ? 'vertical' : 'horizontal'}
                  margin={{ top: 8, right: 24,
                            bottom: horizontal ? 8 : bottomMargin, left: 8 }}>
          {grid}
          {horizontal ? (
            <>
              <XAxis type="number" {...axisProps} tickFormatter={yFormat as never} />
              <YAxis type="category" dataKey={x} width={140} {...axisProps} />
            </>
          ) : (
            <>
              <XAxis dataKey={x} {...axisProps}
                     tickFormatter={xFormat as never} label={xAxisLabel} />
              <YAxis {...axisProps} tickFormatter={yFormat as never} />
            </>
          )}
          {tooltip}
          {legend}
          {series.map((s, i) => (
            <Bar key={s.key} dataKey={s.key} name={s.label} fill={colors[i]}
                 radius={horizontal ? [0, 4, 4, 0] : [4, 4, 0, 0]}>
              {highlightKey && series.length === 1
                ? data.map((row, ri) => (
                    <Cell key={ri}
                          fill={String(row[x]) === highlightKey
                            ? colors[0] : 'var(--text-muted)'} />
                  ))
                : null}
            </Bar>
          ))}
        </BarChart>
      );
    }

    if (kind === 'area') {
      return (
        <AreaChart data={data} margin={{ top: 8, right: 24, bottom: bottomMargin, left: 8 }}>
          <defs>
            {series.map((s, i) => (
              <linearGradient key={s.key} id={`${titleId}-g${i}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={colors[i]} stopOpacity={0.35} />
                <stop offset="100%" stopColor={colors[i]} stopOpacity={0.03} />
              </linearGradient>
            ))}
          </defs>
          {grid}
          <XAxis dataKey={x} {...axisProps} tickFormatter={xFormat as never}
                 label={xAxisLabel} />
          <YAxis {...axisProps} tickFormatter={yFormat as never} />
          {tooltip}
          {legend}
          {series.map((s, i) => (
            <Area key={s.key} type="monotone" dataKey={s.key} name={s.label}
                  stroke={colors[i]} strokeWidth={2}
                  fill={`url(#${titleId}-g${i})`} />
          ))}
        </AreaChart>
      );
    }

    if (kind === 'scatter') {
      return (
        <ScatterChart margin={{ top: 8, right: 24, bottom: 8, left: 8 }}>
          {grid}
          <XAxis type="number" dataKey={x} {...axisProps} name={xLabel ?? x} />
          <YAxis type="number" dataKey={series[0].key} {...axisProps}
                 name={series[0].label} tickFormatter={yFormat as never} />
          {tooltip}
          {legend}
          {series.map((s, i) => (
            <Scatter key={s.key} data={data} name={s.label} fill={colors[i]}
                     // 2px surface ring keeps overlapping points separable
                     stroke="var(--surface-1)" strokeWidth={2} />
          ))}
        </ScatterChart>
      );
    }

    return (
      <LineChart data={data} margin={{ top: 8, right: 32, bottom: bottomMargin, left: 8 }}>
        {grid}
        <XAxis dataKey={x} {...axisProps} tickFormatter={xFormat as never}
               label={xAxisLabel} />
        <YAxis {...axisProps} tickFormatter={yFormat as never} />
        {tooltip}
        {legend}
        {series.map((s, i) => (
          <Line key={s.key} type="monotone" dataKey={s.key} name={s.label}
                stroke={colors[i]} strokeWidth={2} dot={false}
                activeDot={{ r: 4, strokeWidth: 2, stroke: 'var(--surface-1)' }}
                label={labelPoints ? undefined : undefined} />
        ))}
      </LineChart>
    );
  }

  return (
    <figure style={{ margin: 0 }}>
      <div style={{
        display: 'flex', alignItems: 'baseline', gap: 'var(--space-3)',
        marginBottom: 'var(--space-2)',
      }}>
        {yLabel && (
          <span style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}>
            {yLabel}
          </span>
        )}
        {tableView && (
          <button
            type="button"
            onClick={() => setShowTable((v) => !v)}
            aria-pressed={showTable}
            style={{
              marginLeft: 'auto', padding: '3px 9px',
              borderRadius: 'var(--radius-sm)', border: '1px solid var(--border)',
              background: showTable ? 'var(--surface-2)' : 'transparent',
              color: 'var(--text-secondary)', fontSize: 'var(--text-xs)',
              fontWeight: 600, cursor: 'pointer',
            }}
          >
            {showTable ? 'Chart' : 'Table'}
          </button>
        )}
      </div>

      {showTable ? (
        <div style={{ overflowX: 'auto', maxHeight: height + 40 }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 'var(--text-sm)' }}>
            <thead>
              <tr style={{ borderBottom: '2px solid var(--axis)' }}>
                <th style={cellHead}>{xLabel ?? x}</th>
                {series.map((s, i) => (
                  <th key={s.key} style={{ ...cellHead, textAlign: 'right' }}>
                    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                      <span aria-hidden style={{
                        width: 8, height: 8, borderRadius: 2, background: colors[i],
                      }} />
                      {s.label}
                    </span>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.map((row, ri) => (
                <tr key={ri} style={{ borderBottom: '1px solid var(--gridline)' }}>
                  <td style={cell}>{xFormat ? xFormat(row[x]) : String(row[x])}</td>
                  {series.map((s) => (
                    <td key={s.key} style={{ ...cell, textAlign: 'right' }}>
                      {typeof row[s.key] === 'number'
                        ? yFormat(row[s.key] as number)
                        : String(row[s.key] ?? '—')}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={height}>
          {body()}
        </ResponsiveContainer>
      )}

      {caption && (
        <figcaption style={{
          marginTop: 'var(--space-2)', color: 'var(--text-muted)',
          fontSize: 'var(--text-xs)', lineHeight: 1.45,
        }}>
          {caption}
        </figcaption>
      )}
    </figure>
  );
}

const cellHead: React.CSSProperties = {
  textAlign: 'left', padding: '7px 9px', color: 'var(--text-muted)',
  fontWeight: 600, fontSize: 'var(--text-xs)', textTransform: 'uppercase',
  letterSpacing: 0.3, whiteSpace: 'nowrap', position: 'sticky', top: 0,
  background: 'var(--surface-1)',
};

const cell: React.CSSProperties = {
  padding: '7px 9px', color: 'var(--text-primary)', whiteSpace: 'nowrap',
};
