/**
 * LoadFan -- project-12-specific.
 *
 * Recent actual load against the day-ahead band issued for it, followed by the
 * band for the day after the data ends. Showing the track record beside the
 * forecast lets a reader judge the band by how often it actually held.
 */

export interface RecentPoint {
  time: string; actual: number; q10: number; q50: number; q90: number; yesterday: number;
}
export interface NextPoint { time: string; q10: number; q50: number; q90: number }

export interface Coverage { hours: number; inside: number; pct: number | null }

interface Props {
  recent: RecentPoint[];
  nextDay: NextPoint[];
  days: number;
  /** Band coverage over the window shown, as computed by the API. */
  coverage?: Coverage;
}

export function LoadFan({ recent, nextDay, days, coverage }: Props) {
  const past = recent.slice(-24 * days);
  const all = [
    ...past.map((p) => ({ ...p, future: false })),
    ...nextDay.map((p) => ({ ...p, actual: NaN, yesterday: NaN, future: true })),
  ];
  const W = 1000, H = 330, L = 48, R = 16, T = 26, B = 34;
  const hi = Math.max(...all.map((p) => Math.max(p.q90, Number.isFinite(p.actual) ? p.actual : 0)));
  const x = (i: number) => L + (i / (all.length - 1)) * (W - L - R);
  const y = (v: number) => T + (1 - v / (hi * 1.05)) * (H - T - B);

  const band = (pts: typeof all, offset: number) =>
    pts.map((p, i) => `${i ? 'L' : 'M'}${x(i + offset)},${y(p.q90)}`).join(' ') + ' ' +
    [...pts].reverse().map((p, i) => `L${x(offset + pts.length - 1 - i)},${y(p.q10)}`).join(' ') + ' Z';

  const line = (vals: number[], offset = 0) =>
    vals.map((v, i) => `${i ? 'L' : 'M'}${x(i + offset)},${y(v)}`).join(' ');

  const split = past.length - 0.5;
  const ticks = [0, 1, 2, 3, 4].map((k) => (hi * 1.05 * k) / 4);
  const dayMarks = all.map((p, i) => ({ i, p })).filter(({ p }) => p.time.endsWith('00:00:00'));

  return (
    <figure style={{ margin: 0 }}>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" style={{ width: '100%', background: 'var(--surface-1)',
        border: '1px solid var(--border)', borderRadius: 'var(--radius)' }}
        aria-label={`Hourly load for the last ${days} days against the forecast band, then tomorrow's band`}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={L} x2={W - R} y1={y(t)} y2={y(t)} stroke="var(--gridline)" />
            <text x={L - 6} y={y(t) + 4} textAnchor="end" fontSize={11} fill="var(--text-muted)">{t.toFixed(1)}</text>
          </g>
        ))}
        {dayMarks.map(({ i, p }) => (
          <text key={p.time} x={x(i)} y={H - 12} textAnchor="middle" fontSize={10} fill="var(--text-muted)">
            {p.time.slice(5, 10)}
          </text>
        ))}
        <path d={band(all.slice(0, past.length), 0)} fill="var(--series-1)" opacity={0.16} />
        <path d={band(all.slice(past.length), past.length)} fill="var(--series-2)" opacity={0.25} />
        <line x1={x(split)} x2={x(split)} y1={T} y2={H - B} stroke="var(--axis)" strokeDasharray="4 4" />
        <path d={line(past.map((p) => p.q50))} fill="none" stroke="var(--series-1)" strokeWidth={1.5} opacity={0.8} />
        <path d={line(past.map((p) => p.actual))} fill="none" stroke="var(--text-primary)" strokeWidth={1.4} />
        <path d={line(nextDay.map((p) => p.q50), past.length)} fill="none" stroke="var(--series-2)" strokeWidth={2} />
        <text x={x(split) + 6} y={T + 12} fontSize={11} fill="var(--text-secondary)">tomorrow →</text>
        <text x={L - 6} y={12} textAnchor="end" fontSize={10} fill="var(--text-muted)">kW</text>
      </svg>
      <figcaption style={{ display: 'flex', gap: 'var(--space-4)', flexWrap: 'wrap', marginTop: 'var(--space-2)',
        fontSize: 'var(--text-xs)', color: 'var(--text-secondary)' }}>
        <Key color="var(--text-primary)" label="Actual load" />
        <Key color="var(--series-1)" label="Forecast median, issued a day ahead" />
        <Key color="var(--series-1)" label="80% band" swatch />
        <Key color="var(--series-2)" label="Tomorrow" swatch />
        {coverage && (
          <span style={{ marginLeft: 'auto', color: 'var(--text-muted)' }}>
            Actual load fell inside the band in {coverage.inside} of {coverage.hours} hours shown
            ({coverage.pct}%).
          </span>
        )}
      </figcaption>
    </figure>
  );
}

function Key({ color, label, swatch }: { color: string; label: string; swatch?: boolean }) {
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
      <span aria-hidden style={swatch
        ? { width: 14, height: 10, background: color, opacity: 0.3, borderRadius: 2 }
        : { width: 14, height: 2, background: color }} />
      {label}
    </span>
  );
}
