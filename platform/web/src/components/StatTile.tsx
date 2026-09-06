/**
 * StatTile -- a single headline number.
 *
 * Per the dataviz form heuristic, one number is not a chart. Hero figures use
 * proportional figures; only aligned columns get tabular-nums. Delta direction
 * ships with an arrow glyph and a worded label, never colour alone.
 */

import type { ReactNode } from 'react';

export interface StatTileProps {
  label: string;
  value: ReactNode;
  unit?: string;
  /** Signed relative change, e.g. 0.126 renders as "+12.6%". */
  delta?: number | null;
  /** Set false when a rise is bad (error, latency, drawdown). */
  deltaGoodWhenUp?: boolean;
  deltaLabel?: string;
  hint?: string;
  /** Optional identity swatch, e.g. 'var(--series-1)'. */
  accent?: string;
  size?: 'md' | 'hero';
}

export function StatTile({
  label,
  value,
  unit,
  delta = null,
  deltaGoodWhenUp = true,
  deltaLabel,
  hint,
  accent,
  size = 'md',
}: StatTileProps) {
  const hasDelta = delta !== null && delta !== undefined && Number.isFinite(delta);
  const up = hasDelta && (delta as number) > 0;
  const flat = hasDelta && (delta as number) === 0;
  const good = up === deltaGoodWhenUp;

  const deltaColor = flat
    ? 'var(--text-muted)'
    : good
      ? 'var(--delta-up)'
      : 'var(--status-critical)';
  const arrow = flat ? '→' : up ? '↑' : '↓';
  const worded = flat ? 'no change' : up ? 'increase' : 'decrease';

  return (
    <div
      style={{
        background: 'var(--surface-1)',
        border: '1px solid var(--border)',
        borderRadius: 'var(--radius-lg)',
        padding: size === 'hero' ? 'var(--space-5)' : 'var(--space-4)',
        boxShadow: 'var(--shadow)',
        display: 'flex',
        flexDirection: 'column',
        gap: 'var(--space-1)',
        minWidth: 0,
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
        {accent && (
          <span
            aria-hidden
            style={{
              width: 8,
              height: 8,
              borderRadius: 2,
              background: accent,
              flexShrink: 0,
            }}
          />
        )}
        <span
          style={{
            color: 'var(--text-secondary)',
            fontSize: 'var(--text-xs)',
            fontWeight: 600,
            letterSpacing: 0.3,
            textTransform: 'uppercase',
          }}
        >
          {label}
        </span>
      </div>

      <div style={{ display: 'flex', alignItems: 'baseline', gap: 6, minWidth: 0 }}>
        <span
          style={{
            fontSize: size === 'hero' ? 'var(--text-hero)' : 'var(--text-2xl)',
            fontWeight: 600,
            lineHeight: 1.1,
            color: 'var(--text-primary)',
            overflowWrap: 'anywhere',
          }}
        >
          {value}
        </span>
        {unit && (
          <span style={{ color: 'var(--text-muted)', fontSize: 'var(--text-sm)' }}>
            {unit}
          </span>
        )}
      </div>

      {hasDelta && (
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 4,
            color: deltaColor,
            fontSize: 'var(--text-sm)',
            fontWeight: 600,
          }}
        >
          <span aria-hidden>{arrow}</span>
          <span>
            {(delta as number) > 0 ? '+' : ''}
            {((delta as number) * 100).toFixed(1)}%
          </span>
          {/* Worded direction so meaning never rests on colour alone. */}
          <span className="sr-only" style={srOnly}>{worded}</span>
          {deltaLabel && (
            <span style={{ color: 'var(--text-muted)', fontWeight: 400 }}>
              {deltaLabel}
            </span>
          )}
        </div>
      )}

      {hint && (
        <p
          style={{
            margin: 0,
            color: 'var(--text-muted)',
            fontSize: 'var(--text-xs)',
            lineHeight: 1.4,
          }}
        >
          {hint}
        </p>
      )}
    </div>
  );
}

export function StatRow({ children }: { children: ReactNode }) {
  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
        gap: 'var(--space-3)',
      }}
    >
      {children}
    </div>
  );
}

const srOnly: React.CSSProperties = {
  position: 'absolute',
  width: 1,
  height: 1,
  padding: 0,
  margin: -1,
  overflow: 'hidden',
  clip: 'rect(0,0,0,0)',
  whiteSpace: 'nowrap',
  border: 0,
};
