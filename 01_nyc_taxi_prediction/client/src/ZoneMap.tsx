/**
 * ZoneMap -- project-01-specific.
 *
 * Renders the 263 real TLC taxi zones as inline SVG. Geometry comes from the
 * official TLC shapefile (EPSG:2263), simplified at build time by
 * `ml/build_zone_map.py` -- so this is a 100 KB asset and no mapping library.
 *
 * Selecting origin and destination here is the same action as choosing them in
 * the form; both write to the same state.
 */

import { useMemo, useState } from 'react';
import geo from './zones.geo.json';

interface Zone {
  id: number;
  zone: string;
  borough: string;
  path: string;
  cx: number;
  cy: number;
}

const ZONES = geo.zones as Zone[];
export const ZONE_BY_ID = new Map(ZONES.map((z) => [z.id, z]));

export interface ZoneMapProps {
  pickup?: number | null;
  dropoff?: number | null;
  onPick?: (id: number) => void;
  /** Which endpoint the next click sets. */
  mode?: 'pickup' | 'dropoff';
  height?: number;
  /** Optional magnitude per zone for a sequential fill, e.g. trip volume. */
  values?: Record<number, number>;
  valueLabel?: string;
}

export function ZoneMap({
  pickup = null,
  dropoff = null,
  onPick,
  mode = 'pickup',
  height = 420,
  values,
  valueLabel = 'value',
}: ZoneMapProps) {
  const [hover, setHover] = useState<Zone | null>(null);

  // Sequential ramp: one hue, light to dark. Bounds come from the supplied
  // values; nothing is computed beyond locating each value within them.
  const scale = useMemo(() => {
    if (!values) return null;
    const vs = Object.values(values).filter((v) => Number.isFinite(v));
    if (!vs.length) return null;
    return { min: Math.min(...vs), max: Math.max(...vs) };
  }, [values]);

  const SEQ = [
    'var(--seq-100)', 'var(--seq-200)', 'var(--seq-300)',
    'var(--seq-400)', 'var(--seq-500)', 'var(--seq-600)', 'var(--seq-700)',
  ];

  function fillFor(z: Zone): string {
    if (z.id === pickup) return 'var(--series-1)';
    if (z.id === dropoff) return 'var(--series-2)';
    if (scale && values?.[z.id] !== undefined) {
      const t = scale.max === scale.min
        ? 0.5
        : (values[z.id] - scale.min) / (scale.max - scale.min);
      return SEQ[Math.min(SEQ.length - 1, Math.floor(t * SEQ.length))];
    }
    // Land needs to read as land against the card surface; --surface-2 is too
    // close to --surface-1 to show the coastline at map scale.
    return 'var(--gridline)';
  }

  const route =
    pickup && dropoff && ZONE_BY_ID.has(pickup) && ZONE_BY_ID.has(dropoff)
      ? { a: ZONE_BY_ID.get(pickup)!, b: ZONE_BY_ID.get(dropoff)! }
      : null;

  return (
    <figure style={{ margin: 0, position: 'relative' }}>
      <div style={{
        display: 'flex', alignItems: 'center', gap: 'var(--space-3)',
        marginBottom: 'var(--space-2)', flexWrap: 'wrap',
        fontSize: 'var(--text-xs)', color: 'var(--text-secondary)',
      }}>
        <LegendChip color="var(--series-1)" label="Pickup" />
        <LegendChip color="var(--series-2)" label="Dropoff" />
        {scale && (
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
            <span>{valueLabel}</span>
            {SEQ.map((c) => (
              <span key={c} aria-hidden style={{
                width: 14, height: 9, background: c,
                border: '1px solid var(--border)',
              }} />
            ))}
          </span>
        )}
        {onPick && (
          <span style={{ marginLeft: 'auto', color: 'var(--text-muted)' }}>
            Click a zone to set the <strong>{mode}</strong>
          </span>
        )}
      </div>

      <svg
        viewBox={geo.viewBox}
        role="img"
        aria-label="Map of the 263 New York City taxi zones"
        style={{
          width: '100%', height, display: 'block',
          background: 'var(--surface-1)',
          border: '1px solid var(--border)',
          borderRadius: 'var(--radius)',
        }}
      >
        {ZONES.map((z) => {
          const active = z.id === pickup || z.id === dropoff;
          return (
            <path
              key={z.id}
              d={z.path}
              fill={fillFor(z)}
              stroke={active ? 'var(--surface-1)' : 'var(--surface-1)'}
              strokeWidth={active ? 2 : 0.6}
              onMouseEnter={() => setHover(z)}
              onMouseLeave={() => setHover(null)}
              onClick={onPick ? () => onPick(z.id) : undefined}
              style={{ cursor: onPick ? 'pointer' : 'default' }}
            >
              <title>
                {z.zone} ({z.borough})
                {values?.[z.id] !== undefined ? ` — ${valueLabel}: ${values[z.id]}` : ''}
              </title>
            </path>
          );
        })}

        {route && (
          <g pointerEvents="none">
            <line
              x1={route.a.cx} y1={route.a.cy} x2={route.b.cx} y2={route.b.cy}
              stroke="var(--text-primary)" strokeWidth={2}
              strokeDasharray="6 5" opacity={0.65}
            />
            {[['a', 'var(--series-1)'], ['b', 'var(--series-2)']].map(([k, c]) => {
              const p = route[k as 'a' | 'b'];
              return (
                <circle key={k} cx={p.cx} cy={p.cy} r={7} fill={c}
                        stroke="var(--surface-1)" strokeWidth={2} />
              );
            })}
          </g>
        )}
      </svg>

      {hover && (
        <div style={{
          position: 'absolute', left: 8, bottom: 8, pointerEvents: 'none',
          background: 'var(--surface-1)', border: '1px solid var(--border)',
          borderRadius: 'var(--radius)', padding: '6px 10px',
          boxShadow: 'var(--shadow)', fontSize: 'var(--text-xs)',
        }}>
          <strong>{hover.zone}</strong>
          <span style={{ color: 'var(--text-muted)' }}> · {hover.borough} · #{hover.id}</span>
        </div>
      )}

      <figcaption style={{
        marginTop: 'var(--space-2)', color: 'var(--text-muted)',
        fontSize: 'var(--text-xs)',
      }}>
        {ZONES.length} taxi zones from the official TLC shapefile ({geo.projection}).
        Simplified with {geo.simplification}.
      </figcaption>
    </figure>
  );
}

function LegendChip({ color, label }: { color: string; label: string }) {
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5 }}>
      <span aria-hidden style={{
        width: 9, height: 9, borderRadius: 2, background: color,
      }} />
      {label}
    </span>
  );
}
