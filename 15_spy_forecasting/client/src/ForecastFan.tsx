/**
 * ForecastFan -- project-15-specific.
 *
 * Shows recent SPY closes and the next-period quantile band, in price terms.
 *
 * The band is deliberately drawn to scale against the price history rather
 * than zoomed in. At a one-day horizon the honest 80% interval is well under
 * one percent wide, and rescaling the axis to make it look dramatic is how
 * forecast charts usually mislead.
 */

import { useMemo, useState } from 'react';
import { StatRow, StatTile, useApi } from '@dsfs/ui';

interface PricePoint { date: string; close: number; vix: number }

interface ModelForecast {
  model: string;
  q10_pct: number; q50_pct: number; q90_pct: number;
  price_low: number; price_mid: number; price_high: number;
}

interface ForecastResponse {
  last_close: number;
  last_date: string;
  horizon: number;
  forecasts: ModelForecast[];
}

export function ForecastFan() {
  const prices = useApi<PricePoint[]>('/api/prices');
  const fc = useApi<ForecastResponse>('/api/forecast');
  const [modelIdx, setModelIdx] = useState(0);
  const [window, setWindow] = useState(120);

  const chosen = fc.data?.forecasts[modelIdx];

  const view = useMemo(() => {
    if (!prices.data?.length || !fc.data) return null;
    const tail = prices.data.slice(-window);
    const closes = tail.map((p) => p.close);
    const lo = Math.min(...closes, chosen?.price_low ?? Infinity);
    const hi = Math.max(...closes, chosen?.price_high ?? -Infinity);
    const pad = (hi - lo) * 0.06 || 1;
    return { tail, min: lo - pad, max: hi + pad };
  }, [prices.data, fc.data, window, chosen]);

  if (!view || !fc.data || !chosen) {
    return (
      <p style={{ color: 'var(--text-muted)', fontSize: 'var(--text-sm)' }}>
        {prices.error ?? fc.error ?? 'Loading forecast…'}
      </p>
    );
  }

  const W = 900, H = 320, padL = 56, padR = 120, padY = 18;
  const { tail, min, max } = view;
  const xs = (i: number) => padL + (i / (tail.length - 1)) * (W - padL - padR);
  const ys = (v: number) => padY + (1 - (v - min) / (max - min)) * (H - 2 * padY);

  const line = tail.map((p, i) => `${i === 0 ? 'M' : 'L'}${xs(i).toFixed(1)},${ys(p.close).toFixed(1)}`).join(' ');
  const lastX = xs(tail.length - 1);
  const bandX = lastX + 26;

  const ticks = 5;
  const gridVals = Array.from({ length: ticks }, (_, i) => min + (i / (ticks - 1)) * (max - min));

  return (
    <div>
      <div style={{
        display: 'flex', gap: 'var(--space-3)', alignItems: 'center',
        flexWrap: 'wrap', marginBottom: 'var(--space-3)',
      }}>
        <label style={{ fontSize: 'var(--text-xs)', color: 'var(--text-secondary)' }}>
          Model{' '}
          <select
            value={modelIdx}
            onChange={(e) => setModelIdx(Number(e.target.value))}
            style={{
              padding: '5px 8px', borderRadius: 'var(--radius-sm)',
              border: '1px solid var(--border)', background: 'var(--surface-1)',
              color: 'var(--text-primary)', font: 'inherit',
            }}
          >
            {fc.data.forecasts.map((f, i) => (
              <option key={f.model} value={i}>{f.model}</option>
            ))}
          </select>
        </label>
        <label style={{ fontSize: 'var(--text-xs)', color: 'var(--text-secondary)' }}>
          History{' '}
          <select
            value={window}
            onChange={(e) => setWindow(Number(e.target.value))}
            style={{
              padding: '5px 8px', borderRadius: 'var(--radius-sm)',
              border: '1px solid var(--border)', background: 'var(--surface-1)',
              color: 'var(--text-primary)', font: 'inherit',
            }}
          >
            {[60, 120, 250, 400].map((n) => (
              <option key={n} value={n}>{n} days</option>
            ))}
          </select>
        </label>
        <span style={{ marginLeft: 'auto', fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}>
          Last close ${fc.data.last_close.toFixed(2)} on {fc.data.last_date}
        </span>
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} role="img"
           aria-label={`SPY closing prices with an 80% forecast band from ${chosen.model}`}
           style={{
             width: '100%', background: 'var(--surface-1)',
             border: '1px solid var(--border)', borderRadius: 'var(--radius)',
           }}>
        {gridVals.map((v) => (
          <g key={v}>
            <line x1={padL} x2={W - padR} y1={ys(v)} y2={ys(v)}
                  stroke="var(--gridline)" strokeWidth={1} />
            <text x={padL - 8} y={ys(v) + 4} textAnchor="end"
                  fill="var(--text-muted)" fontSize={11}>
              ${v.toFixed(0)}
            </text>
          </g>
        ))}

        <path d={line} fill="none" stroke="var(--series-1)" strokeWidth={2} />

        {/* 80% band for the next period */}
        <line x1={lastX} y1={ys(fc.data.last_close)} x2={bandX}
              y2={ys(chosen.price_mid)}
              stroke="var(--text-muted)" strokeWidth={1} strokeDasharray="3 3" />
        <rect x={bandX - 7} y={ys(chosen.price_high)} width={14}
              height={Math.max(2, ys(chosen.price_low) - ys(chosen.price_high))}
              fill="var(--series-2)" opacity={0.28} rx={3} />
        <line x1={bandX - 11} x2={bandX + 11} y1={ys(chosen.price_mid)}
              y2={ys(chosen.price_mid)} stroke="var(--series-2)" strokeWidth={2} />

        <text x={bandX + 16} y={ys(chosen.price_high) + 4}
              fill="var(--text-secondary)" fontSize={11}>
          ${chosen.price_high.toFixed(2)}
        </text>
        <text x={bandX + 16} y={ys(chosen.price_mid) + 4}
              fill="var(--text-primary)" fontSize={11} fontWeight={600}>
          ${chosen.price_mid.toFixed(2)}
        </text>
        <text x={bandX + 16} y={ys(chosen.price_low) + 4}
              fill="var(--text-secondary)" fontSize={11}>
          ${chosen.price_low.toFixed(2)}
        </text>
        <text x={bandX + 16} y={ys(chosen.price_low) + 20}
              fill="var(--text-muted)" fontSize={10}>
          80% band
        </text>
      </svg>

      <p style={{
        margin: 'var(--space-2) 0 var(--space-4)', color: 'var(--text-muted)',
        fontSize: 'var(--text-xs)', lineHeight: 1.45,
      }}>
        Drawn to the same scale as the price history. At a one-day horizon the
        honest 80% interval is under one percent wide — rescaling the axis to
        make that band look dramatic is how forecast charts usually mislead.
      </p>

      <StatRow>
        <StatTile label="Median forecast" value={`${chosen.q50_pct >= 0 ? '+' : ''}${chosen.q50_pct.toFixed(3)}`}
                  unit="%" size="hero" accent="var(--series-1)"
                  hint={`Next ${fc.data.horizon}-day log return.`} />
        <StatTile label="80% band" value={`${chosen.q10_pct.toFixed(2)} … ${chosen.q90_pct >= 0 ? '+' : ''}${chosen.q90_pct.toFixed(2)}`}
                  unit="%" hint="Where 8 of 10 next-day outcomes should land." />
        <StatTile label="Price band" value={`$${chosen.price_low.toFixed(0)}–$${chosen.price_high.toFixed(0)}`}
                  hint="Asymmetric, because returns are logarithmic." />
        <StatTile label="Band width" value={(chosen.q90_pct - chosen.q10_pct).toFixed(2)}
                  unit="%" hint="The useful output. The direction mostly is not." />
      </StatRow>
    </div>
  );
}
