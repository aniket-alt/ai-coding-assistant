/**
 * TripEstimator -- project-01-specific.
 *
 * Composes the shared PredictionForm with this project's zone map, and renders
 * the API's response. Every figure shown is a field of that response; nothing
 * is recomputed here.
 */

import { useState } from 'react';
import { PredictionForm, StatTile, StatRow } from '@dsfs/ui';
import type { FieldSpec } from '@dsfs/ui';
import { ZONE_BY_ID, ZoneMap } from './ZoneMap';

interface TripPrediction {
  duration_minutes: number;
  lower_minutes: number;
  upper_minutes: number;
  rmse_minutes: number;
  route_median_distance_miles: number;
  route_trip_count: number;
  is_rare_route: boolean;
  requested_at: string;
}

/** Friday evening rush -- a default that shows the model reacting to time. */
const DEFAULT_WHEN = '2024-02-02T17:00';

export function TripEstimator() {
  // The form is controlled from here so the map and the selects stay in sync:
  // clicking a zone and choosing it from the dropdown are the same action.
  // 132 = JFK Airport, 161 = Midtown Center -- a recognisable, genuinely long
  // route, so the first render is not a trivial two-minute hop.
  const [values, setValues] = useState<Record<string, unknown>>({
    pu_location: 132,
    do_location: 161,
    when: DEFAULT_WHEN,
    passenger_count: 1,
  });
  const [mode, setMode] = useState<'pickup' | 'dropoff'>('pickup');

  const pickup = Number(values.pu_location);
  const dropoff = Number(values.do_location);
  const pick = (id: number) =>
    setValues((v) => ({ ...v, [mode === 'pickup' ? 'pu_location' : 'do_location']: id }));

  const fields: FieldSpec[] = [
    { kind: 'select', name: 'pu_location', label: 'Pickup zone', options: '/api/zones' },
    { kind: 'select', name: 'do_location', label: 'Dropoff zone', options: '/api/zones' },
    { kind: 'datetime', name: 'when', label: 'Pickup time' },
    { kind: 'slider', name: 'passenger_count', label: 'Passengers', min: 1, max: 6 },
  ];

  const puName = ZONE_BY_ID.get(pickup)?.zone ?? '—';
  const doName = ZONE_BY_ID.get(dropoff)?.zone ?? '—';

  return (
    <div style={{
      display: 'grid', gap: 'var(--space-5)',
      gridTemplateColumns: 'minmax(320px, 1fr) minmax(340px, 1.1fr)',
      alignItems: 'start',
    }}>
      <div>
        <PredictionForm
          fields={fields}
          autoSubmit
          values={values}
          onValuesChange={setValues}
          renderResult={(r, { loading }) => {
            const p = r as TripPrediction | null;
            if (!p) {
              return (
                <p style={{ color: 'var(--text-muted)', fontSize: 'var(--text-sm)' }}>
                  {loading ? 'Estimating…' : 'Choose a route to see an estimate.'}
                </p>
              );
            }
            return (
              <div style={{ opacity: loading ? 0.55 : 1, transition: 'opacity .15s' }}>
                <StatRow>
                  <StatTile
                    label="Estimated duration"
                    value={p.duration_minutes}
                    unit="min"
                    size="hero"
                    accent="var(--series-1)"
                    hint={`${puName} → ${doName}`}
                  />
                  <StatTile
                    label="80% interval"
                    value={`${p.lower_minutes}–${p.upper_minutes}`}
                    unit="min"
                    hint={`From the model's measured held-out RMSE of ${p.rmse_minutes} min.`}
                  />
                  <StatTile
                    label="Typical route distance"
                    value={p.route_median_distance_miles}
                    unit="mi"
                    hint="Training-window median for this zone pair."
                  />
                  <StatTile
                    label="Historical trips"
                    value={p.route_trip_count.toLocaleString()}
                    hint={p.is_rare_route
                      ? 'Rare route — the estimate leans on the origin zone.'
                      : 'Well-represented route.'}
                  />
                </StatRow>
                {p.is_rare_route && (
                  <p style={{
                    marginTop: 'var(--space-3)', padding: 'var(--space-3)',
                    borderRadius: 'var(--radius)',
                    borderLeft: '3px solid var(--status-warning)',
                    background: 'var(--surface-2)', fontSize: 'var(--text-sm)',
                    color: 'var(--text-secondary)',
                  }}>
                    <strong>⚠ Sparse route.</strong> Fewer than ten training trips ran
                    between these zones, so the encoding is shrunk toward the global
                    mean and this estimate is less certain than the interval suggests.
                  </p>
                )}
              </div>
            );
          }}
        />
      </div>

      <div>
        <div style={{
          display: 'flex', gap: 'var(--space-1)', marginBottom: 'var(--space-2)',
        }}>
          {(['pickup', 'dropoff'] as const).map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => setMode(m)}
              aria-pressed={mode === m}
              style={{
                padding: '5px 12px', borderRadius: 'var(--radius-sm)',
                border: '1px solid var(--border)',
                background: mode === m ? 'var(--surface-2)' : 'transparent',
                color: mode === m ? 'var(--text-primary)' : 'var(--text-muted)',
                fontSize: 'var(--text-xs)', fontWeight: 600, cursor: 'pointer',
                textTransform: 'capitalize',
              }}
            >
              Set {m}
            </button>
          ))}
        </div>
        <ZoneMap
          pickup={pickup}
          dropoff={dropoff}
          mode={mode}
          onPick={pick}
        />
      </div>
    </div>
  );
}
