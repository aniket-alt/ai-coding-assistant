/**
 * MetricsPanel -- the headline summary of a training run.
 *
 * Reads `/api/metrics` and renders it. It derives nothing: the lift over
 * baseline is `run.lift_over_baseline`, computed once at training time by
 * RunRecord, and the environment block is the environment captured when the
 * model was fitted -- never the serving process's own versions, which would
 * drift from whatever actually trained the model.
 *
 * Baselines themselves belong to ModelLeaderboard, where they appear as rows.
 * This panel reports the already-computed lift and nothing more, so the two
 * components cannot disagree about what the baseline was.
 */

import type { ModelScore, RunRecord } from '../types';
import { StatTile, StatRow } from './StatTile';

export interface MetricSpec {
  key: string;
  label: string;
  unit?: string;
  format?: (v: number) => string;
  goodWhenUp?: boolean;
  hint?: string;
}

export interface MetricsPanelProps {
  run: RunRecord;
  /** Hero tiles. Defaults to the primary metric plus the next three recorded. */
  highlight?: MetricSpec[];
  showSplit?: boolean;
  showBaselineLift?: boolean;
  showEnvironment?: boolean;
}

const defaultFormat = (v: number): string => {
  if (!Number.isFinite(v)) return '—';
  const a = Math.abs(v);
  if (a !== 0 && a < 0.001) return v.toExponential(2);
  if (a >= 1e4) return v.toLocaleString(undefined, { maximumFractionDigits: 0 });
  return v.toFixed(a < 1 ? 4 : 3).replace(/\.?0+$/, '');
};

export function MetricsPanel({
  run,
  highlight,
  showSplit = true,
  showBaselineLift = true,
  showEnvironment = true,
}: MetricsPanelProps) {
  const best: ModelScore | undefined = run.models.find((m) => m.name === run.best_model);

  const specs: MetricSpec[] = highlight ?? deriveHighlight(run, best);
  // `lift_over_baseline` arrives computed from the run record. Recomputing it
  // here would risk a figure that disagrees with the artifact.
  const lift = run.lift_over_baseline;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-4)' }}>
      <StatRow>
        {specs.map((s, i) => {
          const v = best?.metrics[s.key];
          const fmt = s.format ?? defaultFormat;
          return (
            <StatTile
              key={s.key}
              label={s.label}
              value={v === undefined || !Number.isFinite(v) ? '—' : fmt(v)}
              unit={s.unit}
              hint={s.hint}
              accent={i === 0 ? 'var(--series-1)' : undefined}
              size={i === 0 ? 'hero' : 'md'}
            />
          );
        })}
      </StatRow>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))',
          gap: 'var(--space-3)',
        }}
      >
        {showBaselineLift && lift !== null && lift !== undefined && (
          <Card title="Improvement over baseline">
            <p style={{ margin: 0, fontSize: 'var(--text-2xl)', fontWeight: 600 }}>
              {(lift * 100).toFixed(1)}%
            </p>
            <p style={{ margin: '2px 0 0', color: 'var(--text-secondary)', fontSize: 'var(--text-xs)' }}>
              {best?.name} against the naive baseline, on the held-out split.
              Baseline rows are listed in the leaderboard.
            </p>
          </Card>
        )}

        {showSplit && run.split && (
          <Card title="Evaluation split">
            <Row k="Strategy" v={run.split.strategy} />
            <Row k="Train rows" v={run.split.n_train?.toLocaleString()} />
            <Row k="Test rows" v={run.split.n_test?.toLocaleString()} />
            {run.split.embargo && <Row k="Embargo" v={run.split.embargo} />}
            {run.split.train_end && <Row k="Train ends" v={run.split.train_end} />}
            {run.split.test_start && <Row k="Test starts" v={run.split.test_start} />}
          </Card>
        )}

        {showEnvironment && run.env && (
          <Card title="Reproducibility">
            {/* Captured when the model was fitted, read back from metrics.json.
                Not the serving process -- those versions would drift. */}
            <Row k="Trained at" v={run.created_at} />
            <Row k="Duration" v={`${run.duration_seconds}s`} />
            {Object.entries(run.env).map(([k, v]) => (
              <Row key={k} k={k} v={String(v)} />
            ))}
            <p style={{
              margin: 'var(--space-2) 0 0', color: 'var(--text-muted)',
              fontSize: 'var(--text-xs)', fontStyle: 'italic',
            }}>
              Recorded at training time, not by this server.
            </p>
          </Card>
        )}
      </div>
    </div>
  );
}

/** Primary metric first, then whatever else the run recorded. */
function deriveHighlight(run: RunRecord, best?: ModelScore): MetricSpec[] {
  const keys = best ? Object.keys(best.metrics) : [];
  const rest = keys.filter((k) => k !== run.primary_metric).slice(0, 3);
  return [run.primary_metric, ...rest].map((k) => ({
    key: k,
    label: k.toUpperCase().replace(/_/g, ' '),
    goodWhenUp: k === run.primary_metric ? run.higher_is_better : undefined,
  }));
}

function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section style={{
      background: 'var(--surface-1)', border: '1px solid var(--border)',
      borderRadius: 'var(--radius-lg)', padding: 'var(--space-4)',
      boxShadow: 'var(--shadow)',
    }}>
      <h4 style={{
        margin: '0 0 var(--space-2)', fontSize: 'var(--text-xs)',
        textTransform: 'uppercase', letterSpacing: 0.4,
        color: 'var(--text-muted)', fontWeight: 700,
      }}>
        {title}
      </h4>
      {children}
    </section>
  );
}

function Row({ k, v }: { k: string; v?: string }) {
  if (!v) return null;
  return (
    <div style={{
      display: 'flex', justifyContent: 'space-between', gap: 'var(--space-3)',
      padding: '3px 0', fontSize: 'var(--text-sm)',
    }}>
      <span style={{ color: 'var(--text-muted)' }}>{k}</span>
      <span className="tabular-nums" style={{
        color: 'var(--text-primary)', textAlign: 'right', overflowWrap: 'anywhere',
      }}>
        {v}
      </span>
    </div>
  );
}
