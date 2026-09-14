/**
 * TransactionScorer -- project-06-specific.
 *
 * Scores real held-out transactions. The true label travels with each sample,
 * so the view can show when the model gets one wrong -- which it does, for
 * about a third of the fraud. Hiding that would make the demo look better and
 * the evaluation worse.
 */

import { useEffect, useState } from 'react';
import { StatRow, StatTile, apiGet, apiPost } from '@dsfs/ui';

interface Sample {
  id: number;
  label: string;
  is_fraud: number;
  features: Record<string, number>;
}

interface ScoreResult {
  score: number;
  percentile: number;
  rank_from_top: number;
  reference_population: number;
  would_be_reviewed: boolean;
  review_budget: number;
}

export function TransactionScorer() {
  const [samples, setSamples] = useState<Sample[]>([]);
  const [selected, setSelected] = useState<Sample | null>(null);
  const [result, setResult] = useState<ScoreResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    apiGet<Sample[]>('/api/samples?n=12')
      .then((s) => {
        setSamples(s);
        if (s.length) setSelected(s[0]);
      })
      .catch((e: Error) => setError(e.message));
  }, []);

  useEffect(() => {
    if (!selected) return;
    setLoading(true);
    setError(null);
    apiPost<ScoreResult>('/api/score', { features: selected.features })
      .then(setResult)
      .catch((e: Error) => { setError(e.message); setResult(null); })
      .finally(() => setLoading(false));
  }, [selected]);

  const correct = result && selected
    ? result.would_be_reviewed === (selected.is_fraud === 1)
    : null;

  return (
    <div style={{
      display: 'grid', gap: 'var(--space-5)',
      gridTemplateColumns: 'minmax(260px, 0.8fr) minmax(340px, 1.2fr)',
      alignItems: 'start',
    }}>
      <div>
        <h3 style={{ margin: '0 0 var(--space-2)', fontSize: 'var(--text-sm)' }}>
          Held-out transactions
        </h3>
        <p style={{
          margin: '0 0 var(--space-3)', color: 'var(--text-muted)',
          fontSize: 'var(--text-xs)',
        }}>
          Real transactions from the test window, over-sampled for fraud so
          there is something to look at. The model never saw any of them.
        </p>
        <ul style={{ listStyle: 'none', margin: 0, padding: 0 }}>
          {samples.map((s) => {
            const on = s.id === selected?.id;
            return (
              <li key={s.id}>
                <button
                  type="button"
                  onClick={() => setSelected(s)}
                  style={{
                    width: '100%', textAlign: 'left', cursor: 'pointer',
                    padding: 'var(--space-2) var(--space-3)',
                    marginBottom: 4, font: 'inherit',
                    borderRadius: 'var(--radius-sm)',
                    border: '1px solid var(--border)',
                    borderLeft: `3px solid ${s.is_fraud
                      ? 'var(--status-critical)' : 'var(--gridline)'}`,
                    background: on ? 'var(--surface-2)' : 'transparent',
                    color: on ? 'var(--text-primary)' : 'var(--text-secondary)',
                    fontSize: 'var(--text-sm)',
                    fontWeight: on ? 600 : 400,
                  }}
                >
                  {s.label}
                </button>
              </li>
            );
          })}
        </ul>
      </div>

      <div>
        {error && (
          <div role="alert" style={{
            padding: 'var(--space-3)', borderRadius: 'var(--radius)',
            borderLeft: '3px solid var(--status-critical)',
            background: 'var(--surface-2)', fontSize: 'var(--text-sm)',
          }}>
            {error}
          </div>
        )}

        {result && selected && (
          <div style={{ opacity: loading ? 0.55 : 1, transition: 'opacity .15s' }}>
            <StatRow>
              <StatTile
                label="Suspicion score"
                value={result.score < 0.001
                  ? result.score.toExponential(1)
                  : result.score.toFixed(4)}
                size="hero"
                accent="var(--series-1)"
                hint="Model probability that this transaction is fraudulent."
              />
              <StatTile
                label="Rank"
                value={`#${result.rank_from_top.toLocaleString()}`}
                hint={`Out of ${result.reference_population.toLocaleString()} held-out transactions.`}
              />
              <StatTile
                label={`In today's ${result.review_budget} alerts?`}
                value={result.would_be_reviewed ? 'Yes' : 'No'}
                hint={result.would_be_reviewed
                  ? 'An analyst would see this one.'
                  : 'Ranked below the review budget.'}
              />
              <StatTile
                label="Ground truth"
                value={selected.is_fraud ? 'Fraud' : 'Legitimate'}
                hint="The real label, which the model never saw."
              />
            </StatRow>

            {correct !== null && (
              <div style={{
                marginTop: 'var(--space-3)', padding: 'var(--space-3)',
                borderRadius: 'var(--radius)',
                borderLeft: `3px solid ${correct
                  ? 'var(--status-good)' : 'var(--status-warning)'}`,
                background: 'var(--surface-2)', fontSize: 'var(--text-sm)',
                color: 'var(--text-secondary)',
              }}>
                <strong style={{
                  color: correct ? 'var(--delta-up)' : 'var(--status-warning)',
                }}>
                  {correct ? '✓ Correct' : '✗ Missed'}
                </strong>{' '}
                {selected.is_fraud === 1 && !result.would_be_reviewed && (
                  <>This fraud ranks #{result.rank_from_top.toLocaleString()}, below
                  the {result.review_budget}-alert budget, so no analyst would see
                  it. About a third of fraud in this window is missed this way —
                  the budget curve below shows what a larger team would recover.</>
                )}
                {selected.is_fraud === 1 && result.would_be_reviewed && (
                  <>Ranked inside the review budget, so this fraud gets caught.</>
                )}
                {selected.is_fraud === 0 && !result.would_be_reviewed && (
                  <>Correctly left alone — no analyst time spent.</>
                )}
                {selected.is_fraud === 0 && result.would_be_reviewed && (
                  <>A false positive: this legitimate transaction would consume
                  one of the {result.review_budget} review slots.</>
                )}
              </div>
            )}

            <details style={{ marginTop: 'var(--space-4)' }}>
              <summary style={{
                cursor: 'pointer', fontSize: 'var(--text-xs)',
                color: 'var(--text-muted)', fontWeight: 600,
                textTransform: 'uppercase', letterSpacing: 0.3,
              }}>
                Feature vector (V1–V28 are published PCA components)
              </summary>
              <div style={{
                marginTop: 'var(--space-2)',
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fill, minmax(110px, 1fr))',
                gap: '4px var(--space-3)', fontSize: 'var(--text-xs)',
              }}>
                {Object.entries(selected.features).map(([k, v]) => (
                  <div key={k} style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: 'var(--text-muted)' }}>{k}</span>
                    <span className="tabular-nums">{v.toFixed(3)}</span>
                  </div>
                ))}
              </div>
            </details>
          </div>
        )}
      </div>
    </div>
  );
}
