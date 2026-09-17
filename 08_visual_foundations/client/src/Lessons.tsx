/**
 * Project-08-specific lesson views. Each calls one API endpoint and draws what
 * it returns; the probabilities, confusion counts, loss curves and gradients
 * are all computed on the server by the checked implementations.
 */

import { useEffect, useState } from 'react';
import { apiGet, apiPost, Chart, StatRow, StatTile } from '@dsfs/ui';

const label: React.CSSProperties = {
  display: 'block', fontSize: 'var(--text-xs)', color: 'var(--text-muted)', fontWeight: 600,
  textTransform: 'uppercase', letterSpacing: 0.3, marginBottom: 4,
};
const input: React.CSSProperties = {
  width: '100%', boxSizing: 'border-box', padding: '10px 12px', font: 'inherit', color: 'var(--text-primary)',
  background: 'var(--surface-1)', border: '1px solid var(--border)', borderRadius: 'var(--radius)',
};

function useDebounced<T>(value: T, ms: number): T {
  const [v, setV] = useState(value);
  useEffect(() => { const t = setTimeout(() => setV(value), ms); return () => clearTimeout(t); }, [value, ms]);
  return v;
}

interface Explanation {
  prior_log_odds: number; prior_spam: number; log_odds: number; spam_probability: number;
  words: { token: string; count: number; log_ratio: number; unseen?: boolean }[];
}

export function BayesLesson({ examples }: { examples: string[] }) {
  const [text, setText] = useState(examples[0]);
  const debounced = useDebounced(text, 250);
  const [r, setR] = useState<Explanation | null>(null);
  useEffect(() => { if (debounced.trim()) apiPost<Explanation>('/api/classify', { text: debounced }).then(setR).catch(() => setR(null)); }, [debounced]);

  return (
    <div style={{ display: 'grid', gap: 'var(--space-4)' }}>
      <div style={{ display: 'flex', gap: 'var(--space-2)', flexWrap: 'wrap' }}>
        {examples.map((e) => (
          <button key={e} type="button" onClick={() => setText(e)} style={{ ...input, width: 'auto', cursor: 'pointer', fontSize: 'var(--text-xs)',
            borderColor: e === text ? 'var(--series-1)' : 'var(--border)' }}>{e.slice(0, 40)}…</button>
        ))}
      </div>
      <label><span style={label}>Message</span>
        <textarea rows={2} value={text} onChange={(e) => setText(e.target.value)} style={input} />
      </label>
      {r && (
        <>
          <StatRow>
            <StatTile label="Chance it is spam" size="hero" accent="var(--series-1)" unit="%"
                      value={r.spam_probability > 0.999 ? '> 99.9' : r.spam_probability < 0.001 ? '< 0.1' : (r.spam_probability * 100).toFixed(1)}
                      hint={r.spam_probability > 0.999 || r.spam_probability < 0.001 ? 'shown as a bound, because the exact figure would round to certainty' : undefined} />
            <StatTile label="Prior log-odds" value={r.prior_log_odds.toFixed(2)} hint={`${(r.prior_spam * 100).toFixed(1)}% of training messages are spam`} />
            <StatTile label="Total log-odds" value={r.log_odds.toFixed(2)} hint="prior plus every word's evidence" />
          </StatRow>
          <Chart kind="bar" horizontal height={Math.max(160, r.words.length * 26 + 40)} data={r.words} x="token"
                 series={[{ key: 'log_ratio', label: 'Evidence: log P(word | spam) − log P(word | ham)' }]}
                 yFormat={(v) => `${v >= 0 ? '+' : ''}${v.toFixed(1)}`}
                 caption="Positive bars push towards spam, negative towards ham. Words never seen in training contribute nothing." />
        </>
      )}
    </div>
  );
}

interface Confusion { threshold: number; tp: number; fp: number; fn: number; tn: number; precision: number | null; recall: number | null; messages: number }

export function ThresholdLesson() {
  const [t, setT] = useState(0.5);
  const debounced = useDebounced(t, 120);
  const [c, setC] = useState<Confusion | null>(null);
  useEffect(() => { apiGet<Confusion>(`/api/threshold?t=${debounced}`).then(setC).catch(() => setC(null)); }, [debounced]);
  const cell = (n: number, name: string, good: boolean) => (
    <div style={{ padding: 'var(--space-3)', borderRadius: 'var(--radius)', textAlign: 'center',
      background: `color-mix(in srgb, ${good ? 'var(--status-good)' : 'var(--status-critical)'} 14%, var(--surface-1))`,
      border: '1px solid var(--border)' }}>
      <div style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}>{name}</div>
      <div style={{ fontSize: 28, fontWeight: 700 }}>{n}</div>
    </div>
  );
  return (
    <div style={{ display: 'grid', gap: 'var(--space-4)' }}>
      <label><span style={label}>Flag as spam when the chance is at least {t.toFixed(3)}</span>
        <input type="range" min={0} max={1} step={0.001} value={t} onChange={(e) => setT(Number(e.target.value))} style={{ width: '100%' }} />
      </label>
      {c && (
        <div style={{ display: 'grid', gap: 'var(--space-4)', gridTemplateColumns: 'minmax(260px, 1fr) minmax(260px, 1fr)' }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-2)' }}>
            {cell(c.tp, 'Spam caught', true)}{cell(c.fn, 'Spam missed', false)}
            {cell(c.fp, 'Ham wrongly flagged', false)}{cell(c.tn, 'Ham let through', true)}
          </div>
          <StatRow>
            <StatTile label="Precision" value={c.precision === null ? '—' : `${(c.precision * 100).toFixed(1)}%`} hint="of flagged messages, how many are spam" />
            <StatTile label="Recall" value={c.recall === null ? '—' : `${(c.recall * 100).toFixed(1)}%`} hint="of spam, how much is caught" />
          </StatRow>
        </div>
      )}
      {c && <p style={{ margin: 0, fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}>{c.messages.toLocaleString()} held-out messages, scored by the written-out Naive Bayes.</p>}
    </div>
  );
}

interface Descent {
  trajectory: { step: number; mse: number | null }[]; diverged: boolean; final_mse: number | null;
  exact_mse: number; stable_rate: number; lr: number; steps: number; scaled: boolean;
}

export function DescentLesson() {
  const [lr, setLr] = useState('0.1');
  const [steps, setSteps] = useState(300);
  const [scaled, setScaled] = useState(true);
  const [r, setR] = useState<Descent | null>(null);
  const [error, setError] = useState<string | null>(null);
  const rate = Number(lr);
  const debounced = useDebounced(`${lr}|${steps}|${scaled}`, 300);
  useEffect(() => {
    if (!(rate > 0)) { setError('Enter a positive learning rate.'); return; }
    setError(null);
    apiPost<Descent>('/api/gradient-descent', { lr: rate, steps, scaled }).then(setR).catch((e) => setError(String(e)));
  }, [debounced]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div style={{ display: 'grid', gap: 'var(--space-4)' }}>
      <div style={{ display: 'grid', gap: 'var(--space-4)', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', alignItems: 'end' }}>
        <label><span style={label}>Learning rate</span><input value={lr} onChange={(e) => setLr(e.target.value)} style={input} /></label>
        <label><span style={label}>Steps: {steps}</span>
          <input type="range" min={10} max={2000} step={10} value={steps} onChange={(e) => setSteps(Number(e.target.value))} style={{ width: '100%' }} />
        </label>
        <label style={{ fontSize: 'var(--text-sm)' }}>
          <input type="checkbox" checked={scaled} onChange={(e) => setScaled(e.target.checked)} /> Standardise the features first
        </label>
      </div>
      {error && <p style={{ color: 'var(--status-critical)', margin: 0 }}>{error}</p>}
      {r && (
        <>
          <StatRow>
            <StatTile label="Largest stable learning rate" value={r.stable_rate < 0.001 ? r.stable_rate.toExponential(2) : r.stable_rate.toFixed(3)}
                      hint={scaled ? 'scaled features' : 'raw features'} />
            <StatTile label={`Training MSE after ${r.steps} steps`} value={r.diverged ? 'diverged' : (r.final_mse ?? 0).toFixed(2)}
                      accent={r.diverged ? 'var(--status-critical)' : undefined} />
            <StatTile label="Exact least-squares MSE" value={r.exact_mse.toFixed(2)} hint="the lowest any linear model can reach" />
          </StatRow>
          <Chart kind="line" height={300} data={r.trajectory.filter((p) => p.mse !== null)} x="step" xLabel="Step"
                 series={[{ key: 'mse', label: 'Training MSE' }]} yFormat={(v) => (v >= 1e4 ? v.toExponential(1) : v.toFixed(1))}
                 zeroBaseline={false} logScale
                 caption={r.diverged ? 'The learning rate is above the stable limit, so each step overshoots further and the loss grows until the run is stopped.'
                   : 'Mean squared error on the training readings as gradient descent runs from zero weights, on a logarithmic scale.'} />
        </>
      )}
    </div>
  );
}

interface Trace {
  row: number; features: string[]; inputs: number[]; raw_inputs: number[]; z1: number[]; h: number[];
  output: number; target: number; output_mw: number; target_mw: number; loss: number; d_out: number;
  d_h: number[]; d_z1: number[]; W2: number[]; grad_W2: number[]; grad_b2: number; grad_W1: number[][]; grad_b1: number[];
  training_rows: number;
}

export function BackpropLesson() {
  const [row, setRow] = useState(0);
  const debounced = useDebounced(row, 150);
  const [r, setR] = useState<Trace | null>(null);
  useEffect(() => { apiGet<Trace>(`/api/backprop?row=${debounced}`).then(setR).catch(() => setR(null)); }, [debounced]);
  if (!r) return <p style={{ color: 'var(--text-muted)' }}>Loading…</p>;

  const W = 760, H = 430, xs = [90, 380, 670];
  const y = (i: number, n: number) => 30 + (i + 0.5) * ((H - 60) / n);
  const colour = (v: number) => `color-mix(in srgb, ${v >= 0 ? 'var(--series-1)' : 'var(--series-2)'} ${Math.min(100, Math.round(Math.abs(v) * 60))}%, var(--surface-1))`;

  return (
    <div style={{ display: 'grid', gap: 'var(--space-4)' }}>
      <label><span style={label}>Training row {r.row.toLocaleString()} of {r.training_rows.toLocaleString()}</span>
        <input type="range" min={0} max={r.training_rows - 1} value={row} onChange={(e) => setRow(Number(e.target.value))} style={{ width: '100%' }} />
      </label>
      <StatRow>
        <StatTile label="Network output" value={r.output_mw.toFixed(1)} unit="MW" />
        <StatTile label="Actual output" value={r.target_mw.toFixed(1)} unit="MW" />
        <StatTile label="∂loss/∂output" value={r.d_out.toFixed(4)} hint="2 × (output − target), in standardised units" />
      </StatRow>
      <div style={{ overflowX: 'auto' }}>
        <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ maxWidth: W }} role="img" aria-label="Forward values and gradients through the network">
          {r.inputs.map((_, i) => r.h.map((__, j) => (
            <line key={`a${i}-${j}`} x1={xs[0]} y1={y(i, 4)} x2={xs[1]} y2={y(j, 8)} stroke="var(--gridline)" strokeWidth={1} />
          )))}
          {r.h.map((_, j) => (
            <line key={`b${j}`} x1={xs[1]} y1={y(j, 8)} x2={xs[2]} y2={y(0, 1)} stroke="var(--axis)"
                  strokeWidth={1 + Math.min(4, Math.abs(r.W2[j]) * 3)} opacity={0.6} />
          ))}
          {r.inputs.map((v, i) => (
            <g key={`in${i}`} transform={`translate(${xs[0]}, ${y(i, 4)})`}>
              <circle r={22} fill={colour(v)} stroke="var(--border)" />
              <text textAnchor="middle" y={4} fontSize={11} fill="var(--text-primary)">{v.toFixed(2)}</text>
              <text textAnchor="end" x={-30} y={4} fontSize={11} fill="var(--text-muted)">{r.features[i]}</text>
            </g>
          ))}
          {r.h.map((v, j) => (
            <g key={`h${j}`} transform={`translate(${xs[1]}, ${y(j, 8)})`}>
              <circle r={18} fill={colour(v)} stroke="var(--border)" />
              <text textAnchor="middle" y={4} fontSize={10} fill="var(--text-primary)">{v.toFixed(2)}</text>
              <text x={26} y={-4} fontSize={9} fill="var(--text-muted)">∂L/∂z {r.d_z1[j].toFixed(3)}</text>
              <text x={26} y={9} fontSize={9} fill="var(--text-muted)">∂L/∂w₂ {r.grad_W2[j].toFixed(3)}</text>
            </g>
          ))}
          <g transform={`translate(${xs[2]}, ${y(0, 1)})`}>
            <circle r={26} fill={colour(r.output)} stroke="var(--border)" />
            <text textAnchor="middle" y={4} fontSize={11} fill="var(--text-primary)">{r.output.toFixed(2)}</text>
            <text textAnchor="middle" y={44} fontSize={10} fill="var(--text-muted)">target {r.target.toFixed(2)}</text>
          </g>
          <text x={xs[0]} y={H - 6} textAnchor="middle" fontSize={11} fill="var(--text-secondary)">standardised inputs</text>
          <text x={xs[1]} y={H - 6} textAnchor="middle" fontSize={11} fill="var(--text-secondary)">hidden: h = tanh(W₁x + b₁)</text>
          <text x={xs[2]} y={H - 6} textAnchor="middle" fontSize={11} fill="var(--text-secondary)">output = W₂h + b₂</text>
        </svg>
      </div>
      <p style={{ margin: 0, fontSize: 'var(--text-sm)', color: 'var(--text-secondary)' }}>
        Backward, for hidden unit 1: ∂L/∂h = ∂L/∂output × W₂ = {r.d_out.toFixed(4)} × {r.W2[0].toFixed(4)} = {r.d_h[0].toFixed(4)};
        then ∂L/∂z = ∂L/∂h × (1 − h²) = {r.d_h[0].toFixed(4)} × (1 − {r.h[0].toFixed(4)}²) = {r.d_z1[0].toFixed(4)}.
        Every value here is returned by the server's backward pass.
      </p>
    </div>
  );
}
