/**
 * Project-02-specific views: the generation studio, the tokenizer and the
 * attention map. Each renders exactly what one API call returned.
 */

import { useEffect, useState } from 'react';
import { apiPost, StatRow, StatTile } from '@dsfs/ui';

export interface Piece { id: number; text: string }
interface Generation {
  prompt: string; continuation: string; tokens: Piece[]; prompt_tokens: number; generated_tokens: number;
  stopped_at_end_of_story: boolean; ms_per_token: number; device: string; kv_cache: boolean;
}
interface Tokens { tokens: Piece[]; characters: number; count: number; vocab_size: number }
interface Attention { tokens: Piece[]; truncated_to: number; layers: number[][][][] }

const label: React.CSSProperties = {
  display: 'block', fontSize: 'var(--text-xs)', color: 'var(--text-muted)', fontWeight: 600,
  textTransform: 'uppercase', letterSpacing: 0.3, marginBottom: 4,
};
const input: React.CSSProperties = {
  width: '100%', boxSizing: 'border-box', padding: '10px 12px', font: 'inherit', color: 'var(--text-primary)',
  background: 'var(--surface-1)', border: '1px solid var(--border)', borderRadius: 'var(--radius)',
};
const button: React.CSSProperties = {
  padding: '9px 16px', font: 'inherit', fontWeight: 600, borderRadius: 'var(--radius)', cursor: 'pointer',
  border: '1px solid var(--series-1)', background: 'var(--series-1)', color: '#fff',
};

function Slider({ name, value, min, max, step, onChange }: {
  name: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void;
}) {
  return (
    <label style={{ display: 'block' }}>
      <span style={label}>{name}: {value}</span>
      <input type="range" min={min} max={max} step={step} value={value} style={{ width: '100%' }}
             onChange={(e) => onChange(Number(e.target.value))} />
    </label>
  );
}

export function TokenChips({ tokens }: { tokens: Piece[] }) {
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
      {tokens.map((t, i) => (
        <span key={i} title={`id ${t.id}`} style={{
          fontFamily: 'var(--font-mono, monospace)', fontSize: 'var(--text-sm)', padding: '2px 6px',
          borderRadius: 4, whiteSpace: 'pre',
          background: `color-mix(in srgb, var(--series-${(i % 2) + 1}) 16%, transparent)`,
          border: `1px solid color-mix(in srgb, var(--series-${(i % 2) + 1}) 45%, transparent)`,
        }}>{t.text.replace(/ /g, '·').replace(/\n/g, '↵')}</span>
      ))}
    </div>
  );
}

export function GenerationStudio({ prompts }: { prompts: string[] }) {
  const [prompt, setPrompt] = useState(prompts[0] ?? 'Once upon a time,');
  const [params, setParams] = useState({ max_new: 120, temperature: 0.8, top_p: 0.9, repetition_penalty: 1.1, seed: 0 });
  const [cache, setCache] = useState(true);
  const [result, setResult] = useState<Generation | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const run = async () => {
    setBusy(true); setError(null);
    try {
      setResult(await apiPost<Generation>('/api/generate', { prompt, ...params, use_cache: cache }));
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };
  useEffect(() => { void run(); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div style={{ display: 'grid', gap: 'var(--space-4)' }}>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 'var(--space-2)' }}>
        {prompts.map((p) => (
          <button key={p} type="button" onClick={() => setPrompt(p)} style={{
            ...input, width: 'auto', cursor: 'pointer', fontSize: 'var(--text-xs)',
            borderColor: p === prompt ? 'var(--series-1)' : 'var(--border)' }}>{p.slice(0, 42)}…</button>
        ))}
      </div>
      <label><span style={label}>Prompt</span>
        <textarea rows={3} value={prompt} onChange={(e) => setPrompt(e.target.value)} style={input} />
      </label>
      <div style={{ display: 'grid', gap: 'var(--space-4)', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))' }}>
        <Slider name="Temperature" value={params.temperature} min={0} max={1.5} step={0.05}
                onChange={(v) => setParams({ ...params, temperature: v })} />
        <Slider name="Top-p" value={params.top_p} min={0.1} max={1} step={0.05}
                onChange={(v) => setParams({ ...params, top_p: v })} />
        <Slider name="Repetition penalty" value={params.repetition_penalty} min={1} max={1.5} step={0.05}
                onChange={(v) => setParams({ ...params, repetition_penalty: v })} />
        <Slider name="New tokens" value={params.max_new} min={10} max={240} step={10}
                onChange={(v) => setParams({ ...params, max_new: v })} />
        <Slider name="Seed" value={params.seed} min={0} max={50} step={1}
                onChange={(v) => setParams({ ...params, seed: v })} />
      </div>
      <div style={{ display: 'flex', gap: 'var(--space-4)', alignItems: 'center' }}>
        <button type="button" onClick={() => void run()} disabled={busy} style={{ ...button, opacity: busy ? 0.6 : 1 }}>
          {busy ? 'Generating…' : 'Generate'}
        </button>
        <label style={{ fontSize: 'var(--text-sm)' }}>
          <input type="checkbox" checked={cache} onChange={(e) => setCache(e.target.checked)} /> Use the KV cache
        </label>
      </div>
      {error && <p style={{ color: 'var(--status-critical)' }}>{error}</p>}
      {result && (
        <div style={{ opacity: busy ? 0.55 : 1 }}>
          <StatRow>
            <StatTile label="Generated" value={result.generated_tokens} unit="tokens"
                      hint={result.stopped_at_end_of_story ? 'Stopped at the end of a story' : 'Reached the token limit'} />
            <StatTile label="Speed" value={result.ms_per_token.toFixed(1)} unit="ms / token"
                      hint={`${result.device}, KV cache ${result.kv_cache ? 'on' : 'off'}`} />
            <StatTile label="Prompt" value={result.prompt_tokens} unit="tokens" />
          </StatRow>
          <p style={{ fontSize: 'var(--text-lg, 1.1rem)', lineHeight: 1.7, whiteSpace: 'pre-wrap', marginTop: 'var(--space-4)' }}>
            <span style={{ color: 'var(--text-muted)' }}>{result.prompt}</span>{result.continuation}
          </p>
        </div>
      )}
    </div>
  );
}

export function TokenizerView({ initial }: { initial: string }) {
  const [text, setText] = useState(initial);
  const [result, setResult] = useState<Tokens | null>(null);
  useEffect(() => {
    const t = setTimeout(() => { apiPost<Tokens>('/api/tokenize', { text }).then(setResult).catch(() => setResult(null)); }, 250);
    return () => clearTimeout(t);
  }, [text]);
  return (
    <div style={{ display: 'grid', gap: 'var(--space-4)' }}>
      <label><span style={label}>Text</span>
        <textarea rows={3} value={text} onChange={(e) => setText(e.target.value)} style={input} />
      </label>
      {result && (
        <>
          <StatRow>
            <StatTile label="Tokens" value={result.count} />
            <StatTile label="Characters" value={result.characters} />
            <StatTile label="Vocabulary" value={result.vocab_size.toLocaleString()} unit="tokens" />
          </StatRow>
          <TokenChips tokens={result.tokens} />
          <p style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}>· marks a space and ↵ a line break. Hover a token for its id.</p>
        </>
      )}
    </div>
  );
}

export function AttentionView({ initial, heads, layers }: { initial: string; heads: number; layers: number }) {
  const [text, setText] = useState(initial);
  const [layer, setLayer] = useState(0);
  const [head, setHead] = useState(0);
  const [result, setResult] = useState<Attention | null>(null);
  useEffect(() => {
    const t = setTimeout(() => { apiPost<Attention>('/api/attention', { text }).then(setResult).catch(() => setResult(null)); }, 300);
    return () => clearTimeout(t);
  }, [text]);

  const grid = result?.layers[layer]?.[head];
  const n = result?.tokens.length ?? 0;
  const cell = n ? Math.max(10, Math.min(28, Math.floor(560 / n))) : 20;
  return (
    <div style={{ display: 'grid', gap: 'var(--space-4)' }}>
      <label><span style={label}>Prompt (first 48 tokens)</span>
        <input value={text} onChange={(e) => setText(e.target.value)} style={input} />
      </label>
      <div style={{ display: 'grid', gap: 'var(--space-4)', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))' }}>
        <Slider name="Layer" value={layer + 1} min={1} max={layers} step={1} onChange={(v) => setLayer(v - 1)} />
        <Slider name="Head" value={head + 1} min={1} max={heads} step={1} onChange={(v) => setHead(v - 1)} />
      </div>
      {result && grid && (
        <div style={{ overflowX: 'auto' }}>
          <table role="grid" aria-label={`Attention weights, layer ${layer + 1}, head ${head + 1}`}
                 style={{ borderCollapse: 'collapse', fontSize: 11 }}>
            <thead>
              <tr>
                <th />
                {result.tokens.map((t, j) => (
                  <th key={j} style={{ height: 70, verticalAlign: 'bottom', padding: 0 }}>
                    <div style={{ writingMode: 'vertical-rl', transform: 'rotate(180deg)', whiteSpace: 'pre',
                      color: 'var(--text-muted)', fontWeight: 400 }}>{t.text.replace(/\n/g, '↵')}</div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {grid.map((row, i) => (
                <tr key={i}>
                  <th style={{ textAlign: 'right', paddingRight: 6, whiteSpace: 'pre', color: 'var(--text-muted)', fontWeight: 400 }}>
                    {result.tokens[i].text.replace(/\n/g, '↵')}
                  </th>
                  {row.map((w, j) => (
                    <td key={j} title={`${result.tokens[i].text} → ${result.tokens[j].text}: ${w.toFixed(3)}`}
                        style={{ width: cell, height: cell, padding: 0,
                          background: j > i ? 'var(--surface-2)' : `color-mix(in srgb, var(--series-1) ${Math.round(w * 100)}%, var(--surface-1))` }} />
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          <p style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}>
            Each row is a token attending to earlier tokens; darker means more weight. Cells above the diagonal are the future, which the causal mask hides.
          </p>
        </div>
      )}
    </div>
  );
}
