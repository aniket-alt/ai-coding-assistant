/**
 * LeakBenchmark -- project-11-specific.
 *
 * Every planted leak and clean control, with what each detector concluded and
 * how much the leak inflated the reported score. Selecting a row shows the
 * variant's source, so a reader can check exactly what was planted.
 */

import { useEffect, useState } from 'react';
import { Chart, apiGet } from '@dsfs/ui';

export interface Variant {
  id: string; title: string; kind: 'leak' | 'control'; category: string; explanation: string;
  average_precision: number; inflation: number;
  static_detected: boolean; static_rules: string[];
  runtime_detected: boolean; runtime_check: string | null;
}

interface Benchmark { clean_average_precision: number; variants: Variant[] }

export function LeakBenchmark({ data }: { data: Benchmark }) {
  const [selected, setSelected] = useState<string>('noisy_label_proxy');
  const [source, setSource] = useState<string>('');

  useEffect(() => {
    apiGet<{ source: string }>(`/api/corpus/${selected}`)
      .then((r) => setSource(r.source))
      .catch((e: Error) => setSource(`# ${e.message}`));
  }, [selected]);

  const leaks = data.variants.filter((v) => v.kind === 'leak');
  const byDamage = [...leaks].sort((a, b) => b.inflation - a.inflation);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-5)' }}>
      <Chart
        kind="bar" horizontal height={360}
        data={byDamage.map((v) => ({
          leak: v.title,
          inflation: v.inflation,
        }))}
        x="leak"
        series={[{ key: 'inflation', label: 'Change in average precision caused by the leak' }]}
        yFormat={(v) => (v >= 0 ? `+${v.toFixed(3)}` : v.toFixed(3))}
        caption={`Against a clean score of ${data.clean_average_precision.toFixed(4)}. Negative values are leaks that happened to lower the score. The table below shows which detector caught each one.`}
      />

      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', minWidth: 860, borderCollapse: 'collapse', fontSize: 'var(--text-sm)' }}>
          <thead>
            <tr style={{ borderBottom: '2px solid var(--axis)' }}>
              {['Variant', 'Family', 'Average precision', 'Inflation', 'Static audit', 'Runtime assertions'].map((h, i) => (
                <th key={h} style={{ ...th, textAlign: i >= 2 && i <= 3 ? 'right' : 'left' }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.variants.map((v) => {
              const caught = v.static_detected || v.runtime_detected;
              const on = v.id === selected;
              return (
                <tr key={v.id} onClick={() => setSelected(v.id)} aria-selected={on}
                    style={{ borderBottom: '1px solid var(--gridline)', cursor: 'pointer',
                      background: on ? 'var(--surface-2)' : 'transparent',
                      borderLeft: `3px solid ${on ? 'var(--series-1)' : 'transparent'}` }}>
                  <td style={td}>
                    <strong>{v.title}</strong>
                    {v.kind === 'leak' && !caught && (
                      <span style={{ marginLeft: 8, fontSize: 'var(--text-xs)', fontWeight: 700,
                        color: 'var(--status-critical)', border: '1px solid currentColor',
                        borderRadius: 'var(--radius-sm)', padding: '0 5px' }}>MISSED</span>
                    )}
                    <div style={{ color: 'var(--text-muted)', fontSize: 'var(--text-xs)', whiteSpace: 'normal', maxWidth: 360 }}>
                      {v.explanation}
                    </div>
                  </td>
                  <td style={{ ...td, color: 'var(--text-secondary)' }}>{v.kind === 'control' ? 'clean control' : v.category}</td>
                  <td style={num}>{v.average_precision.toFixed(4)}</td>
                  <td style={num}>{v.kind === 'control' ? '—' : (v.inflation >= 0 ? '+' : '') + v.inflation.toFixed(3)}</td>
                  <Verdict kind={v.kind} hit={v.static_detected} detail={v.static_rules.join(', ')} />
                  <Verdict kind={v.kind} hit={v.runtime_detected} detail={v.runtime_check ?? ''} />
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <section>
        <h3 style={{ margin: '0 0 var(--space-2)', fontSize: 'var(--text-sm)' }}>
          Source of <code>{selected}</code>
        </h3>
        <pre style={{
          margin: 0, padding: 'var(--space-4)', overflowX: 'auto', maxHeight: 420,
          background: 'var(--surface-2)', border: '1px solid var(--border)', borderRadius: 'var(--radius)',
          fontSize: 'var(--text-xs)', lineHeight: 1.5,
        }}>
          {source}
        </pre>
      </section>
    </div>
  );
}

function Verdict({ kind, hit, detail }: { kind: string; hit: boolean; detail: string }) {
  // For a leak, a flag is a catch; for a clean control, a flag would be a false alarm.
  const good = kind === 'leak' ? hit : !hit;
  const label = kind === 'leak' ? (hit ? '✓ caught' : '— silent') : (hit ? '✕ false alarm' : '✓ quiet');
  return (
    <td style={{ ...td, fontWeight: 600, color: good ? 'var(--delta-up)' : 'var(--text-muted)' }}>
      {label}
      {hit && detail && (
        <div style={{ fontWeight: 400, color: 'var(--text-muted)', fontSize: 'var(--text-xs)' }}>{detail}</div>
      )}
    </td>
  );
}

const th: React.CSSProperties = {
  padding: '8px 10px', color: 'var(--text-muted)', fontWeight: 600,
  fontSize: 'var(--text-xs)', textTransform: 'uppercase', letterSpacing: 0.3, whiteSpace: 'nowrap',
};
const td: React.CSSProperties = { padding: '9px 10px', color: 'var(--text-primary)', verticalAlign: 'top' };
const num: React.CSSProperties = { ...td, textAlign: 'right', whiteSpace: 'nowrap' };
