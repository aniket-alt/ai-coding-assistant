/**
 * RuleTable -- project-04-specific.
 *
 * The strongest mined rules, each shown with its statistics in the training
 * period beside what it did in the following season. A rule's training
 * confidence alone says little; whether it held up is the useful part.
 */

import { useMemo, useState } from 'react';
import { Chart } from '@dsfs/ui';

export interface Rule {
  antecedent: string;
  consequent: string;
  support_pct: number;
  confidence: number;
  lift: number;
  test_confidence: number;
  test_lift: number;
  survived: boolean;
}

type SortKey = 'lift' | 'confidence' | 'drop';

export function RuleTable({ rules }: { rules: Rule[] }) {
  const [sort, setSort] = useState<SortKey>('lift');

  const rows = useMemo(() => {
    const out = [...rules];
    if (sort === 'lift') out.sort((a, b) => b.lift - a.lift);
    if (sort === 'confidence') out.sort((a, b) => b.confidence - a.confidence);
    // Largest fall in confidence between seasons first.
    if (sort === 'drop') out.sort((a, b) => (a.test_confidence - a.confidence) - (b.test_confidence - b.confidence));
    return out;
  }, [rules, sort]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-5)' }}>
      <Chart
        kind="scatter"
        height={300}
        data={rules}
        x="confidence"
        xLabel="Confidence when mined (before September)"
        series={[{ key: 'test_confidence', label: 'Confidence the following season' }]}
        yFormat={(v) => v.toFixed(2)}
        caption="Each point is one rule. Points on the diagonal kept their confidence; points below it weakened. Most sit close to the line."
      />

      <div style={{ display: 'flex', gap: 'var(--space-1)', alignItems: 'center' }}>
        <span style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}>Sort by</span>
        {([['lift', 'Lift'], ['confidence', 'Confidence'], ['drop', 'Largest decline']] as const).map(([k, label]) => (
          <button key={k} type="button" onClick={() => setSort(k)} aria-pressed={sort === k}
                  style={{
                    padding: '4px 10px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border)',
                    background: sort === k ? 'var(--surface-2)' : 'transparent',
                    color: sort === k ? 'var(--text-primary)' : 'var(--text-muted)',
                    fontSize: 'var(--text-xs)', fontWeight: 600, cursor: 'pointer',
                  }}>
            {label}
          </button>
        ))}
      </div>

      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', minWidth: 820, borderCollapse: 'collapse', fontSize: 'var(--text-sm)' }}>
          <thead>
            <tr style={{ borderBottom: '2px solid var(--axis)' }}>
              {['If the basket has', 'Recommend', 'Support', 'Confidence', 'Lift', 'Next season', 'Held?'].map((h, i) => (
                <th key={h} style={{ ...th, textAlign: i < 2 ? 'left' : 'right' }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i} style={{ borderBottom: '1px solid var(--gridline)' }}>
                <td style={{ ...td, whiteSpace: 'normal', maxWidth: 320 }}>{r.antecedent}</td>
                <td style={{ ...td, whiteSpace: 'normal', maxWidth: 240, fontWeight: 600 }}>{r.consequent}</td>
                <td style={num}>{r.support_pct.toFixed(2)}%</td>
                <td style={num}>{r.confidence.toFixed(2)}</td>
                <td style={num}>{r.lift.toFixed(1)}</td>
                <td style={num}>
                  {r.test_confidence.toFixed(2)}
                  <span style={{ color: 'var(--text-muted)', fontSize: 'var(--text-xs)' }}> · lift {r.test_lift.toFixed(1)}</span>
                </td>
                <td style={{ ...num, fontWeight: 700, color: r.survived ? 'var(--delta-up)' : 'var(--status-critical)' }}>
                  {r.survived ? '✓ yes' : '✕ no'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

const th: React.CSSProperties = {
  padding: '8px 10px', color: 'var(--text-muted)', fontWeight: 600,
  fontSize: 'var(--text-xs)', textTransform: 'uppercase', letterSpacing: 0.3, whiteSpace: 'nowrap',
};
const td: React.CSSProperties = { padding: '9px 10px', color: 'var(--text-primary)', verticalAlign: 'top' };
const num: React.CSSProperties = { ...td, textAlign: 'right', whiteSpace: 'nowrap' };
