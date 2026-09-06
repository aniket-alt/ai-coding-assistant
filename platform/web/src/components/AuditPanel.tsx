/**
 * AuditPanel -- shows the leakage audit in the app itself.
 *
 * Most portfolios bury the audit in a markdown file nobody opens. Publishing
 * it in the running UI, including failures, is the point: the findings here are
 * produced by `dsfs.leakage` on every training run, not written by hand.
 */

import { useMemo, useState } from 'react';
import type { AuditFinding, Severity } from '../types';

const STATUS: Record<Severity, { color: string; icon: string; label: string }> = {
  CRITICAL: { color: 'var(--status-critical)', icon: '✕', label: 'Critical' },
  WARNING: { color: 'var(--status-warning)', icon: '!', label: 'Warning' },
  INFO: { color: 'var(--text-muted)', icon: 'i', label: 'Info' },
};

export function AuditPanel({ findings }: { findings: AuditFinding[] }) {
  const [filter, setFilter] = useState<Severity | 'ALL'>('ALL');

  const counts = useMemo(() => {
    const c: Record<string, number> = { CRITICAL: 0, WARNING: 0, INFO: 0 };
    findings.forEach((f) => {
      c[f.severity] = (c[f.severity] ?? 0) + 1;
    });
    return c;
  }, [findings]);

  const shown = filter === 'ALL' ? findings : findings.filter((f) => f.severity === filter);
  const passed = counts.CRITICAL === 0;

  return (
    <section
      style={{
        background: 'var(--surface-1)',
        border: '1px solid var(--border)',
        borderRadius: 'var(--radius-lg)',
        padding: 'var(--space-5)',
        boxShadow: 'var(--shadow)',
      }}
    >
      <header style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-3)', flexWrap: 'wrap' }}>
        <h3 style={{ margin: 0, fontSize: 'var(--text-lg)' }}>Leakage &amp; reproducibility audit</h3>
        <span
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: 6,
            padding: '3px 10px',
            borderRadius: 999,
            fontSize: 'var(--text-xs)',
            fontWeight: 700,
            color: '#fff',
            background: passed ? 'var(--status-good)' : 'var(--status-critical)',
          }}
        >
          <span aria-hidden>{passed ? '✓' : '✕'}</span>
          {passed ? 'PASS' : 'FAIL'}
        </span>

        <div style={{ marginLeft: 'auto', display: 'flex', gap: 'var(--space-1)' }}>
          {(['ALL', 'CRITICAL', 'WARNING', 'INFO'] as const).map((k) => {
            const n = k === 'ALL' ? findings.length : counts[k] ?? 0;
            const active = filter === k;
            return (
              <button
                key={k}
                type="button"
                onClick={() => setFilter(k)}
                style={{
                  padding: '4px 10px',
                  borderRadius: 'var(--radius-sm)',
                  border: '1px solid var(--border)',
                  background: active ? 'var(--surface-2)' : 'transparent',
                  color: active ? 'var(--text-primary)' : 'var(--text-muted)',
                  fontSize: 'var(--text-xs)',
                  fontWeight: 600,
                  cursor: 'pointer',
                }}
              >
                {k === 'ALL' ? 'All' : STATUS[k].label} ({n})
              </button>
            );
          })}
        </div>
      </header>

      <p style={{ color: 'var(--text-secondary)', fontSize: 'var(--text-sm)', marginTop: 'var(--space-3)' }}>
        Produced by an AST pass over this project&rsquo;s source combined with runtime
        assertions that execute during training. A critical finding fails CI.
      </p>

      {shown.length === 0 ? (
        <p
          style={{
            margin: 0,
            padding: 'var(--space-5)',
            textAlign: 'center',
            color: 'var(--text-muted)',
            background: 'var(--surface-2)',
            borderRadius: 'var(--radius)',
          }}
        >
          {findings.length === 0
            ? 'No findings. Transformers are fitted inside CV folds, splits are disjoint, estimators are seeded.'
            : `No ${filter.toLowerCase()} findings.`}
        </p>
      ) : (
        <ul style={{ listStyle: 'none', margin: 'var(--space-4) 0 0', padding: 0 }}>
          {shown.map((f, i) => {
            const s = STATUS[f.severity];
            return (
              <li
                key={`${f.file}:${f.line}:${i}`}
                style={{
                  display: 'flex',
                  gap: 'var(--space-3)',
                  padding: 'var(--space-3) 0',
                  borderTop: '1px solid var(--gridline)',
                }}
              >
                <span
                  aria-hidden
                  style={{
                    flexShrink: 0,
                    display: 'grid',
                    placeItems: 'center',
                    width: 20,
                    height: 20,
                    borderRadius: '50%',
                    background: s.color,
                    color: '#fff',
                    fontSize: 12,
                    fontWeight: 700,
                  }}
                >
                  {s.icon}
                </span>
                <div style={{ minWidth: 0 }}>
                  <div style={{ display: 'flex', gap: 'var(--space-2)', flexWrap: 'wrap', alignItems: 'baseline' }}>
                    <strong style={{ fontSize: 'var(--text-sm)' }}>{s.label}</strong>
                    <code style={{ color: 'var(--text-secondary)' }}>{f.rule}</code>
                    {f.file && (
                      <code style={{ color: 'var(--text-muted)', fontSize: 'var(--text-xs)' }}>
                        {f.file}:{f.line}
                      </code>
                    )}
                  </div>
                  <p style={{ margin: '2px 0 0', color: 'var(--text-secondary)', fontSize: 'var(--text-sm)' }}>
                    {f.message}
                  </p>
                </div>
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}
