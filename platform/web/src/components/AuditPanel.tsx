/**
 * AuditPanel -- the leakage audit, published in the running app.
 *
 * Two halves, matching `dsfs.leakage`:
 *
 *  - executed assertions: checks that actually ran during training, each
 *    carrying the values it measured. These come from the assertion ledger,
 *    so deleting a check from a training script removes its evidence here
 *    rather than leaving a claim behind.
 *  - static findings: the AST pass over the project source.
 *
 * The component counts only the array it was handed, which is tautological
 * rather than derived. Headline severity counts come from the run record when
 * supplied.
 */

import { useMemo, useState } from 'react';
import type { AuditFinding, Severity } from '../types';

const STATUS: Record<Severity, { color: string; icon: string; label: string }> = {
  CRITICAL: { color: 'var(--status-critical)', icon: '✕', label: 'Critical' },
  WARNING: { color: 'var(--status-warning)', icon: '!', label: 'Warning' },
  INFO: { color: 'var(--text-muted)', icon: 'i', label: 'Info' },
};

/** One entry from the runtime assertion ledger. */
export interface ExecutedAssertion {
  name: string;
  passed: boolean;
  detail: string;
  params?: Record<string, unknown>;
  at?: string;
}

export interface AuditPanelProps {
  findings: AuditFinding[];
  /** From run.audit.runtime_assertions -- checks that executed. */
  runtimeAssertions?: ExecutedAssertion[];
  /** Authoritative counts from run.audit, when available. */
  summary?: { critical?: number; warnings?: number; total?: number };
  title?: string;
  groupBy?: 'severity' | 'rule' | 'file';
  defaultFilter?: Severity | 'ALL';
  onSelectFinding?: (f: AuditFinding) => void;
}

export function AuditPanel({
  findings,
  runtimeAssertions = [],
  summary,
  title = 'Leakage & reproducibility audit',
  defaultFilter = 'ALL',
  onSelectFinding,
}: AuditPanelProps) {
  const [filter, setFilter] = useState<Severity | 'ALL'>(defaultFilter);
  const [openAssertion, setOpenAssertion] = useState<string | null>(null);

  const counts = useMemo(() => {
    const c: Record<string, number> = { CRITICAL: 0, WARNING: 0, INFO: 0 };
    findings.forEach((f) => { c[f.severity] = (c[f.severity] ?? 0) + 1; });
    return c;
  }, [findings]);

  const nCritical = summary?.critical ?? counts.CRITICAL;
  const passed = nCritical === 0;
  const shown = filter === 'ALL' ? findings : findings.filter((f) => f.severity === filter);
  const assertionsPassed = runtimeAssertions.filter((a) => a.passed).length;

  return (
    <section style={{
      background: 'var(--surface-1)', border: '1px solid var(--border)',
      borderRadius: 'var(--radius-lg)', padding: 'var(--space-5)',
      boxShadow: 'var(--shadow)',
    }}>
      <header style={{
        display: 'flex', alignItems: 'center', gap: 'var(--space-3)', flexWrap: 'wrap',
      }}>
        <h3 style={{ margin: 0, fontSize: 'var(--text-lg)' }}>{title}</h3>
        <span style={{
          display: 'inline-flex', alignItems: 'center', gap: 6,
          padding: '3px 10px', borderRadius: 999, fontSize: 'var(--text-xs)',
          fontWeight: 700, color: '#fff',
          background: passed ? 'var(--status-good)' : 'var(--status-critical)',
        }}>
          <span aria-hidden>{passed ? '✓' : '✕'}</span>
          {passed ? 'PASS' : 'FAIL'}
        </span>
      </header>

      {/* -- checks that actually executed ------------------------------- */}
      {runtimeAssertions.length > 0 && (
        <div style={{ marginTop: 'var(--space-4)' }}>
          <h4 style={subhead}>
            Assertions executed during training
            <span style={{
              marginLeft: 8, color: 'var(--text-muted)', fontWeight: 500,
              textTransform: 'none', letterSpacing: 0,
            }}>
              {assertionsPassed}/{runtimeAssertions.length} passed
            </span>
          </h4>
          <p style={{
            margin: '0 0 var(--space-3)', color: 'var(--text-secondary)',
            fontSize: 'var(--text-sm)',
          }}>
            These ran inside the training script and would have halted it on
            failure. Each records the values it measured, so the evidence
            disappears if the check is removed.
          </p>
          <ul style={{ listStyle: 'none', margin: 0, padding: 0 }}>
            {runtimeAssertions.map((a) => {
              const open = openAssertion === a.name;
              const hasParams = a.params && Object.keys(a.params).length > 0;
              return (
                <li key={a.name} style={{
                  border: '1px solid var(--border)', borderRadius: 'var(--radius)',
                  marginBottom: 'var(--space-2)', overflow: 'hidden',
                }}>
                  <button
                    type="button"
                    onClick={() => hasParams && setOpenAssertion(open ? null : a.name)}
                    aria-expanded={hasParams ? open : undefined}
                    style={{
                      width: '100%', display: 'flex', alignItems: 'flex-start',
                      gap: 'var(--space-3)', padding: 'var(--space-3)',
                      background: 'transparent', border: 'none', textAlign: 'left',
                      cursor: hasParams ? 'pointer' : 'default', font: 'inherit',
                    }}
                  >
                    <span aria-hidden style={{
                      flexShrink: 0, display: 'grid', placeItems: 'center',
                      width: 20, height: 20, borderRadius: '50%', color: '#fff',
                      fontSize: 12, fontWeight: 700,
                      background: a.passed ? 'var(--status-good)' : 'var(--status-critical)',
                    }}>
                      {a.passed ? '✓' : '✕'}
                    </span>
                    <span style={{ minWidth: 0, flex: 1 }}>
                      <code style={{ fontWeight: 700, color: 'var(--text-primary)' }}>
                        {a.name}
                      </code>
                      <span style={{
                        marginLeft: 8, fontSize: 'var(--text-xs)', fontWeight: 600,
                        color: a.passed ? 'var(--delta-up)' : 'var(--status-critical)',
                      }}>
                        {a.passed ? 'PASSED' : 'FAILED'}
                      </span>
                      <p style={{
                        margin: '2px 0 0', color: 'var(--text-secondary)',
                        fontSize: 'var(--text-sm)', whiteSpace: 'normal',
                      }}>
                        {a.detail}
                      </p>
                    </span>
                    {hasParams && (
                      <span aria-hidden style={{ color: 'var(--text-muted)', fontSize: 12 }}>
                        {open ? '▾' : '▸'}
                      </span>
                    )}
                  </button>
                  {open && hasParams && (
                    <dl style={{
                      margin: 0, padding: 'var(--space-3)',
                      background: 'var(--surface-2)',
                      borderTop: '1px solid var(--gridline)',
                      display: 'grid', gridTemplateColumns: 'auto 1fr',
                      gap: '4px var(--space-3)', fontSize: 'var(--text-xs)',
                    }}>
                      {Object.entries(a.params!).map(([k, v]) => (
                        <div key={k} style={{ display: 'contents' }}>
                          <dt style={{ color: 'var(--text-muted)' }}>{k}</dt>
                          <dd className="tabular-nums" style={{
                            margin: 0, color: 'var(--text-primary)',
                            overflowWrap: 'anywhere',
                          }}>
                            {Array.isArray(v)
                              ? (v.length ? v.join(', ') : '— none —')
                              : String(v)}
                          </dd>
                        </div>
                      ))}
                    </dl>
                  )}
                </li>
              );
            })}
          </ul>
        </div>
      )}

      {/* -- static findings --------------------------------------------- */}
      <div style={{ marginTop: 'var(--space-5)' }}>
        <div style={{
          display: 'flex', alignItems: 'center', gap: 'var(--space-2)', flexWrap: 'wrap',
        }}>
          <h4 style={{ ...subhead, margin: 0 }}>Static source analysis</h4>
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
                    padding: '4px 10px', borderRadius: 'var(--radius-sm)',
                    border: '1px solid var(--border)',
                    background: active ? 'var(--surface-2)' : 'transparent',
                    color: active ? 'var(--text-primary)' : 'var(--text-muted)',
                    fontSize: 'var(--text-xs)', fontWeight: 600, cursor: 'pointer',
                  }}
                >
                  {k === 'ALL' ? 'All' : STATUS[k].label} ({n})
                </button>
              );
            })}
          </div>
        </div>

        {shown.length === 0 ? (
          <p style={{
            margin: 'var(--space-3) 0 0', padding: 'var(--space-4)',
            textAlign: 'center', color: 'var(--text-muted)',
            background: 'var(--surface-2)', borderRadius: 'var(--radius)',
            fontSize: 'var(--text-sm)',
          }}>
            {findings.length === 0
              ? 'The AST pass found nothing: transformers are fitted inside folds, splits are disjoint, estimators are seeded.'
              : `No ${filter.toLowerCase()} findings.`}
          </p>
        ) : (
          <ul style={{ listStyle: 'none', margin: 'var(--space-3) 0 0', padding: 0 }}>
            {shown.map((f, i) => {
              const s = STATUS[f.severity];
              return (
                <li
                  key={`${f.file}:${f.line}:${i}`}
                  onClick={onSelectFinding ? () => onSelectFinding(f) : undefined}
                  style={{
                    display: 'flex', gap: 'var(--space-3)',
                    padding: 'var(--space-3) 0',
                    borderTop: '1px solid var(--gridline)',
                    cursor: onSelectFinding ? 'pointer' : 'default',
                  }}
                >
                  <span aria-hidden style={{
                    flexShrink: 0, display: 'grid', placeItems: 'center',
                    width: 20, height: 20, borderRadius: '50%',
                    background: s.color, color: '#fff', fontSize: 12, fontWeight: 700,
                  }}>
                    {s.icon}
                  </span>
                  <div style={{ minWidth: 0 }}>
                    <div style={{
                      display: 'flex', gap: 'var(--space-2)', flexWrap: 'wrap',
                      alignItems: 'baseline',
                    }}>
                      <strong style={{ fontSize: 'var(--text-sm)' }}>{s.label}</strong>
                      <code style={{ color: 'var(--text-secondary)' }}>{f.rule}</code>
                      {f.file && (
                        <code style={{ color: 'var(--text-muted)', fontSize: 'var(--text-xs)' }}>
                          {f.file}:{f.line}
                        </code>
                      )}
                    </div>
                    <p style={{
                      margin: '2px 0 0', color: 'var(--text-secondary)',
                      fontSize: 'var(--text-sm)',
                    }}>
                      {f.message}
                    </p>
                  </div>
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </section>
  );
}

const subhead: React.CSSProperties = {
  margin: '0 0 var(--space-2)', fontSize: 'var(--text-xs)',
  textTransform: 'uppercase', letterSpacing: 0.4,
  color: 'var(--text-muted)', fontWeight: 700,
};
