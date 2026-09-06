/**
 * ProvenanceBadge -- renders the dataset receipt.
 *
 * This component is the portfolio's thesis made visible: every app states, in
 * its own UI, exactly which real dataset it was trained on, under what licence,
 * and with what checksum. If a project ever runs on generated data, the badge
 * says so loudly rather than hiding it in a docstring.
 */

import type { Provenance } from '../types';

function humanBytes(n: number): string {
  if (!n) return '—';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let v = n;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i += 1;
  }
  return `${v.toFixed(1)} ${units[i]}`;
}

export interface ProvenanceBadgeProps {
  provenance: Provenance;
  /** `compact` for a header chip, `full` for the data-understanding panel. */
  variant?: 'compact' | 'full';
}

export function ProvenanceBadge({
  provenance: p,
  variant = 'full',
}: ProvenanceBadgeProps) {
  const real = !p.synthetic;
  // Status colours are reserved and always ship with an icon + label, never
  // colour alone.
  const accent = real ? 'var(--status-good)' : 'var(--status-warning)';
  const label = real ? 'Real third-party data' : 'Synthetic data';
  const icon = real ? '✓' : '⚠';

  if (variant === 'compact') {
    return (
      <a
        href={p.source_url}
        target="_blank"
        rel="noreferrer noopener"
        title={`${p.name} — ${p.license}`}
        style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: 'var(--space-2)',
          padding: '4px 10px',
          borderRadius: 999,
          border: '1px solid var(--border)',
          background: 'var(--surface-1)',
          color: 'var(--text-secondary)',
          fontSize: 'var(--text-xs)',
          textDecoration: 'none',
          whiteSpace: 'nowrap',
        }}
      >
        <span aria-hidden style={{ color: accent, fontWeight: 700 }}>{icon}</span>
        <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>
          {p.n_rows ? `${p.n_rows.toLocaleString()} rows` : label}
        </span>
        <span>· {p.name}</span>
      </a>
    );
  }

  const rows: Array<[string, React.ReactNode]> = [
    ['Dataset', p.name],
    [
      'Source',
      <a key="src" href={p.source_url} target="_blank" rel="noreferrer noopener">
        {p.source_url}
      </a>,
    ],
    ['Licence', p.license],
    ['Retrieved', p.retrieved_at || '—'],
    [
      'SHA-256',
      <code key="sha" style={{ wordBreak: 'break-all' }}>
        {p.sha256 ? `${p.sha256.slice(0, 32)}…` : '—'}
      </code>,
    ],
    ['Rows × Columns', `${p.n_rows.toLocaleString()} × ${p.n_cols.toLocaleString()}`],
    ['Size on disk', humanBytes(p.bytes_on_disk)],
  ];

  return (
    <section
      aria-labelledby="prov-heading"
      style={{
        background: 'var(--surface-1)',
        border: '1px solid var(--border)',
        borderRadius: 'var(--radius-lg)',
        padding: 'var(--space-5)',
        boxShadow: 'var(--shadow)',
      }}
    >
      <header
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 'var(--space-2)',
          marginBottom: 'var(--space-4)',
        }}
      >
        <span
          aria-hidden
          style={{
            display: 'grid',
            placeItems: 'center',
            width: 22,
            height: 22,
            borderRadius: '50%',
            background: accent,
            color: '#fff',
            fontSize: 13,
            fontWeight: 700,
          }}
        >
          {icon}
        </span>
        <h3 id="prov-heading" style={{ margin: 0, fontSize: 'var(--text-lg)' }}>
          Dataset provenance
        </h3>
        <span
          style={{
            marginLeft: 'auto',
            fontSize: 'var(--text-xs)',
            fontWeight: 700,
            letterSpacing: 0.4,
            textTransform: 'uppercase',
            color: accent,
          }}
        >
          {label}
        </span>
      </header>

      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 'var(--text-sm)' }}>
        <tbody>
          {rows.map(([k, v]) => (
            <tr key={k} style={{ borderTop: '1px solid var(--gridline)' }}>
              <th
                scope="row"
                style={{
                  textAlign: 'left',
                  padding: '7px 0',
                  color: 'var(--text-muted)',
                  fontWeight: 500,
                  width: '34%',
                }}
              >
                {k}
              </th>
              <td style={{ padding: '7px 0', color: 'var(--text-primary)' }}>{v}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <p
        style={{
          margin: 'var(--space-4) 0 0',
          paddingTop: 'var(--space-3)',
          borderTop: '1px solid var(--gridline)',
          color: 'var(--text-secondary)',
          fontSize: 'var(--text-xs)',
          fontStyle: 'italic',
        }}
      >
        {p.citation}
      </p>
    </section>
  );
}
