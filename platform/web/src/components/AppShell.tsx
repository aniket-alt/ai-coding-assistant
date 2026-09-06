/**
 * AppShell -- the frame every project's client renders inside.
 *
 * Supplies the header (project identity, provenance chip, theme toggle), the
 * tab bar, and a consistent content well, so a project's own code is only its
 * distinctive views.
 */

import { useEffect, useState, type ReactNode } from 'react';
import type { ProjectMeta, Provenance } from '../types';
import { ProvenanceBadge } from './ProvenanceBadge';

export type Theme = 'light' | 'dark' | 'system';

export function useTheme(): [Theme, (t: Theme) => void] {
  const [theme, setTheme] = useState<Theme>(() => {
    try {
      return (localStorage.getItem('dsfs-theme') as Theme) || 'system';
    } catch {
      return 'system';
    }
  });

  useEffect(() => {
    const root = document.documentElement;
    if (theme === 'system') root.removeAttribute('data-theme');
    else root.setAttribute('data-theme', theme);
    try {
      localStorage.setItem('dsfs-theme', theme);
    } catch {
      /* private mode or blocked storage -- theme still applies for this view */
    }
  }, [theme]);

  return [theme, setTheme];
}

export interface Tab {
  id: string;
  label: string;
}

export interface AppShellProps {
  meta: ProjectMeta | null;
  provenance?: Provenance | null;
  tabs: Tab[];
  active: string;
  onTabChange: (id: string) => void;
  repoUrl?: string;
  children: ReactNode;
}

export function AppShell({
  meta,
  provenance,
  tabs,
  active,
  onTabChange,
  repoUrl,
  children,
}: AppShellProps) {
  const [theme, setTheme] = useTheme();
  const next: Theme = theme === 'dark' ? 'light' : theme === 'light' ? 'system' : 'dark';
  const icon = theme === 'dark' ? '☾' : theme === 'light' ? '☀' : '◐';

  return (
    <div style={{ minHeight: '100vh', background: 'var(--plane)' }}>
      <header
        style={{
          position: 'sticky',
          top: 0,
          zIndex: 20,
          background: 'var(--surface-1)',
          borderBottom: '1px solid var(--border)',
        }}
      >
        <div
          style={{
            maxWidth: 1400,
            margin: '0 auto',
            padding: 'var(--space-3) var(--space-5)',
            display: 'flex',
            alignItems: 'center',
            gap: 'var(--space-4)',
            flexWrap: 'wrap',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-3)', minWidth: 0 }}>
            {meta && (
              <span
                aria-hidden
                style={{
                  display: 'grid',
                  placeItems: 'center',
                  width: 30,
                  height: 30,
                  borderRadius: 'var(--radius)',
                  background: 'var(--series-1)',
                  color: '#fff',
                  fontWeight: 700,
                  fontSize: 'var(--text-sm)',
                  flexShrink: 0,
                }}
              >
                {String(meta.number).padStart(2, '0')}
              </span>
            )}
            <div style={{ minWidth: 0 }}>
              <h1
                style={{
                  margin: 0,
                  fontSize: 'var(--text-lg)',
                  fontWeight: 700,
                  whiteSpace: 'nowrap',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                }}
              >
                {meta?.title ?? 'Loading…'}
              </h1>
              {meta?.tagline && (
                <p
                  style={{
                    margin: 0,
                    fontSize: 'var(--text-xs)',
                    color: 'var(--text-muted)',
                    whiteSpace: 'nowrap',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                  }}
                >
                  {meta.tagline}
                </p>
              )}
            </div>
          </div>

          <div
            style={{
              marginLeft: 'auto',
              display: 'flex',
              alignItems: 'center',
              gap: 'var(--space-2)',
            }}
          >
            {provenance && <ProvenanceBadge provenance={provenance} variant="compact" />}
            {repoUrl && (
              <a
                href={repoUrl}
                target="_blank"
                rel="noreferrer noopener"
                style={linkBtn}
              >
                Source
              </a>
            )}
            <a href="/api/docs" target="_blank" rel="noreferrer noopener" style={linkBtn}>
              API
            </a>
            <button
              type="button"
              onClick={() => setTheme(next)}
              aria-label={`Theme: ${theme}. Switch to ${next}.`}
              title={`Theme: ${theme} → ${next}`}
              style={{ ...linkBtn, cursor: 'pointer', minWidth: 34 }}
            >
              <span aria-hidden>{icon}</span>
            </button>
          </div>
        </div>

        <nav
          aria-label="Sections"
          style={{
            maxWidth: 1400,
            margin: '0 auto',
            padding: '0 var(--space-5)',
            display: 'flex',
            gap: 'var(--space-1)',
            overflowX: 'auto',
          }}
        >
          {tabs.map((t) => {
            const on = t.id === active;
            return (
              <button
                key={t.id}
                type="button"
                onClick={() => onTabChange(t.id)}
                aria-current={on ? 'page' : undefined}
                style={{
                  padding: 'var(--space-2) var(--space-3)',
                  border: 'none',
                  borderBottom: `2px solid ${on ? 'var(--series-1)' : 'transparent'}`,
                  background: 'transparent',
                  color: on ? 'var(--text-primary)' : 'var(--text-muted)',
                  fontWeight: on ? 700 : 500,
                  fontSize: 'var(--text-sm)',
                  cursor: 'pointer',
                  whiteSpace: 'nowrap',
                }}
              >
                {t.label}
              </button>
            );
          })}
        </nav>
      </header>

      <main
        style={{
          maxWidth: 1400,
          margin: '0 auto',
          padding: 'var(--space-5)',
          display: 'flex',
          flexDirection: 'column',
          gap: 'var(--space-5)',
        }}
      >
        {children}
      </main>

      <footer
        style={{
          maxWidth: 1400,
          margin: '0 auto',
          padding: 'var(--space-5)',
          borderTop: '1px solid var(--border)',
          color: 'var(--text-muted)',
          fontSize: 'var(--text-xs)',
          display: 'flex',
          gap: 'var(--space-4)',
          flexWrap: 'wrap',
        }}
      >
        <span>
          Project {meta ? String(meta.number).padStart(2, '0') : '—'} of 16 · built with
          Claude Code
        </span>
        {meta?.dataset && <span>Dataset: {meta.dataset}</span>}
        <span style={{ marginLeft: 'auto' }}>dsfs v{meta?.dsfs_version ?? '—'}</span>
      </footer>
    </div>
  );
}

/** Section wrapper with a heading and optional description. */
export function Panel({
  title,
  description,
  actions,
  children,
}: {
  title: string;
  description?: string;
  actions?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section>
      <header
        style={{
          display: 'flex',
          alignItems: 'baseline',
          gap: 'var(--space-3)',
          marginBottom: 'var(--space-3)',
          flexWrap: 'wrap',
        }}
      >
        <h2 style={{ margin: 0, fontSize: 'var(--text-xl)' }}>{title}</h2>
        {description && (
          <p style={{ margin: 0, color: 'var(--text-secondary)', fontSize: 'var(--text-sm)' }}>
            {description}
          </p>
        )}
        {actions && <div style={{ marginLeft: 'auto' }}>{actions}</div>}
      </header>
      {children}
    </section>
  );
}

const linkBtn: React.CSSProperties = {
  padding: '5px 10px',
  borderRadius: 'var(--radius-sm)',
  border: '1px solid var(--border)',
  background: 'var(--surface-1)',
  color: 'var(--text-secondary)',
  fontSize: 'var(--text-xs)',
  fontWeight: 600,
  textDecoration: 'none',
  display: 'inline-flex',
  alignItems: 'center',
  justifyContent: 'center',
};
