/**
 * CrispDmPanel -- walks the six CRISP-DM phases the training run recorded.
 *
 * Because the backend builds this report while it trains, every phase gate,
 * finding and figure shown here reflects the run that actually happened.
 */

import { useState } from 'react';
import type { CrispDmReport, CrispPhase } from '../types';
import { MarkdownMath } from './MarkdownMath';

const PHASE_LABEL: Record<string, string> = {
  business_understanding: 'Business',
  data_understanding: 'Data',
  data_preparation: 'Preparation',
  modeling: 'Modeling',
  evaluation: 'Evaluation',
  deployment: 'Deployment',
};

export function CrispDmPanel({ report }: { report: CrispDmReport }) {
  const [active, setActive] = useState(0);
  const phases = report.phases ?? [];
  if (!phases.length) return null;
  const phase: CrispPhase = phases[Math.min(active, phases.length - 1)];

  return (
    <section
      style={{
        background: 'var(--surface-1)',
        border: '1px solid var(--border)',
        borderRadius: 'var(--radius-lg)',
        boxShadow: 'var(--shadow)',
        overflow: 'hidden',
      }}
    >
      {report.business_question && (
        <div
          style={{
            padding: 'var(--space-4) var(--space-5)',
            borderBottom: '1px solid var(--gridline)',
            background: 'var(--surface-2)',
          }}
        >
          <span
            style={{
              fontSize: 'var(--text-xs)',
              fontWeight: 700,
              textTransform: 'uppercase',
              letterSpacing: 0.4,
              color: 'var(--text-muted)',
            }}
          >
            Business question
          </span>
          <p style={{ margin: '4px 0 0', fontSize: 'var(--text-lg)' }}>
            {report.business_question}
          </p>
        </div>
      )}

      {/* Phase stepper */}
      <nav
        aria-label="CRISP-DM phases"
        style={{
          display: 'flex',
          gap: 0,
          overflowX: 'auto',
          borderBottom: '1px solid var(--gridline)',
        }}
      >
        {phases.map((p, i) => {
          const on = i === active;
          return (
            <button
              key={p.name}
              type="button"
              onClick={() => setActive(i)}
              aria-current={on ? 'step' : undefined}
              style={{
                flex: '1 0 auto',
                padding: 'var(--space-3) var(--space-4)',
                border: 'none',
                borderBottom: `2px solid ${on ? 'var(--series-1)' : 'transparent'}`,
                background: on ? 'var(--surface-1)' : 'transparent',
                color: on ? 'var(--text-primary)' : 'var(--text-muted)',
                fontWeight: on ? 700 : 500,
                fontSize: 'var(--text-sm)',
                cursor: 'pointer',
                whiteSpace: 'nowrap',
                display: 'flex',
                alignItems: 'center',
                gap: 6,
              }}
            >
              <span
                aria-hidden
                style={{
                  display: 'grid',
                  placeItems: 'center',
                  width: 18,
                  height: 18,
                  borderRadius: '50%',
                  fontSize: 11,
                  fontWeight: 700,
                  color: '#fff',
                  background: p.gate_passed ? 'var(--status-good)' : 'var(--status-critical)',
                }}
              >
                {p.gate_passed ? '✓' : '✕'}
              </span>
              {i + 1}. {PHASE_LABEL[p.name] ?? p.name}
            </button>
          );
        })}
      </nav>

      <div style={{ padding: 'var(--space-5)' }}>
        <h3 style={{ margin: '0 0 var(--space-1)', fontSize: 'var(--text-xl)' }}>
          {phase.title}
        </h3>
        {phase.objective && (
          <p style={{ margin: '0 0 var(--space-4)', color: 'var(--text-secondary)' }}>
            {phase.objective}
          </p>
        )}

        {phase.findings?.length > 0 && (
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(190px, 1fr))',
              gap: 'var(--space-3)',
              marginBottom: 'var(--space-5)',
            }}
          >
            {phase.findings.map((f) => (
              <div
                key={f.label}
                style={{
                  background: 'var(--surface-2)',
                  borderRadius: 'var(--radius)',
                  padding: 'var(--space-3)',
                  border: '1px solid var(--border)',
                }}
              >
                <div
                  style={{
                    fontSize: 'var(--text-xs)',
                    color: 'var(--text-muted)',
                    textTransform: 'uppercase',
                    letterSpacing: 0.3,
                    fontWeight: 600,
                  }}
                >
                  {f.label}
                </div>
                <div
                  className="tabular-nums"
                  style={{ fontSize: 'var(--text-xl)', fontWeight: 600, lineHeight: 1.3 }}
                >
                  {f.value}
                </div>
                {f.detail && (
                  <div style={{ fontSize: 'var(--text-xs)', color: 'var(--text-secondary)' }}>
                    {f.detail}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {phase.narrative && <MarkdownMath>{phase.narrative}</MarkdownMath>}

        {phase.figures?.length > 0 && (
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
              gap: 'var(--space-3)',
              marginTop: 'var(--space-4)',
            }}
          >
            {phase.figures.map((src) => (
              <img
                key={src}
                src={src}
                alt=""
                style={{
                  width: '100%',
                  borderRadius: 'var(--radius)',
                  border: '1px solid var(--border)',
                }}
              />
            ))}
          </div>
        )}

        {phase.code_refs?.length > 0 && (
          <p style={{ marginTop: 'var(--space-4)', fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}>
            Source:{' '}
            {phase.code_refs.map((c, i) => (
              <span key={c}>
                {i > 0 && ', '}
                <code>{c}</code>
              </span>
            ))}
          </p>
        )}

        <div
          style={{
            marginTop: 'var(--space-4)',
            padding: 'var(--space-3)',
            borderRadius: 'var(--radius)',
            borderLeft: `3px solid ${phase.gate_passed ? 'var(--status-good)' : 'var(--status-critical)'}`,
            background: 'var(--surface-2)',
            fontSize: 'var(--text-sm)',
          }}
        >
          <strong>
            Phase gate: {phase.gate_passed ? 'PASSED' : 'FAILED'}
          </strong>
          {phase.gate_notes && (
            <span style={{ color: 'var(--text-secondary)' }}> — {phase.gate_notes}</span>
          )}
        </div>
      </div>
    </section>
  );
}
