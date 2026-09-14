/**
 * CrispDmWalkthrough -- the six phases, driven entirely by report data.
 *
 * Nothing about any project is hardcoded here. Phases, findings, figures,
 * code references and gate outcomes all arrive from `/api/crispdm`, which the
 * training run wrote as it executed.
 *
 * `slots` is what keeps this general: a project injects its own visual into a
 * named phase -- the taxi zone map into data_understanding, a cluster scatter
 * into modeling, a forecast fan into evaluation -- without this component
 * knowing that any of those exist.
 */

import { useState } from 'react';
import type { CrispDmReport, CrispPhase, PhaseName } from '../types';
import { MarkdownMath } from './MarkdownMath';

const PHASE_LABEL: Record<string, string> = {
  business_understanding: 'Business',
  data_understanding: 'Data',
  data_preparation: 'Preparation',
  modeling: 'Modeling',
  evaluation: 'Evaluation',
  deployment: 'Deployment',
};

export interface CrispDmWalkthroughProps {
  report: CrispDmReport;
  /** 'stepper' for tabs; 'continuous' for one paper-style scroll. */
  layout?: 'stepper' | 'continuous';
  initialPhase?: PhaseName;
  onPhaseChange?: (p: PhaseName) => void;
  /** Project-specific visuals, keyed by phase. */
  slots?: Partial<Record<PhaseName, React.ReactNode>>;
  /** Prefix for relative figure paths. */
  figureBase?: string;
}

export function CrispDmWalkthrough({
  report,
  layout = 'stepper',
  initialPhase,
  onPhaseChange,
  slots,
  figureBase = '',
}: CrispDmWalkthroughProps) {
  const phases = report.phases ?? [];
  const startIndex = Math.max(
    0,
    initialPhase ? phases.findIndex((p) => p.name === initialPhase) : 0,
  );
  const [active, setActive] = useState(startIndex);

  if (!phases.length) return null;

  const select = (i: number) => {
    setActive(i);
    onPhaseChange?.(phases[i].name);
  };

  return (
    <section style={{
      background: 'var(--surface-1)', border: '1px solid var(--border)',
      borderRadius: 'var(--radius-lg)', boxShadow: 'var(--shadow)',
      overflow: 'hidden',
    }}>
      {report.business_question && (
        <div style={{
          padding: 'var(--space-4) var(--space-5)',
          borderBottom: '1px solid var(--gridline)',
          background: 'var(--surface-2)',
        }}>
          <span style={{
            fontSize: 'var(--text-xs)', fontWeight: 700, textTransform: 'uppercase',
            letterSpacing: 0.4, color: 'var(--text-muted)',
          }}>
            Business question
          </span>
          <p style={{ margin: '4px 0 0', fontSize: 'var(--text-lg)' }}>
            {report.business_question}
          </p>
        </div>
      )}

      {layout === 'stepper' && (
        <nav aria-label="CRISP-DM phases" style={{
          display: 'flex', overflowX: 'auto', borderBottom: '1px solid var(--gridline)',
        }}>
          {phases.map((p, i) => {
            const on = i === active;
            return (
              <button
                key={p.name}
                type="button"
                onClick={() => select(i)}
                aria-current={on ? 'step' : undefined}
                style={{
                  flex: '1 0 auto', padding: 'var(--space-3) var(--space-4)',
                  border: 'none',
                  borderBottom: `2px solid ${on ? 'var(--series-1)' : 'transparent'}`,
                  background: on ? 'var(--surface-1)' : 'transparent',
                  color: on ? 'var(--text-primary)' : 'var(--text-muted)',
                  fontWeight: on ? 700 : 500, fontSize: 'var(--text-sm)',
                  cursor: 'pointer', whiteSpace: 'nowrap',
                  display: 'flex', alignItems: 'center', gap: 6,
                }}
              >
                <GateDot passed={p.gate_passed} />
                {i + 1}. {PHASE_LABEL[p.name] ?? p.name}
              </button>
            );
          })}
        </nav>
      )}

      {layout === 'stepper' ? (
        <PhaseBody phase={phases[active]} slot={slots?.[phases[active].name]}
                   figureBase={figureBase} />
      ) : (
        phases.map((p, i) => (
          <div key={p.name} style={{
            borderTop: i === 0 ? 'none' : '1px solid var(--gridline)',
          }}>
            <PhaseBody phase={p} slot={slots?.[p.name]} figureBase={figureBase} />
          </div>
        ))
      )}
    </section>
  );
}

function PhaseBody({
  phase,
  slot,
  figureBase,
}: {
  phase: CrispPhase;
  slot?: React.ReactNode;
  figureBase: string;
}) {
  return (
    <div style={{ padding: 'var(--space-5)' }}>
      <h3 style={{
        margin: '0 0 var(--space-1)', fontSize: 'var(--text-xl)',
        display: 'flex', alignItems: 'center', gap: 'var(--space-2)',
      }}>
        <GateDot passed={phase.gate_passed} />
        {phase.title}
      </h3>
      {phase.objective && (
        <p style={{ margin: '0 0 var(--space-4)', color: 'var(--text-secondary)' }}>
          {phase.objective}
        </p>
      )}

      {phase.findings?.length > 0 && (
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(190px, 1fr))',
          gap: 'var(--space-3)', marginBottom: 'var(--space-5)',
        }}>
          {phase.findings.map((f) => (
            <div key={f.label} style={{
              background: 'var(--surface-2)', borderRadius: 'var(--radius)',
              padding: 'var(--space-3)', border: '1px solid var(--border)',
            }}>
              <div style={{
                fontSize: 'var(--text-xs)', color: 'var(--text-muted)',
                textTransform: 'uppercase', letterSpacing: 0.3, fontWeight: 600,
              }}>
                {f.label}
              </div>
              <div className="tabular-nums" style={{
                fontSize: 'var(--text-xl)', fontWeight: 600, lineHeight: 1.3,
              }}>
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

      {/* Project-specific visual for this phase, supplied by the caller. */}
      {slot && <div style={{ marginTop: 'var(--space-4)' }}>{slot}</div>}

      {phase.figures?.length > 0 && (
        <div style={{
          display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
          gap: 'var(--space-3)', marginTop: 'var(--space-4)',
        }}>
          {phase.figures.map((src) => (
            <img key={src} src={`${figureBase}${src}`} alt=""
                 style={{
                   width: '100%', borderRadius: 'var(--radius)',
                   border: '1px solid var(--border)',
                 }} />
          ))}
        </div>
      )}

      {phase.code_refs?.length > 0 && (
        <p style={{
          marginTop: 'var(--space-4)', fontSize: 'var(--text-xs)',
          color: 'var(--text-muted)',
        }}>
          Source:{' '}
          {phase.code_refs.map((c, i) => (
            <span key={c}>{i > 0 && ', '}<code>{c}</code></span>
          ))}
        </p>
      )}

      <div style={{
        marginTop: 'var(--space-4)', padding: 'var(--space-3)',
        borderRadius: 'var(--radius)',
        borderLeft: `3px solid ${phase.gate_passed ? 'var(--status-good)' : 'var(--status-critical)'}`,
        background: 'var(--surface-2)', fontSize: 'var(--text-sm)',
      }}>
        <strong>Phase gate: {phase.gate_passed ? 'PASSED' : 'FAILED'}</strong>
        {phase.gate_notes && (
          <span style={{ color: 'var(--text-secondary)' }}> — {phase.gate_notes}</span>
        )}
      </div>
    </div>
  );
}

function GateDot({ passed }: { passed: boolean }) {
  return (
    <span aria-hidden style={{
      display: 'grid', placeItems: 'center', width: 18, height: 18,
      borderRadius: '50%', fontSize: 11, fontWeight: 700, color: '#fff',
      flexShrink: 0,
      background: passed ? 'var(--status-good)' : 'var(--status-critical)',
    }}>
      {passed ? '✓' : '✕'}
    </span>
  );
}
