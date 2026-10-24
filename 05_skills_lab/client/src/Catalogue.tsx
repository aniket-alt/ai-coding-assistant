/**
 * Project-05-specific views: the catalogue list, one skill's comparison, and
 * the live re-run. Every number shown is returned by the API; the client only
 * decides how many decimals a difference deserves.
 */

import { useEffect, useState } from 'react';
import { apiPost, Chart, StatRow, StatTile } from '@dsfs/ui';

export interface Side { label: string; mean: number; sd: number }
export interface Gap { mean: number; sd: number; ci_low: number; ci_high: number; decisive: boolean; flatters: boolean; negligible: boolean }
export interface SkillRecord {
  id: string; name: string; category: string; question: string; dataset: string; metric: string;
  metric_meaning: string; higher_is_better: boolean; seeds: number;
  careless: Side; careful: Side; gap: Gap; verdict: string; explanation: string; pitfalls: string[]; seconds: number;
  recorded?: { seeds: number; gap_mean: number; ci_low: number; ci_high: number };
}

/** Small differences are the point of several entries, so they keep their digits. */
export function show(value: number): string {
  const size = Math.abs(value);
  if (size === 0) return '0';
  if (size < 1e-4) return value.toExponential(1);
  if (size < 0.01) return value.toFixed(5);
  return value.toFixed(4);
}

export const signed = (v: number) => `${v >= 0 ? '+' : '−'}${show(Math.abs(v))}`;

/** The difference as "how much better the careless version looks", whichever way the metric runs. */
export const flattery = (s: SkillRecord) => s.gap.mean * (s.higher_is_better ? 1 : -1);

const verdictColour = (s: SkillRecord) =>
  !s.gap.decisive ? 'var(--text-secondary)' : s.gap.flatters ? 'var(--status-critical)' : 'var(--delta-up)';

const card: React.CSSProperties = {
  background: 'var(--surface-1)', border: '1px solid var(--border)', borderRadius: 'var(--radius)',
  padding: 'var(--space-3)', textAlign: 'left', font: 'inherit', color: 'var(--text-primary)', cursor: 'pointer',
  display: 'grid', gap: 4,
};

export function CatalogueList({ skills, selected, onSelect }: {
  skills: SkillRecord[]; selected: string; onSelect: (id: string) => void;
}) {
  const categories = [...new Set(skills.map((s) => s.category))];
  return (
    <div style={{ display: 'grid', gap: 'var(--space-4)' }}>
      {categories.map((category) => (
        <section key={category}>
          <h3 style={{ margin: '0 0 var(--space-2)', fontSize: 'var(--text-sm)', color: 'var(--text-secondary)' }}>{category}</h3>
          <div style={{ display: 'grid', gap: 'var(--space-2)', gridTemplateColumns: 'repeat(auto-fill, minmax(290px, 1fr))' }}>
            {skills.filter((s) => s.category === category).map((s) => (
              <button key={s.id} type="button" onClick={() => onSelect(s.id)}
                      style={{ ...card, outline: selected === s.id ? '2px solid var(--series-1)' : 'none' }}>
                <span style={{ fontWeight: 600 }}>{s.name}</span>
                <span style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}>{s.dataset} · {s.metric}</span>
                <span style={{ fontSize: 'var(--text-sm)', color: verdictColour(s) }}>
                  {s.gap.decisive
                    ? `${signed(flattery(s))} ${s.gap.flatters ? 'in the careless version’s favour' : 'against the careless version'}`
                    : 'no measurable difference'}
                </span>
              </button>
            ))}
          </div>
        </section>
      ))}
    </div>
  );
}

export function SkillDetail({ skill }: { skill: SkillRecord }) {
  const [seeds, setSeeds] = useState(10);
  const [fresh, setFresh] = useState<SkillRecord | null>(null);
  const [busy, setBusy] = useState(false);
  useEffect(() => { setFresh(null); setSeeds(10); }, [skill.id]);

  const rerun = async () => {
    setBusy(true);
    try { setFresh(await apiPost<SkillRecord>(`/api/skills/${skill.id}/rerun?seeds=${seeds}`, {})); }
    finally { setBusy(false); }
  };

  const shown = fresh ?? skill;
  const bars = [
    { side: shown.careless.label, value: shown.careless.mean },
    { side: shown.careful.label, value: shown.careful.mean },
  ];
  return (
    <div style={{ display: 'grid', gap: 'var(--space-4)' }}>
      <p style={{ margin: 0, fontSize: 'var(--text-lg, 1.05rem)' }}>{skill.question}</p>
      <StatRow>
        <StatTile label={shown.careless.label} value={show(shown.careless.mean)} accent="var(--status-critical)"
                  hint={`${shown.metric} · sd ${show(shown.careless.sd)} across ${shown.seeds} seeds`} />
        <StatTile label={shown.careful.label} value={show(shown.careful.mean)} accent="var(--status-good)"
                  hint={`${shown.metric} · sd ${show(shown.careful.sd)} across ${shown.seeds} seeds`} />
        <StatTile label={shown.higher_is_better ? 'Paired difference' : 'Paired difference (lower is better)'}
                  value={signed(shown.gap.mean)}
                  hint={`95% interval ${signed(shown.gap.ci_low)} to ${signed(shown.gap.ci_high)}`} />
      </StatRow>
      <p style={{ margin: 0, color: verdictColour(shown), fontWeight: 600 }}>
        {shown.verdict}
        {shown.gap.negligible && shown.gap.decisive && ' — but the difference is far too small to matter in practice'}
      </p>
      <Chart kind="bar" horizontal height={170} data={bars} x="side"
             series={[{ key: 'value', label: shown.metric }]} yFormat={(v) => show(v)}
             caption={shown.metric_meaning} />
      <p style={{ margin: 0, lineHeight: 1.65 }}>{shown.explanation}</p>
      <div>
        <div style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 0.3 }}>How it goes wrong</div>
        <ul style={{ margin: '4px 0 0', paddingLeft: 20, lineHeight: 1.6 }}>
          {shown.pitfalls.map((p) => <li key={p}>{p}</li>)}
        </ul>
      </div>
      <div style={{ borderTop: '1px solid var(--gridline)', paddingTop: 'var(--space-3)', display: 'grid', gap: 'var(--space-2)' }}>
        <label style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: 0.3 }}>
          Run it again on {seeds} seeds
          <input type="range" min={2} max={40} value={seeds} onChange={(e) => setSeeds(Number(e.target.value))} style={{ width: '100%' }} />
        </label>
        <div style={{ display: 'flex', gap: 'var(--space-2)', alignItems: 'center', flexWrap: 'wrap' }}>
          <button type="button" onClick={() => void rerun()} disabled={busy}
                  style={{ padding: '9px 16px', font: 'inherit', fontWeight: 600, borderRadius: 'var(--radius)',
                    border: '1px solid var(--series-1)', background: 'var(--series-1)', color: '#fff', cursor: 'pointer', opacity: busy ? 0.6 : 1 }}>
            {busy ? 'Running…' : 'Run now'}
          </button>
          {fresh && fresh.recorded && (
            <span style={{ fontSize: 'var(--text-sm)', color: 'var(--text-secondary)' }}>
              {fresh.seeds} seeds just now: {signed(fresh.gap.mean)} [{signed(fresh.gap.ci_low)}, {signed(fresh.gap.ci_high)}] in {fresh.seconds}s ·
              {' '}recorded over {fresh.recorded.seeds}: {signed(fresh.recorded.gap_mean)} [{signed(fresh.recorded.ci_low)}, {signed(fresh.recorded.ci_high)}]
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
