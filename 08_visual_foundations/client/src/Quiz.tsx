/**
 * Quiz -- the same server-checked quiz as project 10, over this course's lessons.
 *
 * Questions arrive without answers. Each choice is checked by the server,
 * whose explanation quotes figures from the training run, so the quiz teaches
 * what this data actually showed.
 */

import { useState } from 'react';
import { apiPost, useApi } from '@dsfs/ui';

interface Question { id: string; phase: string; question: string; options: string[] }
interface Verdict { correct: boolean; answer: number; explanation: string }

export function Quiz() {
  const questions = useApi<Question[]>('/api/quiz');
  const [verdicts, setVerdicts] = useState<Record<string, Verdict & { choice: number }>>({});

  const answer = async (q: Question, choice: number) => {
    if (verdicts[q.id]) return;
    const v = await apiPost<Verdict>(`/api/quiz/${q.id}`, { choice });
    setVerdicts((prev) => ({ ...prev, [q.id]: { ...v, choice } }));
  };

  if (!questions.data) return <p style={{ color: 'var(--text-muted)' }}>{questions.error ?? 'Loading quiz…'}</p>;
  const answered = Object.values(verdicts);
  const score = answered.filter((v) => v.correct).length;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-4)' }}>
      <p style={{ margin: 0, color: 'var(--text-secondary)', fontSize: 'var(--text-sm)' }}>
        {answered.length === 0
          ? `${questions.data.length} questions. Answers are checked by the server and explained with this run's numbers.`
          : `${score} of ${answered.length} correct so far.`}
      </p>
      {questions.data.map((q, n) => {
        const v = verdicts[q.id];
        return (
          <section key={q.id} style={{ background: 'var(--surface-1)', border: '1px solid var(--border)',
            borderRadius: 'var(--radius-lg)', padding: 'var(--space-4)' }}>
            <div style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 0.3 }}>
              Question {n + 1} · {q.phase}
            </div>
            <p style={{ margin: '4px 0 var(--space-3)', fontWeight: 600 }}>{q.question}</p>
            <div style={{ display: 'grid', gap: 'var(--space-2)', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))' }}>
              {q.options.map((o, i) => {
                const chosen = v?.choice === i;
                const right = v && i === v.answer;
                return (
                  <button key={o} type="button" onClick={() => void answer(q, i)} disabled={!!v}
                          style={{
                            textAlign: 'left', padding: '10px 12px', borderRadius: 'var(--radius)', font: 'inherit',
                            cursor: v ? 'default' : 'pointer',
                            border: `1px solid ${right ? 'var(--status-good)' : chosen ? 'var(--status-critical)' : 'var(--border)'}`,
                            background: right || chosen ? 'var(--surface-2)' : 'transparent',
                            color: 'var(--text-primary)',
                          }}>
                    {right ? '✓ ' : chosen ? '✕ ' : ''}{o}
                  </button>
                );
              })}
            </div>
            {v && (
              <p style={{ margin: 'var(--space-3) 0 0', fontSize: 'var(--text-sm)', color: 'var(--text-secondary)' }}>
                <strong style={{ color: v.correct ? 'var(--delta-up)' : 'var(--status-critical)' }}>
                  {v.correct ? 'Correct.' : 'Not quite.'}
                </strong>{' '}
                {v.explanation}
              </p>
            )}
          </section>
        );
      })}
    </div>
  );
}
