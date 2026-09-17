/**
 * Workspace views -- project-00-specific. They group and arrange the tasks the
 * server holds; every change goes back through useTasks, which sends the
 * task's version so a conflicting edit is refused rather than lost.
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import { apiGet } from '@dsfs/ui';
import type { Task } from './useTasks';

type Update = (t: Task, f: Partial<Task>) => Promise<boolean>;

const PRIORITY = { 1: 'urgent', 2: 'high', 3: 'normal', 4: 'low' } as const;
const PRIORITY_COLOUR = { 1: 'var(--status-critical)', 2: 'var(--series-4)', 3: 'var(--series-1)', 4: 'var(--text-muted)' } as const;
const STATUSES: { id: Task['status']; label: string }[] = [
  { id: 'todo', label: 'To do' }, { id: 'doing', label: 'In progress' }, { id: 'review', label: 'In review' }, { id: 'done', label: 'Done' },
];

const card: React.CSSProperties = {
  background: 'var(--surface-1)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', padding: '10px 12px',
};
const chip = (colour: string): React.CSSProperties => ({
  fontSize: 11, padding: '1px 6px', borderRadius: 999, border: `1px solid ${colour}`, color: colour, whiteSpace: 'nowrap',
});
const inputStyle: React.CSSProperties = {
  width: '100%', boxSizing: 'border-box', padding: '10px 12px', font: 'inherit', color: 'var(--text-primary)',
  background: 'var(--surface-1)', border: '1px solid var(--border)', borderRadius: 'var(--radius)',
};

function dueLabel(due: string | null, today: string): { text: string; colour: string } | null {
  if (!due) return null;
  const days = Math.round((Date.parse(due) - Date.parse(today)) / 86400000);
  if (days < 0) return { text: `${-days}d overdue`, colour: 'var(--status-critical)' };
  if (days === 0) return { text: 'today', colour: 'var(--series-4)' };
  if (days === 1) return { text: 'tomorrow', colour: 'var(--series-1)' };
  return { text: due, colour: 'var(--text-muted)' };
}

export function TaskCard({ task, today, update, draggable = false }: { task: Task; today: string; update: Update; draggable?: boolean }) {
  const due = dueLabel(task.status === 'done' ? null : task.due, today);
  return (
    <div style={{ ...card, cursor: draggable ? 'grab' : 'default', opacity: task.status === 'done' ? 0.7 : 1 }}
         draggable={draggable} onDragStart={(e) => e.dataTransfer.setData('text/task', String(task.id))}>
      <div style={{ display: 'flex', gap: 8, alignItems: 'flex-start' }}>
        <input type="checkbox" aria-label={`Mark “${task.title}” done`} checked={task.status === 'done'}
               onChange={(e) => void update(task, { status: e.target.checked ? 'done' : 'todo' })} style={{ marginTop: 3 }} />
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontWeight: 500, textDecoration: task.status === 'done' ? 'line-through' : 'none' }}>{task.title}</div>
          <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', marginTop: 4 }}>
            <span style={chip(PRIORITY_COLOUR[task.priority])}>{PRIORITY[task.priority]}</span>
            {task.tags.map((t) => <span key={t} style={chip('var(--text-secondary)')}>#{t}</span>)}
            {due && <span style={chip(due.colour)}>{due.text}</span>}
            {task.estimate_min !== null && (
              <span style={chip('var(--text-muted)')}>{task.spent_min ? `${task.spent_min} / ` : ''}{task.estimate_min} min</span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

interface Preview { title: string; priority_name: string; tags: string[]; due: string | null; estimate_min: number | null; unparsed: string[] }

export function QuickEntry({ create }: { create: (text: string) => Promise<Task | null> }) {
  const [text, setText] = useState('Prepare churn model review !high #work @friday ~45m');
  const [preview, setPreview] = useState<Preview | null>(null);
  useEffect(() => {
    if (!text.trim()) { setPreview(null); return; }
    const t = setTimeout(() => { apiGet<Preview>(`/api/parse?text=${encodeURIComponent(text)}`).then(setPreview).catch(() => setPreview(null)); }, 150);
    return () => clearTimeout(t);
  }, [text]);
  return (
    <form onSubmit={(e) => { e.preventDefault(); if (preview?.title) void create(text).then(() => setText('')); }}
          style={{ display: 'grid', gap: 'var(--space-2)' }}>
      <input value={text} onChange={(e) => setText(e.target.value)} style={inputStyle} aria-label="Quick entry"
             placeholder="Add a task: !high #tag @tomorrow ~30m" />
      {preview && (
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', alignItems: 'center', fontSize: 'var(--text-sm)' }}>
          <span style={{ color: 'var(--text-muted)' }}>Will create:</span>
          <strong>{preview.title || '(no title)'}</strong>
          <span style={chip('var(--text-secondary)')}>{preview.priority_name}</span>
          {preview.tags.map((t) => <span key={t} style={chip('var(--text-secondary)')}>#{t}</span>)}
          {preview.due && <span style={chip('var(--series-1)')}>due {preview.due}</span>}
          {preview.estimate_min && <span style={chip('var(--text-muted)')}>{preview.estimate_min} min</span>}
          {preview.unparsed.length > 0 && <span style={{ color: 'var(--text-muted)' }}>kept as text: {preview.unparsed.join(' ')}</span>}
        </div>
      )}
    </form>
  );
}

export function ListView({ tasks, today, update }: { tasks: Task[]; today: string; update: Update }) {
  const groups = useMemo(() => {
    if (!today) return [];  // today's date arrives with the first load; nothing to group before it
    const open = tasks.filter((t) => t.status !== 'done').sort((a, b) => (a.due ?? '9999').localeCompare(b.due ?? '9999') || a.priority - b.priority);
    const in7 = new Date(Date.parse(today) + 7 * 86400000).toISOString().slice(0, 10);
    return [
      { label: 'Overdue', items: open.filter((t) => t.due && t.due < today) },
      { label: 'Today', items: open.filter((t) => t.due === today) },
      { label: 'Next seven days', items: open.filter((t) => t.due && t.due > today && t.due <= in7) },
      { label: 'Later', items: open.filter((t) => t.due && t.due > in7) },
      { label: 'No date', items: open.filter((t) => !t.due) },
    ].filter((g) => g.items.length);
  }, [tasks, today]);
  return (
    <div style={{ display: 'grid', gap: 'var(--space-4)' }}>
      {groups.map((g) => (
        <section key={g.label}>
          <h3 style={{ margin: '0 0 var(--space-2)', fontSize: 'var(--text-sm)', color: g.label === 'Overdue' ? 'var(--status-critical)' : 'var(--text-secondary)' }}>
            {g.label} · {g.items.length}
          </h3>
          <div style={{ display: 'grid', gap: 'var(--space-2)' }}>
            {g.items.map((t) => <TaskCard key={t.id} task={t} today={today} update={update} />)}
          </div>
        </section>
      ))}
    </div>
  );
}

function DropZone({ onDrop, children, label }: { onDrop: (id: number) => void; children: React.ReactNode; label: string }) {
  const [over, setOver] = useState(false);
  return (
    <div onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)}
         onDrop={(e) => { e.preventDefault(); setOver(false); onDrop(Number(e.dataTransfer.getData('text/task'))); }}
         aria-label={label}
         style={{ background: over ? 'var(--surface-2)' : 'transparent', borderRadius: 'var(--radius-lg)', padding: 'var(--space-2)',
                  border: '1px dashed var(--border)', minHeight: 120, display: 'grid', gap: 'var(--space-2)', alignContent: 'start' }}>
      {children}
    </div>
  );
}

export function Board({ tasks, today, update }: { tasks: Task[]; today: string; update: Update }) {
  const byId = new Map(tasks.map((t) => [t.id, t]));
  return (
    <div style={{ display: 'grid', gap: 'var(--space-3)', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))' }}>
      {STATUSES.map((s) => {
        const items = tasks.filter((t) => t.status === s.id).sort((a, b) => a.priority - b.priority);
        const shown = s.id === 'done' ? items.slice(-6) : items;
        return (
          <div key={s.id}>
            <h3 style={{ margin: '0 0 var(--space-2)', fontSize: 'var(--text-sm)' }}>{s.label} · {items.length}</h3>
            <DropZone label={`${s.label} column`} onDrop={(id) => { const t = byId.get(id); if (t && t.status !== s.id) void update(t, { status: s.id }); }}>
              {shown.map((t) => <TaskCard key={t.id} task={t} today={today} update={update} draggable />)}
              {s.id === 'done' && items.length > shown.length && (
                <div style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}>and {items.length - shown.length} more</div>
              )}
            </DropZone>
          </div>
        );
      })}
    </div>
  );
}

export function Matrix({ tasks, today, update }: { tasks: Task[]; today: string; update: Update }) {
  const open = tasks.filter((t) => t.status !== 'done');
  const byId = new Map(open.map((t) => [t.id, t]));
  const quadrants = [
    { title: 'Do first', hint: 'important and urgent', important: true, urgent: true },
    { title: 'Schedule', hint: 'important, not urgent', important: true, urgent: false },
    { title: 'Delegate', hint: 'urgent, not important', important: false, urgent: true },
    { title: 'Eliminate', hint: 'neither', important: false, urgent: false },
  ];
  return (
    <div style={{ display: 'grid', gap: 'var(--space-3)', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))' }}>
      {quadrants.map((q) => {
        const items = open.filter((t) => t.important === q.important && t.urgent === q.urgent);
        return (
          <div key={q.title}>
            <h3 style={{ margin: '0 0 var(--space-2)', fontSize: 'var(--text-sm)' }}>{q.title} · {items.length}
              <span style={{ fontWeight: 400, color: 'var(--text-muted)' }}> — {q.hint}</span></h3>
            <DropZone label={`${q.title} quadrant`} onDrop={(id) => { const t = byId.get(id); if (t) void update(t, { important: q.important, urgent: q.urgent }); }}>
              {items.map((t) => <TaskCard key={t.id} task={t} today={today} update={update} draggable />)}
            </DropZone>
          </div>
        );
      })}
    </div>
  );
}

export function Calendar({ tasks, today }: { tasks: Task[]; today: string }) {
  const [offset, setOffset] = useState(0);
  if (!today) return <p style={{ color: 'var(--text-muted)' }}>Loading…</p>;
  const base = new Date(`${today}T00:00:00`);
  const first = new Date(base.getFullYear(), base.getMonth() + offset, 1);
  const start = new Date(first);
  start.setDate(1 - ((first.getDay() + 6) % 7));
  const days = Array.from({ length: 42 }, (_, i) => new Date(start.getFullYear(), start.getMonth(), start.getDate() + i));
  const iso = (d: Date) => `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
  const byDay = new Map<string, Task[]>();
  tasks.filter((t) => t.due).forEach((t) => byDay.set(t.due!, [...(byDay.get(t.due!) ?? []), t]));
  const button: React.CSSProperties = { ...inputStyle, width: 'auto', cursor: 'pointer', padding: '6px 12px' };
  return (
    <div>
      <div style={{ display: 'flex', gap: 'var(--space-2)', alignItems: 'center', marginBottom: 'var(--space-3)' }}>
        <button type="button" style={button} onClick={() => setOffset(offset - 1)} aria-label="Previous month">‹</button>
        <strong>{first.toLocaleDateString('en-GB', { month: 'long', year: 'numeric' })}</strong>
        <button type="button" style={button} onClick={() => setOffset(offset + 1)} aria-label="Next month">›</button>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(7, minmax(0, 1fr))', gap: 4 }}>
        {['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'].map((d) => (
          <div key={d} style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)', padding: 4 }}>{d}</div>
        ))}
        {days.map((d) => {
          const key = iso(d);
          const items = byDay.get(key) ?? [];
          return (
            <div key={key} style={{ ...card, padding: 6, minHeight: 84, opacity: d.getMonth() === first.getMonth() ? 1 : 0.45,
              borderColor: key === today ? 'var(--series-1)' : 'var(--border)' }}>
              <div style={{ fontSize: 'var(--text-xs)', fontWeight: key === today ? 700 : 400 }}>{d.getDate()}</div>
              {items.slice(0, 3).map((t) => (
                <div key={t.id} title={t.title} style={{ fontSize: 11, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                  borderLeft: `3px solid ${PRIORITY_COLOUR[t.priority]}`, paddingLeft: 4, marginTop: 2,
                  textDecoration: t.status === 'done' ? 'line-through' : 'none' }}>{t.title}</div>
              ))}
              {items.length > 3 && <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>+{items.length - 3} more</div>}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function chime() {
  try {
    const ctx = new AudioContext();
    [660, 880].forEach((freq, i) => {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.frequency.value = freq;
      gain.gain.setValueAtTime(0.0001, ctx.currentTime + i * 0.25);
      gain.gain.exponentialRampToValueAtTime(0.2, ctx.currentTime + i * 0.25 + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + i * 0.25 + 0.6);
      osc.connect(gain).connect(ctx.destination);
      osc.start(ctx.currentTime + i * 0.25);
      osc.stop(ctx.currentTime + i * 0.25 + 0.7);
    });
  } catch { /* audio unavailable: the timer still records the session */ }
}

export function Focus({ tasks, onFinished }: { tasks: Task[]; onFinished: (taskId: number | null, minutes: number) => Promise<void> }) {
  const open = tasks.filter((t) => t.status !== 'done');
  const [taskId, setTaskId] = useState<number | null>(open[0]?.id ?? null);
  const [mode, setMode] = useState<'focus' | 'break'>('focus');
  const [length, setLength] = useState(25);
  const [left, setLeft] = useState(25 * 60);
  const [running, setRunning] = useState(false);
  const startedAt = useRef<number | null>(null);
  const [log, setLog] = useState<string[]>([]);

  useEffect(() => { if (!running) setLeft((mode === 'focus' ? length : 5) * 60); }, [mode, length, running]);
  useEffect(() => {
    if (!running) return;
    const tick = setInterval(() => {
      const elapsed = Math.floor((Date.now() - (startedAt.current ?? Date.now())) / 1000);
      const remaining = (mode === 'focus' ? length : 5) * 60 - elapsed;
      if (remaining <= 0) {
        setRunning(false);
        chime();
        if (mode === 'focus') {
          void onFinished(taskId, length);
          setLog((l) => [`${new Date().toLocaleTimeString()} · ${length} min on ${tasks.find((t) => t.id === taskId)?.title ?? 'no task'}`, ...l]);
          setMode('break');
        } else {
          setMode('focus');
        }
      } else {
        setLeft(remaining);
      }
    }, 250);
    return () => clearInterval(tick);
  }, [running, mode, length, taskId, onFinished, tasks]);

  const mm = String(Math.floor(left / 60)).padStart(2, '0');
  const ss = String(left % 60).padStart(2, '0');
  const button: React.CSSProperties = { padding: '9px 16px', font: 'inherit', fontWeight: 600, borderRadius: 'var(--radius)', cursor: 'pointer',
    border: '1px solid var(--series-1)', background: 'var(--series-1)', color: '#fff' };
  return (
    <div style={{ display: 'grid', gap: 'var(--space-4)', justifyItems: 'center', textAlign: 'center' }}>
      <div style={{ fontSize: 'var(--text-sm)', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 1 }}>{mode === 'focus' ? 'Focus' : 'Break'}</div>
      <div style={{ fontSize: 88, fontWeight: 700, fontVariantNumeric: 'tabular-nums', lineHeight: 1 }} aria-live="polite">{mm}:{ss}</div>
      <div style={{ display: 'flex', gap: 'var(--space-2)', flexWrap: 'wrap', justifyContent: 'center' }}>
        <select value={taskId ?? ''} onChange={(e) => setTaskId(e.target.value ? Number(e.target.value) : null)} style={{ ...inputStyle, width: 320 }} aria-label="Task to focus on">
          <option value="">No task</option>
          {open.map((t) => <option key={t.id} value={t.id}>{t.title}</option>)}
        </select>
        <select value={length} onChange={(e) => setLength(Number(e.target.value))} disabled={running} style={{ ...inputStyle, width: 120 }} aria-label="Focus length">
          {[1, 15, 25, 45, 50].map((m) => <option key={m} value={m}>{m} min</option>)}
        </select>
      </div>
      <div style={{ display: 'flex', gap: 'var(--space-2)' }}>
        <button type="button" style={button} onClick={() => { startedAt.current = Date.now() - ((mode === 'focus' ? length : 5) * 60 - left) * 1000; setRunning(!running); }}>
          {running ? 'Pause' : 'Start'}
        </button>
        <button type="button" style={{ ...button, background: 'transparent', color: 'var(--text-primary)', borderColor: 'var(--border)' }}
                onClick={() => { setRunning(false); setLeft((mode === 'focus' ? length : 5) * 60); }}>Reset</button>
      </div>
      <p style={{ margin: 0, fontSize: 'var(--text-sm)', color: 'var(--text-muted)', maxWidth: 560 }}>
        A finished focus session adds its minutes to the task on the server, so time spent shows on the task in every tab.
      </p>
      {log.map((l) => <div key={l} style={{ fontSize: 'var(--text-sm)' }}>{l}</div>)}
    </div>
  );
}
