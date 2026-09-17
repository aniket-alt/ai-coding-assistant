/**
 * Project 00 client -- the task workspace.
 *
 * Every view reads the same live task state (useTasks) and writes back through
 * it. Analytics come from /api/analytics and the verification results from
 * /api/evidence; the client computes neither.
 */

import { useCallback, useState } from 'react';
import {
  apiPost, AppShell, AuditPanel, Chart, CrispDmWalkthrough, ModelLeaderboard, Panel, ProvenanceBadge,
  StatRow, StatTile, useApi, useAudit, useCrispDm, useMeta, useMetrics, useProvenance,
} from '@dsfs/ui';
import type { Tab } from '@dsfs/ui';
import { useTasks } from './useTasks';
import { Board, Calendar, Focus, ListView, Matrix, QuickEntry } from './Views';

const TABS: Tab[] = [
  { id: 'list', label: 'List' },
  { id: 'board', label: 'Board' },
  { id: 'matrix', label: 'Matrix' },
  { id: 'calendar', label: 'Calendar' },
  { id: 'focus', label: 'Focus' },
  { id: 'analytics', label: 'Analytics' },
  { id: 'evidence', label: 'Evidence' },
  { id: 'crispdm', label: 'CRISP-DM' },
  { id: 'audit', label: 'Audit' },
];

interface Analytics {
  today: string; heatmap: { date: string; completed: number }[]; streak_days: number;
  completed_this_week: number; completed_last_week: number;
  by_status: { status: string; tasks: number }[]; by_priority: { priority: string; open: number }[];
  overdue: number; estimate_accuracy: { tasks: number; estimated_min: number; spent_min: number };
  focus_minutes_week: number; live_subscribers: number;
}
interface LiveRow { subscribers: number; changes: number; deliveries: number; subscribers_complete_in_order: number; p50_ms: number; p95_ms: number; p99_ms: number; max_ms: number; writes_per_second: number }
interface Evidence {
  parser: { cases: number; today: string; round_trips: { checked: number; failures: number };
            full: { exact_match_pct: number; failures: { text: string }[] }; baseline: { exact_match_pct: number } };
  live: LiveRow[];
  conflicts: { rounds: number; writers: number; accepted: number; refused_409: number; rounds_not_exactly_one_winner: number; final_version: number; expected_final_version: number };
  machine: string;
}

export function App() {
  const [tab, setTab] = useState('list');
  const meta = useMeta();
  const metrics = useMetrics();
  const crispdm = useCrispDm();
  const audit = useAudit();
  const provenance = useProvenance();
  const ws = useTasks();
  const analytics = useApi<Analytics>('/api/analytics', [tab, ws.tasks.length]);
  const evidence = useApi<Evidence>('/api/evidence');
  const run = metrics.data;
  const open = ws.tasks.filter((t) => t.status !== 'done').length;

  const onFocusDone = useCallback(async (taskId: number | null, minutes: number) => {
    await apiPost('/api/focus', { task_id: taskId, minutes });
  }, []);

  const leaderboard = run && (
    <ModelLeaderboard run={run} entityLabel="Parser" showTiming={false} columns={[
      { key: 'exact_match_pct', label: 'Exact parses', format: (v) => `${v.toFixed(1)}%` },
      { key: 'fields_correct_pct', label: 'Fields correct', format: (v) => `${v.toFixed(1)}%` },
      { key: 'due_pct', label: 'Due date', format: (v) => `${v.toFixed(1)}%` },
      { key: 'tags_pct', label: 'Tags', format: (v) => `${v.toFixed(1)}%` },
    ]} />
  );

  const status = (
    <div style={{ display: 'flex', gap: 'var(--space-4)', alignItems: 'center', flexWrap: 'wrap', fontSize: 'var(--text-sm)', color: 'var(--text-secondary)' }}>
      <span><span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: 4, marginRight: 6,
        background: ws.live === 'live' ? 'var(--status-good)' : 'var(--series-4)' }} />{ws.live === 'live' ? 'Live' : 'Connecting…'}</span>
      <span>{open} open · {ws.tasks.length - open} done</span>
      {ws.lastLatency !== null && <span>last change arrived {Math.max(0, ws.lastLatency)} ms after the server sent it</span>}
      <span style={{ color: 'var(--text-muted)' }}>Open this page in a second tab and edit a task to see both stay in sync.</span>
    </div>
  );

  const workspace = (title: string, body: React.ReactNode, description?: string) => (
    <Panel title={title} description={description}>
      <div style={{ display: 'grid', gap: 'var(--space-4)' }}>
        {status}
        {ws.notice && (
          <div role="status" style={{ padding: 'var(--space-3)', borderRadius: 'var(--radius)', border: '1px solid var(--series-4)',
            background: 'color-mix(in srgb, var(--series-4) 10%, var(--surface-1))', display: 'flex', justifyContent: 'space-between', gap: 8 }}>
            <span>{ws.notice}</span>
            <button type="button" onClick={ws.clearNotice} style={{ border: 'none', background: 'none', cursor: 'pointer', color: 'inherit' }} aria-label="Dismiss">✕</button>
          </div>
        )}
        <QuickEntry create={ws.create} />
        {body}
      </div>
    </Panel>
  );

  return (
    <AppShell meta={meta.data} provenance={provenance.data} tabs={TABS} active={tab}
              onTabChange={setTab} repoUrl="https://github.com/aniket-alt/ai-coding-assistant">
      {tab === 'list' && workspace('Tasks', <ListView tasks={ws.tasks} today={ws.today} update={ws.update} />,
        'Grouped by when they are due. Type in quick entry to add one; the preview shows how the server will read it.')}
      {tab === 'board' && workspace('Board', <Board tasks={ws.tasks} today={ws.today} update={ws.update} />, 'Drag a card to change its status.')}
      {tab === 'matrix' && workspace('Eisenhower matrix', <Matrix tasks={ws.tasks} today={ws.today} update={ws.update} />, 'Open tasks by importance and urgency. Drag a card to reclassify it.')}
      {tab === 'calendar' && workspace('Calendar', <Calendar tasks={ws.tasks} today={ws.today} />, 'Tasks on their due dates.')}
      {tab === 'focus' && (
        <Panel title="Focus timer" description="Pomodoro sessions, recorded against a task when they finish.">
          <Focus tasks={ws.tasks} onFinished={onFocusDone} />
        </Panel>
      )}

      {tab === 'analytics' && analytics.data && (
        <>
          <Panel title="Momentum" description="Computed by the server from every task's completion time.">
            <StatRow>
              <StatTile label="Streak" value={analytics.data.streak_days} unit="days" hint="consecutive days with a completed task" />
              <StatTile label="Completed this week" value={analytics.data.completed_this_week}
                        hint={`${analytics.data.completed_last_week} the week before`} />
              <StatTile label="Overdue" value={analytics.data.overdue} accent={analytics.data.overdue ? 'var(--status-critical)' : undefined} />
              <StatTile label="Focus this week" value={analytics.data.focus_minutes_week} unit="min" />
            </StatRow>
          </Panel>
          <Panel title="Completions per day" description="The last five weeks.">
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(7, 28px)', gap: 4 }} role="img"
                 aria-label="Heatmap of tasks completed per day">
              {analytics.data.heatmap.map((d) => (
                <div key={d.date} title={`${d.date}: ${d.completed} completed`}
                     style={{ width: 28, height: 28, borderRadius: 4, border: '1px solid var(--border)',
                       background: d.completed ? `color-mix(in srgb, var(--status-good) ${Math.min(100, 25 + d.completed * 25)}%, var(--surface-1))` : 'var(--surface-2)' }} />
              ))}
            </div>
            <p style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}>Oldest first, one square per day; darker squares had more completions. Hover for counts.</p>
          </Panel>
          <Panel title="Workload" description="Where tasks sit now.">
            <div style={{ display: 'grid', gap: 'var(--space-4)', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))' }}>
              <Chart kind="bar" height={220} data={analytics.data.by_status} x="status" series={[{ key: 'tasks', label: 'Tasks' }]} />
              <Chart kind="bar" height={220} data={analytics.data.by_priority} x="priority" series={[{ key: 'open', label: 'Open tasks' }]} />
            </div>
            <p style={{ fontSize: 'var(--text-sm)', color: 'var(--text-secondary)' }}>
              For {analytics.data.estimate_accuracy.tasks} finished tasks with both an estimate and tracked time,
              {' '}{analytics.data.estimate_accuracy.estimated_min} minutes were estimated and {analytics.data.estimate_accuracy.spent_min} spent.
            </p>
          </Panel>
        </>
      )}

      {tab === 'evidence' && evidence.data && (
        <>
          <Panel title="Does quick entry understand what was typed?"
                 description={`${evidence.data.parser.cases} labelled phrases written against the grammar, resolved as if today were ${evidence.data.parser.today}. The same author wrote the parser and the phrases, so generated round trips are the independent check.`}>
            {leaderboard}
            <StatRow>
              <StatTile label="Generated round trips" value={evidence.data.parser.round_trips.checked.toLocaleString()}
                        hint={`${evidence.data.parser.round_trips.failures} failed to recover every field`} />
            </StatRow>
          </Panel>
          <Panel title="Does every tab see every change?" description={`A real server and ${evidence.data.live[evidence.data.live.length - 1].changes} changes, on ${evidence.data.machine}. Latency is server send to client receipt on one machine, so network time is not included.`}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', minWidth: 640, borderCollapse: 'collapse', fontSize: 'var(--text-sm)' }}>
                <thead><tr style={{ borderBottom: '2px solid var(--axis)' }}>
                  {['Open tabs', 'Tabs with every change, in order', 'Deliveries', 'p50', 'p95', 'p99', 'Writes/s'].map((h, i) => (
                    <th key={h} style={{ ...th, textAlign: i ? 'right' : 'left' }}>{h}</th>))}
                </tr></thead>
                <tbody>
                  {evidence.data.live.map((r) => (
                    <tr key={r.subscribers} style={{ borderBottom: '1px solid var(--gridline)' }}>
                      <td style={td}>{r.subscribers}</td>
                      <td style={{ ...td, textAlign: 'right' }}>{r.subscribers_complete_in_order} of {r.subscribers}</td>
                      <td style={{ ...td, textAlign: 'right' }}>{r.deliveries.toLocaleString()}</td>
                      <td style={{ ...td, textAlign: 'right' }}>{r.p50_ms.toFixed(1)} ms</td>
                      <td style={{ ...td, textAlign: 'right' }}>{r.p95_ms.toFixed(1)} ms</td>
                      <td style={{ ...td, textAlign: 'right' }}>{r.p99_ms.toFixed(1)} ms</td>
                      <td style={{ ...td, textAlign: 'right' }}>{r.writes_per_second.toFixed(0)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>
          <Panel title="Can an edit be silently lost?" description="Simultaneous edits to one task, all based on the same version.">
            <StatRow>
              <StatTile label="Rounds" value={evidence.data.conflicts.rounds} hint={`${evidence.data.conflicts.writers} writers each`} />
              <StatTile label="Edits accepted" value={evidence.data.conflicts.accepted} />
              <StatTile label="Refused with 409" value={evidence.data.conflicts.refused_409} hint="each would otherwise have overwritten another" />
              <StatTile label="Final version" value={evidence.data.conflicts.final_version} hint={`expected ${evidence.data.conflicts.expected_final_version}`} />
            </StatRow>
          </Panel>
        </>
      )}

      {tab === 'crispdm' && crispdm.data && (
        <CrispDmWalkthrough report={crispdm.data} slots={{ evaluation: leaderboard || undefined }} />
      )}

      {tab === 'audit' && (
        <>
          <AuditPanel findings={audit.data ?? []}
                      runtimeAssertions={(run?.audit?.runtime_assertions as never) ?? []}
                      summary={{ critical: run?.audit?.critical as number | undefined }} />
          {provenance.data && <ProvenanceBadge provenance={provenance.data} />}
        </>
      )}
    </AppShell>
  );
}

const th: React.CSSProperties = {
  padding: '8px 10px', color: 'var(--text-muted)', fontWeight: 600, fontSize: 'var(--text-xs)',
  textTransform: 'uppercase', letterSpacing: 0.3, whiteSpace: 'nowrap',
};
const td: React.CSSProperties = { padding: '9px 10px', color: 'var(--text-primary)', whiteSpace: 'nowrap' };
