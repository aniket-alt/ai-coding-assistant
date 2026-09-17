/**
 * The workspace's live task state.
 *
 * Loads every task once, then applies server-sent events. Each event carries a
 * global sequence number: if one is skipped (a dropped connection, a sleeping
 * laptop), the hook reloads everything rather than showing a stale board.
 * Updates send the version they were based on; a 409 means someone else
 * changed the task first, so the server's copy replaces ours and the user is
 * told, instead of either change disappearing silently.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { apiGet } from '@dsfs/ui';

export interface Task {
  id: number; title: string; notes: string; status: 'todo' | 'doing' | 'review' | 'done';
  priority: 1 | 2 | 3 | 4; important: boolean; urgent: boolean; tags: string[];
  due: string | null; estimate_min: number | null; spent_min: number;
  created_at: string; updated_at: string; completed_at: string | null; version: number;
}

interface Snapshot { tasks: Task[]; seq: number; today: string }
interface ChangeEvent { seq: number; type: 'created' | 'updated' | 'deleted'; origin: string | null; at: number; task: Task }

export const CLIENT_ID = `tab-${Math.random().toString(36).slice(2, 8)}`;

export function useTasks() {
  const [tasks, setTasks] = useState<Map<number, Task>>(new Map());
  const [today, setToday] = useState<string>('');
  const [live, setLive] = useState<'connecting' | 'live' | 'reconnecting'>('connecting');
  const [notice, setNotice] = useState<string | null>(null);
  const [lastLatency, setLastLatency] = useState<number | null>(null);
  const seq = useRef(0);

  const reload = useCallback(async () => {
    const snap = await apiGet<Snapshot>('/api/tasks');
    setTasks(new Map(snap.tasks.map((t) => [t.id, t])));
    setToday(snap.today);
    seq.current = snap.seq;
  }, []);

  useEffect(() => {
    void reload();
    const source = new EventSource('/api/events');
    source.addEventListener('hello', () => setLive('live'));
    source.onmessage = (msg) => {
      const ev = JSON.parse(msg.data) as ChangeEvent;
      if (ev.seq <= seq.current) return;
      if (ev.seq !== seq.current + 1) {
        void reload();  // a gap: something was missed, so start again from the server's state
        return;
      }
      seq.current = ev.seq;
      setLastLatency(Date.now() - ev.at);
      setTasks((prev) => {
        const next = new Map(prev);
        if (ev.type === 'deleted') next.delete(ev.task.id);
        else if ((prev.get(ev.task.id)?.version ?? 0) <= ev.task.version) next.set(ev.task.id, ev.task);
        return next;
      });
      if (ev.origin && ev.origin !== CLIENT_ID && ev.type !== 'created') {
        setNotice(`Another tab ${ev.type} “${ev.task.title}”.`);
      }
    };
    source.onerror = () => { setLive('reconnecting'); void reload(); };
    return () => source.close();
  }, [reload]);

  const send = useCallback(async (method: 'POST' | 'PATCH' | 'DELETE', path: string, body?: unknown) => {
    const res = await fetch(path, { method, headers: { 'Content-Type': 'application/json', 'X-Client-Id': CLIENT_ID },
                                    body: body ? JSON.stringify(body) : undefined });
    const data = await res.json().catch(() => ({}));
    return { status: res.status, data };
  }, []);

  const update = useCallback(async (task: Task, fields: Partial<Task>) => {
    const { status, data } = await send('PATCH', `/api/tasks/${task.id}`, { version: task.version, fields });
    if (status === 409) {
      const current = data.detail.current as Task;
      setTasks((prev) => new Map(prev).set(current.id, current));
      setNotice(`“${current.title}” was changed in another tab first, so your edit was not applied. It now shows the latest version.`);
      return false;
    }
    if (status === 200) setTasks((prev) => new Map(prev).set(data.id, data as Task));
    else setNotice(typeof data.detail === 'string' ? data.detail : 'The change was rejected.');
    return status === 200;
  }, [send]);

  const create = useCallback(async (text: string, extra: Partial<Task> = {}) => {
    const { status, data } = await send('POST', '/api/tasks', { text, ...extra });
    if (status !== 200) {
      setNotice(typeof data.detail === 'string' ? data.detail : 'The task could not be created.');
      return null;
    }
    setTasks((prev) => new Map(prev).set(data.id, data as Task));
    return data as Task;
  }, [send]);

  const remove = useCallback(async (task: Task) => {
    const { status } = await send('DELETE', `/api/tasks/${task.id}?version=${task.version}`);
    if (status === 409) setNotice(`“${task.title}” changed in another tab, so it was not deleted.`);
    else if (status === 200) setTasks((prev) => { const next = new Map(prev); next.delete(task.id); return next; });
  }, [send]);

  return { tasks: [...tasks.values()], today, live, notice, clearNotice: () => setNotice(null), lastLatency, update, create, remove, reload };
}
