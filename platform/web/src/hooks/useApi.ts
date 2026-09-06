/**
 * Typed data access for the platform endpoints every backend exposes.
 *
 * Each project's Vite config proxies /api to its own FastAPI port, so the
 * client never hard-codes a host.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import type {
  AuditFinding,
  CrispDmReport,
  ProjectMeta,
  Provenance,
  RunRecord,
} from '../types';
import { isNotBuilt } from '../types';

export interface AsyncState<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
  /** True when the endpoint responded "not generated yet" rather than failing. */
  notBuilt: boolean;
  reload: () => void;
}

export async function apiGet<T>(path: string, signal?: AbortSignal): Promise<T> {
  const res = await fetch(path, {
    signal,
    headers: { Accept: 'application/json' },
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = (await res.json()) as { detail?: string };
      if (body?.detail) detail = body.detail;
    } catch {
      /* response had no JSON body; keep the status line */
    }
    throw new Error(detail);
  }
  return (await res.json()) as T;
}

export async function apiPost<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const b = (await res.json()) as { detail?: string };
      if (b?.detail) detail = b.detail;
    } catch {
      /* no JSON body */
    }
    throw new Error(detail);
  }
  return (await res.json()) as T;
}

/** Fetch a JSON endpoint, tracking loading / error / not-yet-generated states. */
export function useApi<T>(path: string, deps: unknown[] = []): AsyncState<T> {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [notBuilt, setNotBuilt] = useState(false);
  const [nonce, setNonce] = useState(0);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    setLoading(true);
    setError(null);
    setNotBuilt(false);

    apiGet<T>(path, ctrl.signal)
      .then((payload) => {
        if (ctrl.signal.aborted) return;
        if (isNotBuilt(payload)) {
          setNotBuilt(true);
          setData(null);
        } else {
          setData(payload);
        }
      })
      .catch((e: unknown) => {
        if (ctrl.signal.aborted) return;
        setError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => {
        if (!ctrl.signal.aborted) setLoading(false);
      });

    return () => ctrl.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [path, nonce, ...deps]);

  const reload = useCallback(() => setNonce((n) => n + 1), []);
  return { data, loading, error, notBuilt, reload };
}

/* -- the standard platform endpoints -------------------------------- */

export const useMeta = () => useApi<ProjectMeta>('/api/meta');
export const useMetrics = () => useApi<RunRecord>('/api/metrics');
export const useCrispDm = () => useApi<CrispDmReport>('/api/crispdm');
export const useProvenance = () => useApi<Provenance>('/api/provenance');
export const useAudit = () => useApi<AuditFinding[]>('/api/audit');
