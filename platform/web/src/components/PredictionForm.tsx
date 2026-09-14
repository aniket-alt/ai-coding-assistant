/**
 * PredictionForm -- schema-driven inference input.
 *
 * The form owns the inputs, validation, debouncing, submission, and the
 * loading and error states. The project owns only how a result looks, via
 * `renderResult`. That split is what lets the same component serve a taxi
 * booking, a customer feature vector, a forecast horizon and an anomaly
 * sample without knowing anything about any of them.
 *
 * It renders exactly what the API returns and computes no figure of its own.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { apiGet, apiPost } from '../hooks/useApi';

export interface Option {
  value: string | number;
  label: string;
  group?: string;
}

export type FieldSpec =
  | {
      kind: 'select';
      name: string;
      label: string;
      /** Inline options, or an endpoint path that returns them. */
      options: Option[] | string;
      searchable?: boolean;
      help?: string;
      required?: boolean;
    }
  | {
      kind: 'multiselect';
      name: string;
      label: string;
      options: Option[] | string;
      max?: number;
      help?: string;
    }
  | {
      kind: 'number';
      name: string;
      label: string;
      min?: number;
      max?: number;
      step?: number;
      unit?: string;
      help?: string;
    }
  | {
      kind: 'slider';
      name: string;
      label: string;
      min: number;
      max: number;
      step?: number;
      unit?: string;
      help?: string;
    }
  | { kind: 'datetime'; name: string; label: string; help?: string }
  | { kind: 'toggle'; name: string; label: string; help?: string };

export interface PredictionFormProps {
  fields: FieldSpec[];
  endpoint?: string;
  initialValues?: Record<string, unknown>;
  /**
   * Controlled mode. Supply both to drive the inputs from outside -- needed
   * whenever another view edits the same values, such as a map that sets an
   * origin the select must also reflect.
   */
  values?: Record<string, unknown>;
  onValuesChange?: (v: Record<string, unknown>) => void;
  /** Re-predict as inputs change, debounced. */
  autoSubmit?: boolean;
  debounceMs?: number;
  submitLabel?: string;
  renderResult: (result: unknown, meta: { loading: boolean }) => React.ReactNode;
  onResult?: (r: unknown) => void;
  /** Rendered between the inputs and the result, e.g. a map. */
  children?: React.ReactNode;
}

/** Options declared as an endpoint are fetched once and shared by field name. */
function useRemoteOptions(fields: FieldSpec[]) {
  const [store, setStore] = useState<Record<string, Option[]>>({});

  const endpoints = useMemo(() => {
    const out: Record<string, string> = {};
    fields.forEach((f) => {
      if ((f.kind === 'select' || f.kind === 'multiselect') && typeof f.options === 'string') {
        out[f.name] = f.options;
      }
    });
    return out;
  }, [fields]);

  useEffect(() => {
    let cancelled = false;
    Object.entries(endpoints).forEach(([name, path]) => {
      apiGet<Record<string, unknown>[]>(path)
        .then((rows) => {
          if (cancelled) return;
          const opts: Option[] = rows.map((r) => ({
            value: (r.id ?? r.value ?? r.key) as string | number,
            label: String(r.label ?? r.name ?? r.id ?? r.value),
            group: r.group ? String(r.group) : r.borough ? String(r.borough) : undefined,
          }));
          setStore((s) => ({ ...s, [name]: opts }));
        })
        .catch(() => {
          if (!cancelled) setStore((s) => ({ ...s, [name]: [] }));
        });
    });
    return () => { cancelled = true; };
  }, [endpoints]);

  return store;
}

export function PredictionForm({
  fields,
  endpoint = '/api/predict',
  initialValues = {},
  values: controlled,
  onValuesChange,
  autoSubmit = false,
  debounceMs = 400,
  submitLabel = 'Predict',
  renderResult,
  onResult,
  children,
}: PredictionFormProps) {
  const remote = useRemoteOptions(fields);
  const [internal, setInternal] = useState<Record<string, unknown>>(initialValues);

  // Controlled when the caller supplies `values`; uncontrolled otherwise.
  const values = controlled ?? internal;
  const setValues = (next: Record<string, unknown>) => {
    if (controlled) onValuesChange?.(next);
    else setInternal(next);
  };
  const [result, setResult] = useState<unknown>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const optionsFor = (f: FieldSpec): Option[] => {
    if (f.kind !== 'select' && f.kind !== 'multiselect') return [];
    return typeof f.options === 'string' ? (remote[f.name] ?? []) : f.options;
  };

  const submit = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const r = await apiPost<unknown>(endpoint, values);
      setResult(r);
      onResult?.(r);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setResult(null);
    } finally {
      setLoading(false);
    }
  }, [endpoint, values, onResult]);

  useEffect(() => {
    if (!autoSubmit) return;
    const ready = fields.every((f) => {
      if (f.kind === 'select' && (f as { required?: boolean }).required !== false) {
        return values[f.name] !== undefined && values[f.name] !== '';
      }
      return true;
    });
    if (!ready) return;
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(() => { void submit(); }, debounceMs);
    return () => { if (timer.current) clearTimeout(timer.current); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [values, autoSubmit, debounceMs]);

  const set = (name: string, v: unknown) => setValues({ ...values, [name]: v });

  return (
    <form
      onSubmit={(e) => { e.preventDefault(); void submit(); }}
      style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-4)' }}
    >
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
        gap: 'var(--space-3)',
      }}>
        {fields.map((f) => (
          <Field
            key={f.name}
            spec={f}
            value={values[f.name]}
            options={optionsFor(f)}
            onChange={(v) => set(f.name, v)}
          />
        ))}
      </div>

      {!autoSubmit && (
        <button
          type="submit"
          disabled={loading}
          style={{
            alignSelf: 'flex-start', padding: '9px 18px',
            borderRadius: 'var(--radius)', border: 'none',
            background: 'var(--series-1)', color: '#fff',
            fontWeight: 700, fontSize: 'var(--text-sm)',
            cursor: loading ? 'wait' : 'pointer', opacity: loading ? 0.7 : 1,
          }}
        >
          {loading ? 'Predicting…' : submitLabel}
        </button>
      )}

      {children}

      {error && (
        <div role="alert" style={{
          padding: 'var(--space-3)', borderRadius: 'var(--radius)',
          borderLeft: '3px solid var(--status-critical)',
          background: 'var(--surface-2)', fontSize: 'var(--text-sm)',
          color: 'var(--text-secondary)',
        }}>
          <strong style={{ color: 'var(--status-critical)' }}>Prediction failed.</strong>{' '}
          {error}
        </div>
      )}

      {(result !== null || loading) && renderResult(result, { loading })}
    </form>
  );
}

function Field({
  spec,
  value,
  options,
  onChange,
}: {
  spec: FieldSpec;
  value: unknown;
  options: Option[];
  onChange: (v: unknown) => void;
}) {
  const id = `f-${spec.name}`;
  const label = (
    <label htmlFor={id} style={{
      display: 'block', fontSize: 'var(--text-xs)', fontWeight: 600,
      color: 'var(--text-secondary)', textTransform: 'uppercase',
      letterSpacing: 0.3, marginBottom: 4,
    }}>
      {spec.label}
    </label>
  );

  const help = spec.help ? (
    <p style={{
      margin: '4px 0 0', fontSize: 'var(--text-xs)', color: 'var(--text-muted)',
    }}>
      {spec.help}
    </p>
  ) : null;

  const grouped = useMemo(() => {
    const g: Record<string, Option[]> = {};
    options.forEach((o) => {
      const key = o.group ?? '';
      (g[key] ??= []).push(o);
    });
    return g;
  }, [options]);

  if (spec.kind === 'select') {
    return (
      <div>
        {label}
        <select
          id={id}
          value={String(value ?? '')}
          onChange={(e) => onChange(Number.isNaN(Number(e.target.value))
            ? e.target.value : Number(e.target.value))}
          style={input}
        >
          <option value="">Select…</option>
          {Object.entries(grouped).map(([g, opts]) =>
            g ? (
              <optgroup key={g} label={g}>
                {opts.map((o) => (
                  <option key={String(o.value)} value={String(o.value)}>{o.label}</option>
                ))}
              </optgroup>
            ) : (
              opts.map((o) => (
                <option key={String(o.value)} value={String(o.value)}>{o.label}</option>
              ))
            ),
          )}
        </select>
        {help}
      </div>
    );
  }

  if (spec.kind === 'slider') {
    const v = Number(value ?? spec.min);
    return (
      <div>
        {label}
        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
          <input
            id={id} type="range" min={spec.min} max={spec.max}
            step={spec.step ?? 1} value={v}
            onChange={(e) => onChange(Number(e.target.value))}
            style={{ flex: 1, accentColor: 'var(--series-1)' }}
          />
          <output className="tabular-nums" style={{
            minWidth: 52, textAlign: 'right', fontSize: 'var(--text-sm)',
            fontWeight: 600,
          }}>
            {v}{spec.unit ? ` ${spec.unit}` : ''}
          </output>
        </div>
        {help}
      </div>
    );
  }

  if (spec.kind === 'number') {
    return (
      <div>
        {label}
        <input
          id={id} type="number" min={spec.min} max={spec.max} step={spec.step ?? 1}
          value={value === undefined || value === null ? '' : String(value)}
          onChange={(e) => onChange(e.target.value === '' ? undefined : Number(e.target.value))}
          style={input}
        />
        {help}
      </div>
    );
  }

  if (spec.kind === 'datetime') {
    return (
      <div>
        {label}
        <input
          id={id} type="datetime-local"
          value={typeof value === 'string' ? value : ''}
          onChange={(e) => onChange(e.target.value)}
          style={input}
        />
        {help}
      </div>
    );
  }

  if (spec.kind === 'toggle') {
    return (
      <div>
        {label}
        <input
          id={id} type="checkbox" checked={Boolean(value)}
          onChange={(e) => onChange(e.target.checked)}
          style={{ accentColor: 'var(--series-1)', width: 18, height: 18 }}
        />
        {help}
      </div>
    );
  }

  // multiselect
  const selected = Array.isArray(value) ? (value as (string | number)[]) : [];
  return (
    <div>
      {label}
      <select
        id={id} multiple
        value={selected.map(String)}
        onChange={(e) => {
          const picked = [...e.target.selectedOptions].map((o) =>
            Number.isNaN(Number(o.value)) ? o.value : Number(o.value));
          onChange(spec.kind === 'multiselect' && spec.max
            ? picked.slice(0, spec.max) : picked);
        }}
        style={{ ...input, height: 120 }}
      >
        {options.map((o) => (
          <option key={String(o.value)} value={String(o.value)}>{o.label}</option>
        ))}
      </select>
      {help}
    </div>
  );
}

const input: React.CSSProperties = {
  width: '100%', padding: '8px 10px', borderRadius: 'var(--radius-sm)',
  border: '1px solid var(--border)', background: 'var(--surface-1)',
  color: 'var(--text-primary)', fontSize: 'var(--text-sm)',
  fontFamily: 'inherit',
};
