/**
 * TypeScript mirrors of the `dsfs` Python dataclasses.
 *
 * These are the wire contract between every project's FastAPI backend and its
 * React client. Keep them in sync with:
 *   platform/dsfs/provenance.py, crispdm.py, metrics.py, leakage.py
 */

/* -- provenance.py --------------------------------------------------- */

export interface Provenance {
  slug: string;
  name: string;
  source_url: string;
  license: string;
  citation: string;
  retrieved_at: string;
  sha256: string;
  n_rows: number;
  n_cols: number;
  bytes_on_disk: number;
  /** True only for deliberately generated data. Surfaced prominently in the UI. */
  synthetic: boolean;
  notes: string;
  extra: Record<string, unknown>;
}

/* -- crispdm.py ------------------------------------------------------ */

export type PhaseName =
  | 'business_understanding'
  | 'data_understanding'
  | 'data_preparation'
  | 'modeling'
  | 'evaluation'
  | 'deployment';

export const PHASE_ORDER: readonly PhaseName[] = [
  'business_understanding',
  'data_understanding',
  'data_preparation',
  'modeling',
  'evaluation',
  'deployment',
] as const;

export interface CrispFinding {
  label: string;
  value: string;
  detail: string;
}

export interface CrispPhase {
  name: PhaseName;
  title: string;
  objective: string;
  /** Markdown; may contain $KaTeX$ math. Render with <MarkdownMath>. */
  narrative: string;
  findings: CrispFinding[];
  figures: string[];
  code_refs: string[];
  gate_passed: boolean;
  gate_notes: string;
}

export interface CrispDmReport {
  project_slug: string;
  project_title: string;
  business_question: string;
  phases: CrispPhase[];
}

/* -- metrics.py ------------------------------------------------------ */

export interface ModelScore {
  name: string;
  family: string;
  metrics: Record<string, number>;
  params: Record<string, unknown>;
  fit_seconds: number;
  predict_ms_per_1k: number;
  notes: string;
}

export interface SplitInfo {
  strategy: string;
  n_train: number;
  n_test: number;
  n_val: number;
  train_end: string;
  test_start: string;
  embargo: string;
  notes: string;
}

export interface RunRecord {
  project_slug: string;
  project_title: string;
  task: string;
  /** The metric the leaderboard sorts by. */
  primary_metric: string;
  higher_is_better: boolean;
  provenance: Provenance;
  split: SplitInfo;
  models: ModelScore[];
  baseline: Record<string, number>;
  audit: Record<string, unknown>;
  env: Record<string, string>;
  created_at: string;
  duration_seconds: number;
  best_model: string | null;
  best_score: number | null;
  lift_over_baseline: number | null;
  extra: Record<string, unknown>;
}

/* -- leakage.py ------------------------------------------------------ */

export type Severity = 'CRITICAL' | 'WARNING' | 'INFO';

export interface AuditFinding {
  severity: Severity;
  rule: string;
  message: string;
  file: string;
  line: number;
}

/* -- app.py ---------------------------------------------------------- */

export interface ProjectMeta {
  slug: string;
  number: number;
  title: string;
  tagline: string;
  domain: string;
  api_port: number;
  web_port: number;
  dataset: string;
  techniques: string[];
  dsfs_version: string;
}

/** Shape returned when an artifact has not been generated yet. */
export interface NotBuilt {
  available: false;
  detail: string;
}

export function isNotBuilt(v: unknown): v is NotBuilt {
  return typeof v === 'object' && v !== null && (v as NotBuilt).available === false;
}
