/**
 * FlowForge domain types.
 *
 * Identifiers are branded so a node id cannot be passed where a workflow id is
 * expected, even though both are strings at runtime. Node configurations are a
 * discriminated union on `kind`, and every switch over it ends in
 * `assertNever`, so adding a kind without handling it is a compile error.
 */

declare const brand: unique symbol;
export type Brand<T, Name extends string> = T & { readonly [brand]: Name };

export type WorkflowId = Brand<string, 'WorkflowId'>;
export type NodeId = Brand<string, 'NodeId'>;
export type RunId = Brand<string, 'RunId'>;

export const workflowId = (s: string): WorkflowId => s as WorkflowId;
export const nodeId = (s: string): NodeId => s as NodeId;
export const runId = (s: string): RunId => s as RunId;

export type NodeConfig =
  | { kind: 'source'; dataset: string; rows: number }
  | { kind: 'transform'; operation: 'impute' | 'scale' | 'encode' | 'join-features' }
  | { kind: 'train'; model: string }
  | { kind: 'evaluate'; metric: string }
  | { kind: 'condition'; metric: string; atLeast: number }
  | { kind: 'join'; mode: 'all' | 'any' }
  | { kind: 'action'; action: 'deploy' | 'alert' | 'report' };

export type NodeKind = NodeConfig['kind'];

export interface WorkflowNode {
  id: NodeId;
  title: string;
  config: NodeConfig;
  /** Simulated work, in milliseconds. */
  durationMs: number;
  /** Probability that one attempt fails, for exercising retries. */
  failureRate?: number;
  retries?: number;
}

export interface Edge {
  from: NodeId;
  to: NodeId;
  /** For edges leaving a condition node: taken when the condition is true or false. */
  branch?: 'true' | 'false';
}

export interface Workflow {
  id: WorkflowId;
  name: string;
  nodes: WorkflowNode[];
  edges: Edge[];
}

export type Result<T, E> = { ok: true; value: T } | { ok: false; error: E };
export const ok = <T>(value: T): Result<T, never> => ({ ok: true, value });
export const err = <E>(error: E): Result<never, E> => ({ ok: false, error });

export type GraphError =
  | { kind: 'cycle'; path: NodeId[] }
  | { kind: 'unknown-node'; edge: Edge; missing: NodeId }
  | { kind: 'duplicate-node'; id: NodeId }
  | { kind: 'self-loop'; id: NodeId };

export type NodeState = 'pending' | 'running' | 'succeeded' | 'failed' | 'skipped';

export type RunEvent =
  | { type: 'run-started'; run: RunId; at: number; order: NodeId[] }
  | { type: 'node-started'; run: RunId; at: number; node: NodeId; attempt: number; running: number }
  | { type: 'node-retrying'; run: RunId; at: number; node: NodeId; attempt: number; delayMs: number }
  | { type: 'node-succeeded'; run: RunId; at: number; node: NodeId; attempt: number; output?: number }
  | { type: 'node-failed'; run: RunId; at: number; node: NodeId; attempts: number }
  | { type: 'node-skipped'; run: RunId; at: number; node: NodeId; reason: string }
  | { type: 'run-finished'; run: RunId; at: number; succeeded: number; failed: number; skipped: number };

export function assertNever(x: never): never {
  throw new Error(`Unhandled case: ${JSON.stringify(x)}`);
}

/** A human description of any node, exhaustive over every kind. */
export function describe(config: NodeConfig): string {
  switch (config.kind) {
    case 'source': return `Load ${config.dataset} (${config.rows.toLocaleString('en-US')} rows)`;
    case 'transform': return `Transform: ${config.operation}`;
    case 'train': return `Train ${config.model}`;
    case 'evaluate': return `Evaluate ${config.metric}`;
    case 'condition': return `Continue if ${config.metric} ≥ ${config.atLeast}`;
    case 'join': return config.mode === 'all' ? 'Wait for every input' : 'Wait for any input';
    case 'action': return `Action: ${config.action}`;
    default: return assertNever(config);
  }
}
