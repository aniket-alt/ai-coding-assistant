/**
 * Compile-time guarantees, checked by `tsc`.
 *
 * Every line marked @ts-expect-error must fail to type-check. If a change to
 * the types ever lets one of them compile, tsc reports the directive itself as
 * unused and the type-check fails -- so these are tests, not comments.
 */

import type { NodeConfig, NodeId, Result, WorkflowId } from '../src/types.ts';
import { assertNever, nodeId, workflowId } from '../src/types.ts';

declare function removeNode(workflow: WorkflowId, node: NodeId): void;

const wf = workflowId('churn-pipeline');
const node = nodeId('train-model');

removeNode(wf, node);
// @ts-expect-error -- arguments swapped: a node id is not a workflow id
removeNode(node, wf);
// @ts-expect-error -- a raw string is not a node id until it is branded
removeNode(wf, 'train-model');

// @ts-expect-error -- a condition needs its threshold
export const incomplete: NodeConfig = { kind: 'condition', metric: 'AUC' };
// @ts-expect-error -- parameters belong to their kind: a source has no model
export const mixed: NodeConfig = { kind: 'source', dataset: 'bank', rows: 10, model: 'LightGBM' };

export function missingCase(config: NodeConfig): string {
  switch (config.kind) {
    case 'source': case 'transform': case 'train': case 'evaluate': case 'condition': case 'join':
      return config.kind;
    default:
      // @ts-expect-error -- 'action' is unhandled, so config is not narrowed to never
      return assertNever(config);
  }
}

export function readResult(r: Result<number, string>): number {
  // @ts-expect-error -- the value is only reachable after checking ok
  const unchecked: number = r.value;
  void unchecked;
  return r.ok ? r.value : r.error.length;
}
