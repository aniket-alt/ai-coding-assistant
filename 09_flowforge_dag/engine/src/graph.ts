/**
 * Graph algorithms over a workflow.
 *
 * `kahn` is the engine's ordering. `dfsOrder` and `naiveOrder` exist to be
 * compared with it: the first is the other textbook linear-time method, the
 * second the quadratic approach people write when they do not know either.
 * All three work on dense integer indices for speed; `validate` maps a
 * workflow's ids onto them.
 */

import type { Edge, GraphError, NodeId, Result, Workflow } from './types.ts';
import { err, ok } from './types.ts';

export interface IndexedGraph {
  n: number;
  /** Compressed adjacency: successors of i are targets[offsets[i] .. offsets[i + 1]). */
  offsets: Int32Array;
  targets: Int32Array;
  ids: NodeId[];
}

export function fromEdges(n: number, edges: ReadonlyArray<readonly [number, number]>, ids?: NodeId[]): IndexedGraph {
  const offsets = new Int32Array(n + 1);
  for (const [u] of edges) offsets[u + 1]++;
  for (let i = 0; i < n; i++) offsets[i + 1] += offsets[i];
  const fill = offsets.slice(0, n);
  const targets = new Int32Array(edges.length);
  for (const [u, v] of edges) targets[fill[u]++] = v;
  return { n, offsets, targets, ids: ids ?? Array.from({ length: n }, (_, i) => `n${i}` as NodeId) };
}

export function validate(wf: Workflow): Result<IndexedGraph, GraphError> {
  const index = new Map<NodeId, number>();
  for (const node of wf.nodes) {
    if (index.has(node.id)) return err({ kind: 'duplicate-node', id: node.id });
    index.set(node.id, index.size);
  }
  const pairs: [number, number][] = [];
  for (const e of wf.edges) {
    if (e.from === e.to) return err({ kind: 'self-loop', id: e.from });
    const u = index.get(e.from);
    const v = index.get(e.to);
    if (u === undefined) return err({ kind: 'unknown-node', edge: e, missing: e.from });
    if (v === undefined) return err({ kind: 'unknown-node', edge: e, missing: e.to });
    pairs.push([u, v]);
  }
  const g = fromEdges(wf.nodes.length, pairs, wf.nodes.map((x) => x.id));
  const order = kahn(g);
  if (!order.ok) return err({ kind: 'cycle', path: order.error.map((i) => g.ids[i]) });
  return ok(g);
}

/**
 * Kahn's algorithm: repeatedly take a node with no remaining inputs. O(V + E).
 * Ties are broken by index, so the same graph always yields the same order.
 * On a cycle, returns one concrete cycle as a closed path of indices.
 */
export function kahn(g: IndexedGraph): Result<Int32Array, number[]> {
  const indegree = new Int32Array(g.n);
  for (let i = 0; i < g.targets.length; i++) indegree[g.targets[i]]++;
  const queue = new Int32Array(g.n);
  let head = 0;
  let tail = 0;
  for (let i = 0; i < g.n; i++) if (indegree[i] === 0) queue[tail++] = i;
  while (head < tail) {
    const u = queue[head++];
    for (let k = g.offsets[u]; k < g.offsets[u + 1]; k++) {
      const v = g.targets[k];
      if (--indegree[v] === 0) queue[tail++] = v;
    }
  }
  if (tail === g.n) return ok(queue);
  return err(findCycle(g, indegree));
}

/** Every node left with inputs lies on or downstream of a cycle; walk back from one until a node repeats. */
function findCycle(g: IndexedGraph, indegree: Int32Array): number[] {
  const parent = new Int32Array(g.n).fill(-1);
  for (let u = 0; u < g.n; u++) {
    if (indegree[u] === 0) continue;
    for (let k = g.offsets[u]; k < g.offsets[u + 1]; k++) {
      const v = g.targets[k];
      if (indegree[v] > 0 && parent[v] === -1) parent[v] = u;
    }
  }
  let start = 0;
  while (indegree[start] === 0) start++;
  const seenAt = new Map<number, number>();
  const walk: number[] = [];
  let cur = start;
  while (!seenAt.has(cur)) {
    seenAt.set(cur, walk.length);
    walk.push(cur);
    cur = parent[cur];
  }
  const cycle = walk.slice(seenAt.get(cur)).reverse();
  return [...cycle, cycle[0]];
}

/** Longest distance from any source: nodes on the same level have no dependency between them and can run in parallel. */
export function levels(g: IndexedGraph, order: Int32Array): Int32Array {
  const level = new Int32Array(g.n);
  for (const u of order) {
    for (let k = g.offsets[u]; k < g.offsets[u + 1]; k++) {
      const v = g.targets[k];
      if (level[u] + 1 > level[v]) level[v] = level[u] + 1;
    }
  }
  return level;
}

/** Reverse post-order of an iterative depth-first search. O(V + E). Returns null on a cycle. */
export function dfsOrder(g: IndexedGraph): Int32Array | null {
  const state = new Uint8Array(g.n); // 0 new, 1 on stack, 2 done
  const out = new Int32Array(g.n);
  let pos = g.n;
  const stack = new Int32Array(g.n);
  const edgeAt = new Int32Array(g.n);
  for (let s = 0; s < g.n; s++) {
    if (state[s]) continue;
    let top = 0;
    stack[0] = s;
    edgeAt[s] = g.offsets[s];
    state[s] = 1;
    while (top >= 0) {
      const u = stack[top];
      if (edgeAt[u] < g.offsets[u + 1]) {
        const v = g.targets[edgeAt[u]++];
        if (state[v] === 1) return null;
        if (state[v] === 0) {
          state[v] = 1;
          edgeAt[v] = g.offsets[v];
          stack[++top] = v;
        }
      } else {
        state[u] = 2;
        out[--pos] = u;
        top--;
      }
    }
  }
  return out;
}

/** The quadratic method: scan every remaining node, each round, for one with no unplaced parents. */
export function naiveOrder(g: IndexedGraph): Int32Array | null {
  const parents: number[][] = Array.from({ length: g.n }, () => []);
  for (let u = 0; u < g.n; u++) for (let k = g.offsets[u]; k < g.offsets[u + 1]; k++) parents[g.targets[k]].push(u);
  const placed = new Uint8Array(g.n);
  const out = new Int32Array(g.n);
  for (let round = 0; round < g.n; round++) {
    let found = -1;
    for (let v = 0; v < g.n && found < 0; v++) {
      if (!placed[v] && parents[v].every((p) => placed[p])) found = v;
    }
    if (found < 0) return null;
    placed[found] = 1;
    out[round] = found;
  }
  return out;
}

/** True when every edge goes from earlier to later in `order`, and `order` is a permutation. */
export function respectsEdges(g: IndexedGraph, order: ArrayLike<number>): boolean {
  if (order.length !== g.n) return false;
  const position = new Int32Array(g.n).fill(-1);
  for (let i = 0; i < order.length; i++) {
    if (position[order[i]] !== -1) return false;
    position[order[i]] = i;
  }
  for (let u = 0; u < g.n; u++) {
    for (let k = g.offsets[u]; k < g.offsets[u + 1]; k++) if (position[u] >= position[g.targets[k]]) return false;
  }
  return true;
}

export function edgeList(g: IndexedGraph): Edge[] {
  const out: Edge[] = [];
  for (let u = 0; u < g.n; u++) for (let k = g.offsets[u]; k < g.offsets[u + 1]; k++) out.push({ from: g.ids[u], to: g.ids[g.targets[k]] });
  return out;
}
