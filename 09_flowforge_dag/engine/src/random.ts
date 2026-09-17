/**
 * Seeded randomness and random graphs, for property tests and benchmarks.
 */

import type { IndexedGraph } from './graph.ts';
import { fromEdges } from './graph.ts';

/** Mulberry32: a small, fast 32-bit generator; the same seed always gives the same stream. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * A random DAG: nodes are shuffled, and edges only run from earlier to later in
 * the hidden shuffle, so the graph is acyclic without the algorithms being able
 * to read that from node numbering.
 */
export function randomDag(n: number, edgesPerNode: number, rand: () => number): IndexedGraph {
  const hidden = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [hidden[i], hidden[j]] = [hidden[j], hidden[i]];
  }
  const edges: [number, number][] = [];
  const seen = new Set<number>();
  const target = Math.floor(n * edgesPerNode);
  let guard = 0;
  while (edges.length < target && n > 1 && guard++ < target * 4) {
    let a = Math.floor(rand() * n);
    let b = Math.floor(rand() * n);
    if (a === b) continue;
    if (a > b) [a, b] = [b, a];
    const key = hidden[a] * n + hidden[b];
    if (seen.has(key)) continue;
    seen.add(key);
    edges.push([hidden[a], hidden[b]]);
  }
  return fromEdges(n, edges);
}

/** A random DAG with one back edge added along an existing path, which is guaranteed to create a cycle. */
export function randomCyclic(n: number, edgesPerNode: number, rand: () => number): IndexedGraph {
  const g = randomDag(n, edgesPerNode, rand);
  const edges: [number, number][] = [];
  for (let u = 0; u < g.n; u++) for (let k = g.offsets[u]; k < g.offsets[u + 1]; k++) edges.push([u, g.targets[k]]);
  if (edges.length === 0) edges.push([0, 1]);
  const [u, v] = edges[Math.floor(rand() * edges.length)];
  edges.push([v, u]);
  return fromEdges(n, edges);
}
