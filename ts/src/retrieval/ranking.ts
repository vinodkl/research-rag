import type { IndexedChunk, SearchResult } from "../types/index.js";

export type { SearchResult } from "../types/index.js";

export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) throw new Error("vectors must have matching dimensions");
  let dot = 0;
  let aNorm = 0;
  let bNorm = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    aNorm += a[i] ** 2;
    bNorm += b[i] ** 2;
  }
  if (aNorm === 0 || bNorm === 0) throw new Error("vectors must not be zero-length");
  return dot / (Math.sqrt(aNorm) * Math.sqrt(bNorm));
}

export function topK(chunks: IndexedChunk[], query: number[], k = 5): SearchResult[] {
  if (k < 1) throw new Error("k must be positive");
  return chunks
    .map((chunk) => ({ chunk, score: cosineSimilarity(query, chunk.embedding) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, k);
}
