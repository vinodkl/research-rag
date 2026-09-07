import { expandQuery } from "./query-expansion.js";
import { retrieveChunks } from "./retrieve-chunks.js";
import type { SearchResult } from "../types/index.js";

const RRF_K = 60;

/**
 * Merge ranked lists by giving higher positions more points.
 *
 * Example:
 *   query 1: A, B, C, D, E
 *   query 2: B, C, F, G, H
 *
 * A rank-1 result gets 1 / 61 points, rank 2 gets 1 / 62, and so on.
 * B receives points from both lists, so it rises above chunks found once:
 *   B = 1 / 62 + 1 / 61
 *   A = 1 / 61
 * After all lists are processed, chunks are sorted by their combined points
 * and the first `limit` unique chunks are returned. Fusion only decides order;
 * each returned SearchResult keeps its original vector score and chunk data.
 */
export function fuseRankings(rankings: SearchResult[][], limit: number): SearchResult[] {
  const combined = new Map<string, { result: SearchResult; fusion: number }>();
  for (const ranking of rankings) {
    ranking.forEach((result, rank) => {
      const current = combined.get(result.chunk.id) ?? { result, fusion: 0 };
      current.fusion += 1 / (RRF_K + rank + 1);
      current.result = current.result.score >= result.score ? current.result : result;
      combined.set(result.chunk.id, current);
    });
  }
  return [...combined.values()]
    .sort((a, b) => b.fusion - a.fusion || b.result.score - a.result.score)
    .slice(0, limit)
    .map(({ result }) => result);
}

export async function retrieveChunksWithQueryExpansion(
  question: string,
  limit = 10,
  perQueryLimit = 10,
): Promise<SearchResult[]> {
  let queries: string[];
  try {
    queries = await expandQuery(question);
  } catch {
    queries = [question];
  }
  const rankings = await Promise.all(queries.map((query) => retrieveChunks(query, perQueryLimit)));
  return fuseRankings(rankings, limit);
}
