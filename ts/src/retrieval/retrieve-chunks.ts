import { readIndex } from "../offline/5-validate-index.js";
import { embedQuery } from "./query-embedding.js";
import { topK } from "./ranking.js";
import type { SearchResult } from "../types/index.js";

/** Embed a question and return the highest-scoring indexed chunks. */
export async function retrieveChunks(question: string, k = 5): Promise<SearchResult[]> {
  const index = await readIndex();
  const query = await embedQuery(question);
  return topK(index.chunks, query, k);
}
