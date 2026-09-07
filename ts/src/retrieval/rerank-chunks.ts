import type OpenAI from "openai";
import { completeJson } from "../llm/chat.js";
import { chunkRankingResponseFormat } from "../llm/schemas.js";
import { models } from "../config/settings.js";
import type { RankingResponse, SearchResult } from "../types/index.js";

const SYSTEM = `Rank the supplied research-paper passages for the original question.
Return every chunk_id exactly once, strongest evidence first. Mark each as direct,
supporting, or irrelevant. Do not answer the question.`;


export function validateRanking(value: unknown, candidates: SearchResult[], limit: number): SearchResult[] {
  if (!value || typeof value !== "object") throw new Error("reranker returned invalid JSON");
  const ranking = (value as Partial<RankingResponse>).ranking;
  if (!Array.isArray(ranking)) throw new Error("reranker returned no ranking");
  const byId = new Map(candidates.map((result) => [result.chunk.id, result]));
  if (ranking.length !== byId.size || new Set(ranking.map((item) => item?.chunk_id)).size !== ranking.length) {
    throw new Error("reranker did not return every candidate exactly once");
  }
  const results: SearchResult[] = [];
  for (const item of ranking) {
    if (!item || !byId.has(item.chunk_id)) throw new Error("reranker returned an unknown chunk");
    if (item.relevance !== "irrelevant") results.push(byId.get(item.chunk_id)!);
  }
  return results.slice(0, limit);
}

export async function rerankChunks(
  question: string,
  candidates: SearchResult[],
  limit = 5,
  api?: OpenAI,
): Promise<SearchResult[]> {
  if (!candidates.length) return [];
  const passages = candidates
    .map(({ chunk }) => `[id: ${chunk.id}] ${chunk.title} - ${chunk.section}\n${chunk.text}`)
    .join("\n\n");
  const value = await completeJson(
    {
      model: models.rerank,
      messages: [
        { role: "system", content: SYSTEM },
        { role: "user", content: `Question: ${question}\n\nPassages:\n\n${passages}` },
      ],
      response_format: chunkRankingResponseFormat,
    },
    api,
  );
  return validateRanking(value, candidates, limit);
}
