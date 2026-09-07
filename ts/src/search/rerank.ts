import "dotenv/config";
import OpenAI from "openai";
import type { SearchResult } from "./retrieve.js";

const MODEL = process.env.OPENAI_RERANK_MODEL ?? "gpt-4o-mini";
const SYSTEM = `Rank the supplied research-paper passages for the original question.
Return every chunk_id exactly once, strongest evidence first. Mark each as direct,
supporting, or irrelevant. Do not answer the question.`;

interface RankingItem {
  chunk_id: string;
  relevance: "direct" | "supporting" | "irrelevant";
}

interface RankingResponse {
  ranking: RankingItem[];
}

const RESPONSE_FORMAT = {
  type: "json_schema" as const,
  json_schema: {
    name: "chunk_ranking",
    strict: true,
    schema: {
      type: "object",
      properties: {
        ranking: {
          type: "array",
          items: {
            type: "object",
            properties: {
              chunk_id: { type: "string" },
              relevance: { type: "string", enum: ["direct", "supporting", "irrelevant"] },
            },
            required: ["chunk_id", "relevance"],
            additionalProperties: false,
          },
        },
      },
      required: ["ranking"],
      additionalProperties: false,
    },
  },
};

let client: OpenAI | null = null;
function openaiClient(): OpenAI {
  if (client) return client;
  const key = process.env.OPENAI_API_KEY;
  if (!key) throw new Error("OPENAI_API_KEY is not set (copy ts/.env.example -> ts/.env)");
  client = new OpenAI({ apiKey: key });
  return client;
}

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

export async function rerank(
  question: string,
  candidates: SearchResult[],
  limit = 5,
  api?: OpenAI,
): Promise<SearchResult[]> {
  if (!candidates.length) return [];
  const passages = candidates
    .map(({ chunk }) => `[id: ${chunk.id}] ${chunk.title} - ${chunk.section}\n${chunk.text}`)
    .join("\n\n");
  const response = await (api ?? openaiClient()).chat.completions.create({
    model: MODEL,
    messages: [
      { role: "system", content: SYSTEM },
      { role: "user", content: `Question: ${question}\n\nPassages:\n\n${passages}` },
    ],
    response_format: RESPONSE_FORMAT,
  });
  const content = response.choices[0]?.message.content;
  if (!content) throw new Error("reranker returned no content");
  return validateRanking(JSON.parse(content), candidates, limit);
}
