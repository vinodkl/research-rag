import "dotenv/config";
import OpenAI from "openai";
import type { SearchResult } from "../search/retrieve.js";

const MODEL = process.env.OPENAI_GENERATION_MODEL ?? "gpt-4o-mini";
const GROUNDED_SYSTEM = `Answer the question using only the supplied research-paper passages.
If the passages do not contain enough evidence, say that plainly instead of guessing.
Return JSON with an answer string and a citations array. Each citation must contain a chunk_id and a short verbatim quote.`;
const PLAIN_SYSTEM = `Answer the user's question normally. Return JSON with an answer string and an empty citations array.`;
const RESPONSE_FORMAT = {
  type: "json_schema" as const,
  json_schema: {
    name: "grounded_answer",
    strict: true,
    schema: {
      type: "object",
      properties: {
        answer: { type: "string" },
        citations: {
          type: "array",
          items: {
            type: "object",
            properties: {
              chunk_id: { type: "string" },
              quote: { type: "string", minLength: 11, maxLength: 300 },
            },
            required: ["chunk_id", "quote"],
            additionalProperties: false,
          },
        },
      },
      required: ["answer", "citations"],
      additionalProperties: false,
    },
  },
};

export interface Citation {
  chunk_id: string;
  quote: string;
}

export interface GeneratedAnswer {
  answer: string;
  citations: Citation[];
}

let client: OpenAI | null = null;

function openaiClient(): OpenAI {
  if (client) return client;
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) throw new Error("OPENAI_API_KEY is not set (copy ts/.env.example -> ts/.env)");
  client = new OpenAI({ apiKey });
  return client;
}

export function validateGeneratedAnswer(value: unknown, results?: SearchResult[]): GeneratedAnswer {
  if (!value || typeof value !== "object") throw new Error("generation returned invalid JSON");
  const raw = value as { answer?: unknown; citations?: unknown };
  if (typeof raw.answer !== "string" || !raw.answer.trim()) {
    throw new Error("generation returned an empty answer");
  }
  if (!Array.isArray(raw.citations)) throw new Error("generation returned invalid citations");
  if (!results) return { answer: raw.answer, citations: [] };

  const chunks = new Map(results.map(({ chunk }) => [chunk.id, chunk]));
  console.log('response', value)
  const seen = new Set<string>();
  const citations: Citation[] = [];
  for (const item of raw.citations) {
    if (!item || typeof item !== "object") continue;
    const citation = item as { chunk_id?: unknown; quote?: unknown };
    if (typeof citation.chunk_id !== "string" || typeof citation.quote !== "string") continue;
    const quote = citation.quote.trim();
    const chunk = chunks.get(citation.chunk_id);
    const key = `${citation.chunk_id}:${quote.toLowerCase().replace(/\s+/g, " ")}`;
    if (
      chunk && quote.length > 10 && quote.length <= 300 &&
      !seen.has(key) &&
      chunk.text.toLowerCase().replace(/\s+/g, " ").includes(quote.toLowerCase().replace(/\s+/g, " "))
    ) {
      seen.add(key);
      citations.push({ chunk_id: citation.chunk_id, quote });
    }
  }
  if (!citations.length) {
    const received = raw.citations.length;
    const ids = raw.citations
      .filter((item): item is { chunk_id?: unknown } => !!item && typeof item === "object")
      .map((item) => String(item.chunk_id ?? "missing"))
      .join(", ");
    const quoteLengths = raw.citations
      .filter((item): item is { quote?: unknown } => !!item && typeof item === "object")
      .map((item) => String(item.quote ?? "").length)
      .join(", ");
    throw new Error(`generation returned no valid citations (received ${received}; ids: ${ids || "none"}; quote lengths: ${quoteLengths || "none"})`);
  }
  return { answer: raw.answer, citations };
}

export async function generateAnswer(
  question: string,
  results?: SearchResult[],
  api?: OpenAI,
): Promise<GeneratedAnswer> {
  if (results && !results.length) throw new Error("cannot generate an answer without search results");
  const activeApi = api ?? openaiClient();
  const passages = results
    ? results
        .map(
          ({ chunk }) =>
            `[id: ${chunk.id}] ${chunk.title} - ${chunk.section} (p.${chunk.page})\n${chunk.text}`,
        )
        .join("\n\n")
    : "";

  const response = await activeApi.chat.completions.create({
    model: MODEL,
    messages: [
      { role: "system", content: results ? GROUNDED_SYSTEM : PLAIN_SYSTEM },
      {
        role: "user",
        content: results
          ? `Passages:\n\n${passages}\n\nQuestion: ${question}`
          : `Question: ${question}`,
      },
    ],
    response_format: RESPONSE_FORMAT,
  });
  const content = response.choices[0]?.message.content;
  if (!content) throw new Error("generation returned no content");
  return validateGeneratedAnswer(JSON.parse(content), results);
}
