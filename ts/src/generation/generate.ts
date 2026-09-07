import type OpenAI from "openai";
import { completeJson } from "../llm/chat.js";
import { groundedAnswerResponseFormat } from "../llm/schemas.js";
import { models } from "../config/settings.js";
import type { Citation, GeneratedAnswer, SearchResult } from "../types/index.js";

export type { Citation, GeneratedAnswer } from "../types/index.js";

/** Squash case and punctuation, then tokenize for ordered-word matching. */
function words(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim()
    .split(" ")
    .filter(Boolean);
}

/**
 * True when every quote word appears inside chunkText in order (gaps allowed).
 * Chunking can interleave figure captions mid-paragraph, so a strict contiguous
 * substring rejects genuine verbatim quotes that merely span such an insertion;
 * ordered-word matching tolerates that while still requiring the quote's words
 * to exist in the real text.
 */
export function quoteInChunk(quote: string, chunkText: string): boolean {
  const hay = words(chunkText);
  const needle = words(quote);
  if (!needle.length) return false;
  let i = 0;
  for (const word of hay) {
    if (word === needle[i]) i += 1;
    if (i === needle.length) return true;
  }
  return false;
}

const GROUNDED_SYSTEM = `Answer the question using only the supplied research-paper passages.
If the passages do not contain enough evidence, say that plainly instead of guessing.
Return JSON with an answer string and citations. Every citation must have a chunk_id and a short verbatim quote.
Quotes are checked mechanically: copy them character-for-character from a single passage, do not paraphrase, do not shorten mid-sentence, and do not join text from separate sentences.`;
export function validateGeneratedAnswer(value: unknown, results: SearchResult[]): GeneratedAnswer {
  if (!value || typeof value !== "object") throw new Error("generation returned invalid JSON");
  const raw = value as { answer?: unknown; citations?: unknown };
  if (typeof raw.answer !== "string" || !raw.answer.trim()) {
    throw new Error("generation returned an empty answer");
  }
  if (!Array.isArray(raw.citations)) throw new Error("generation returned invalid citations");

  const chunks = new Map(results.map(({ chunk }) => [chunk.id, chunk]));
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
      chunk &&
      quote.length > 10 &&
      quote.length <= 300 &&
      !seen.has(key) &&
      quoteInChunk(quote, chunk.text)
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
  results: SearchResult[],
  api?: OpenAI,
): Promise<GeneratedAnswer> {
  if (results && !results.length) throw new Error("cannot generate an answer without search results");
  const passages = results
    .map(
      ({ chunk }) =>
        `[id: ${chunk.id}] ${chunk.title} - ${chunk.section} (p.${chunk.page})\n${chunk.text}`,
    )
    .join("\n\n");

  const value = await completeJson(
    {
      model: models.generation,
      messages: [
        { role: "system", content: GROUNDED_SYSTEM },
        { role: "user", content: `Passages:\n\n${passages}\n\nQuestion: ${question}` },
      ],
      response_format: groundedAnswerResponseFormat,
    },
    api,
  );
  return validateGeneratedAnswer(value, results);
}
