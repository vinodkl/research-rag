import type OpenAI from "openai";
import { completeJson } from "../llm/chat.js";
import { queryRewriteResponseFormat } from "../llm/schemas.js";
import { models } from "../config/settings.js";
import type { QueryRewriteResponse } from "../types/index.js";

const SYSTEM = "Rewrite the user's question as one concise search query for machine-learning papers. Do not answer it.";

export async function expandQuery(question: string, api?: OpenAI): Promise<string[]> {
  const value = await completeJson(
    {
      model: models.query,
      messages: [
        { role: "system", content: SYSTEM },
        { role: "user", content: question },
      ],
      response_format: queryRewriteResponseFormat,
    },
    api,
  );
  const rewrite = (value as QueryRewriteResponse).rewrite;

  return typeof rewrite === "string" && rewrite.trim() && rewrite.trim() !== question.trim()
    ? [question, rewrite.trim()]
    : [question];
}
