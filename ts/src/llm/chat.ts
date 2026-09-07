import type OpenAI from "openai";
import type { ChatCompletionCreateParamsNonStreaming } from "openai/resources/chat/completions/completions.js";
import { openaiClient } from "./client.js";

type JsonChatRequest = Pick<
  ChatCompletionCreateParamsNonStreaming,
  "model" | "messages" | "response_format"
>;

/** Shared JSON chat transport. Callers still own prompts and response validation. */
export async function completeJson(request: JsonChatRequest, api?: OpenAI): Promise<unknown> {
  const response = await (api ?? openaiClient()).chat.completions.create(request);
  const content = response.choices[0]?.message.content;
  if (!content) throw new Error("model returned no content");
  return JSON.parse(content);
}
