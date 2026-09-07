import "dotenv/config";

export const models = {
  embedding: process.env.OPENAI_EMBEDDING_MODEL ?? "text-embedding-3-large",
  caption: process.env.OPENAI_CAPTION_MODEL ?? "deepseek-v4-flash-vision-exp",
  query: process.env.OPENAI_QUERY_MODEL ?? "gpt-4o-mini",
  rerank: process.env.OPENAI_RERANK_MODEL ?? "gpt-4o-mini",
  generation: process.env.OPENAI_GENERATION_MODEL ?? "gpt-4o-mini",
} as const;
