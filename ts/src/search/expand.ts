import "dotenv/config";
import OpenAI from "openai";

const MODEL = process.env.OPENAI_QUERY_MODEL ?? "gpt-4o-mini";
const SYSTEM = "Rewrite the user's question as one concise search query for machine-learning papers. Do not answer it.";

let client: OpenAI | null = null;
function openaiClient(): OpenAI {
  if (client) return client;
  const key = process.env.OPENAI_API_KEY;
  if (!key) throw new Error("OPENAI_API_KEY is not set (copy ts/.env.example -> ts/.env)");
  client = new OpenAI({ apiKey: key });
  return client;
}

const RESPONSE_FORMAT = {
  type: "json_schema" as const,
  json_schema: {
    name: "query_rewrite",
    strict: true,
    schema: {
      type: "object",
      properties: { rewrite: { type: "string", minLength: 3, maxLength: 500 } },
      required: ["rewrite"],
      additionalProperties: false,
    },
  },
};

export async function expandQuery(question: string, api?: OpenAI): Promise<string[]> {
  const response = await (api ?? openaiClient()).chat.completions.create({
    model: MODEL,
    messages: [
      { role: "system", content: SYSTEM },
      { role: "user", content: question },
    ],
    response_format: RESPONSE_FORMAT,
  });
  const content = response.choices[0]?.message.content;
  if (!content) return [question];
  const rewrite = (JSON.parse(content) as { rewrite?: unknown }).rewrite;

  console.log(`Expanded query:`, question, rewrite);
  return typeof rewrite === "string" && rewrite.trim() && rewrite.trim() !== question.trim()
    ? [question, rewrite.trim()]
    : [question];
}
