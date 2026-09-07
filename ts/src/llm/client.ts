import "dotenv/config";
import OpenAI from "openai";

type ClientOptions = {
  envKey: string;
  missingKeyMessage: string;
  baseURL?: string;
};

const clients = new Map<string, OpenAI>();

function clientFor({ envKey, missingKeyMessage, baseURL }: ClientOptions): OpenAI {
  const cacheKey = `${envKey}:${baseURL ?? ""}`;
  const cached = clients.get(cacheKey);
  if (cached) return cached;

  const apiKey = process.env[envKey];
  if (!apiKey) throw new Error(missingKeyMessage);
  const client = new OpenAI({ apiKey, ...(baseURL ? { baseURL } : {}) });
  clients.set(cacheKey, client);
  return client;
}

export function openaiClient(): OpenAI {
  return clientFor({
    envKey: "OPENAI_API_KEY",
    missingKeyMessage: "OPENAI_API_KEY is not set (copy ts/.env.example -> ts/.env)",
  });
}

export function deepseekClient(): OpenAI {
  return clientFor({
    envKey: "DEEPSEEK_API_KEY",
    baseURL: "https://api.deepseek.com",
    missingKeyMessage: "DEEPSEEK_API_KEY is not set (copy ts/.env.example -> ts/.env)",
  });
}
