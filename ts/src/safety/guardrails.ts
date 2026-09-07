import type { SearchResult } from "../search/retrieve.js";

const INJECTION = /(?:ignore|disregard|forget)\s+(?:the\s+|all\s+)?(?:previous|prior|above)\s+(?:instructions?|prompts?)|(?:reveal|show|print)\s+(?:your\s+)?(?:system|developer)\s+(?:prompt|instructions?)/i;
const UNSAFE = /\b(?:write|generate|produce|give me)\b.{0,50}\b(?:hate speech|racial slurs?|pornographic|nsfw)\b|\b(?:how (?:can|do) i|instructions? (?:for|to))\b.{0,50}\b(?:kill|harm|attack)\s+(?:a person|someone|people)\b/i;
const OFF_TOPIC = /\b(?:best|nearest|recommend)\s+(?:pizza|restaurant|cafe)\b|\b(?:weather|temperature)\s+(?:today|tomorrow)\b|\b(?:book|find)\s+(?:me\s+)?(?:a\s+)?(?:flight|hotel)\b/i;
const SENSITIVE = [
  { pattern: /(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?![\w.-])/g, kind: "EMAIL" },
  { pattern: /(?<!\w)(?:sk|rk|pk)-[A-Za-z0-9_-]{16,}(?!\w)/g, kind: "API_TOKEN" },
  { pattern: /(?<!\w)(?:\+\d{1,3}[ .-]?)?(?:\(?\d{3}\)?[ .-])\d{3}[ .-]\d{4}(?!\w)/g, kind: "PHONE" },
];

export function normalize(text: string): string {
  return text.normalize("NFKC").replace(/[\u200b\ufeff]/g, "").replace(/\s+/g, " ").trim();
}

export function checkQuestion(question: string): string | null {
  const text = normalize(question);
  if (text.length < 3 || text.length > 500) return "Please ask a question between 3 and 500 characters.";
  if (INJECTION.test(text)) return "That looks like an attempt to change my instructions, so I did not run it.";
  if (UNSAFE.test(text)) return "I cannot help with that request.";
  if (OFF_TOPIC.test(text)) return "I can only answer questions about the indexed machine-learning papers.";
  return null;
}

export function sanitizeQuestion(question: string): string {
  let text = normalize(question);
  for (const { pattern, kind } of SENSITIVE) {
    text = text.replace(pattern, `[${kind}]`);
  }
  return text;
}

export function checkRetrieval(results: SearchResult[], minimumScore = 0.1): string | null {
  if (!results.some(({ score }) => Number.isFinite(score) && score >= minimumScore)) {
    return "I could not find anything in the indexed papers about that.";
  }
  return null;
}
