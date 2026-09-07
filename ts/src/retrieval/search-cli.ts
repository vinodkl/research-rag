import path from "node:path";
import { fileURLToPath } from "node:url";
import type { SearchResult } from "../types/index.js";
import { retrieveChunks } from "./retrieve-chunks.js";

export { retrieveChunks } from "./retrieve-chunks.js";

export function formatResults(question: string, results: SearchResult[]): string {
  if (results.length === 0) return `Search: ${question}\n\nNo matches found.`;
  const lines = [`Search: ${question}`, `Matches: ${results.length}`];
  for (const [i, { chunk, score }] of results.entries()) {
    lines.push(
      `\n${"─".repeat(72)}\n[${i + 1}] score ${score.toFixed(4)}\n` +
        `${chunk.title} | ${chunk.section} | page ${chunk.page}\n\n${chunk.text.trim()}`,
    );
  }
  return lines.join("\n");
}

async function main() {
  const question = process.argv.slice(2).join(" ").trim();
  if (!question) throw new Error('usage: npm run search -- "your question"');

  console.log(formatResults(question, await retrieveChunks(question)));
}

if (process.argv[1] && path.resolve(process.argv[1]) === path.resolve(fileURLToPath(import.meta.url))) {
  main().catch((err) => {
    console.error(err);
    process.exitCode = 1;
  });
}
