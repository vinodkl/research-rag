import path from "node:path";
import { fileURLToPath } from "node:url";
import { generateAnswer } from "./generate.js";
import { search } from "../search/search.js";
import { rerank } from "../search/rerank.js";

async function main() {
  const args = process.argv.slice(2);
  const plain = args.includes("--no-rag");
  const question = args.filter((arg) => arg !== "--no-rag").join(" ").trim();
  if (!question) throw new Error('usage: npm run ask -- [--no-rag] "your question"');

  let results = plain ? undefined : await search(question, 10);
  if (results) {
    try {
      results = await rerank(question, results, 5);
    } catch (err) {
      console.warn(`Reranking unavailable (${(err as Error).name}); using vector order`);
      results = results.slice(0, 5);
    }
  }
  const answer = await generateAnswer(question, results);
  console.log(`\nAnswer${plain ? " (plain LLM, RAG bypassed)" : " (RAG)"}\n──────\n${answer.answer}`);
  if (answer.citations.length > 0) {
    console.log("\nSources\n───────");
    for (const citation of answer.citations) {
      console.log(`[${citation.chunk_id}] “${citation.quote}”`);
    }
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === path.resolve(fileURLToPath(import.meta.url))) {
  main().catch((err) => {
    console.error(err);
    process.exitCode = 1;
  });
}
