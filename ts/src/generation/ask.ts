import path from "node:path";
import { fileURLToPath } from "node:url";
import { generateAnswer } from "./generate.js";
import { retrieveChunksWithQueryExpansion } from "../retrieval/retrieve-chunks-with-query-expansion.js";
import { rerankChunks } from "../retrieval/rerank-chunks.js";
import { checkQuestion, checkRetrieval, sanitizeQuestion } from "../safety/guardrails.js";

async function main() {
  const rawQuestion = process.argv.slice(2).join(" ").trim();
  if (!rawQuestion) throw new Error('usage: npm run ask -- "your question"');
  const refusal = checkQuestion(rawQuestion);
  if (refusal) {
    console.log(refusal);
    return;
  }
  const question = sanitizeQuestion(rawQuestion);

  let results = await retrieveChunksWithQueryExpansion(question, 10, 10);
  if (results) {
    const refusal = checkRetrieval(results);
    if (refusal) {
      console.log(refusal);
      return;
    }
  }
  if (results) {
    try {
      results = await rerankChunks(question, results, 5);
    } catch (err) {
      console.warn(`Reranking unavailable (${(err as Error).name}); using vector order`);
      results = results.slice(0, 5);
    }
  }
  let answer;
  try {
    answer = await generateAnswer(question, results);
  } catch (err) {
    console.error(err instanceof Error ? err.message : err);
    console.log(
      "\nI could not produce an answer with verifiable citations, so I refused rather than guess.",
    );
    return;
  }
  console.log(`\nAnswer (RAG)\n──────\n${answer.answer}`);
  if (answer.citations.length > 0) {
    console.log("\nSources\n───────");
    for (const citation of answer.citations) {
      console.log(`Citation: [${citation.chunk_id}] “${citation.quote}”`);
    }
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === path.resolve(fileURLToPath(import.meta.url))) {
  main().catch((err) => {
    console.error(err);
    process.exitCode = 1;
  });
}
