import assert from "node:assert/strict";
import test from "node:test";
import { generateAnswer, validateGeneratedAnswer } from "./generate.js";

const result = {
  score: 0.9,
  chunk: {
    id: "demo:0",
    paper: "demo",
    title: "Demo",
    section: "1 Intro",
    page: 1,
    text: "Attention connects tokens in the sequence.",
    embedding: [1, 0],
  },
};

test("generateAnswer rejects an empty retrieval result", async () => {
  await assert.rejects(
    () => generateAnswer("What is attention?", []),
    /without search results/,
  );
});

test("validateGeneratedAnswer keeps quotes found in retrieved chunks", () => {
  const answer = validateGeneratedAnswer(
    {
      answer: "Attention connects tokens.",
      citations: [{ chunk_id: "demo:0", quote: "Attention connects tokens in the sequence." }],
    },
    [result],
  );
  assert.equal(answer.citations[0].chunk_id, "demo:0");
});

test("validateGeneratedAnswer rejects fabricated citations", () => {
  assert.throws(
    () => validateGeneratedAnswer({ answer: "Guess", citations: [{ chunk_id: "demo:0", quote: "Not in the paper" }] }, [result]),
    /no valid citations/,
  );
});
