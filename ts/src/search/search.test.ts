import assert from "node:assert/strict";
import test from "node:test";
import { formatResults } from "./search.js";

test("formatResults presents ranked matches with useful metadata", () => {
  const result = formatResults("What is attention?", [
    {
      score: 0.98765,
      chunk: {
        id: "demo:0",
        paper: "demo",
        title: "Demo paper",
        section: "1 Introduction",
        page: 2,
        text: "Attention connects tokens.",
        embedding: [1, 0],
      },
    },
  ]);

  assert.match(result, /Search: What is attention\?/);
  assert.match(result, /\[1\] score 0\.9877/);
  assert.match(result, /Demo paper \| 1 Introduction \| page 2/);
  assert.match(result, /Attention connects tokens\./);
});
