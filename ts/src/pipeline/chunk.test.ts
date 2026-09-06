import assert from "node:assert/strict";
import test from "node:test";
import { chunkPaper } from "./3-chunk.js";

test("chunkPaper starts chunks at numbered headings and keeps metadata", () => {
  const body = "A".repeat(220);
  const chunks = chunkPaper("demo", "Demo", [`1 Introduction\n${body}\n2 Method\n${body}`]);

  assert.equal(chunks.length, 2);
  assert.equal(chunks[0].section, "1 Introduction");
  assert.equal(chunks[0].page, 1);
  assert.equal(chunks[1].id, "demo:1");
});

test("chunkPaper splits oversized sections", () => {
  const chunks = chunkPaper("demo", "Demo", [`1 Introduction\n${"A".repeat(2600)}`]);
  assert.ok(chunks.length > 1);
  assert.ok(chunks.every((chunk) => chunk.text.length <= 2500));
});
