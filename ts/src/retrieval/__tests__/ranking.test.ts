import assert from "node:assert/strict";
import test from "node:test";
import { cosineSimilarity, topK } from "../ranking.js";

const chunk = (id: string, embedding: number[]) => ({
  id,
  paper: "demo",
  title: "Demo",
  section: "1 Intro",
  page: 1,
  text: id,
  embedding,
});

test("cosineSimilarity measures vector direction", () => {
  assert.equal(cosineSimilarity([1, 0], [1, 0]), 1);
  assert.equal(cosineSimilarity([1, 0], [0, 1]), 0);
  assert.equal(cosineSimilarity([1, 0], [-1, 0]), -1);
});

test("topK ranks chunks and limits results", () => {
  const results = topK(
    [chunk("far", [0, 1]), chunk("near", [2, 0]), chunk("middle", [1, 1])],
    [1, 0],
    2,
  );
  assert.deepEqual(results.map((result) => result.chunk.id), ["near", "middle"]);
});
