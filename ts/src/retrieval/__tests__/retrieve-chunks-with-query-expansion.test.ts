import assert from "node:assert/strict";
import test from "node:test";
import { fuseRankings } from "../retrieve-chunks-with-query-expansion.js";

const result = (id: string, score: number) => ({
  score,
  chunk: { id, paper: "demo", title: "Demo", section: "Intro", page: 1, text: id, embedding: [1, 0] },
});

test("fuseRankings combines query rankings without duplicate chunks", () => {
  const fused = fuseRankings(
    [[result("a", 0.9), result("b", 0.8)], [result("b", 0.8), result("c", 0.7)]],
    3,
  );
  assert.deepEqual(fused.map(({ chunk }) => chunk.id), ["b", "a", "c"]);
});
