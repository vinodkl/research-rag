import assert from "node:assert/strict";
import test from "node:test";
import { validateRanking } from "../rerank-chunks.js";

const make = (id: string, score: number) => ({
  score,
  chunk: { id, paper: "demo", title: "Demo", section: "Intro", page: 1, text: id, embedding: [1, 0] },
});

test("validateRanking follows model order and removes irrelevant chunks", () => {
  const results = validateRanking(
    { ranking: [
      { chunk_id: "b", relevance: "direct" },
      { chunk_id: "a", relevance: "irrelevant" },
    ] },
    [make("a", 0.8), make("b", 0.7)],
    5,
  );
  assert.deepEqual(results.map(({ chunk }) => chunk.id), ["b"]);
});

test("validateRanking rejects incomplete rankings", () => {
  assert.throws(
    () => validateRanking({ ranking: [{ chunk_id: "a", relevance: "direct" }] }, [make("a", 1), make("b", 0.5)], 5),
    /every candidate exactly once/,
  );
});
