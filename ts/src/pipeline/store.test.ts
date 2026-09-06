import assert from "node:assert/strict";
import test from "node:test";
import { validateIndex } from "./5-store.js";

test("validateIndex accepts a consistent local index", () => {
  const index = validateIndex({ model: "demo", dimensions: 2, chunks: [{ embedding: [0, 1] }] });
  assert.equal(index.chunks.length, 1);
});

test("validateIndex rejects vectors with the wrong dimensions", () => {
  assert.throws(
    () => validateIndex({ model: "demo", dimensions: 2, chunks: [{ embedding: [1] }] }),
    /wrong dimensions/,
  );
});
