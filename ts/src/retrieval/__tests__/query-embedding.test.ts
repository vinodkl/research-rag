import assert from "node:assert/strict";
import test from "node:test";
import { embedQuery } from "../query-embedding.js";

test("embedQuery rejects an empty question before calling the provider", async () => {
  await assert.rejects(() => embedQuery("  "), /question must not be empty/);
});
