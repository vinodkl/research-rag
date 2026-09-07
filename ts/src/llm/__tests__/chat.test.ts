import assert from "node:assert/strict";
import test from "node:test";
import { completeJson } from "../chat.js";

const request = {
  model: "test-model",
  messages: [{ role: "user" as const, content: "hello" }],
  response_format: { type: "json_object" as const },
};

test("completeJson parses the shared chat response", async () => {
  const api = {
    chat: {
      completions: {
        create: async () => ({ choices: [{ message: { content: '{"ok":true}' } }] }),
      },
    },
  } as never;

  assert.deepEqual(await completeJson(request, api), { ok: true });
});
