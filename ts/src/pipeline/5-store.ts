/** Step 5 - Store: validate the local embedding index.
 *
 * `index.json` is the local store for this learning pipeline. Search can read
 * it directly, so this stage only checks the model/vector contract and reports
 * what is available.
 */

import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import type { IndexedChunk } from "../types/index.js";

export interface Index {
  model: string;
  dimensions: number;
  chunks: IndexedChunk[];
}

export function validateIndex(value: unknown): Index {
  if (!value || typeof value !== "object") throw new Error("index.json must contain an object");
  const index = value as Partial<Index>;
  if (typeof index.model !== "string" || !index.model) throw new Error("index model is missing");
  if (!Number.isInteger(index.dimensions) || (index.dimensions ?? 0) < 0) {
    throw new Error("index dimensions must be a non-negative integer");
  }
  if (!Array.isArray(index.chunks)) throw new Error("index chunks are missing");
  for (const chunk of index.chunks) {
    if (!chunk || !Array.isArray(chunk.embedding) || chunk.embedding.length !== index.dimensions) {
      throw new Error("index contains a vector with the wrong dimensions");
    }
    if (!chunk.embedding.every((n) => Number.isFinite(n))) {
      throw new Error("index contains a non-finite vector value");
    }
  }
  return index as Index;
}

const INDEX_PATH = path.join(import.meta.dirname, "..", "..", "data", "index.json");

export async function readIndex(filePath = INDEX_PATH): Promise<Index> {
  return validateIndex(JSON.parse(await readFile(filePath, "utf8")));
}

async function main() {
  const index = await readIndex();
  console.log(`store: ${index.chunks.length} chunks (${index.model}, ${index.dimensions}-dim)`);
}

if (process.argv[1] && path.resolve(process.argv[1]) === path.resolve(fileURLToPath(import.meta.url))) {
  main().catch((err) => {
    console.error(err);
    process.exitCode = 1;
  });
}
