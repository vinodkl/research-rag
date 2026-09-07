/**
 * Step 4 - Embed: each chunk's text becomes a vector.
 *
 * Ported from the Python repo's src/research_rag/retrieval/embeddings.py.
 *
 * An embedding model maps text to a fixed-length vector so that similar
 * meanings land near each other. "Near" has a precise definition here: cosine
 * similarity. OpenAI's embeddings come back already normalised to length 1, so
 * cosine similarity is just the dot product - which is what the next stage's
 * search will compute.
 *
 * The model is a contract: the SAME model must embed the chunks now and the
 * query at question time. Swap the model without re-embedding every chunk and
 * search silently returns nonsense, so the model id is written into the output
 * file (see `MODEL`) for the validation stage to check.
 *
 * Embeddings cost money per call, so - like captions.ts - this stage never
 * pays twice for text it has already embedded: a previous data/index.json is
 * reused for every chunk whose text is byte-for-byte unchanged.
 */

import { createHash } from "node:crypto";
import { readFile, writeFile, mkdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { openaiClient } from "../llm/client.js";
import { models } from "../config/settings.js";
import type { Chunk, IndexedChunk } from "../types/index.js";

export const MODEL = models.embedding;
const BATCH = 128; // texts per API request (matches the Python BATCH)

/** List of texts -> matrix of unit-length vectors, one row per text. */
export async function embed(texts: string[]): Promise<number[][]> {
  const api = openaiClient();
  const vectors: number[][] = [];
  for (let start = 0; start < texts.length; start += BATCH) {
    const batch = texts.slice(start, start + BATCH);
    const response = await api.embeddings.create({ model: MODEL, input: batch });
    // The API guarantees response.data is returned in input order.
    vectors.push(...response.data.map((item) => item.embedding));
  }
  return vectors;
}

const DATA_DIR = path.join(import.meta.dirname, "..", "..", "data");
const CHUNKS_PATH = path.join(DATA_DIR, "chunks.json");
const INDEX_PATH = path.join(DATA_DIR, "index.json");

/** What the validation stage reads: the model contract plus every embedded chunk. */
interface Index {
  model: string;
  dimensions: number;
  chunks: IndexedChunk[];
}

function textHash(text: string): string {
  return createHash("sha256").update(text).digest("hex");
}

/** Previous run's vectors, keyed by text hash - empty if there is no cache or
 * it was built with a different model. */
async function loadCache(): Promise<Map<string, number[]>> {
  try {
    const previous: Index = JSON.parse(await readFile(INDEX_PATH, "utf-8"));
    if (previous.model !== MODEL) return new Map();
    return new Map(previous.chunks.map((c) => [textHash(c.text), c.embedding]));
  } catch {
    return new Map(); // no index.json yet
  }
}

async function main() {
  const chunks: Chunk[] = JSON.parse(await readFile(CHUNKS_PATH, "utf-8"));
  const cache = await loadCache();

  const missing = chunks.filter((c) => !cache.has(textHash(c.text)));
  console.log(
    `${chunks.length} chunks: ${chunks.length - missing.length} cached, ` +
      `${missing.length} to embed (${MODEL})`,
  );

  if (missing.length > 0) {
    const fresh = await embed(missing.map((c) => c.text));
    missing.forEach((chunk, i) => cache.set(textHash(chunk.text), fresh[i]));
  }

  const indexed: IndexedChunk[] = chunks.map((chunk) => ({
    ...chunk,
    embedding: cache.get(textHash(chunk.text))!,
  }));

  const index: Index = {
    model: MODEL,
    dimensions: indexed[0]?.embedding.length ?? 0,
    chunks: indexed,
  };

  await mkdir(DATA_DIR, { recursive: true });
  await writeFile(INDEX_PATH, JSON.stringify(index, null, 2));
  console.log(
    `saved ${indexed.length} embedded chunks (${index.dimensions}-dim) -> ` +
      `${path.relative(process.cwd(), INDEX_PATH)}`,
  );
}

if (process.argv[1] && path.resolve(process.argv[1]) === path.resolve(fileURLToPath(import.meta.url))) {
  main().catch((err) => {
    console.error(err);
    process.exitCode = 1;
  });
}
