/**
 * Step 2 - Fetch: download each paper's PDF to data/papers/{id}.pdf.
 *
 * Kept deliberately dumb: no retries, no concurrency limiting, no caching
 * headers. The one thing worth doing is skipping a paper that's already on
 * disk, so re-running this script after a partial failure is cheap.
 */

import { mkdir, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import path from "node:path";
import { PAPERS } from "../config/papers.js";

const OUT_DIR = path.join(import.meta.dirname, "..", "..", "data", "papers");

async function fetchPaper(id: string, url: string): Promise<void> {
  const dest = path.join(OUT_DIR, `${id}.pdf`);
  if (existsSync(dest)) {
    console.log(`skip  ${id} (already downloaded)`);
    return;
  }

  console.log(`fetch ${id} <- ${url}`);
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`${id}: HTTP ${response.status} fetching ${url}`);
  }
  const bytes = Buffer.from(await response.arrayBuffer());
  await writeFile(dest, bytes);
  console.log(`saved ${id} (${(bytes.length / 1024).toFixed(0)} KB)`);
}

async function main() {
  await mkdir(OUT_DIR, { recursive: true });
  for (const paper of PAPERS) {
    await fetchPaper(paper.id, paper.url);
  }
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
