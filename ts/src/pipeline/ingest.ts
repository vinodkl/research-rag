/** Run the TypeScript ingestion stages in order. */

import { execFileSync } from "node:child_process";
import path from "node:path";

const root = path.join(import.meta.dirname, "..", "..");
const tsx = path.join(root, "node_modules", ".bin", "tsx");
const stages = [
  "src/pipeline/1-fetch.ts",
  "src/pipeline/2-parse/index.ts",
  "src/pipeline/3-chunk.ts",
  "src/pipeline/4-embed.ts",
  "src/pipeline/5-validate-index.ts",
];

for (const stage of stages) {
  console.log(`\n==> ${stage}`);
  execFileSync(tsx, [path.join(root, stage)], { cwd: root, stdio: "inherit" });
}
console.log("\ningest complete");
