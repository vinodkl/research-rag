/** Run the TypeScript ingestion stages in order. */

import { execFileSync } from "node:child_process";
import path from "node:path";

const root = path.join(import.meta.dirname, "..", "..");
const tsx = path.join(root, "node_modules", ".bin", "tsx");
const stages = [
  "src/offline/1-fetch.ts",
  "src/offline/2-parse/index.ts",
  "src/offline/3-chunk.ts",
  "src/offline/4-embed.ts",
  "src/offline/5-validate-index.ts",
];

for (const stage of stages) {
  console.log(`\n==> ${stage}`);
  execFileSync(tsx, [path.join(root, stage)], { cwd: root, stdio: "inherit" });
}
console.log("\ningest complete");
