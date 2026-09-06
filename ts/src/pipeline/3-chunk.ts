/**
 * Step 4 - Chunk: split a paper at its section headings.
 *
 * Ported from the Python repo's src/research_rag/ingestion/chunking.py.
 * The chunk is the atomic unit of retrieval, so chunk boundaries decide what
 * an answer can see. Papers already carry the right boundaries: their
 * numbered section headings ("1 Introduction", "2 Method", "3.1 Setup" each
 * start a new chunk) - but nothing in the parsed text *says* "this line is a
 * heading" (see parse.ts). We have to guess from shape alone, with two
 * defences against false positives:
 *   - a blacklist: "Figure 3", "Table 2", "Equation 5" look like headings
 *     but are not,
 *   - monotonicity: real section numbers only ever increase through a paper
 *     (1, 2, 3, 3.1, 3.2, 4 ...), so "2 GPUs were used" mid-paper is rejected
 *     once we're already past section 5.
 *
 * Two size policies keep chunks retrieval-friendly:
 *   - a section longer than MAX_CHARS splits at paragraph breaks,
 *   - a section shorter than MIN_CHARS merges into the next one.
 */

import { readFile, writeFile, mkdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";
import type { Chunk } from "../types/index.js";
import { PAPERS } from "../config/papers.js";

// The notebook's pattern: a section number, then a Title-Case (or ALL-CAPS)
// phrase. The title charset includes ?!()/. because real headings use them
// ("3 AREN'T EXISTING SOLUTIONS GOOD ENOUGH?").
const HEADING = /^\s*(\d{1,2}(?:\.\d{1,2}){0,2})\.?\s+([A-Z][a-zA-Z][\w\-,'&:?!()/. ]{1,60})\s*$/;
// Some layouts (ICLR style) emit the number and the title on separate lines.
const NUMBER_ALONE = /^\s*(\d{1,2}(?:\.\d{1,2}){0,2})\.?\s*$/;
const TITLE_ALONE = /^\s*([A-Z][a-zA-Z][\w\-,'&:?!()/. ]{1,60})\s*$/;
// Things that look numbered but are not sections (also from the notebook).
const BLACKLIST =
  /^(Figure|Fig|Table|Tab|Equation|Eq|Section|Algorithm|Appendix|Theorem|Lemma|Input|Output|Step|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|Proceedings)\b/i;
// Unnumbered headings every paper has.
const KNOWN =
  /^\s*(Abstract|Introduction|Related Work|Conclusion|References|Acknowledgements|Appendix)\s*$/i;

const MAX_CHARS = 2500; // ~600 tokens: fits the embedding model comfortably
const MIN_CHARS = 200; // anything shorter merges forward

interface Section {
  section: string;
  page: number;
  text: string;
}

export function chunkPaper(paper: string, title: string, pages: string[]): Chunk[] {
  let sections = splitAtHeadings(pages);
  sections = mergeSmall(sections);

  const chunks: Chunk[] = [];
  for (const { section, page, text } of sections) {
    for (const part of splitLarge(text)) {
      chunks.push({
        id: `${paper}:${chunks.length}`,
        paper,
        title,
        section,
        page,
        text: part,
      });
    }
  }
  return chunks;
}

/** "3.1" -> [3, 1], so section numbers compare in document order. */
function parseNum(num: string): number[] {
  return num
    .split(".")
    .filter((part) => part !== "")
    .map(Number);
}

/** Lexicographic tuple comparison, same semantics as Python tuple `<`/`>`. */
function compareTuples(a: number[], b: number[]): number {
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) {
    if (a[i] !== b[i]) return a[i] - b[i];
  }
  return a.length - b.length;
}

/** Walk every line; an accepted heading starts a new section. */
function splitAtHeadings(pages: string[]): Section[] {
  const sections: Section[] = [];
  let section = "Front matter";
  let pageOfSection = 1;
  let lines: string[] = [];
  let prevNum: number[] = [0];

  // Monotonicity (the notebook's rule): real section numbers only move
  // forward, so "2 GPUs were used" inside section 5 is rejected. Jumps are
  // capped at two whole sections - enough to survive one missed heading
  // without letting table values like "70.0" register as section seventy.
  function accept(num: number[]): boolean {
    if (compareTuples(num, prevNum) > 0 && num[0] <= prevNum[0] + 2 && num[0] <= 30) {
      prevNum = num;
      return true;
    }
    return false;
  }

  pages.forEach((page, pageIndex) => {
    const pageNumber = pageIndex + 1;
    const pageLines = page.split("\n");
    let i = 0;
    while (i < pageLines.length) {
      const line = pageLines[i];
      let heading: string | null = null;

      const headingMatch = HEADING.exec(line);
      const numberAloneMatch = !headingMatch ? NUMBER_ALONE.exec(line) : null;

      if (headingMatch && !BLACKLIST.test(headingMatch[2])) {
        if (accept(parseNum(headingMatch[1]))) {
          heading = line.trim();
        }
      } else if (numberAloneMatch && i + 1 < pageLines.length) {
        // ICLR-style layout: "2" on one line, "PROBLEM STATEMENT" on the next.
        const titleMatch = TITLE_ALONE.exec(pageLines[i + 1]);
        if (
          titleMatch &&
          !BLACKLIST.test(titleMatch[1]) &&
          accept(parseNum(numberAloneMatch[1]))
        ) {
          heading = `${numberAloneMatch[1]} ${titleMatch[1].trim()}`;
          i += 1; // the title line is consumed by the heading
        }
      } else if (KNOWN.test(line)) {
        heading = line.trim();
      }

      if (heading) {
        sections.push({ section, page: pageOfSection, text: lines.join("\n") });
        section = heading;
        pageOfSection = pageNumber;
        lines = [];
      } else {
        lines.push(line);
      }
      i += 1;
    }
  });
  sections.push({ section, page: pageOfSection, text: lines.join("\n") });

  return sections
    .map(({ section, page, text }) => ({ section, page, text: text.trim() }))
    .filter((s) => s.text !== "");
}

/** A tiny section joins the one after it, keeping its own heading. */
function mergeSmall(sections: Section[]): Section[] {
  const merged: Section[] = [];
  let carry: Section | null = null;

  for (let { section, page, text } of sections) {
    if (carry) {
      section = carry.section;
      page = carry.page;
      text = carry.text + "\n\n" + text;
      carry = null;
    }
    if (text.length < MIN_CHARS) {
      carry = { section, page, text };
    } else {
      merged.push({ section, page, text });
    }
  }
  if (carry) merged.push(carry); // a tiny final section has nothing to join
  return merged;
}

/** Cut an oversized section at the paragraph break nearest the size limit. */
function splitLarge(text: string): string[] {
  const parts: string[] = [];
  while (text.length > MAX_CHARS) {
    let cut = text.lastIndexOf("\n\n", MAX_CHARS);
    if (cut < MAX_CHARS / 2) cut = MAX_CHARS; // no good break point: cut at the limit
    parts.push(text.slice(0, cut).trim());
    text = text.slice(cut).trim();
  }
  parts.push(text);
  return parts.filter((p) => p !== "");
}

const PARSED_DIR = path.join(import.meta.dirname, "..", "..", "data", "parsed");
const OUT_DIR = path.join(import.meta.dirname, "..", "..", "data");

async function main() {
  await mkdir(OUT_DIR, { recursive: true });
  const allChunks: Chunk[] = [];

  for (const paper of PAPERS) {
    const raw = await readFile(path.join(PARSED_DIR, `${paper.id}.json`), "utf-8");
    const pages: string[] = JSON.parse(raw);
    const chunks = chunkPaper(paper.id, paper.title, pages);
    allChunks.push(...chunks);
    console.log(`chunk ${paper.id}: ${pages.length} pages -> ${chunks.length} chunks`);
  }

  const dest = path.join(OUT_DIR, "chunks.json");
  await writeFile(dest, JSON.stringify(allChunks, null, 2));
  console.log(`saved ${allChunks.length} chunks -> ${path.relative(process.cwd(), dest)}`);
}

if (process.argv[1] && path.resolve(process.argv[1]) === path.resolve(fileURLToPath(import.meta.url))) {
  main().catch((err) => {
    console.error(err);
    process.exitCode = 1;
  });
}
