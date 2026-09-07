/**
 * Step 3 - Parse: PDF -> per-page text, figure captions appended.
 *
 * Ported from the Python repo's src/research_rag/ingestion/parsing.py, using
 * `mupdf` - the same MuPDF engine PyMuPDF wraps, via Artifex's official WASM
 * build - in place of the earlier pdf-parse/pdf.js version. MuPDF's own text
 * extraction handles reading order (including multi-column layouts) properly
 * instead of the Y-coordinate guess this file used to make by hand.
 *
 * Figures need one extra move: an embedding model can only index text, so
 * anything said by a figure is invisible to retrieval unless it is written
 * down. Almost always the authors already did that - "Figure 3: Training
 * loss versus steps..." sits in the text layer and gets indexed like any
 * other sentence. Only when a page has figure images but no such caption do
 * we ask a vision model for one, appended as "[Figure: ...]".
 */

import { readFile, writeFile, mkdir } from "node:fs/promises";
import { execFileSync } from "node:child_process";
import path from "node:path";
import * as mupdf from "mupdf";
import { PAPERS } from "../../config/papers.js";
import { captionImage } from "./captions.js";

const PAPERS_DIR = path.join(import.meta.dirname, "..", "..", "..", "data", "papers");
const OUT_DIR = path.join(import.meta.dirname, "..", "..", "..", "data", "parsed");

// An author-written caption already on the page: "Figure 3: ..." / "Fig. 2."
const HAS_CAPTION = /^(Figure|Fig\.?)\s*\d+\s*[:.]/im;
const MIN_PIXELS = 40_000; // ~200x200: below this it is an icon, not a figure

export async function parsePaper(id: string): Promise<string[]> {
  const filePath = path.join(PAPERS_DIR, `${id}.pdf`);
  const bytes = await readFile(filePath);
  const doc = mupdf.Document.openDocument(bytes, "application/pdf");

  const pages: string[] = [];
  const pageCount = doc.countPages();
  for (let i = 0; i < pageCount; i++) {
    try {
      const page = doc.loadPage(i);
      try {
        // "preserve-images" is required for onImageBlock to fire at all - the
        // default text extraction silently reports zero images on any PDF,
        // which is exactly why every paper in the corpus looked image-free.
        const stext = page.toStructuredText("preserve-images");
        try {
          let text = stext.asText();
          // Authors caption their own figures; vision is the fallback, not the rule.
          if (!HAS_CAPTION.test(text)) {
            for (const image of pageFigures(stext)) {
              try {
                const pixmap = image.toPixmap();
                try {
                  const caption = await captionImage(pixmap.asPNG());
                  if (caption) text += `\n[Figure: ${caption}]\n`;
                } finally {
                  pixmap.destroy();
                }
              } finally {
                image.destroy();
              }
            }
          }
          pages.push(text);
        } finally {
          stext.destroy();
        }
      } finally {
        page.destroy();
      }
    } catch (err) {
      // A rare mupdf WASM bug (confirmed here: a Type3 font glyph in
      // clip.pdf's page 42) traps mid-operation - and unlike a normal JS
      // exception, that trap leaves the WASM module's internal state
      // corrupted for the rest of THIS process: every page after the first
      // one to trap fails the same way, even pages that parsed cleanly
      // moments earlier. There is no in-process recovery from that (only a
      // fresh process, i.e. the next paper, gets a clean module again), so
      // salvage what we already have and stop rather than let corrupted
      // state produce garbage for the remaining pages.
      console.warn(
        `${id}: page ${i + 1} crashed while parsing (${(err as Error).message}) - ` +
          `salvaged ${pages.length}/${pageCount} pages, stopping here`,
      );
      pages.push(
        `[PARSE ERROR: mupdf failed on this page onward - ${(err as Error).message}]`,
      );
      break;
    }
  }
  // Best-effort only: after a trap above, doc.destroy() may itself throw
  // against the corrupted module. This worker process exits right after
  // writing its output either way, so a missed destroy() here costs nothing.
  try {
    doc.destroy();
  } catch {
    // ignored - see comment above
  }
  return pages;
}

/** Every figure-sized image on a page. `walk` is synchronous, so we collect
 * the images here and caption them (async) afterward, one at a time. */
function pageFigures(stext: mupdf.StructuredText): mupdf.Image[] {
  const images: mupdf.Image[] = [];
  stext.walk({
    onImageBlock: (_bbox, _transform, image) => {
console.log("image", image.getWidth(), image.getHeight());
      if (image.getWidth() * image.getHeight() >= MIN_PIXELS) {
        images.push(image);
      } else {
        image.destroy(); // too small to be a figure - free it now, not later
      }
    },
  });
  return images;
}

const TSX_BIN = path.join(import.meta.dirname, "..", "..", "..", "node_modules", ".bin", "tsx");

/** Worker mode: parse exactly one paper, write its JSON, then exit. */
async function parseOnePaperCli(id: string): Promise<void> {
  await mkdir(OUT_DIR, { recursive: true });
  const pages = await parsePaper(id);
  const dest = path.join(OUT_DIR, `${id}.json`);
  await writeFile(dest, JSON.stringify(pages, null, 2));
  console.log(`saved ${id}: ${pages.length} pages -> ${path.relative(process.cwd(), dest)}`);
}

async function main() {
  const paperArgIndex = process.argv.indexOf("--paper");
  if (paperArgIndex !== -1) {
    await parseOnePaperCli(process.argv[paperArgIndex + 1]);
    return;
  }

  // Orchestrator mode: one child process per paper, not one big loop in this
  // process. mupdf's WASM heap only ever grows, and some of what it holds
  // (font/glyph/colorspace caches) lives at the library-context level, not
  // the Document level, so per-object .destroy() calls above cannot reach
  // it. A single long-lived process eventually runs out of WASM memory no
  // matter how disciplined the cleanup is - confirmed by two different
  // "memory access out of bounds" crashes, at different documents, after
  // adding those destroy() calls. A fresh process per paper gets a fresh
  // heap the OS reclaims unconditionally on exit, and keeps one unusually
  // memory-hungry PDF from taking down the whole batch.
  await mkdir(OUT_DIR, { recursive: true });
  const failed: string[] = [];
  for (const paper of PAPERS) {
    console.log(`parse ${paper.id}`);
    try {
      execFileSync(TSX_BIN, [import.meta.filename, "--paper", paper.id], { stdio: "inherit" });
    } catch {
      console.error(`FAILED: ${paper.id}`);
      failed.push(paper.id);
    }
  }
  if (failed.length > 0) {
    console.log(`\n${failed.length} paper(s) failed to parse: ${failed.join(", ")}`);
    process.exitCode = 1;
  }
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
