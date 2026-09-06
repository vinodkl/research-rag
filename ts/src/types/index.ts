/**
 * Shared shapes for the ingestion pipeline. Mirrors the Python repo's
 * `research_rag.models.Chunk` (src/research_rag/models.py) — same fields,
 * same purpose: this is the one struct that flows through every stage.
 */

export interface Paper {
  id: string;
  title: string;
  url: string;
}

export interface Chunk {
  id: string; // `${paper}:${index}`, e.g. "attention:4"
  paper: string;
  title: string;
  section: string;
  page: number;
  text: string;
}

/** A Chunk plus its embedding vector — the unit stored in index.json. */
export interface IndexedChunk extends Chunk {
  embedding: number[];
}
