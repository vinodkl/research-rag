/** Shared domain and provider-boundary shapes used across the TypeScript app. */

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

export interface Index {
  model: string;
  dimensions: number;
  chunks: IndexedChunk[];
}

export interface SearchResult {
  chunk: IndexedChunk;
  score: number;
}

export interface Citation {
  chunk_id: string;
  quote: string;
}

export interface GeneratedAnswer {
  answer: string;
  citations: Citation[];
}

export interface QueryRewriteResponse {
  rewrite?: unknown;
}

export type Relevance = "direct" | "supporting" | "irrelevant";

export interface RankingItem {
  chunk_id: string;
  relevance: Relevance;
}

export interface RankingResponse {
  ranking: RankingItem[];
}
