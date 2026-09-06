import type { Paper } from "../types/index.js";

/**
 * The learning corpus: 3 papers, small enough to read every intermediate
 * output by hand. Same PDFs the Python repo's config/papers.yaml points at,
 * so you can compare results if curious.
 */
export const PAPERS: Paper[] = [
  {
    id: "attention",
    title: "Attention Is All You Need",
    url: "https://arxiv.org/pdf/1706.03762",
  },
  {
    id: "bert",
    title: "BERT: Pre-training of Deep Bidirectional Transformers",
    url: "https://arxiv.org/pdf/1810.04805",
  },
  {
    id: "roformer",
    title: "RoFormer: Enhanced Transformer with Rotary Position Embedding",
    url: "https://arxiv.org/pdf/2104.09864",
  },
];
