export const groundedAnswerResponseFormat = {
  type: "json_schema" as const,
  json_schema: {
    name: "grounded_answer",
    strict: true,
    schema: {
      type: "object",
      properties: {
        answer: { type: "string" },
        citations: {
          type: "array",
          items: {
            type: "object",
            properties: {
              chunk_id: { type: "string" },
              quote: { type: "string", minLength: 11, maxLength: 300 },
            },
            required: ["chunk_id", "quote"],
            additionalProperties: false,
          },
        },
      },
      required: ["answer", "citations"],
      additionalProperties: false,
    },
  },
};

export const queryRewriteResponseFormat = {
  type: "json_schema" as const,
  json_schema: {
    name: "query_rewrite",
    strict: true,
    schema: {
      type: "object",
      properties: { rewrite: { type: "string", minLength: 3, maxLength: 500 } },
      required: ["rewrite"],
      additionalProperties: false,
    },
  },
};

export const chunkRankingResponseFormat = {
  type: "json_schema" as const,
  json_schema: {
    name: "chunk_ranking",
    strict: true,
    schema: {
      type: "object",
      properties: {
        ranking: {
          type: "array",
          items: {
            type: "object",
            properties: {
              chunk_id: { type: "string" },
              relevance: { type: "string", enum: ["direct", "supporting", "irrelevant"] },
            },
            required: ["chunk_id", "relevance"],
            additionalProperties: false,
          },
        },
      },
      required: ["ranking"],
      additionalProperties: false,
    },
  },
};
