import { embed } from "../pipeline/4-embed.js";

export async function embedQuery(question: string): Promise<number[]> {
  if (!question.trim()) throw new Error("question must not be empty");
  const [vector] = await embed([question]);
  return vector;
}
