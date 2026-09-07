/**
 * Figure captioning: describe a figure image with a vision model, so an
 * embedding model (which only sees text) can find it.
 *
 * Ported from the Python repo's src/research_rag/ingestion/parsing.py
 * `_caption`/`_load_cache`/`_save_cache`, using DeepSeek's V4 Flash vision
 * model instead of gpt-5.6-luna, via DeepSeek's OpenAI-compatible API.
 *
 * Note the exact model ID: `deepseek-v4-flash-vision-exp`, not
 * `deepseek-v4-flash` - the plain flash model returns a 400 ("This model
 * does not support image") on any image content; only the `-vision-exp`
 * variant accepts it. Confirmed with a live call before wiring this in.
 *
 * Captions are cached by image content hash in data/captions.json, so
 * re-parsing never pays the vision bill twice for the same figure.
 */

import { createHash } from "node:crypto";
import { readFile, writeFile, mkdir } from "node:fs/promises";
import path from "node:path";
import { deepseekClient } from "../../llm/client.js";
import { models } from "../../config/settings.js";

const CACHE_PATH = path.join(import.meta.dirname, "..", "..", "..", "data", "captions.json");
const CAPTION_MODEL = models.caption;

let cache: Record<string, string> | null = null;

async function loadCache(): Promise<Record<string, string>> {
  if (cache) return cache;
  let loaded: Record<string, string>;
  try {
    loaded = JSON.parse(await readFile(CACHE_PATH, "utf-8"));
  } catch {
    loaded = {}; // no cache file yet
  }
  cache = loaded;
  return cache;
}

async function saveCache(): Promise<void> {
  if (!cache) return;
  await mkdir(path.dirname(CACHE_PATH), { recursive: true });
  await writeFile(CACHE_PATH, JSON.stringify(cache, null, 0));
}

/** One sentence describing a figure, cached by content hash. */
export async function captionImage(png: Uint8Array): Promise<string> {
  const store = await loadCache();
  const key = createHash("md5").update(png).digest("hex");
  if (store[key]) return store[key];

  const dataUrl = `data:image/png;base64,${Buffer.from(png).toString("base64")}`;
  const completion = await deepseekClient().chat.completions.create({
    model: CAPTION_MODEL,
    messages: [
      {
        role: "user",
        content: [
          {
            type: "text",
            text: "Describe this figure from an ML paper in 1-2 sentences: what it shows and the key takeaway. No preamble.",
          },
          { type: "image_url", image_url: { url: dataUrl, detail: "low" } },
        ],
      },
    ],
    max_tokens: 400,
  });

console.log("caption", completion.choices[0]?.message.content);
  const caption = (completion.choices[0]?.message.content ?? "").trim();
  if (caption) {
    store[key] = caption; // an empty response is not worth caching or inserting
    await saveCache();
  }
  return caption;
}
