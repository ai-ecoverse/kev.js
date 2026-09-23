import { readFileSync, existsSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { Tokenizer } from "@huggingface/tokenizers";
import type { DecisionRecord, QuestionMeta, Answer, SystemOneRequest } from "../src/api.ts";

export const root = fileURLToPath(new URL("..", import.meta.url));

export interface Fixture {
  name: string;
  request: SystemOneRequest;
  record: DecisionRecord;
  meta: QuestionMeta[];
  encoding: { ids: number[]; pos: number[]; decide_idx: number[]; opt_idx: number[][]; state_len: number };
  probs: number[][];
  answers: Record<string, Answer>;
}

/** KEV_MODEL=kev-4b selects fixtures/kev-4b.json and dist/models/kev-4b (default kev-0.8b) */
export const model = process.env.KEV_MODEL ?? "kev-0.8b";
export const fixtures: Fixture[] = JSON.parse(readFileSync(`${root}/fixtures/${model}.json`, "utf8")).fixtures;

export const modelDir = process.env.KEV_MODEL_DIR ?? `${root}/public/models/${model}`;
export const haveModel = existsSync(`${modelDir}/manifest.json`);

const HF = "https://huggingface.co/ai-ecoverse/kev.js/resolve/main";

/** fetch with retries: connecting to the HF CDN can time out transiently */
async function fetchText(url: string, attempts = 4): Promise<string> {
  for (let i = 1; ; i++) {
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
      return await res.text();
    } catch (e) {
      if (i >= attempts || String(e).includes("HTTP 4")) throw e;
      await new Promise((r) => setTimeout(r, 500 * i));
    }
  }
}

/** The tokenizer the bundle ships: the local bundle or build output when present, else the published copy on Hugging
 * Face (cached in the OS temp dir), so CI can run the encoding tests without weights. */
export async function tokenizer(): Promise<Tokenizer> {
  const { tokenizer: tok, tokenizer_config: cfg } = await tokenizerFiles();
  return new Tokenizer(JSON.parse(tok), JSON.parse(cfg));
}

/** tokenizer.json and tokenizer_config.json as text, from the same places as tokenizer(). */
export async function tokenizerFiles(): Promise<{ tokenizer: string; tokenizer_config: string }> {
  // follow the manifest: files live under a revision directory (r-<sha>/)
  const read = async (f: "tokenizer" | "tokenizer_config") => {
    const localManifest = `${modelDir}/manifest.json`;
    if (existsSync(localManifest)) return readFileSync(`${modelDir}/${JSON.parse(readFileSync(localManifest, "utf8")).files[f]}`, "utf8");
    const build = `${root}/export/build/${model}/tokenizer/${f}.json`;
    if (existsSync(build)) return readFileSync(build, "utf8");
    const manifest = JSON.parse(await fetchText(`${HF}/${model}/manifest.json`));
    const path = manifest.files[f] as string;
    const cached = join(tmpdir(), `kev-js-${model}-${path.replaceAll("/", "_")}`);   // path carries the revision
    if (!existsSync(cached)) writeFileSync(cached, await fetchText(`${HF}/${model}/${path}`));
    return readFileSync(cached, "utf8");
  };
  return { tokenizer: await read("tokenizer"), tokenizer_config: await read("tokenizer_config") };
}
