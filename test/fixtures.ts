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

/** The tokenizer the bundle ships: the local bundle or build output when present, else the published copy on Hugging
 * Face (cached in the OS temp dir), so CI can run the encoding tests without weights. */
export async function tokenizer(): Promise<Tokenizer> {
  const local = [modelDir, `${root}/export/build/${model}/tokenizer`].find((d) => existsSync(`${d}/tokenizer.json`));
  const read = async (f: string) => {
    if (local) return readFileSync(`${local}/${f}`, "utf8");
    const cached = join(tmpdir(), `kev-js-${model}-${f}`);
    if (!existsSync(cached)) {
      const res = await fetch(`${HF}/${model}/${f}`);
      if (!res.ok) throw new Error(`${model}/${f}: HTTP ${res.status}`);
      writeFileSync(cached, await res.text());
    }
    return readFileSync(cached, "utf8");
  };
  return new Tokenizer(JSON.parse(await read("tokenizer.json")), JSON.parse(await read("tokenizer_config.json")));
}
