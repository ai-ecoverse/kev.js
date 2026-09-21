import { readFileSync, existsSync } from "node:fs";
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

export const fixtures: Fixture[] = JSON.parse(readFileSync(`${root}/fixtures/kev-0.8b.json`, "utf8")).fixtures;

export const modelDir = process.env.KEV_MODEL_DIR ?? `${root}/dist/models/kev-0.8b`;
export const haveModel = existsSync(`${modelDir}/manifest.json`);

export function tokenizer(): Tokenizer {
  // same tokenizer files the bundle ships (copied from the base model by kev_web_export.merge)
  const dir = haveModel ? modelDir : `${root}/export/build/kev-0.8b/tokenizer`;
  return new Tokenizer(JSON.parse(readFileSync(`${dir}/tokenizer.json`, "utf8")), JSON.parse(readFileSync(`${dir}/tokenizer_config.json`, "utf8")));
}
