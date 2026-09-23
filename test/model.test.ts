// Full runtime (src/model.ts) on onnxruntime-node against the PyTorch reference probabilities.
// Needs the packaged bundle (dist/models/kev-0.8b); KEV_VARIANTS=fp32,q8 picks the variants (default: all in the manifest).
import "./no-float16.ts";
import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import * as ort from "onnxruntime-node";
import { Kev, PointerHead, type KevManifest, type OrtModule } from "../src/index.ts";
import { fixtures, haveModel, modelDir, tokenizer } from "./fixtures.ts";
import { bounds, Parity } from "./parity.ts";

const manifest: KevManifest | null = haveModel ? JSON.parse(readFileSync(`${modelDir}/manifest.json`, "utf8")) : null;
// default: the variants on disk (fetch-model downloads one); KEV_VARIANTS names them, and then a missing one fails
const variants = process.env.KEV_VARIANTS?.split(",")
  ?? Object.entries(manifest?.variants ?? {}).filter(([, v]) => existsSync(`${modelDir}/${v.model}`)).map(([name]) => name);

async function load(variant: string): Promise<Kev> {
  const v = manifest!.variants[variant];
  const session = await ort.InferenceSession.create(`${modelDir}/${v.model}`, Kev.sessionOptions(manifest!, variant, false));
  const head = PointerHead.fromSafetensors(readFileSync(`${modelDir}/${manifest!.files.head}`).buffer as ArrayBuffer);
  return new Kev({ ort: ort as unknown as OrtModule, session, head, tokenizer: await tokenizer(), manifest: manifest!, variant, options: { temperature: 1 } });
}

for (const variant of variants) {
  test(`${variant}: probabilities match the PyTorch reference`, { skip: !haveModel && "no bundle" }, async () => {
    const kev = await load(variant);
    const parity = new Parity();
    for (const f of fixtures) {
      const r = await kev.systemOne(f.request);
      const probs = await kev.probs(f.record);   // second call: exercises the state cache
      probs.forEach((p, k) => parity.add(`${f.name} q${k}`, f.probs[k], p));
      if (variant === "fp32") assert.deepEqual(Object.fromEntries(Object.entries(r.answers).map(([k, a]) => [k, a.type])), Object.fromEntries(Object.entries(f.answers).map(([k, a]) => [k, a.type])));
      assert.equal(r.usage.input_tokens, f.encoding.ids.length);
    }
    console.log(`${variant}: ${parity.summary()}`);
    assert.deepEqual(parity.violations(bounds(manifest!, variant)), []);
    await kev.release();
  });
}
