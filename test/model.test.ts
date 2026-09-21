// Full runtime (src/model.ts) on onnxruntime-node against the PyTorch reference probabilities.
// Needs the packaged bundle (dist/models/kev-0.8b); KEV_VARIANTS=fp32,q8 picks the variants (default: all in the manifest).
import "./no-float16.ts";
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import * as ort from "onnxruntime-node";
import { Kev, PointerHead, type KevManifest, type OrtModule } from "../src/index.ts";
import { fixtures, haveModel, modelDir, tokenizer } from "./fixtures.ts";

const manifest: KevManifest | null = haveModel ? JSON.parse(readFileSync(`${modelDir}/manifest.json`, "utf8")) : null;
const variants = process.env.KEV_VARIANTS?.split(",") ?? Object.keys(manifest?.variants ?? {});

async function load(variant: string): Promise<Kev> {
  const v = manifest!.variants[variant];
  const session = await ort.InferenceSession.create(`${modelDir}/${v.model}`, Kev.sessionOptions(manifest!, variant, false));
  const head = PointerHead.fromSafetensors(readFileSync(`${modelDir}/${manifest!.files.head}`).buffer as ArrayBuffer);
  return new Kev({ ort: ort as unknown as OrtModule, session, head, tokenizer: tokenizer(), manifest: manifest!, variant });
}

for (const variant of variants) {
  test(`${variant}: probabilities match the PyTorch reference`, { skip: !haveModel && "no bundle" }, async () => {
    const kev = await load(variant);
    const bound = variant === "fp32" ? 1e-4 : (manifest!.variants[variant].parity?.max_abs_dp ?? 0.1) + 1e-3;
    let worst = 0;
    for (const f of fixtures) {
      const r = await kev.systemOne(f.request);
      const probs = await kev.probs(f.record);   // second call: exercises the state cache
      probs.forEach((p, k) => p.forEach((x, j) => { worst = Math.max(worst, Math.abs(x - f.probs[k][j])); }));
      if (variant === "fp32") assert.deepEqual(Object.fromEntries(Object.entries(r.answers).map(([k, a]) => [k, a.type])), Object.fromEntries(Object.entries(f.answers).map(([k, a]) => [k, a.type])));
      assert.equal(r.usage.input_tokens, f.encoding.ids.length);
    }
    console.log(`${variant}: max |dp| ${worst.toExponential(2)} over ${fixtures.length} fixtures`);
    assert.ok(worst <= bound, `max |dp| ${worst} > ${bound}`);
    await kev.release();
  });
}
