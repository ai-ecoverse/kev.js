// Image requests against the PyTorch reference (fixtures/<model>.json and the eval images, from
// `export/kev_web_export/vision_fixtures.py`; KEV_VISION_MODEL, default kev-4b-vision): Qwen's image preprocessing, the
// mRoPE positions and request validation always; with a vision bundle in public/models/<model> (KEV_VISION_DIR; not
// in git), the probabilities through the
// vision graph and the spliced decoder on onnxruntime-node. Node has no JPEG decoder here, so the two JPEG photos are
// left to the browser tests.
import "./no-float16.ts";
import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import * as ort from "onnxruntime-node";
import { Kev, PointerHead, ValidationError, preprocess, ropePositions, smartResize, validate, visionInputs, imageKey,
  type KevManifest, type OrtModule, type SystemOneRequest, type VisionConfig } from "../src/index.ts";
import { Tokenizer } from "@huggingface/tokenizers";
import { root } from "./fixtures.ts";
import { MEAN_ABS_DP, Parity } from "./parity.ts";
import { decodePng } from "./png.ts";

interface VisionFixture {
  set: string; id: string; image: string; size: [number, number]; request: SystemOneRequest; labels: number[];
  grid: [number, number]; image_tokens: number; probs: number[][];
  pixels: { shape: [number, number]; sum: number; sum_sq: number; head: number[]; tail: number[] };
}
const name = process.env.KEV_VISION_MODEL ?? "kev-4b-vision";
const fxPath = `${root}/fixtures/${name}.json`;
const fx: { run: string; max_pixels: number; image_token: number; vision_start: number; vision_end: number; fixtures: VisionFixture[] } | null =
  existsSync(fxPath) ? JSON.parse(readFileSync(fxPath, "utf8")) : null;
const png = (f: VisionFixture) => f.image.endsWith(".png");
const image = (f: VisionFixture) => decodePng(readFileSync(`${root}/${f.image}`));

/** Qwen3.5-4B's preprocessor and vision_config with kev.js's pixel cap, as kev_web_export.vision_onnx writes vision.json.
 * The preprocessing keys are the same for every Qwen3.5 size. */
const VISION: VisionConfig = {
  patch_size: 16, merge_size: 2, temporal_patch_size: 2, hidden_size: 1024, out_hidden_size: 2560, num_heads: 16, num_grid_per_side: 48,
  rope_theta: 10000, image_mean: [0.5, 0.5, 0.5], image_std: [0.5, 0.5, 0.5], min_pixels: 65536, max_pixels: 768 * 768,
  image_token: 248056, vision_start: 248053, vision_end: 248054,
};

test("smartResize, vision inputs and mRoPE positions", () => {
  assert.deepEqual(smartResize(632, 760, VISION), [640, 768]);
  assert.deepEqual(smartResize(100, 100, VISION), [256, 256]);        // below min_pixels
  assert.deepEqual(smartResize(800, 1280, VISION), [576, 960]);       // above the cap: 552,960 pixels
  const [t, h, w] = ropePositions([1, 2, 9, 9, 9, 9, 9, 9, 3], 4, 6, 9);   // 2 x 3 merged grid after two text tokens
  assert.deepEqual(t, [0, 1, 2, 2, 2, 2, 2, 2, 5]);
  assert.deepEqual(h, [0, 1, 2, 2, 2, 3, 3, 3, 5]);
  assert.deepEqual(w, [0, 1, 2, 3, 4, 2, 3, 4, 5]);
  const vi = visionInputs(4, 4, VISION);
  // 4 x 4 patches: the first (block order) is patch (0, 0), the table's corner; the last is (3, 3), the opposite corner
  assert.deepEqual(Array.from(vi.posIdx.subarray(0, 4), Number), [0, 1, 48, 49]);
  assert.deepEqual(Array.from(vi.posW.subarray(0, 4)), [1, 0, 0, 0]);
  assert.deepEqual(Array.from(vi.posIdx.subarray(60, 64), Number), [2303, 2303, 2303, 2303]);
  assert.equal(vi.posW[60], 1);
  assert.ok(vi.cos.subarray(0, 64).every((c) => c === 1) && vi.sin.subarray(0, 64).every((x) => x === 0));
});

test("validate accepts RGBA images and rejects malformed ones; text requests are untouched", () => {
  const q = { q: { type: "noul" as const, instructions: "Is it red?" } };
  const img = { width: 2, height: 1, data: new Uint8ClampedArray(8) };
  assert.equal(validate({ state: "x", questions: q, image: img }).image, img);
  assert.throws(() => validate({ state: "x", questions: q, image: { width: 2, height: 2, data: new Uint8Array(8) } }), ValidationError);
  assert.throws(() => validate({ state: "x", questions: q, image: "cat.png" }), ValidationError);
  assert.equal(validate({ state: "x", questions: q }).image, undefined);
  assert.notEqual(imageKey(img), imageKey({ ...img, data: Uint8ClampedArray.of(0, 0, 0, 0, 0, 0, 0, 1) }));
});

test("preprocess reproduces Qwen's pixel_values", { skip: fx ? false : `no fixtures/${name}.json` }, () => {
  assert.equal(fx!.max_pixels, VISION.max_pixels);
  let worstMean = 0, n = 0;
  for (const f of fx!.fixtures.filter(png)) {
    const img = image(f);
    assert.deepEqual([img.width, img.height], f.size, f.id);
    const p = preprocess(img, VISION);
    assert.deepEqual([p.gridH, p.gridW], f.grid, f.id);
    assert.deepEqual([p.data.length / f.pixels.shape[1], f.pixels.shape[1]], f.pixels.shape, f.id);
    let sum = 0, sq = 0;
    for (const x of p.data) { sum += x; sq += x * x; }
    // resized images: allow a resampler that is not torchvision's to differ by a rounding step here and there
    worstMean = Math.max(worstMean, Math.abs(sum - f.pixels.sum) / p.data.length);
    assert.ok(Math.abs(sq - f.pixels.sum_sq) / p.data.length < 2e-3, `${f.set}/${f.id}: sum of squares ${sq} vs ${f.pixels.sum_sq}`);
    f.pixels.head.forEach((v, i) => assert.ok(Math.abs(p.data[i] - v) < 0.02, `${f.set}/${f.id}: head[${i}] ${p.data[i]} vs ${v}`));
    n++;
  }
  console.log(`${n} images: mean |pixel difference| per value at most ${worstMean.toExponential(2)} (one uint8 step is ${(2 / 255).toExponential(2)})`);
  assert.ok(worstMean < 2e-3);
});

const dir = process.env.KEV_VISION_DIR ?? `${root}/public/models/${name}`;
const manifest: KevManifest | null = existsSync(`${dir}/manifest.json`) ? JSON.parse(readFileSync(`${dir}/manifest.json`, "utf8")) : null;
const bundleTokenizer = () => new Tokenizer(JSON.parse(readFileSync(`${dir}/${manifest!.files.tokenizer}`, "utf8")),
  JSON.parse(readFileSync(`${dir}/${manifest!.files.tokenizer_config}`, "utf8")));
const skip = !fx ? "no fixtures" : !manifest ? `no vision bundle in ${dir} (see README: Images)` : false;

test("image requests match PyTorch through the vision graph and the spliced decoder", { skip }, async () => {
  assert.equal(fx!.run, manifest!.run, "fixtures and weights come from the same checkpoint commit");
  const variant = process.env.KEV_VARIANTS?.split(",")[0] ?? Object.keys(manifest!.variants)[0];
  const v = manifest!.variants[variant], t = manifest!.vision!;
  const session = await ort.InferenceSession.create(`${dir}/${v.model}`, Kev.sessionOptions(manifest!, variant, false));
  const vision = await ort.InferenceSession.create(`${dir}/${t.model}`, { graphOptimizationLevel: "all" });
  const head = PointerHead.fromSafetensors(readFileSync(`${dir}/${manifest!.files.head}`).buffer as ArrayBuffer);
  const kev = new Kev({ ort: ort as unknown as OrtModule, session, vision, head, tokenizer: bundleTokenizer(), manifest: manifest!, variant, options: { temperature: 1 } });
  const pre = ["patch_size", "merge_size", "temporal_patch_size", "image_mean", "image_std", "min_pixels", "max_pixels", "image_token", "vision_start", "vision_end"] as const;
  for (const k of pre) assert.deepEqual(t.config[k], VISION[k], `vision.json ${k} matches the preprocessing tested above`);
  const bySet = new Map<string, Parity>();
  let vision_ms = 0, decoder_ms = 0, n = 0;
  for (const f of fx!.fixtures.filter(png)) {
    const probs = await kev.probs(recordOf(f.request), image(f));
    assert.equal(kev.timing.imageTokens, f.image_tokens, f.id);
    vision_ms += kev.timing.vision; decoder_ms += kev.timing.state + kev.timing.branches; n++;
    const parity = bySet.get(f.set) ?? new Parity(); bySet.set(f.set, parity);
    probs.forEach((p, k) => parity.add(`${f.set}/${f.id} q${k}`, f.probs[k], p));
  }
  const res = await kev.systemOne({ ...fx!.fixtures[0].request, image: image(fx!.fixtures[0]) });
  assert.ok(res.usage.input_tokens > fx!.fixtures[0].image_tokens, "usage counts the image tokens");
  assert.equal((await kev.systemOne({ ...fx!.fixtures[0].request, image: image(fx!.fixtures[0]) })).usage.input_tokens, res.usage.input_tokens);
  assert.ok(kev.timing.cached, "the same image and text hit the state cache");
  console.log(`${variant}, ${n} images: vision ${(vision_ms / n).toFixed(0)} ms, decoder ${(decoder_ms / n).toFixed(0)} ms per image (onnxruntime-node CPU)`);
  const worst = Math.max(...Object.values((t.parity?.[variant] ?? {}) as Record<string, { max_abs_dp: number }>).map((s) => s.max_abs_dp), 0.05);
  for (const [s, parity] of bySet) {
    console.log(`${variant} ${s}: ${parity.summary()}`);
    assert.deepEqual(parity.violations({ max: variant === "fp32" ? 1e-4 : 2 * worst, mean: variant === "fp32" ? 1e-4 : MEAN_ABS_DP }), []);
  }
  await kev.release();
});

/** The internal record of a request, as toRecord builds it (imported lazily to keep this file's imports readable). */
import { toRecord } from "../src/index.ts";
const recordOf = (req: SystemOneRequest) => toRecord(req).record;
