// kev-ministral (src/ministral.ts, src/pixtral.ts, loadKevMinistral). Always: Pixtral preprocessing and the token
// layout against the reference (fixtures/kev-ministral-3b.json, from the kev-ministral demo's model.ts at 6fe9246),
// with the published tokenizer and demo screenshots fetched from Hugging Face at the pinned revision; the loader on a
// synthetic bundle; the runtime on a scripted ONNX Runtime (lora_scale, the state cache, chunked prefill, the image
// splice, generation). With the bundle on disk (KEV_MINISTRAL_DIR, default public/models/kev-ministral-3b), the
// probabilities and greedy generations on onnxruntime-node (KEV_MINISTRAL_EP, default cpu).
import "./no-float16.ts";
import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Tokenizer } from "@huggingface/tokenizers";
import {
  escapeSpecials, isMinistralManifest, KevMinistral, loadKev, loadKevMinistral, ministralFiles, ministralHead, ministralParts, pixtralPreprocess,
  pixtralSize, pixtralTokens, sampleTopK, toRecord, validate, type MinistralManifest, type OrtModule, type SystemOneRequest,
} from "../src/index.ts";
import { root } from "./fixtures.ts";
import { Parity } from "./parity.ts";
import { decodePng } from "./png.ts";

interface Fixture {
  name: string; image: string; size: [number, number]; gold: string; also: string[]; request: SystemOneRequest;
  state: number[]; rows: { ids: number[]; decide: number; opts: number[] }[];
  pixels: { height: number; width: number; sum: number; sum_sq: number; head: number[]; tail: number[] };
  image_tokens: number; probs: number[][]; probs_text: number[][];
}
interface GenerationFixture { name: string; prompt: string; image: string | null; max_new: number; input_tokens: number; ids: number[]; text: string }
const fx: { repo: string; revision: string; subfolder: string; fixtures: Fixture[]; generation: GenerationFixture[] } =
  JSON.parse(readFileSync(`${root}/fixtures/kev-ministral-3b.json`, "utf8"));

const modelDir = process.env.KEV_MINISTRAL_DIR ?? `${root}/public/models/kev-ministral-3b`;
const haveModel = existsSync(`${modelDir}/manifest.json`);
const ep = process.env.KEV_MINISTRAL_EP ?? "cpu";

/** A file of the published repo at the fixtures' revision, cached in the OS temp dir (the tokenizer is 17 MB). */
async function published(path: string): Promise<Buffer> {
  const cached = join(tmpdir(), `kev-ministral-${fx.revision.slice(0, 12)}-${path.replaceAll("/", "_")}`);
  if (!existsSync(cached)) {
    const url = `https://huggingface.co/${fx.repo}/resolve/${fx.revision}/${path}`;
    let body: ArrayBuffer | undefined;
    for (let i = 1; !body; i++) {
      try {
        const res = await fetch(url);
        if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
        body = await res.arrayBuffer();
      } catch (e) {
        if (i >= 4 || String(e).includes("HTTP 4")) throw e;
        await new Promise((r) => setTimeout(r, 500 * i));
      }
    }
    const part = `${cached}.${process.pid}.part`;
    writeFileSync(part, Buffer.from(body));
    renameSync(part, cached);
  }
  return readFileSync(cached);
}

const bundleFile = async (p: string) => (haveModel ? readFile(`${modelDir}/${p}`) : published(`${fx.subfolder}/${p}`));
const screenshot = async (f: { image: string }) => decodePng(await published(f.image));

let tok: Tokenizer | undefined;
async function tokenizer() {
  tok ??= new Tokenizer(JSON.parse((await bundleFile("tokenizer.json")).toString()), JSON.parse((await bundleFile("tokenizer_config.json")).toString()));
  return tok;
}

const MANIFEST: MinistralManifest = {
  name: "kev-ministral-test", base: "mistralai/Ministral-3-3B-Base-2512", run: "mn-test",
  files: { embed_tokens: ["embed_tokens.onnx_data"], vision_encoder: ["vision_encoder.onnx_data"], decoder_model_merged: ["decoder_model_merged.onnx_data", "decoder_model_merged.onnx_data_1"] },
  kv_heads: 2, head_dim: 3, layers: 2,
};

/** head.json and head.bin of a pointer head over D = 4 hidden values: q reads value 0, k reads value 1. */
const D = 4;
function headFiles(temperature = 1) {
  const dp = 2, f = new Float32Array(2 * dp * D + 2 * dp);
  f[0] = 1;                 // q.weight[0][0]
  f[dp * D + dp + 1] = 1;   // k.weight[0][1]
  return { json: { dim: D, dp, temperature, delimiter_ids: [20, 21, 22, 23, 26] }, bin: new Uint8Array(f.buffer) };
}

test("pixtralSize fits the longest edge, then rounds each side up to 28 px", () => {
  assert.deepEqual(pixtralSize(576, 1024), [588, 1036]);
  assert.deepEqual(pixtralSize(90, 90), [112, 112]);
  assert.deepEqual(pixtralSize(900, 1700), [840, 1540]);
  assert.deepEqual(pixtralSize(1, 20000), [28, 1540]);       // a sliver keeps one row of patches
  assert.deepEqual(pixtralTokens({ rows: 2, cols: 3 }, { img: 10, brk: 12, end: 13 }), [10, 10, 10, 12, 10, 10, 10, 13]);
  assert.throws(() => pixtralPreprocess({ width: 2, height: 2, data: new Uint8Array(15) }), RangeError);
  const flat = pixtralPreprocess({ width: 28, height: 28, data: new Uint8Array(28 * 28 * 4).fill(255) });
  assert.deepEqual([flat.height, flat.width, flat.rows, flat.cols], [28, 28, 1, 1]);
  assert.ok(Math.abs(flat.data[0] - (1 - 0.48145466) / 0.26862954) < 1e-6);
});

test("Pixtral preprocessing reproduces the reference on every demo screenshot", async () => {
  for (const f of fx.fixtures) {
    const img = await screenshot(f);
    assert.deepEqual([img.height, img.width], f.size, f.name);
    const px = pixtralPreprocess(img);
    assert.deepEqual([px.height, px.width], [f.pixels.height, f.pixels.width], f.name);
    assert.equal(pixtralTokens(px, { img: 10, brk: 12, end: 13 }).length, f.image_tokens, f.name);
    let sum = 0, sq = 0;
    for (const x of px.data) { sum += x; sq += x * x; }
    assert.ok(Math.abs(sum - f.pixels.sum) <= 1e-6 * Math.abs(f.pixels.sum) + 1e-3, `${f.name}: sum ${sum} vs ${f.pixels.sum}`);
    assert.ok(Math.abs(sq - f.pixels.sum_sq) <= 1e-6 * f.pixels.sum_sq, `${f.name}: sum of squares ${sq} vs ${f.pixels.sum_sq}`);
    assert.deepEqual(Array.from(px.data.subarray(0, 4)), f.pixels.head, f.name);
    assert.deepEqual(Array.from(px.data.subarray(px.data.length - 4)), f.pixels.tail, f.name);
  }
});

test("control token names in caller text are broken with a word joiner, longest first", () => {
  const esc = escapeSpecials(["<s>", "[IMG]", "[IMG_END]", "x"]);
  assert.equal(esc("a [IMG_END] b [IMG] <s>"), "a [\u2060IMG_END] b [\u2060IMG] <\u2060s>");
  assert.equal(esc("plain x"), "plain x");
});

test("the token layout matches the reference: <s> <state> state, and one row per question", async () => {
  const kev = new KevMinistral({ ort: fakeOrt().ort, ...fakeSessions(), tokenizer: await tokenizer(), manifest: MANIFEST, head: ministralHead(headFiles().json, headFiles().bin) });
  for (const f of fx.fixtures) {
    const enc = kev.encode(toRecord(validate(f.request)).record);
    assert.deepEqual([1, ...enc.state], f.state, f.name);
    assert.deepEqual(enc.branches.map((b) => ({ ids: b.ids, decide: b.decide, opts: b.opts })), f.rows, f.name);
  }
  const ids = kev.userTokens("[IMG] <SPECIAL_20> [INST]");
  assert.ok(!ids.includes(10) && !ids.includes(20) && !ids.includes(3), `control tokens from text: ${ids}`);
});

test("manifest helpers and the pointer head files", () => {
  assert.ok(isMinistralManifest(MANIFEST));
  assert.ok(!isMinistralManifest({ files: { head: "h" }, variants: {} }));
  assert.ok(!isMinistralManifest(null));
  assert.deepEqual(ministralParts(MANIFEST, false).map((g) => g.model), ["onnx/embed_tokens.onnx", "onnx/decoder_model_merged.onnx"]);
  assert.deepEqual(ministralFiles({ ...MANIFEST, sizes: { "head.bin": 9 } }).slice(0, 5), [
    { path: "tokenizer.json", bytes: undefined }, { path: "tokenizer_config.json", bytes: undefined }, { path: "head.json", bytes: undefined },
    { path: "head.bin", bytes: 9 }, { path: "onnx/embed_tokens.onnx", bytes: undefined },
  ]);
  assert.equal(ministralFiles(MANIFEST).length, 4 + 2 + 3 + 2);
  const { json, bin } = headFiles(2.2);
  const padded = new Uint8Array(bin.length + 1); padded.set(bin, 1);
  const { head, special } = ministralHead(json, padded.subarray(1));
  assert.deepEqual(special, { state: 20, q: 21, opt: 22, opt_end: 23, decide: 26 });
  assert.equal(head.temperature, 2.2);
  assert.deepEqual(head.logits(Float32Array.of(2, 0, 0, 0), [Float32Array.of(0, 1, 0, 0), Float32Array.of(0, 3, 0, 0)]).map((x) => x * Math.SQRT2), [2, 6]);
  assert.throws(() => ministralHead(json, bin.subarray(4)), /head\.bin has/);
});

test("sampleTopK keeps the k best and draws among them", () => {
  const row = Float32Array.of(0, 5, 1, 4, 3);
  assert.equal(sampleTopK(row, 1, 2, () => 0), 1);
  assert.equal(sampleTopK(row, 1, 2, () => 0.999), 3);
  assert.equal(sampleTopK(row, 1e-3, 3, () => 0.5), 1);
  assert.equal(sampleTopK(row, 1, 1, () => 1), 1);
});

// ------------------------------------------------------------------ a scripted ONNX Runtime

type T = { type: string; data: Float32Array | BigInt64Array; dims: number[]; getData: () => Promise<unknown>; dispose: () => void; disposed?: boolean };
interface DecoderCall { n: number; pastLen: number; lora: number; fetches: string[]; first: number }

/** Tensors that record disposal, an embedding table whose row for token t is [t, t + 0.5, 0, 0], a vision encoder
 * whose features are [-1, -2, 0, 0], and a decoder whose hidden states are its inputs_embeds and whose logits favour
 * the token after the last one it saw (EOS after `stopAfter`). */
function fakeOrt(stopAfter = 1e9) {
  const calls: DecoderCall[] = [];
  const all: T[] = [];
  class Tensor {
    disposed = false;
    constructor(readonly type: string, readonly data: Float32Array | BigInt64Array, readonly dims: number[]) { all.push(this as unknown as T); }
    async getData() { return this.data; }
    dispose() { this.disposed = true; }
  }
  const ort = { Tensor, InferenceSession: { create: async () => { throw new Error("use fakeSessions"); } } } as unknown as OrtModule;
  const tensor = (type: string, data: Float32Array | BigInt64Array, dims: number[]) => new Tensor(type, data, dims) as unknown as T;
  return { ort, calls, all, tensor, stopAfter };
}

function fakeSessions(o: ReturnType<typeof fakeOrt> = fakeOrt(), vocab = 131072) {
  const released: string[] = [];
  const session = (name: string, inputNames: string[], outputNames: string[], run: (feeds: Record<string, T>, fetches?: string[]) => Record<string, T>) =>
    ({ inputNames, outputNames, run: async (f: Record<string, T>, fetches?: string[]) => run(f, fetches), release: async () => { released.push(name); } }) as unknown as import("onnxruntime-common").InferenceSession;
  const embed = session("embed", ["input_ids"], ["inputs_embeds"], (f) => {
    const ids = Array.from(f.input_ids.data as BigInt64Array, Number), e = new Float32Array(ids.length * D);
    ids.forEach((t, i) => { e[i * D] = t; e[i * D + 1] = t + 0.5; });
    return { inputs_embeds: o.tensor("float32", e, [1, ids.length, D]) };
  });
  const vision = session("vision", ["pixel_values"], ["image_features"], (f) => {
    const [, , h, w] = f.pixel_values.dims, n = (h / 28) * (w / 28), e = new Float32Array(n * D);
    for (let i = 0; i < n; i++) { e[i * D] = -1; e[i * D + 1] = -2; }
    return { image_features: o.tensor("float32", e, [1, n, D]) };
  });
  const decoder = session("decoder", [], [], (f, fetches = []) => {
    const emb = f.inputs_embeds.data as Float32Array, n = f.inputs_embeds.dims[1], pastLen = f["past_key_values.0.key"].dims[2];
    assert.equal(Number((f.attention_mask.data as BigInt64Array).length), pastLen + n);
    for (const [k, t] of Object.entries(f)) if (k.startsWith("past_key_values.")) assert.ok(!t.disposed, `${k} was disposed before use`);
    o.calls.push({ n, pastLen, lora: (f.lora_scale.data as Float32Array)[0], fetches, first: emb[0] });
    const out: Record<string, T> = {};
    for (const name of fetches) {
      if (name === "hidden_states") out[name] = o.tensor("float32", emb.slice(), [1, n, D]);
      else if (name === "logits") {
        const l = new Float32Array(n * vocab);
        for (let i = 0; i < n; i++) { const last = emb[i * D]; l[i * vocab + (pastLen + i + 1 >= o.stopAfter ? 2 : (last + 1) % vocab)] = 1; l[i * vocab + 20] = 2; }
        out[name] = o.tensor("float32", l, [1, n, vocab]);
      } else out[name] = o.tensor("float32", new Float32Array(0), [1, 2, pastLen + n, 3]);
    }
    return out;
  });
  return { embed, decoder, vision, released };
}

async function fakeKev(o = fakeOrt(), options = {}, vision = true) {
  const s = fakeSessions(o);
  const { json, bin } = headFiles();
  const kev = new KevMinistral({ ort: o.ort, embed: s.embed, decoder: s.decoder, vision: vision ? s.vision : undefined, tokenizer: await tokenizer(), manifest: MANIFEST, head: ministralHead(json, bin), options });
  return { kev, o, s };
}

const REQ: SystemOneRequest = { state: "A page", questions: { a: { type: "noul", instructions: "Is it?" }, b: { type: "choice", instructions: "Which?", criteria: { x: "", y: "", z: "" } } } };

test("decisions: lora_scale 1, the state runs once and is cached, each question continues it", async () => {
  const { kev, o } = await fakeKev();
  const enc = kev.encode(toRecord(REQ).record);
  const res = await kev.systemOne(REQ);
  assert.deepEqual(o.calls.map((c) => [c.n, c.pastLen, c.lora]), [[enc.state.length + 1, 0, 1], ...enc.branches.map((b) => [b.ids.length, enc.state.length + 1, 1])]);
  assert.deepEqual(o.calls[0].fetches.filter((f) => !f.startsWith("present")), []);
  assert.deepEqual(o.calls[1].fetches.slice(0, 2), ["hidden_states", "present.0.key"]);
  assert.equal(o.calls[0].first, 1, "the state starts with <s>");
  assert.equal(res.usage.input_tokens, enc.tokens + 1);
  assert.equal(res.model, "kev-ministral-test");
  assert.ok(o.all.filter((t) => t.dims.length === 4 && t.dims[2] > enc.state.length + 1).every((t) => t.disposed), "question rows' caches are released");
  assert.deepEqual(Object.keys(res.answers), ["a", "b"]);
  assert.ok(Math.abs(Object.values((res.answers.b as { probabilities: Record<string, number> }).probabilities).reduce((a, b) => a + b) - 1) < 1e-3);
  o.calls.length = 0;
  const answered: string[] = [];
  await kev.systemOne(REQ, { onAnswer: (id) => answered.push(id) });
  assert.deepEqual(answered, ["a", "b"]);
  assert.equal(o.calls.length, 2, "a repeated state is not run again");
  assert.ok(kev.timing.cached);
  await kev.probs(toRecord({ ...REQ, state: "Another page, with a longer state" }).record);
  const evicted = o.all.filter((t) => t.dims.length === 4 && t.dims[2] === enc.state.length + 1);
  assert.ok(evicted.length > 0 && evicted.every((t) => t.disposed), "the evicted state's cache is released");
});

test("decisions: a long state is prefilled in chunks; stateCacheSize 0 keeps nothing", async () => {
  const { kev, o } = await fakeKev(fakeOrt(), { prefillChunk: 3, stateCacheSize: 0 });
  const enc = kev.encode(toRecord(REQ).record);
  await kev.probs(toRecord(REQ).record);
  const n = enc.state.length + 1, chunks = Math.ceil(n / 3);
  assert.deepEqual(o.calls.slice(0, chunks).map((c) => [c.n, c.pastLen]), Array.from({ length: chunks }, (_, i) => [Math.min(3, n - 3 * i), 3 * i]));
  assert.ok(o.all.filter((t) => t.dims.length === 4).every((t) => t.disposed), "every cache is released");
  await kev.probs(toRecord(REQ).record);
  assert.equal(o.calls.length, 2 * (chunks + 2));
  const t = await tokenizer();
  assert.throws(() => new KevMinistral({ ort: o.ort, ...fakeSessions(), tokenizer: t, manifest: MANIFEST, head: ministralHead(headFiles().json, headFiles().bin), options: { prefillChunk: 0 } }), /prefillChunk/);
});

test("decisions with an image: the block follows <state>, the features fill its [IMG] positions", async () => {
  const { kev, o } = await fakeKev(fakeOrt(), { temperature: 1 });
  const image = { width: 56, height: 28, data: new Uint8Array(56 * 28 * 4) };
  const rec = toRecord(REQ).record, enc = kev.encode(rec);
  const res = await kev.systemOne({ ...REQ, image });
  assert.equal(kev.timing.imageTokens, 3);
  assert.equal(res.usage.input_tokens, enc.tokens + 1 + 3);
  assert.equal(o.calls[0].n, enc.state.length + 1 + 3);
  assert.equal(o.calls[1].pastLen, enc.state.length + 4);
  await kev.probs(rec);
  assert.equal(o.calls.at(-3)!.n, enc.state.length + 1, "the same text without the image is another state");
  const noVision = (await fakeKev(fakeOrt(), {}, false)).kev;
  assert.equal(noVision.acceptsImages, false);
  await assert.rejects(noVision.probs(rec, image), /without its vision encoder/);
  await assert.rejects(noVision.generate("x", { image }), /without its vision encoder/);
  await assert.rejects(kev.probs(rec, { width: 2, height: 2, data: new Uint8Array(3) }), RangeError);
});

test("decisions: Kev's state limit, truncation and date facts", async () => {
  const long = { ...REQ, state: "A page with many more words than four tokens" };
  const { kev } = await fakeKev(fakeOrt(), { maxState: 4 });
  await assert.rejects(kev.systemOne(long), (e: Error) => e.name === "ContextOverflow" && /over the 4-token limit/.test(e.message));
  const t = (await fakeKev(fakeOrt(), { maxState: 4, truncateStates: true, dateFacts: true })).kev;
  const res = await t.systemOne({ ...REQ, state: "From 2024-01-01 to 2024-01-03 and more words" });
  assert.equal(res.truncated, true);
  assert.equal(res.usage.state_tokens_used, 4);
  assert.ok(res.usage.state_tokens! > 4);
  const plain = (await fakeKev(fakeOrt(), { maxBranch: 4 })).kev;
  assert.throws(() => plain.encode({ state: "x", questions: [{ instr: "a long question", options: ["a", "b"] }] }), /branch too long/);
});

test("generation: lora_scale 0, logits for the last position only, greedy until EOS or maxTokens", async () => {
  const o = fakeOrt(8);
  const { kev } = await fakeKev(o);
  const tokens: number[] = [];
  const r = await kev.generate([1, 2000], { onToken: (_, n) => tokens.push(n) });
  assert.deepEqual(o.calls.map((c) => [c.n, c.pastLen, c.lora, c.fetches.includes("logits")]), [[1, 0, 0, false], [1, 1, 0, true], ...[2, 3, 4, 5, 6, 7].map((p) => [1, p, 0, true])]);
  assert.deepEqual(r.ids, [2001, 2002, 2003, 2004, 2005, 2006]);
  assert.equal(r.finish, "stop");
  assert.equal(r.inputTokens, 2);
  assert.deepEqual(tokens, [1, 2, 3, 4, 5, 6]);
  assert.ok(o.all.filter((t) => t.dims.length === 4).every((t) => t.disposed), "every cache is released");
  const short = await (await fakeKev()).kev.generate("Hello", { maxTokens: 2 });
  assert.equal(short.finish, "length");
  assert.equal(short.ids.length, 2);
  const raw = await (await fakeKev()).kev.generate([1], { maxTokens: 1, maskSpecial: false });
  assert.deepEqual(raw.ids, [20], "control tokens only when maskSpecial is false");
  const sampled = await (await fakeKev()).kev.generate([1, 2000], { maxTokens: 1, temperature: 1, topK: 1, random: () => 0.5 });
  assert.deepEqual(sampled.ids, [2001]);
  const stopped = await (await fakeKev()).kev.generate([1]);
  assert.deepEqual([stopped.ids, stopped.finish], [[], "stop"], "EOS is not part of the text");
  await assert.rejects(kev.generate("x", { maxTokens: 0 }), RangeError);
});

test("generation with an image: at the [IMG] placeholder, else after <s>", async () => {
  const image = { width: 28, height: 56, data: new Uint8Array(28 * 56 * 4) };
  const a = await fakeKev();
  const r = await a.kev.generate("[IMG]\nThis shows", { image, maxTokens: 1 });
  const plain = a.kev.tokenizer.encode("[IMG]\nThis shows", { add_special_tokens: true }).ids;
  assert.equal(r.inputTokens, plain.length - 1 + 4);
  await a.kev.probs(toRecord(REQ).record, { width: 56, height: 28, data: new Uint8Array(56 * 28 * 4) });
  const decided = { ...a.kev.timing };
  await a.kev.generate("[IMG] again", { image, maxTokens: 1 });
  assert.deepEqual(a.kev.timing, decided, "generation leaves the last decision's timing alone");
  assert.equal(a.o.calls[0].n, r.inputTokens - 1);
  assert.ok(r.ms.image >= 0);
  const b = await fakeKev();
  const s = await b.kev.generate([1, 100], { image, maxTokens: 1 });
  assert.equal(s.inputTokens, 6);
  await assert.rejects(b.kev.generate("[IMG] and [IMG]", { image }), /more than one \[IMG\]/);
  await assert.rejects(b.kev.generate("[IMG] alone"), /no image/);
});

test("release frees every session and cached state", async () => {
  const { kev, o, s } = await fakeKev();
  await kev.probs(toRecord(REQ).record);
  await kev.release();
  assert.deepEqual(s.released.sort(), ["decoder", "embed", "vision"]);
  assert.ok(o.all.filter((t) => t.dims.length === 4).every((t) => t.disposed));
});

// ------------------------------------------------------------------ loader

async function syntheticBundle() {
  const enc = (x: string) => new TextEncoder().encode(x);
  const { json, bin } = headFiles();
  return new Map<string, Uint8Array>([
    ["manifest.json", enc(JSON.stringify(MANIFEST))], ["tokenizer.json", await bundleFile("tokenizer.json")], ["tokenizer_config.json", await bundleFile("tokenizer_config.json")],
    ["head.json", enc(JSON.stringify(json))], ["head.bin", bin],
    ["onnx/embed_tokens.onnx", enc("embed")], ["onnx/embed_tokens.onnx_data", enc("E")],
    ["onnx/decoder_model_merged.onnx", enc("decoder")], ["onnx/decoder_model_merged.onnx_data", enc("D0")], ["onnx/decoder_model_merged.onnx_data_1", enc("D1")],
    ["onnx/vision_encoder.onnx", enc("vision")], ["onnx/vision_encoder.onnx_data", enc("V")],
  ]);
}

function recordingOrt(fail?: string, env?: object) {
  const created: { graph: Uint8Array | string; options: Omit<import("onnxruntime-common").InferenceSession.SessionOptions, "externalData"> & { externalData?: { path: string; data: Uint8Array }[] } }[] = [];
  const released: string[] = [];
  const name = (g: Uint8Array | string) => (typeof g === "string" ? g : new TextDecoder().decode(g));
  const ort = { ...fakeOrt().ort, ...(env ? { env } : {}), InferenceSession: { create: async (graph: Uint8Array | string, options: never) => {
    if (fail && name(graph).includes(fail)) throw new Error(`${fail} failed`);
    created.push({ graph, options });
    return { inputNames: [], outputNames: [], release: async () => { released.push(name(graph)); } };
  } } } as unknown as OrtModule;
  return { ort, created, released, name };
}

test("loadKevMinistral reads a bundle from a read function: three sessions, the decoder's cache on the GPU", async () => {
  const files = await syntheticBundle();
  const reads: string[] = [];
  const { ort, created, name } = recordingOrt();
  const phases: string[] = [];
  const kev = await loadKevMinistral(async (p) => { reads.push(p); return files.get(p)!; }, { ort, executionProviders: ["webgpu"], onPhase: (p) => phases.push(p) });
  assert.deepEqual(reads.sort(), [...files.keys()].sort());
  assert.deepEqual(created.map((c) => name(c.graph)), ["embed", "decoder", "vision"]);
  assert.deepEqual(created[1].options.externalData!.map((x) => [x.path, new TextDecoder().decode(x.data)]), [["decoder_model_merged.onnx_data", "D0"], ["decoder_model_merged.onnx_data_1", "D1"]]);
  assert.deepEqual(created[1].options.preferredOutputLocation, { hidden_states: "cpu", logits: "cpu", "present.0.key": "gpu-buffer", "present.0.value": "gpu-buffer", "present.1.key": "gpu-buffer", "present.1.value": "gpu-buffer" });
  assert.equal(created[0].options.preferredOutputLocation, undefined);
  assert.deepEqual(phases, ["manifest", "download", "session", "ready"]);
  assert.ok(kev.acceptsImages);
  assert.equal(kev.temperature, 1);
});

test("loadKevMinistral: filePath hands ORT the graphs' paths; vision: false skips the encoder", async () => {
  const files = await syntheticBundle();
  const reads: string[] = [];
  const { ort, created } = recordingOrt();
  const kev = await loadKevMinistral(async (p) => { reads.push(p); return files.get(p)!; }, { ort, executionProviders: ["cpu"], filePath: (p) => `/m/${p}`, vision: false, temperature: 2 });
  assert.deepEqual(created.map((c) => c.graph), ["/m/onnx/embed_tokens.onnx", "/m/onnx/decoder_model_merged.onnx"]);
  assert.ok(created.every((c) => c.options.externalData === undefined && c.options.preferredOutputLocation === undefined));
  assert.deepEqual(reads.sort(), ["head.bin", "head.json", "manifest.json", "tokenizer.json", "tokenizer_config.json"]);
  assert.equal(kev.acceptsImages, false);
  assert.equal(kev.temperature, 2);
});

test("loadKevMinistral from a URL, and a failed session releases the ones before it", async () => {
  const files = await syntheticBundle();
  const real = globalThis.fetch;
  globalThis.fetch = (async (input: RequestInfo | URL) => new Response(files.get(String(input).replace("https://m.test/b/", "")) as BodyInit)) as typeof fetch;
  try {
    const ok = recordingOrt();
    await loadKevMinistral("https://m.test/b", { ort: ok.ort, cacheName: null });
    assert.equal(ok.created.length, 3);
    const bad = recordingOrt("vision");
    await assert.rejects(loadKevMinistral("https://m.test/b", { ort: bad.ort, cacheName: null }), /vision failed/);
    assert.deepEqual(bad.released, ["embed", "decoder"]);
    await assert.rejects(loadKev("https://m.test/b", { ort: ok.ort, cacheName: null }), /is a kev-ministral bundle: load it with loadKevMinistral/);
    files.set("manifest.json", new TextEncoder().encode(JSON.stringify({ name: "kev", files: {}, variants: {} })));
    await assert.rejects(loadKevMinistral("https://m.test/b", { ort: ok.ort, cacheName: null }), /not a kev-ministral bundle/);
  } finally {
    globalThis.fetch = real;
  }
});

// ------------------------------------------------------------------ the published bundle

test(`the published bundle on onnxruntime-node (${ep}) matches the reference`, { skip: !haveModel && `no bundle in ${modelDir}` }, async () => {
  const ort = await import("onnxruntime-node");
  const kev = await loadKevMinistral((p) => readFile(`${modelDir}/${p}`), {
    ort: ort as unknown as OrtModule, executionProviders: [ep], filePath: (p) => `${modelDir}/${p}`,
    sessionOptions: ep === "cpu" ? { intraOpNumThreads: Number(process.env.KEV_MINISTRAL_THREADS ?? 4) } : {},
  });
  const parity = new Parity();
  let right = 0;
  for (const f of fx.fixtures) {
    const rec = toRecord(validate(f.request)).record;
    const withImage = await kev.probs(rec, await screenshot(f));
    parity.add(`${f.name} image`, f.probs[0], withImage[0]);
    parity.add(`${f.name} text`, f.probs_text[0], (await kev.probs(rec))[0]);
    const keys = Object.keys((f.request.questions.action as { criteria: object }).criteria);
    const pick = keys[withImage[0].indexOf(Math.max(...withImage[0]))];
    if (pick === f.gold || f.also.includes(pick)) right++;
  }
  console.log(`kev-ministral ${ep}: ${parity.summary()}; ${right}/${fx.fixtures.length} demo samples right with the screenshot`);
  assert.deepEqual(parity.violations({ max: 0.25, mean: 0.02 }), []);
  for (const g of fx.generation) {
    const r = await kev.generate(g.prompt, { maxTokens: g.max_new, image: g.image ? await screenshot(g as { image: string }) : undefined });
    assert.equal(r.inputTokens, g.input_tokens, g.name);
    const ref = g.ids.filter((t) => t !== 2);
    let same = 0;
    while (same < ref.length && r.ids[same] === ref[same]) same++;
    console.log(`${g.name}: ${same} / ${ref.length} tokens as the reference: ${JSON.stringify(r.text)}`);
    assert.ok(same >= Math.min(10, ref.length), `${g.name}: only ${same} tokens as the reference`);
  }
  await kev.release();
});

test("loadKevMinistral: a caller's preferredOutputLocation never reaches onnxruntime-node; on onnxruntime-web the decoder keeps its own", async () => {
  const files = await syntheticBundle();
  const read = async (p: string) => files.get(p)!;
  const node = recordingOrt(undefined, { versions: { node: "1.30.0" } });
  await loadKevMinistral(read, { ort: node.ort, executionProviders: ["webgpu"], sessionOptions: { preferredOutputLocation: "cpu", logSeverityLevel: 3 } });
  assert.ok(node.created.every((c) => !("preferredOutputLocation" in c.options) && c.options.logSeverityLevel === 3));
  const web = recordingOrt(undefined, { versions: { web: "1.30.0" } });
  await loadKevMinistral(read, { ort: web.ort, executionProviders: ["webgpu"], sessionOptions: { preferredOutputLocation: "cpu" } });
  assert.equal((web.created[1].options.preferredOutputLocation as Record<string, string>)["present.0.key"], "gpu-buffer");
  assert.equal(web.created[0].options.preferredOutputLocation, "cpu");
});

test("loadKevMinistral releases every session when the instance cannot be built, and checks the head before any", async () => {
  const files = await syntheticBundle();
  const read = async (p: string) => files.get(p)!;
  const late = recordingOrt();
  await assert.rejects(loadKevMinistral(read, { ort: late.ort, prefillChunk: 0 }), /prefillChunk/);
  assert.deepEqual(late.released.sort(), ["decoder", "embed", "vision"]);
  files.set("head.bin", new Uint8Array(8));
  const early = recordingOrt();
  await assert.rejects(loadKevMinistral(read, { ort: early.ort }), /head\.bin has 8 bytes/);
  assert.equal(early.created.length, 0);
});

test("loadKevMinistral announces only the files it reads, and each one completes", async () => {
  const files = await syntheticBundle();
  const sizes = Object.fromEntries([...files].filter(([p]) => p !== "manifest.json").map(([p, b]) => [p, b.length]));
  files.set("manifest.json", new TextEncoder().encode(JSON.stringify({ ...MANIFEST, sizes })));
  const progress: { file: string; loaded: number; total: number }[] = [];
  await loadKevMinistral(async (p) => files.get(p)!, { ort: recordingOrt().ort, vision: false, onProgress: (p) => progress.push(p) });
  const announced = new Set(progress.map((p) => p.file));
  assert.ok(![...announced].some((p) => p.includes("vision")), "the skipped encoder is not announced");
  for (const f of announced) assert.ok(progress.some((p) => p.file === f && p.total > 0 && p.loaded === p.total), `${f} never completes`);
  assert.equal(announced.size, Object.keys(sizes).length - 2);
});
