// One load of the kev-ministral bundle serving two task shapes on WebGPU:
//   decide():   Kev decisions — state (+ image) once, each question row on its cache, lora_scale = 1, pointer head over
//               hidden_states at <decide> and every </opt>
//   generate(): text / image-to-text generation from the same decoder session with lora_scale = 0:
//               the stock Ministral-3-3B-Base, greedy
// Three sessions (embed_tokens, vision_encoder, decoder) are created once; nothing is loaded twice.

import * as ort from "onnxruntime-web/webgpu";
import { Tokenizer } from "@huggingface/tokenizers";
import { toRecord, type SystemOneRequest } from "../../src/api.ts";
import { encode, makeUserTokens, type SpecialTokens } from "./encode.ts";
import { imageTokens, preprocess, type ImageLike } from "./vision.ts";

type Tensor = ort.Tensor;
type Past = Record<string, Tensor>;

export interface Manifest { files: Record<string, string[]>; kv_heads: number; head_dim: number; layers: number }

export type Progress = (file: string, loaded: number, total: number, cached: boolean) => void;
const CACHE = "kev-ministral-bundle-v1";

/** A bundle file as bytes, from Cache Storage when an earlier load stored it (the decoder alone is 4.1 GB), else
 * streamed from the network with progress and then cached. */
async function bytes(url: string, progress?: Progress): Promise<Uint8Array> {
  const name = url.split("/").pop()!;
  const cache = typeof caches !== "undefined" ? await caches.open(CACHE).catch(() => undefined) : undefined;
  const hit = await cache?.match(url);
  if (hit) {
    const b = new Uint8Array(await hit.arrayBuffer());
    progress?.(name, b.length, b.length, true);
    return b;
  }
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  const total = Number(r.headers.get("content-length") || 0);
  const out = total ? new Uint8Array(total) : undefined;
  const parts: Uint8Array[] = [];
  let loaded = 0;
  const reader = r.body!.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (out) out.set(value, loaded); else parts.push(value);
    loaded += value.length;
    progress?.(name, loaded, total || loaded, false);
  }
  let b = out;
  if (!b) { b = new Uint8Array(loaded); let o = 0; for (const p of parts) { b.set(p, o); o += p.length; } }
  await cache?.put(url, new Response(b)).catch(() => { /* quota: run uncached next time */ });
  return b;
}

export interface GenerateOptions {
  /** 0 = greedy (the parity checks use this) */
  temperature?: number;
  topK?: number;
  /** never emit control tokens ([IMG], [INST], <SPECIAL_n>, the decision delimiters, ...) */
  maskSpecial?: boolean;
  onToken?: (text: string, tokens: number) => void;
}

export class KevMinistral {
  private constructor(
    private embed: ort.InferenceSession, private vision: ort.InferenceSession, private dec: ort.InferenceSession,
    private tok: Tokenizer, private user: (t: string) => number[], private sp: SpecialTokens,
    private ids: { bos: number; img: number; brk: number; end: number; eos: number },
    private head: { qw: Float32Array; qb: Float32Array; kw: Float32Array; kb: Float32Array; dim: number; dp: number; T: number },
    private m: Manifest,
    private specialIds: number[] = [],
  ) {}

  static async load(base: string, log: (s: string) => void = () => {}, progress?: Progress): Promise<KevMinistral> {
    const m: Manifest = await (await fetch(`${base}/manifest.json`)).json();
    const session = async (name: string, outLoc?: Record<string, "cpu" | "gpu-buffer">) => {
      const t0 = performance.now();
      const model = await bytes(`${base}/onnx/${name}.onnx`, progress);
      const externalData = [];
      for (const f of m.files[name]) externalData.push({ path: f, data: await bytes(`${base}/onnx/${f}`, progress) });
      const s = await ort.InferenceSession.create(model, { executionProviders: ["webgpu"], externalData, ...(outLoc ? { preferredOutputLocation: outLoc } : {}) });
      log(`${name}: ${((performance.now() - t0) / 1000).toFixed(1)} s`);
      return s;
    };
    const decOut: Record<string, "cpu" | "gpu-buffer"> = { hidden_states: "cpu", logits: "gpu-buffer" };
    for (let l = 0; l < m.layers; l++) { decOut[`present.${l}.key`] = "gpu-buffer"; decOut[`present.${l}.value`] = "gpu-buffer"; }
    const embed = await session("embed_tokens"), vision = await session("vision_encoder"), dec = await session("decoder_model_merged", decOut);
    const tj = await (await fetch(`${base}/tokenizer.json`)).json(), tc = await (await fetch(`${base}/tokenizer_config.json`)).json();
    const tok = new Tokenizer(tj, tc);
    const added: { id: number; content: string; special: boolean }[] = tj.added_tokens;
    const specialIds = added.filter((a) => a.special && a.content !== "</s>").map((a) => a.id);
    const byName = (s: string) => added.find((a) => a.content === s)!.id;
    const hj = await (await fetch(`${base}/head.json`)).json();
    const hb = new Float32Array((await bytes(`${base}/head.bin`)).buffer);
    const D = hj.dim, P = hj.dp;
    let o = 0;
    const qw = hb.subarray(o, o += P * D), qb = hb.subarray(o, o += P), kw = hb.subarray(o, o += P * D), kb = hb.subarray(o, o += P);
    const [state, q, opt, opt_end, decide] = hj.delimiter_ids as number[];
    return new KevMinistral(embed, vision, dec, tok, makeUserTokens(tok, added.map((a) => a.content)), { state, q, opt, opt_end, decide },
      { bos: byName("<s>"), img: byName("[IMG]"), brk: byName("[IMG_BREAK]"), end: byName("[IMG_END]"), eos: byName("</s>") },
      { qw, qb, kw, kb, dim: D, dp: P, T: hj.temperature }, m, specialIds);
  }

  private emptyPast(): Past {
    const past: Past = {};
    for (let l = 0; l < this.m.layers; l++) for (const kv of ["key", "value"])
      past[`past_key_values.${l}.${kv}`] = new ort.Tensor("float32", new Float32Array(0), [1, this.m.kv_heads, 0, this.m.head_dim]);
    return past;
  }

  private async embeds(ids: number[]): Promise<Float32Array> {
    const out = await this.embed.run({ input_ids: new ort.Tensor("int64", BigInt64Array.from(ids.map(BigInt)), [1, ids.length]) });
    return (await out.inputs_embeds.getData()) as Float32Array;
  }

  /** One decoder pass: returns hidden_states (CPU), the logits tensor (GPU) and the new cache. Disposes the old cache
   * unless `keep` (a state prefix reused by several question rows). */
  private async step(emb: Float32Array, n: number, past: Past, pastLen: number, lora: boolean, want: ("hidden_states" | "logits")[], keep = false) {
    const feeds: Record<string, Tensor> = {
      inputs_embeds: new ort.Tensor("float32", emb, [1, n, 3072]),
      attention_mask: new ort.Tensor("int64", new BigInt64Array(pastLen + n).fill(1n), [1, pastLen + n]),
      ...past,
    };
    // onnxruntime-web only accepts declared (required) inputs, so lora_scale is always fed: 1 = decision model, 0 = base
    feeds.lora_scale = new ort.Tensor("float32", new Float32Array([lora ? 1 : 0]), [1]);
    const presents = Object.keys(past).map((k) => k.replace("past_key_values.", "present."));
    const out = await this.dec.run(feeds, [...want, ...presents]);
    if (!keep) for (const t of Object.values(past)) t.dispose?.();
    const next: Past = {};
    for (const p of presents) next[p.replace("present.", "past_key_values.")] = out[p];
    return { out, past: next };
  }

  private pointer(h: Float32Array, row: number, opts: number[]): number[] {
    const { qw, qb, kw, kb, dim, dp, T } = this.head;
    const proj = (w: Float32Array, b: Float32Array, at: number) => {
      const v = new Float32Array(dp);
      for (let i = 0; i < dp; i++) { let s = b[i]; for (let j = 0; j < dim; j++) s += w[i * dim + j] * h[(at) * dim + j]; v[i] = s; }
      return v;
    };
    const qv = proj(qw, qb, row);
    const z = opts.map((o) => { const kv = proj(kw, kb, o); let s = 0; for (let i = 0; i < dp; i++) s += kv[i] * qv[i]; return s / Math.sqrt(dp) / T; });
    const mx = Math.max(...z), e = z.map((x) => Math.exp(x - mx)), sum = e.reduce((a, b) => a + b, 0);
    return e.map((x) => x / sum);
  }

  /** Kev decision for a System One request with an optional screenshot. Returns per-question probabilities, the ids
   * fed (for parity checks against Python) and timings. */
  async decide(req: SystemOneRequest, image?: ImageLike) {
    const t0 = performance.now();
    const { record, meta } = toRecord(req);
    const enc = encode(this.user, record, this.sp);
    let stateIds = [this.ids.bos, ...enc.state];
    let pix: ReturnType<typeof preprocess> | undefined, feats: Float32Array | undefined;
    let imgIds: number[] = [];
    if (image) {
      pix = preprocess(image);
      imgIds = imageTokens(pix, this.ids);
      const vo = await this.vision.run({ pixel_values: new ort.Tensor("float32", pix.data, [1, 3, pix.height, pix.width]) });
      feats = (await vo.image_features.getData()) as Float32Array;
      vo.image_features.dispose?.();
      stateIds = [stateIds[0], stateIds[1], ...imgIds, ...stateIds.slice(2)];
    }
    const t1 = performance.now();
    const emb = await this.embeds(stateIds);
    if (feats) {
      let f = 0;
      for (let i = 0; i < stateIds.length; i++) if (stateIds[i] === this.ids.img) { emb.set(feats.subarray(f * 3072, (f + 1) * 3072), i * 3072); f++; }
      if (f * 3072 !== feats.length) throw new Error(`image features ${feats.length / 3072} != [IMG] tokens ${f}`);
    }
    const pre = await this.step(emb, stateIds.length, this.emptyPast(), 0, true, []);
    pre.out.logits?.dispose?.();
    const probs: number[][] = [];
    for (const b of enc.branches) {
      const r = await this.step(await this.embeds(b.ids), b.ids.length, pre.past, stateIds.length, true, ["hidden_states"], true);
      const h = (await r.out.hidden_states.getData()) as Float32Array;
      probs.push(this.pointer(h, b.decide, b.opts));
      for (const t of Object.values(r.past)) t.dispose?.();
    }
    for (const t of Object.values(pre.past)) t.dispose?.();
    return { probs, meta, stateIds, rows: enc.branches, imgIds, pixels: pix && { height: pix.height, width: pix.width, sample: Array.from(pix.data.subarray(0, 4)) },
             pixelData: pix?.data, ms: { vision: t1 - t0, total: performance.now() - t0 } };
  }

  /** Greedy generation from the stock base (lora_scale at its default 0). `prompt` may contain one "[IMG]" placeholder,
   * expanded to the image block when `image` is given. */
  async generate(prompt: string, maxNew: number, image?: ImageLike, o: GenerateOptions = {}) {
    const t0 = performance.now();
    let ids = this.tok.encode(prompt, { add_special_tokens: true }).ids;
    let feats: Float32Array | undefined;
    if (image) {
      const pix = preprocess(image);
      const at = ids.indexOf(this.ids.img);
      ids = [...ids.slice(0, at), ...imageTokens(pix, this.ids), ...ids.slice(at + 1)];
      const vo = await this.vision.run({ pixel_values: new ort.Tensor("float32", pix.data, [1, 3, pix.height, pix.width]) });
      feats = (await vo.image_features.getData()) as Float32Array; vo.image_features.dispose?.();
    }
    const emb = await this.embeds(ids);
    if (feats) { let f = 0; for (let i = 0; i < ids.length; i++) if (ids[i] === this.ids.img) emb.set(feats.subarray(f * 3072, (f++ + 1) * 3072), i * 3072); }
    const inputIds = ids;
    let r = await this.step(emb, ids.length, this.emptyPast(), 0, false, ["logits"]);
    let len = ids.length;
    const outIds: number[] = [];
    const tFirst = performance.now();
    for (let i = 0; i < maxNew; i++) {
      const lg = (await r.out.logits.getData()) as Float32Array;
      r.out.logits.dispose?.();
      const n = lg.length / 131072, row = lg.subarray((n - 1) * 131072);
      if (o.maskSpecial) for (const s of this.specialIds) row[s] = -Infinity;
      const best = o.temperature ? sample(row, o.temperature, o.topK ?? 40) : argmax(row);
      outIds.push(best);
      o.onToken?.(this.tok.decode(outIds, { skip_special_tokens: true }), outIds.length);
      if (best === this.ids.eos || i === maxNew - 1) break;
      r = await this.step(await this.embeds([best]), 1, r.past, len, false, ["logits"]);
      len += 1;
    }
    for (const t of Object.values(r.past)) t.dispose?.();
    const tEnd = performance.now();
    return { inputIds, outIds, text: this.tok.decode(outIds, { skip_special_tokens: false }),
             ms: { prefill: tFirst - t0, perToken: (tEnd - tFirst) / Math.max(1, outIds.length - 1) } };
  }
}

function argmax(row: Float32Array): number {
  let best = 0; for (let v = 1; v < row.length; v++) if (row[v] > row[best]) best = v;
  return best;
}

/** Top-k sampling at a temperature. */
function sample(row: Float32Array, temperature: number, k: number): number {
  // one pass keeping the k best (sorted descending): sorting all 131,072 logits per token would dominate the step
  const idx: number[] = [];
  for (let v = 0; v < row.length; v++) {
    const x = row[v];
    if (idx.length === k && x <= row[idx[k - 1]]) continue;
    let j = idx.length < k ? idx.length : k - 1;
    if (idx.length < k) idx.push(v);
    while (j > 0 && row[idx[j - 1]] < x) { idx[j] = idx[j - 1]; j--; }
    idx[j] = v;
  }
  const mx = row[idx[0]], w = idx.map((i) => Math.exp((row[i] - mx) / temperature));
  let r = Math.random() * w.reduce((a, b) => a + b, 0);
  for (let j = 0; j < idx.length; j++) { r -= w[j]; if (r <= 0) return idx[j]; }
  return idx[idx.length - 1];
}
