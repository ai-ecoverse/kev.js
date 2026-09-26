// Kev runtime over onnxruntime (web or node): the state runs once, every question runs as a continuation of the
// state's cache (Gated DeltaNet recurrent + conv states, KV of the full-attention layers), and the pointer head reads
// each question's <decide> and </opt> hidden states. Exact isolation: branches never see each other.

import type { InferenceSession, Tensor, TypedTensor } from "onnxruntime-common";
import { Tokenizer } from "@huggingface/tokenizers";
import { pyJsonDumps, toAnswers, toRecord, validate, withDateFacts, type Answer, type DecisionRecord, type SystemOneRequest, type SystemOneResponse } from "./api.ts";
import { encode, type Encoding, type EncodeOptions, type SpecialTokens } from "./encode.ts";
import { PointerHead } from "./head.ts";
import { imageKey, preprocess, ropePositions, smartResize, visionInputs, type ImageLike, type VisionConfig } from "./vision.ts";

/** The subset of the onnxruntime-web / onnxruntime-node module the runtime uses (both satisfy it). */
export interface OrtModule {
  InferenceSession: { create(model: Uint8Array | string, options?: InferenceSession.SessionOptions): Promise<InferenceSession> };
  Tensor: new (type: Tensor.Type, data: Tensor.DataType, dims: readonly number[]) => Tensor;
}

export interface IOInfo {
  name: string;
  type: "float16" | "float32" | "int64";
  shape: (number | string)[];
  /** cache inputs: the shape of an empty cache (batch 1, no past tokens) */
  empty?: number[];
}

export interface VariantManifest {
  model: string;
  data: string[];
  bytes: number;
  io_dtype: "float16" | "float32";
  /** bytes per file, so download progress is right even without a content-length header */
  sizes?: Record<string, number>;
  inputs: IOInfo[];
  outputs: IOInfo[];
  /** parity against the fp32 PyTorch model on the bundled fixtures */
  parity?: { max_abs_dp: number; argmax_flips: number; questions: number };
  /** rows of the graph's rotary tables (SERVE_MAX_BRANCH = 73,728 after postprocess): no position may reach it */
  max_positions?: number;
}

/** The vision tower of a bundle that takes images (kev_web_export.vision_onnx + package --vision). Its decoder
 * variants have an `image_embeds` input (kev_web_export.splice). */
export interface VisionManifest {
  model: string;
  data: string[];
  bytes: number;
  sizes?: Record<string, number>;
  config: VisionConfig;
  /** image fixtures through this tower and each variant, against fp32 PyTorch */
  parity?: Record<string, unknown>;
}

/** Where the time of the last request went, in ms. Image requests fill preprocess and vision. */
export interface Timing { preprocess: number; vision: number; state: number; branches: number; cached: boolean; imageTokens: number }

export interface KevManifest {
  name: string;
  /** the checkpoint the weights come from, repo@commit */
  run: string;
  /** digest of the bundle's files, which the loader keys Cache Storage by; older manifests have none (then `run`) */
  revision?: string;
  base: string;
  hidden_size: number;
  head_dim: number;
  special: SpecialTokens;
  max_state: number;
  max_branch: number;
  /** Fitted pointer-head temperature from head.pt. Absent on bundles exported before 2026-09-21. */
  temperature?: number;
  files: { head: string; tokenizer: string; tokenizer_config: string };
  variants: Record<string, VariantManifest>;
  vision?: VisionManifest;
}

export interface KevOptions extends EncodeOptions {
  /** Override the checkpoint temperature. Unset = manifest.temperature, else the night-2 fitted value, else 1 (raw). */
  temperature?: number;
  /** kev.serve's KEV_DATE_FACTS: append day counts between absolute dates in the state. Default false. */
  dateFacts?: boolean;
  /** states whose caches are kept for reuse (LRU). Default 4, like kev.serve. 0 disables. */
  stateCacheSize?: number;
}

/** Night-2 Qwen3.5 checkpoints: the same LoRA later gained a fitted T in head.pt (2026-09-21 calibration republish). */
const NIGHT2_TEMPERATURE: { sha: string; t: number }[] = [
  { sha: "225679690cdd1de6fceb1258b1bddf61c493cee9", t: 2.406050072164233 },
  { sha: "54f4f8777356cd5bbbb6c6919c657f26e6f2f6d8", t: 2.406050072164233 },
  { sha: "d842b1f6c7d8780d0686b9730381b422fe0308a0", t: 2.406050072164233 },
  { sha: "4bc64c6b4c4881148661ffb823ce21fcfdc79a0e", t: 2.1435469250725863 },
  { sha: "e226ccc58b9a4ae9fbbb272cf0d1d0ce6412fe54", t: 2.1435469250725863 },
  { sha: "485ace8703592fcf405488b262449990824cfed1", t: 2.1435469250725863 },
  { sha: "70dd4088ebf4eb82d15ef57a863a5b9a98b94d6c", t: 2.1435469250725863 },
  { sha: "442e597d71840506c326c8c2f5eedd42aeac7bbd", t: 2.2973967099940698 },
  { sha: "3cf1ab729d2b7bd678ec11cbbcdcb78c71fe4b95", t: 2.2973967099940698 },
  { sha: "2629c06a5aeb0feb3b9783bafed17ed8f39ecf5c", t: 2.2973967099940698 },
  { sha: "b54583620720cf4766b81100075f732b20789127", t: 2.2973967099940698 },
];

/** Resolve the serving temperature the way kev.evaluate.load does: explicit override, else checkpoint, else 1. */
export function temperatureFor(run: string, manifestTemperature?: number): number {
  if (manifestTemperature != null) return manifestTemperature;
  const rev = run.includes("@") ? run.slice(run.indexOf("@") + 1).toLowerCase() : "";
  if (!rev) return 1;
  for (const { sha, t } of NIGHT2_TEMPERATURE) {
    if (sha.startsWith(rev) || rev.startsWith(sha.slice(0, 7))) return t;
  }
  return 1;
}

type Past = Map<string, Tensor>;

/** A state run once: its token count, caches, and the first text position of every question branch after it. */
interface CachedState { len: number; past: Past; next: number; imageTokens: number }

const halfToFloat = (() => {
  const f = new Float32Array(1), u = new Uint32Array(f.buffer);
  return (h: number) => {
    const s = (h & 0x8000) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    if (e === 0) return (s ? -1 : 1) * m * 2 ** -24;
    if (e === 31) { u[0] = s | 0x7f800000 | (m << 13); return f[0]; }
    u[0] = s | ((e + 112) << 23) | (m << 13);
    return f[0];
  };
})();

function toFloat32(t: Tensor): Float32Array {
  const d = t.data as ArrayLike<number>;
  if (d instanceof Float32Array) return d;
  if (d instanceof Uint16Array) { const out = new Float32Array(d.length); for (let i = 0; i < d.length; i++) out[i] = halfToFloat(d[i]); return out; }
  return Float32Array.from(d);   // Float16Array, where the runtime returns one
}

export class Kev {
  readonly manifest: KevManifest;
  readonly variant: string;
  readonly tokenizer: Tokenizer;
  private ort: OrtModule;
  private session: InferenceSession;
  private head: PointerHead;
  private opts: Required<Pick<KevOptions, "stateCacheSize" | "dateFacts">> & EncodeOptions;
  private cache = new Map<string, CachedState>();
  private queue: Promise<unknown> = Promise.resolve();
  private v: VariantManifest;
  private vision?: InferenceSession;
  private imageInput: boolean;
  /** Where the time of the last request went. */
  timing: Timing = { preprocess: 0, vision: 0, state: 0, branches: 0, cached: false, imageTokens: 0 };

  constructor(a: { ort: OrtModule; session: InferenceSession; head: PointerHead; tokenizer: Tokenizer; manifest: KevManifest; variant: string; options?: KevOptions; vision?: InferenceSession }) {
    this.ort = a.ort; this.session = a.session; this.head = a.head; this.tokenizer = a.tokenizer; this.vision = a.vision;
    this.manifest = a.manifest; this.variant = a.variant; this.v = a.manifest.variants[a.variant];
    if (!this.v) throw new Error(`unknown variant ${a.variant}; have ${Object.keys(a.manifest.variants)}`);
    this.opts = { stateCacheSize: 4, dateFacts: false, maxState: 65536, maxBranch: 73728, ...a.options };
    this.head.temperature = a.options?.temperature ?? temperatureFor(a.manifest.run, a.manifest.temperature);
    this.imageInput = this.v.inputs.some((i) => i.name === "image_embeds");
    if (this.vision && !(this.imageInput && a.manifest.vision)) throw new Error(`variant ${a.variant} of ${a.manifest.name} takes no image_embeds`);
  }

  /** Whether requests may carry an image (the bundle has a vision tower and it was loaded). */
  get acceptsImages() { return this.vision !== undefined; }

  /** Serving temperature actually in use (checkpoint or override). */
  get temperature() { return this.head.temperature; }

  /** Session options that keep the state caches on the GPU between runs (WebGPU); harmless elsewhere. */
  static sessionOptions(manifest: KevManifest, variant: string, gpu: boolean): InferenceSession.SessionOptions {
    const v = manifest.variants[variant];
    return {
      graphOptimizationLevel: "all",
      ...(gpu ? { preferredOutputLocation: Object.fromEntries(v.outputs.map((o) => [o.name, o.name === "hidden_states" ? "cpu" : "gpu-buffer"])) } : {}),
    };
  }

  encode(rec: DecisionRecord): Encoding {
    return encode(this.tokenizer, rec, this.manifest.special, this.opts);
  }

  private serial<T>(fn: () => Promise<T>): Promise<T> {
    const run = this.queue.then(fn, fn);
    this.queue = run.catch(() => undefined);
    return run;
  }

  private tensor(type: "int64", data: number[], dims: number[]): Tensor {
    return new this.ort.Tensor(type, BigInt64Array.from(data, BigInt), dims);
  }

  private emptyPast(): Past {
    const past: Past = new Map();
    const f16 = this.v.io_dtype === "float16";
    for (const i of this.v.inputs) {
      if (!i.empty) continue;
      const n = i.empty.reduce((a, b) => a * b, 1);
      past.set(i.name, new this.ort.Tensor(i.type, f16 ? new Uint16Array(n) : new Float32Array(n), i.empty));
    }
    return past;
  }

  /** pos: text positions (all three mRoPE sections equal) or [3][S] mRoPE positions. A graph with an image_embeds
   * input gets the image's rows, or one row of zeros for text (its output is then the text graph's). */
  private feeds(ids: number[], pos: number[] | number[][], past: Past, pastLen: number, imageEmbeds?: Tensor): Record<string, Tensor> {
    const S = ids.length;
    const p3 = typeof pos[0] === "number" ? [pos, pos, pos] as number[][] : pos as number[][];
    const feeds: Record<string, Tensor> = {
      input_ids: this.tensor("int64", ids, [1, S]),
      attention_mask: this.tensor("int64", new Array(pastLen + S).fill(1), [1, pastLen + S]),
      position_ids: this.tensor("int64", p3.flat(), [3, 1, S]),
      ...Object.fromEntries(past),
    };
    if (this.imageInput) {
      const d = this.manifest.hidden_size;
      feeds.image_embeds = imageEmbeds ?? new this.ort.Tensor(this.v.io_dtype, this.v.io_dtype === "float16" ? new Uint16Array(d) : new Float32Array(d), [1, d]);
    }
    return feeds;
  }

  private async runState(ids: number[], pos: number[] | number[][], imageEmbeds?: Tensor): Promise<CachedState> {
    const empty = this.emptyPast();
    const presentNames = this.v.outputs.map((o) => o.name).filter((n) => n.startsWith("present"));
    const out = await this.session.run(this.feeds(ids, pos, empty, 0, imageEmbeds), presentNames);
    const past: Past = new Map();
    for (const n of presentNames)
      past.set(n.endsWith(".key") || n.endsWith(".value") ? n.replace("present.", "past_key_values.") : n.replace("present.", "past."), out[n]);
    const last = typeof pos[0] === "number" ? (pos as number[]) : (pos as number[][]).flat();
    return { len: ids.length, past, next: last.reduce((a, b) => Math.max(a, b), -1) + 1, imageTokens: 0 };
  }

  /** The vision tower on one image: [image tokens, hidden] rows for the <|image_pad|> tokens, and the patch grid. */
  private async imageEmbeds(image: ImageLike): Promise<{ embeds: Tensor; gridH: number; gridW: number }> {
    const c = this.manifest.vision!.config;
    if (this.v.io_dtype !== "float32") throw new Error("image input needs a float32-io variant (q8f32)");
    const t0 = performance.now();
    const { data, gridH, gridW } = preprocess(image, c);
    const { posIdx, posW, cos, sin } = visionInputs(gridH, gridW, c);
    const P = gridH * gridW, hd = c.hidden_size / c.num_heads;
    const t1 = performance.now();
    const out = await this.vision!.run({
      patches: new this.ort.Tensor("float32", data, [P, data.length / P]),
      pos_idx: new this.ort.Tensor("int64", posIdx, [P, 4]),
      pos_w: new this.ort.Tensor("float32", posW, [P, 4]),
      cos: new this.ort.Tensor("float32", cos, [P, hd]),
      sin: new this.ort.Tensor("float32", sin, [P, hd]),
    });
    this.timing.preprocess = t1 - t0; this.timing.vision = performance.now() - t1;
    return { embeds: out.image_embeds, gridH, gridW };
  }

  private disposeState(s: CachedState) {
    for (const t of s.past.values()) (t as Tensor & { dispose?: () => void }).dispose?.();
  }

  /** Run a state: text only, or with an image first (<state> <|vision_start|> <|image_pad|> x N <|vision_end|> text,
   * where Qwen's chat template also puts the image before the text), with Qwen3.5's mRoPE positions. */
  private async newState(enc: Encoding, image?: ImageLike): Promise<CachedState> {
    this.timing.preprocess = this.timing.vision = 0; this.timing.imageTokens = 0;
    if (!image) return this.runState(enc.state, enc.state.map((_, i) => i));
    const { ids, pos, n } = this.imageLayout(enc, image);
    const { embeds } = await this.imageEmbeds(image);
    if (embeds.dims[0] !== n) throw new Error(`vision tower returned ${embeds.dims[0]} rows for ${n} image tokens`);
    this.timing.imageTokens = n;
    try {
      return { ...await this.runState(ids, pos, embeds), imageTokens: n };
    } finally {
      (embeds as Tensor & { dispose?: () => void }).dispose?.();
    }
  }

  /** The state's token ids and mRoPE positions with an image in front, from the image's size alone (Qwen's resize
   * fixes the patch grid, so this needs no vision pass). */
  private imageLayout(enc: Encoding, image: ImageLike): { ids: number[]; pos: number[][]; n: number } {
    const c = this.manifest.vision!.config;
    const [H, W] = smartResize(image.height, image.width, c);
    const gridH = H / c.patch_size, gridW = W / c.patch_size, n = (gridH * gridW) / (c.merge_size * c.merge_size);
    const ids = [enc.state[0], c.vision_start, ...new Array(n).fill(c.image_token), c.vision_end, ...enc.state.slice(1)];
    return { ids, pos: ropePositions(ids, gridH, gridW, c.image_token, c.merge_size), n };
  }

  /** The largest position a request uses: the state's (after an image, mRoPE advances by the image's larger side in
   * merged patches, not by its token count) or its longest question's. */
  lastPosition(enc: Encoding, image?: ImageLike): number {
    let next = enc.state.length;
    if (image) next = this.imageLayout(enc, image).pos.flat().reduce((a, b) => Math.max(a, b), -1) + 1;
    let last = next - 1;
    for (const b of enc.branches) for (const p of b.pos) last = Math.max(last, p - enc.state.length + next);
    return last;
  }

  /** Per-question probabilities for an encoded request, with an optional image in front of the state. onQuestion
   * fires as each question finishes. */
  probsEncoded(enc: Encoding, onQuestion?: (index: number, probs: number[]) => void, image?: ImageLike): Promise<number[][]> {
    if (image && !this.vision) throw new Error(this.manifest.vision ? "loaded without the vision tower (vision: false)" : `${this.manifest.name} takes no images; load a vision bundle`);
    const limit = this.v.max_positions, last = limit ? this.lastPosition(enc, image) : 0;
    if (limit && last >= limit) {
      return Promise.reject(new RangeError(`the request needs ${last + 1} positions; this graph's rotary tables cover ${limit}: shorten the state or the questions${image ? ", or use a smaller image" : ""}`));
    }
    return this.serial(async () => {
      const key = (image ? `${imageKey(image)}|` : "") + enc.state.join(",");
      const t0 = performance.now();
      let st = this.cache.get(key);
      this.timing.cached = !!st;
      if (st) { this.cache.delete(key); this.timing.preprocess = this.timing.vision = 0; this.timing.imageTokens = st.imageTokens; }
      else st = await this.newState(enc, image);
      const t1 = performance.now();
      this.timing.state = t1 - t0 - this.timing.preprocess - this.timing.vision;
      const d = this.manifest.hidden_size;
      const probs: number[][] = [];
      try {
        for (const [i, b] of enc.branches.entries()) {
          // branch positions continue the state's: len for text, max(mRoPE positions) + 1 after an image
          const pos = st.next === enc.state.length ? b.pos : b.pos.map((p) => p - enc.state.length + st!.next);
          const out = await this.session.run(this.feeds(b.ids, pos, st.past, st.len), ["hidden_states"]);
          const h = toFloat32(out.hidden_states as TypedTensor<"float32">);
          const row = (i: number) => h.subarray(i * d, (i + 1) * d);
          const p = this.head.probs(row(b.decide), b.opts.map(row));
          probs.push(p); onQuestion?.(i, p);
        }
        this.timing.branches = performance.now() - t1;
      } finally {
        if (this.opts.stateCacheSize > 0) {
          this.cache.set(key, st);
          while (this.cache.size > this.opts.stateCacheSize) {
            const [k, old] = this.cache.entries().next().value!;
            this.cache.delete(k); this.disposeState(old);
          }
        } else this.disposeState(st);
      }
      return probs;
    });
  }

  async probs(rec: DecisionRecord, image?: ImageLike): Promise<number[][]> {
    return this.probsEncoded(this.encode(rec), undefined, image);
  }

  /** POST /v1/systemone. onAnswer fires per question, so a caller can show answers as they land. `image` (RGBA
   * pixels) goes in front of the state on a vision bundle; without it the request is exactly the text request. */
  async systemOne(input: SystemOneRequest | unknown, opts: { onAnswer?: (id: string, answer: Answer, index: number) => void; dateFacts?: boolean } = {}): Promise<SystemOneResponse> {
    let req = validate(input);
    if (opts.dateFacts ?? this.opts.dateFacts) req = { ...req, state: withDateFacts(req.state) };
    const { record, meta } = toRecord(req);
    const enc = this.encode(record);
    const t0 = performance.now();
    const probs = await this.probsEncoded(enc, opts.onAnswer && ((i, p) => {
      const m = meta[i];
      opts.onAnswer!(m.id, toAnswers([p], [m])[m.id], i);
    }), req.image);
    const latency = performance.now() - t0;
    const answers = toAnswers(probs, meta);
    const outputTokens = this.tokenizer.encode(pyJsonDumps(answers), { add_special_tokens: false }).ids.length;
    const inputTokens = enc.tokens + (req.image ? this.timing.imageTokens + 2 : 0);
    return { model: req.model ?? "kev-latest", answers, usage: { input_tokens: inputTokens, output_tokens: outputTokens }, latency_ms: Math.round(latency * 10) / 10 };
  }

  /** POST /v1/systemone/separate: each question in its own request against the same state. */
  async systemOneSeparate(input: SystemOneRequest | unknown, opts: { dateFacts?: boolean } = {}): Promise<SystemOneResponse> {
    const req = validate(input);
    const merged: SystemOneResponse = { model: req.model ?? "kev-latest", answers: {}, usage: { input_tokens: 0, output_tokens: 0 }, latency_ms: 0 };
    for (const [id, q] of Object.entries(req.questions)) {
      const r = await this.systemOne({ ...req, questions: { [id]: q } }, opts);
      Object.assign(merged.answers, r.answers);
      merged.usage.input_tokens += r.usage.input_tokens; merged.latency_ms += r.latency_ms;
    }
    merged.usage.output_tokens = this.tokenizer.encode(pyJsonDumps(merged.answers), { add_special_tokens: false }).ids.length;
    merged.latency_ms = Math.round(merged.latency_ms * 10) / 10;
    return merged;
  }

  clearCache() { for (const s of this.cache.values()) this.disposeState(s); this.cache.clear(); }

  async release() { this.clearCache(); await this.session.release(); await this.vision?.release(); }
}
