// kev-ministral (ai-ecoverse/kev-ministral, kev-ministral-3b/) over onnxruntime (web or node): Ministral-3-3B-Base,
// Kev's decision LoRA left unmerged in the decoder behind a `lora_scale` input, a Kev pointer head and the Pixtral
// vision encoder. One load serves two task shapes from the same three sessions (embed_tokens, the decoder, the
// vision encoder), so no weight is loaded twice:
//
//   decisions, lora_scale = 1: <s> <state> [image block] state, run once; each question as a continuation of its
//   cache (<q> instructions <opt> option </opt> ... <decide>), and the pointer head over the hidden states at
//   <decide> and every </opt>. Exact isolation: questions never see each other.
//   generation, lora_scale = 0: the stock base model (not instruction-tuned: it continues text), on text or on an
//   image followed by text, greedy or sampled.
//
// A port of the kev-ministral demo (demo/ministral/model.ts, parity-checked against PyTorch by its authors), with
// Kev's request handling (src/model.ts): validation, date facts, state limits, a state cache, timing.

import type { InferenceSession, Tensor } from "onnxruntime-common";
import { Tokenizer } from "@huggingface/tokenizers";
import { pyJsonDumps, toAnswers, toRecord, validate, withDateFacts, type Answer, type DecisionRecord, type SystemOneRequest, type SystemOneResponse } from "./api.ts";
import { ContextOverflow, encode, type Encoding, type SpecialTokens } from "./encode.ts";
import { PointerHead } from "./head.ts";
import type { KevOptions, OrtModule, Timing } from "./model.ts";
import { pixtralPreprocess, pixtralTokens } from "./pixtral.ts";
import { imageKey, type ImageLike } from "./vision.ts";

/** manifest.json of a kev-ministral bundle. Graphs are onnx/<name>.onnx, their weights onnx/<file>. */
export interface MinistralManifest {
  name: string;
  base: string;
  /** the training run the LoRA and the head come from */
  run: string;
  /** digest of the bundle's files, which the loader keys Cache Storage by when present (else `run`) */
  revision?: string;
  note?: string;
  /** the external weight files of each graph, in onnx/ */
  files: { embed_tokens: string[]; decoder_model_merged: string[]; vision_encoder?: string[] };
  /** bytes per bundle path, when the manifest records them (needed for readSync) */
  sizes?: Record<string, number>;
  kv_heads: number;
  head_dim: number;
  layers: number;
}

/** The pointer head as head.json describes it: head.bin holds q.weight [dp, dim], q.bias, k.weight, k.bias (fp32). */
export interface MinistralHeadJson { dim: number; dp: number; temperature: number; delimiter_ids: number[]; delimiters?: string[]; run?: string }

/** Bundle files besides the graphs, at the bundle's root. */
export const MINISTRAL_FILES = { tokenizer: "tokenizer.json", tokenizer_config: "tokenizer_config.json", head_json: "head.json", head_bin: "head.bin" } as const;

/** Whether a manifest.json is a kev-ministral bundle's (loadKevMinistral) rather than a Kev bundle's (loadKev). */
export function isMinistralManifest(m: unknown): m is MinistralManifest {
  const f = (m as { files?: Record<string, unknown> } | null)?.files;
  return !!f && Array.isArray(f.embed_tokens) && Array.isArray(f.decoder_model_merged) && Number.isInteger((m as MinistralManifest).layers);
}

export interface MinistralPart { name: "embed_tokens" | "decoder_model_merged" | "vision_encoder"; model: string; data: string[] }

/** The graphs a load creates, in order: embed_tokens, the decoder and, unless `vision` is false, the vision encoder. */
export function ministralParts(m: MinistralManifest, vision = true): MinistralPart[] {
  const names: MinistralPart["name"][] = ["embed_tokens", "decoder_model_merged", ...(vision && m.files.vision_encoder ? ["vision_encoder" as const] : [])];
  return names.map((name) => ({ name, model: `onnx/${name}.onnx`, data: (m.files[name] ?? []).map((f) => `onnx/${f}`) }));
}

/** Every file a load reads besides manifest.json, with its size when the manifest records one: for a download or
 * resume step. */
export function ministralFiles(m: MinistralManifest, vision = true): { path: string; bytes?: number }[] {
  const paths = [...Object.values(MINISTRAL_FILES), ...ministralParts(m, vision).flatMap((g) => [g.model, ...g.data])];
  return paths.map((path) => ({ path, bytes: m.sizes?.[path] }));
}

/** The pointer head and the delimiter token ids from head.json and head.bin. */
export function ministralHead(json: MinistralHeadJson, bin: Uint8Array): { head: PointerHead; special: SpecialTokens } {
  const { dim, dp } = json;
  if (bin.byteLength !== (2 * dp * dim + 2 * dp) * 4) throw new Error(`head.bin has ${bin.byteLength} bytes, expected ${(2 * dp * dim + 2 * dp) * 4} for dp ${dp}, dim ${dim}`);
  const f = new Float32Array(bin.slice().buffer);   // copy: a view into a larger buffer need not be 4-aligned
  let o = 0;
  const take = (shape: number[]) => { const n = shape.reduce((a, b) => a * b, 1); return { shape, data: f.subarray(o, (o += n)) }; };
  const weights = { "q.weight": take([dp, dim]), "q.bias": take([dp]), "k.weight": take([dp, dim]), "k.bias": take([dp]) };
  const [state, q, opt, opt_end, decide] = json.delimiter_ids;
  return { head: new PointerHead(weights, json.temperature), special: { state, q, opt, opt_end, decide } };
}

const WORD_JOINER = "\u2060";

/** Caller text can never produce a control token: Mistral's tokenizer parses names such as `[IMG]` or `<SPECIAL_20>`
 * out of plain text, so a word joiner goes after the first character of every special token's name, longest names
 * first (the rule kev-ministral was trained with). */
export function escapeSpecials(names: string[]): (text: string) => string {
  const sorted = names.filter((s) => s.length > 1).sort((a, b) => b.length - a.length);
  return (text) => {
    for (const s of sorted) if (text.includes(s)) text = text.split(s).join(s[0] + WORD_JOINER + s.slice(1));
    return text;
  };
}

export interface KevMinistralOptions extends KevOptions {
  /** Tokens per decoder pass when a state or prompt is prefilled. Default 1024. The decoder returns logits for every
   * position it runs (131,072 floats each), so this bounds that buffer (512 MB at 1024) for long states and images. */
  prefillChunk?: number;
}

export interface GenerateOptions {
  /** at most this many new tokens; default 256 */
  maxTokens?: number;
  /** an image for the prompt's one `[IMG]` placeholder, or after <s> when the prompt has none */
  image?: ImageLike;
  /** 0 (default) is greedy */
  temperature?: number;
  /** sample among the k most likely tokens; default 40 */
  topK?: number;
  /** never emit control tokens ([IMG], [INST], <SPECIAL_n>, the decision delimiters, ...); default true */
  maskSpecial?: boolean;
  /** a uniform random number in [0, 1) for sampling; default Math.random */
  random?: () => number;
  /** the text so far after each new token */
  onToken?: (text: string, tokens: number) => void;
}

export interface Generation {
  text: string;
  ids: number[];
  inputTokens: number;
  finish: "stop" | "length";
  /** image: preprocess and encoder; prefill: up to the first new token; perToken: each one after it */
  ms: { image: number; prefill: number; perToken: number };
}

type Past = Record<string, Tensor>;
type Disposable = Tensor & { dispose?: () => void };

/** A state run once: its token count and its KV cache. */
interface CachedState { len: number; past: Past; imageTokens: number }

const dispose = (past: Past) => { for (const t of Object.values(past)) (t as Disposable).dispose?.(); };

function argmax(row: Float32Array): number {
  let best = 0;
  for (let v = 1; v < row.length; v++) if (row[v] > row[best]) best = v;
  return best;
}

/** Top-k sampling at a temperature: one pass keeps the k best, since sorting all 131,072 logits per token would
 * dominate the step. */
export function sampleTopK(row: Float32Array, temperature: number, k: number, random: () => number = Math.random): number {
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
  let r = random() * w.reduce((a, b) => a + b, 0);
  for (let j = 0; j < idx.length; j++) { r -= w[j]; if (r <= 0) return idx[j]; }
  return idx[idx.length - 1];
}

export class KevMinistral {
  readonly manifest: MinistralManifest;
  readonly tokenizer: Tokenizer;
  private ort: OrtModule;
  private embed: InferenceSession;
  private decoder: InferenceSession;
  private vision?: InferenceSession;
  private head: PointerHead;
  private special: SpecialTokens;
  private ids: { bos: number; eos: number; img: number; brk: number; end: number };
  private controls: number[];
  private escape: (text: string) => string;
  private hidden: number;
  private opts: Required<Pick<KevMinistralOptions, "stateCacheSize" | "dateFacts" | "truncateStates" | "maxState" | "maxBranch" | "prefillChunk">>;
  private cache = new Map<string, CachedState>();
  private queue: Promise<unknown> = Promise.resolve();
  /** Where the time of the last decision request went. */
  timing: Timing = { preprocess: 0, vision: 0, state: 0, branches: 0, cached: false, imageTokens: 0 };

  constructor(a: { ort: OrtModule; embed: InferenceSession; decoder: InferenceSession; vision?: InferenceSession; tokenizer: Tokenizer; manifest: MinistralManifest;
    head: { head: PointerHead; special: SpecialTokens }; options?: KevMinistralOptions }) {
    this.ort = a.ort; this.embed = a.embed; this.decoder = a.decoder; this.vision = a.vision; this.tokenizer = a.tokenizer; this.manifest = a.manifest;
    this.head = a.head.head; this.special = a.head.special; this.hidden = this.head.d;
    const { stateCacheSize = 1, dateFacts = false, truncateStates = false, maxState = 65536, maxBranch = 73728, prefillChunk = 1024 } = a.options ?? {};
    if (!(prefillChunk >= 1)) throw new RangeError(`prefillChunk must be at least 1, got ${prefillChunk}`);
    this.opts = { stateCacheSize, dateFacts, truncateStates, maxState, maxBranch, prefillChunk: Math.floor(prefillChunk) };
    if (a.options?.temperature != null) this.head.temperature = a.options.temperature;
    const added = [...a.tokenizer.get_added_tokens_decoder()];
    const id = (name: string) => {
      const found = added.find(([, t]) => t.content === name);
      if (!found) throw new Error(`the tokenizer has no ${name} token`);
      return found[0];
    };
    this.ids = { bos: id("<s>"), eos: id("</s>"), img: id("[IMG]"), brk: id("[IMG_BREAK]"), end: id("[IMG_END]") };
    this.controls = added.filter(([, t]) => t.special && t.content !== "</s>").map(([i]) => i);
    this.escape = escapeSpecials(added.map(([, t]) => t.content));
  }

  /** Session options for one graph: on onnxruntime-web's WebGPU the decoder keeps its KV cache on the GPU between
   * runs. */
  static sessionOptions(name: MinistralPart["name"], manifest: MinistralManifest, gpu: boolean): InferenceSession.SessionOptions {
    if (name !== "decoder_model_merged" || !gpu) return { graphOptimizationLevel: "all" };
    const loc: Record<string, "cpu" | "gpu-buffer"> = { hidden_states: "cpu", logits: "cpu" };
    for (let l = 0; l < manifest.layers; l++) loc[`present.${l}.key`] = loc[`present.${l}.value`] = "gpu-buffer";
    return { graphOptimizationLevel: "all", preferredOutputLocation: loc };
  }

  /** Whether requests may carry an image (the vision encoder was loaded). */
  get acceptsImages() { return this.vision !== undefined; }

  /** Serving temperature of the pointer head (head.json's, or the override). */
  get temperature() { return this.head.temperature; }

  /** Token ids of caller text, with control token names escaped. */
  userTokens(text: string): number[] {
    return this.tokenizer.encode(this.escape(text), { add_special_tokens: false }).ids;
  }

  /** The state and question rows of a record (without <s> and the image, which the state run adds), under Kev's
   * limits: a state over maxState tokens is refused with a ContextOverflow unless truncateStates. */
  encode(rec: DecisionRecord): Encoding {
    const { maxState, maxBranch, truncateStates } = this.opts;
    try {
      return encode(this.tokenizer, rec, this.special, { maxState, maxBranch, strict: !truncateStates, user: (t) => this.userTokens(t) });
    } catch (e) {
      if (!(e instanceof ContextOverflow) || e.maxState == null) throw e;
      const n = (x: number) => x.toLocaleString("en-US");
      throw new ContextOverflow(`state is ${n(e.stateTokens!)} tokens, over the ${n(e.maxState)}-token limit (the <state> token included): `
        + `shorten the document or split it across requests; or load with { truncateStates: true } to read only its first ${n(e.maxState)} tokens `
        + "(responses then say truncated: true)", e.stateTokens, e.maxState);
    }
  }

  private serial<T>(fn: () => Promise<T>): Promise<T> {
    const run = this.queue.then(fn, fn);
    this.queue = run.catch(() => undefined);
    return run;
  }

  private emptyPast(): Past {
    const past: Past = {};
    const { layers, kv_heads, head_dim } = this.manifest;
    for (let l = 0; l < layers; l++) for (const kv of ["key", "value"])
      past[`past_key_values.${l}.${kv}`] = new this.ort.Tensor("float32", new Float32Array(0), [1, kv_heads, 0, head_dim]);
    return past;
  }

  /** inputs_embeds of token ids, as [S, hidden] floats. */
  private async embeddings(ids: number[]): Promise<Float32Array> {
    const out = await this.embed.run({ input_ids: new this.ort.Tensor("int64", BigInt64Array.from(ids, BigInt), [1, ids.length]) });
    const t = out[this.embed.outputNames[0]] as Disposable;
    const data = (await t.getData()) as Float32Array;
    t.dispose?.();
    return data;
  }

  /** The image's pixel values through the vision encoder, its token block, and the time each took. */
  private async encodeImage(image: ImageLike): Promise<{ features: Float32Array; tokens: number[]; preprocess: number; vision: number }> {
    if (!this.vision) throw new Error("this kev-ministral instance was loaded without its vision encoder (vision: false)");
    const t0 = performance.now();
    const px = pixtralPreprocess(image);
    const t1 = performance.now();
    const out = await this.vision.run({ [this.vision.inputNames[0]]: new this.ort.Tensor("float32", px.data, [1, 3, px.height, px.width]) });
    const t = out[this.vision.outputNames[0]] as Disposable;
    const features = (await t.getData()) as Float32Array;
    t.dispose?.();
    return { features, tokens: pixtralTokens(px, this.ids), preprocess: t1 - t0, vision: performance.now() - t1 };
  }

  /** inputs_embeds for ids, with the image's features at its [IMG] positions. */
  private async embedWithImage(ids: number[], features?: Float32Array): Promise<Float32Array> {
    const emb = await this.embeddings(ids);
    if (!features) return emb;
    const d = this.hidden;
    let f = 0;
    for (let i = 0; i < ids.length; i++) if (ids[i] === this.ids.img) { emb.set(features.subarray(f * d, (f + 1) * d), i * d); f++; }
    if (f * d !== features.length) throw new Error(`the vision encoder returned ${features.length / d} rows for ${f} [IMG] tokens`);
    return emb;
  }

  /** One decoder pass over n positions after pastLen cached ones, returning `want` and the new cache. The old cache is
   * disposed unless `keep`. The new cache is always fetched: onnxruntime-node's WebGPU EP fails a GroupQueryAttention
   * run whose present outputs are left out. */
  private async step(emb: Float32Array, n: number, past: Past, pastLen: number, lora: boolean, want: ("hidden_states" | "logits")[], keep = false) {
    const feeds: Record<string, Tensor> = {
      inputs_embeds: new this.ort.Tensor("float32", emb, [1, n, this.hidden]),
      attention_mask: new this.ort.Tensor("int64", new BigInt64Array(pastLen + n).fill(1n), [1, pastLen + n]),
      // onnxruntime-web accepts only declared inputs and requires each: 1 = the decision model, 0 = the stock base
      lora_scale: new this.ort.Tensor("float32", new Float32Array([lora ? 1 : 0]), [1]),
      ...past,
    };
    const presents = Object.keys(past).map((k) => k.replace("past_key_values.", "present."));
    let out: InferenceSession.OnnxValueMapType;
    try {
      out = await this.decoder.run(feeds, [...want, ...presents]);
    } finally {
      if (!keep) dispose(past);
    }
    const next: Past = {};
    for (const p of presents) next[p.replace("present.", "past_key_values.")] = out[p] as Tensor;
    return { out, past: next };
  }

  /** Run positions pastLen.. in passes of at most prefillChunk tokens; the last pass returns `want`. */
  private async prefill(emb: Float32Array, n: number, past: Past, pastLen: number, lora: boolean, want: ("hidden_states" | "logits")[]) {
    const chunk = this.opts.prefillChunk, d = this.hidden;
    let at = 0;
    for (; n - at > chunk; at += chunk) ({ past } = await this.step(emb.subarray(at * d, (at + chunk) * d), chunk, past, pastLen + at, lora, []));
    return this.step(emb.subarray(at * d, n * d), n - at, past, pastLen + at, lora, want);
  }

  /** The state (<s> <state> [image block] text) through the decision model. */
  private async newState(enc: Encoding, image?: ImageLike): Promise<CachedState> {
    this.timing.preprocess = this.timing.vision = 0; this.timing.imageTokens = 0;
    let ids = [this.ids.bos, ...enc.state];
    let features: Float32Array | undefined;
    if (image) {
      const img = await this.encodeImage(image);
      features = img.features;
      ids = [ids[0], ids[1], ...img.tokens, ...ids.slice(2)];
      this.timing.preprocess = img.preprocess; this.timing.vision = img.vision; this.timing.imageTokens = img.tokens.length;
    }
    const emb = await this.embedWithImage(ids, features);
    const { past } = await this.prefill(emb, ids.length, this.emptyPast(), 0, true, []);
    return { len: ids.length, past, imageTokens: this.timing.imageTokens };
  }

  /** Per-question probabilities for an encoded request, with an optional image in front of the state. onQuestion
   * fires as each question finishes. */
  probsEncoded(enc: Encoding, onQuestion?: (index: number, probs: number[]) => void, image?: ImageLike): Promise<number[][]> {
    if (image && !this.vision) return Promise.reject(new Error("this kev-ministral instance was loaded without its vision encoder (vision: false)"));
    return this.serial(async () => {
      const key = (image ? `${imageKey(image)}|` : "") + enc.state.join(",");
      const t0 = performance.now();
      let st = this.cache.get(key);
      this.timing.cached = !!st;
      if (st) { this.cache.delete(key); this.timing.preprocess = this.timing.vision = 0; this.timing.imageTokens = st.imageTokens; }
      else st = await this.newState(enc, image);
      const t1 = performance.now();
      this.timing.state = t1 - t0 - this.timing.preprocess - this.timing.vision;
      const d = this.hidden;
      const probs: number[][] = [];
      try {
        for (const [i, b] of enc.branches.entries()) {
          const { out, past } = await this.step(await this.embeddings(b.ids), b.ids.length, st.past, st.len, true, ["hidden_states"], true);
          dispose(past);
          const t = out.hidden_states as Disposable;
          const h = (await t.getData()) as Float32Array;
          t.dispose?.();
          const row = (k: number) => h.subarray(k * d, (k + 1) * d);
          const p = this.head.probs(row(b.decide), b.opts.map(row));
          probs.push(p); onQuestion?.(i, p);
        }
        this.timing.branches = performance.now() - t1;
      } finally {
        if (this.opts.stateCacheSize > 0) {
          this.cache.set(key, st);
          while (this.cache.size > this.opts.stateCacheSize) {
            const [k, old] = this.cache.entries().next().value!;
            this.cache.delete(k); dispose(old.past);
          }
        } else dispose(st.past);
      }
      return probs;
    });
  }

  async probs(rec: DecisionRecord, image?: ImageLike): Promise<number[][]> {
    return this.probsEncoded(this.encode(rec), undefined, image);
  }

  /** POST /v1/systemone, as Kev.systemOne: `image` (RGBA pixels) goes in front of the state. */
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
    const inputTokens = enc.tokens + 1 + (req.image ? this.timing.imageTokens : 0);
    const res: SystemOneResponse = { model: req.model ?? this.manifest.name, answers, usage: { input_tokens: inputTokens, output_tokens: outputTokens }, latency_ms: Math.round(latency * 10) / 10 };
    if (this.opts.truncateStates) {
      res.usage.state_tokens = enc.stateTokens; res.usage.state_tokens_used = enc.state.length;
      res.truncated = enc.stateTruncated;
    }
    return res;
  }

  /** Text from the stock base model (lora_scale = 0), from a prompt or its token ids (which then start with <s>).
   * The base model is not instruction-tuned: start the text you want continued. */
  generate(prompt: string | number[], o: GenerateOptions = {}): Promise<Generation> {
    const { maxTokens = 256, image, temperature = 0, topK = 40, maskSpecial = true, random, onToken } = o;
    if (!(maxTokens >= 1)) return Promise.reject(new RangeError(`maxTokens must be at least 1, got ${maxTokens}`));
    if (image && !this.vision) return Promise.reject(new Error("this kev-ministral instance was loaded without its vision encoder (vision: false)"));
    return this.serial(async () => {
      const t0 = performance.now();
      let ids = typeof prompt === "string" ? this.tokenizer.encode(prompt, { add_special_tokens: true }).ids : [...prompt];
      const slots = ids.filter((t) => t === this.ids.img).length;
      if (slots > (image ? 1 : 0)) throw new RangeError(image ? "the prompt has more than one [IMG] placeholder for one image" : "the prompt has an [IMG] placeholder but no image");
      let features: Float32Array | undefined;
      let imageMs = 0;
      if (image) {
        const img = await this.encodeImage(image);
        features = img.features;
        imageMs = img.preprocess + img.vision;
        const at = slots ? ids.indexOf(this.ids.img) : ids[0] === this.ids.bos ? 1 : 0;
        ids = [...ids.slice(0, at), ...img.tokens, ...ids.slice(at + (slots ? 1 : 0))];
      }
      const inputTokens = ids.length;
      const emb = await this.embedWithImage(ids, features);
      // all but the last position without logits, so only the last position's row is computed for the CPU
      let past = this.emptyPast();
      if (ids.length > 1) ({ past } = await this.prefill(emb.subarray(0, (ids.length - 1) * this.hidden), ids.length - 1, past, 0, false, []));
      let r = await this.step(emb.subarray((ids.length - 1) * this.hidden), 1, past, ids.length - 1, false, ["logits"]);
      let len = ids.length;
      const out: number[] = [];
      const tFirst = performance.now();
      let finish: Generation["finish"] = "length";
      try {
        for (;;) {
          const t = r.out.logits as Disposable;
          const row = ((await t.getData()) as Float32Array).slice(-t.dims[t.dims.length - 1]);
          t.dispose?.();
          if (maskSpecial) for (const s of this.controls) row[s] = -Infinity;
          const next = temperature > 0 ? sampleTopK(row, temperature, topK, random) : argmax(row);
          if (next === this.ids.eos) { finish = "stop"; break; }
          out.push(next);
          onToken?.(this.tokenizer.decode(out, { skip_special_tokens: true }), out.length);
          if (out.length >= maxTokens) break;
          r = await this.step(await this.embeddings([next]), 1, r.past, len, false, ["logits"]);
          len += 1;
        }
      } finally {
        dispose(r.past);
      }
      const tEnd = performance.now();
      return {
        text: out.length ? this.tokenizer.decode(out, { skip_special_tokens: true }) : "", ids: out, inputTokens, finish,
        ms: { image: imageMs, prefill: tFirst - t0, perToken: (tEnd - tFirst) / Math.max(1, out.length - 1) },
      };
    });
  }

  clearCache() { for (const s of this.cache.values()) dispose(s.past); this.cache.clear(); }

  async release() {
    this.clearCache();
    await this.decoder.release(); await this.embed.release(); await this.vision?.release();
  }
}
