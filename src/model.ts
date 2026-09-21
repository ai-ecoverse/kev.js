// Kev runtime over onnxruntime (web or node): the state runs once, every question runs as a continuation of the
// state's cache (Gated DeltaNet recurrent + conv states, KV of the full-attention layers), and the pointer head reads
// each question's <decide> and </opt> hidden states. Exact isolation: branches never see each other.

import type { InferenceSession, Tensor, TypedTensor } from "onnxruntime-common";
import { Tokenizer } from "@huggingface/tokenizers";
import { pyJsonDumps, toAnswers, toRecord, validate, type Answer, type DecisionRecord, type SystemOneRequest, type SystemOneResponse } from "./api.ts";
import { encode, type Encoding, type EncodeOptions, type SpecialTokens } from "./encode.ts";
import { PointerHead } from "./head.ts";

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
}

export interface KevManifest {
  name: string;
  run: string;
  base: string;
  hidden_size: number;
  head_dim: number;
  special: SpecialTokens;
  max_state: number;
  max_branch: number;
  files: { head: string; tokenizer: string; tokenizer_config: string };
  variants: Record<string, VariantManifest>;
}

export interface KevOptions extends EncodeOptions {
  /** kev.serve's KEV_TEMPERATURE: p^(1/T) renormalised. Default 1 (off). */
  temperature?: number;
  /** states whose caches are kept for reuse (LRU). Default 4, like kev.serve. 0 disables. */
  stateCacheSize?: number;
}

type Past = Map<string, Tensor>;

interface CachedState { len: number; past: Past }

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
  private opts: Required<Pick<KevOptions, "temperature" | "stateCacheSize">> & EncodeOptions;
  private cache = new Map<string, CachedState>();
  private queue: Promise<unknown> = Promise.resolve();
  private v: VariantManifest;

  constructor(a: { ort: OrtModule; session: InferenceSession; head: PointerHead; tokenizer: Tokenizer; manifest: KevManifest; variant: string; options?: KevOptions }) {
    this.ort = a.ort; this.session = a.session; this.head = a.head; this.tokenizer = a.tokenizer;
    this.manifest = a.manifest; this.variant = a.variant; this.v = a.manifest.variants[a.variant];
    if (!this.v) throw new Error(`unknown variant ${a.variant}; have ${Object.keys(a.manifest.variants)}`);
    this.opts = { temperature: 1, stateCacheSize: 4, maxState: 8192, maxBranch: 8192, ...a.options };
  }

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

  private feeds(ids: number[], pos: number[], past: Past, pastLen: number): Record<string, Tensor> {
    const S = ids.length;
    return {
      input_ids: this.tensor("int64", ids, [1, S]),
      attention_mask: this.tensor("int64", new Array(pastLen + S).fill(1), [1, pastLen + S]),
      position_ids: this.tensor("int64", [...pos, ...pos, ...pos], [3, 1, S]),   // text positions: all three mRoPE sections equal
      ...Object.fromEntries(past),
    };
  }

  private async runState(ids: number[], pos: number[]): Promise<CachedState> {
    const empty = this.emptyPast();
    const presentNames = this.v.outputs.map((o) => o.name).filter((n) => n.startsWith("present"));
    const out = await this.session.run(this.feeds(ids, pos, empty, 0), presentNames);
    const past: Past = new Map();
    for (const n of presentNames)
      past.set(n.endsWith(".key") || n.endsWith(".value") ? n.replace("present.", "past_key_values.") : n.replace("present.", "past."), out[n]);
    return { len: ids.length, past };
  }

  private disposeState(s: CachedState) {
    for (const t of s.past.values()) (t as Tensor & { dispose?: () => void }).dispose?.();
  }

  /** Per-question probabilities for an encoded request. onQuestion fires as each question finishes. */
  probsEncoded(enc: Encoding, onQuestion?: (index: number, probs: number[]) => void): Promise<number[][]> {
    return this.serial(async () => {
      const key = enc.state.join(",");
      let st = this.cache.get(key);
      if (st) this.cache.delete(key);
      else st = await this.runState(enc.state, enc.state.map((_, i) => i));
      const d = this.manifest.hidden_size;
      const probs: number[][] = [];
      try {
        for (const [i, b] of enc.branches.entries()) {
          const out = await this.session.run(this.feeds(b.ids, b.pos, st.past, st.len), ["hidden_states"]);
          const h = toFloat32(out.hidden_states as TypedTensor<"float32">);
          const row = (i: number) => h.subarray(i * d, (i + 1) * d);
          const p = this.head.probs(row(b.decide), b.opts.map(row), this.opts.temperature);
          probs.push(p); onQuestion?.(i, p);
        }
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

  async probs(rec: DecisionRecord): Promise<number[][]> {
    return this.probsEncoded(this.encode(rec));
  }

  /** POST /v1/systemone. onAnswer fires per question, so a caller can show answers as they land. */
  async systemOne(input: SystemOneRequest | unknown, opts: { onAnswer?: (id: string, answer: Answer, index: number) => void } = {}): Promise<SystemOneResponse> {
    const req = validate(input);
    const { record, meta } = toRecord(req);
    const enc = this.encode(record);
    const t0 = performance.now();
    const probs = await this.probsEncoded(enc, opts.onAnswer && ((i, p) => {
      const m = meta[i];
      opts.onAnswer!(m.id, toAnswers([p], [m])[m.id], i);
    }));
    const latency = performance.now() - t0;
    const answers = toAnswers(probs, meta);
    const outputTokens = this.tokenizer.encode(pyJsonDumps(answers), { add_special_tokens: false }).ids.length;
    return { model: req.model ?? "kev-latest", answers, usage: { input_tokens: enc.tokens, output_tokens: outputTokens }, latency_ms: Math.round(latency * 10) / 10 };
  }

  /** POST /v1/systemone/separate: each question in its own request against the same state. */
  async systemOneSeparate(input: SystemOneRequest | unknown): Promise<SystemOneResponse> {
    const req = validate(input);
    const merged: SystemOneResponse = { model: req.model ?? "kev-latest", answers: {}, usage: { input_tokens: 0, output_tokens: 0 }, latency_ms: 0 };
    for (const [id, q] of Object.entries(req.questions)) {
      const r = await this.systemOne({ ...req, questions: { [id]: q } });
      Object.assign(merged.answers, r.answers);
      merged.usage.input_tokens += r.usage.input_tokens; merged.latency_ms += r.latency_ms;
    }
    merged.usage.output_tokens = this.tokenizer.encode(pyJsonDumps(merged.answers), { add_special_tokens: false }).ids.length;
    merged.latency_ms = Math.round(merged.latency_ms * 10) / 10;
    return merged;
  }

  clearCache() { for (const s of this.cache.values()) this.disposeState(s); this.cache.clear(); }

  async release() { this.clearCache(); await this.session.release(); }
}
