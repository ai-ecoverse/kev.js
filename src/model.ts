// Kev runtime over onnxruntime (web or node): the state runs once, every question runs as a continuation of the
// state's cache (Gated DeltaNet recurrent + conv states, KV of the full-attention layers), and the pointer head reads
// each question's <decide> and </opt> hidden states. Exact isolation: branches never see each other.

import type { InferenceSession, Tensor, TypedTensor } from "onnxruntime-common";
import { Tokenizer } from "@huggingface/tokenizers";
import { pyJsonDumps, toAnswers, toRecord, validate, withDateFacts, type Answer, type DecisionRecord, type SystemOneRequest, type SystemOneResponse } from "./api.ts";
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
  /** Fitted pointer-head temperature from head.pt. Absent on bundles exported before 2026-09-21. */
  temperature?: number;
  files: { head: string; tokenizer: string; tokenizer_config: string };
  variants: Record<string, VariantManifest>;
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
  private opts: Required<Pick<KevOptions, "stateCacheSize" | "dateFacts">> & EncodeOptions;
  private cache = new Map<string, CachedState>();
  private queue: Promise<unknown> = Promise.resolve();
  private v: VariantManifest;

  constructor(a: { ort: OrtModule; session: InferenceSession; head: PointerHead; tokenizer: Tokenizer; manifest: KevManifest; variant: string; options?: KevOptions }) {
    this.ort = a.ort; this.session = a.session; this.head = a.head; this.tokenizer = a.tokenizer;
    this.manifest = a.manifest; this.variant = a.variant; this.v = a.manifest.variants[a.variant];
    if (!this.v) throw new Error(`unknown variant ${a.variant}; have ${Object.keys(a.manifest.variants)}`);
    this.opts = { stateCacheSize: 4, dateFacts: false, maxState: 8192, maxBranch: 8192, ...a.options };
    this.head.temperature = a.options?.temperature ?? temperatureFor(a.manifest.run, a.manifest.temperature);
  }

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
          const p = this.head.probs(row(b.decide), b.opts.map(row));
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
  async systemOne(input: SystemOneRequest | unknown, opts: { onAnswer?: (id: string, answer: Answer, index: number) => void; dateFacts?: boolean } = {}): Promise<SystemOneResponse> {
    let req = validate(input);
    if (opts.dateFacts ?? this.opts.dateFacts) req = { ...req, state: withDateFacts(req.state) };
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

  async release() { this.clearCache(); await this.session.release(); }
}
