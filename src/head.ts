// Kev's pointer head: logits_k = (Wk h_opt_k + bk) · (Wq h_decide + bq) / sqrt(dp), softmax over the options.
// Weights ship as fp32 safetensors (head.safetensors, written by kev_web_export.merge).

export interface F32Tensor { shape: number[]; data: Float32Array }

export function parseSafetensors(buf: ArrayBuffer): Record<string, F32Tensor> {
  const n = Number(new DataView(buf).getBigUint64(0, true));
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, n))) as Record<string, { dtype: string; shape: number[]; data_offsets: [number, number] }>;
  const out: Record<string, F32Tensor> = {};
  for (const [name, t] of Object.entries(header)) {
    if (name === "__metadata__") continue;
    if (t.dtype !== "F32") throw new Error(`${name}: expected F32, got ${t.dtype}`);
    const [a, b] = t.data_offsets;
    out[name] = { shape: t.shape, data: new Float32Array(buf.slice(8 + n + a, 8 + n + b)) };   // copy: offsets need not be 4-aligned
  }
  return out;
}

export class PointerHead {
  readonly d: number;
  readonly dp: number;
  private qw: Float32Array; private qb: Float32Array; private kw: Float32Array; private kb: Float32Array;

  constructor(weights: Record<string, F32Tensor>) {
    const q = weights["q.weight"], k = weights["k.weight"];
    if (!q || !k) throw new Error("head weights must contain q.weight and k.weight");
    [this.dp, this.d] = q.shape;
    this.qw = q.data; this.qb = weights["q.bias"].data; this.kw = k.data; this.kb = weights["k.bias"].data;
  }

  static fromSafetensors(buf: ArrayBuffer) { return new PointerHead(parseSafetensors(buf)); }

  private project(w: Float32Array, b: Float32Array, h: Float32Array): Float64Array {
    const out = new Float64Array(this.dp);
    for (let i = 0; i < this.dp; i++) {
      let s = b[i];
      const row = i * this.d;
      for (let j = 0; j < this.d; j++) s += w[row + j] * h[j];
      out[i] = s;
    }
    return out;
  }

  logits(hDecide: Float32Array, hOpts: Float32Array[]): number[] {
    const q = this.project(this.qw, this.qb, hDecide);
    const scale = 1 / Math.sqrt(this.dp);
    return hOpts.map((h) => {
      const k = this.project(this.kw, this.kb, h);
      let s = 0;
      for (let i = 0; i < this.dp; i++) s += k[i] * q[i];
      return s * scale;
    });
  }

  /** Softmax probabilities; temperature != 1 matches kev.serve's KEV_TEMPERATURE (p^(1/T), renormalised). */
  probs(hDecide: Float32Array, hOpts: Float32Array[], temperature = 1): number[] {
    const z = this.logits(hDecide, hOpts).map((x) => x / temperature);
    const m = Math.max(...z);
    const e = z.map((x) => Math.exp(x - m));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map((x) => x / s);
  }
}
