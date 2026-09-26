// How close a variant must stay to the PyTorch reference, on any platform. fp32 is exact to 1e-4. A quantized
// variant's error is heavy-tailed and its tail depends on the int8 kernels, which differ by CPU (x86-64 runners with and
// without AVX-512 differ), WASM and WebGPU: kev-0.8b q8f32 moves mmlu/test/8047 by 0.086 (WebGPU, WASM), 0.0967 (arm64,
// the manifest's measurement), 0.0993 and 0.124 (two x86-64 runners), while the mean over questions stays near 0.008.
// So: the mean catches a runtime bug (it shifts many questions), over at least MEAN_MIN_QUESTIONS (a handful of
// questions measures those questions: WebGPU on SwiftShader, two tickets and one near-tie among their six questions,
// has mean 0.018); the maximum is twice the manifest's figure, but not below MAX_ABS_DP_FLOOR — packaging on a lucky
// CPU (ARM measured 0.026 for the 64k rebuild) would otherwise fail x86 CI on the same known heavy-tail question —
// and an answer may change only where the reference itself is a near-tie (its margin between the two answers is at
// most NEAR_TIE; readme-ticket's department question, 0.442 / 0.421, flips on some x86-64 runners).
import type { KevManifest } from "../src/index.ts";

export const NEAR_TIE = 0.05;
export const MEAN_ABS_DP = 0.02;
export const MEAN_MIN_QUESTIONS = 20;
/** Floor for 2× measured max |dp|: covers known x86/WASM/WebGPU tails (~0.12) when packaging got a lucky low figure. */
export const MAX_ABS_DP_FLOOR = 0.25;

export const argmax = (p: number[]) => p.indexOf(Math.max(...p));

/** Bounds for one variant: the largest |dp| of any question, and the mean of those over the questions. */
export function bounds(manifest: KevManifest, variant: string): { max: number; mean: number } {
  if (variant === "fp32") return { max: 1e-4, mean: 1e-4 };
  const measured = manifest.variants[variant].parity?.max_abs_dp ?? 0.1;
  return { max: Math.max(2 * measured, MAX_ABS_DP_FLOOR), mean: MEAN_ABS_DP };
}

/** Compares probabilities with the reference, question by question, and keeps the worst deviation and every flip. */
export class Parity {
  worst = 0;
  worstAt = "";
  questions = 0;
  sum = 0;
  flips: { at: string; margin: number }[] = [];

  add(at: string, ref: number[], got: number[]) {
    this.questions++;
    const d = Math.max(...got.map((x, j) => Math.abs(x - ref[j])));
    this.sum += d;
    if (d > this.worst) { this.worst = d; this.worstAt = at; }
    const a = argmax(ref), b = argmax(got);
    if (a !== b) this.flips.push({ at, margin: ref[a] - ref[b] });
  }

  /** Every bound this comparison breaks, empty when it passes. */
  violations(b: { max: number; mean: number }): string[] {
    const out: string[] = [];
    if (this.worst > b.max) out.push(`max |dp| ${this.worst} at ${this.worstAt} > ${b.max}`);
    if (this.questions >= MEAN_MIN_QUESTIONS && this.mean > b.mean) out.push(`mean |dp| ${this.mean} > ${b.mean} over ${this.questions} questions`);
    for (const f of this.clearFlips) out.push(`answer changed at ${f.at}, reference margin ${f.margin} > ${NEAR_TIE}`);
    return out;
  }

  /** Flips on a clear reference answer (margin above NEAR_TIE). */
  get clearFlips() { return this.flips.filter((f) => f.margin > NEAR_TIE); }

  /** mean over questions of each question's largest |dp| */
  get mean() { return this.questions ? this.sum / this.questions : 0; }

  summary() {
    const flips = this.flips.map((f) => `${f.at} (margin ${f.margin.toFixed(3)})`).join(", ") || "none";
    return `mean |dp| ${this.mean.toExponential(2)}, max ${this.worst.toExponential(2)} (${this.worstAt}) over ${this.questions} questions; flips: ${flips}`;
  }
}
