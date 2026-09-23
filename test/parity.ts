// How close a quantized variant must stay to the PyTorch reference, on any platform. manifest parity is measured on
// the export machine's onnxruntime CPU kernels; int8 kernels differ between arm64, x86-64, WASM and WebGPU, so the same
// record moves a little differently on each (kev-0.8b q8f32, mmlu/test/8047: 0.0967 arm64, 0.0993 x86-64, 0.086
// WebGPU on Metal). The probability bound allows one percentage point for that.
//
// An answer may change only where the reference itself is a near-tie: its probability for its own answer exceeds the
// one for the new answer by at most NEAR_TIE. On x86-64, q8f32 flips readme-ticket's department question, which the
// reference splits 0.442 / 0.421; arm64 does not. A flip on a clear answer fails.
import type { KevManifest } from "../src/index.ts";

export const PLATFORM_SLACK = 0.01;
export const NEAR_TIE = 0.05;

export const argmax = (p: number[]) => p.indexOf(Math.max(...p));

export function maxAbsDp(manifest: KevManifest, variant: string): number {
  if (variant === "fp32") return 1e-4;
  return (manifest.variants[variant].parity?.max_abs_dp ?? 0.1) + PLATFORM_SLACK;
}

/** Compares probabilities with the reference, question by question, and keeps the worst deviation and every flip. */
export class Parity {
  worst = 0;
  worstAt = "";
  questions = 0;
  flips: { at: string; margin: number }[] = [];

  add(at: string, ref: number[], got: number[]) {
    this.questions++;
    const d = Math.max(...got.map((x, j) => Math.abs(x - ref[j])));
    if (d > this.worst) { this.worst = d; this.worstAt = at; }
    const a = argmax(ref), b = argmax(got);
    if (a !== b) this.flips.push({ at, margin: ref[a] - ref[b] });
  }

  /** Flips on a clear reference answer (margin above NEAR_TIE). */
  get clearFlips() { return this.flips.filter((f) => f.margin > NEAR_TIE); }

  summary() {
    const flips = this.flips.map((f) => `${f.at} (margin ${f.margin.toFixed(3)})`).join(", ") || "none";
    return `max |dp| ${this.worst.toExponential(2)} (${this.worstAt}) over ${this.questions} questions; flips: ${flips}`;
  }
}
