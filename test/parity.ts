// How close a quantized variant must stay to the PyTorch reference, on any platform. manifest parity is measured on
// the export machine's onnxruntime CPU kernels; int8 kernels differ between arm64, x86-64, WASM and WebGPU, so the same
// record moves a little differently on each (kev-0.8b q8f32, mmlu/test/8047: 0.0967 arm64, 0.0993 x86-64, 0.086
// WebGPU on Metal). The probability bound allows one percentage point for that; the answers may not change more
// often than at export (argmax flips), which is what a caller sees.
import type { KevManifest } from "../src/index.ts";

export const PLATFORM_SLACK = 0.01;

export function parityBound(manifest: KevManifest, variant: string): { maxAbsDp: number; argmaxFlips: number } {
  if (variant === "fp32") return { maxAbsDp: 1e-4, argmaxFlips: 0 };
  const p = manifest.variants[variant].parity;
  return { maxAbsDp: (p?.max_abs_dp ?? 0.1) + PLATFORM_SLACK, argmaxFlips: p?.argmax_flips ?? 0 };
}

export const argmax = (p: number[]) => p.indexOf(Math.max(...p));
