// Kev runs in a worker so a multi-second load or a long request never blocks the page.
import * as ort from "onnxruntime-web/webgpu";
import wasm from "onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm?url";
import mjs from "onnxruntime-web/ort-wasm-simd-threaded.asyncify.mjs?url";
import { loadKev, type DecisionRecord, type Kev, type OrtModule } from "../src/index.ts";

ort.env.wasm.wasmPaths = { wasm, mjs };
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(8, navigator.hardwareConcurrency || 4) : 1;

export type WorkerRequest =
  | { type: "load"; baseUrl: string; variant: string; device: "webgpu" | "wasm"; verbose?: boolean }
  | { type: "run"; id: number; request: unknown; mode: "packed" | "separate" | "probs"; dateFacts?: boolean };

export type WorkerResponse =
  | { type: "progress"; file: string; loaded: number; total: number }
  | { type: "phase"; phase: "manifest" | "download" | "session" | "warmup" }
  | { type: "ready"; variant: string; device: string; loadMs: number; warmupMs: number; temperature: number }
  | { type: "partial"; id: number; qid: string; answer: unknown; index: number }
  | { type: "result"; id: number; response: unknown }
  | { type: "error"; id?: number; message: string };

let kev: Kev | null = null;
const post = (m: WorkerResponse) => (self as unknown as Worker).postMessage(m);

self.onmessage = async (e: MessageEvent<WorkerRequest>) => {
  const m = e.data;
  try {
    if (m.type === "load") {
      await kev?.release(); kev = null;
      ort.env.logLevel = m.verbose ? "verbose" : "warning";
      const t0 = performance.now();
      kev = await loadKev(m.baseUrl, {
        ort: ort as unknown as OrtModule, variant: m.variant, executionProviders: [m.device],
        onProgress: (p) => post({ type: "progress", ...p }),
        onPhase: (phase) => { if (phase !== "ready") post({ type: "phase", phase }); },
        ...(m.verbose ? { sessionOptions: { logSeverityLevel: 0 as const, logVerbosityLevel: 0 } } : {}),
      });
      const t1 = performance.now();
      post({ type: "phase", phase: "warmup" });
      await kev.systemOne({ state: "warm up", questions: { q: { type: "noul", instructions: "Is this a warm-up?" } } });   // compiles the shaders
      kev.clearCache();
      post({ type: "ready", variant: m.variant, device: m.device, loadMs: t1 - t0, warmupMs: performance.now() - t1, temperature: kev.temperature });
    } else if (m.type === "run") {
      if (!kev) throw new Error("model not loaded");
      const response = m.mode === "separate" ? await kev.systemOneSeparate(m.request, { dateFacts: m.dateFacts })
        : m.mode === "probs" ? await kev.probs(m.request as DecisionRecord)   // raw probabilities for a rendered record (parity checks)
        : await kev.systemOne(m.request, { dateFacts: m.dateFacts, onAnswer: (qid, answer, index) => post({ type: "partial", id: m.id, qid, answer, index }) });
      post({ type: "result", id: m.id, response });
    }
  } catch (err) {
    post({ type: "error", id: m.type === "run" ? m.id : undefined, message: err instanceof Error ? err.message : String(err) });
  }
};
