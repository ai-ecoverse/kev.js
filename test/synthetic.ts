// A synthetic model bundle for loader tests: the layout kev_web_export.package writes, with a real tokenizer, a tiny
// pointer head and placeholder graph and weight bytes. Shared by the Node tests and the browser harness, so it touches
// no Node APIs.
import type { KevManifest, OrtModule } from "../src/index.ts";

/** A safetensors file with a 2 x 3 pointer head. */
export function headFile(): Uint8Array {
  const t = (off: number, shape: number[]) => ({ dtype: "F32", shape, data_offsets: [off * 4, (off + shape.reduce((a, b) => a * b)) * 4] });
  const header = new TextEncoder().encode(JSON.stringify({ "q.weight": t(0, [2, 3]), "q.bias": t(6, [2]), "k.weight": t(8, [2, 3]), "k.bias": t(14, [2]) }));
  const out = new Uint8Array(8 + header.length + 16 * 4);
  new DataView(out.buffer).setBigUint64(0, BigInt(header.length), true);
  out.set(header, 8);
  return out;
}

/** bundle path -> bytes, manifest.json included. */
export function syntheticBundle(tok: { tokenizer: string; tokenizer_config: string }): Map<string, Uint8Array> {
  const enc = (s: string) => new TextEncoder().encode(s);
  const files = new Map<string, Uint8Array>([
    ["r-abc/tokenizer.json", enc(tok.tokenizer)], ["r-abc/tokenizer_config.json", enc(tok.tokenizer_config)],
    ["r-abc/head.safetensors", headFile()], ["r-abc/v/model.onnx", enc("graph")],
    ["r-abc/v/model.onnx.data_0", enc("shard zero")], ["r-abc/v/model.onnx.data_1", enc("shard one!")],
  ]);
  const sizes = Object.fromEntries([...files].map(([p, b]) => [p, b.length]));
  const manifest: Partial<KevManifest> = {
    name: "kev-test", run: "jaredpalmer/kev-test@abc", special: { state: 1, q: 2, opt: 3, opt_end: 4, decide: 5 } as KevManifest["special"],
    files: { head: "r-abc/head.safetensors", tokenizer: "r-abc/tokenizer.json", tokenizer_config: "r-abc/tokenizer_config.json" },
    variants: { v: { model: "r-abc/v/model.onnx", data: ["r-abc/v/model.onnx.data_0", "r-abc/v/model.onnx.data_1"], bytes: 20, io_dtype: "float32", sizes, inputs: [], outputs: [] } },
  };
  files.set("manifest.json", enc(JSON.stringify(manifest)));
  return files;
}

export interface Created { graph: Uint8Array; options: { externalData?: { path: string; data: Uint8Array }[] } }

/** An ort whose InferenceSession.create records what it was given. */
export function stubOrt(): { ort: OrtModule; created: Created[] } {
  const created: Created[] = [];
  const mod = { InferenceSession: { create: async (graph: Uint8Array, options: Created["options"]) => { created.push({ graph, options }); return { release: async () => {} }; } } };
  return { ort: mod as unknown as OrtModule, created };
}

/** Run fn with fetch replaced by one that throws: a local source must not touch the network. */
export async function withoutFetch<T>(fn: () => Promise<T>): Promise<T> {
  const real = globalThis.fetch;
  globalThis.fetch = (async (input: RequestInfo | URL) => { throw new Error(`unexpected fetch ${String(input)}`); }) as typeof fetch;
  try { return await fn(); } finally { globalThis.fetch = real; }
}
