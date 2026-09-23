// loadKev from local sources (a read function, a directory handle): the bundle is read in place, never fetched.
import "./no-float16.ts";
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import * as ort from "onnxruntime-node";
import { loadKev, modelFiles, type KevManifest, type OrtModule, type Progress, type ReadModelFile } from "../src/index.ts";
import { fixtures, haveModel, modelDir, tokenizerFiles } from "./fixtures.ts";

/** A safetensors file with a 2 x 3 pointer head. */
function headFile(): Uint8Array {
  const t = (off: number, shape: number[]) => ({ dtype: "F32", shape, data_offsets: [off * 4, (off + shape.reduce((a, b) => a * b)) * 4] });
  const header = new TextEncoder().encode(JSON.stringify({ "q.weight": t(0, [2, 3]), "q.bias": t(6, [2]), "k.weight": t(8, [2, 3]), "k.bias": t(14, [2]) }));
  const out = new Uint8Array(8 + header.length + 16 * 4);
  new DataView(out.buffer).setBigUint64(0, BigInt(header.length), true);
  out.set(header, 8);
  return out;
}

/** A bundle as kev_web_export.package lays it out, with placeholder graph and weight bytes. */
async function bundle(): Promise<Map<string, Uint8Array>> {
  const tok = await tokenizerFiles();
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

/** An ort whose InferenceSession.create records what it was given. */
function stubOrt() {
  const created: { graph: Uint8Array; options: { externalData?: { path: string; data: Uint8Array }[] } }[] = [];
  const mod = { InferenceSession: { create: async (graph: Uint8Array, options: object) => { created.push({ graph, options }); return { release: async () => {} }; } } };
  return { ort: mod as unknown as OrtModule, created };
}

/** A FileSystemDirectoryHandle over a path -> bytes map, the parts loadKev uses: nested directories, files as Blobs. */
function directory(files: Map<string, Uint8Array>, prefix = ""): FileSystemDirectoryHandle {
  const missing = (name: string) => Object.assign(new Error(`${prefix}${name} not found`), { name: "NotFoundError" });
  return {
    kind: "directory", name: prefix,
    async getDirectoryHandle(name: string) {
      if (![...files.keys()].some((p) => p.startsWith(`${prefix}${name}/`))) throw missing(name);
      return directory(files, `${prefix}${name}/`);
    },
    async getFileHandle(name: string) {
      const bytes = files.get(`${prefix}${name}`);
      if (!bytes) throw missing(name);
      return { kind: "file", name, getFile: async () => new Blob([bytes as BlobPart]) };
    },
  } as unknown as FileSystemDirectoryHandle;
}

async function withoutFetch<T>(fn: () => Promise<T>): Promise<T> {
  const real = globalThis.fetch;
  globalThis.fetch = (async (input: RequestInfo | URL) => { throw new Error(`unexpected fetch ${String(input)}`); }) as typeof fetch;
  try { return await fn(); } finally { globalThis.fetch = real; }
}

test("loadKev reads every file from a read function, by bundle path, without fetch", async () => {
  const files = await bundle();
  const reads: string[] = [];
  // a VFS hands back views into larger buffers: only the view's bytes may be used
  const read: ReadModelFile = async (p) => {
    reads.push(p);
    const b = files.get(p);
    if (!b) throw new Error(`no ${p}`);
    const padded = new Uint8Array(b.length + 8); padded.set(b, 4);
    return padded.subarray(4, 4 + b.length);
  };
  const { ort, created } = stubOrt();
  const progress: Progress[] = [];
  const kev = await withoutFetch(() => loadKev(read, { ort, variant: "v", executionProviders: ["cpu"], onProgress: (p) => progress.push(p) }));
  assert.deepEqual(reads.sort(), [...files.keys()].sort());
  assert.equal(new TextDecoder().decode(created[0].graph), "graph");
  assert.deepEqual(created[0].options.externalData!.map((x) => [x.path, new TextDecoder().decode(x.data)]),
    [["model.onnx.data_0", "shard zero"], ["model.onnx.data_1", "shard one!"]]);
  assert.equal(kev.manifest.run, "jaredpalmer/kev-test@abc");
  const done = progress.filter((p) => p.total > 0 && p.loaded === p.total).map((p) => p.file);
  for (const { path } of modelFiles(kev.manifest, "v")) assert.ok(done.includes(path), `no progress for ${path}`);
});

test("loadKev reads a bundle from a directory handle (OPFS layout)", async () => {
  const files = await bundle();
  const { ort, created } = stubOrt();
  await withoutFetch(() => loadKev(directory(files), { ort, variant: "v", executionProviders: ["cpu"] }));
  assert.equal(created[0].options.externalData!.length, 2);
});

test("a short local file is reported by name", async () => {
  const files = await bundle();
  files.set("r-abc/v/model.onnx.data_1", new TextEncoder().encode("shard"));
  const { ort } = stubOrt();
  await assert.rejects(loadKev(directory(files), { ort, variant: "v" }), /model\.onnx\.data_1: expected 10 bytes, read 5/);
});

test("modelFiles lists what a variant loads, with sizes", async () => {
  const files = await bundle();
  const manifest = JSON.parse(new TextDecoder().decode(files.get("manifest.json"))) as KevManifest;
  files.delete("manifest.json");
  assert.deepEqual(modelFiles(manifest), [...files].map(([path, b]) => ({ path, bytes: b.length })));
  assert.throws(() => modelFiles(manifest, "nope"), /unknown variant nope/);
});

test("q8f32 loaded from local files matches the PyTorch reference", { skip: !haveModel && "no bundle" }, async () => {
  const kev = await withoutFetch(() => loadKev((p) => readFile(`${modelDir}/${p}`), { ort: ort as unknown as OrtModule, variant: "q8f32", executionProviders: ["cpu"], temperature: 1 }));
  const bound = (kev.manifest.variants.q8f32.parity?.max_abs_dp ?? 0.1) + 1e-3;
  for (const f of fixtures.slice(0, 3)) {
    const probs = await kev.probs(f.record);
    probs.forEach((p, k) => p.forEach((x, j) => assert.ok(Math.abs(x - f.probs[k][j]) <= bound, `${f.name} q${k} opt${j}`)));
  }
  await kev.release();
});
