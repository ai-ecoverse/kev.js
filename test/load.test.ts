// loadKev from local sources (a read function, a directory handle): the bundle is read in place, never fetched.
import "./no-float16.ts";
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import * as ort from "onnxruntime-node";
import { loadKev, modelFiles, type KevManifest, type OrtModule, type Progress, type ReadModelFile } from "../src/index.ts";
import { fixtures, haveModel, modelDir, tokenizerFiles } from "./fixtures.ts";
import { bounds, Parity, MAX_ABS_DP_FLOOR } from "./parity.ts";
import { stubOrt, syntheticBundle, withoutFetch } from "./synthetic.ts";

async function bundle() {
  return syntheticBundle(await tokenizerFiles());
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
  const parity = new Parity();
  for (const f of fixtures.slice(0, 3)) (await kev.probs(f.record)).forEach((p, k) => parity.add(`${f.name} q${k}`, f.probs[k], p));
  console.log(`local files, q8f32: ${parity.summary()}`);
  assert.deepEqual(parity.violations(bounds(kev.manifest, "q8f32")), []);
  await kev.release();
});

test("the parity rule tolerates a flip on a near-tie and fails one on a clear answer", () => {
  const p = new Parity();
  p.add("near-tie", [0.442, 0.421, 0.137], [0.43, 0.44, 0.13]);
  p.add("clear", [0.7, 0.3], [0.45, 0.55]);
  p.add("same", [0.9, 0.1], [0.85, 0.15]);
  assert.deepEqual(p.flips.map((f) => f.at), ["near-tie", "clear"]);
  assert.deepEqual(p.clearFlips.map((f) => f.at), ["clear"]);
  assert.equal(p.worstAt, "clear");
  assert.equal(p.violations({ max: 1, mean: 1 }).length, 1);                        // the clear flip
  assert.equal(p.violations({ max: 0.2, mean: 0.01 }).length, 2);                   // flip and max 0.25; 3 questions: no mean verdict
});

test("the parity rule catches a runtime-wide shift through the mean, over enough questions", () => {
  const p = new Parity();
  for (let i = 0; i < 6; i++) p.add(`q${i}`, [0.8, 0.2], [0.77, 0.23]);           // 0.03 everywhere, no flip, small max
  assert.deepEqual(p.violations({ max: 0.19, mean: 0.02 }), []);                    // six questions: no mean verdict
  for (let i = 6; i < 20; i++) p.add(`q${i}`, [0.8, 0.2], [0.77, 0.23]);
  assert.deepEqual(p.violations({ max: 0.19, mean: 0.02 }).map((v) => v.split(" ")[0]), ["mean"]);
});

test("bounds floors a lucky packaging measurement so x86 CI still fits known tails", () => {
  const low = { variants: { q8f32: { parity: { max_abs_dp: 0.026387 } } } } as unknown as import("../src/index.ts").KevManifest;
  const high = { variants: { q8f32: { parity: { max_abs_dp: 0.2 } } } } as unknown as import("../src/index.ts").KevManifest;
  assert.equal(bounds(low, "q8f32").max, MAX_ABS_DP_FLOOR);                         // 2×0.026 < floor
  assert.equal(bounds(high, "q8f32").max, 0.4);                                     // 2×0.2 > floor
  assert.deepEqual(bounds(low, "fp32"), { max: 1e-4, mean: 1e-4 });
});

/** An in-memory Cache Storage (the parts loadKev uses) that logs deletions into `events`. */
function memoryCaches(events: string[]) {
  const stores = new Map<string, Map<string, Response>>();
  const cache = (m: Map<string, Response>) => ({
    match: async (key: string) => m.get(key)?.clone(),
    put: async (key: string, res: Response) => { m.set(key, res); },
    delete: async (req: Request | string) => { const k = typeof req === "string" ? req : req.url; events.push(`evict ${k}`); return m.delete(k); },
    keys: async () => [...m.keys()].map((url) => ({ url }) as Request),
  });
  return { open: async (name: string) => cache(stores.get(name) ?? stores.set(name, new Map()).get(name)!) } as unknown as CacheStorage;
}

test("a bundle rebuilt for the same checkpoint is fetched again, and the old copy is evicted before the download", async () => {
  const base = "https://models.test/kev-test";
  const files = await bundle();
  const manifest = JSON.parse(new TextDecoder().decode(files.get("manifest.json"))) as KevManifest;
  const serve = (m: KevManifest, graph: string) => {
    files.set("manifest.json", new TextEncoder().encode(JSON.stringify(m)));
    files.set("r-abc/v/model.onnx", new TextEncoder().encode(graph));
  };
  const events: string[] = [];
  const realFetch = globalThis.fetch, realCaches = (globalThis as { caches?: CacheStorage }).caches;
  globalThis.fetch = (async (input: RequestInfo | URL) => {
    const path = String(input).slice(base.length + 1);
    events.push(`fetch ${path}`);
    const b = files.get(path);
    return b ? new Response(b as BodyInit) : new Response(null, { status: 404 });
  }) as typeof fetch;
  (globalThis as { caches?: CacheStorage }).caches = memoryCaches(events);
  try {
    // same checkpoint (run), same file sizes, different bytes: only the content revision tells them apart
    serve({ ...manifest, revision: "aaaa" }, "graph");
    const first = stubOrt();
    await loadKev(base, { ort: first.ort, variant: "v", executionProviders: ["cpu"] });
    assert.equal(new TextDecoder().decode(first.created[0].graph), "graph");
    events.length = 0;
    serve({ ...manifest, revision: "bbbb" }, "GRAPH");
    const second = stubOrt();
    await loadKev(base, { ort: second.ort, variant: "v", executionProviders: ["cpu"] });
    assert.equal(new TextDecoder().decode(second.created[0].graph), "GRAPH", "the rebuilt graph, not the cached one");
    const evictions = events.flatMap((e, i) => (e.startsWith("evict") ? [i] : []));
    const firstFileFetch = events.findIndex((e) => e.startsWith("fetch r-abc/"));
    assert.equal(evictions.length, 6, "every file of the old revision is evicted");
    assert.ok(evictions.every((i) => i < firstFileFetch), `evicted before the download: ${events.join(", ")}`);
    assert.ok(events.filter((e) => e.startsWith("evict")).every((e) => e.includes("kev-rev=aaaa")));
  } finally {
    globalThis.fetch = realFetch;
    (globalThis as { caches?: CacheStorage }).caches = realCaches;
  }
});
