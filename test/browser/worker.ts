// The browser cases, run in a module worker against the real OPFS and, for the model cases, the real onnxruntime-web.
import * as ort from "onnxruntime-web/webgpu";
import wasm from "onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm?url";
import mjs from "onnxruntime-web/ort-wasm-simd-threaded.asyncify.mjs?url";
import { loadKev, modelFiles, type KevManifest, type OrtModule } from "../../src/index.ts";
import type { Fixture } from "../fixtures.ts";
import { maxAbsDp, Parity } from "../parity.ts";
import { stubOrt, syntheticBundle, withoutFetch } from "../synthetic.ts";

ort.env.wasm.wasmPaths = { wasm, mjs };
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;
ort.env.logLevel = "warning";

const HF = "https://huggingface.co/ai-ecoverse/kev.js/resolve/main";
const post = (m: unknown) => (self as unknown as Worker).postMessage(m);
const say = (line: string) => post({ log: line });
const text = (b: Uint8Array) => new TextDecoder().decode(b);

// ---- OPFS helpers -------------------------------------------------------------------------------------------------

/** A fresh directory at the OPFS root. */
async function freshDir(name: string): Promise<FileSystemDirectoryHandle> {
  const root = await navigator.storage.getDirectory();
  await root.removeEntry(name, { recursive: true }).catch(() => {});
  return root.getDirectoryHandle(name, { create: true });
}

async function fileHandle(dir: FileSystemDirectoryHandle, path: string, create = false): Promise<FileSystemFileHandle> {
  const parts = path.split("/");
  for (const part of parts.slice(0, -1)) dir = await dir.getDirectoryHandle(part, { create });
  return dir.getFileHandle(parts[parts.length - 1], { create });
}

async function writeFile(dir: FileSystemDirectoryHandle, path: string, data: Uint8Array | ReadableStream<Uint8Array>) {
  const w = await (await fileHandle(dir, path, true)).createWritable();
  if (data instanceof Uint8Array) { await w.write(data as Uint8Array<ArrayBuffer>); await w.close(); } else await data.pipeTo(w);   // pipeTo closes w
}

async function sizeOf(dir: FileSystemDirectoryHandle, path: string): Promise<number> {
  try { return (await (await fileHandle(dir, path)).getFile()).size; } catch { return -1; }
}

// ---- cases --------------------------------------------------------------------------------------------------------

/** Tokenizer texts for the synthetic bundle: the local bundle's when served, else the published copy. */
async function tokenizerTexts(): Promise<{ tokenizer: string; tokenizer_config: string }> {
  let base = "/models/kev-0.8b";
  let res = await fetch(`${base}/manifest.json`);
  if (!res.ok) { base = `${HF}/kev-0.8b`; res = await fetch(`${base}/manifest.json`); }
  const m = (await res.json()) as KevManifest;
  const get = async (p: string) => (await fetch(`${base}/${p}`)).text();
  return { tokenizer: await get(m.files.tokenizer), tokenizer_config: await get(m.files.tokenizer_config) };
}

async function syntheticInOpfs(name: string, edit?: (files: Map<string, Uint8Array>) => void) {
  const files = syntheticBundle(await tokenizerTexts());
  edit?.(files);
  const dir = await freshDir(name);
  for (const [path, bytes] of files) await writeFile(dir, path, bytes);
  return { dir, files };
}

/** What a stub session was created from: the graph and each external-data entry, as text. */
function created(c: ReturnType<typeof stubOrt>["created"]) {
  return { graph: text(c[0].graph), externalData: c[0].options.externalData!.map((x) => [x.path, text(x.data)]) };
}

const cases = {
  /** loadKev(FileSystemDirectoryHandle) over the real OPFS. */
  async "opfs-directory"() {
    const { dir } = await syntheticInOpfs("kev-test-directory");
    const { ort: stub, created: c } = stubOrt();
    const kev = await withoutFetch(() => loadKev(dir, { ort: stub, variant: "v", executionProviders: ["wasm"] }));
    return { run: kev.manifest.run, ...created(c) };
  },

  /** loadKev(read function) where the function returns OPFS File objects, as a VFS over OPFS would. */
  async "opfs-function"() {
    const { dir } = await syntheticInOpfs("kev-test-function");
    const reads: string[] = [];
    const read = async (path: string) => { reads.push(path); return (await fileHandle(dir, path)).getFile(); };
    const { ort: stub, created: c } = stubOrt();
    const kev = await withoutFetch(() => loadKev(read, { ort: stub, variant: "v", executionProviders: ["wasm"] }));
    return { run: kev.manifest.run, reads: reads.sort(), expected: ["manifest.json", ...modelFiles(kev.manifest, "v").map((f) => f.path)].sort(), ...created(c) };
  },

  /** A file cut short in OPFS (an interrupted download) fails the load with its name. */
  async "opfs-short"() {
    const { dir } = await syntheticInOpfs("kev-test-short", (files) => files.set("r-abc/v/model.onnx.data_1", new TextEncoder().encode("shard")));
    const { ort: stub } = stubOrt();
    try {
      await withoutFetch(() => loadKev(dir, { ort: stub, variant: "v" }));
      return { error: null };
    } catch (e) {
      return { error: e instanceof Error ? e.message : String(e) };
    }
  },

  /** The published model, copied into OPFS, loaded from there on onnxruntime-web and compared with PyTorch. */
  async model(o: { model?: string; variant?: string; ep?: "wasm" | "webgpu"; limit?: number }) {
    const model = o.model ?? "kev-0.8b", variant = o.variant ?? "q8f32", ep = o.ep ?? "wasm";
    const res = await fetch(`/models/${model}/manifest.json`);
    if (!res.ok) return { skip: `no bundle at public/models/${model} (node --import tsx scripts/fetch-model.ts ${model} ${variant})` };
    const manifestText = await res.text();
    const manifest = JSON.parse(manifestText) as KevManifest;
    let adapter: string | null = null;
    if (ep === "webgpu") {
      const a = await (navigator as Navigator & { gpu?: { requestAdapter(): Promise<{ info?: Record<string, string> } | null> } }).gpu?.requestAdapter();
      if (!a) return { skip: "no WebGPU adapter in this browser" };
      adapter = [a.info?.vendor, a.info?.architecture, a.info?.description].filter(Boolean).join(" ") || "unknown";
    }

    // copy into OPFS the way a download step would: modelFiles, skipping files already at their size, manifest last
    const dir = await (await navigator.storage.getDirectory()).getDirectoryHandle(`kev-model-${model}`, { create: true });
    const t0 = performance.now();
    let copied = 0;
    for (const { path, bytes } of modelFiles(manifest, variant)) {
      if (bytes && (await sizeOf(dir, path)) === bytes) continue;
      const r = await fetch(`/models/${model}/${path}`);
      if (!r.ok || !r.body) throw new Error(`/models/${model}/${path}: HTTP ${r.status}`);
      await writeFile(dir, path, r.body);
      copied++;
    }
    await writeFile(dir, "manifest.json", new TextEncoder().encode(manifestText));
    say(`${model} ${variant}: ${copied} files copied into OPFS in ${Math.round(performance.now() - t0)} ms`);

    // load from OPFS, recording every fetch: model files must not be among them (onnxruntime may fetch its own wasm)
    const fetched: string[] = [];
    const realFetch = globalThis.fetch;
    globalThis.fetch = ((input: RequestInfo | URL, init?: RequestInit) => {
      fetched.push(typeof input === "string" ? input : input instanceof URL ? input.href : input.url);
      return realFetch(input, init);
    }) as typeof fetch;
    const t1 = performance.now();
    let kev;
    try {
      kev = await loadKev(dir, { ort: ort as unknown as OrtModule, variant, executionProviders: [ep], temperature: 1 });
    } finally {
      globalThis.fetch = realFetch;
    }
    const loadMs = performance.now() - t1;
    say(`${model} ${variant}: loaded from OPFS on ${ep}${adapter ? ` (${adapter})` : ""} in ${Math.round(loadMs)} ms`);

    const fx = (await (await fetch(`/fixtures/${model}.json`)).json()) as { run: string; fixtures: Fixture[] };
    const parity = new Parity();
    const t2 = performance.now();
    const sample = fx.fixtures.slice(0, o.limit ?? fx.fixtures.length);
    for (const f of sample) {
      (await kev.probs(f.record)).forEach((p, k) => parity.add(`${f.name} q${k}`, f.probs[k], p));
    }
    const msPerFixture = (performance.now() - t2) / sample.length;
    const answers = (await kev.systemOne(sample[0].request)).answers;
    await kev.release();
    return {
      ep, adapter, run: manifest.run, fixtureRun: fx.run, fixtures: sample.length, summary: parity.summary(),
      worst: parity.worst, worstAt: parity.worstAt, clearFlips: parity.clearFlips, bound: maxAbsDp(manifest, variant),
      answerKeys: Object.keys(answers), expectedAnswerKeys: Object.keys(sample[0].answers),
      modelFetches: fetched.filter((u) => u.includes("/models/")), caches: await caches.keys(),
      loadMs: Math.round(loadMs), msPerFixture: Math.round(msPerFixture),
    };
  },
};

export type CaseName = keyof typeof cases;
export type CaseResult = Record<string, unknown>;

self.onmessage = async (e: MessageEvent<{ id: number; name: CaseName; options: Record<string, unknown> }>) => {
  const { id, name, options } = e.data;
  try {
    post({ id, result: await (cases[name] as (o: Record<string, unknown>) => Promise<CaseResult>)(options) });
  } catch (err) {
    post({ id, error: err instanceof Error ? `${err.message}\n${err.stack ?? ""}` : String(err) });
  }
};
