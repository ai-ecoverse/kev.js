// Load a packaged Kev model (kev_web_export.package output): manifest, tokenizer, head, ONNX graph and its external
// weights. From a URL, weight files (0.8 GB for kev-0.8b q8) are kept in Cache Storage when available and streamed
// with progress. A model already on disk (OPFS, a picked directory, a virtual file system) is read in place.

import { Tokenizer } from "@huggingface/tokenizers";
import type { InferenceSession } from "onnxruntime-common";
import { PointerHead } from "./head.ts";
import { Kev, type KevManifest, type KevOptions, type OrtModule } from "./model.ts";

export interface Progress { file: string; loaded: number; total: number }

/** Reads one file of a packaged model by its path in the bundle ("manifest.json", "r-<sha>/q8f32/model.onnx"). */
export type ReadModelFile = (path: string) => Promise<Uint8Array | ArrayBuffer | Blob>;

/** Where loadKev reads a model: a base URL (fetched, kept in Cache Storage), a directory holding the bundle (OPFS via
 * navigator.storage.getDirectory(), or showDirectoryPicker()), or a function that reads a file by its bundle path. The
 * last two are read in place: no fetch, no Cache Storage. */
export type ModelSource = string | FileSystemDirectoryHandle | ReadModelFile;

/** Coarse stage of loadKev, for a status line that keeps moving after the download finishes. */
export type LoadPhase = "manifest" | "download" | "session" | "ready";

/** Reads `length` bytes of one file of a packaged model at `offset`, synchronously, by its path in the bundle: an OPFS
 * sync access handle in a worker, or a virtual file system with sync reads. */
export type ReadModelFileSync = (path: string, offset: number, length: number) => Uint8Array;

export interface LoadOptions extends KevOptions {
  ort: OrtModule;
  /** variant in manifest.variants; default: the first one listed */
  variant?: string;
  /** e.g. ["webgpu"] or ["wasm"]; default ["webgpu", "wasm"] */
  executionProviders?: InferenceSession.SessionOptions["executionProviders"];
  onProgress?: (p: Progress) => void;
  onPhase?: (phase: LoadPhase) => void;
  /** Cache Storage bucket for model files; null disables caching. Default "kev-web-v1". Only URL sources are cached. */
  cacheName?: string | null;
  sessionOptions?: InferenceSession.SessionOptions;
  /** Load the vision tower of a bundle that has one (manifest.vision), so requests may carry an image. Default true;
   * false loads only the decoder, which then answers text requests exactly as the text bundle does. */
  vision?: boolean;
  /** weight files fetched at once. Default 2: on a bandwidth-limited link more streams are slower in aggregate
   * (measured through a tunnel: 1.2 MB/s with one stream, 0.75 MB/s across six). */
  concurrency?: number;
  /** The file system path of a bundle file, for an ONNX Runtime that opens files itself (onnxruntime-node). Given one,
   * the decoder and the vision tower are created from their graphs' paths and ORT reads their weights from the files
   * beside them, so loadKev never holds the weights: peak memory stays near the model's size instead of twice it. */
  filePath?: (path: string) => string;
  /** Reads part of a weight file synchronously (onnxruntime-web, local sources). onnxruntime-web asks for each weight's
   * bytes while it builds the session, in no particular order of files; with this, loadKev reads just those bytes when
   * ORT asks, instead of every weight file before the session starts, so one tensor at a time is held outside ORT.
   * Needs the manifest's file sizes. */
  readSync?: ReadModelFileSync;
}

async function readWithProgress(res: Response, file: string, onProgress?: (p: Progress) => void, expected = 0): Promise<Uint8Array> {
  const total = expected || Number(res.headers.get("content-length") ?? 0);
  if (!res.body || !onProgress) return new Uint8Array(await res.arrayBuffer());
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); loaded += value.length;
    onProgress({ file, loaded, total: total || loaded });
  }
  if (chunks.length === 1) return chunks[0];
  const out = new Uint8Array(loaded);
  let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out;
}

/** The Cache Storage key for one file of one bundle revision. Weights are republished under the same URLs with the
 * same sizes, so the key carries the revision: a new or rebuilt bundle never reads an old one's cached bytes. */
const cacheKey = (url: string, rev?: string) => (rev ? `${url}${url.includes("?") ? "&" : "?"}kev-rev=${encodeURIComponent(rev)}` : url);

export async function fetchFile(url: string, o: { cacheName?: string | null; onProgress?: (p: Progress) => void; file?: string; bytes?: number; rev?: string } = {}): Promise<Uint8Array> {
  const file = o.file ?? url;
  const cache = o.cacheName !== null && typeof caches !== "undefined" ? await caches.open(o.cacheName ?? "kev-web-v1") : null;
  const key = cacheKey(url, o.rev);
  const hit = await cache?.match(key);
  if (hit) {
    const data = await readWithProgress(hit, file, o.onProgress, o.bytes);
    if (!o.bytes || data.length === o.bytes) return data;
    await cache!.delete(key);   // truncated entry (e.g. a reload mid-download): fetch it again
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  // read first and cache the bytes afterwards: cache.put(res.clone()) would download the whole body before it
  // resolves, so a multi-hundred-MB file would report no progress at all
  const data = await readWithProgress(res, file, o.onProgress, o.bytes);
  if (o.bytes && data.length !== o.bytes) throw new Error(`${file}: expected ${o.bytes} bytes, received ${data.length}`);
  if (cache) {
    try {
      await cache.put(key, new Response(data as BodyInit, { headers: { "content-type": res.headers.get("content-type") ?? "application/octet-stream", "content-length": String(data.length) } }));
    } catch { /* quota: run uncached */ }
  }
  return data;
}

/** A ReadModelFile over a directory that holds the bundle as published (manifest.json at its root). */
export function directoryReader(dir: FileSystemDirectoryHandle): ReadModelFile {
  return async (path) => {
    const parts = path.split("/").filter(Boolean);
    let d = dir;
    for (const part of parts.slice(0, -1)) d = await d.getDirectoryHandle(part);
    return (await d.getFileHandle(parts[parts.length - 1])).getFile();
  };
}

/** Every file a variant loads besides manifest.json, with its size when the manifest records one. A download or resume
 * step can fetch exactly these, and skip those already at their size. A vision bundle adds its tower's files unless
 * `vision` is false. */
export function modelFiles(manifest: KevManifest, variant = Object.keys(manifest.variants)[0], vision = true): { path: string; bytes?: number }[] {
  const v = manifest.variants[variant];
  if (!v) throw new Error(`unknown variant ${variant}; have ${Object.keys(manifest.variants).join(", ")}`);
  const files = [manifest.files.tokenizer, manifest.files.tokenizer_config, manifest.files.head, v.model, ...v.data]
    .map((path) => ({ path, bytes: v.sizes?.[path] }));
  const t = vision ? manifest.vision : undefined;
  return t ? [...files, ...[t.model, ...t.data].map((path) => ({ path, bytes: t.sizes?.[path] }))] : files;
}

/** One file from a local source, as bytes. Views and buffers may come from another realm (a VFS over postMessage), so
 * they are recognised by shape rather than instanceof. */
async function readLocal(read: ReadModelFile, file: string, o: { onProgress?: (p: Progress) => void; bytes?: number } = {}): Promise<Uint8Array> {
  const got = await read(file);
  const data = ArrayBuffer.isView(got) ? new Uint8Array(got.buffer, got.byteOffset, got.byteLength)
    : typeof (got as Blob).arrayBuffer === "function" ? new Uint8Array(await (got as Blob).arrayBuffer())
    : new Uint8Array(got as ArrayBuffer);
  if (o.bytes && data.length !== o.bytes) throw new Error(`${file}: expected ${o.bytes} bytes, read ${data.length} (incomplete download?)`);
  o.onProgress?.({ file, loaded: data.length, total: data.length });
  return data;
}

const join = (base: string, path: string) => `${base.replace(/\/$/, "")}/${path}`;

/** Delete this model's cached files from other revisions: a superseded bundle is gigabytes of quota. */
async function dropOtherRevisions(baseUrl: string, rev: string, cacheName?: string | null) {
  if (cacheName === null || typeof caches === "undefined") return;
  const cache = await caches.open(cacheName ?? "kev-web-v1");
  const prefix = `${baseUrl.replace(/\/$/, "")}/`;
  for (const req of await cache.keys()) {
    if (!req.url.startsWith(prefix)) continue;
    const got = new URL(req.url).searchParams.get("kev-rev");
    if (got !== rev) await cache.delete(req);
  }
}

/** Run jobs with a bounded number in flight: a model is split into many shards, and hundreds of parallel
 * requests are slower than a handful and make progress jumpy. */
async function pool<T>(jobs: (() => Promise<T>)[], limit: number): Promise<T[]> {
  const out = new Array<T>(jobs.length);
  let next = 0;
  await Promise.all(Array.from({ length: Math.min(limit, jobs.length) }, async () => {
    for (let i = next++; i < jobs.length; i = next++) out[i] = await jobs[i]();
  }));
  return out;
}

/** A weight file onnxruntime-web reads on demand. It keeps each mounted file as given and only takes its byteLength and
 * subarray(start, end) while it creates the session, so a Uint8Array that reads those bytes at each subarray stands in
 * for the file. ORT swallows what subarray throws, so `read` records its errors for loadKev to rethrow. */
class LazyFile extends Uint8Array {
  constructor(private readonly size: number, private readonly read: (start: number, end: number) => Uint8Array) {
    super(0);
  }
  override get byteLength(): number { return this.size; }
  override get length(): number { return this.size; }
  override subarray(start = 0, end = this.size): Uint8Array<ArrayBuffer> {
    const got = this.read(start, end);
    return new Uint8Array(got.buffer, got.byteOffset, got.byteLength) as Uint8Array<ArrayBuffer>;
  }
}

export async function loadKev(source: ModelSource, o: LoadOptions): Promise<Kev> {
  const baseUrl = typeof source === "string" ? source : null;
  const read = baseUrl !== null ? null : typeof source === "function" ? source : directoryReader(source as FileSystemDirectoryHandle);
  o.onPhase?.("manifest");
  const decode = (b: Uint8Array) => new TextDecoder().decode(b);
  // the manifest is always fetched fresh: it names the revision everything else is cached under
  const manifest = JSON.parse(decode(read ? await readLocal(read, "manifest.json")
    : await fetchFile(join(baseUrl!, "manifest.json"), { cacheName: null }))) as KevManifest;
  const variant = o.variant ?? Object.keys(manifest.variants)[0];
  const v = manifest.variants[variant];
  if (!v) throw new Error(`unknown variant ${variant}; have ${Object.keys(manifest.variants).join(", ")}`);
  const tower = o.vision === false ? undefined : manifest.vision;
  const sizes = { ...v.sizes, ...tower?.sizes };
  // the content digest when the manifest has one: a bundle rebuilt for the same checkpoint (a spliced decoder at the
  // text graph's URL) must not be read from the old bytes, which `run` alone would allow
  const rev = manifest.revision ?? manifest.run;
  // before fetching: a quota that fits one bundle but not two would otherwise refuse the new files and keep the old
  if (baseUrl !== null) await dropOtherRevisions(baseUrl, rev, o.cacheName);
  o.onPhase?.("download");
  // announce every file up front so the total does not grow as downloads start
  for (const [p, bytes] of Object.entries(sizes)) o.onProgress?.({ file: p, loaded: 0, total: bytes });
  const get = (p: string) => read ? readLocal(read, p, { onProgress: o.onProgress, bytes: sizes[p] })
    : fetchFile(join(baseUrl!, p), { cacheName: o.cacheName, onProgress: o.onProgress, file: p, bytes: sizes[p], rev });
  const [tokJson, tokCfg] = (await Promise.all([get(manifest.files.tokenizer), get(manifest.files.tokenizer_config)])).map(decode);
  // onnxruntime-node opens a graph by path and reads its weights itself; onnxruntime-web reads lazy weights on demand
  const byPath = read ? o.filePath : undefined;
  const readSync = read && !byPath ? o.readSync : undefined;
  let failure: Error | undefined;
  const failed = (e: unknown) => (failure ??= e instanceof Error ? e : new Error(String(e)));
  const lazy = (p: string) => {
    let loaded = 0;
    return new LazyFile(sizes[p], (start, end) => {
      let got: Uint8Array;
      try {
        got = readSync!(p, start, end - start);
      } catch (e) {
        throw failed(e);
      }
      if (got.byteLength !== end - start) throw failed(new Error(`${p}: expected ${sizes[p]} bytes, but it ends before byte ${end} (incomplete download?)`));
      loaded += got.byteLength;
      o.onProgress?.({ file: p, loaded, total: sizes[p] });
      return got;
    });
  };
  const graphs = new Set([v.model, ...(tower ? [tower.model] : [])]);
  const weights = new Set([...v.data, ...(tower?.data ?? [])]);
  const skipped = (p: string) => { if (sizes[p]) o.onProgress?.({ file: p, loaded: sizes[p], total: sizes[p] }); };
  const load = async (p: string): Promise<Uint8Array | string> => {
    if (byPath && graphs.has(p)) { skipped(p); return byPath(p); }
    if (!weights.has(p) || !(byPath || (readSync && sizes[p]))) return get(p);
    if (readSync) return lazy(p);
    skipped(p);
    return "";
  };
  const towerFiles = tower ? [tower.model, ...tower.data] : [];
  const [head, graph, ...rest] = await pool([manifest.files.head, v.model, ...v.data, ...towerFiles].map((p) => () => load(p)), o.concurrency ?? 2);
  const data = rest.slice(0, v.data.length), towerData = rest.slice(v.data.length);
  const external = (paths: string[], files: (Uint8Array | string)[]) =>
    byPath ? {} : { externalData: paths.map((p, i) => ({ path: p.split("/").pop()!, data: files[i] as Uint8Array })) };
  const tokenizer = new Tokenizer(JSON.parse(tokJson), JSON.parse(tokCfg));
  o.onPhase?.("session");
  const eps = o.executionProviders ?? ["webgpu", "wasm"];
  const gpu = eps.some((e) => (typeof e === "string" ? e : e.name) === "webgpu");
  const create = async (model: Uint8Array | string, options: InferenceSession.SessionOptions) => {
    try {
      return await o.ort.InferenceSession.create(model, options);
    } catch (e) {
      throw failure ?? e;
    }
  };
  const session = await create(graph, {
    ...Kev.sessionOptions(manifest, variant, gpu),
    executionProviders: eps,
    ...external(v.data, data),
    ...o.sessionOptions,
  });
  // the tower's weights are fp16, cast to fp32 when the session loads; its output stays on the CPU for the decoder
  let vision: InferenceSession | undefined;
  try {
    // with filePath or readSync the tower's weights are read while its session is created: a failure there must not
    // leave the decoder loaded
    vision = tower && await create(towerData[0], {
      graphOptimizationLevel: "all",
      executionProviders: eps,
      ...external(tower.data, towerData.slice(1)),
      ...o.sessionOptions,
    });
  } catch (e) {
    await session.release().catch(() => undefined);
    throw e;
  }
  if (readSync) for (const p of weights) skipped(p);
  o.onPhase?.("ready");
  return new Kev({ ort: o.ort, session, head: PointerHead.fromSafetensors((head as Uint8Array).slice().buffer), tokenizer, manifest, variant, options: o, vision });
}
