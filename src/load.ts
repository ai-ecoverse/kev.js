// Load a packaged Kev model (kev_web_export.package output) from a URL: manifest, tokenizer, head, ONNX graph and
// its external weights. Weight files are large (0.8 GB for kev-0.8b q8), so they are kept in Cache Storage when
// available and streamed with progress.

import { Tokenizer } from "@huggingface/tokenizers";
import type { InferenceSession } from "onnxruntime-common";
import { PointerHead } from "./head.ts";
import { Kev, type KevManifest, type KevOptions, type OrtModule } from "./model.ts";

export interface Progress { file: string; loaded: number; total: number }

/** Coarse stage of loadKev, for a status line that keeps moving after the download finishes. */
export type LoadPhase = "manifest" | "download" | "session" | "ready";

export interface LoadOptions extends KevOptions {
  ort: OrtModule;
  /** variant in manifest.variants; default: the first one listed */
  variant?: string;
  /** e.g. ["webgpu"] or ["wasm"]; default ["webgpu", "wasm"] */
  executionProviders?: InferenceSession.SessionOptions["executionProviders"];
  onProgress?: (p: Progress) => void;
  onPhase?: (phase: LoadPhase) => void;
  /** Cache Storage bucket for model files; null disables caching. Default "kev-web-v1". */
  cacheName?: string | null;
  sessionOptions?: InferenceSession.SessionOptions;
  /** weight files fetched at once. Default 2: on a bandwidth-limited link more streams are slower in aggregate
   * (measured through a tunnel: 1.2 MB/s with one stream, 0.75 MB/s across six). */
  concurrency?: number;
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

export async function fetchFile(url: string, o: { cacheName?: string | null; onProgress?: (p: Progress) => void; file?: string; bytes?: number } = {}): Promise<Uint8Array> {
  const file = o.file ?? url;
  const cache = o.cacheName !== null && typeof caches !== "undefined" ? await caches.open(o.cacheName ?? "kev-web-v1") : null;
  const hit = await cache?.match(url);
  if (hit) {
    const data = await readWithProgress(hit, file, o.onProgress, o.bytes);
    if (!o.bytes || data.length === o.bytes) return data;
    await cache!.delete(url);   // truncated entry (e.g. a reload mid-download): fetch it again
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  // read first and cache the bytes afterwards: cache.put(res.clone()) would download the whole body before it
  // resolves, so a multi-hundred-MB file would report no progress at all
  const data = await readWithProgress(res, file, o.onProgress, o.bytes);
  if (o.bytes && data.length !== o.bytes) throw new Error(`${file}: expected ${o.bytes} bytes, received ${data.length}`);
  if (cache) {
    try {
      await cache.put(url, new Response(data as BodyInit, { headers: { "content-type": res.headers.get("content-type") ?? "application/octet-stream", "content-length": String(data.length) } }));
    } catch { /* quota: run uncached */ }
  }
  return data;
}

const join = (base: string, path: string) => `${base.replace(/\/$/, "")}/${path}`;

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

export async function loadKev(baseUrl: string, o: LoadOptions): Promise<Kev> {
  o.onPhase?.("manifest");
  const text = async (p: string) => new TextDecoder().decode(await fetchFile(join(baseUrl, p), { cacheName: null }));
  const manifest = JSON.parse(await text("manifest.json")) as KevManifest;
  const variant = o.variant ?? Object.keys(manifest.variants)[0];
  const v = manifest.variants[variant];
  if (!v) throw new Error(`unknown variant ${variant}; have ${Object.keys(manifest.variants).join(", ")}`);
  o.onPhase?.("download");
  // announce every file up front so the total does not grow as downloads start
  for (const [p, bytes] of Object.entries(v.sizes ?? {})) o.onProgress?.({ file: p, loaded: 0, total: bytes });
  const get = (p: string) => fetchFile(join(baseUrl, p), { cacheName: o.cacheName, onProgress: o.onProgress, file: p, bytes: v.sizes?.[p] });
  const [tokJson, tokCfg] = await Promise.all([text(manifest.files.tokenizer), text(manifest.files.tokenizer_config)]);
  const [head, graph, ...data] = await pool([manifest.files.head, v.model, ...v.data].map((p) => () => get(p)), o.concurrency ?? 2);
  const tokenizer = new Tokenizer(JSON.parse(tokJson), JSON.parse(tokCfg));
  o.onPhase?.("session");
  const eps = o.executionProviders ?? ["webgpu", "wasm"];
  const gpu = eps.some((e) => (typeof e === "string" ? e : e.name) === "webgpu");
  const session = await o.ort.InferenceSession.create(graph, {
    ...Kev.sessionOptions(manifest, variant, gpu),
    executionProviders: eps,
    externalData: v.data.map((p, i) => ({ path: p.split("/").pop()!, data: data[i] })),
    ...o.sessionOptions,
  });
  o.onPhase?.("ready");
  return new Kev({ ort: o.ort, session, head: PointerHead.fromSafetensors(head.slice().buffer), tokenizer, manifest, variant, options: o });
}
