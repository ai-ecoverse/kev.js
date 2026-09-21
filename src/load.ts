// Load a packaged Kev model (kev_web_export.package output) from a URL: manifest, tokenizer, head, ONNX graph and
// its external weights. Weight files are large (0.8 GB for kev-0.8b q8), so they are kept in Cache Storage when
// available and streamed with progress.

import { Tokenizer } from "@huggingface/tokenizers";
import type { InferenceSession } from "onnxruntime-common";
import { PointerHead } from "./head.ts";
import { Kev, type KevManifest, type KevOptions, type OrtModule } from "./model.ts";

export interface Progress { file: string; loaded: number; total: number }

export interface LoadOptions extends KevOptions {
  ort: OrtModule;
  /** variant in manifest.variants; default: the first one listed */
  variant?: string;
  /** e.g. ["webgpu"] or ["wasm"]; default ["webgpu", "wasm"] */
  executionProviders?: InferenceSession.SessionOptions["executionProviders"];
  onProgress?: (p: Progress) => void;
  /** Cache Storage bucket for model files; null disables caching. Default "kev-web-v1". */
  cacheName?: string | null;
  sessionOptions?: InferenceSession.SessionOptions;
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
  if (hit) return readWithProgress(hit, file, o.onProgress, o.bytes);
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  // read first and cache the bytes afterwards: cache.put(res.clone()) would download the whole body before it
  // resolves, so a multi-hundred-MB file would report no progress at all
  const data = await readWithProgress(res, file, o.onProgress, o.bytes);
  if (cache) {
    try {
      await cache.put(url, new Response(data as BodyInit, { headers: { "content-type": res.headers.get("content-type") ?? "application/octet-stream", "content-length": String(data.length) } }));
    } catch { /* quota: run uncached */ }
  }
  return data;
}

const join = (base: string, path: string) => `${base.replace(/\/$/, "")}/${path}`;

export async function loadKev(baseUrl: string, o: LoadOptions): Promise<Kev> {
  const text = async (p: string) => new TextDecoder().decode(await fetchFile(join(baseUrl, p), { cacheName: null }));
  const manifest = JSON.parse(await text("manifest.json")) as KevManifest;
  const variant = o.variant ?? Object.keys(manifest.variants)[0];
  const v = manifest.variants[variant];
  if (!v) throw new Error(`unknown variant ${variant}; have ${Object.keys(manifest.variants).join(", ")}`);
  const get = (p: string) => fetchFile(join(baseUrl, p), { cacheName: o.cacheName, onProgress: o.onProgress, file: p, bytes: v.sizes?.[p] });
  const [tokJson, tokCfg, head, graph, ...data] = await Promise.all([
    text(manifest.files.tokenizer), text(manifest.files.tokenizer_config), get(manifest.files.head), get(v.model), ...v.data.map(get),
  ]);
  const tokenizer = new Tokenizer(JSON.parse(tokJson), JSON.parse(tokCfg));
  const eps = o.executionProviders ?? ["webgpu", "wasm"];
  const gpu = eps.some((e) => (typeof e === "string" ? e : e.name) === "webgpu");
  const session = await o.ort.InferenceSession.create(graph, {
    ...Kev.sessionOptions(manifest, variant, gpu),
    executionProviders: eps,
    externalData: v.data.map((p, i) => ({ path: p.split("/").pop()!, data: data[i] })),
    ...o.sessionOptions,
  });
  return new Kev({ ort: o.ort, session, head: PointerHead.fromSafetensors(head.slice().buffer), tokenizer, manifest, variant, options: o });
}
