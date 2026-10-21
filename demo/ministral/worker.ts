// Demo worker: owns the single KevMinistral instance (WebGPU in a worker keeps the page responsive). Every task the
// page sends (decide, describe an image, complete text) runs on the same three sessions, loaded once by the library's
// loadKevMinistral.
import * as ort from "onnxruntime-web/webgpu";
import { loadKevMinistral, toRecord, type ImageLike, type KevMinistral, type SystemOneRequest } from "../../src/index.ts";

let model: KevMinistral | undefined;

type Msg =
  | { id: number; type: "load"; base: string }
  | { id: number; type: "decide"; req: SystemOneRequest; image?: string | Blob }
  | { id: number; type: "generate"; prompt: string; maxNew: number; image?: string | Blob; temperature: number };

/** RGBA pixels of an image URL as the browser decodes it, without color management (PIL reads raw values). */
async function loadImage(src: string | Blob): Promise<ImageLike> {
  const blob = typeof src === "string" ? await (await fetch(src)).blob() : src;
  const bmp = await createImageBitmap(blob, { colorSpaceConversion: "none", premultiplyAlpha: "none" });
  const ctx = new OffscreenCanvas(bmp.width, bmp.height).getContext("2d", { willReadFrequently: true })!;
  ctx.drawImage(bmp, 0, 0);
  const d = ctx.getImageData(0, 0, bmp.width, bmp.height);
  return { width: d.width, height: d.height, data: d.data };
}

self.onmessage = async (e: MessageEvent<Msg>) => {
  const m = e.data;
  const reply = (body: object) => (self as unknown as Worker).postMessage({ id: m.id, ...body });
  try {
    if (m.type === "load") {
      const t0 = performance.now();
      // the weights' Hugging Face commit is in the base URL, so Cache Storage entries never outlive their bundle
      model = await loadKevMinistral(m.base, {
        ort, executionProviders: ["webgpu"], cacheName: "kev-ministral-bundle-v1",
        onProgress: ({ file, loaded, total }) => reply({ progress: { file, loaded, total } }),
        onPhase: (phase) => reply({ log: phase }),
      });
      const adapter = await (navigator as unknown as { gpu?: GPU }).gpu?.requestAdapter();
      reply({ done: { seconds: (performance.now() - t0) / 1000, gpu: adapter?.info?.description || adapter?.info?.vendor || "WebGPU" } });
    } else if (m.type === "decide") {
      const img = m.image ? await loadImage(m.image) : undefined;
      const t0 = performance.now();
      const { record, meta } = toRecord(m.req);
      const probs = await model!.probs(record, img);
      const t = model!.timing;
      reply({ done: { probs, keys: meta.map((q) => q.keys), imageTokens: t.imageTokens,
        tokens: model!.encode(record).tokens + 1 + t.imageTokens, ms: { vision: t.preprocess + t.vision, total: performance.now() - t0 } } });
    } else if (m.type === "generate") {
      const img = m.image ? await loadImage(m.image) : undefined;
      const r = await model!.generate(m.prompt, {
        maxTokens: m.maxNew, image: img, temperature: m.temperature, topK: 40, maskSpecial: true, onToken: (text, n) => reply({ token: { text, n } }),
      });
      reply({ done: { text: r.text, tokens: r.ids.length, inputTokens: r.inputTokens, ms: r.ms } });
    }
  } catch (err) {
    reply({ error: String((err as Error)?.message ?? err) });
  }
};
