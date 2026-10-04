// Demo worker: owns the single KevMinistral instance (WebGPU in a worker keeps the page responsive). Every task the
// page sends (decide, describe an image, complete text) runs on the same three sessions.
import { KevMinistral } from "./model.ts";
import { loadImage } from "./vision.ts";

let model: KevMinistral | undefined;

type Msg =
  | { id: number; type: "load"; base: string }
  | { id: number; type: "decide"; req: any; image?: string | Blob }
  | { id: number; type: "generate"; prompt: string; maxNew: number; image?: string | Blob; temperature: number };

self.onmessage = async (e: MessageEvent<Msg>) => {
  const m = e.data;
  const reply = (body: object) => (self as any).postMessage({ id: m.id, ...body });
  try {
    if (m.type === "load") {
      const t0 = performance.now();
      model = await KevMinistral.load(m.base, (s) => reply({ log: s }), (file, loaded, total, cached) => reply({ progress: { file, loaded, total, cached } }));
      const adapter = await (navigator as any).gpu?.requestAdapter();
      reply({ done: { seconds: (performance.now() - t0) / 1000, gpu: adapter?.info?.description || adapter?.info?.vendor || "WebGPU" } });
    } else if (m.type === "decide") {
      const img = m.image ? await loadImage(m.image) : undefined;
      const r = await model!.decide(m.req, img);
      reply({ done: { probs: r.probs, keys: r.meta.map((q: any) => q.keys), tokens: r.stateIds.length, imageTokens: r.imgIds.length, ms: r.ms } });
    } else if (m.type === "generate") {
      const img = m.image ? await loadImage(m.image) : undefined;
      const r = await model!.generate(m.prompt, m.maxNew, img, {
        temperature: m.temperature, topK: 40, maskSpecial: true, onToken: (text, n) => reply({ token: { text, n } }),
      });
      reply({ done: { text: r.text, tokens: r.outIds.length, inputTokens: r.inputIds.length, ms: r.ms } });
    }
  } catch (err: any) {
    reply({ error: String(err?.message ?? err) });
  }
};
