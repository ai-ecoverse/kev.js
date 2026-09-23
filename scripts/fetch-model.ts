// Download one variant of a published bundle into public/models/<model>, the layout `npm run dev`, the browser tests and
// KEV_MODEL_DIR expect. Files already at their manifest size are kept, so an interrupted or cached download resumes.
//   node --import tsx scripts/fetch-model.ts kev-0.8b q8f32
import { createWriteStream, existsSync, statSync } from "node:fs";
import { mkdir, rename, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import { fileURLToPath } from "node:url";
import type { ReadableStream } from "node:stream/web";
import { modelFiles, type KevManifest } from "../src/index.ts";

const [model = "kev-0.8b", variant = "q8f32"] = process.argv.slice(2);
const base = process.env.KEV_HF_BASE ?? "https://huggingface.co/ai-ecoverse/kev.js/resolve/main";
const out = join(fileURLToPath(new URL("..", import.meta.url)), "public", "models", model);

async function get(url: string, attempts = 4): Promise<Response> {
  for (let i = 1; ; i++) {
    try {
      const res = await fetch(url);
      if (res.ok) return res;
      if (res.status < 500 || i >= attempts) throw new Error(`${url}: HTTP ${res.status}`);
    } catch (e) {
      if (i >= attempts || String(e).includes("HTTP 4")) throw e;
    }
    await new Promise((r) => setTimeout(r, 1000 * i));
  }
}

const manifestText = await (await get(`${base}/${model}/manifest.json`)).text();
const manifest = JSON.parse(manifestText) as KevManifest;
let fetched = 0, kept = 0;
for (const { path, bytes } of modelFiles(manifest, variant)) {
  const dest = join(out, path);
  if (bytes && existsSync(dest) && statSync(dest).size === bytes) { kept++; continue; }
  await mkdir(dirname(dest), { recursive: true });
  const res = await get(`${base}/${model}/${path}`);
  await pipeline(Readable.fromWeb(res.body as ReadableStream), createWriteStream(`${dest}.part`));
  const size = statSync(`${dest}.part`).size;
  if (bytes && size !== bytes) throw new Error(`${path}: expected ${bytes} bytes, received ${size}`);
  await rename(`${dest}.part`, dest);
  fetched++;
}
await writeFile(join(out, "manifest.json"), manifestText);   // last: a manifest on disk means its files are complete
console.log(`${model} ${variant} (${manifest.run}): ${fetched} fetched, ${kept} kept -> ${out}`);
