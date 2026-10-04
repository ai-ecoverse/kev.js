import { defineConfig } from "vite";
import { fileURLToPath } from "node:url";

// Models are served from ../dist (dist/models/<name>/manifest.json ...), produced by kev_web_export.package.
// Cross-origin isolation enables multi-threaded WASM (SharedArrayBuffer) for the CPU fallback.
const isolation = { "Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Embedder-Policy": "require-corp" };

// GitHub Pages serves the project at /<repo>/; PAGES_BASE sets that prefix in CI.
export default defineConfig({
  base: process.env.PAGES_BASE ?? "/",
  root: fileURLToPath(new URL(".", import.meta.url)),
  publicDir: fileURLToPath(new URL("../public", import.meta.url)),   // public/models/<name>/... from kev_web_export.package
  server: { headers: isolation, host: "127.0.0.1", port: 5173, allowedHosts: [".getbb.app"] },   // bb connect exposes the dev server over https
  preview: { headers: isolation, host: "127.0.0.1", port: 4173 },
  optimizeDeps: { exclude: ["onnxruntime-web"] },
  worker: { format: "es" },
  build: {
    outDir: fileURLToPath(new URL("../dist-demo", import.meta.url)), target: "es2023", copyPublicDir: false, emptyOutDir: true,
    // the homepage, and the kev-ministral demo (one Ministral-3-3B download: decisions, image description, completion)
    rollupOptions: { input: { index: fileURLToPath(new URL("index.html", import.meta.url)), ministral: fileURLToPath(new URL("ministral.html", import.meta.url)) } },
  },
});
