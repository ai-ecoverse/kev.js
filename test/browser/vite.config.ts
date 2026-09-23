import { defineConfig, type Plugin } from "vite";
import { createReadStream, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";

// The browser test harness: test/browser served on its own port, models from public/models like `npm run dev`, and the
// Python fixtures under /fixtures. Cross-origin isolated, so the WASM backend runs multi-threaded as in the demo.
const repo = fileURLToPath(new URL("../..", import.meta.url));
const isolation = { "Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Embedder-Policy": "require-corp" };

const fixtures: Plugin = {
  name: "kev-fixtures",
  configureServer(server) {
    server.middlewares.use("/fixtures/", (req, res, next) => {
      const file = `${repo}/fixtures/${decodeURIComponent(req.url ?? "").split("?")[0].replace(/^\/+/, "")}`;
      if (!/^[\w.-]+\.json$/.test(file.split("/").pop() ?? "") || !existsSync(file)) return next();
      res.setHeader("content-type", "application/json");
      createReadStream(file).pipe(res);
    });
  },
};

export default defineConfig({
  root: fileURLToPath(new URL(".", import.meta.url)),
  publicDir: `${repo}/public`,
  appType: "mpa",   // a missing model file is a 404, not index.html
  plugins: [fixtures],
  server: { headers: isolation, host: "127.0.0.1", port: 5174, strictPort: true },
  optimizeDeps: { exclude: ["onnxruntime-web"] },
  worker: { format: "es" },
});
