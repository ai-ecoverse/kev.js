// Browser tests: the loader against the real OPFS, and the published model loaded from OPFS on onnxruntime-web.
// KEV_REQUIRE_MODEL=1 turns a missing bundle into a failure, KEV_REQUIRE_WEBGPU=1 a missing WebGPU adapter (CI sets
// what it provides), so a green run means the case ran.
import { writeFileSync } from "node:fs";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test as base, chromium, expect, type BrowserContext, type Page } from "@playwright/test";
import type { CaseName, CaseResult } from "./worker.ts";

// One persistent profile for the whole run, as a user's browser has: OPFS gets the on-disk quota (an incognito
// context's is in memory, and a second copy of the 0.8B bundle exceeded it on CI), and the model copied into OPFS by
// the first model case is found there by the next, which is the resume path.
const test = base.extend<object, { profile: BrowserContext }>({
  profile: [async ({ channel, launchOptions }, use) => {
    const dir = await mkdtemp(join(tmpdir(), "kev-browser-"));
    const context = await chromium.launchPersistentContext(dir, { ...launchOptions, channel });
    await use(context);
    await context.close();
    await rm(dir, { recursive: true, force: true });
  }, { scope: "worker" }],
  page: async ({ profile, baseURL }, use) => {
    const page = await profile.newPage();
    await page.goto(baseURL!);
    await use(page);
    await page.close();
  },
});

const run = (page: Page, name: CaseName, options: Record<string, unknown> = {}) =>
  page.evaluate(([n, o]) => window.harness.run(n as CaseName, o as Record<string, unknown>), [name, options] as const) as Promise<CaseResult>;

test.beforeEach(async ({ page }) => {
  page.on("console", (m) => { if (m.type() === "error" || m.text().startsWith("kev-")) console.log(`[browser] ${m.text()}`); });
  await page.waitForFunction(() => window.harness?.ready);
  expect(await page.evaluate(() => self.crossOriginIsolated)).toBe(true);
});

const synthetic = { graph: "graph", externalData: [["model.onnx.data_0", "shard zero"], ["model.onnx.data_1", "shard one!"]] };

test("loadKev reads a bundle from an OPFS directory handle", async ({ page }) => {
  expect(await run(page, "opfs-directory")).toEqual({ run: "jaredpalmer/kev-test@abc", ...synthetic });
});

test("loadKev reads OPFS File objects through a read function, each bundle file once", async ({ page }) => {
  const r = await run(page, "opfs-function");
  expect(r.reads).toEqual(r.expected);
  expect(r).toMatchObject({ run: "jaredpalmer/kev-test@abc", ...synthetic });
});

test("a file cut short in OPFS fails the load by name", async ({ page }) => {
  expect((await run(page, "opfs-short")).error).toMatch(/model\.onnx\.data_1: expected 10 bytes, read 5/);
});

for (const ep of ["wasm", "webgpu"] as const) {
  test(`Kev-0.8B q8f32 from OPFS on ${ep} matches the PyTorch reference`, async ({ page }) => {
    test.setTimeout(40 * 60_000);
    // every fixture by default (WASM: 1-3 s each). KEV_WEBGPU_FIXTURES caps WebGPU alone: on SwiftShader, the CPU
    // Vulkan that CI's GPU-less runners have, one fixture takes about 150 s. KEV_BROWSER_FIXTURES caps both.
    const cap = (ep === "webgpu" && Number(process.env.KEV_WEBGPU_FIXTURES)) || Number(process.env.KEV_BROWSER_FIXTURES);
    const limit = cap || undefined;
    const r = await run(page, "model", { ep, limit });
    console.log(`[${ep}] ${r.skip ?? `copied ${r.copied} files into OPFS`}`);
    if (r.skip) {
      const required = process.env[String(r.skip).includes("WebGPU") ? "KEV_REQUIRE_WEBGPU" : "KEV_REQUIRE_MODEL"] === "1";
      expect(required ? r.skip : undefined, "required case could not run").toBeUndefined();
      test.skip(true, String(r.skip));
    }
    console.log(`[${ep}${r.adapter ? `, ${r.adapter}` : ""}] ${r.fixtures} fixtures, ${r.summary}; load ${r.loadMs} ms, ${r.msPerFixture} ms/fixture`);
    expect(r.fixtureRun, "fixtures and weights come from the same checkpoint").toBe(r.run);
    expect(r.violations, "parity with the PyTorch reference (test/parity.ts)").toEqual([]);
    expect(r.answerKeys).toEqual(r.expectedAnswerKeys);
    expect(r.modelFetches, "no model file is fetched when loading from OPFS").toEqual([]);
    expect(r.caches, "nothing is written to Cache Storage").toEqual([]);
  });
}

// Image requests on a vision bundle (public/models/<KEV_VISION_MODEL>, default kev-4b-vision, built locally: README,
// Images) against fixtures/<model>.json. KEV_VISION_EP picks the backend (default webgpu), KEV_VISION_REPORT saves
// the per-set report as JSON.
test("a Kev vision bundle matches the PyTorch image fixtures", async ({ page }) => {
  test.setTimeout(60 * 60_000);
  const ep = process.env.KEV_VISION_EP ?? "webgpu";
  const r = await run(page, "vision", { ep, model: process.env.KEV_VISION_MODEL ?? "kev-4b-vision", variant: process.env.KEV_VISION_VARIANT ?? "q8f32", limit: Number(process.env.KEV_BROWSER_FIXTURES) || undefined });
  if (r.skip) test.skip(true, String(r.skip));
  for (const [s, v] of Object.entries(r.sets as Record<string, Record<string, unknown>>))
    console.log(`[${ep}${r.adapter ? `, ${r.adapter}` : ""}] ${s}: ${JSON.stringify(v)}`);
  if (process.env.KEV_VISION_REPORT) writeFileSync(process.env.KEV_VISION_REPORT, JSON.stringify(r, null, 1));
  expect(r.fixtureRun, "fixtures and weights come from the same checkpoint").toBe(r.run);
  for (const v of Object.values(r.sets as Record<string, { violations: string[]; max_mean_pixel_diff: number }>)) {
    expect(v.violations, "parity with the PyTorch reference (test/parity.ts)").toEqual([]);
    expect(v.max_mean_pixel_diff, "the browser decodes and resizes the images as PIL and Qwen's processor do").toBeLessThan(2e-3);
  }
});
