// Browser tests: the loader against the real OPFS, and the published model loaded from OPFS on onnxruntime-web.
// KEV_REQUIRE_MODEL=1 turns a missing bundle into a failure, KEV_REQUIRE_WEBGPU=1 a missing WebGPU adapter (CI sets
// what it provides), so a green run means the case ran.
import { expect, test, type Page } from "@playwright/test";
import type { CaseName, CaseResult } from "./worker.ts";

const run = (page: Page, name: CaseName, options: Record<string, unknown> = {}) =>
  page.evaluate(([n, o]) => window.harness.run(n as CaseName, o as Record<string, unknown>), [name, options] as const) as Promise<CaseResult>;

test.beforeEach(async ({ page }) => {
  page.on("console", (m) => { if (m.type() === "error" || m.text().startsWith("kev-")) console.log(`[browser] ${m.text()}`); });
  await page.goto("/");
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
    test.setTimeout(20 * 60_000);
    // every fixture by default (WASM: about a second each); KEV_BROWSER_FIXTURES caps the count for a slow adapter
    const limit = Number(process.env.KEV_BROWSER_FIXTURES) || undefined;
    const r = await run(page, "model", { ep, limit });
    if (r.skip) {
      const required = process.env[String(r.skip).includes("WebGPU") ? "KEV_REQUIRE_WEBGPU" : "KEV_REQUIRE_MODEL"] === "1";
      expect(required ? r.skip : undefined, "required case could not run").toBeUndefined();
      test.skip(true, String(r.skip));
    }
    console.log(`[${ep}${r.adapter ? `, ${r.adapter}` : ""}] ${r.fixtures} fixtures, ${r.summary}; load ${r.loadMs} ms, ${r.msPerFixture} ms/fixture`);
    expect(r.fixtureRun, "fixtures and weights come from the same checkpoint").toBe(r.run);
    expect(r.worst as number, `max |dp| at ${r.worstAt}`).toBeLessThanOrEqual(r.bound as number);
    expect(r.clearFlips, "answers changed only where the reference is a near-tie").toEqual([]);
    expect(r.answerKeys).toEqual(r.expectedAnswerKeys);
    expect(r.modelFetches, "no model file is fetched when loading from OPFS").toEqual([]);
    expect(r.caches, "nothing is written to Cache Storage").toEqual([]);
  });
}
