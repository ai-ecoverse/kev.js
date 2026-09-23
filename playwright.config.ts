import { defineConfig } from "@playwright/test";

// Browser tests (test/browser): Chromium against the harness dev server. WebGPU is enabled where the platform has an
// adapter; on Linux without a GPU that is SwiftShader's Vulkan.
// KEV_WEBGPU_SWIFTSHADER=1 forces that software adapter elsewhere too, to reproduce CI.
const swiftshader = process.platform === "linux" || process.env.KEV_WEBGPU_SWIFTSHADER === "1";
const webgpu = ["--enable-unsafe-webgpu", ...(swiftshader ? ["--enable-features=Vulkan", "--use-vulkan=swiftshader", "--use-webgpu-adapter=swiftshader", "--disable-vulkan-surface"] : [])];

export default defineConfig({
  testDir: "test/browser",
  testMatch: "*.spec.ts",
  timeout: 120_000,
  workers: 1,
  reporter: process.env.CI ? [["list"], ["github"]] : "list",
  use: { baseURL: "http://127.0.0.1:5174", channel: "chromium", launchOptions: { args: webgpu } },
  webServer: { command: "vite --config test/browser/vite.config.ts", url: "http://127.0.0.1:5174", reuseExistingServer: !process.env.CI, timeout: 60_000 },
});
