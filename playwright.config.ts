import { defineConfig } from "@playwright/test";

// Browser tests (test/browser): Chromium against the harness dev server. WebGPU is enabled where the platform has an
// adapter; on Linux without a GPU that is SwiftShader's Vulkan.
// KEV_WEBGPU_SWIFTSHADER=1 forces that software adapter elsewhere too, to reproduce CI.
const swiftshader = process.platform === "linux" || process.env.KEV_WEBGPU_SWIFTSHADER === "1";
const webgpu = ["--enable-unsafe-webgpu", ...(swiftshader ? ["--enable-features=Vulkan", "--use-vulkan=swiftshader", "--use-webgpu-adapter=swiftshader", "--disable-vulkan-surface"] : [])];

// KEV_HARNESS_PORT moves the harness off 5174 when another dev server holds it (reuseExistingServer would test that one)
const port = Number(process.env.KEV_HARNESS_PORT ?? 5174);

export default defineConfig({
  testDir: "test/browser",
  testMatch: "*.spec.ts",
  timeout: 120_000,
  workers: 1,
  reporter: process.env.CI ? [["list"], ["github"]] : "list",
  // KEV_BROWSER_CHANNEL=chrome runs the installed Google Chrome: on macOS, Playwright's headless Chromium reports a
  // Metal adapter but stalls for minutes in the first Kev-4B pass, where Chrome answers in 0.4 s
  use: { baseURL: `http://127.0.0.1:${port}`, channel: process.env.KEV_BROWSER_CHANNEL ?? "chromium", launchOptions: { args: webgpu } },
  webServer: { command: "vite --config test/browser/vite.config.ts", url: `http://127.0.0.1:${port}`, reuseExistingServer: !process.env.CI, timeout: 60_000 },
});
