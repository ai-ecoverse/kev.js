// Screenshot an eval set's generated pages: every item of <set>/questions.json with a `page` is rendered by Playwright's
// Chromium at its `size` (device scale 1) into its `image`.
//   node scripts/render-vision-set.mjs eval/vision-v2
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { chromium } from "playwright";

const dir = resolve(process.argv[2] ?? "eval/vision-v2");
const { items } = JSON.parse(readFileSync(`${dir}/questions.json`, "utf8"));
const browser = await chromium.launch();
let n = 0;
for (const it of items.filter((i) => i.page)) {
  const [width, height] = it.size;
  const page = await browser.newPage({ viewport: { width, height }, deviceScaleFactor: 1 });
  await page.goto(pathToFileURL(`${dir}/${it.page}`).href);
  await page.screenshot({ path: `${dir}/${it.image}` });
  await page.close();
  n++;
}
await browser.close();
console.log(`${n} pages -> ${dir}/images (Chromium ${browser.version()})`);
