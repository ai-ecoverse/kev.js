// Evaluate code in the demo tab of a Chrome running with --remote-debugging-port (default 9222).
//
//   node scripts/cdp.mjs '<expression>'      # awaited, returned by value
//   node scripts/cdp.mjs -f file.js          # file contains the expression
//   node scripts/cdp.mjs --shot out.png      # screenshot the tab
//
// Attaches straight to the page target: Playwright's connectOverCDP enumerates every tab, which times out on a
// browser with a hundred of them.
import { readFileSync, writeFileSync } from "node:fs";

const port = process.env.CDP_PORT ?? "9222";
const match = process.env.MATCH ?? "5173";
const targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
const target = targets.find((t) => t.type === "page" && t.url.includes(match));
if (!target) { console.error(`no page matching ${match}`); process.exit(1); }

const shot = process.argv[2] === "--shot" ? process.argv[3] : null;
const expression = process.argv[2] === "-f" ? readFileSync(process.argv[3], "utf8") : process.argv[2];
const ws = new WebSocket(target.webSocketDebuggerUrl);
const pending = new Map();
let id = 0;
const send = (method, params) => new Promise((resolve, reject) => { pending.set(++id, { resolve, reject }); ws.send(JSON.stringify({ id, method, params })); });
ws.onmessage = (e) => {
  const m = JSON.parse(e.data);
  if (m.id && pending.has(m.id)) { const { resolve, reject } = pending.get(m.id); pending.delete(m.id); m.error ? reject(new Error(m.error.message)) : resolve(m.result); }
};
await new Promise((r) => (ws.onopen = r));
if (shot) {
  const { data } = await send("Page.captureScreenshot", { format: "png", captureBeyondViewport: true });
  writeFileSync(shot, Buffer.from(data, "base64"));
  ws.close();
  console.log(shot);
  process.exit(0);
}
const res = await send("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true, userGesture: true });
ws.close();
if (res.exceptionDetails) { console.error(res.exceptionDetails.exception?.description ?? res.exceptionDetails.text); process.exit(1); }
console.log(JSON.stringify(res.result.value, null, 2));
