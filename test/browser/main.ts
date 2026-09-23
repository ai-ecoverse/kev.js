// Page side of the harness: the cases run in a module worker, where the demo and SLICC run Kev. Playwright calls
// window.harness.run(name, options) and gets the case's result, or its error message.
import type { CaseName, CaseResult } from "./worker.ts";

const worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
const pending = new Map<number, { resolve: (r: CaseResult) => void; reject: (e: Error) => void }>();
let next = 0;
const log = document.getElementById("log")!;

worker.onmessage = (e: MessageEvent<{ id: number; result?: CaseResult; error?: string; log?: string }>) => {
  const m = e.data;
  if (m.log) { log.textContent += `${m.log}\n`; console.log(m.log); return; }
  const p = pending.get(m.id);
  pending.delete(m.id);
  if (m.error !== undefined) p?.reject(new Error(m.error)); else p?.resolve(m.result!);
};

declare global { interface Window { harness: { ready: boolean; run: (name: CaseName, options?: Record<string, unknown>) => Promise<CaseResult> } } }

window.harness = {
  ready: true,
  run: (name, options = {}) => new Promise((resolve, reject) => {
    const id = next++;
    pending.set(id, { resolve, reject });
    worker.postMessage({ id, name, options });
  }),
};
