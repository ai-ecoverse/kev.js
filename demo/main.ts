import { presets } from "./presets.ts";
import type { WorkerRequest, WorkerResponse } from "./worker.ts";
import type { Answer, KevManifest, SystemOneResponse } from "../src/index.ts";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const models = ["kev-0.8b", "kev-4b"];   // directories under /models
const worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
const verbose = new URLSearchParams(location.search).has("verbose");   // ?verbose: ORT logs, incl. node placement per EP
let nextId = 1;
const pending = new Map<number, { resolve: (r: SystemOneResponse) => void; reject: (e: Error) => void }>();


const hasWebGPU = "gpu" in navigator && !!(await (navigator as Navigator & { gpu: { requestAdapter(): Promise<unknown> } }).gpu.requestAdapter().catch(() => null));
if (!hasWebGPU) { $<HTMLSelectElement>("device").value = "wasm"; ($<HTMLSelectElement>("device").options[0]).disabled = true; }

for (const m of models) $<HTMLSelectElement>("model").add(new Option(m, m));
for (const name of Object.keys(presets)) $<HTMLSelectElement>("preset").add(new Option(name, name));
const setPreset = () => { $<HTMLTextAreaElement>("request").value = JSON.stringify(presets[$<HTMLSelectElement>("preset").value], null, 2); };
$("preset").onchange = setPreset; setPreset();

async function refreshVariants() {
  const manifest = (await (await fetch(`models/${$<HTMLSelectElement>("model").value}/manifest.json`)).json()) as KevManifest;
  const sel = $<HTMLSelectElement>("variant"); sel.innerHTML = "";
  for (const [name, v] of Object.entries(manifest.variants)) {
    if (name === "fp32") continue;   // fp32 is for Node parity tests; too large for a tab
    sel.add(new Option(`${name} (${(v.bytes / 1e6).toFixed(0)} MB)`, name));
  }
}
$("model").onchange = refreshVariants;
await refreshVariants();

const status = (s: string, cls = "") => { const el = $("status"); el.textContent = s; el.className = cls; };
const send = (m: WorkerRequest) => worker.postMessage(m);

$("load").onclick = () => {

  $<HTMLButtonElement>("run").disabled = $<HTMLButtonElement>("separate").disabled = true;
  status("Loading…");
  send({ type: "load", baseUrl: new URL(`models/${$<HTMLSelectElement>("model").value}`, location.href).href, variant: $<HTMLSelectElement>("variant").value, device: $<HTMLSelectElement>("device").value as "webgpu" | "wasm", verbose });
};

const progress = new Map<string, { loaded: number; total: number }>();
worker.onmessage = (e: MessageEvent<WorkerResponse>) => {
  const m = e.data;
  if (m.type === "progress") {
    progress.set(m.file, m);
    const loaded = [...progress.values()].reduce((s, p) => s + p.loaded, 0), total = [...progress.values()].reduce((s, p) => s + p.total, 0);
    $("bar").style.width = `${(100 * loaded) / Math.max(total, 1)}%`;
    status(`Downloading ${(loaded / 1e6).toFixed(0)} / ${(total / 1e6).toFixed(0)} MB`);
  } else if (m.type === "ready") {
    $("bar").style.width = "100%";
    status(`Ready: ${m.variant} on ${m.device}. Load ${(m.loadMs / 1000).toFixed(1)} s, warm-up ${(m.warmupMs / 1000).toFixed(1)} s.`);
    $<HTMLButtonElement>("run").disabled = $<HTMLButtonElement>("separate").disabled = false;
  } else if (m.type === "result") {
    pending.get(m.id)?.resolve(m.response as SystemOneResponse); pending.delete(m.id);
  } else if (m.type === "error") {
    if (m.id !== undefined) { pending.get(m.id)?.reject(new Error(m.message)); pending.delete(m.id); }
    else status(`Error: ${m.message}`, "err");
  }
};

function systemOne(request: unknown, mode: "packed" | "separate" | "probs" = "packed"): Promise<SystemOneResponse> {
  const id = nextId++;
  return new Promise((resolve, reject) => { pending.set(id, { resolve, reject }); send({ type: "run", id, request, mode }); });
}
// console access: await kev.systemOne({...})
(window as unknown as { kev: unknown }).kev = { systemOne: (r: unknown) => systemOne(r), systemOneSeparate: (r: unknown) => systemOne(r, "separate"), probs: (rec: unknown) => systemOne(rec, "probs") };

function bars(a: Answer): { name: string; p: number }[] {
  if (a.type === "noul") return [{ name: "yes", p: a.noul }, { name: "no", p: 1 - a.noul }];
  if (a.type === "choice") return Object.entries(a.probabilities).map(([name, p]) => ({ name, p }));
  return Object.entries(a.probabilities).map(([i, p]) => ({ name: `${i} · ${a.legend[i]}`, p }));
}

function summary(a: Answer): string {
  if (a.type === "noul") return `p(yes) = ${a.noul.toFixed(2)}`;
  if (a.type === "choice") return `${a.choice} · confidence ${a.confidence.toFixed(2)}`;
  return `score ${a.score.toFixed(2)} · confidence ${a.confidence.toFixed(2)}`;
}

function show(r: SystemOneResponse, other?: SystemOneResponse) {
  const el = $("answers"); el.innerHTML = "";
  const meta = document.createElement("div"); meta.className = "meta";
  meta.textContent = `${r.latency_ms} ms · ${r.usage.input_tokens} input tokens` + (other ? ` · separate: ${other.latency_ms} ms` : "");
  el.append(meta);
  for (const [id, a] of Object.entries(r.answers)) {
    const q = document.createElement("div"); q.className = "q";
    q.innerHTML = `<h3></h3>`; q.querySelector("h3")!.append(id, Object.assign(document.createElement("small"), { textContent: `${a.type} · ${summary(a)}` }));
    const bs = bars(a); const top = Math.max(...bs.map((b) => b.p));
    for (const b of bs) {
      const row = document.createElement("div"); row.className = `opt${b.p === top ? " top" : ""}`;
      row.innerHTML = `<span class="name"></span><span class="track"><span class="fill"></span></span><span class="val"></span>`;
      (row.querySelector(".name") as HTMLElement).textContent = b.name; (row.querySelector(".name") as HTMLElement).title = b.name;
      (row.querySelector(".fill") as HTMLElement).style.width = `${b.p * 100}%`;
      (row.querySelector(".val") as HTMLElement).textContent = b.p.toFixed(2);
      q.append(row);
    }
    if (other) {
      const o = other.answers[id]; const c = document.createElement("div"); c.className = "cmp";
      c.textContent = `separate pass: ${summary(o)}${JSON.stringify(o) === JSON.stringify(a) ? " (identical)" : ""}`;
      q.append(c);
    }
    el.append(q);
  }
  $("raw").textContent = JSON.stringify(other ? { packed: r, separate: other } : r, null, 2);
}

async function run(mode: "packed" | "separate") {
  let req: unknown;
  try { req = JSON.parse($<HTMLTextAreaElement>("request").value); } catch (e) { $("answers").innerHTML = `<p class="err">Invalid JSON: ${(e as Error).message}</p>`; return; }
  $<HTMLButtonElement>("run").disabled = $<HTMLButtonElement>("separate").disabled = true;
  try {
    const r = await systemOne(req);
    show(r, mode === "separate" ? await systemOne(req, "separate") : undefined);
  } catch (e) {
    const p = document.createElement("p"); p.className = "err"; p.textContent = (e as Error).message; $("answers").replaceChildren(p);
  } finally {
    $<HTMLButtonElement>("run").disabled = $<HTMLButtonElement>("separate").disabled = false;
  }
}
$("run").onclick = () => run("packed");
$("separate").onclick = () => run("separate");
document.addEventListener("keydown", (e) => { if ((e.metaKey || e.ctrlKey) && e.key === "Enter" && !$<HTMLButtonElement>("run").disabled) run("packed"); });

