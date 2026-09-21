import { presets } from "./presets.ts";
import type { WorkerRequest, WorkerResponse } from "./worker.ts";
import type { Answer, KevManifest, SystemOneResponse } from "../src/index.ts";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const models = ["kev-0.8b", "kev-4b", "kev-9b"];

// Weights live on Hugging Face for the published page; a dev server with public/models/ serves them locally.
// Override with VITE_MODEL_BASE, or ?models=<url> for a one-off.
const HF_BASE = "https://huggingface.co/ai-ecoverse/kev.js/resolve/main";
const params = new URLSearchParams(location.search);
const MODEL_BASE = params.get("models") ?? (import.meta.env.VITE_MODEL_BASE as string | undefined) ?? (import.meta.env.DEV ? "models" : HF_BASE);
const modelUrl = (name: string) => new URL(`${MODEL_BASE.replace(/\/$/, "")}/${name}`, location.href).href;
const worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
const verbose = params.has("verbose");   // ?verbose: ORT logs, incl. node placement per EP
const store = { get: (k: string) => localStorage.getItem(`kev-web:${k}`), set: (k: string, v: string) => localStorage.setItem(`kev-web:${k}`, v) };

let nextId = 1;
const pending = new Map<number, { resolve: (r: SystemOneResponse) => void; reject: (e: Error) => void; onPartial?: (qid: string, a: Answer) => void }>();

const hasWebGPU = "gpu" in navigator && !!(await (navigator as Navigator & { gpu: { requestAdapter(): Promise<unknown> } }).gpu.requestAdapter().catch(() => null));

for (const m of models) $<HTMLSelectElement>("model").add(new Option(m, m));
for (const name of Object.keys(presets)) $<HTMLSelectElement>("preset").add(new Option(name, name));

const device = $<HTMLSelectElement>("device");
device.value = hasWebGPU ? store.get("device") ?? "webgpu" : "wasm";
if (!hasWebGPU) { device.options[0].disabled = true; device.title = "This browser has no WebGPU adapter"; }
$<HTMLSelectElement>("model").value = store.get("model") ?? models[0];

const request = $<HTMLTextAreaElement>("request");
const setPreset = () => { request.value = JSON.stringify(presets[$<HTMLSelectElement>("preset").value], null, 2); store.set("request", request.value); };
$("preset").onchange = setPreset;
request.value = store.get("request") ?? "";
request.oninput = () => store.set("request", request.value);
if (!request.value) setPreset();

/** Models whose weights finished downloading here before; the loader still checks every file's size. */
const cacheKey = (model: string, variant: string) => `cached:${model}/${variant}`;

async function refreshVariants() {
  const model = $<HTMLSelectElement>("model").value;
  const sel = $<HTMLSelectElement>("variant"); sel.innerHTML = "";
  const res = await fetch(`${modelUrl(model)}/manifest.json`);
  if (!res.ok) { sel.innerHTML = ""; sel.add(new Option("unavailable", "")); status(`${model} is not published yet.`, "err"); return; }
  const manifest = (await res.json()) as KevManifest;
  for (const [name, v] of Object.entries(manifest.variants)) {
    if (name === "fp32") continue;   // fp32 is for Node parity tests; too large for a tab
    const cached = store.get(cacheKey(model, name)) ? ", cached" : "";
    sel.add(new Option(`${name} (${(v.bytes / 1e6).toFixed(0)} MB${cached})`, name));
  }
  const want = store.get(`variant:${model}`);
  if (want && [...sel.options].some((o) => o.value === want)) sel.value = want;
  updateLoadButton();
}

function updateLoadButton() {
  const model = $<HTMLSelectElement>("model").value, variant = $<HTMLSelectElement>("variant").value;
  const cached = !!store.get(cacheKey(model, variant));
  $("load").textContent = cached ? "Load (cached)" : "Download & load";
}

$("model").onchange = () => { store.set("model", $<HTMLSelectElement>("model").value); void refreshVariants(); };
$("variant").onchange = () => { store.set(`variant:${$<HTMLSelectElement>("model").value}`, $<HTMLSelectElement>("variant").value); updateLoadButton(); };
$("device").onchange = () => store.set("device", device.value);
await refreshVariants();

const status = (s: string, cls = "") => { const el = $("status"); el.textContent = s; el.className = cls; };
const send = (m: WorkerRequest) => worker.postMessage(m);
const setBusy = (busy: boolean) => {
  $<HTMLButtonElement>("run").disabled = $<HTMLButtonElement>("separate").disabled = busy || !ready;
  $<HTMLButtonElement>("load").disabled = busy;
  document.body.classList.toggle("busy", busy);
};
let ready = false;
let loadStart = 0;
const mb = (n: number) => (n / 1e6).toFixed(0);

function load() {
  const model = $<HTMLSelectElement>("model").value, variant = $<HTMLSelectElement>("variant").value;
  ready = false; loadStart = performance.now(); progress.clear();
  setBusy(true);
  $("progress").classList.add("active");
  $("bar").style.width = "0%";
  status("Starting…");
  send({ type: "load", baseUrl: modelUrl(model), variant, device: device.value as "webgpu" | "wasm", verbose });
}
$("load").onclick = load;

$("reset").onclick = async () => {
  for (const k of await caches.keys()) await caches.delete(k);
  for (const m of models) for (const v of ["q8f32", "q8", "fp16"]) localStorage.removeItem(`kev-web:${cacheKey(m, v)}`);
  await refreshVariants();
  status("Cache cleared. The next load downloads the weights again.");
};

const progress = new Map<string, { loaded: number; total: number }>();
worker.onmessage = (e: MessageEvent<WorkerResponse>) => {
  const m = e.data;
  if (m.type === "progress") {
    progress.set(m.file, m);
    const loaded = [...progress.values()].reduce((s, p) => s + p.loaded, 0), total = [...progress.values()].reduce((s, p) => s + p.total, 0);
    $("bar").style.width = `${(100 * loaded) / Math.max(total, 1)}%`;
    const secs = (performance.now() - loadStart) / 1000;
    const rate = loaded / Math.max(secs, 0.001);
    const left = total > loaded ? ` · ${Math.ceil((total - loaded) / Math.max(rate, 1))} s left` : "";
    const mbs = rate / 1e6;
    status(`${mb(loaded)} / ${mb(total)} MB · ${mbs < 10 ? mbs.toFixed(1) : mbs.toFixed(0)} MB/s${left}`);
  } else if (m.type === "phase") {
    if (m.phase === "manifest") status("Fetching the model manifest…");
    else if (m.phase === "session") { $("bar").style.width = "100%"; status("Uploading weights to the GPU…"); }
    else if (m.phase === "warmup") status("Compiling shaders…");
  } else if (m.type === "ready") {
    ready = true;
    $("progress").classList.remove("active");
    store.set(cacheKey($<HTMLSelectElement>("model").value, m.variant), "1");
    void refreshVariants();
    status(`Ready · ${m.variant} on ${m.device} · load ${(m.loadMs / 1000).toFixed(1)} s, warm-up ${(m.warmupMs / 1000).toFixed(1)} s`);
    setBusy(false);
  } else if (m.type === "partial") {
    pending.get(m.id)?.onPartial?.(m.qid, m.answer as Answer);
  } else if (m.type === "result") {
    pending.get(m.id)?.resolve(m.response as SystemOneResponse); pending.delete(m.id);
  } else if (m.type === "error") {
    $("progress").classList.remove("active");
    if (m.id !== undefined) { pending.get(m.id)?.reject(new Error(m.message)); pending.delete(m.id); }
    else { status(`Error: ${m.message}`, "err"); setBusy(false); }
  }
};

function systemOne(req: unknown, mode: "packed" | "separate" | "probs" = "packed", onPartial?: (qid: string, a: Answer) => void): Promise<SystemOneResponse> {
  const id = nextId++;
  return new Promise((resolve, reject) => { pending.set(id, { resolve, reject, onPartial }); send({ type: "run", id, request: req, mode }); });
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

/** One block per question, created empty so the layout is final before the first answer lands. */
function skeleton(ids: string[]) {
  const el = $("answers"); el.innerHTML = "";
  const meta = document.createElement("div"); meta.className = "meta"; meta.id = "run-meta"; meta.textContent = "running…";
  el.append(meta);
  for (const id of ids) {
    const q = document.createElement("div"); q.className = "q pending"; q.id = `q-${id}`;
    const h = document.createElement("h3"); h.append(id, Object.assign(document.createElement("small"), { textContent: "…" }));
    q.append(h, Object.assign(document.createElement("div"), { className: "rows" }));
    el.append(q);
  }
}

function fill(id: string, a: Answer, cmp?: Answer) {
  const q = document.getElementById(`q-${id}`);
  if (!q) return;
  q.classList.remove("pending");
  q.querySelector("h3 small")!.textContent = `${a.type} · ${summary(a)}`;
  const rows = q.querySelector(".rows") as HTMLElement; rows.innerHTML = "";
  const bs = bars(a); const top = Math.max(...bs.map((b) => b.p));
  for (const b of bs) {
    const row = document.createElement("div"); row.className = `opt${b.p === top ? " top" : ""}`;
    row.innerHTML = `<span class="name"></span><span class="track"><span class="fill"></span></span><span class="val"></span>`;
    const name = row.querySelector(".name") as HTMLElement;
    name.textContent = b.name; name.title = b.name;
    (row.querySelector(".val") as HTMLElement).textContent = b.p.toFixed(2);
    rows.append(row);
    requestAnimationFrame(() => { (row.querySelector(".fill") as HTMLElement).style.width = `${b.p * 100}%`; });   // animate from 0
  }
  if (cmp) {
    const c = document.createElement("div"); c.className = "cmp";
    c.textContent = `separate pass: ${summary(cmp)}${JSON.stringify(cmp) === JSON.stringify(a) ? " (identical)" : ""}`;
    q.append(c);
  }
}

async function run(mode: "packed" | "separate") {
  let req: { questions?: Record<string, unknown> };
  try { req = JSON.parse(request.value); } catch (e) { $("answers").innerHTML = `<p class="err">Invalid JSON: ${(e as Error).message}</p>`; return; }
  setBusy(true);
  skeleton(Object.keys(req.questions ?? {}));
  try {
    const r = await systemOne(req, "packed", (qid, a) => fill(qid, a));
    const other = mode === "separate" ? await systemOne(req, "separate") : undefined;
    for (const [id, a] of Object.entries(r.answers)) fill(id, a, other?.answers[id]);
    $("run-meta")!.textContent = `${r.latency_ms} ms · ${r.usage.input_tokens} input tokens` + (other ? ` · separate: ${other.latency_ms} ms` : "");
    $("raw").textContent = JSON.stringify(other ? { packed: r, separate: other } : r, null, 2);
  } catch (e) {
    const p = document.createElement("p"); p.className = "err"; p.textContent = (e as Error).message; $("answers").replaceChildren(p);
  } finally {
    setBusy(false);
  }
}
$("run").onclick = () => run("packed");
$("separate").onclick = () => run("separate");
document.addEventListener("keydown", (e) => { if ((e.metaKey || e.ctrlKey) && e.key === "Enter" && !$<HTMLButtonElement>("run").disabled) run("packed"); });

/** Published numbers for the models table; measured on an M4 Max (see the repo README). */
const MODEL_FACTS: Record<string, { size: string; acc: string; ms: string }> = {
  "kev-0.8b": { size: "822 MB", acc: "0.657 / 0.488", ms: "112 ms" },
  "kev-4b": { size: "4.7 GB", acc: "0.770 / 0.339", ms: "364 ms" },
  "kev-9b": { size: "—", acc: "—", ms: "—" },
};
const table = document.getElementById("model-table");
if (table) table.innerHTML = models.map((m) => {
  const f = MODEL_FACTS[m];
  return `<tr><td><code>${m}</code></td><td>${f.size}</td><td>${f.acc}</td><td>${f.ms}</td></tr>`;
}).join("");

setBusy(false);
if (store.get(cacheKey($<HTMLSelectElement>("model").value, $<HTMLSelectElement>("variant").value))) load();   // weights are local: start without a click
else status("Pick a model and press Download & load.");
