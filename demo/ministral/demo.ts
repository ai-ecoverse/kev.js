// Demo page: three task shapes on one KevMinistral instance in a worker.
const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
// Weights and example screenshots come from Hugging Face, pinned to a commit so Cache Storage never mixes revisions.
// ?base=<url> points at another bundle (e.g. a local one), ?samples=<url> at another samples.json.
const REPO = "https://huggingface.co/ai-ecoverse/kev-ministral/resolve/79c6ec85c85610d33de7826840adf086ae7e30a8";
const params = new URLSearchParams(location.search);
const BASE = params.get("base") ?? `${REPO}/kev-ministral-3b`;
const SAMPLES = params.get("samples") ?? `${REPO}/demo/samples.json`;

const worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
let nextId = 1;
const pending = new Map<number, { onMsg: (m: any) => void; resolve: (v: any) => void; reject: (e: Error) => void }>();
worker.onmessage = (e) => {
  const m = e.data, p = pending.get(m.id);
  if (!p) return;
  if (m.error) { pending.delete(m.id); p.reject(new Error(m.error)); }
  else if (m.done) { pending.delete(m.id); p.resolve(m.done); }
  else p.onMsg(m);
};
function call(msg: object, onMsg: (m: any) => void = () => {}): Promise<any> {
  const id = nextId++;
  return new Promise((resolve, reject) => { pending.set(id, { onMsg, resolve, reject }); worker.postMessage({ id, ...msg }); });
}

// ------------------------------------------------------------------ load
const runButtons = ["decide", "describe", "complete"].map((id) => $<HTMLButtonElement>(id));
let busy = false;
const setBusy = (b: boolean) => { busy = b; for (const x of runButtons) x.disabled = b || !loaded; };
let loaded = false;
const progRows = new Map<string, HTMLElement>();
const gb = (n: number) => (n / 1e9).toFixed(2) + " GB";
$("load").onclick = async () => {
  $<HTMLButtonElement>("load").disabled = true;
  $("status").textContent = "loading…";
  try {
    const r = await call({ type: "load", base: BASE }, (m) => {
      if (!m.progress) return;
      const { file, loaded: l, total, cached } = m.progress;
      if (total < 1e6) return;   // graph files and configs: not worth a row
      let row = progRows.get(file);
      if (!row) {
        row = document.createElement("div"); row.className = "prog";
        row.innerHTML = `<span>${file}</span><div class="track"><div class="fill" style="width:0"></div></div><span></span>`;
        $("progress").appendChild(row); progRows.set(file, row);
      }
      (row.querySelector(".fill") as HTMLElement).style.width = `${(100 * l) / total}%`;
      (row.lastElementChild as HTMLElement).textContent = cached ? "cached" : `${gb(l)}`;
    });
    loaded = true; setBusy(false);
    $("status").textContent = `loaded once in ${r.seconds.toFixed(1)} s · ${r.gpu}`;
    $("load").classList.add("hidden");
    setTimeout(() => { $("progress").innerHTML = ""; }, 1500);
  } catch (e: any) {
    $("status").textContent = "load failed: " + e.message;
    $<HTMLButtonElement>("load").disabled = false;
  }
};

// ------------------------------------------------------------------ tabs
document.querySelectorAll<HTMLButtonElement>(".tab").forEach((t) => (t.onclick = () => {
  document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("on", x === t));
  for (const name of ["decide", "describe", "complete"]) $(`tab-${name}`).classList.toggle("hidden", name !== t.dataset.tab);
}));

// ------------------------------------------------------------------ decide
interface Sample { id: string; source: string; site: string; goal: string; state: string; image: string; gold: string; browser_correct: boolean;
                   question: { type: string; instructions: string; criteria: Record<string, string> } }
const samples: Sample[] = (await (await fetch(SAMPLES)).json()).map((s: Sample) => ({ ...s, image: new URL(s.image, SAMPLES).href }));
let decideImage: string | Blob | undefined;
let currentGold: string | undefined;
const sel = $<HTMLSelectElement>("sample");
samples.forEach((s, i) => sel.add(new Option(`${s.source} · ${s.site} — ${s.goal.slice(0, 70)}`, String(i))));
function showSample(i: number) {
  const s = samples[i];
  $<HTMLImageElement>("shot").src = s.image; decideImage = s.image; currentGold = s.gold;
  $<HTMLTextAreaElement>("state").value = s.state;
  $<HTMLInputElement>("qtext").value = s.question.instructions;
  $<HTMLTextAreaElement>("opts").value = Object.entries(s.question.criteria).map(([k, v]) => `${k}: ${v}`).join("\n");
  $("bars").innerHTML = ""; $("verdict").textContent = ""; $("dmeta").textContent = "";
}
sel.onchange = () => showSample(Number(sel.value));
showSample(0);
$("upload-btn").onclick = () => $("upload").click();
$<HTMLInputElement>("upload").onchange = () => {
  const f = $<HTMLInputElement>("upload").files?.[0];
  if (!f) return;
  decideImage = f; currentGold = undefined;
  $<HTMLImageElement>("shot").src = URL.createObjectURL(f);
};
$("decide").onclick = async () => {
  if (busy) return;
  const criteria: Record<string, string> = {};
  for (const line of $<HTMLTextAreaElement>("opts").value.split("\n")) {
    // keys contain colons themselves (click:e17, type:e482), so key and description are split at the first ": "
    const at = line.indexOf(": "); if (!line.trim()) continue;
    if (at < 0) criteria[line.trim()] = line.trim(); else criteria[line.slice(0, at).trim()] = line.slice(at + 2).trim();
  }
  if (Object.keys(criteria).length < 2) { $("verdict").textContent = "need at least two options"; return; }
  const req = { state: $<HTMLTextAreaElement>("state").value, questions: { action: { type: "choice", instructions: $<HTMLInputElement>("qtext").value, criteria } } };
  setBusy(true); $("dmeta").textContent = "deciding…"; $("bars").innerHTML = "";
  try {
    const noimg = $<HTMLInputElement>("noimg").checked;
    const r = await call({ type: "decide", req, image: noimg ? undefined : decideImage });
    const p: number[] = r.probs[0], keys: string[] = r.keys[0];
    const order = keys.map((k, i) => [k, p[i]] as [string, number]).sort((a, b) => b[1] - a[1]);
    const shown = order.slice(0, 10);
    if (currentGold && !shown.some(([k]) => k === currentGold)) { const g = order.find(([k]) => k === currentGold); if (g) shown.push(g); }
    $("bars").innerHTML = shown.map(([k, v], i) =>
      `<div class="b${i === 0 ? " top" : ""}${k === currentGold ? " gold" : ""}"><span class="k" title="${k}: ${criteria[k] ?? ""}">${k} <span class="note">${(criteria[k] ?? "").slice(0, 40)}</span></span>` +
      `<div class="track"><div class="fill" style="width:${(100 * v).toFixed(1)}%"></div></div><span>${(100 * v).toFixed(1)}%</span></div>`).join("");
    const choice = order[0][0];
    const v = $("verdict");
    if (currentGold) { v.className = `verdict meta ${choice === currentGold ? "ok" : "bad"}`; v.textContent = choice === currentGold ? `✓ picked the gold action (${choice})` : `✗ picked ${choice}; gold is ${currentGold}`; }
    else { v.className = "verdict meta"; v.textContent = `picked ${choice}`; }
    $("dmeta").textContent = `${r.tokens} tokens in context (${r.imageTokens} from the screenshot) · ${(r.ms.total / 1000).toFixed(2)} s`;
  } catch (e: any) { $("dmeta").textContent = "error: " + e.message; }
  setBusy(false);
};

// ------------------------------------------------------------------ describe
let describeImage: string | Blob | undefined;
const dsel = $<HTMLSelectElement>("dsample");
samples.forEach((s, i) => dsel.add(new Option(`${s.site} — ${s.goal.slice(0, 60)}`, String(i))));
dsel.onchange = () => { describeImage = samples[Number(dsel.value)].image; $<HTMLImageElement>("dshot").src = describeImage as string; };
dsel.onchange(new Event("change"));
$("dupload-btn").onclick = () => $("dupload").click();
$<HTMLInputElement>("dupload").onchange = () => {
  const f = $<HTMLInputElement>("dupload").files?.[0];
  if (f) { describeImage = f; $<HTMLImageElement>("dshot").src = URL.createObjectURL(f); }
};

async function generate(prompt: string, maxNew: number, temperature: number, image: string | Blob | undefined, out: HTMLElement, meta: HTMLElement) {
  if (busy) return;
  setBusy(true);
  const shown = prompt.replace("[IMG]", "🖼").replace(/</g, "&lt;");
  out.innerHTML = `<span class="p">${shown}</span><span class="g"></span>`;
  const g = out.querySelector(".g") as HTMLElement;
  meta.textContent = "generating…";
  try {
    const r = await call({ type: "generate", prompt, maxNew, image, temperature }, (m) => { if (m.token) g.textContent = m.token.text; });
    g.textContent = r.text.replace("</s>", "");
    meta.textContent = `${r.inputTokens} prompt tokens · ${r.tokens} generated · prefill ${(r.ms.prefill / 1000).toFixed(2)} s · ${r.ms.perToken.toFixed(0)} ms/token`;
  } catch (e: any) { meta.textContent = "error: " + e.message; }
  setBusy(false);
}
$("describe").onclick = () => {
  const p = $<HTMLTextAreaElement>("dprompt").value;
  generate(p.includes("[IMG]") ? p : "[IMG]\n" + p, Number($<HTMLInputElement>("dmax").value), Number($<HTMLInputElement>("dtemp").value), describeImage, $("dout"), $("dgmeta"));
};

// ------------------------------------------------------------------ complete
const presets: [string, string][] = [
  ["Summarize (TL;DR)", "Article: The city council voted on Tuesday to close two downtown streets to cars every Sunday from May to September. Supporters said the change would bring more people to local shops and cafes, while some business owners worried that customers who drive in from the suburbs would stay away. The trial will be reviewed in October.\n\nTL;DR:"],
  ["Summarize release notes", "Release notes: Version 2.4 adds offline mode, so notes you write without a connection are saved locally and synced when you reconnect. Search is now about three times faster on large notebooks. The old export format is deprecated and will be removed in version 3.0.\n\nTL;DR:"],
  ["Complete a list", "The three most important things to check before deploying a web application are"],
  ["Question answering", "Q: What is the capital of Australia?\nA:"],
  ["Write code", "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n"],
  ["German", "Die Stadt liegt an einem breiten Fluss, über den drei Brücken führen. Im Sommer"],
];
const psel = $<HTMLSelectElement>("preset");
presets.forEach(([n], i) => psel.add(new Option(n, String(i))));
psel.onchange = () => { $<HTMLTextAreaElement>("cprompt").value = presets[Number(psel.value)][1]; };
psel.onchange(new Event("change"));
$("complete").onclick = () => generate($<HTMLTextAreaElement>("cprompt").value, Number($<HTMLInputElement>("cmax").value),
  Number($<HTMLInputElement>("ctemp").value), undefined, $("cout"), $("cmeta"));
