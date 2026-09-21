// Browser parity + accuracy for one packaged model, run in the demo tab: fixture parity against PyTorch, the
// 300-record reference set (accuracy, Brier, |dp| against fp32) and latency. Used as
//   node scripts/cdp.mjs "$(cat scripts/browser-eval.js)('kev-4b', 'q8f32')"
(async (model, variant) => {
  const deadline = Date.now() + 25 * 60 * 1000;
  const until = async (f, what) => { while (!f()) { if (Date.now() > deadline) throw new Error("timeout: " + what); await new Promise(r => setTimeout(r, 300)); } };
  await until(() => typeof window.kev === "object" && document.getElementById("variant").options.length > 0, "page init");
  const sel = (id, v) => { const e = document.getElementById(id); e.value = v; e.dispatchEvent(new Event("change")); };
  sel("model", model);
  await until(() => [...document.getElementById("variant").options].some(o => o.value === variant), "variants");
  sel("variant", variant);
  document.getElementById("load").click();
  const t0 = performance.now();
  await until(() => /Ready|Error/.test(document.getElementById("status").textContent), "load");
  const status = document.getElementById("status").textContent;
  if (/Error/.test(status)) return { status };
  const get = async (f) => (await fetch(`/@fs/Users/trieloff/Developer/ai-ecoverse/kev-web/fixtures/${f}`)).json();
  const am = a => a.indexOf(Math.max(...a));
  const fx = await get(`${model}.json`);
  let fw = 0, ff = 0, fn = 0;
  for (const f of fx.fixtures) (await window.kev.probs(f.record)).forEach((p, k) => { fn++; fw = Math.max(fw, ...p.map((x, j) => Math.abs(x - f.probs[k][j]))); if (am(p) !== am(f.probs[k])) ff++; });
  const ev = await get(`${model}-transfer-v4-dev300.json`);
  let c = 0, cr = 0, b = 0, n = 0, w = 0, s = 0, fl = 0; const t1 = performance.now();
  for (const r of ev.records) (await window.kev.probs(r.record)).forEach((p, k) => {
    const ref = r.probs[k], y = r.labels[k]; n++; c += am(p) === y; cr += am(ref) === y; if (am(p) !== am(ref)) fl++;
    b += p.reduce((t, x, j) => t + (x - (j === y)) ** 2, 0); const d = Math.max(...p.map((x, j) => Math.abs(x - ref[j]))); w = Math.max(w, d); s += d; });
  const perRecord = (performance.now() - t1) / ev.records.length;
  const req = fx.fixtures[0].request; const lat = [];
  for (let i = 0; i < 3; i++) lat.push((await window.kev.systemOne({ ...req, state: req.state + " ".repeat(i + 1) })).latency_ms);
  const cached = (await window.kev.systemOne(req)).latency_ms;
  return { model, variant, status, fixture_run: fx.run, evalset_run: ev.run,
    fixtures: { questions: fn, max_abs_dp: +fw.toFixed(4), flips: ff },
    evalset: { n, accuracy: +(c / n).toFixed(4), ref_accuracy: +(cr / n).toFixed(4), brier: +(b / n).toFixed(4), ref_brier: +ev.summary.brier.toFixed(4), max_abs_dp: +w.toFixed(4), mean_abs_dp: +(s / n).toFixed(4), flips: fl, ms_per_record: +perRecord.toFixed(0) },
    latency_3q_ms: lat, latency_cached_ms: cached };
})
