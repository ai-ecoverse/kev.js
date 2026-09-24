"""Score Kev behind Qwen3.5's stock vision tower on eval/vision-v1, against text-only baselines.

    uv run --group vision python -m kev_web_export.vision_eval --run jaredpalmer/kev-4b@<sha> --device mps \\
        --set ../eval/vision-v1 --out ../eval/vision-v1/results/kev-4b

Conditions (the questions are the same; only the state changes):
  image    the item's image, then its neutral context text
  caption  no image; the context plus a text caption that holds everything the questions need (the text upper bound)
  omitted  no image; the context text only (what Kev can guess from the question alone)
  shuffled another item's image of the same family, then the context (does the answer follow the image?)

Metrics come from kev.metrics (the benchmark's own scorer): accuracy, Brier (sum over options, uniform = 1 - 1/K),
NLL, ECE and mean top probability, at the raw logits (T = 1) and at the checkpoint's serving temperature. Flatness is
the mean entropy divided by log K (1 = uniform). Writes rows.json (per question and condition, with logits) and
report.json, both naming the pinned run."""
import argparse, json, math, os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from . import KEV_ROOT  # noqa: F401
from kev.api import question_keys
from kev.metrics import metrics, probabilities_at_temperature
from .vision import VisionKev

CONDITIONS = ["image", "caption", "omitted", "shuffled"]
# serving temperatures of the night-2 checkpoints whose head.pt predates the fitted value (src/model.ts NIGHT2_TEMPERATURE)
NIGHT2_T = {"225679690cdd1de6fceb1258b1bddf61c493cee9": 2.406050072164233, "4bc64c6b4c4881148661ffb823ce21fcfdc79a0e": 2.1435469250725863,
            "442e597d71840506c326c8c2f5eedd42aeac7bbd": 2.2973967099940698}


def request(item, state):
    return {"state": state, "questions": {k: {kk: vv for kk, vv in q.items() if kk != "label"} for k, q in item["questions"].items()}}


def flatness(rows, temperature):
    return float(np.mean([-(p * np.log(np.maximum(p, 1e-12))).sum() / math.log(len(p)) for p in (probabilities_at_temperature(r, temperature) for r in rows)]))


def summary(rows, temperature):
    out = {}
    for t, name in [(1.0, "raw"), (temperature, "served")]:
        m = metrics(rows, t)
        out[name] = {k: m[k] for k in ("n", "acc", "brier", "nll", "ece", "mean_conf")} | {"flatness": flatness(rows, t)}
    out["chance_acc"] = float(np.mean([1 / len(r["p"]) for r in rows]))
    out["uniform_brier"] = float(np.mean([1 - 1 / len(r["p"]) for r in rows]))
    return out


def accuracy_ci(rows, n=2000, seed=0):
    """95% interval of accuracy, resampling images (an image's questions are not independent)."""
    by = {}
    for r in rows: by.setdefault(r["item"], []).append(float(np.argmax(r["p"]) == r["label"]))
    groups, rng = list(by.values()), np.random.default_rng(seed)
    accs = []
    for _ in range(n):
        pick = [groups[i] for i in rng.integers(0, len(groups), len(groups))]
        accs.append(sum(map(sum, pick)) / sum(map(len, pick)))
    return [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))]


def write_report(rows, items, run, base, version, temperature, out):
    fam = sorted({it["family"] for it in items})
    report = {"run": run, "base": base, "set": version, "temperature": temperature,
              "images": len(items), "questions": sum(len(it["questions"]) for it in items), "conditions": {}}
    for cond in CONDITIONS:
        cr = [r for r in rows if r["condition"] == cond]
        report["conditions"][cond] = summary(cr, temperature) | {"acc_ci95": accuracy_ci(cr),
            "by_type": {t: summary([r for r in cr if r["type"] == t], temperature)["raw"] for t in ("noul", "choice", "score")},
            "by_family": {f: summary([r for r in cr if r["family"] == f], temperature)["raw"]["acc"] for f in fam}}
    key = lambda r: (r["item"], r["question"])
    img = {key(r): np.array(r["p"]) for r in rows if r["condition"] == "image"}
    for other in ("omitted", "shuffled", "caption"):
        o = {key(r): np.array(r["p"]) for r in rows if r["condition"] == other}
        report[f"image_vs_{other}"] = {"mean_max_abs_dp": float(np.mean([np.abs(img[k] - o[k]).max() for k in img])),
                                       "answers_changed": int(sum(img[k].argmax() != o[k].argmax() for k in img))}
    os.makedirs(out, exist_ok=True)
    json.dump(report, open(f"{out}/report.json", "w"), indent=1)
    print(f"{'condition':<9} {'acc':>6} {'95% CI':>13} {'brier':>6} {'brier@T':>7} {'nll':>6} {'conf':>5} {'flat':>5} {'flat@T':>6}")
    for cond, s in report["conditions"].items():
        lo, hi = s["acc_ci95"]
        print(f"{cond:<9} {s['raw']['acc']:6.3f} [{lo:.3f}, {hi:.3f}] {s['raw']['brier']:6.3f} {s['served']['brier']:7.3f} {s['raw']['nll']:6.3f} "
              f"{s['raw']['mean_conf']:5.2f} {s['raw']['flatness']:5.2f} {s['served']['flatness']:6.2f}")
    print(f"chance acc {report['conditions']['image']['chance_acc']:.3f}, uniform brier {report['conditions']['image']['uniform_brier']:.3f}, T = {temperature:.3f} -> {out}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="jaredpalmer/kev-4b@4bc64c6b4c4881148661ffb823ce21fcfdc79a0e")
    ap.add_argument("--set", default="../eval/vision-v1")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rescore", action="store_true", help="recompute report.json from the rows.json already in --out")
    a = ap.parse_args()
    data = json.load(open(f"{a.set}/questions.json"))
    items = data["items"]
    if a.rescore:
        old = json.load(open(f"{a.out}/report.json"))
        write_report(json.load(open(f"{a.out}/rows.json")), items, old["run"], old["base"], data["version"], old["temperature"], a.out)
        return
    vk = VisionKev(a.run, a.device)
    temperature = vk.meta.temperature if vk.meta.temperature != 1.0 else NIGHT2_T.get(vk.run.partition("@")[2], 1.0)
    embeds = {}
    for it in items:
        embeds[it["id"]] = vk.image_embeds(Image.open(f"{a.set}/{it['image']}").convert("RGB"))
    fam = {}
    for it in items: fam.setdefault(it["family"], []).append(it["id"])
    shuffled = {i: ids[(k + 1) % len(ids)] for ids in fam.values() for k, i in enumerate(ids)}

    rows = []
    for n, it in enumerate(items):
        for cond in CONDITIONS:
            state = f"{it['context']}\n{it['caption']}" if cond == "caption" else it["context"]
            e = embeds[it["id"]] if cond == "image" else embeds[shuffled[it["id"]]] if cond == "shuffled" else None
            logits, meta = vk.logits(request(it, state), embeds=e)
            for z, m in zip(logits, meta):
                q = it["questions"][m["id"]]
                rows.append({"item": it["id"], "family": it["family"], "question": m["id"], "type": q["type"], "condition": cond,
                             "keys": question_keys(q["type"], q.get("criteria")), "label": q["label"],
                             "logits": z.tolist(), "p": F.softmax(z, -1).tolist()})
        print(f"[{n + 1}/{len(items)}] {it['id']}: " + "  ".join(
            f"{r['question']} {'ok' if np.argmax(r['p']) == r['label'] else 'x '} {max(r['p']):.2f}" for r in rows if r["item"] == it["id"] and r["condition"] == "image"), flush=True)

    os.makedirs(a.out, exist_ok=True)
    json.dump(rows, open(f"{a.out}/rows.json", "w"))
    write_report(rows, items, vk.run, f"{vk.meta.base}@{vk.meta.base_revision}", data["version"], temperature, a.out)


if __name__ == "__main__":
    main()
