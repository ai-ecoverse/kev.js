"""Parity of an ONNX vision graph + spliced decoder with the PyTorch image fixtures (kev_web_export.vision_fixtures).

    uv run --group vision python -m kev_web_export.vision_parity --decoder build/kev-4b/fp32-vision/model.onnx \\
        --vision build/kev-4b-vision/fp32.onnx --build build/kev-4b --vision-json build/kev-4b-vision/vision.json \\
        --fixtures ../fixtures/kev-4b-vision.json [--text-fixtures ../fixtures/kev-4b.json] [--out report.json]

Reports, per eval set, the mean over questions of each question's largest |dp| and the maximum (kev.js's test/parity.ts
definition), answers changed against the reference and their reference margins, accuracy of both, and the latency
split (Qwen's processor, the vision graph, the state pass, the question branches). --text-fixtures runs kev.js's
text-only fixtures through the spliced graph with a zero image row: the text path must not move."""
import argparse, json, os, time
import numpy as np
from PIL import Image
from . import KEV_ROOT  # noqa: F401
from kev.model import load_tokenizer
from .ort_vision import OrtVisionKev, vision_inputs

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def check_vision_inputs(cfg, base):
    """The numpy port against transformers' helpers and rotary module, on a few grids."""
    import torch
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionRotaryEmbedding
    from transformers.vision_utils import get_vision_interpolation_indices_and_weights, get_vision_position_ids
    vc = AutoConfig.from_pretrained(base).vision_config
    rot = Qwen3_5VisionRotaryEmbedding(vc)
    worst = 0.0
    for gh, gw in [(28, 28), (38, 60), (18, 32)]:
        thw = torch.tensor([[1, gh, gw]])
        idx, w = get_vision_interpolation_indices_and_weights(thw, num_grid_per_side=cfg["num_grid_per_side"], mode="bilinear", align_corners=True,
                                                              spatial_merge_size=cfg["merge_size"])
        cos, sin = rot(torch.zeros(1), get_vision_position_ids(thw, cfg["merge_size"]))
        i2, w2, c2, s2 = vision_inputs(gh, gw, cfg)
        if not np.array_equal(idx.numpy(), i2): raise SystemExit(f"position-table indices differ at grid {gh}x{gw}")
        worst = max(worst, float(np.abs(w.numpy() - w2).max()), float(np.abs(cos.numpy() - c2).max()), float(np.abs(sin.numpy() - s2).max()))
    print(f"vision_inputs vs transformers: indices equal, max |d| weights/cos/sin {worst:.2e}")


def compare(rows):
    d = [max(abs(a - b) for a, b in zip(g, r)) for g, r, _ in rows]
    flips = [{"at": at, "margin": round(float(max(r) - r[int(np.argmax(g))]), 4)} for g, r, at in rows if np.argmax(g) != np.argmax(r)]
    return {"questions": len(rows), "mean_abs_dp": float(np.mean(d)), "max_abs_dp": float(np.max(d)), "flips": flips}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoder", required=True); ap.add_argument("--vision", required=True)
    ap.add_argument("--build", required=True, help="dir with kev.json and head.safetensors")
    ap.add_argument("--vision-json", required=True)
    ap.add_argument("--fixtures", required=True); ap.add_argument("--text-fixtures")
    ap.add_argument("--out")
    a = ap.parse_args()
    from huggingface_hub import snapshot_download
    from transformers import AutoImageProcessor
    kev = json.load(open(f"{a.build}/kev.json"))
    fx = json.load(open(a.fixtures))
    if fx["run"] != kev["run"]: raise SystemExit(f"{a.fixtures} is from {fx['run']}, the build from {kev['run']}")
    cfg = json.load(open(a.vision_json))
    base = snapshot_download(kev["base"], revision=kev["base_revision"])
    check_vision_inputs(cfg, base)
    proc = AutoImageProcessor.from_pretrained(base, size={"shortest_edge": cfg["min_pixels"], "longest_edge": cfg["max_pixels"]})
    if cfg["max_pixels"] != fx["max_pixels"]: raise SystemExit(f"vision.json caps images at {cfg['max_pixels']} pixels, the fixtures at {fx['max_pixels']}")
    tok = load_tokenizer(kev["base"], revision=kev["base_revision"])
    t0 = time.perf_counter()
    rt = OrtVisionKev(a.decoder, f"{a.build}/head.safetensors", a.vision, cfg, tok, proc)
    load_s = time.perf_counter() - t0
    report = {"run": kev["run"], "decoder": a.decoder, "vision": a.vision, "load_s": round(load_s, 1), "sets": {}}
    by_set, timing = {}, {}
    for f in fx["fixtures"]:
        img = Image.open(os.path.join(ROOT, f["image"])).convert("RGB")
        emb = rt.embeds(img)
        got = rt.probs(f["request"], embeds=emb)
        ids = list(f["request"]["questions"])
        rows = by_set.setdefault(f["set"], [])
        for k, (g, r) in enumerate(zip(got, f["probs"])):
            rows.append((g.tolist(), r, f"{f['id']}/{ids[k]}", f["labels"][k]))
        for k, v in rt.timing.items(): timing.setdefault(f["set"], {}).setdefault(k, []).append(v)
    for s, rows in by_set.items():
        c = compare([(g, r, at) for g, r, at, _ in rows])
        c["accuracy"] = float(np.mean([np.argmax(g) == y for g, _, _, y in rows])); c["ref_accuracy"] = float(np.mean([np.argmax(r) == y for _, r, _, y in rows]))
        c["median_ms"] = {k: round(float(np.median(v)), 1) for k, v in timing[s].items()}
        report["sets"][s] = c
        print(f"{s}: mean |dp| {c['mean_abs_dp']:.2e}, max {c['max_abs_dp']:.2e} over {c['questions']} questions, {len(c['flips'])} flips {c['flips']}; "
              f"acc {c['accuracy']:.3f} (ref {c['ref_accuracy']:.3f}); median ms {c['median_ms']}")
    if a.text_fixtures:
        tf = json.load(open(a.text_fixtures))
        rows = []
        for f in tf["fixtures"]:
            for k, (g, r) in enumerate(zip(rt.kev.probs(f["encoding"]), f["probs"])): rows.append((g.tolist(), r, f"{f['name']}/q{k}"))
        c = compare(rows); report["text"] = c
        print(f"text fixtures through the spliced graph: mean |dp| {c['mean_abs_dp']:.2e}, max {c['max_abs_dp']:.2e} over {c['questions']} questions, {len(c['flips'])} flips")
    if a.out: json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
