"""PyTorch reference fixtures for image requests: Kev + the stock vision tower in fp32 on the CPU (exact path).

    uv run --group vision python -m kev_web_export.vision_fixtures --run jaredpalmer/kev-4b@<sha> \\
        --set ../eval/vision-v1 --set ../eval/vision-v2 --out ../fixtures/kev-4b-vision.json

One fixture per image (the `image` condition of kev_web_export.vision_eval: the image, then the item's context). Each
records what the browser must reproduce: the request, the processor's patch grid and statistics of pixel_values (for
the preprocessing port) and the raw (T = 1) probabilities per question. Text-only parity is the existing
fixtures/<model>.json, run through the spliced graph with a zero image row."""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from .vision import VisionKev
from .vision_eval import request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="jaredpalmer/kev-4b@4bc64c6b4c4881148661ffb823ce21fcfdc79a0e")
    ap.add_argument("--set", action="append", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(os.cpu_count())
    vk = VisionKev(a.run, a.device)
    fixtures = []
    for s in a.set:
        data = json.load(open(f"{s}/questions.json"))
        for it in data["items"]:
            path = os.path.join(s, it["image"])
            img = Image.open(path).convert("RGB")
            px = vk.proc(images=[img], return_tensors="pt")
            pv = px["pixel_values"].double()
            t0 = time.perf_counter()
            emb = vk.image_embeds(img)
            t1 = time.perf_counter()
            req = request(it, it["context"])
            logits, meta = vk.logits(req, embeds=emb)
            t2 = time.perf_counter()
            _, gh, gw = emb
            fixtures.append({"set": data["version"], "id": it["id"], "image": os.path.relpath(os.path.abspath(path), ROOT), "size": list(img.size),
                             "request": req, "labels": [it["questions"][m["id"]]["label"] for m in meta],
                             "grid": [gh, gw], "image_tokens": int(emb[0].shape[0]),
                             "pixels": {"shape": list(pv.shape), "sum": float(pv.sum()), "sum_sq": float((pv * pv).sum()),
                                        "head": pv.flatten()[:64].tolist(), "tail": pv.flatten()[-64:].tolist()},
                             "probs": [F.softmax(z, -1).tolist() for z in logits], "logits": [z.tolist() for z in logits],
                             "torch_ms": {"vision": round((t1 - t0) * 1e3), "decoder": round((t2 - t1) * 1e3)}})
            print(f"{data['version']}/{it['id']}: grid {gh}x{gw}, {emb[0].shape[0]} image tokens, vision {t1 - t0:.1f}s, decoder {t2 - t1:.1f}s", flush=True)
    json.dump({"run": vk.run, "base": f"{vk.meta.base}@{vk.meta.base_revision}", "device": a.device, "dtype": "float32",
               "max_pixels": vk.proc.size["longest_edge"], "image_token": vk.image_token, "vision_start": vk.vs, "vision_end": vk.ve,
               "fixtures": fixtures}, open(a.out, "w"))
    print(f"{len(fixtures)} fixtures -> {a.out}")


if __name__ == "__main__":
    main()
