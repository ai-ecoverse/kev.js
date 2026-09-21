"""Fold a Kev LoRA into its Qwen3.5 base (fp32, exact) and write a checkpoint the onnxruntime-genai builder can read.

The builder expects the full Qwen3_5ForConditionalGeneration layout (model.language_model.*, model.visual.*, mtp.*), so
the base checkpoint is copied and only the language-model tensors are replaced by the merged ones. The pointer head and
the tokenizer/encoding constants the browser runtime needs are written next to it."""
import argparse, json, os, shutil
import torch
from safetensors.torch import load_file, save_file
from . import KEV_ROOT  # noqa: F401  (puts kev on sys.path)
from kev.evaluate import load, resolve_run
from kev.model import SPECIAL, MAX_STATE, MAX_BRANCH
from .pin import pin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="jaredpalmer/kev-0.8b", help="Kev run dir or Hub id[@rev]")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    a.run = pin(a.run)                                          # exact commit: the Hub ids are republished in place
    print(f"run: {a.run}")
    run = resolve_run(a.run)
    meta = torch.load(f"{run}/head.pt", map_location="cpu")
    tok, m = load(run, "cpu")                                  # fp32, LoRA merged (kev.evaluate.load)
    merged = {f"model.language_model.{k}": v.detach().contiguous() for k, v in m.lm.state_dict().items()}

    from huggingface_hub import snapshot_download
    base = snapshot_download(meta["base"], revision=meta.get("base_revision"))
    idx = json.load(open(f"{base}/model.safetensors.index.json"))["weight_map"]
    sd = {}
    for f in sorted(set(idx.values())): sd.update(load_file(f"{base}/{f}"))
    missing = [k for k in merged if k not in sd]
    if missing: raise SystemExit(f"merged keys not in base checkpoint: {missing[:5]}")
    changed = sum(not torch.equal(sd[k].float(), v) for k, v in merged.items())
    sd.update(merged)                                          # language model -> fp32 merged; vision/mtp untouched

    ckpt = os.path.join(a.out, "merged")
    os.makedirs(ckpt, exist_ok=True)
    save_file(sd, f"{ckpt}/model.safetensors", metadata={"format": "pt"})
    for f in os.listdir(base):
        if f.endswith((".json", ".txt", ".jinja")) and f != "model.safetensors.index.json":
            shutil.copy(f"{base}/{f}", ckpt)
    cfg = json.load(open(f"{ckpt}/config.json")); cfg.setdefault("text_config", {})["dtype"] = "float32"
    cfg.setdefault("eos_token_id", cfg["text_config"].get("eos_token_id"))   # the builder reads it from the top level
    json.dump(cfg, open(f"{ckpt}/config.json", "w"), indent=2)

    head = {k: v.float().contiguous() for k, v in meta["head"].items()}
    save_file(head, f"{a.out}/head.safetensors")
    kev = {
        "run": a.run, "base": meta["base"], "base_revision": meta.get("base_revision"), "lora": meta.get("lora"),
        "hidden_size": m.lm.config.hidden_size, "head_dim": meta.get("head_dim", 256),
        "temperature": float(meta.get("temperature", 1.0)),
        "special": {name: tok.convert_tokens_to_ids(t) for name, t in zip(["state", "q", "opt", "opt_end", "decide"], SPECIAL)},
        "max_state": MAX_STATE, "max_branch": MAX_BRANCH, "pad_id": tok.pad_token_id if tok.pad_token_id is not None else 0,
    }
    json.dump(kev, open(f"{a.out}/kev.json", "w"), indent=2)
    tok.save_pretrained(f"{a.out}/tokenizer")
    print(f"merged {changed}/{len(merged)} language-model tensors changed -> {ckpt}; head + kev.json + tokenizer -> {a.out}")


if __name__ == "__main__":
    main()
