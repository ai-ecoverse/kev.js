"""Assemble the web bundle: manifest.json + tokenizer + pointer head + one directory per ONNX variant.

    uv run python -m kev_web_export.package --build build/kev-0.8b --out ../dist/models/kev-0.8b \
        --variant q8=web-q8 --variant fp16=web-fp16 --fixtures ../fixtures/kev-0.8b.json

Large files are hard-linked, not copied. With --fixtures, each variant's parity against the PyTorch reference is
measured (CPU EP) and recorded in the manifest."""
import argparse, json, os, shutil
import onnx
from .ort_runtime import OrtKev

ONNX_TYPES = {onnx.TensorProto.FLOAT16: "float16", onnx.TensorProto.FLOAT: "float32", onnx.TensorProto.INT64: "int64"}


def io_info(model_path):
    m = onnx.load(model_path, load_external_data=False)
    def info(v, inp):
        shape = [d.dim_param or d.dim_value for d in v.type.tensor_type.shape.dim]
        out = {"name": v.name, "type": ONNX_TYPES[v.type.tensor_type.elem_type], "shape": shape}
        if inp and v.name.startswith("past_key_values."):
            out["empty"] = [1, shape[1], 0, 256]          # kv_cache_dim = head_dim of the full-attention layers
        elif inp and v.name.startswith("past."):
            out["empty"] = [1] + shape[1:]
        return out
    return [info(v, True) for v in m.graph.input], [info(v, False) for v in m.graph.output]


def link(src, dst):
    if os.path.exists(dst): os.remove(dst)
    try: os.link(src, dst)
    except OSError: shutil.copy(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True, help="merge output dir (kev.json, head.safetensors, tokenizer/, variant dirs)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", action="append", required=True, help="name=dir, e.g. q8=web-q8")
    ap.add_argument("--fixtures")
    a = ap.parse_args()
    kev = json.load(open(f"{a.build}/kev.json"))
    os.makedirs(a.out, exist_ok=True)
    for f in ("tokenizer.json", "tokenizer_config.json"): link(f"{a.build}/tokenizer/{f}", f"{a.out}/{f}")
    link(f"{a.build}/head.safetensors", f"{a.out}/head.safetensors")
    fixtures = json.load(open(a.fixtures))["fixtures"] if a.fixtures else None
    variants = {}
    for spec in a.variant:
        name, d = spec.split("=", 1)
        os.makedirs(f"{a.out}/{name}", exist_ok=True)
        link(f"{a.build}/{d}/model.onnx", f"{a.out}/{name}/model.onnx")
        data = sorted((f for f in os.listdir(f"{a.build}/{d}") if f.startswith("model.onnx.data")), key=lambda f: (len(f), f))
        for f in data: link(f"{a.build}/{d}/{f}", f"{a.out}/{name}/{f}")
        inputs, outputs = io_info(f"{a.out}/{name}/model.onnx")
        v = {"model": f"{name}/model.onnx", "data": [f"{name}/{f}" for f in data],
             "bytes": sum(os.path.getsize(f"{a.out}/{name}/{f}") for f in ["model.onnx", *data]),
             "io_dtype": next(o["type"] for o in outputs if o["name"] == "hidden_states"),
             "sizes": {**{f"{name}/{f}": os.path.getsize(f"{a.out}/{name}/{f}") for f in ["model.onnx", *data]},
                       "head.safetensors": os.path.getsize(f"{a.out}/head.safetensors")},
             "inputs": inputs, "outputs": outputs}
        if fixtures:
            import numpy as np
            rt = OrtKev(f"{a.out}/{name}/model.onnx", f"{a.out}/head.safetensors")
            worst, flips, n = 0.0, 0, 0
            for f in fixtures:
                for g, ref in zip(rt.probs(f["encoding"]), f["probs"]):
                    worst = max(worst, float(np.abs(g - np.array(ref)).max())); flips += int(g.argmax() != int(np.argmax(ref))); n += 1
            v["parity"] = {"max_abs_dp": round(worst, 6), "argmax_flips": flips, "questions": n}
            print(name, v["parity"])
        variants[name] = v
    manifest = {"name": os.path.basename(os.path.normpath(a.out)), **{k: kev[k] for k in ("run", "base", "hidden_size", "head_dim", "special", "max_state", "max_branch")},
                "files": {"head": "head.safetensors", "tokenizer": "tokenizer.json", "tokenizer_config": "tokenizer_config.json"}, "variants": variants}
    json.dump(manifest, open(f"{a.out}/manifest.json", "w"), indent=2)
    print(f"{a.out}/manifest.json: {', '.join(f'{k} {v['bytes'] / 1e6:.0f} MB' for k, v in variants.items())}")


if __name__ == "__main__":
    main()
