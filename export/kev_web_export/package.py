"""Assemble the web bundle: manifest.json + tokenizer + pointer head + one directory per ONNX variant.

    uv run python -m kev_web_export.package --build build/kev-0.8b --out ../dist/models/kev-0.8b \
        --variant q8=web-q8 --variant fp16=web-fp16 --fixtures ../fixtures/kev-0.8b.json

Large files are hard-linked, not copied. The manifest's `revision` is a digest of every file it names, and browsers
key their cache by it, so a bundle rebuilt for the same checkpoint (a spliced decoder at the text graph's URL) is never
served from stale cached bytes; `run` stays as provenance. Each variant records `max_positions`, the rows its rotary
tables keep. With --fixtures, each variant's parity against the PyTorch reference is
measured (CPU EP) and recorded in the manifest.

--vision <kev_web_export.vision_onnx out dir> makes a bundle that takes images: the tower's browser graph goes under
r-<sha>/vision/ and manifest.json gains `vision` (files, sizes, vision.json as `config`, and --vision-parity's report
if given). Its variants must be spliced decoders (kev_web_export.splice), which take `image_embeds`.

    uv run python -m kev_web_export.package --update ../public/models/kev-0.8b

adds `revision` and `max_positions` to an existing bundle's manifest without touching its files."""
import argparse, hashlib, json, os, shutil, sys
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


def max_positions(model_path):
    """How many positions the graph's rotary tables cover (postprocess.py --rope-positions): longer rows can't run."""
    m = onnx.load(model_path, load_external_data=False)
    return next(t.dims[0] for t in m.graph.initializer if t.name == "cos_cache")


def bundle_files(manifest):
    """Every file a manifest names, besides itself."""
    files = set(manifest["files"].values())
    for v in manifest["variants"].values(): files |= {v["model"], *v["data"]}
    if manifest.get("vision"): files |= {manifest["vision"]["model"], *manifest["vision"]["data"]}
    return files


def content_revision(out, files):
    """A digest of the files' contents, which clients key their cache by."""
    h = hashlib.sha256()
    for f in sorted(files):
        d = hashlib.sha256()
        with open(f"{out}/{f}", "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b""): d.update(chunk)
        h.update(f"{f}\0{d.hexdigest()}\n".encode())
    return h.hexdigest()[:16]


def with_revision(out, manifest):
    """The manifest with max_positions per variant and `revision` right after `run`."""
    for v in manifest["variants"].values(): v["max_positions"] = max_positions(f"{out}/{v['model']}")
    rev = content_revision(out, bundle_files(manifest))
    items = [(k, v) for k, v in manifest.items() if k != "revision"]
    i = next(n for n, (k, _) in enumerate(items) if k == "run") + 1
    return dict(items[:i] + [("revision", rev)] + items[i:])


def update(out):
    manifest = with_revision(out, json.load(open(f"{out}/manifest.json")))
    json.dump(manifest, open(f"{out}/manifest.json", "w"), indent=2)
    print(f"{out}/manifest.json: revision {manifest['revision']}, max_positions {({k: v['max_positions'] for k, v in manifest['variants'].items()})}")


def link(src, dst):
    if os.path.exists(dst): os.remove(dst)
    try: os.link(src, dst)
    except OSError: shutil.copy(src, dst)


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--update": return update(sys.argv[2])
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True, help="merge output dir (kev.json, head.safetensors, tokenizer/, variant dirs)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", action="append", required=True, help="name=dir, e.g. q8=web-q8")
    ap.add_argument("--fixtures")
    ap.add_argument("--vision", help="vision_onnx output dir (web/ and vision.json)")
    ap.add_argument("--vision-parity", action="append", default=[], help="name=report.json from vision_parity, recorded per name")
    a = ap.parse_args()
    kev = json.load(open(f"{a.build}/kev.json"))
    # Everything but the manifest lives under a revision directory, so publishing a new checkpoint never overwrites a
    # file an older manifest points at: the switch is the single commit that replaces manifest.json.
    rev = kev["run"].partition("@")[2][:7] or "local"
    r = f"r-{rev}"
    os.makedirs(f"{a.out}/{r}", exist_ok=True)
    for f in ("tokenizer.json", "tokenizer_config.json"): link(f"{a.build}/tokenizer/{f}", f"{a.out}/{r}/{f}")
    link(f"{a.build}/head.safetensors", f"{a.out}/{r}/head.safetensors")
    fixtures = None
    if a.fixtures:
        fx = json.load(open(a.fixtures))
        # parity is only meaningful against the same checkpoint the weights were exported from
        if fx["run"] != kev["run"]:
            raise SystemExit(f"{a.fixtures} is from {fx['run']}, but {a.build} was exported from {kev['run']}")
        fixtures = fx["fixtures"]
    variants = {}
    for spec in a.variant:
        name, d = spec.split("=", 1)
        vdir = f"{r}/{name}"
        os.makedirs(f"{a.out}/{vdir}", exist_ok=True)
        link(f"{a.build}/{d}/model.onnx", f"{a.out}/{vdir}/model.onnx")
        data = sorted((f for f in os.listdir(f"{a.build}/{d}") if f.startswith("model.onnx.data")), key=lambda f: (len(f), f))
        for f in data: link(f"{a.build}/{d}/{f}", f"{a.out}/{vdir}/{f}")
        inputs, outputs = io_info(f"{a.out}/{vdir}/model.onnx")
        v = {"model": f"{vdir}/model.onnx", "data": [f"{vdir}/{f}" for f in data],
             "bytes": sum(os.path.getsize(f"{a.out}/{vdir}/{f}") for f in ["model.onnx", *data]),
             "io_dtype": next(o["type"] for o in outputs if o["name"] == "hidden_states"),
             "sizes": {**{f"{vdir}/{f}": os.path.getsize(f"{a.out}/{vdir}/{f}") for f in ["model.onnx", *data]},
                       f"{r}/head.safetensors": os.path.getsize(f"{a.out}/{r}/head.safetensors"),
                       **{f"{r}/{f}": os.path.getsize(f"{a.out}/{r}/{f}") for f in ("tokenizer.json", "tokenizer_config.json")}},
             "inputs": inputs, "outputs": outputs}
        if fixtures:
            import numpy as np
            rt = OrtKev(f"{a.out}/{vdir}/model.onnx", f"{a.out}/{r}/head.safetensors")
            worst, flips, n = 0.0, 0, 0
            for f in fixtures:
                for g, ref in zip(rt.probs(f["encoding"]), f["probs"]):
                    worst = max(worst, float(np.abs(g - np.array(ref)).max())); flips += int(g.argmax() != int(np.argmax(ref))); n += 1
            v["parity"] = {"max_abs_dp": round(worst, 6), "argmax_flips": flips, "questions": n}
            print(name, v["parity"])
        variants[name] = v
    vision = None
    if a.vision:
        cfg = json.load(open(f"{a.vision}/vision.json"))
        if cfg["run"] != kev["run"]: raise SystemExit(f"{a.vision} was exported for {cfg['run']}, the decoder from {kev['run']}")
        if not all(any(i["name"] == "image_embeds" for i in v["inputs"]) for v in variants.values()):
            raise SystemExit("--vision needs spliced decoders (kev_web_export.splice): a variant has no image_embeds input")
        vd = f"{r}/vision"
        os.makedirs(f"{a.out}/{vd}", exist_ok=True)
        files = ["model.onnx", *sorted((f for f in os.listdir(f"{a.vision}/web") if f.startswith("model.onnx.data")), key=lambda f: (len(f), f))]
        for f in files: link(f"{a.vision}/web/{f}", f"{a.out}/{vd}/{f}")
        sizes = {f"{vd}/{f}": os.path.getsize(f"{a.out}/{vd}/{f}") for f in files}
        config = {k: v for k, v in cfg.items() if k not in ("run", "base", "inputs", "outputs")}
        vision = {"model": f"{vd}/model.onnx", "data": [f"{vd}/{f}" for f in files[1:]], "bytes": sum(sizes.values()), "sizes": sizes, "config": config}
        if a.vision_parity:
            vision["parity"] = {}
            for spec in a.vision_parity:
                name, path = spec.split("=", 1)
                rep = json.load(open(path))
                vision["parity"][name] = {s: {k: (round(x, 6) if isinstance(x, float) else len(x) if k == "flips" else x) for k, x in c.items()}
                                          for s, c in rep["sets"].items()}
    # drop files from earlier packagings (removed variants, old shard layouts): the directory is published as is
    keep = {"manifest.json", f"{r}/tokenizer.json", f"{r}/tokenizer_config.json", f"{r}/head.safetensors"}
    for v in variants.values(): keep |= {v["model"], *v["data"]}
    if vision: keep |= {vision["model"], *vision["data"]}
    for root, _, fs in os.walk(a.out, topdown=False):
        for f in fs:
            rel = os.path.relpath(os.path.join(root, f), a.out)
            if rel not in keep: os.remove(os.path.join(root, f)); print("removed stale", rel)
        if root != a.out and not os.listdir(root): os.rmdir(root)
    keys = ("run", "base", "hidden_size", "head_dim", "special", "max_state", "max_branch", "temperature")
    manifest = {"name": os.path.basename(os.path.normpath(a.out)), **{k: kev[k] for k in keys if k in kev},
                "files": {"head": f"{r}/head.safetensors", "tokenizer": f"{r}/tokenizer.json", "tokenizer_config": f"{r}/tokenizer_config.json"},
                "variants": variants, **({"vision": vision} if vision else {})}
    manifest = with_revision(a.out, manifest)
    json.dump(manifest, open(f"{a.out}/manifest.json", "w"), indent=2)
    print(f"{a.out}/manifest.json: {', '.join(f'{k} {v['bytes'] / 1e6:.0f} MB' for k, v in variants.items())}")


if __name__ == "__main__":
    main()
