"""Shrink a builder export for the browser without touching the transformer body.

- The embedding table is the single largest tensor (vocab 248k x hidden) and the builder only quantizes it to int4
  (GatherBlockQuantized). --embed int8 stores it as int8 with one scale per row, dequantized after the lookup with
  standard ops (Gather, Cast, Mul), so every execution provider can run it.
- The rotary cos/sin caches are sized for the base model's 262k context. Kev never sees more than a few thousand
  positions (8,192 tokens per state + question at serving time), so --rope-positions trims them."""
import argparse, os
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="builder output dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--embed", choices=["keep", "int8"], default="int8")
    ap.add_argument("--rope-positions", type=int, default=8192)
    a = ap.parse_args()
    m = onnx.load(f"{a.src}/model.onnx", load_external_data=True)
    g = m.graph
    inits = {t.name: t for t in g.initializer}

    for name in ("cos_cache", "sin_cache"):
        t = inits[name]; arr = numpy_helper.to_array(t)
        if arr.shape[0] > a.rope_positions:
            t.CopyFrom(numpy_helper.from_array(np.ascontiguousarray(arr[: a.rope_positions]), name))

    if a.embed == "int8":
        (node,) = [n for n in g.node if n.op_type == "Gather" and n.input[0] == "model.embed_tokens.weight"]
        w = numpy_helper.to_array(inits["model.embed_tokens.weight"])
        wf = w.astype(np.float32)
        scale = np.maximum(np.abs(wf).max(axis=1), 1e-12) / 127.0
        q = np.clip(np.rint(wf / scale[:, None]), -127, 127).astype(np.int8)
        err = float(np.abs(q.astype(np.float32) * scale[:, None] - wf).max())
        g.initializer.remove(inits["model.embed_tokens.weight"])
        g.initializer.extend([numpy_helper.from_array(q, "model.embed_tokens.weight_Q8"),
                              numpy_helper.from_array(scale.astype(w.dtype), "model.embed_tokens.weight_scales")])
        out, ids, p = node.output[0], node.input[1], "/model/embed_tokens"
        io = helper.np_dtype_to_tensor_dtype(w.dtype)
        new = [helper.make_node("Gather", ["model.embed_tokens.weight_Q8", ids], [f"{p}/GatherQ8/output_0"], name=f"{p}/GatherQ8"),
               helper.make_node("Cast", [f"{p}/GatherQ8/output_0"], [f"{p}/CastQ8/output_0"], name=f"{p}/CastQ8", to=io),
               helper.make_node("Gather", ["model.embed_tokens.weight_scales", ids], [f"{p}/GatherScale/output_0"], name=f"{p}/GatherScale"),
               helper.make_node("Unsqueeze", [f"{p}/GatherScale/output_0", f"{p}/axes_last"], [f"{p}/Unsqueeze/output_0"], name=f"{p}/Unsqueeze"),
               helper.make_node("Mul", [f"{p}/CastQ8/output_0", f"{p}/Unsqueeze/output_0"], [out], name=f"{p}/MulScale")]
        g.initializer.append(numpy_helper.from_array(np.array([-1], np.int64), f"{p}/axes_last"))
        i = list(g.node).index(node); g.node.remove(node)
        for k, n in enumerate(new): g.node.insert(i + k, n)
        print(f"embedding -> int8 per-row (max abs error {err:.2e})")

    os.makedirs(a.out, exist_ok=True)
    for f in os.listdir(a.src):
        if f.endswith((".json", ".jinja")): os.system(f"cp '{a.src}/{f}' '{a.out}/'")
    if os.path.exists(f"{a.out}/model.onnx.data"): os.remove(f"{a.out}/model.onnx.data")
    onnx.save(m, f"{a.out}/model.onnx", save_as_external_data=True, all_tensors_to_one_file=True, location="model.onnx.data", size_threshold=1024)
    print(f"{a.out}: {os.path.getsize(f'{a.out}/model.onnx.data') / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
