"""Export the stock Qwen3.5 vision tower (ViT + patch merger) of a Kev checkpoint's base to ONNX.

    uv run --group vision python -m kev_web_export.vision_onnx --run jaredpalmer/kev-4b@<sha> --out build/kev-4b/vision

Kev never changes these weights, so they come straight from the base snapshot. Adapted from cua-s1.js
(export/four_b/vision.py). Qwen3_5VisionModel.forward derives everything that depends on the image's size from
grid_thw: the bilinear resampling of the learned 48x48 position table, the 2D rotary angles and the packed-sequence
boundaries. Those are data-dependent loops, so here they are inputs, computed by the caller (`vision_inputs`, and
its port in src/vision.ts). What remains is plain tensor math over one image's patches in spatial-merge-block order:

    patches [P, 1536] -> patch projection (the Conv3d, whose kernel equals its stride, as a MatMul)
                      + sum_k pos_embed[pos_idx[:, k]] * pos_w[:, k]
                      -> 24 blocks (full attention within the image, 2D rotary from cos/sin)
                      -> merger -> image_embeds [P/4, hidden]

`Core` is checked against transformers' own `visual(pixel_values, grid_thw)` before export, and the ONNX graph against
`Core`. Two graphs are written: fp32.onnx (the parity reference) and web/ (fp16 weights cast to fp32 when the session
loads, fp32 compute, 32 MB shards). cua-s1.js measured int8 weights at 4.6% RMS off on the image embeddings; fp16
weights stay within 0.2% for half the bytes of fp32."""
import argparse, json, os
import numpy as np
import torch
from torch import nn
from .vision import load_visual, MAX_PIXELS
from .pin import pin

INPUTS = ["patches", "pos_idx", "pos_w", "cos", "sin"]


def vision_inputs(v, grid_h: int, grid_w: int):
    """The size-dependent inputs for one image of grid_h x grid_w patches, from transformers' own helpers."""
    from transformers.vision_utils import get_vision_interpolation_indices_and_weights, get_vision_position_ids
    thw = torch.tensor([[1, grid_h, grid_w]])
    idx, w = get_vision_interpolation_indices_and_weights(thw, num_grid_per_side=v.num_grid_per_side, mode=v.interpolation_mode,
                                                          align_corners=v.interpolation_align_corners, spatial_merge_size=v.spatial_merge_size)
    pos = get_vision_position_ids(thw, v.spatial_merge_size)
    cos, sin = v.rotary_pos_emb(torch.zeros(1), pos)
    return idx, w.float(), cos.float(), sin.float()


def rotate_half(x):
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


class Core(nn.Module):
    def __init__(self, v):
        super().__init__()
        self.v = v
        p = v.patch_embed.proj
        self.patch_w = nn.Parameter(p.weight.detach().reshape(p.weight.shape[0], -1).clone())
        self.patch_b = nn.Parameter(p.bias.detach().clone())

    def forward(self, patches, pos_idx, pos_w, cos, sin):
        v = self.v
        h = patches @ self.patch_w.T + self.patch_b
        h = h + (v.pos_embed(pos_idx) * pos_w[:, :, None]).sum(1)
        cos, sin = cos[None], sin[None]                                                                   # [1, P, d]
        for blk in v.blocks:
            a = blk.attn
            P = h.shape[0]
            q, k, val = a.qkv(blk.norm1(h)).reshape(P, 3, a.num_heads, -1).permute(1, 2, 0, 3).unbind(0)   # [heads, P, d]
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin
            att = torch.softmax((q @ k.transpose(-1, -2)) * a.scaling, dim=-1) @ val                       # full attention
            h = h + a.proj(att.transpose(0, 1).reshape(P, -1))
            h = h + blk.mlp(blk.norm2(h))
        return v.merger(h)


def strip_traces(path):
    import onnx
    m = onnx.load(path, load_external_data=True)
    for node in m.graph.node:
        keep = [p for p in node.metadata_props if p.key != "pkg.torch.onnx.stack_trace"]
        del node.metadata_props[:]; node.metadata_props.extend(keep)
    return m


def half_weights(src: str, out_dir: str, shard_mb: int = 32):
    """The browser graph: float initializers of 64k+ elements stored as fp16 and cast to fp32 at load; compute in fp32."""
    import onnx
    from onnx import helper, numpy_helper, TensorProto
    from .postprocess import save_sharded
    m = onnx.load(src, load_external_data=True)
    g = m.graph
    for t in list(g.initializer):
        if t.data_type != TensorProto.FLOAT or np.prod(t.dims) < 65536: continue
        name, arr = t.name, numpy_helper.to_array(t)
        g.initializer.remove(t)
        g.initializer.append(numpy_helper.from_array(arr.astype(np.float16), f"{name}_f16"))
        g.node.insert(0, helper.make_node("Cast", [f"{name}_f16"], [name], name=f"{name}/Cast", to=TensorProto.FLOAT))
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(out_dir):
        if f.startswith("model.onnx"): os.remove(f"{out_dir}/{f}")
    save_sharded(m, out_dir, shard_mb * 1_000_000)
    return f"{out_dir}/model.onnx"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="jaredpalmer/kev-4b@4bc64c6b4c4881148661ffb823ce21fcfdc79a0e")
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=18)
    a = ap.parse_args()
    from huggingface_hub import snapshot_download
    from transformers import AutoImageProcessor
    from PIL import Image
    from . import KEV_ROOT  # noqa: F401
    from kev.checkpoint import Checkpoint
    run = pin(a.run)
    meta = Checkpoint(run).meta
    base = snapshot_download(meta.base, revision=meta.base_revision)
    cfg, v = load_visual(base)
    core = Core(v).eval()
    proc = AutoImageProcessor.from_pretrained(base, size={"shortest_edge": 65536, "longest_edge": MAX_PIXELS})
    rng = np.random.default_rng(0)
    for w, h in [(760, 632), (448, 448), (500, 300)]:
        img = Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))
        out = proc(images=[img], return_tensors="pt")
        (_, gh, gw), = out["image_grid_thw"].tolist()
        with torch.no_grad():
            ref = v(out["pixel_values"].float(), grid_thw=out["image_grid_thw"]).pooler_output
            got = core(out["pixel_values"].float(), *vision_inputs(v, gh, gw))
        print(f"{w}x{h} -> grid {gh}x{gw}: Core vs visual max |d| {float((ref - got).abs().max()):.2e} (|ref| max {float(ref.abs().max()):.1f})")

    os.makedirs(a.out, exist_ok=True)
    img = Image.fromarray(rng.integers(0, 256, (632, 760, 3), dtype=np.uint8))
    out = proc(images=[img], return_tensors="pt")
    (_, gh, gw), = out["image_grid_thw"].tolist()
    sample = (out["pixel_values"].float(), *vision_inputs(v, gh, gw))
    from torch.export import Dim
    P = 4 * Dim("merged", min=1, max=16384)   # the merger folds 2x2 blocks of consecutive patches
    path = f"{a.out}/fp32.onnx"
    with torch.enable_grad():
        program = torch.onnx.export(core, sample, dynamo=True, opset_version=a.opset, input_names=INPUTS, output_names=["image_embeds"],
                                    dynamic_shapes={"patches": {0: P}, "pos_idx": {0: P}, "pos_w": {0: P}, "cos": {0: P}, "sin": {0: P}})
    program.save(path, external_data=True)
    import onnx
    m = strip_traces(path)
    os.remove(f"{path}.data")          # onnx.save appends to an existing external-data file
    onnx.save(m, path, save_as_external_data=True, location="fp32.onnx.data")
    del m

    import onnxruntime as ort
    web = half_weights(path, f"{a.out}/web")
    for label, p in [("fp32", path), ("web fp16 weights", web)]:
        sess = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
        for w, h in [(760, 632), (1024, 640)]:
            img = Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))
            out = proc(images=[img], return_tensors="pt")
            (_, gh, gw), = out["image_grid_thw"].tolist()
            ins = (out["pixel_values"].float(), *vision_inputs(v, gh, gw))
            with torch.no_grad(): ref = core(*ins).numpy()
            got = sess.run(["image_embeds"], {k: t.numpy() for k, t in zip(INPUTS, ins)})[0]
            rms = float(np.sqrt(((ref - got) ** 2).mean()) / np.sqrt((ref ** 2).mean()))
            print(f"ONNX {label} {w}x{h}: max |d| vs Core {float(np.abs(ref - got).max()):.2e}, relative RMS {rms:.2e}")
    c = cfg.vision_config
    json.dump({"run": run, "base": f"{meta.base}@{meta.base_revision}", "inputs": INPUTS, "outputs": ["image_embeds"],
               "patch_size": v.patch_size, "merge_size": v.spatial_merge_size, "temporal_patch_size": v.patch_embed.temporal_patch_size,
               "hidden_size": c.hidden_size, "out_hidden_size": c.out_hidden_size, "num_heads": c.num_heads,
               "num_grid_per_side": v.num_grid_per_side, "rope_theta": c.rope_parameters["rope_theta"],
               "image_mean": proc.image_mean, "image_std": proc.image_std, "min_pixels": proc.size["shortest_edge"], "max_pixels": proc.size["longest_edge"],
               "image_token": cfg.image_token_id, "vision_start": cfg.vision_start_token_id, "vision_end": cfg.vision_end_token_id},
              open(f"{a.out}/vision.json", "w"), indent=2)
    print(f"-> {path}; browser -> {web}")


if __name__ == "__main__":
    main()
