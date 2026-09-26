"""Image requests against onnxruntime: the vision graph (kev_web_export.vision_onnx) and a spliced decoder
(kev_web_export.splice), in the browser's order. The algorithm src/vision.ts and src/model.ts are written against.

    image -> Qwen's processor (patches in 2x2-block order, grid) -> vision graph (+ position-table taps, 2D rotary)
          -> image_embeds [grid / 4, hidden]
    state = <state> <|vision_start|> <|image_pad|> x N <|vision_end|> text, run once with mRoPE positions
    each question = its branch as a continuation of the state's cache, text positions from max(state positions) + 1

`vision_inputs` is a numpy port of transformers' get_vision_interpolation_indices_and_weights,
get_vision_position_ids and Qwen3_5VisionRotaryEmbedding for one image (checked against them in vision_parity)."""
import time
import numpy as np
import onnxruntime as ort
from . import KEV_ROOT  # noqa: F401
from kev.api import SystemOneRequest, to_record
from kev.model import encode, rows_of
from .ort_runtime import OrtKev
from .vision import rope_positions


def block_rc(P, grid_w, m):
    """(row, col) of each patch in spatial-merge-block order."""
    i = np.arange(P)
    bw = grid_w // m
    return (i // (m * m * bw)) * m + (i // m) % m, ((i // (m * m)) % bw) * m + i % m


def vision_inputs(grid_h, grid_w, c):
    P, side, m = grid_h * grid_w, c["num_grid_per_side"], c["merge_size"]
    head_dim = c["hidden_size"] // c["num_heads"]; half = head_dim // 2
    inv_freq = (1.0 / (c["rope_theta"] ** (np.arange(0, half, 2, dtype=np.float32) / half))).astype(np.float32)
    r, col = block_rc(P, grid_w, m)

    def taps(x, size):   # bilinear, align_corners=True, into the side x side table
        src = (x.astype(np.float32) * np.float32(side - 1) / np.float32(max(size - 1, 1))).astype(np.float32)
        f = np.floor(src)
        return [(np.minimum(f, side - 1), 1 - np.abs(src - f)), (np.minimum(f + 1, side - 1), np.maximum(0, 1 - np.abs(src - f - 1)))]
    idx, w = [], []
    for hi, hw in taps(r, grid_h):
        for wi, ww in taps(col, grid_w):
            idx.append(hi * side + wi); w.append(hw * ww)
    fh, fw = r[:, None].astype(np.float32) * inv_freq, col[:, None].astype(np.float32) * inv_freq
    ang = np.concatenate([fh, fw], 1)
    ang = np.concatenate([ang, ang], 1)
    return np.stack(idx, 1).astype(np.int64), np.stack(w, 1).astype(np.float32), np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


class OrtVisionKev:
    def __init__(self, decoder, head, vision, cfg, tokenizer, proc, providers=("CPUExecutionProvider",)):
        self.kev = OrtKev(decoder, head, providers)
        self.vis = ort.InferenceSession(vision, providers=list(providers))
        self.cfg, self.tok, self.proc = cfg, tokenizer, proc
        self.timing = {}

    def embeds(self, image):
        t0 = time.perf_counter()
        px = self.proc(images=[image], return_tensors="np")
        _, gh, gw = px["image_grid_thw"][0].tolist()
        t1 = time.perf_counter()
        idx, w, cos, sin = vision_inputs(gh, gw, self.cfg)
        emb = self.vis.run(["image_embeds"], {"patches": px["pixel_values"].astype(np.float32), "pos_idx": idx, "pos_w": w, "cos": cos, "sin": sin})[0]
        self.timing.update(preprocess_ms=(t1 - t0) * 1e3, vision_ms=(time.perf_counter() - t1) * 1e3)
        return emb, gh, gw

    def probs(self, req, image=None, embeds=None):
        rec, _ = to_record(SystemOneRequest.model_validate(req))
        for q in rec["questions"]: q["label"] = 0
        enc = encode(self.tok, rec, max_state=65536, max_branch=73728)
        S, _, rows = rows_of(enc)
        if embeds is None and image is not None: embeds = self.embeds(image)
        c = self.cfg
        t0 = time.perf_counter()
        if embeds is not None:
            emb, gh, gw = embeds
            S = S[:1] + [c["vision_start"]] + [c["image_token"]] * emb.shape[0] + [c["vision_end"]] + S[1:]
            pos = np.array(rope_positions(S, gh, gw, c["image_token"], c["merge_size"]))
        else:
            emb, pos = None, np.arange(len(S))[None].repeat(3, 0)
        _, past = self.kev.run(S, pos, self.kev.empty_past(), 0, emb)
        t1 = time.perf_counter()
        nxt, out = int(pos.max()) + 1, []
        for r in rows:
            h, _ = self.kev.run(r["ids"], list(range(nxt, nxt + len(r["ids"]))), past, len(S))
            out.append(self.kev.head(h[r["decide"]], h[r["opts"]]))
        self.timing.update(state_ms=(t1 - t0) * 1e3, branches_ms=(time.perf_counter() - t1) * 1e3, state_tokens=len(S))
        return out
