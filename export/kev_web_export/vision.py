"""Kev behind Qwen3.5's stock vision tower: the PyTorch reference for image questions.

Kev fine-tunes only Qwen3.5's text model (a LoRA on the language-model projections; no `model.visual.*` tensors in the
adapter), and Qwen3.5 base checkpoints are natively multimodal. So the base's unmodified vision tower and merger can sit
in front of Kev's merged text model, as Jev-Omni does with Gemma's encoders. Nothing here is trained.

Layout of a state with an image (the image comes first, as Qwen's chat template puts it before the user's text):

    <state> <|vision_start|> <|image_pad|> x N <|vision_end|> state text...   then each question branch as usual

The image tokens are inserted directly, never through `user_tokens`, so callers still cannot forge them from text.
Positions are Qwen3.5's 3D mRoPE (`rope_positions`, checked against `Qwen3_5Model.get_rope_index` in `check`):
text counts up on all three axes, the image's tokens get (start, start + row, start + column) over the merged grid,
and text after it resumes at start + max(rows, columns). Every question runs as one causal row (state + branch), which
is `DecisionModel.forward_rows_batch` with image embeddings spliced in at the `<|image_pad|>` tokens.

    uv run --group vision python -m kev_web_export.vision --run jaredpalmer/kev-4b@4bc64c6b4c4881148661ffb823ce21fcfdc79a0e"""
import argparse, json
import torch
import torch.nn.functional as F
from safetensors import safe_open
from . import KEV_ROOT  # noqa: F401  (puts kev on sys.path)
from kev.api import SystemOneRequest, to_record
from kev.checkpoint import Checkpoint, LoadOptions
from kev.model import rows_of
from .pin import pin

# Largest image area fed to the tower, after Qwen's smart_resize (the base allows 16.7M). The tower's attention is full
# over the image's patches, so its memory grows with the square of the area: at 589,824 pixels (768 x 768) that is 2,304
# patches and a 340 MB fp32 score matrix per layer, which WebGPU buffers hold; 576 tokens reach the text model.
MAX_PIXELS = 768 * 768


def load_visual(base: str, dtype=torch.float32):
    """The stock Qwen3_5VisionModel (ViT + patch merger) from a base snapshot's `model.visual.*` tensors."""
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
    cfg = AutoConfig.from_pretrained(base)
    vcfg = cfg.vision_config
    vcfg._attn_implementation = "eager"
    v = Qwen3_5VisionModel(vcfg)
    idx = json.load(open(f"{base}/model.safetensors.index.json"))["weight_map"]
    sd = {}
    for f in sorted({f for k, f in idx.items() if k.startswith("model.visual.")}):
        with safe_open(f"{base}/{f}", "pt") as st:
            sd.update({k[len("model.visual."):]: st.get_tensor(k) for k in st.keys() if k.startswith("model.visual.")})
    missing, unexpected = v.load_state_dict(sd, strict=False)
    if [m for m in missing if not m.endswith("inv_freq")] or unexpected:
        raise SystemExit(f"vision weights: missing {missing}, unexpected {unexpected}")
    return cfg, v.to(dtype).eval()


def rope_positions(ids, grid_h, grid_w, image_token, merge=2):
    """[3, L] mRoPE positions for token ids holding at most one image run of (grid_h / merge) x (grid_w / merge)."""
    t, h, w, cur, i = [], [], [], 0, 0
    while i < len(ids):
        if ids[i] != image_token:
            t.append(cur); h.append(cur); w.append(cur); cur += 1; i += 1; continue
        gh, gw = grid_h // merge, grid_w // merge
        for r in range(gh):
            for c in range(gw):
                t.append(cur); h.append(cur + r); w.append(cur + c)
        i += gh * gw; cur += max(gh, gw)
    return [t, h, w]


class VisionKev:
    """A Kev checkpoint (fp32, LoRA merged, raw logits) with the stock vision tower of its own base in front."""

    def __init__(self, run: str, device="cpu", max_pixels: int = MAX_PIXELS):
        from huggingface_hub import snapshot_download
        from transformers import AutoImageProcessor
        self.run = pin(run)
        ck = Checkpoint(self.run)
        self.meta = ck.meta
        self.tok, self.model = ck.load(device, LoadOptions(temperature=1.0))
        self.base = snapshot_download(ck.meta.base, revision=ck.meta.base_revision)
        self.config, self.visual = load_visual(self.base)
        self.visual.to(device)
        self.proc = AutoImageProcessor.from_pretrained(self.base, size={"shortest_edge": 65536, "longest_edge": max_pixels})
        self.device = device
        c = self.config
        self.image_token, self.vs, self.ve = c.image_token_id, c.vision_start_token_id, c.vision_end_token_id
        self.merge = c.vision_config.spatial_merge_size
        hv, ht = c.vision_config.out_hidden_size, self.model.lm.config.hidden_size
        if hv != ht: raise SystemExit(f"vision merger outputs {hv}, Kev's text model takes {ht}")

    @torch.no_grad()
    def image_embeds(self, image):
        out = self.proc(images=[image], return_tensors="pt")
        (_, gh, gw), = out["image_grid_thw"].tolist()
        emb = self.visual(out["pixel_values"].to(self.device, torch.float32), grid_thw=out["image_grid_thw"].to(self.device)).pooler_output
        return emb, gh, gw

    def _inputs(self, ids, embeds):
        """(inputs_embeds [1, L, d], position_ids [3, 1, L]) for a full row; image embeddings at the <|image_pad|> tokens."""
        x = self.model.lm.get_input_embeddings()(torch.tensor([ids], device=self.device))
        if embeds is None:
            return x, torch.arange(len(ids), device=self.device)[None, None].expand(3, 1, -1)
        emb, gh, gw = embeds
        x[0, torch.tensor(ids, device=self.device) == self.image_token] = emb.to(x.dtype)
        return x, torch.tensor(rope_positions(ids, gh, gw, self.image_token, self.merge), device=self.device)[:, None, :]

    @torch.no_grad()
    def logits(self, req: dict, image=None, embeds=None, prefix=True):
        """Raw pointer logits (T = 1) per question, in request order. `embeds` = (image_embeds, grid_h, grid_w) reuses
        a tower pass; `image` runs one; neither is text-only. prefix=True runs the state once and each branch on a copy
        of its cache (kev.serve's path, `_branch_rows_from_prefix`); False runs every question as its full row
        (`forward_rows_batch`). The two agree to float noise (checked in `check`)."""
        import copy
        from transformers import DynamicCache
        rec, meta = to_record(SystemOneRequest.model_validate(req))
        enc = self.model.encode(self.tok, rec, max_state=65536, max_branch=73728)
        S, _, rows = rows_of(enc)
        if embeds is None and image is not None: embeds = self.image_embeds(image)
        if embeds is not None:
            S = S[:1] + [self.vs] + [self.image_token] * embeds[0].shape[0] + [self.ve] + S[1:]
        lm, out = self.model.lm, []
        if prefix:
            x, pos = self._inputs(S, embeds)
            cache = lm(inputs_embeds=x, position_ids=pos, past_key_values=DynamicCache(config=lm.config), use_cache=True).past_key_values
            nxt = int(pos.max()) + 1
        for r in rows:
            if prefix:
                x = lm.get_input_embeddings()(torch.tensor([r["ids"]], device=self.device))
                pos = torch.arange(nxt, nxt + len(r["ids"]), device=self.device)[None, None].expand(3, 1, -1)
                h = lm(inputs_embeds=x, position_ids=pos, past_key_values=copy.deepcopy(cache), use_cache=True).last_hidden_state[0].float()
                d, oi = r["decide"], r["opts"]
            else:
                x, pos = self._inputs(S + r["ids"], embeds)
                h = lm(inputs_embeds=x, position_ids=pos).last_hidden_state[0].float()
                d, oi = len(S) + r["decide"], [len(S) + o for o in r["opts"]]
            out.append(self.model.head(h[d], h[torch.tensor(oi, device=self.device)]).cpu())
        return out, meta


def check(vk: VisionKev):
    """Wiring checks: positions against transformers' own get_rope_index, and a text-only request against the plain
    DecisionModel path (no image = bit-for-bit the Kev path)."""
    from types import SimpleNamespace
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
    from PIL import Image
    img = Image.new("RGB", (320, 224), (200, 30, 30))
    emb, gh, gw = vk.image_embeds(img)
    ids = [1, 2, vk.vs] + [vk.image_token] * emb.shape[0] + [vk.ve, 5, 6, 7]
    stub = SimpleNamespace(config=vk.config, get_vision_position_ids=lambda *a, **k: Qwen3_5Model.get_vision_position_ids(None, *a, **k))
    t = torch.tensor([ids])
    ref, _ = Qwen3_5Model.get_rope_index(stub, t, mm_token_type_ids=(t == vk.image_token).int(), image_grid_thw=torch.tensor([[1, gh, gw]]))
    ok = torch.equal(ref[:, 0], torch.tensor(rope_positions(ids, gh, gw, vk.image_token, vk.merge)))
    print(f"mRoPE positions match get_rope_index: {ok} (grid {gh}x{gw}, {emb.shape[0]} image tokens, embeds {tuple(emb.shape)})")
    req = {"state": "I was charged twice. Please fix this ASAP.", "questions": {"billing": {"type": "noul", "instructions": "Is this ticket about billing?"}}}
    got, _ = vk.logits(req)
    rec, _ = to_record(SystemOneRequest.model_validate(req))
    with torch.no_grad(): want = vk.model.forward(vk.model.encode(vk.tok, rec))
    print(f"text-only path vs DecisionModel.forward: max |d logit| {max(float((a - b.cpu()).abs().max()) for a, b in zip(got, want)):.2e}, p(yes) {float(F.softmax(got[0], -1)[1]):.4f}")
    req = {"state": "A picture.", "questions": {"red": {"type": "noul", "instructions": "Is the picture red?"},
                                                "c": {"type": "choice", "instructions": "What color is it?", "criteria": {"red": None, "green": None, "blue": None}}}}
    a, _ = vk.logits(req, embeds=(emb, gh, gw), prefix=True)
    b, _ = vk.logits(req, embeds=(emb, gh, gw), prefix=False)
    print(f"image request, state prefix vs full rows: max |d logit| {max(float((x - y).abs().max()) for x, y in zip(a, b)):.2e}; "
          f"red p(yes) {float(F.softmax(a[0], -1)[1]):.4f}, color {F.softmax(a[1], -1).numpy().round(3).tolist()}")
    return ok


@torch.no_grad()
def describe(vk: VisionKev, image, prompt="This picture shows", tokens=24):
    """Greedy continuation by the stock base (no LoRA, its tied LM head) through the same image splice and positions:
    if the wiring were wrong, the base could not describe the picture."""
    from transformers import AutoModelForCausalLM
    lm = AutoModelForCausalLM.from_pretrained(vk.base, dtype=torch.float32).to(vk.device).eval()
    emb, gh, gw = vk.image_embeds(image)
    ids = [vk.vs] + [vk.image_token] * emb.shape[0] + [vk.ve] + vk.tok(prompt, add_special_tokens=False).input_ids
    n0 = len(ids)
    for _ in range(tokens):
        x = lm.get_input_embeddings()(torch.tensor([ids], device=vk.device))
        x[0, torch.tensor(ids, device=vk.device) == vk.image_token] = emb
        pos = torch.tensor(rope_positions(ids, gh, gw, vk.image_token, vk.merge), device=vk.device)[:, None, :]
        ids.append(int(lm(inputs_embeds=x, position_ids=pos).logits[0, -1].argmax()))
    del lm
    return prompt + vk.tok.decode(ids[n0:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="jaredpalmer/kev-4b@4bc64c6b4c4881148661ffb823ce21fcfdc79a0e")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--describe", nargs="*", default=[], help="images the stock base should describe through the splice")
    a = ap.parse_args()
    vk = VisionKev(a.run, a.device)
    print(f"run {vk.run}; base {vk.meta.base}@{vk.meta.base_revision}; text model {type(vk.model.lm).__name__}, "
          f"hidden {vk.model.lm.config.hidden_size}; vision out {vk.config.vision_config.out_hidden_size}")
    check(vk)
    from PIL import Image
    for f in a.describe:
        print(f"{f}: {describe(vk, Image.open(f).convert('RGB'))!r}")


if __name__ == "__main__":
    main()
