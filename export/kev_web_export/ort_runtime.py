"""The browser algorithm, in Python against onnxruntime: run the state once, then every question as a continuation of
the state's cache (Kev's hybrid serving path, kev.model._branch_rows_from_prefix), and apply the pointer head.

This is the reference the JS runtime is written against; parity.py checks it against the PyTorch fixtures."""
import numpy as np
import onnxruntime as ort
from safetensors.numpy import load_file


class OrtKev:
    def __init__(self, model_path, head_path, providers=("CPUExecutionProvider",)):
        self.sess = ort.InferenceSession(model_path, providers=list(providers))
        self.inputs = {i.name: i for i in self.sess.get_inputs()}
        self.dtype = np.float16 if any(i.type == "tensor(float16)" for i in self.sess.get_inputs()) else np.float32
        h = load_file(head_path)
        self.qw, self.qb, self.kw, self.kb = (h[k].astype(np.float32) for k in ("q.weight", "q.bias", "k.weight", "k.bias"))
        self.scale = 1 / np.sqrt(self.qw.shape[0])

    def empty_past(self):
        past = {}
        for name, i in self.inputs.items():
            if name.startswith("past_key_values."):
                past[name] = np.zeros((1, i.shape[1], 0, 256), self.dtype)
            elif name.startswith("past."):
                past[name] = np.zeros([1] + i.shape[1:], self.dtype)
        return past

    def run(self, ids, pos, past, past_len, image_embeds=None):
        """pos: text positions [S] (all three mRoPE sections equal) or mRoPE positions [3, S]. image_embeds: rows for the
        <|image_pad|> tokens of ids, on a graph spliced by kev_web_export.splice (one zero row when there are none)."""
        pos = np.asarray(pos, np.int64)
        feeds = {"input_ids": np.array([ids], np.int64),
                 "attention_mask": np.ones((1, past_len + len(ids)), np.int64),
                 "position_ids": (pos.reshape(3, 1, -1) if pos.ndim == 2 else np.broadcast_to(pos, (3, 1, len(pos)))).copy(), **past}
        if "image_embeds" in self.inputs:
            feeds["image_embeds"] = np.zeros((1, self.inputs["image_embeds"].shape[1]), self.dtype) if image_embeds is None else image_embeds.astype(self.dtype)
        names = [o.name for o in self.sess.get_outputs()]
        out = dict(zip(names, self.sess.run(names, feeds)))
        present = {n.replace("present.", "past_key_values.", 1) if n.endswith((".key", ".value")) else n.replace("present.", "past.", 1): v
                   for n, v in out.items() if n.startswith("present.")}
        return out["hidden_states"][0].astype(np.float32), present

    def head(self, h_decide, h_opts):
        z = (h_opts @ self.kw.T + self.kb) @ (self.qw @ h_decide + self.qb) * self.scale
        e = np.exp(z - z.max()); return e / e.sum()

    def probs(self, enc):
        Ls = enc["state_len"]
        _, state_past = self.run(enc["ids"][:Ls], enc["pos"][:Ls], self.empty_past(), 0)
        out, start = [], Ls
        for d, oi in zip(enc["decide_idx"], enc["opt_idx"]):
            end = d + 1
            h, _ = self.run(enc["ids"][start:end], enc["pos"][start:end], state_past, Ls)   # ORT never mutates inputs: the state cache is reused as is
            out.append(self.head(h[d - start], h[[o - start for o in oi]]))
            start = end
        return out
