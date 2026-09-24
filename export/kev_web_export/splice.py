"""Give a Kev decoder graph an `image_embeds` input, without touching its weights.

    uv run python -m kev_web_export.splice --src build/kev-4b/onnx-fp32-cpu --out build/kev-4b/fp32-vision
    uv run python -m kev_web_export.splice --src ../public/models/kev-4b/r-4bc64c6/q8f32 --out build/kev-4b/q8f32-vision

`image_embeds` is [image tokens, hidden]: each <|image_pad|> position of input_ids takes the next row, in order,
instead of the token's embedding, which is transformers' masked_scatter of the vision features into inputs_embeds
(from cua-s1.js, export/four_b/postprocess.py). A request without an image, and every question branch, passes one row
of zeros; with no <|image_pad|> in input_ids the graph's output is the text graph's, bit for bit (a Where that
selects the text embedding everywhere).

Only model.onnx is rewritten. The external-data files are hard-linked (copied where links fail), so the spliced graph
reads the very bytes of the text graph: works on the builder's output (embedding = a Gather) and on a postprocessed
or published bundle (embedding = int8 slices ending in /model/embed_tokens/MulScale)."""
import argparse, os, shutil
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto


def embedding_output(g):
    for n in g.node:
        if n.name == "/model/embed_tokens/MulScale": return n.output[0]
    (n,) = [n for n in g.node if n.op_type == "Gather" and n.input[0] == "model.embed_tokens.weight"]
    return n.output[0]


def splice_image_embeds(g, token_id, hidden):
    embedded = embedding_output(g)
    (producer,) = [n for n in g.node if embedded in n.output]
    text = f"{embedded}_text"
    producer.output[list(producer.output).index(embedded)] = text
    io = next(o for o in g.output if o.name == "hidden_states").type.tensor_type.elem_type
    g.input.append(helper.make_tensor_value_info("image_embeds", io, ["image_tokens", hidden]))
    p = "/model/image_embeds"
    g.initializer.extend([numpy_helper.from_array(np.array(token_id, np.int64), f"{p}/token_id"),
                          numpy_helper.from_array(np.array(1, np.int64), f"{p}/one"),
                          numpy_helper.from_array(np.array(0, np.int64), f"{p}/zero"),
                          numpy_helper.from_array(np.array(1, np.int64), f"{p}/axis"),
                          numpy_helper.from_array(np.array([-1], np.int64), f"{p}/last")])
    nodes = [
        helper.make_node("Equal", ["input_ids", f"{p}/token_id"], [f"{p}/mask"], name=f"{p}/Equal"),                # [B, S]
        helper.make_node("Cast", [f"{p}/mask"], [f"{p}/mask_i"], name=f"{p}/Cast", to=TensorProto.INT64),
        helper.make_node("CumSum", [f"{p}/mask_i", f"{p}/axis"], [f"{p}/count"], name=f"{p}/CumSum"),               # 1, 2, ... at image tokens
        helper.make_node("Sub", [f"{p}/count", f"{p}/one"], [f"{p}/row"], name=f"{p}/Sub"),
        helper.make_node("Max", [f"{p}/row", f"{p}/zero"], [f"{p}/row0"], name=f"{p}/Max"),                         # text tokens before the image
        helper.make_node("Gather", ["image_embeds", f"{p}/row0"], [f"{p}/rows"], name=f"{p}/Gather", axis=0),     # [B, S, hidden]
        helper.make_node("Unsqueeze", [f"{p}/mask", f"{p}/last"], [f"{p}/mask3"], name=f"{p}/Unsqueeze"),
        helper.make_node("Where", [f"{p}/mask3", f"{p}/rows", text], [embedded], name=f"{p}/Where"),
    ]
    i = list(g.node).index(producer) + 1
    for k, n in enumerate(nodes): g.node.insert(i + k, n)


def link(src, dst):
    if os.path.exists(dst): os.remove(dst)
    try: os.link(src, dst)
    except OSError: shutil.copy(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="directory holding model.onnx and its model.onnx.data* files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--image-token-id", type=int, default=248056)   # <|image_pad|> in every Qwen3.5 tokenizer
    a = ap.parse_args()
    m = onnx.load(f"{a.src}/model.onnx", load_external_data=False)
    if any(i.name == "image_embeds" for i in m.graph.input): raise SystemExit(f"{a.src} already has image_embeds")
    hs = next(o for o in m.graph.output if o.name == "hidden_states")
    hidden = hs.type.tensor_type.shape.dim[-1].dim_value
    if not hidden: raise SystemExit("hidden_states has no static hidden size")
    splice_image_embeds(m.graph, a.image_token_id, hidden)
    os.makedirs(a.out, exist_ok=True)
    data = [f for f in os.listdir(a.src) if f.startswith("model.onnx.data")]
    for f in data: link(f"{a.src}/{f}", f"{a.out}/{f}")
    for f in os.listdir(a.src):
        if f.endswith((".json", ".jinja")): shutil.copy(f"{a.src}/{f}", a.out)
    with open(f"{a.out}/model.onnx", "wb") as fh: fh.write(m.SerializeToString())
    print(f"{a.out}/model.onnx: image_embeds [image_tokens, {hidden}] at token {a.image_token_id}; {len(data)} data files linked")


if __name__ == "__main__":
    main()
