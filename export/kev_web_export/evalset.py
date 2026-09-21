"""A larger labelled evaluation set for the browser, with reference probabilities from the fp32 ONNX export
(which matches the PyTorch model to 3e-5 on the fixtures, and is ~50x faster than PyTorch's reference DeltaNet on CPU).

Scores accuracy and Brier for the reference so the browser numbers can be compared record by record."""
import argparse, json, os
import numpy as np
from . import KEV_ROOT  # noqa: F401
from .pin import pin
from kev.api import SystemOneRequest, to_record
from kev.evaluate import load_tokenizer
from kev.model import encode, rows_of
from .fixtures import load_dev
from .ort_runtime import OrtKev


def label_index(meta, y):
    return int(y) if meta["type"] == "noul" else meta["keys"].index(y) if meta["type"] == "choice" else int(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--head", required=True)
    ap.add_argument("--base", default="Qwen/Qwen3.5-0.8B-Base"); ap.add_argument("--base_revision", default="dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68")
    ap.add_argument("--suite", default="evals/v4/transfer-v4"); ap.add_argument("--n", type=int, default=300); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tok = load_tokenizer(a.base, revision=a.base_revision)
    rt = OrtKev(a.model, a.head)
    recs, correct, brier, n = [], 0, 0.0, 0
    for fx in load_dev(a.suite, a.n, a.seed):
        rec, meta = to_record(SystemOneRequest.model_validate(fx["request"]))
        for q in rec["questions"]: q["label"] = 0
        enc = encode(tok, rec, strict=True)
        S, _, _ = rows_of(enc)
        e = {"ids": enc["ids"], "pos": enc["pos"], "decide_idx": enc["decide_idx"], "opt_idx": enc["opt_idx"], "state_len": len(S)}
        probs = [p.tolist() for p in rt.probs(e)]
        labels = [label_index(m, fx["labels"][m["id"]]) for m in meta]
        for p, y in zip(probs, labels):
            correct += int(np.argmax(p) == y); onehot = np.eye(len(p))[y]; brier += float(((np.array(p) - onehot) ** 2).sum()); n += 1
        recs.append({"name": fx["name"], "record": {"state": rec["state"], "questions": [{"instr": q["instr"], "options": q["options"]} for q in rec["questions"]]},
                     "probs": probs, "labels": labels})
    summary = {"questions": n, "accuracy": correct / n, "brier": brier / n}
    kev_json = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(a.model))), "kev.json")
    run = json.load(open(kev_json))["run"] if os.path.exists(kev_json) else None   # the exported checkpoint's pinned commit
    json.dump({"run": run, "reference": a.model, "suite": a.suite, "summary": summary, "records": recs}, open(a.out, "w"))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
