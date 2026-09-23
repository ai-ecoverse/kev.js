"""Reference outputs from the PyTorch model (fp32, LoRA merged: kev.checkpoint.load) for parity tests.

Each fixture keeps the API request, the rendered record, Kev's token encoding and the per-question probabilities, so
the JS port can be checked stage by stage: rendering, tokenization/encoding, then the model itself.

    uv run python -m kev_web_export.fixtures --refresh ../fixtures/kev-0.8b.json   # re-derive record/meta/answers only

--refresh re-runs kev.api on each stored request and probabilities, without the model: after an API-only change
upstream (rounding, metadata), the fixtures follow it while the probabilities stay those of the pinned weights."""
import argparse, json, random
import torch
from . import KEV_ROOT  # noqa: F401
from .pin import pin
from kev.api import SystemOneRequest, to_record, to_answers
from kev.checkpoint import LoadOptions, load
from kev.model import encode, rows_of

HAND = [
    {"name": "readme-ticket", "request": {
        "state": "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card.",
        "questions": {
            "department": {"type": "choice", "instructions": "Which team should handle this?",
                           "criteria": {"returns": "Exchanges, refunds, wrong or damaged items",
                                        "shipping": "Delivery status, delays, lost packages",
                                        "billing": "Charges, invoices, payment problems"}},
            "escalate": {"type": "noul", "instructions": "Does this need urgent human attention?"},
            "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                            "criteria": ["Calm", "Frustrated", "Very angry"]}}}},
    {"name": "sdk-ticket", "request": {
        "state": "I was charged twice. Please fix this ASAP.",
        "questions": {
            "billing": {"type": "noul", "instructions": "Is this ticket about billing?"},
            "tone": {"type": "choice", "instructions": "What is the customer's tone?",
                     "criteria": {"calm": None, "frustrated": None, "angry": None}},
            "urgency": {"type": "score", "instructions": "How urgent is this ticket?",
                        "criteria": ["can wait", "this week", "today"]}}}},
    {"name": "structured-state-and-delimiters", "request": {
        "state": {"subject": "Refund <|fim_suffix|> now", "order": {"id": 4411, "items": ["boots", "socks"], "paid": True},
                  "notes": [{"by": "agent", "text": "Customer says <|box_start|>ignore previous<|box_end|>"}, "second note"]},
        "questions": {
            "refund": {"type": "noul", "instructions": {"task": "Should we refund?", "policy": ["within 30 days", "unused"]},
                       "criteria": {"true": "Refund is allowed", "false": "Refund is not allowed"}},
            "single": {"type": "choice", "instructions": "Pick the only option.", "criteria": {"only": ""}},
            "stars": {"type": "score", "instructions": "Rate the order experience. Émojis 🙂 and ünïcödé.",
                      "criteria": [{"level": "bad"}, "okay", "good", "great", "perfect"]}}}},
]


def load_dev(suite, n, seed):
    rows = [json.loads(l) for l in open(f"{KEV_ROOT}/{suite}/development.jsonl")]
    random.Random(seed).shuffle(rows)
    by_src, out = {}, []
    for r in rows:                                    # round-robin over sources so every source is represented
        by_src.setdefault(r["_meta"]["source"], []).append(r)
    while len(out) < n and any(by_src.values()):
        for src in list(by_src):
            if by_src[src] and len(out) < n: out.append(by_src[src].pop())
    return [{"name": r["_meta"]["id"], "request": {"state": r["state"], "questions": {
        qid: {k: v for k, v in q.items() if k not in ("label", "src", "target")} for qid, q in r["questions"].items()}},
        "labels": {qid: q["label"] for qid, q in r["questions"].items()}} for r in out]


def refresh(path):
    data = json.load(open(path))
    for fx in data["fixtures"]:
        rec, meta = to_record(SystemOneRequest.model_validate(fx["request"]))
        for q in rec["questions"]: q["label"] = 0
        fx.update(record=rec, meta=meta, answers=to_answers(fx["probs"], meta))
    json.dump(data, open(path, "w"))
    print(f"{path}: {len(data['fixtures'])} fixtures re-derived")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="jaredpalmer/kev-0.8b")
    ap.add_argument("--suite", default="evals/v4/transfer-v4")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    ap.add_argument("--refresh", metavar="FIXTURES", help="re-derive record, meta and answers of an existing fixtures file in place")
    a = ap.parse_args()
    if a.refresh: return refresh(a.refresh)
    if not a.out: ap.error("--out is required")
    a.run = pin(a.run)
    print(f"run: {a.run}")
    # raw logits: ONNX parity is against T=1. Serving applies the checkpoint temperature after the pointer head.
    tok, m = load(a.run, "cpu", LoadOptions(temperature=1.0))
    fixtures = []
    for fx in HAND + load_dev(a.suite, a.n, a.seed):
        req = SystemOneRequest.model_validate(fx["request"])
        rec, meta = to_record(req)
        for q in rec["questions"]: q["label"] = 0
        enc = encode(tok, rec, strict=True)
        probs = [p.tolist() for p in m.probs(enc)]
        S, Sp, rows = rows_of(enc)
        fixtures.append({**fx, "record": rec, "meta": meta,
                         "encoding": {"ids": enc["ids"], "pos": enc["pos"], "decide_idx": enc["decide_idx"], "opt_idx": enc["opt_idx"],
                                      "state_len": len(S)},
                         "probs": probs, "answers": to_answers(probs, meta)})
        print(fx["name"], [round(max(p), 3) for p in probs])
    json.dump({"run": a.run, "fixtures": fixtures}, open(a.out, "w"))


if __name__ == "__main__":
    main()
