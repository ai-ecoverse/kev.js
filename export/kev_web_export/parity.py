"""Compare an ONNX export (via ort_runtime) with the PyTorch fixtures: max |dp|, argmax agreement, per-fixture timing."""
import argparse, json, time
import numpy as np
from .ort_runtime import OrtKev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--fixtures", required=True)
    ap.add_argument("--provider", default="CPUExecutionProvider")
    a = ap.parse_args()
    kev = OrtKev(a.model, a.head, providers=(a.provider,))
    fx = json.load(open(a.fixtures))["fixtures"]
    worst, flips, n, t0 = 0.0, 0, 0, time.time()
    for f in fx:
        got = kev.probs(f["encoding"])
        for g, ref in zip(got, f["probs"]):
            ref = np.array(ref); d = float(np.abs(g - ref).max()); worst = max(worst, d); n += 1
            if int(g.argmax()) != int(ref.argmax()): flips += 1; print(f"  argmax flip in {f['name']}: {np.round(ref, 3)} -> {np.round(g, 3)}")
    print(json.dumps({"model": a.model, "questions": n, "max_abs_dp": worst, "argmax_flips": flips, "seconds": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
