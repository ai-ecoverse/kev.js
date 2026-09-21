"""Publish packaged model bundles to a Hugging Face repo (default ai-ecoverse/kev.js).

    HF_TOKEN=... uv run python -m kev_web_export.upload_hf --models ../public/models --repo ai-ecoverse/kev.js

Each model directory becomes a folder in the repo (kev-0.8b/, kev-4b/, kev-9b/), which is what the browser runtime
expects: <base>/<model>/manifest.json and the files it names. A model card is written from the manifests."""
import argparse, json, os
from huggingface_hub import HfApi

CARD = """---
license: apache-2.0
library_name: kev.js
pipeline_tag: text-classification
base_model:
{bases}
tags: [kev, decision-model, onnx, onnxruntime-web, webgpu, quantized, int8]
---

# kev.js weights

Browser-ready exports of [Kev](https://github.com/jaredpalmer/kev), Jared Palmer's family of small decision models.
They answer yes/no, multiple-choice and rating questions with calibrated probabilities, and run in a browser on
WebGPU through [@ai-ecoverse/kev.js](https://github.com/ai-ecoverse/kev.js). This repo holds only converted
weights: no training, evaluation or model design here is ours.

```js
import * as ort from "onnxruntime-web/webgpu";
import {{ loadKev }} from "@ai-ecoverse/kev.js";

const kev = await loadKev("https://huggingface.co/{repo}/resolve/main/kev-0.8b", {{ ort, variant: "q8f32" }});
const res = await kev.systemOne({{
  state: "I was charged twice. Please fix this ASAP.",
  questions: {{ billing: {{ type: "noul", instructions: "Is this ticket about billing?" }} }},
}});
```

## Contents

{table}

Every bundle is int8 weights with fp32 activations, split into 32 MB files so any CDN or proxy can serve them.
`manifest.json` lists the files, their sizes, the tokenizer and the pointer head, plus the measured deviation from
the original fp32 PyTorch model on a fixture set.

## Provenance and licenses

- Models and training: [jaredpalmer/kev](https://github.com/jaredpalmer/kev) (Apache-2.0). Source checkpoints:
  {runs}.
- Base models: [Qwen3.5](https://huggingface.co/Qwen) (Apache-2.0).
- Architecture described in [Jev's Architecture Unmasked](https://archerhume.com/posts/jevs-architecture-unmasked).
  The API shapes follow [TypeSafe's System One](https://docs.typesafe.ai/api); Jev is TypeSafe's hosted model and is
  not affiliated with this repo.
- Conversion: LoRA merged in fp32, exported with the onnxruntime-genai model builder without the LM head, embeddings
  quantized to int8 per row. Details in the [kev.js README](https://github.com/ai-ecoverse/kev.js#readme).
"""


def published_files(name, manifest):
    """Repo paths of one model's download: manifest, tokenizer, head and every non-fp32 variant's files. fp32 is the
    local parity reference, 3 GB+, and not meant for browsers."""
    f = manifest["files"]
    paths = ["manifest.json", f["head"], f["tokenizer"], f["tokenizer_config"]]
    for v, meta in manifest["variants"].items():
        if v != "fp32": paths += [meta["model"], *meta["data"]]
    return [f"{name}/{p}" for p in paths]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="../public/models")
    ap.add_argument("--repo", default="ai-ecoverse/kev.js")
    ap.add_argument("--only", help="comma-separated model names; default all")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    everything = sorted(d for d in os.listdir(a.models) if os.path.exists(f"{a.models}/{d}/manifest.json"))
    names = a.only.split(",") if a.only else everything
    # the card describes every packaged model, not just the ones this run uploads
    manifests = {n: json.load(open(f"{a.models}/{n}/manifest.json")) for n in everything}

    rows = ["| Folder | Variant | Download | Base | Source checkpoint |", "|---|---|---|---|---|"]
    for n, m in manifests.items():
        for v, meta in m["variants"].items():
            if v == "fp32": continue
            rows.append(f"| `{n}` | `{v}` | {meta['bytes'] / 1e9:.2f} GB | `{m['base']}` | `{m['run']}` |")
    card = CARD.format(repo=a.repo, table="\n".join(rows),
                       bases="\n".join(f"- {b}" for b in sorted({m["base"] for m in manifests.values()})),
                       runs=", ".join(f"`{m['run']}`" for m in manifests.values()))

    total = sum(os.path.getsize(f"{a.models}/{p}") for n in names for p in published_files(n, manifests[n]))
    print(f"{a.repo}: {', '.join(names)} ({total / 1e9:.1f} GB)")
    if a.dry_run:
        print(card)
        return
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(a.repo, repo_type="model", private=a.private, exist_ok=True)
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=a.repo, repo_type="model")
    remote = set(api.list_repo_files(a.repo))
    for n in names:
        files = published_files(n, manifests[n])
        payload = [f for f in files if f != f"{n}/manifest.json"]
        print(f"uploading {n}: {len(files)} files")
        # 1. the revision's files, under their r-<sha>/ directory: nothing a live manifest points at is touched.
        #    Only what the manifest names is sent; upload_large_folder resumes and parallelises.
        api.upload_large_folder(repo_id=a.repo, repo_type="model", folder_path=a.models, allow_patterns=payload, num_workers=4)
        # 2. the manifest, alone: this commit is the switch to the new revision
        api.upload_file(path_or_fileobj=f"{a.models}/{n}/manifest.json", path_in_repo=f"{n}/manifest.json", repo_id=a.repo,
                        repo_type="model", commit_message=f"{n}: {manifests[n]['run']}")
        # 3. whatever the new manifest no longer names (older revisions, the pre-revision flat layout)
        stale = sorted(f for f in remote if f.startswith(f"{n}/") and f not in files)
        if stale:
            api.delete_files(repo_id=a.repo, repo_type="model", delete_patterns=stale, commit_message=f"{n}: drop {len(stale)} superseded files")
            print(f"  removed {len(stale)} superseded files")
    print(f"https://huggingface.co/{a.repo}")


if __name__ == "__main__":
    main()
