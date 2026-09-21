"""Resolve a Kev run to an exact Hub commit: jaredpalmer/kev-9b -> jaredpalmer/kev-9b@<sha>.

The Kev checkpoints are republished under the same repo ids (all three were updated on 2026-09-21), so an unpinned id
means "whatever main is at the moment". The export and its reference fixtures must come from the same commit; every
step takes the pinned form, and package.py refuses to combine different ones.

    uv run python -m kev_web_export.pin jaredpalmer/kev-9b      # prints the pinned run"""
import os, sys


def pin(run: str) -> str:
    if os.path.isdir(run): return run                      # a local run directory is already fixed
    repo, _, rev = run.partition("@")
    if rev and len(rev) == 40: return run                  # already a full commit sha
    from huggingface_hub import HfApi
    return f"{repo}@{HfApi().model_info(repo, revision=rev or None).sha}"


if __name__ == "__main__":
    print(pin(sys.argv[1]))
