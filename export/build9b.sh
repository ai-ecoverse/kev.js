#!/usr/bin/env bash
# Kev-9B end to end: merge, int8 build, 32 MB shards, PyTorch reference fixtures, parity, package.
# No fp32 ONNX reference here (it would be ~36 GB); parity is measured straight against PyTorch.
set -euo pipefail
cd "$(dirname "$0")"
log() { echo "[9b] $*"; }
# FROM=postprocess skips the merge and build (the slow, network-bound part) when resuming after a failure
if [ "${FROM:-merge}" = merge ]; then
  log "merge"
  uv run python -m kev_web_export.merge --run jaredpalmer/kev-9b --out build/kev-9b
  log "build q8f32-webgpu"
  ./build.sh build/kev-9b q8f32-webgpu
fi
log "postprocess"
uv run python -m kev_web_export.postprocess --src build/kev-9b/onnx-q8f32-webgpu --out build/kev-9b/web-q8f32
log "fixtures (PyTorch reference)"
uv run python -m kev_web_export.fixtures --run jaredpalmer/kev-9b --n 20 --out ../fixtures/kev-9b.json
log "parity"
uv run python -m kev_web_export.parity --model build/kev-9b/web-q8f32/model.onnx --head build/kev-9b/head.safetensors --fixtures ../fixtures/kev-9b.json
log "package"
uv run python -m kev_web_export.package --build build/kev-9b --out ../public/models/kev-9b --variant q8f32=web-q8f32 --fixtures ../fixtures/kev-9b.json
log "done"
