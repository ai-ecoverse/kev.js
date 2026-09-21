#!/usr/bin/env bash
# Build the ONNX variants of a merged Kev checkpoint: build.sh build/kev-0.8b [variant...]
# variants: fp32-cpu (parity reference), q4f16-webgpu, q4-webgpu, fp16-webgpu, q4-cpu, q8f16-webgpu,
#           q4k-webgpu, q4kmix-webgpu, q4klin-webgpu, q4mm-webgpu (int4 MatMuls, embeddings left unquantized)
set -euo pipefail
dir=$1; shift
variants=${*:-fp32-cpu q4f16-webgpu q4-webgpu fp16-webgpu}
common=(exclude_lm_head=true exclude_mtp=true)
for v in $variants; do
  case $v in
    fp32-cpu)     args=(-p fp32 -e cpu --extra_options "${common[@]}") ;;
    q4-cpu)       args=(-p int4 -e cpu --extra_options "${common[@]}") ;;
    q4f16-webgpu) args=(-p int4 -e webgpu --extra_options "${common[@]}") ;;
    q4-webgpu)    args=(-p int4 -e webgpu --extra_options "${common[@]}" use_webgpu_fp32=true) ;;
    fp16-webgpu)  args=(-p fp16 -e webgpu --extra_options "${common[@]}") ;;
    q8f16-webgpu) args=(-p int8 -e webgpu --extra_options "${common[@]}") ;;
    q8f32-webgpu) args=(-p int8 -e webgpu --extra_options "${common[@]}" use_webgpu_fp32=true) ;;
    q4k-webgpu)   args=(-p int4 -e webgpu --extra_options "${common[@]}" algo_config=k_quant) ;;
    q4kmix-webgpu) args=(-p int4 -e webgpu --extra_options "${common[@]}" algo_config=k_quant_mixed) ;;
    q4klin-webgpu) args=(-p int4 -e webgpu --extra_options "${common[@]}" algo_config=k_quant_linear) ;;
    q8all-webgpu) args=(-p int8 -e webgpu --extra_options "${common[@]}" op_types_to_quantize=MatMul/Gather) ;;
    q4b16-webgpu) args=(-p int4 -e webgpu --extra_options "${common[@]}" block_size=16) ;;
    q4e16-webgpu) args=(-p int4 -e webgpu --extra_options "${common[@]}" nodes_to_exclude=/model/embed_tokens/Gather) ;;
    q4mm-webgpu)  args=(-p int4 -e webgpu --extra_options "${common[@]}" op_types_to_quantize=MatMul) ;;
    *) echo "unknown variant $v" >&2; exit 2 ;;
  esac
  echo "== $v"
  # the builder can abort in process teardown on macOS (libc++ "recursive_mutex lock failed") after everything is
  # written, so success is judged by genai_config.json, which it writes last
  rm -rf "$dir/onnx-$v"
  uv run python -m onnxruntime_genai.models.builder -i "$dir/merged" -o "$dir/onnx-$v" -c build/cache "${args[@]}" > "$dir/onnx-$v.log" 2>&1 || true
  [ -f "$dir/onnx-$v/genai_config.json" ] || { tr '\r' '\n' < "$dir/onnx-$v.log" | tail -20; exit 1; }
  du -sh "$dir/onnx-$v/model.onnx.data"
done
