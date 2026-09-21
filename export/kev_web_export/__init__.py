"""Export a Kev checkpoint for the browser: merge LoRA, build ONNX (hidden states, no LM head), export the pointer head."""
import os, sys

# kev is not an installable package; use the pinned submodule
KEV_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "vendor", "kev"))
if KEV_ROOT not in sys.path: sys.path.insert(0, KEV_ROOT)
