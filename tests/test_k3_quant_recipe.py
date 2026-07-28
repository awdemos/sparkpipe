#!/usr/bin/env python3
"""K3's quantisation must follow the checkpoint's recipe, not a global choice.

config.json's quantization_config quantises weights to 4 bits at group 32 and
carries an ignore list: self_attn, shared_experts, the dense mlp projections,
lm_head and the vision tower. The report's deployment section says the
quantisation-aware training ran from SFT onward - so the routed experts were
trained INTO that grid and nothing else was.

That makes the ignore list a correctness constraint rather than a preference. A
tensor that never saw QAT has no protection in a 4-bit grid, which is the same
reason a derived factorisation of those experts could not be stored at MXFP4
either: the grid is not the protection, the training into the grid is.

This gate checks that only the two routed-expert GEMMs take the quantised
Format and every other projection in the layer names a high-precision one.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LAYER = ROOT / "inference" / "llms" / "kimi_k3" / "layer.cuh"
BIND = ROOT / "inference" / "llms" / "kimi_k3" / "bind.cu"
# the only tensors the checkpoint quantises
QUANTISED = ("expert_w1", "expert_w2")


def main():
    text = re.sub(r"//[^\n]*", "", LAYER.read_text())
    failures = 0
    generic = []
    for match in re.finditer(r"K3Project<(\w+)>\(b,\s*[\w>\-\.]+,\s*b->(\w+)", text):
        fmt, tensor = match.group(1), match.group(2)
        if fmt == "Format" and not any(q in tensor for q in QUANTISED):
            generic.append(tensor)
            failures += 1
    for tensor in generic:
        print(f"  FAIL {tensor} takes the quantised Format; the checkpoint's "
              f"ignore list excludes it from QAT")
    # the expert GEMMs must still be reachable by Format, or nothing is quantised
    # K3Project is generic - its body says LmGemmLaunch<Format> and its CALLERS
    # choose the format. Counting its single call site as a quantised GEMM
    # flagged correct code, which is the false positive that teaches a reader to
    # skip the gate. Only direct launches outside the helper count.
    helper = re.search(r"static int32_t K3Project\b.*?\n\}", text, re.S)
    outside = text.replace(helper.group(0), "") if helper else text
    expert_gemms = len(re.findall(r"LmGemmLaunch<Format", outside))
    if expert_gemms != 2:
        print(f"  FAIL {expert_gemms} expert GEMMs take Format, expected 2")
        failures += 1
    # and Format must be the checkpoint's, not something else 4-bit
    bind = BIND.read_text()
    if "K3LaunchSlice<LmMxfp4" not in bind:
        print("  FAIL the slice does not instantiate LmMxfp4; the checkpoint is "
              "MXFP4 group 32 and a different 4-bit grid is a different model")
        failures += 1
    print(f"projections checked {len(re.findall(r'K3Project<', text))}, "
          f"expert GEMMs {expert_gemms}")
    if failures:
        print(f"\nFAIL ({failures})")
        return 1
    print("\nonly the routed experts are quantised, and at the grid they were "
          "trained in")
    return 0


if __name__ == "__main__":
    sys.exit(main())
