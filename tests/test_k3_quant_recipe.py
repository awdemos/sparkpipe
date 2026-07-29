#!/usr/bin/env python3
"""K3's quantisation must follow the checkpoint's recipe, not a global choice.

THIS GATE CURRENTLY FAILS, AND IT IS RIGHT TO. A second external audit found
that the expert path quantises its ACTIVATIONS to MXFP4. The checkpoint sets
input_activations null: it quantises weights and says nothing about
activations, so an inference stack runs BF16 activations against streamed
MXFP4 weights.

The fix needs an asymmetric GEMM - A at 16 stored bits with no scale, B at 4
with an E8M0 scale every 32, existing BF16 MMA and FP32 accumulators - and that
is a change to the tile machinery rather than a call-site edit. It is not done.
The gate stays strict rather than being softened to green, because a suite that
passes over a known-wrong data path is worth less than one that says where it
is wrong. docs/K3_WEIGHT_ONLY_MXFP4.md has the full requirement.

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
    # ACTIVATIONS ARE NOT QUANTISED, AND THIS GATE DID NOT CHECK.
    #
    # The checkpoint sets input_activations null. It quantises weights and says
    # nothing about activations, so an inference stack runs BF16 activations
    # against streamed MXFP4 weights. This file passed while the expert path
    # quantised its activations to MXFP4, because it only ever asked which
    # WEIGHT projections took the quantised Format.
    #
    # A second audit found it. The check is one line and its absence was the
    # whole of the defect.
    for match in re.finditer(r"K3Quantise<(\w+)>", text):
        if match.group(1) == "Format":
            print("  FAIL an activation is quantised with the weight Format; "
                  "the checkpoint sets input_activations null")
            failures += 1

    # the expert GEMMs must be WEIGHT-ONLY launches of the Format: BF16
    # activations against the quantised stream, E8M0 decoded in the load. A
    # symmetric LmGemmLaunch<Format> would quantise the activations again, so
    # its count outside the helper must be zero, not two.
    helper = re.search(r"static int32_t K3Project\b.*?\n\}", text, re.S)
    outside = text.replace(helper.group(0), "") if helper else text
    expert_gemms = len(re.findall(r"LmGemmWeightOnlyLaunch<Format", outside))
    if expert_gemms != 2:
        print(f"  FAIL {expert_gemms} weight-only expert GEMMs take Format, "
              f"expected 2")
        failures += 1
    symmetric = len(re.findall(r"LmGemmLaunch<Format", outside))
    if symmetric != 0:
        print(f"  FAIL {symmetric} symmetric GEMMs take Format; a symmetric "
              f"launch quantises the activations the checkpoint leaves alone")
        failures += 1
    if "K3Quantise" in text:
        print("  FAIL an activation quantiser still exists in the layer; the "
              "recipe has no place for one")
        failures += 1
    # and the route expansion must be the BF16 gather, or the first expert GEMM
    # reads rows that were never expanded
    if "LmGatherRowsKernel" not in text:
        print("  FAIL the route expansion gather is missing; the quantiser "
              "used to do it implicitly")
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
