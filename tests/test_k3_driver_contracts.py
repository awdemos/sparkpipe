#!/usr/bin/env python3
"""K3's driver-side performance contracts, gated as source.

Two contracts live in inference/llms/kimi_k3 and nowhere else, and both are
the kind that compiles while it rots:

  1. THE BF16 KDA STATE OPTION FAILS CLOSED. The slot is 6 MiB of fp32 per
     sequence per layer, ~40% of a B64 step's bytes (the roadmap's K3 state
     correction), so the half-width option is the biggest batch lever in the
     model - and LmDeltaRuleKernel still addresses the pool as float. A flag
     that launches against a half-width pool does not crash; it mis-strides
     every head and sequence and decodes fluently. So the consumer flag must
     exist, the bind must propagate it, the pool arithmetic must be exactly
     half, and every launch path must refuse it until the kernel grows the
     bf16-store variant the flag's comment specifies. Default off, and
     nothing in the tree may set it.

  2. THE LAYER PATH STAYS GRAPH-CAPTURABLE. Roadmap D10/D1 put CUDA graphs
     first on the attack list (~3,300 launches per K3 token). Capture breaks
     on host-device traffic between launches, so the layer, slice, engine
     and bind sources must never name a synchronising or copying CUDA call
     outside a comment.

The gather/indirect-A contract (roadmap D9) is a comment plus one structural
fact: the packed-row-to-source-token map the indirect GEMM would consume is
the same one the gather consumes today, so the gate holds them paired.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
K3 = ROOT / "inference" / "llms" / "kimi_k3"


def defines(path):
    text = path.read_text().replace("\\\n", " ")
    values = {}
    for name, value in re.findall(r"#define (K3_\w+)[ \t]+([^\n]+)", text):
        expression = re.sub(r"\(u?int\d+_t\)", "", value.strip())
        expression = re.sub(r"(?<=\d)ull|(?<=\d)u", "", expression)
        expression = re.sub(r"K3_\w+",
                            lambda m: str(values.get(m.group(0), 0)),
                            expression)
        try:
            values[name] = int(eval(expression))
        except Exception:
            pass
    return values


def main():
    config = defines(K3 / "config.h")
    layer = (K3 / "layer.cuh").read_text()
    slice_ = (K3 / "slice.cuh").read_text()
    failures = 0

    # -- the slot arithmetic: fp32 slot, and the option is exactly half -------
    heads = config.get("K3_KDA_HEADS", 0)
    slot = config.get("K3_KDA_STATE_SLOT_BYTES", 0)
    slot_bf16 = config.get("K3_KDA_STATE_SLOT_BYTES_BF16", 0)
    expect = heads * config.get("K3_KDA_KEY_DIM", 0) \
        * config.get("K3_KDA_VALUE_DIM", 0) * 4
    if slot != expect or slot_bf16 * 2 != slot or slot_bf16 == 0:
        print(f"  FAIL state slot {slot}, bf16 {slot_bf16}, expected "
              f"{expect} and exactly half")
        failures += 1

    # -- the flag exists on both structs and the bind propagates it -----------
    if "uint32_t kda_state_bf16;" not in layer:
        print("  FAIL K3LayerBuffers lost the bf16-state flag")
        failures += 1
    if "uint32_t kda_state_bf16;" not in slice_:
        print("  FAIL K3SliceState lost the bf16-state flag")
        failures += 1
    if "buffers->kda_state_bf16 = state->kda_state_bf16;" not in slice_:
        print("  FAIL the bind no longer propagates the state dtype flag")
        failures += 1
    if "K3_KDA_STATE_SLOT_BYTES_BF16" not in slice_:
        print("  FAIL the bind does not stride the pool by the flag; a pool "
              "bound at one width and launched at another aliases sequences")
        failures += 1

    # -- every launch path REFUSES the flag while the kernel is fp32-only -----
    for name, text, where in (
            ("layer.cuh", layer, "K3LayerKda"),
            ("slice.cuh", slice_, "K3FoldAccepted")):
        if not re.search(
                r"kda_state_bf16 != 0u \)\s*\n\s*return\(LM_LAUNCH_ERR_SHAPE\)",
                text):
            print(f"  FAIL {name}: {where} does not fail closed on the "
                  f"bf16-state flag; the delta kernel addresses the pool as "
                  f"float and a launched flag mis-strides every head")
            failures += 1

    # -- default off: the only assignment is the bind's propagation -----------
    for name in ("layer.cuh", "slice.cuh", "bind.cu", "unity.cu",
                 "engine.h", "dspark.h", "pipeline_sideband.h"):
        text = re.sub(r"//[^\n]*", "", (K3 / name).read_text())
        for match in re.finditer(r"kda_state_bf16\s*=\s*([^;\n]+);", text):
            value = match.group(1).strip()
            if value not in ("state->kda_state_bf16",):
                print(f"  FAIL {name} sets kda_state_bf16 to {value!r}; the "
                      f"option is admission-time and off by default, and no "
                      f"driver source may turn it on while launches refuse it")
                failures += 1

    # -- the layer path stays capturable: no host-device traffic --------------
    for name in ("layer.cuh", "slice.cuh", "bind.cu", "unity.cu", "engine.h"):
        text = re.sub(r"//[^\n]*", "", (K3 / name).read_text())
        for call in ("cudaMemcpy", "cudaMalloc", "cudaFree",
                     "cudaStreamSynchronize", "cudaDeviceSynchronize",
                     "cudaMemcpyAsync"):
            if call in text:
                print(f"  FAIL {name} names {call}; host-device traffic "
                      f"between launches is what breaks CUDA graph capture "
                      f"(roadmap D10)")
                failures += 1

    # -- the indirect-A contract: the gather and its removal map stay paired --
    gather = re.search(r"LmGatherRowsKernel<K3_LAYER_THREADS>.*?"
                       r"route_source_token.*?"
                       r"route_gather_bf16", layer, re.S)
    if gather is None:
        print("  FAIL the gather no longer reads route_source_token into "
              "route_gather_bf16; that map is also the consumer-side "
              "contract for the indirect-A grouped GEMM (roadmap D9)")
        failures += 1
    if "activation_row_index" not in layer:
        print("  FAIL the indirect-A contract comment is gone; the double-"
              "touch fix is specified there for the GEMM's owner")
        failures += 1

    if failures:
        print(f"\nFAIL ({failures})")
        return 1
    print("\nthe bf16 state option is wired, refused, and off; the layer "
          "path captures; the gather keeps its removal contract")
    return 0


if __name__ == "__main__":
    sys.exit(main())
