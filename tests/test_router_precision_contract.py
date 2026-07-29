#!/usr/bin/env python3
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def main():
    gemm = (ROOT / "inference/kernels/gemm.cuh").read_text()
    launch = (ROOT / "runtime/gemm.cuh").read_text()
    models = [
        "deepseek_v4",
        "glm5_2",
        "kimi_k3",
        "mimo_2_5",
    ]
    failures = []
    if "void *output_f32;" not in gemm:
        failures.append("GEMM has no FP32 output")
    if "(args->output_bf16 == 0) == (args->output_f32 == 0)" not in launch:
        failures.append("GEMM output selection is not fail-closed")
    for model in models:
        source = (
            ROOT / "inference/llms" / model / "layer.cuh"
        ).read_text()
        if "gemm.output_f32 = b->router_logits;" not in source:
            failures.append(f"{model} router is not FP32")
        if "output_bf16 = b->router_logits;" in source:
            failures.append(f"{model} router still writes BF16")
    kimi = (
        ROOT / "inference/llms/kimi_k3/layer.cuh"
    ).read_text()
    if "LmTopkSmallKernel<K3_LAYER_THREADS,K3_TOP_K,true,1u,1u,true>" not in kimi:
        failures.append("K3 FP32 logits do not take sigmoid in top-k")
    if failures:
        print("\n".join(failures))
        return 1
    print("router logits stay FP32 through selection")
    return 0


if __name__ == "__main__":
    sys.exit(main())
