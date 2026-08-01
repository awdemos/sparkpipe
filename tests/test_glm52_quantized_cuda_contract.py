#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path


def require(text: str, needle: str, label: str) -> None:
    if needle not in text:
        raise AssertionError(f"{label} is missing {needle!r}")


def forbid(text: str, needle: str, label: str) -> None:
    if needle in text:
        raise AssertionError(f"{label} contains forbidden {needle!r}")


def count_exact(text: str, needle: str, expected: int, label: str) -> None:
    actual = text.count(needle)
    if actual != expected:
        raise AssertionError(
            f"{label} contains {actual} copies of {needle!r}, expected {expected}"
        )


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    layer = (repository / "inference/llms/glm5_2/layer.cuh").read_text(
        encoding="utf-8"
    )
    binder = (repository / "inference/llms/glm5_2/bind.cu").read_text(
        encoding="utf-8"
    )
    unity = (repository / "inference/llms/glm5_2/unity.cu").read_text(
        encoding="utf-8"
    )
    stage_pack = (repository / "runtime/pack/stage_pack.py").read_text(
        encoding="utf-8"
    )
    expert_pack = (repository / "runtime/pack/fp8_resident_pack.py").read_text(
        encoding="utf-8"
    )
    targets = json.loads(
        (repository / "model_contracts/must_work_targets.json").read_text(
            encoding="utf-8"
        )
    )

    require(
        layer,
        "static int32_t Glm52LaunchBf16Linear(",
        "GLM 5.2 BF16 non-expert linear path",
    )
    require(
        layer,
        "LmAbsorbedProject<LmBf16Format>(",
        "GLM 5.2 BF16 attention projections",
    )
    require(
        layer,
        "LmGemmLaunch<\n        LmBf16Format,",
        "GLM 5.2 BF16 router",
    )
    count_exact(
        layer,
        "LmGemmWeightOnlyLaunch<\n        LmFp8,",
        2,
        "GLM 5.2 FP8 expert GEMMs",
    )
    count_exact(
        layer,
        "gemm.scale_a = LmScaleTensorNone();",
        4,
        "GLM 5.2 BF16 activation scale bypass",
    )
    count_exact(
        layer,
        "gemm.scale_b = LmScaleTensorBlockF32(",
        2,
        "GLM 5.2 FP8 expert scale planes",
    )
    forbid(
        layer,
        "LmQuantiseRowsKernel",
        "GLM 5.2 shipping layer path",
    )
    forbid(
        layer,
        "LmGemmLaunch<LmFp8",
        "GLM 5.2 symmetric FP8 execution",
    )

    for field in (
        "attention_norm_weight_bf16",
        "post_attention_norm_weight_bf16",
        "attention_output_weight_bf16",
        "dense_gate_weight_bf16",
        "dense_up_weight_bf16",
        "dense_down_weight_bf16",
        "moe_router_weight_bf16",
    ):
        require(binder, field, "GLM 5.2 BF16 binder")
    for field in (
        "w1_weight_fp8_e4m3",
        "w1_scale_inv_f32",
        "w2_weight_fp8_e4m3",
        "w2_scale_inv_f32",
    ):
        require(binder, field, "GLM 5.2 FP8 expert binder")
    require(
        binder,
        "SparkGlm52ResidentDecodeStageFp8MoePlanIsUsable(node)",
        "GLM 5.2 FP8 plan validation",
    )

    require(
        unity,
        "Glm52GemmBf16(",
        "GLM 5.2 BF16 GEMM export",
    )
    require(
        unity,
        "Glm52GemmFp8ExpertWeightBf16Activation(",
        "GLM 5.2 FP8-weight/BF16-activation export",
    )
    require(
        unity,
        "Glm52LayerMoeFp8ExpertWeightBf16Activation(",
        "GLM 5.2 mixed-precision MoE export",
    )
    for forbidden in (
        "Glm52LayerAttentionFp8",
        "Glm52LayerDenseMlpFp8",
        "Glm52LayerMoeFp8(",
        "Glm52LayerAttentionInt7",
        "Glm52LayerDenseMlpInt7",
        "Glm52LayerMoeInt7",
        "Glm52GemmInt7",
        "Glm52GemmInt6",
        "Glm52GemmInt8",
        "Glm52GemmNvfp4",
        "LmQuantiseRowsKernel<LmFp8",
    ):
        forbid(unity, forbidden, "GLM 5.2 shipping unity surface")

    require(
        stage_pack,
        "validate_bf16_non_expert_source_dtypes(",
        "GLM 5.2 stage-pack precision validator",
    )
    require(
        stage_pack,
        'f"{model_quantization} non-expert weights must be BF16: "',
        "GLM 5.2 stage-pack BF16 rejection",
    )
    require(
        expert_pack,
        'expected FP8 E4M3',
        "GLM 5.2 expert-pack FP8 rejection",
    )
    require(
        expert_pack,
        '"format": "sparkpipe.glm52.fp8.resident_moe_pack.v1"',
        "GLM 5.2 FP8 expert-pack identity",
    )

    glm_targets = [
        target for target in targets["targets"]
        if target.get("model_family") == "glm52"
    ]
    if len(glm_targets) != 1:
        raise AssertionError("must-work manifest must contain exactly one GLM 5.2 target")
    precision = glm_targets[0]
    if precision["routed_expert_weight_format"] != "fp8_e4m3":
        raise AssertionError("GLM 5.2 routed experts are not pinned to FP8 E4M3")
    if precision["routed_expert_activation_format"] != "bf16":
        raise AssertionError("GLM 5.2 expert activations are not pinned to BF16")
    if precision["non_expert_weight_format"] != "bf16":
        raise AssertionError("GLM 5.2 non-expert weights are not pinned to BF16")
    if precision["non_expert_activation_format"] != "bf16":
        raise AssertionError("GLM 5.2 non-expert activations are not pinned to BF16")
    if precision["accumulator_format"] != "fp32":
        raise AssertionError("GLM 5.2 accumulators are not pinned to FP32")

    print("PASS GLM-5.2 FP8-expert/BF16-rest CUDA source contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
