#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = {
    "flash": (
        ROOT / "model_contracts" / "dsv4_flash_authoritative.json",
        ROOT / "model-families" / "dsv4" / "include" / "sparkpipe" / "spark_dsv4_flash_model.h",
        ROOT / "model_contracts" / "dsv4_flash.json",
    ),
    "pro": (
        ROOT / "model_contracts" / "dsv4_pro_authoritative.json",
        ROOT / "model-families" / "dsv4" / "include" / "sparkpipe" / "spark_dsv4_pro_model.h",
        ROOT / "model_contracts" / "dsv4_pro.json",
    ),
}


def require_equal(actual: Any, expected: Any, description: str) -> None:
    if actual != expected:
        raise ValueError(f"{description}: expected {expected!r}, got {actual!r}")


def validate_contract(variant: str, contract: dict[str, Any]) -> None:
    model = contract["model"]
    attention = contract["attention"]
    hyper_connections = contract["hyper_connections"]
    moe = contract["moe"]
    precision = contract["precision"]
    ratios = attention["compression_ratios"]

    require_equal(contract["schema_version"], 1, f"{variant} schema version")
    require_equal(contract["architecture"], "DeepseekV4ForCausalLM", f"{variant} architecture")
    require_equal(model["mtp_layer_count"], 1, f"{variant} MTP layer count")
    require_equal(len(ratios), model["layer_count"] + model["mtp_layer_count"], f"{variant} compression-ratio count")
    require_equal(moe["experts_per_token"], 6, f"{variant} top-k")
    require_equal(moe["shared_expert_count"], 1, f"{variant} shared experts")
    require_equal(moe["hash_routed_layer_count"], 3, f"{variant} hash-routed layers")
    require_equal(hyper_connections["stream_count"], 4, f"{variant} hyper-connection streams")
    require_equal(precision["routed_expert_weight_format"], "checkpoint_fp4", f"{variant} expert precision")
    require_equal(precision["non_expert_linear_weight_format"], "fp8_e4m3_block_128x128", f"{variant} non-expert precision")
    require_equal(contract["qualification"]["cuda_target"], "sm_121a", f"{variant} CUDA target")
    require_equal(contract["qualification"]["production_ready"], False, f"{variant} readiness")
    if variant == "flash":
        require_equal(model["hidden_dimension"], 4096, "Flash hidden dimension")
        require_equal(model["layer_count"], 43, "Flash layer count")
        require_equal(model["attention_head_count"], 64, "Flash attention heads")
        require_equal(moe["routed_expert_count"], 256, "Flash routed experts")
        require_equal(attention["index_top_k"], 512, "Flash index top-k")
        require_equal(ratios[:2], [0, 0], "Flash bootstrap attention layers")
    elif variant == "pro":
        require_equal(model["hidden_dimension"], 7168, "Pro hidden dimension")
        require_equal(model["layer_count"], 61, "Pro layer count")
        require_equal(model["attention_head_count"], 128, "Pro attention heads")
        require_equal(moe["routed_expert_count"], 384, "Pro routed experts")
        require_equal(attention["index_top_k"], 1024, "Pro index top-k")
        require_equal(ratios[:2], [128, 128], "Pro bootstrap attention layers")
    else:
        raise ValueError(f"unknown DSV4 variant: {variant}")


def macro_prefix(variant: str) -> str:
    return f"SPARK_DSV4_{variant.upper()}"


def c_float(value: float) -> str:
    if value == int(value):
        return f"{int(value)}.0f"
    return f"{value:.10g}f"


def render_header(variant: str, contract: dict[str, Any]) -> str:
    model = contract["model"]
    attention = contract["attention"]
    hyper_connections = contract["hyper_connections"]
    moe = contract["moe"]
    precision = contract["precision"]
    ratios = attention["compression_ratios"]
    prefix = macro_prefix(variant)
    guard = f"SPARKPIPE_SPARK_DSV4_{variant.upper()}_MODEL_H"

    defines = [
        ("HIDDEN_DIMENSION", model["hidden_dimension"]),
        ("LAYER_COUNT", model["layer_count"]),
        ("MTP_LAYER_COUNT", model["mtp_layer_count"]),
        ("VOCAB_COUNT", model["vocabulary_size"]),
        ("MAXIMUM_CONTEXT_TOKENS", model["maximum_context_tokens"]),
        ("ATTENTION_HEAD_COUNT", model["attention_head_count"]),
        ("KV_HEAD_COUNT", model["kv_head_count"]),
        ("HEAD_DIMENSION", model["head_dimension"]),
        ("QK_ROPE_HEAD_DIMENSION", model["qk_rope_head_dimension"]),
        ("QUERY_LORA_RANK", model["query_lora_rank"]),
        ("OUTPUT_LORA_RANK", model["output_lora_rank"]),
        ("OUTPUT_GROUP_COUNT", model["output_group_count"]),
        ("INDEX_HEAD_COUNT", attention["index_head_count"]),
        ("INDEX_HEAD_DIMENSION", attention["index_head_dimension"]),
        ("INDEX_TOP_K", attention["index_top_k"]),
        ("SLIDING_WINDOW_TOKENS", attention["sliding_window_tokens"]),
        ("YARN_FACTOR", attention["yarn_factor"]),
        ("YARN_ORIGINAL_CONTEXT_TOKENS", attention["yarn_original_context_tokens"]),
        ("HYPER_CONNECTION_STREAM_COUNT", hyper_connections["stream_count"]),
        ("HYPER_CONNECTION_SINKHORN_ITERATIONS", hyper_connections["sinkhorn_iterations"]),
        ("ROUTED_EXPERT_COUNT", moe["routed_expert_count"]),
        ("SHARED_EXPERT_COUNT", moe["shared_expert_count"]),
        ("EXPERTS_PER_TOKEN", moe["experts_per_token"]),
        ("EXPERT_INTERMEDIATE_DIMENSION", moe["expert_intermediate_dimension"]),
        ("HASH_ROUTED_LAYER_COUNT", moe["hash_routed_layer_count"]),
        ("FP8_WEIGHT_BLOCK_ROWS", 128),
        ("FP8_WEIGHT_BLOCK_COLUMNS", 128),
    ]

    lines = [
        f"#ifndef {guard}",
        f"#define {guard}",
        "",
        "#include <stdint.h>",
        "",
        "/* Generated by tools/generate_dsv4_contracts.py. */",
        "",
    ]
    for suffix, value in defines:
        lines.append(f"#define {prefix}_{suffix} {value}u")
    lines.extend([
        f"#define {prefix}_RMS_NORM_EPSILON {c_float(model['rms_norm_epsilon'])}",
        f"#define {prefix}_ROPE_THETA {c_float(attention['rope_theta'])}",
        f"#define {prefix}_COMPRESSED_ROPE_THETA {c_float(attention['compressed_rope_theta'])}",
        f"#define {prefix}_HYPER_CONNECTION_EPSILON {c_float(hyper_connections['epsilon'])}",
        f"#define {prefix}_ROUTED_SCALING_FACTOR {c_float(moe['routed_scaling_factor'])}",
        f"#define {prefix}_SWIGLU_LIMIT {c_float(moe['swiglu_limit'])}",
        f"#define {prefix}_EXPERT_WEIGHTS_ARE_CHECKPOINT_FP4 1u",
        f"#define {prefix}_NON_EXPERT_WEIGHTS_ARE_FP8_E4M3 1u",
        f"#define {prefix}_DYNAMIC_ACTIVATIONS_ARE_FP8_E4M3 1u",
        f"#define {prefix}_RESIDUAL_STORAGE_IS_BF16 1u",
        f"#define {prefix}_ACCUMULATION_IS_FP32 1u",
        "",
        f"static const uint16_t SparkDsv4{variant.title()}CompressionRatios[{len(ratios)}u] =",
        "{",
    ])
    for offset in range(0, len(ratios), 12):
        group = ratios[offset:offset + 12]
        lines.append("    " + ", ".join(f"{value}u" for value in group) + ("," if offset + 12 < len(ratios) else ""))
    lines.extend([
        "};",
        "",
        f"static inline uint16_t SparkDsv4{variant.title()}BackboneCompressionRatio(uint32_t layer_index)",
        "{",
        f"    if (layer_index >= {prefix}_LAYER_COUNT)",
        "    {",
        "        return 0u;",
        "    }",
        f"    return SparkDsv4{variant.title()}CompressionRatios[layer_index];",
        "}",
        "",
        f"static inline uint16_t SparkDsv4{variant.title()}MtpCompressionRatio(void)",
        "{",
        f"    return SparkDsv4{variant.title()}CompressionRatios[{prefix}_LAYER_COUNT];",
        "}",
        "",
        f"#endif",
        "",
    ])
    return "\n".join(lines)


def render_normalized_contract(variant: str, contract: dict[str, Any]) -> str:
    result = dict(contract)
    result["generated"] = {
        "variant": variant,
        "header": f"model-families/dsv4/include/sparkpipe/spark_dsv4_{variant}_model.h",
        "compression_ratio_interpretation": "first layer_count entries are backbone layers; final entry is the one declared MTP layer",
    }
    return json.dumps(result, indent=2, sort_keys=True) + "\n"


def write_or_check(path: Path, content: str, check: bool) -> bool:
    if check:
        return path.exists() and path.read_text(encoding="utf-8") == content
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    stale: list[str] = []
    for variant, (source_path, header_path, normalized_path) in CONTRACTS.items():
        contract = json.loads(source_path.read_text(encoding="utf-8"))
        validate_contract(variant, contract)
        outputs = {
            header_path: render_header(variant, contract),
            normalized_path: render_normalized_contract(variant, contract),
        }
        for path, content in outputs.items():
            if not write_or_check(path, content, args.check):
                stale.append(str(path.relative_to(ROOT)))
    if stale:
        print("stale generated DSV4 contract files:")
        for path in stale:
            print(path)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
