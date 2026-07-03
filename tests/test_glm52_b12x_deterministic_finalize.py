#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def require_contains(path: str, needle: str) -> None:
    text = (ROOT / path).read_text()
    if needle not in text:
        raise AssertionError(f"missing {needle!r} in {path}")


def main() -> int:
    require_contains(
        "modules/glm52_sm121_b12x_compiled_backend/include/sparkpipe/spark_glm52_sm121_b12x_generated_kernel_table.h",
        "SPARK_GLM52_SM121_B12X_GENERATED_MANIFEST_REQUIRED_FLAGS",
    )
    require_contains(
        "modules/glm52_sm121_b12x_compiled_backend/source/spark_flashinfer_b12x_compiled_moe_backend.cu",
        "SparkGlm52B12xDeterministicFc2FinalizeKernel",
    )
    require_contains(
        "modules/glm52_sm121_b12x_compiled_backend/source/spark_flashinfer_b12x_compiled_moe_backend.cu",
        "generated_arguments.output_bf16 =\n        state->workspaces[bucket_index].route_output_bf16;",
    )
    require_contains(
        "third_party/flashinfer/flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_static_kernel.py",
        "st_global_i32(get_ptr_as_int64(token_map, map_idx), pair_idx)",
    )
    require_contains(
        "third_party/flashinfer/flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_micro_kernel.py",
        "unique_tok = local_expert_idx",
    )
    require_contains(
        "third_party/flashinfer/flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dynamic_kernel.py",
        "get_ptr_as_int64(token_map, phys_row), pair_idx",
    )
    require_contains(
        "third_party/flashinfer/flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dispatch.py",
        "deterministic_route_output",
    )
    require_contains(
        "third_party/flashinfer/flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dispatch.py",
        "scatter_rows = routed_rows if _sparkpipe_b12x_deterministic_route_output_enabled() else m",
    )
    require_contains(
        "tools/glm52_b12x_aot_compile.py",
        "SPARKPIPE_B12X_DETERMINISTIC_ROUTE_OUTPUT",
    )
    require_contains(
        "tools/glm52_b12x_aot_compile.py",
        "token_count = int(key[5])",
    )
    require_contains(
        "tools/glm52_b12x_aot_compile.py",
        "token_count = int(key[4])",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
