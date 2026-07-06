#!/usr/bin/env python3
from pathlib import Path


def test_final_stage_has_hidden_only_builtin_launcher(root: Path) -> None:
    source = (root / "modules" / "glm52_resident_decode_stage" / "source" /
              "spark_glm52_sm121_required_decode_stage.cu").read_text(
                  encoding="utf-8")
    for bucket in (16, 32, 64, 128, 256, 512, 1024):
        needle = (
            "SPARK_GLM52_EXACT_PP13_BUILTIN_AOT_LAUNCHER_ENTRY"
            f"(stage_index, SPARK_GLM52_STAGE_PLAN_BUCKET_B{bucket}, final_token_stage)")
        assert needle in source
    assert "SPARK_GLM52_EXACT_PP13_BUILTIN_AOT_LAUNCHER_BUCKETS(12u, 0u)" in source
    assert "SPARK_GLM52_EXACT_PP13_BUILTIN_AOT_LAUNCHER_BUCKETS(12u, 1u)" in source


def test_exact_pp13_final_stage_can_run_hidden_only(root: Path) -> None:
    source = (root / "modules" / "glm52_resident_decode_stage" / "source" /
              "spark_glm52_sm121_required_decode_stage.cu").read_text(
                  encoding="utf-8")
    rejected_contract = (
        "final_token_stage !=\n"
        "            (exact_stage_slice_plan->first_layer_index + 6u =="
    )
    allowed_contract = (
        "(final_token_stage != 0u &&\n"
        "         exact_stage_slice_plan->first_layer_index + 6u !="
    )
    assert rejected_contract not in source
    assert allowed_contract in source


def test_pp13_rank_capacity_is_not_fixed_batch(root: Path) -> None:
    source = (root / "modules" / "glm52_resident_decode_stage" / "source" /
              "spark_glm52_pp13_node_context_builder_cuda.cu").read_text(
                  encoding="utf-8")
    reserved_block_start = source.index("node->reserved_execution_flags =")
    reserved_block_end = source.index("if ((state->rank_plan.flags &", reserved_block_start)
    reserved_block = source[reserved_block_start:reserved_block_end]
    assert "SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_FIXED_ACTIVE_BATCH" not in reserved_block


def test_pp13_rank_does_not_enable_dsa_fragment_transport(root: Path) -> None:
    source = (root / "modules" / "glm52_resident_decode_stage" / "source" /
              "spark_glm52_pp13_node_context_builder_cuda.cu").read_text(
                  encoding="utf-8")
    assert ("~SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_DSA_KV_FRAGMENT_TRANSPORT"
            in source)
    assert "layer->node.dsa_kv_fragment_prefetch_plan = 0;" in source
    assert "layer->node.dsa_kv_fragment_save_plan = 0;" in source


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    test_final_stage_has_hidden_only_builtin_launcher(root)
    test_exact_pp13_final_stage_can_run_hidden_only(root)
    test_pp13_rank_capacity_is_not_fixed_batch(root)
    test_pp13_rank_does_not_enable_dsa_fragment_transport(root)


if __name__ == "__main__":
    main()
