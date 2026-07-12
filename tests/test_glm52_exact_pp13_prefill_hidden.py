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


def test_prebound_linear_plan_accepts_smaller_active_count(root: Path) -> None:
    source = (root / "modules" / "glm52_resident_decode_stage" / "source" /
              "spark_glm52_sm121_required_decode_stage.cu").read_text(
                  encoding="utf-8")
    start = source.index(
        "static SparkStatus SparkGlm52ResidentDecodeStageMaybeLaunchPreboundLinearPlan(")
    end = source.index(
        "static SparkStatus SparkGlm52ResidentDecodeStageLaunchMtpDraft(",
        start)
    function_body = source[start:end]
    assert "linear_plan_active_mismatch" not in function_body
    assert "active_sequence_count != linear_plan->maximum_active_sequence_count" not in function_body


def test_fp8_linear_plans_require_scaled_gemm_backend(root: Path) -> None:
    source = (root / "modules" / "glm52_resident_decode_stage" / "source" /
              "spark_glm52_sm121_required_decode_stage.cu").read_text(
                  encoding="utf-8")
    start = source.index(
        "static SparkStatus SparkGlm52ResidentDecodeStageLaunchBlackwellBuiltInQuantizedTensorCoreLinearPlan(")
    end = source.index(
        "static SparkStatus SparkGlm52ResidentDecodeStageLaunchLinear(",
        start)
    function_body = source[start:end]
    fp8_start = function_body.index(
        "SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_FP8_E4M3")
    fp8_end = function_body.index("grid = dim3(", fp8_start)
    fp8_branch = function_body[fp8_start:fp8_end]
    assert "if (backend == 0)" in fp8_branch
    assert "return SPARK_STATUS_MODULE_NOT_VALIDATED;" in fp8_branch
    assert "LaunchFp8E4m3ActivationWeightLinearScaledGemmBackend" in fp8_branch
    assert "SupportedQuantizedBf16WmmaLinearKernel" not in fp8_branch


def test_pp13_builder_binds_all_fp8_linear_plans(root: Path) -> None:
    source = (root / "modules" / "glm52_resident_decode_stage" / "source" /
              "spark_glm52_pp13_node_context_builder_cuda.cu").read_text(
                  encoding="utf-8")
    start = source.index("static SparkStatus SparkGlm52Pp13BuilderBindLayerPlans(")
    end = source.index("static SparkStatus SparkGlm52Pp13BuilderBindFp8Moe(", start)
    function_body = source[start:end]
    regular_bind = function_body.index(
        "SparkGlm52Sm121RequiredDecodeStageBindBlackwellQuantizedRegularLinearPlans(")
    fp8_bind = function_body.index(
        "SparkGlm52Sm121RequiredDecodeStageBindFp8E4m3LinearPlansScaledGemmBackend(")
    assert fp8_bind > regular_bind
    assert "&state->fp8_scaled_gemm_backend" in function_body[fp8_bind:]


def test_serial_prefill_progresses_runner_after_each_token(root: Path) -> None:
    source = (root / "modules" / "glm52_resident_decode_stage" / "source" /
              "spark_glm52_pp13_node_context_builder_cuda.cu").read_text(
                  encoding="utf-8")
    sync_call = "cudaStreamSynchronize(state->stream)"
    progress_call = "SparkGlm52ResidentDecodeStageProductionRunnerProgress("
    start = source.index("static SparkStatus SparkGlm52Pp13BuilderPrefill(")
    end = source.index("static SparkStatus SparkGlm52Pp13BuilderDecode(", start)
    function_body = source[start:end]
    assert sync_call in function_body
    assert progress_call in function_body
    assert function_body.index(progress_call, function_body.index(sync_call)) > function_body.index(sync_call)


def test_speculative_verify_exposes_the_full_verifier_vector(root: Path) -> None:
    source = (root / "modules" / "glm52_resident_decode_stage" / "source" /
              "spark_glm52_pp13_node_context_builder_cuda.cu").read_text(
                  encoding="utf-8")
    start = source.index(
        "static SparkStatus SparkGlm52Pp13BuilderFinalizeSpeculativeVerify(")
    end = source.index(
        "static SparkStatus SparkGlm52Pp13BuilderFinalizeCapturedCompletion(",
        start)
    function_body = source[start:end]
    assert ("state->speculative_verify_draft_count + 1u;" in
            function_body)
    assert ("state->speculative_verify_accepted_count + 1u;" not in
            function_body)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    test_final_stage_has_hidden_only_builtin_launcher(root)
    test_exact_pp13_final_stage_can_run_hidden_only(root)
    test_pp13_rank_capacity_is_not_fixed_batch(root)
    test_pp13_rank_does_not_enable_dsa_fragment_transport(root)
    test_prebound_linear_plan_accepts_smaller_active_count(root)
    test_fp8_linear_plans_require_scaled_gemm_backend(root)
    test_pp13_builder_binds_all_fp8_linear_plans(root)
    test_serial_prefill_progresses_runner_after_each_token(root)
    test_speculative_verify_exposes_the_full_verifier_vector(root)


if __name__ == "__main__":
    main()
