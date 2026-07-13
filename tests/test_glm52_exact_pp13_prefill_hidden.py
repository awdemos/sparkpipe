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


def test_pp13_builder_matches_validated_tiled_attention(root: Path) -> None:
    source = (root / "modules" / "glm52_resident_decode_stage" / "source" /
              "spark_glm52_pp13_node_context_builder_cuda.cu").read_text(
                  encoding="utf-8")
    wire_start = source.index("static void SparkGlm52Pp13BuilderWireLayer(")
    wire_end = source.index(
        "static void SparkGlm52Pp13BuilderConfigureMtpLayer(", wire_start)
    wire_body = source[wire_start:wire_end]
    assert "node->key_nope_cache_bf16 = layer->key_nope_cache;" in wire_body
    assert "node->value_cache_bf16 = layer->value_cache;" in wire_body
    assert ("SPARK_GLM52_RESIDENT_DECODE_STAGE_ATTENTION_EXECUTION_"
            "TILED_ONLINE_SOFTMAX" in wire_body)
    assert ("SPARK_GLM52_RESIDENT_DECODE_STAGE_ATTENTION_EXECUTION_"
            "ABSORBED_LATENT" not in wire_body)
    assert ("SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_"
            "TILED_ONLINE_ATTENTION" in wire_body)
    assert source.count("ALLOC_FIELD(key_nope_cache,") == 2
    assert source.count("ALLOC_FIELD(value_cache,") == 2
    assert source.count("ZERO_FIELD(key_nope_cache,") == 2
    assert source.count("ZERO_FIELD(value_cache,") == 2


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
    runner_start = source.index(
        "static SparkStatus SparkGlm52Pp13BuilderRunPrefillFrame(")
    prefill_start = source.index(
        "static SparkStatus SparkGlm52Pp13BuilderPrefill(", runner_start)
    decode_start = source.index(
        "static SparkStatus SparkGlm52Pp13BuilderDecode(", prefill_start)
    runner_body = source[runner_start:prefill_start]
    prefill_body = source[prefill_start:decode_start]
    assert sync_call in runner_body
    assert progress_call in runner_body
    assert runner_body.index(progress_call, runner_body.index(sync_call)) > runner_body.index(sync_call)
    assert "SparkGlm52Pp13BuilderRunPrefillFrame(" in prefill_body


def test_prefill_probe_hashes_the_exact_stage_input(root: Path) -> None:
    source = (root / "modules" / "glm52_resident_decode_stage" / "source" /
              "spark_glm52_pp13_node_context_builder_cuda.cu").read_text(
                  encoding="utf-8")
    probe_start = source.index(
        "static void SparkGlm52Pp13BuilderMaybeProbePrefillInputHidden(")
    prepare_start = source.index(
        "static SparkStatus SparkGlm52Pp13BuilderPreparePrefillFrame(",
        probe_start)
    probe_body = source[probe_start:prepare_start]
    prefill_start = source.index(
        "static SparkStatus SparkGlm52Pp13BuilderPrefill(", prepare_start)
    decode_start = source.index(
        "static SparkStatus SparkGlm52Pp13BuilderDecode(", prefill_start)
    prefill_body = source[prefill_start:decode_start]
    assert "SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_BF16_BYTES" in probe_body
    assert "SparkGlm52Pp13BuilderProbeFnv64(" in probe_body
    assert "SparkGlm52Pp13BuilderMaybeProbePrefillInputHidden(" in prefill_body


def test_fp8_phase_probe_targets_the_first_divergent_layer(root: Path) -> None:
    cuda_source = (root / "modules" / "glm52_resident_decode_stage" /
                   "source" /
                   "spark_glm52_sm121_required_decode_stage.cu").read_text(
                       encoding="utf-8")
    builder_source = (root / "modules" / "glm52_resident_decode_stage" /
                      "source" /
                      "spark_glm52_pp13_node_context_builder_cuda.cu").read_text(
                          encoding="utf-8")
    probe_start = cuda_source.index(
        "static void SparkGlm52ResidentDecodeStageDeviceHashProbe(")
    probe_end = cuda_source.index(
        "static bool SparkGlm52ResidentDecodeStageFp8KvCachePlanIsUsableCuda(",
        probe_start)
    assert "node_context->layer_index != 0u" in cuda_source[
        probe_start:probe_end]
    assert "fp8_layer0_attention_probe" in builder_source


def test_fp8_validator_preserves_quantized_dense_execution(root: Path) -> None:
    source = (root / "modules" / "glm52_resident_decode_stage" /
              "validation" /
              "spark_glm52_resident_decode_stage_cuda_validation.cu").read_text(
                  encoding="utf-8")
    start = source.index("static bool SparkValidationBindRequiredLinearPlans(")
    end = source.index(
        "static bool SparkValidationInitializeDenseLayerCacheAliases(", start)
    function_body = source[start:end]
    assert function_body.count("use_quantized_dense_plans == 0u &&") == 2


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

    start = source.index(
        "static SparkStatus SparkGlm52Pp13BuilderEmitWideDecodeCompletions(")
    end = source.index(
        "static SparkStatus SparkGlm52Pp13BuilderLoadMtpPreviousTargets(",
        start)
    function_body = source[start:end]
    assert "completion.token_count = work_packet->rows_per_lane;" in function_body
    assert ("completion.token_count = accepted_draft_count + 1u;" not in
            function_body)
    assert "execution_row_base + token_index" in function_body


def test_mtp_full_vocab_workspace_uses_logical_lane_capacity(root: Path) -> None:
    source = (root / "modules" / "glm52_resident_decode_stage" / "source" /
              "spark_glm52_pp13_node_context_builder_cuda.cu").read_text(
                  encoding="utf-8")
    start = source.index(
        "static SparkStatus SparkGlm52Pp13BuilderLaunchMtpDraftPlan(")
    end = source.index(
        "static SparkStatus SparkGlm52Pp13BuilderLoadMtpWeights(", start)
    function_body = source[start:end]
    assert ("active_sequence_count,state->rank_plan.logical_lane_capacity," in
            function_body)
    assert ("active_sequence_count,state->rank_plan.execution_row_capacity," not in
            function_body)


def test_mtp_linear_plans_use_logical_rows(root: Path) -> None:
    source = (root / "modules" / "glm52_resident_decode_stage" / "source" /
              "spark_glm52_pp13_node_context_builder_cuda.cu").read_text(
                  encoding="utf-8")
    stage_start = source.index(
        "static SparkStatus SparkGlm52Pp13BuilderPrepareStageLinearPlanRows(")
    mtp_start = source.index(
        "static SparkStatus SparkGlm52Pp13BuilderPrepareMtpLinearPlanRows(",
        stage_start)
    stage_body = source[stage_start:mtp_start]
    assert "state->mtp_layer.linear_binding" not in stage_body

    draft_start = source.index(
        "static SparkStatus SparkGlm52Pp13BuilderLaunchMtpDraftPlan(")
    draft_end = source.index(
        "static SparkStatus SparkGlm52Pp13BuilderLoadMtpWeights(", draft_start)
    draft_body = source[draft_start:draft_end]
    prepare = "SparkGlm52Pp13BuilderPrepareMtpLinearPlanRows("
    draft_loop = "for (draft_index = 0u;"
    assert draft_body.index(prepare) < draft_body.index(draft_loop)
    assert "state,active_sequence_count);" in draft_body


def test_mtp_runtime_failures_name_the_failing_phase(root: Path) -> None:
    source = (root / "modules" / "glm52_resident_decode_stage" / "source" /
              "spark_glm52_pp13_node_context_builder_cuda.cu").read_text(
                  encoding="utf-8")
    start = source.index(
        "static SparkStatus SparkGlm52Pp13BuilderLaunchMtpLayer(")
    end = source.index(
        "static SparkStatus SparkGlm52Pp13BuilderLoadMtpWeights(", start)
    body = source[start:end]
    for phase in (
            "mtp_metadata",
            "mtp_fusion",
            "mtp_eh_projection",
            "mtp_required_layer",
            "mtp_prepare_linear_rows",
            "mtp_full_vocab_greedy",
            "mtp_store_draft"):
        assert '"' + phase + '"' in body


def test_layer_body_failures_are_never_silent(root: Path) -> None:
    source = (root / "modules" / "glm52_resident_decode_stage" / "source" /
              "spark_glm52_sm121_required_decode_stage.cu").read_text(
                  encoding="utf-8")
    start = source.index(
        "static SparkStatus SparkGlm52ResidentDecodeStageTraceLayerBodyStatus(",
        source.index("{", source.index(
            "static SparkStatus SparkGlm52ResidentDecodeStageTraceLayerBodyStatus(")))
    end = source.index("\n}\n", start)
    body = source[start:end]
    assert "getenv" not in body
    assert "layer_body_failed" in body


def test_plain_wide_decode_bypasses_dspark_finalizer(root: Path) -> None:
    source = (root / "modules" / "glm52_resident_decode_stage" / "source" /
              "spark_glm52_pp13_node_context_builder_cuda.cu").read_text(
                  encoding="utf-8")
    start = source.index(
        "static void SparkGlm52Pp13BuilderCompletePendingWork(")
    end = source.index(
        "static SparkStatus SparkGlm52Pp13BuilderBuildResidentKvTable(",
        start)
    function_body = source[start:end]
    plain_decode_guard = (
        "SparkGlm52Pp13BuilderWorkIsPlainDecodeBatch(work_packet) == 0u")
    captured_finalize = "SparkGlm52Pp13BuilderFinalizeCapturedCompletion("
    wide_finalize = "SparkGlm52Pp13BuilderEmitWideDecodeCompletions("
    assert plain_decode_guard in function_body
    assert function_body.index(plain_decode_guard) < function_body.index(
        captured_finalize)
    assert function_body.index(captured_finalize) < function_body.index(
        wide_finalize)


def test_resident_block_stride_is_independent_of_the_physical_pool(
        root: Path) -> None:
    builder = (root / "modules" / "glm52_resident_decode_stage" / "source" /
               "spark_glm52_pp13_node_context_builder_cuda.cu").read_text(
                   encoding="utf-8")
    module = (root / "modules" / "glm52_resident_decode_stage" / "source" /
              "spark_glm52_resident_decode_stage_module.c").read_text(
                  encoding="utf-8")
    assignment = (
        "node->max_blocks_per_sequence =\n"
        "\t\tSPARK_GLM52_PP13_BUILDER_MAX_BLOCKS_PER_SEQUENCE;")
    assert builder.count(assignment) == 2
    assert "SparkGlm52Pp13BuilderResidentMaxBlocksPerSequence" not in builder
    assert ("node_context->max_blocks_per_sequence >\n"
            "            node_context->kv_block_count") not in module
    assert "physical_block_index >= node_context->kv_block_count" in module


def test_service_backend_namespaces_ids_per_live_session(root: Path) -> None:
    source = (root / "src" /
              "spark_glm52_pp13_service_backend.c").read_text(
                  encoding="utf-8")
    assert "state->request_api.next_sequence_id = state->session_id_base" in source
    assert "service_configuration.request_id_base = state->session_id_base" in source
    assert "state->cuda_resident_next_sequence_number = state->session_id_base" in source


def test_final_event_pump_detects_disconnect_before_send(root: Path) -> None:
    source = (root / "tools" /
              "sparkpipe_glm52_pp13_rank_daemon.c").read_text(
                  encoding="utf-8")
    start = source.index("static uint32_t SparkGlm52Pp13DaemonPumpFinalEvents(")
    end = source.index("static SparkStatus SparkGlm52Pp13DaemonInitialize(", start)
    function_body = source[start:end]
    receive = "SparkGlm52Pp13DaemonPumpFinalEventReceive(runtime)"
    send = "SparkGlm52Pp13DaemonPumpFinalEventSend(runtime)"
    assert function_body.index(receive) < function_body.index(send)


def test_rank_queue_does_not_overtake_a_deferred_sequence_position(
        root: Path) -> None:
    source = (root / "tools" /
              "sparkpipe_glm52_pp13_rank_daemon.c").read_text(
                  encoding="utf-8")
    start = source.index("static uint32_t SparkGlm52Pp13DaemonPumpQueuedWork(")
    end = source.index("static void SparkGlm52Pp13DaemonHandleWork(", start)
    function_body = source[start:end]
    predecessor = "SparkGlm52Pp13DaemonHasQueuedPredecessor(runtime,packet)"
    forward = "SparkGlm52Pp13DaemonForwardWork(runtime,packet)"
    submit = "SparkGlm52Pp13DaemonSubmitWork(runtime,packet)"
    assert predecessor in source
    assert function_body.index(predecessor) < function_body.index(forward)
    assert function_body.index(predecessor) < function_body.index(submit)


def test_short_context_bypasses_indexshare_for_exact_prefix_attention(
        root: Path) -> None:
    source = (root / "modules" / "glm52_resident_decode_stage" / "source" /
              "spark_glm52_sm121_required_decode_stage.cu").read_text(
                  encoding="utf-8")
    start = source.index(
        "static SparkStatus SparkGlm52ResidentDecodeStageLaunchSparseIndexSelection(")
    end = source.index(
        "static uint32_t SparkGlm52ResidentDecodeStageFp8AmaxProbeEnabled(",
        start)
    function_body = source[start:end]
    prefix = "SparkGlm52ResidentDecodeStageLaunchContextPrefixSparseIndices("
    shared = "SPARK_GLM52_RESIDENT_DECODE_STAGE_SPARSE_INDEX_DSA_INDEXSHARE_SHARED"
    assert "pipeline_slot->dsa_candidate_count <=" in function_body
    assert function_body.index(prefix) < function_body.index(shared)


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
    test_prefill_probe_hashes_the_exact_stage_input(root)
    test_fp8_phase_probe_targets_the_first_divergent_layer(root)
    test_fp8_validator_preserves_quantized_dense_execution(root)
    test_speculative_verify_exposes_the_full_verifier_vector(root)
    test_mtp_full_vocab_workspace_uses_logical_lane_capacity(root)
    test_plain_wide_decode_bypasses_dspark_finalizer(root)
    test_resident_block_stride_is_independent_of_the_physical_pool(root)
    test_service_backend_namespaces_ids_per_live_session(root)
    test_final_event_pump_detects_disconnect_before_send(root)
    test_rank_queue_does_not_overtake_a_deferred_sequence_position(root)
    test_short_context_bypasses_indexshare_for_exact_prefix_attention(root)


if __name__ == "__main__":
    main()
