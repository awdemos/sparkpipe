// GLM 5.2's answer to the stage's two validation questions: is this node
// context runnable, and is this slice context runnable. The 1,577 lines
// that used to sit inside the common module - fifteen functions of glm
// projection catalogs, glm dimension checks, glm MoE plan requirements -
// are glm source now, reached through the two neutral entry points the
// common module declares and THE LINKER RESOLVES. rules.mk already
// compiles the common module once per model family; the seam was always
// the link step, it just had nothing model-side to link.

#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include "sparkpipe/spark_glm52_resident_decode_stage_firmware.h"

static SparkStatus SparkValidateGlm52ResidentDecodeStageRequiredQuantizedProjectionPlan(
    const SparkResidentDecodeStageNodeContext *node_context,
    uint32_t plan_index,
    uint32_t input_dimension,
    uint32_t output_dimension)
{
    if (!SparkGlm52ResidentDecodeStageLinearPlanIsUsable(
            node_context,
            plan_index,
            input_dimension,
            output_dimension) ||
        !SparkGlm52ResidentDecodeStageLinearPlanHasQuantizedProjectionKind(
            node_context,
            plan_index))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkValidateGlm52ResidentDecodeStageRequiredLinearPlan(
    const SparkResidentDecodeStageNodeContext *node_context,
    uint32_t plan_index,
    uint32_t input_dimension,
    uint32_t output_dimension)
{
    return SparkGlm52ResidentDecodeStageLinearPlanIsUsable(
            node_context,
            plan_index,
            input_dimension,
            output_dimension)
        ? SPARK_STATUS_OK
        : SPARK_STATUS_INVALID_ARGUMENT;
}

static SparkStatus SparkValidateGlm52ResidentDecodeStageRequiredProjectionPlans(
    const SparkResidentDecodeStageNodeContext *node_context)
{
    SparkStatus status;

    if (node_context->projection_mode ==
        SPARK_RESIDENT_DECODE_STAGE_PROJECTION_LOWERED_BF16)
    {
        status = SparkValidateGlm52ResidentDecodeStageRequiredLinearPlan(
            node_context,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_QUERY_LATENT,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
            SPARK_RESIDENT_DECODE_STAGE_QUERY_LATENT_PROJECTION_DIMENSION);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        status = SparkValidateGlm52ResidentDecodeStageRequiredLinearPlan(
            node_context,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_QUERY_ROPE,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
            SPARK_RESIDENT_DECODE_STAGE_QUERY_ROPE_PROJECTION_DIMENSION);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        status = SparkValidateGlm52ResidentDecodeStageRequiredLinearPlan(
            node_context,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_KEY_ROPE,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_ROPE_DIMENSION);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        return SparkValidateGlm52ResidentDecodeStageRequiredLinearPlan(
            node_context,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_KV_LATENT,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION);
    }

    if (node_context->projection_mode ==
            SPARK_RESIDENT_DECODE_STAGE_PROJECTION_RAW_FP8_E4M3 ||
        SparkResidentDecodeStageProjectionModeUsesQuantizedPlan(
            node_context))
    {
        status = SparkValidateGlm52ResidentDecodeStageRequiredQuantizedProjectionPlan(
            node_context,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_RAW_QUERY_A,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        status = SparkValidateGlm52ResidentDecodeStageRequiredQuantizedProjectionPlan(
            node_context,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_RAW_QUERY_B,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_B_DIMENSION);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        status = SparkValidateGlm52ResidentDecodeStageRequiredQuantizedProjectionPlan(
            node_context,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_RAW_KV_A,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_KV_A_DIMENSION);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        return SparkValidateGlm52ResidentDecodeStageRequiredQuantizedProjectionPlan(
            node_context,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_RAW_KV_B,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_KV_B_DIMENSION);
    }

    status = SparkValidateGlm52ResidentDecodeStageRequiredLinearPlan(
        node_context,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_RAW_QUERY_A,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkValidateGlm52ResidentDecodeStageRequiredLinearPlan(
        node_context,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_RAW_QUERY_B,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_B_DIMENSION);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkValidateGlm52ResidentDecodeStageRequiredLinearPlan(
        node_context,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_RAW_KV_A,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_KV_A_DIMENSION);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    return SparkValidateGlm52ResidentDecodeStageRequiredLinearPlan(
        node_context,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_RAW_KV_B,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_KV_B_DIMENSION);
}

static SparkStatus SparkValidateGlm52ResidentDecodeStageRequiredProjectionAndOutputPlans(
    const SparkResidentDecodeStageNodeContext *node_context)
{
    SparkStatus status;

    status = SparkValidateGlm52ResidentDecodeStageRequiredProjectionPlans(
        node_context);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (node_context->projection_mode ==
            SPARK_RESIDENT_DECODE_STAGE_PROJECTION_RAW_FP8_E4M3 ||
        SparkResidentDecodeStageProjectionModeUsesQuantizedPlan(
            node_context))
    {
        return SparkValidateGlm52ResidentDecodeStageRequiredQuantizedProjectionPlan(
            node_context,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_ATTENTION_OUTPUT,
            SPARK_RESIDENT_DECODE_STAGE_ATTENTION_PROJECTION_DIMENSION,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION);
    }
    return SparkValidateGlm52ResidentDecodeStageRequiredLinearPlan(
        node_context,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_ATTENTION_OUTPUT,
        SPARK_RESIDENT_DECODE_STAGE_ATTENTION_PROJECTION_DIMENSION,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION);
}

static SparkStatus SparkValidateGlm52ResidentDecodeStageRequiredDenseMlpPlans(
    const SparkResidentDecodeStageNodeContext *node_context)
{
    SparkStatus status;

    if (SparkResidentDecodeStageMlpExecutionUsesQuantizedPlan(
            node_context))
    {
        status = SparkValidateGlm52ResidentDecodeStageRequiredQuantizedProjectionPlan(
            node_context,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_DENSE_GATE,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
            node_context->dense_intermediate_dimension);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        status = SparkValidateGlm52ResidentDecodeStageRequiredQuantizedProjectionPlan(
            node_context,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_DENSE_UP,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
            node_context->dense_intermediate_dimension);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        return SparkValidateGlm52ResidentDecodeStageRequiredQuantizedProjectionPlan(
            node_context,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_DENSE_DOWN,
            node_context->dense_intermediate_dimension,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION);
    }

    status = SparkValidateGlm52ResidentDecodeStageRequiredLinearPlan(
        node_context,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_DENSE_GATE,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
        node_context->dense_intermediate_dimension);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkValidateGlm52ResidentDecodeStageRequiredLinearPlan(
        node_context,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_DENSE_UP,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
        node_context->dense_intermediate_dimension);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    return SparkValidateGlm52ResidentDecodeStageRequiredLinearPlan(
        node_context,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_DENSE_DOWN,
        node_context->dense_intermediate_dimension,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION);
}

static SparkStatus SparkValidateGlm52ResidentDecodeStageFullStageFastPath(
    const SparkResidentDecodeStageNodeContext *node_context)
{
    if (!SparkResidentDecodeStageFullStagePlanIsUsable(node_context))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkResidentDecodeStageExecutionFlagIsSet(
            node_context,
            SPARK_RESIDENT_DECODE_STAGE_EXECUTION_FORBID_DEBUG_SYNCHRONIZATION) &&
        (node_context->launch_check_mode !=
             SPARK_RESIDENT_DECODE_STAGE_LAUNCH_CHECK_NONE ||
         node_context->phase_clock_mode !=
             SPARK_RESIDENT_DECODE_STAGE_PHASE_CLOCK_DISABLED))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkResidentDecodeStageExecutionFlagIsSet(
            node_context,
            SPARK_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_BULK_PREFILL) &&
        !SparkResidentDecodeStageBulkPrefillPlanIsUsable(node_context))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkResidentDecodeStageExecutionFlagIsSet(
            node_context,
            SPARK_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_VALIDATED_LATENCY) &&
        node_context->validated_stage_latency_ns == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkValidateGlm52ResidentDecodeStageTensorCoreAlignment(
    const SparkResidentDecodeStageNodeContext *node_context)
{
    uint32_t pipeline_slot_index;
    uint32_t hidden_output_only;

    if (node_context == 0 || node_context->pipeline_slots == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    hidden_output_only =
        (node_context->reserved_execution_flags &
         SPARK_RESIDENT_DECODE_STAGE_EXECUTION_OUTPUT_HIDDEN_ONLY) != 0u;
    if (!SparkResidentDecodeStagePointerHasTensorCoreAlignment(
            node_context->mla_cache_bf16) ||
        !SparkResidentDecodeStagePointerHasTensorCoreAlignment(
            node_context->attention_norm_weight_bf16))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (hidden_output_only == 0u &&
        (!SparkResidentDecodeStagePointerHasTensorCoreAlignment(
             node_context->final_norm_weight_bf16) ||
         !SparkResidentDecodeStagePointerHasTensorCoreAlignment(
             node_context->restricted_lm_head_weight_bf16)))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    for (pipeline_slot_index = 0u;
         pipeline_slot_index < node_context->pipeline_slot_count;
         ++pipeline_slot_index)
    {
        const SparkResidentDecodeStagePipelineSlot *pipeline_slot;

        pipeline_slot = &node_context->pipeline_slots[pipeline_slot_index];
        if (!SparkResidentDecodeStagePointerHasTensorCoreAlignment(
                pipeline_slot->input_hidden_bf16) ||
            !SparkResidentDecodeStagePointerHasTensorCoreAlignment(
                pipeline_slot->normalized_hidden_bf16) ||
            !SparkResidentDecodeStagePointerHasTensorCoreAlignment(
                pipeline_slot->query_latent_bf16) ||
            !SparkResidentDecodeStagePointerHasTensorCoreAlignment(
                pipeline_slot->attention_output_latent_bf16) ||
            !SparkResidentDecodeStagePointerHasTensorCoreAlignment(
                pipeline_slot->attention_projected_hidden_bf16) ||
            !SparkResidentDecodeStagePointerHasTensorCoreAlignment(
                pipeline_slot->post_attention_hidden_bf16) ||
            !SparkResidentDecodeStagePointerHasTensorCoreAlignment(
                pipeline_slot->layer_output_hidden_bf16))
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkValidateGlm52ResidentDecodeStageFastPathContract(
    const SparkResidentDecodeStageNodeContext *node_context)
{
    uint32_t known_flags;
    SparkStatus status;

    known_flags = SPARK_RESIDENT_DECODE_STAGE_EXECUTION_KNOWN_FLAGS;
    if ((node_context->reserved_execution_flags & ~known_flags) != 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkResidentDecodeStageExecutionFlagIsSet(
            node_context,
            SPARK_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_TENSOR_CORE_ALIGNMENT))
    {
        status = SparkValidateGlm52ResidentDecodeStageTensorCoreAlignment(
            node_context);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }
    if (SparkResidentDecodeStageExecutionFlagIsSet(
            node_context,
            SPARK_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_FULL_STAGE_PLAN))
    {
        return SparkValidateGlm52ResidentDecodeStageFullStageFastPath(
            node_context);
    }
    if (SparkResidentDecodeStageExecutionFlagIsSet(
            node_context,
            SPARK_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_PREBOUND_PROJECTIONS))
    {
        if (!SparkResidentDecodeStageProjectionBackendIsPrebound(
                node_context))
        {
            SparkResidentDecodeStageReportValidationFailure(
                node_context,
                "fast_projection_backend",
                SPARK_STATUS_INVALID_ARGUMENT);
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        status = SparkValidateGlm52ResidentDecodeStageRequiredProjectionAndOutputPlans(
            node_context);
        if (status != SPARK_STATUS_OK)
        {
            SparkResidentDecodeStageReportValidationFailure(
                node_context,
                "fast_projection_plans",
                status);
            return status;
        }
    }
    if (SparkResidentDecodeStageExecutionFlagIsSet(
            node_context,
            SPARK_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_TILED_ONLINE_ATTENTION) &&
        node_context->attention_execution_mode !=
            SPARK_RESIDENT_DECODE_STAGE_ATTENTION_EXECUTION_TILED_ONLINE_SOFTMAX)
    {
        SparkResidentDecodeStageReportValidationFailure(
            node_context,
            "fast_attention_mode",
            SPARK_STATUS_INVALID_ARGUMENT);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkResidentDecodeStageExecutionFlagIsSet(
            node_context,
            SPARK_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_GRAPH_REPLAY) &&
        node_context->enable_cuda_graph_replay == 0u)
    {
        SparkResidentDecodeStageReportValidationFailure(
            node_context,
            "fast_graph_replay",
            SPARK_STATUS_INVALID_ARGUMENT);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkResidentDecodeStageExecutionFlagIsSet(
            node_context,
            SPARK_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_PRESELECTED_SPARSE_INDICES) &&
        node_context->sparse_index_mode !=
            SPARK_RESIDENT_DECODE_STAGE_SPARSE_INDEX_PRESELECTED)
    {
        SparkResidentDecodeStageReportValidationFailure(
            node_context,
            "fast_sparse_indices",
            SPARK_STATUS_INVALID_ARGUMENT);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkResidentDecodeStageExecutionFlagIsSet(
            node_context,
            SPARK_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_FAST_MLP))
    {
        if (node_context->layer_progression_mode ==
            SPARK_RESIDENT_DECODE_STAGE_LAYER_DENSE_BF16_MLP)
        {
            if (node_context->mlp_execution_mode !=
                    SPARK_RESIDENT_DECODE_STAGE_MLP_EXECUTION_PREBOUND_TENSOR_CORE &&
                node_context->mlp_execution_mode !=
                    SPARK_RESIDENT_DECODE_STAGE_MLP_EXECUTION_PREBOUND_QUANTIZED_TENSOR_CORE)
            {
                SparkResidentDecodeStageReportValidationFailure(
                    node_context,
                    "fast_dense_mlp_mode",
                    SPARK_STATUS_INVALID_ARGUMENT);
                return SPARK_STATUS_INVALID_ARGUMENT;
            }
            status = SparkValidateGlm52ResidentDecodeStageRequiredDenseMlpPlans(
                node_context);
            if (status != SPARK_STATUS_OK)
            {
                SparkResidentDecodeStageReportValidationFailure(
                    node_context,
                    "fast_dense_mlp_plans",
                    status);
                return status;
            }
        }
        if (SparkGlm52ResidentDecodeStageB12xMoeDispatchPlanIsRequiredForLayer(
                node_context) &&
            node_context->mlp_execution_mode !=
                SPARK_RESIDENT_DECODE_STAGE_MLP_EXECUTION_FLASHINFER_B12X_MOE)
        {
            SparkResidentDecodeStageReportValidationFailure(
                node_context,
                "fast_b12x_moe_mode",
                SPARK_STATUS_INVALID_ARGUMENT);
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        if (SparkGlm52ResidentDecodeStageB12xMoeDispatchPlanIsRequiredForLayer(
                node_context) &&
            !SparkGlm52ResidentDecodeStageB12xMoeDispatchPlanIsUsable(node_context))
        {
            SparkResidentDecodeStageReportValidationFailure(
                node_context,
                "fast_b12x_moe_plan",
                SPARK_STATUS_INVALID_ARGUMENT);
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        if (node_context->layer_progression_mode ==
                SPARK_RESIDENT_DECODE_STAGE_LAYER_ROUTED_FP8_TOPK &&
            !SparkGlm52ResidentDecodeStageFp8MoePlanIsUsable(node_context))
        {
            SparkResidentDecodeStageReportValidationFailure(
                node_context,
                "fast_fp8_moe_plan",
                SPARK_STATUS_INVALID_ARGUMENT);
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        }
    if (SparkResidentDecodeStageExecutionFlagIsSet(
            node_context,
            SPARK_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_FAST_MOE_ROUTER) &&
        !SparkGlm52ResidentDecodeStageRouterLinearPlanIsProductionFast(node_context))
    {
        SparkResidentDecodeStageReportValidationFailure(
            node_context,
            "fast_router_plan",
            SPARK_STATUS_INVALID_ARGUMENT);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkResidentDecodeStageExecutionFlagIsSet(
            node_context,
            SPARK_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_FAST_RESTRICTED_LOGITS) &&
        node_context->restricted_logits_plan == 0 &&
        !SparkGlm52ResidentDecodeStageLinearPlanIsUsable(
            node_context,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_RESTRICTED_LOGITS,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
            SPARK_RESIDENT_DECODE_STAGE_RESTRICTED_VOCAB_COUNT))
    {
        SparkResidentDecodeStageReportValidationFailure(
            node_context,
            "fast_restricted_logits",
            SPARK_STATUS_INVALID_ARGUMENT);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkResidentDecodeStageMtpDraftRequired(node_context) &&
        !SparkResidentDecodeStageMtpDraftPlanIsUsable(
            node_context->mtp_draft_plan))
    {
        SparkResidentDecodeStageReportValidationFailure(
            node_context,
            "fast_mtp_plan",
            SPARK_STATUS_INVALID_ARGUMENT);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkResidentDecodeStageExecutionFlagIsSet(
            node_context,
            SPARK_RESIDENT_DECODE_STAGE_EXECUTION_FORBID_DEBUG_SYNCHRONIZATION) &&
        (node_context->launch_check_mode !=
             SPARK_RESIDENT_DECODE_STAGE_LAUNCH_CHECK_NONE ||
         node_context->phase_clock_mode !=
             SPARK_RESIDENT_DECODE_STAGE_PHASE_CLOCK_DISABLED))
    {
        SparkResidentDecodeStageReportValidationFailure(
            node_context,
            "fast_debug_sync",
            SPARK_STATUS_INVALID_ARGUMENT);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkResidentDecodeStageExecutionFlagIsSet(
            node_context,
            SPARK_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_BULK_PREFILL) &&
        !SparkResidentDecodeStageBulkPrefillPlanIsUsable(node_context))
    {
        SparkResidentDecodeStageReportValidationFailure(
            node_context,
            "fast_bulk_prefill",
            SPARK_STATUS_INVALID_ARGUMENT);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkResidentDecodeStageExecutionFlagIsSet(
            node_context,
            SPARK_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_VALIDATED_LATENCY) &&
        node_context->validated_stage_latency_ns == 0u)
    {
        SparkResidentDecodeStageReportValidationFailure(
            node_context,
            "fast_latency",
            SPARK_STATUS_INVALID_ARGUMENT);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkValidateGlm52ResidentDecodeStagePipelineSlot(
    const SparkResidentDecodeStagePipelineSlot *pipeline_slot)
{
    if (pipeline_slot == 0 || pipeline_slot->cuda_stream == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (!SparkResidentDecodeStagePointerIsAligned(
            pipeline_slot->input_hidden_bf16,
            2u) ||
        !SparkResidentDecodeStagePointerIsAligned(
            pipeline_slot->normalized_hidden_bf16,
            2u) ||
        !SparkResidentDecodeStagePointerIsAligned(
            pipeline_slot->query_latent_bf16,
            2u) ||
        !SparkResidentDecodeStagePointerIsAligned(
            pipeline_slot->query_rope_input_bf16,
            2u) ||
        !SparkResidentDecodeStagePointerIsAligned(
            pipeline_slot->key_rope_input_bf16,
            2u) ||
        !SparkResidentDecodeStagePointerIsAligned(
            pipeline_slot->current_kv_latent_bf16,
            2u) ||
        !SparkResidentDecodeStagePointerIsAligned(
            pipeline_slot->positions,
            4u) ||
        !SparkResidentDecodeStagePointerIsAligned(
            pipeline_slot->slot_mapping,
            4u) ||
        !SparkResidentDecodeStagePointerIsAligned(
            pipeline_slot->block_table,
            4u) ||
        !SparkResidentDecodeStagePointerIsAligned(
            pipeline_slot->context_lengths,
            4u) ||
        !SparkResidentDecodeStagePointerIsAligned(
            pipeline_slot->first_block_token_offsets,
            4u) ||
        !SparkResidentDecodeStagePointerIsAligned(
            pipeline_slot->sparse_token_indices,
            4u) ||
        !SparkResidentDecodeStagePointerIsAligned(
            pipeline_slot->rotated_query_rope_bf16,
            2u) ||
        !SparkResidentDecodeStagePointerIsAligned(
            pipeline_slot->attention_output_latent_bf16,
            2u) ||
        !SparkResidentDecodeStagePointerIsAligned(
            pipeline_slot->attention_projected_hidden_bf16,
            2u) ||
        !SparkResidentDecodeStagePointerIsAligned(
            pipeline_slot->post_attention_hidden_bf16,
            2u) ||
        !SparkResidentDecodeStagePointerIsAligned(
            pipeline_slot->mtp_draft_hidden_bf16,
            2u) ||
        !SparkResidentDecodeStagePointerIsAligned(
            pipeline_slot->restricted_logits,
            4u) ||
        !SparkResidentDecodeStagePointerIsAligned(
            pipeline_slot->mtp_draft_logits,
            4u) ||
        !SparkResidentDecodeStagePointerIsAligned(
            pipeline_slot->restricted_selected_token_ids,
            4u) ||
        !SparkResidentDecodeStagePointerIsAligned(
            pipeline_slot->restricted_selected_token_scores,
            4u) ||
        !SparkResidentDecodeStagePointerIsAligned(
            pipeline_slot->mtp_draft_token_ids,
            4u) ||
        !SparkResidentDecodeStagePointerIsAligned(
            pipeline_slot->mtp_draft_token_budgets,
            4u) ||
        !SparkResidentDecodeStagePointerIsAligned(
            pipeline_slot->mtp_target_token_ids,
            4u) ||
        !SparkResidentDecodeStagePointerIsAligned(
            pipeline_slot->mtp_accept_mask,
            4u) ||
        !SparkResidentDecodeStagePointerIsAligned(
            pipeline_slot->mtp_committed_token_ids,
            4u) ||
        !SparkResidentDecodeStagePointerIsAligned(
            pipeline_slot->mtp_event_counters,
            4u) ||
        !SparkResidentDecodeStagePointerIsAligned(
            pipeline_slot->phase_clock_cycles,
            8u))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkValidateGlm52ResidentDecodeStageRawPipelineSlot(
    const SparkResidentDecodeStagePipelineSlot *pipeline_slot)
{
    if (!SparkResidentDecodeStagePointerIsAligned(
            pipeline_slot->raw_query_a_bf16,
            2u) ||
        !SparkResidentDecodeStagePointerIsAligned(
            pipeline_slot->raw_query_a_normalized_bf16,
            2u) ||
        !SparkResidentDecodeStagePointerIsAligned(
            pipeline_slot->raw_query_b_bf16,
            2u) ||
        !SparkResidentDecodeStagePointerIsAligned(
            pipeline_slot->raw_kv_a_bf16,
            2u) ||
        !SparkResidentDecodeStagePointerIsAligned(
            pipeline_slot->raw_kv_a_normalized_bf16,
            2u) ||
        !SparkResidentDecodeStagePointerIsAligned(
            pipeline_slot->raw_kv_b_bf16,
            2u))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkValidateGlm52ResidentDecodeStageMoePipelineSlot(
    const SparkResidentDecodeStagePipelineSlot *pipeline_slot)
{
    if (!SparkResidentDecodeStagePointerIsAligned(
            pipeline_slot->post_attention_normalized_hidden_bf16,
            2u) ||
        !SparkResidentDecodeStagePointerIsAligned(
            pipeline_slot->moe_topk_expert_ids,
            4u) ||
        !SparkResidentDecodeStagePointerIsAligned(
            pipeline_slot->moe_topk_weights,
            4u) ||
        (pipeline_slot->moe_router_logits != 0 &&
         !SparkResidentDecodeStagePointerIsAligned(
            pipeline_slot->moe_router_logits,
            4u)) ||
        !SparkResidentDecodeStagePointerIsAligned(
            pipeline_slot->moe_gate_bf16,
            2u) ||
        !SparkResidentDecodeStagePointerIsAligned(
            pipeline_slot->moe_up_bf16,
            2u) ||
        !SparkResidentDecodeStagePointerIsAligned(
            pipeline_slot->moe_intermediate_bf16,
            2u) ||
        !SparkResidentDecodeStagePointerIsAligned(
            pipeline_slot->moe_route_output_bf16,
            2u) ||
        !SparkResidentDecodeStagePointerIsAligned(
            pipeline_slot->layer_output_hidden_bf16,
            2u))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkValidateGlm52ResidentDecodeStageProjectionPointers(
    const SparkResidentDecodeStageNodeContext *node_context)
{
    if (node_context->projection_mode ==
        SPARK_RESIDENT_DECODE_STAGE_PROJECTION_LOWERED_BF16)
    {
        if (!SparkResidentDecodeStagePointerIsAligned(
                node_context->query_latent_weight_bf16,
                2u) ||
            !SparkResidentDecodeStagePointerIsAligned(
                node_context->query_rope_weight_bf16,
                2u) ||
            !SparkResidentDecodeStagePointerIsAligned(
                node_context->key_rope_weight_bf16,
                2u) ||
            !SparkResidentDecodeStagePointerIsAligned(
                node_context->kv_latent_weight_bf16,
                2u) ||
            !SparkResidentDecodeStagePointerIsAligned(
                node_context->attention_output_weight_bf16,
                2u))
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        return SPARK_STATUS_OK;
    }
    if (node_context->projection_mode ==
        SPARK_RESIDENT_DECODE_STAGE_PROJECTION_RAW_BF16)
    {
        if (!SparkResidentDecodeStagePointerIsAligned(
                node_context->raw_query_a_weight_bf16,
                2u) ||
            !SparkResidentDecodeStagePointerIsAligned(
                node_context->raw_query_a_norm_weight_bf16,
                2u) ||
            !SparkResidentDecodeStagePointerIsAligned(
                node_context->raw_query_b_weight_bf16,
                2u) ||
            !SparkResidentDecodeStagePointerIsAligned(
                node_context->raw_kv_a_weight_bf16,
                2u) ||
            !SparkResidentDecodeStagePointerIsAligned(
                node_context->raw_kv_a_norm_weight_bf16,
                2u) ||
            !SparkResidentDecodeStagePointerIsAligned(
                node_context->raw_kv_b_weight_bf16,
                2u) ||
            !SparkResidentDecodeStagePointerIsAligned(
                node_context->attention_output_weight_bf16,
                2u))
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        return SPARK_STATUS_OK;
    }
    if (node_context->projection_mode ==
        SPARK_RESIDENT_DECODE_STAGE_PROJECTION_RAW_FP8_E4M3)
    {
        if (!SparkResidentDecodeStagePointerIsAligned(
                node_context->raw_query_a_norm_weight_bf16,
                2u) ||
            !SparkResidentDecodeStagePointerIsAligned(
                node_context->raw_kv_a_norm_weight_bf16,
                2u))
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        if (node_context->projection_backend_mode ==
                SPARK_RESIDENT_DECODE_STAGE_PROJECTION_BACKEND_PREBOUND_TENSOR_CORE &&
            SparkValidateGlm52ResidentDecodeStageRequiredProjectionAndOutputPlans(
                node_context) == SPARK_STATUS_OK)
        {
            return SPARK_STATUS_OK;
        }
        if (!SparkResidentDecodeStagePointerIsAligned(
                node_context->raw_query_a_weight_fp8_e4m3,
                1u) ||
            !SparkResidentDecodeStagePointerIsAligned(
                node_context->raw_query_a_weight_scale_inv_f32,
                4u) ||
            !SparkResidentDecodeStagePointerIsAligned(
                node_context->raw_query_b_weight_fp8_e4m3,
                1u) ||
            !SparkResidentDecodeStagePointerIsAligned(
                node_context->raw_query_b_weight_scale_inv_f32,
                4u) ||
            !SparkResidentDecodeStagePointerIsAligned(
                node_context->raw_kv_a_weight_fp8_e4m3,
                1u) ||
            !SparkResidentDecodeStagePointerIsAligned(
                node_context->raw_kv_a_weight_scale_inv_f32,
                4u) ||
            !SparkResidentDecodeStagePointerIsAligned(
                node_context->raw_kv_b_weight_fp8_e4m3,
                1u) ||
            !SparkResidentDecodeStagePointerIsAligned(
                node_context->raw_kv_b_weight_scale_inv_f32,
                4u) ||
            !SparkResidentDecodeStagePointerIsAligned(
                node_context->attention_output_weight_fp8_e4m3,
                1u) ||
            !SparkResidentDecodeStagePointerIsAligned(
                node_context->attention_output_weight_scale_inv_f32,
                4u))
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        return SPARK_STATUS_OK;
    }
    if (SparkResidentDecodeStageProjectionModeUsesQuantizedPlan(
            node_context))
    {
        if (node_context->projection_backend_mode !=
                SPARK_RESIDENT_DECODE_STAGE_PROJECTION_BACKEND_PREBOUND_TENSOR_CORE ||
            !SparkResidentDecodeStagePointerIsAligned(
                node_context->raw_query_a_norm_weight_bf16,
                2u) ||
            !SparkResidentDecodeStagePointerIsAligned(
                node_context->raw_kv_a_norm_weight_bf16,
                2u))
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        return SparkValidateGlm52ResidentDecodeStageRequiredProjectionAndOutputPlans(
            node_context);
    }
    return SPARK_STATUS_INVALID_ARGUMENT;
}

static SparkStatus SparkValidateGlm52ResidentDecodeStageLayerPointers(
    const SparkResidentDecodeStageNodeContext *node_context)
{
    if (node_context->layer_progression_mode ==
        SPARK_RESIDENT_DECODE_STAGE_LAYER_ATTENTION_ONLY)
    {
        return SPARK_STATUS_OK;
    }
    if (node_context->layer_progression_mode ==
        SPARK_RESIDENT_DECODE_STAGE_LAYER_DENSE_BF16_MLP)
    {
        if (node_context->dense_intermediate_dimension == 0u ||
            node_context->dense_intermediate_dimension >
                SPARK_GLM52_RESIDENT_DECODE_STAGE_DENSE_INTERMEDIATE_DIMENSION ||
            !SparkResidentDecodeStagePointerIsAligned(
                node_context->post_attention_norm_weight_bf16,
                2u))
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        if (SparkResidentDecodeStageMlpExecutionUsesQuantizedPlan(
                node_context))
        {
            return SparkValidateGlm52ResidentDecodeStageRequiredDenseMlpPlans(
                node_context);
        }
        if (!SparkResidentDecodeStagePointerIsAligned(
                node_context->dense_gate_weight_bf16,
                2u) ||
            !SparkResidentDecodeStagePointerIsAligned(
                node_context->dense_up_weight_bf16,
                2u) ||
            !SparkResidentDecodeStagePointerIsAligned(
                node_context->dense_down_weight_bf16,
                2u))
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        return SPARK_STATUS_OK;
    }
    if (node_context->layer_progression_mode ==
        SPARK_RESIDENT_DECODE_STAGE_LAYER_ROUTER_BF16_TOPK_ONLY)
    {
        if (node_context->moe_expert_count == 0u ||
            node_context->moe_expert_count >
                SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT ||
            node_context->moe_top_k !=
                SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_TOP_K ||
            !SparkResidentDecodeStagePointerIsAligned(
                node_context->post_attention_norm_weight_bf16,
                2u) ||
            !SparkGlm52ResidentDecodeStageRouterWeightOrPlanIsUsable(
                node_context) ||
            !SparkResidentDecodeStagePointerIsAligned(
                node_context->moe_router_score_bias_f32,
                4u))
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        return SPARK_STATUS_OK;
    }
    if (node_context->layer_progression_mode ==
        SPARK_RESIDENT_DECODE_STAGE_LAYER_ROUTED_NVFP4_TOPK)
    {
        if (node_context->moe_expert_count !=
                SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT ||
            node_context->moe_top_k !=
                SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_TOP_K ||
            node_context->moe_intermediate_dimension !=
                SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION ||
            node_context->mlp_execution_mode !=
                SPARK_RESIDENT_DECODE_STAGE_MLP_EXECUTION_FLASHINFER_B12X_MOE ||
            !SparkResidentDecodeStagePointerIsAligned(
                node_context->post_attention_norm_weight_bf16,
                2u) ||
            !SparkGlm52ResidentDecodeStageRouterWeightOrPlanIsUsable(
                node_context) ||
            !SparkResidentDecodeStagePointerIsAligned(
                node_context->moe_router_score_bias_f32,
                4u) ||
            !SparkGlm52ResidentDecodeStageB12xMoeDispatchPlanIsUsable(node_context))
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        return SPARK_STATUS_OK;
    }
    if (node_context->layer_progression_mode ==
        SPARK_RESIDENT_DECODE_STAGE_LAYER_ROUTED_FP8_TOPK)
    {
        if (node_context->moe_expert_count !=
                SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT ||
            node_context->moe_top_k !=
                SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_TOP_K ||
            node_context->moe_intermediate_dimension !=
                SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION ||
            node_context->dense_intermediate_dimension !=
                SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION ||
            node_context->mlp_execution_mode !=
                SPARK_GLM52_RESIDENT_DECODE_STAGE_MLP_EXECUTION_FP8_EXPERT_TENSOR_CORE ||
            node_context->projection_mode !=
                SPARK_RESIDENT_DECODE_STAGE_PROJECTION_RAW_FP8_E4M3 ||
            !SparkResidentDecodeStagePointerIsAligned(
                node_context->post_attention_norm_weight_bf16,
                2u) ||
            !SparkGlm52ResidentDecodeStageRouterWeightOrPlanIsUsable(
                node_context) ||
            !SparkResidentDecodeStagePointerIsAligned(
                node_context->moe_router_score_bias_f32,
                4u) ||
            !SparkGlm52ResidentDecodeStageFp8MoePlanIsUsable(node_context))
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        return SparkValidateGlm52ResidentDecodeStageRequiredDenseMlpPlans(
            node_context);
    }
    return SPARK_STATUS_INVALID_ARGUMENT;
}

SparkStatus SparkResidentDecodeStageModelValidateNodeContext(
    const SparkResidentDecodeStageNodeContext *node_context)
{
    uint64_t represented_token_capacity;
    uint32_t storage_token_capacity;
    uint32_t pipeline_slot_index;
    uint32_t hidden_output_only;

    if (node_context == 0 ||
        node_context->abi_version !=
            SPARK_RESIDENT_DECODE_STAGE_NODE_CONTEXT_ABI_VERSION)
    {
        SparkResidentDecodeStageReportValidationFailure(
            node_context,
            "abi",
            SPARK_STATUS_ABI_MISMATCH);
        return SPARK_STATUS_ABI_MISMATCH;
    }
    hidden_output_only =
        (node_context->reserved_execution_flags &
         SPARK_RESIDENT_DECODE_STAGE_EXECUTION_OUTPUT_HIDDEN_ONLY) != 0u;
    storage_token_capacity = node_context->kv_storage_token_capacity != 0u
        ? node_context->kv_storage_token_capacity
        : node_context->cache_token_capacity;
    if (node_context->pipeline_slot_count == 0u ||
        node_context->pipeline_slot_count >
            SPARK_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT ||
        node_context->max_active_sequence_count == 0u ||
        node_context->cache_token_capacity == 0u ||
        node_context->kv_block_count == 0u ||
        node_context->max_blocks_per_sequence == 0u ||
        node_context->position_count == 0u ||
        node_context->dsa_candidate_capacity == 0u ||
        node_context->dsa_score_row_capacity == 0u ||
        !SparkResidentDecodeStagePointerIsAligned(
            node_context->dsa_score_tiles_f32,
            4u) ||
        node_context->projection_mode >
            SPARK_RESIDENT_DECODE_STAGE_PROJECTION_RAW_MXFP4_E2M1 ||
        node_context->layer_progression_mode >
            SPARK_RESIDENT_DECODE_STAGE_LAYER_ROUTED_FP8_TOPK ||
        node_context->sparse_index_mode >
            SPARK_RESIDENT_DECODE_STAGE_SPARSE_INDEX_DSA_INDEXSHARE_SHARED ||
        node_context->launch_check_mode >
            SPARK_RESIDENT_DECODE_STAGE_LAUNCH_CHECK_SYNC_ON_ERROR ||
        node_context->phase_clock_mode >
            SPARK_RESIDENT_DECODE_STAGE_PHASE_CLOCK_DEVICE_CLOCK64 ||
        node_context->projection_backend_mode >
            SPARK_RESIDENT_DECODE_STAGE_PROJECTION_BACKEND_PREBOUND_TENSOR_CORE ||
        node_context->mlp_execution_mode >
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MLP_EXECUTION_FP8_EXPERT_TENSOR_CORE ||
        node_context->attention_execution_mode >
            SPARK_RESIDENT_DECODE_STAGE_ATTENTION_EXECUTION_ABSORBED_LATENT ||
        !SparkResidentDecodeStageModelQuantizationModeIsSupported(
            node_context->model_quantization_mode) ||
        node_context->reserved1 != 0u ||
        SparkResidentDecodeStageEffectiveKvBlockTokenCount(node_context) == 0u ||
        SparkResidentDecodeStageEffectiveKvBlockTokenCount(node_context) >
            SPARK_KV_CACHE_MAX_BLOCK_TOKENS ||
        !SparkResidentDecodeStageLayerMatchesModelQuantization(
            node_context) ||
        ((node_context->sparse_index_mode ==
            SPARK_RESIDENT_DECODE_STAGE_SPARSE_INDEX_DSA_INDEXSHARE_FULL ||
          node_context->sparse_index_mode ==
            SPARK_RESIDENT_DECODE_STAGE_SPARSE_INDEX_DSA_INDEXSHARE_SHARED) &&
         (node_context->selected_token_indices_by_layer == 0 ||
          node_context->dsa_indexshare_selected_token_count !=
            SPARK_RESIDENT_DECODE_STAGE_SELECTED_TOKEN_COUNT ||
          node_context->dsa_indexshare_layer_count == 0u ||
          node_context->layer_index >=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_COUNT ||
          node_context->dsa_indexshare_source_layer_index >=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_COUNT ||
          node_context->layer_index <
            node_context->dsa_cache_first_layer_index ||
          node_context->dsa_indexshare_source_layer_index <
            node_context->dsa_cache_first_layer_index ||
          node_context->layer_index -
                node_context->dsa_cache_first_layer_index >=
            node_context->dsa_indexshare_layer_count ||
          node_context->dsa_indexshare_source_layer_index -
                node_context->dsa_cache_first_layer_index >=
            node_context->dsa_indexshare_layer_count ||
          node_context->dsa_indexshare_group_end_layer_exclusive >
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_COUNT ||
          node_context->dsa_indexshare_source_layer_index >=
            node_context->dsa_indexshare_group_end_layer_exclusive)) ||
        (node_context->sparse_index_mode ==
            SPARK_RESIDENT_DECODE_STAGE_SPARSE_INDEX_DSA_INDEXSHARE_FULL &&
         (node_context->dsa_index_head_count !=
            SPARK_RESIDENT_DECODE_STAGE_DSA_INDEX_HEAD_COUNT ||
          node_context->dsa_index_head_dimension !=
            SPARK_RESIDENT_DECODE_STAGE_DSA_INDEX_HEAD_DIMENSION ||
          ((node_context->index_query_weight_fp8_e4m3 == 0) !=
              (node_context->index_query_weight_scale_inv_f32 == 0)) ||
          ((node_context->index_key_weight_fp8_e4m3 == 0) !=
              (node_context->index_key_weight_scale_inv_f32 == 0)) ||
          (node_context->index_query_weight_bf16 == 0 &&
              node_context->index_query_weight_fp8_e4m3 == 0) ||
          (node_context->index_key_weight_bf16 == 0 &&
              node_context->index_key_weight_fp8_e4m3 == 0) ||
          node_context->index_weights_proj_weight_bf16 == 0 ||
          node_context->index_key_norm_weight_bf16 == 0 ||
          node_context->index_key_norm_bias_bf16 == 0 ||
          node_context->key_index_cache_bf16 == 0 ||
          !isfinite(node_context->index_softmax_scale) ||
          node_context->index_softmax_scale <= 0.0f)) ||
        !isfinite(node_context->qk_scale) ||
        node_context->qk_scale <= 0.0f ||
        !isfinite(node_context->rms_norm_epsilon) ||
        node_context->rms_norm_epsilon <= 0.0f ||
        !SparkResidentDecodeStagePointerIsAligned(
            node_context->cos_table,
            4u) ||
        !SparkResidentDecodeStagePointerIsAligned(
            node_context->sin_table,
            4u) ||
        (SparkResidentDecodeStageExecutionFlagIsSet(
             node_context,
             SPARK_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_FP8_KV_CACHE)
            ? node_context->mla_cache_bf16 != 0 ||
                node_context->key_nope_cache_bf16 != 0 ||
                node_context->value_cache_bf16 != 0
            : !SparkResidentDecodeStagePointerIsAligned(
                  node_context->mla_cache_bf16,
                  4u) ||
                (node_context->attention_execution_mode !=
                    SPARK_RESIDENT_DECODE_STAGE_ATTENTION_EXECUTION_ABSORBED_LATENT &&
                 (!SparkResidentDecodeStagePointerIsAligned(
                      node_context->key_nope_cache_bf16,
                      4u) ||
                  !SparkResidentDecodeStagePointerIsAligned(
                      node_context->value_cache_bf16,
                      4u)))) ||
        !SparkResidentDecodeStagePointerIsAligned(
            node_context->attention_norm_weight_bf16,
            2u) ||
        node_context->pipeline_slots == 0)
    {
        SparkResidentDecodeStageReportValidationFailure(
            node_context,
            "base",
            SPARK_STATUS_INVALID_ARGUMENT);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (storage_token_capacity < node_context->cache_token_capacity)
    {
        SparkResidentDecodeStageReportValidationFailure(
            node_context,
            "kv_storage_capacity",
            SPARK_STATUS_INVALID_ARGUMENT);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkResidentDecodeStageEffectiveModelQuantizationMode(
            node_context) ==
            SPARK_RESIDENT_DECODE_STAGE_MODEL_QUANTIZATION_FP8_E4M3_8BIT &&
        node_context->projection_mode !=
            SPARK_RESIDENT_DECODE_STAGE_PROJECTION_RAW_FP8_E4M3)
    {
        SparkResidentDecodeStageReportValidationFailure(
            node_context,
            "fp8_projection_mode",
            SPARK_STATUS_INVALID_ARGUMENT);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if ((node_context->mtp_draft_plan != 0 &&
            !SparkResidentDecodeStageMtpDraftPlanIsUsable(
                node_context->mtp_draft_plan)) ||
        (SparkResidentDecodeStageMtpDraftRequired(node_context) &&
            !SparkResidentDecodeStageMtpDraftPlanIsUsable(
                node_context->mtp_draft_plan)))
    {
        SparkResidentDecodeStageReportValidationFailure(
            node_context,
            "mtp_draft_plan",
            SPARK_STATUS_INVALID_ARGUMENT);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if (hidden_output_only == 0u &&
        (!SparkResidentDecodeStagePointerIsAligned(
             node_context->final_norm_weight_bf16,
             2u) ||
         !SparkResidentDecodeStagePointerIsAligned(
             node_context->restricted_lm_head_weight_bf16,
             2u) ||
         !SparkResidentDecodeStagePointerIsAligned(
             node_context->restricted_token_ids,
             4u)))
    {
        SparkResidentDecodeStageReportValidationFailure(
            node_context,
            "final_output_pointers",
            SPARK_STATUS_INVALID_ARGUMENT);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (hidden_output_only == 0u &&
        SparkResidentDecodeStageMtpDraftRequired(node_context) &&
        (!SparkResidentDecodeStagePointerIsAligned(
             node_context->mtp_mxfp4_weight_payload_u8,
             1u) ||
         !SparkResidentDecodeStagePointerIsAligned(
             node_context->mtp_mxfp4_scale_e8m0_u8,
             1u)))
    {
        SparkResidentDecodeStageReportValidationFailure(
            node_context,
            "mtp_weight_pointers",
            SPARK_STATUS_INVALID_ARGUMENT);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if (SparkResidentDecodeStageProjectionBackendIsPrebound(
            node_context) &&
        (node_context->linear_plans == 0 ||
         node_context->linear_plan_count <
             SPARK_RESIDENT_DECODE_STAGE_LINEAR_PLAN_COUNT))
    {
        SparkResidentDecodeStageReportValidationFailure(
            node_context,
            "linear_plan_table",
            SPARK_STATUS_INVALID_ARGUMENT);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (node_context->mlp_execution_mode ==
            SPARK_RESIDENT_DECODE_STAGE_MLP_EXECUTION_FLASHINFER_B12X_MOE &&
        !SparkGlm52ResidentDecodeStageB12xMoeDispatchPlanIsUsable(node_context))
    {
        SparkResidentDecodeStageReportValidationFailure(
            node_context,
            "b12x_moe_plan",
            SPARK_STATUS_INVALID_ARGUMENT);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (node_context->mlp_execution_mode ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MLP_EXECUTION_FP8_EXPERT_TENSOR_CORE &&
        !SparkGlm52ResidentDecodeStageFp8MoePlanIsUsable(node_context))
    {
        SparkResidentDecodeStageReportValidationFailure(
            node_context,
            "fp8_moe_plan",
            SPARK_STATUS_INVALID_ARGUMENT);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkResidentDecodeStageExecutionFlagIsSet(
            node_context,
            SPARK_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_MODEL_QUANTIZATION) &&
        node_context->model_quantization_mode ==
            SPARK_RESIDENT_DECODE_STAGE_MODEL_QUANTIZATION_AUTO)
    {
        SparkResidentDecodeStageReportValidationFailure(
            node_context,
            "model_quantization",
            SPARK_STATUS_INVALID_ARGUMENT);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkResidentDecodeStageExecutionFlagIsSet(
            node_context,
            SPARK_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_FP8_KV_CACHE) &&
        (!SparkResidentDecodeStageFp8KvCachePlanIsUsable(node_context) ||
         (node_context->attention_execution_mode ==
              SPARK_RESIDENT_DECODE_STAGE_ATTENTION_EXECUTION_ABSORBED_LATENT &&
          !SparkResidentDecodeStageUsesCompressedFp8Mla(node_context)) ||
         (node_context->attention_execution_mode ==
              SPARK_RESIDENT_DECODE_STAGE_ATTENTION_EXECUTION_TILED_ONLINE_SOFTMAX &&
          SparkResidentDecodeStageUsesCompressedFp8Mla(node_context))))
    {
        SparkResidentDecodeStageReportValidationFailure(
            node_context,
            "fp8_kv_cache_plan",
            SPARK_STATUS_INVALID_ARGUMENT);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkValidateGlm52ResidentDecodeStageFastPathContract(
            node_context) != SPARK_STATUS_OK)
    {
        SparkResidentDecodeStageReportValidationFailure(
            node_context,
            "fast_path_contract",
            SPARK_STATUS_INVALID_ARGUMENT);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkValidateGlm52ResidentDecodeStageProjectionPointers(
            node_context) != SPARK_STATUS_OK)
    {
        SparkResidentDecodeStageReportValidationFailure(
            node_context,
            "projection_pointers",
            SPARK_STATUS_INVALID_ARGUMENT);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkValidateGlm52ResidentDecodeStageLayerPointers(
            node_context) != SPARK_STATUS_OK)
    {
        SparkResidentDecodeStageReportValidationFailure(
            node_context,
            "layer_pointers",
            SPARK_STATUS_INVALID_ARGUMENT);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    represented_token_capacity =
        (uint64_t)node_context->kv_block_count *
        (uint64_t)SparkResidentDecodeStageEffectiveKvBlockTokenCount(
            node_context);
    if ((uint64_t)node_context->cache_token_capacity >
            represented_token_capacity)
    {
        SparkResidentDecodeStageReportValidationFailure(
            node_context,
            "kv_capacity",
            SPARK_STATUS_INVALID_ARGUMENT);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if (node_context->enable_cuda_graph_replay != 0u)
    {
        if (node_context->cuda_pipeline_slot_states == 0)
        {
            SparkResidentDecodeStageReportValidationFailure(
                node_context,
                "cuda_slot_states",
                SPARK_STATUS_INVALID_ARGUMENT);
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
    }

    for (pipeline_slot_index = 0u;
         pipeline_slot_index < node_context->pipeline_slot_count;
         ++pipeline_slot_index)
    {
        const SparkResidentDecodeStagePipelineSlot *pipeline_slot;
        SparkStatus status;

        pipeline_slot = &node_context->pipeline_slots[pipeline_slot_index];
        if (pipeline_slot->dsa_candidate_count == 0u ||
            pipeline_slot->dsa_candidate_count >
                node_context->dsa_candidate_capacity)
        {
            SparkResidentDecodeStageReportValidationFailure(
                node_context,
                "dsa_candidate_count",
                SPARK_STATUS_INVALID_ARGUMENT);
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        status = SparkValidateGlm52ResidentDecodeStagePipelineSlot(
            pipeline_slot);
        if (status != SPARK_STATUS_OK)
        {
            SparkResidentDecodeStageReportValidationFailure(
                node_context,
                "pipeline_slot",
                status);
            return status;
        }
        if (node_context->sparse_index_mode ==
                SPARK_RESIDENT_DECODE_STAGE_SPARSE_INDEX_DSA_INDEXSHARE_FULL &&
            (!SparkResidentDecodeStagePointerIsAligned(
                 node_context->pipeline_slots[pipeline_slot_index].query_index_heads_bf16,
                 2u) ||
             !SparkResidentDecodeStagePointerIsAligned(
                 node_context->pipeline_slots[pipeline_slot_index].current_key_index_bf16,
                 2u) ||
             !SparkResidentDecodeStagePointerIsAligned(
                 node_context->pipeline_slots[pipeline_slot_index].index_head_weights_bf16,
                 2u)))
        {
            SparkResidentDecodeStageReportValidationFailure(
                node_context,
                "dsa_pipeline_slot",
                SPARK_STATUS_INVALID_ARGUMENT);
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        if (node_context->projection_mode !=
                SPARK_RESIDENT_DECODE_STAGE_PROJECTION_LOWERED_BF16 &&
            SparkValidateGlm52ResidentDecodeStageRawPipelineSlot(
                &node_context->pipeline_slots[pipeline_slot_index]) !=
                SPARK_STATUS_OK)
        {
            SparkResidentDecodeStageReportValidationFailure(
                node_context,
                "raw_pipeline_slot",
                SPARK_STATUS_INVALID_ARGUMENT);
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        if (node_context->layer_progression_mode !=
                SPARK_RESIDENT_DECODE_STAGE_LAYER_ATTENTION_ONLY &&
            SparkValidateGlm52ResidentDecodeStageMoePipelineSlot(
                &node_context->pipeline_slots[pipeline_slot_index]) !=
                SPARK_STATUS_OK)
        {
            SparkResidentDecodeStageReportValidationFailure(
                node_context,
                "moe_pipeline_slot",
                SPARK_STATUS_INVALID_ARGUMENT);
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        if (SparkResidentDecodeStageRequiresNvfp4RouteSlotCache(
                node_context) &&
            !SparkResidentDecodeStagePointerIsAligned(
                node_context->pipeline_slots[pipeline_slot_index].moe_bound_expert_slots,
                4u))
        {
            SparkResidentDecodeStageReportValidationFailure(
                node_context,
                "route_slot_cache",
                SPARK_STATUS_INVALID_ARGUMENT);
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        if (SparkResidentDecodeStageExecutionFlagIsSet(
                node_context,
                SPARK_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_FAST_MOE_ROUTER) &&
            !SparkResidentDecodeStagePointerIsAligned(
                node_context->pipeline_slots[pipeline_slot_index].moe_router_logits,
                4u))
        {
            SparkResidentDecodeStageReportValidationFailure(
                node_context,
                "router_logits_slot",
                SPARK_STATUS_INVALID_ARGUMENT);
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        if (node_context->cuda_pipeline_slot_states != 0 &&
            node_context->cuda_pipeline_slot_states[pipeline_slot_index].abi_version !=
                SPARK_RESIDENT_DECODE_STAGE_CUDA_SLOT_STATE_ABI_VERSION)
        {
            SparkResidentDecodeStageReportValidationFailure(
                node_context,
                "cuda_slot_state_abi",
                SPARK_STATUS_ABI_MISMATCH);
            return SPARK_STATUS_ABI_MISMATCH;
        }
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkResidentDecodeStageModelValidateSliceNodeContext(
    const SparkResidentDecodeStageSliceNodeContext *slice_node_context,
    const SparkResidentDecodeStageNodeContext **first_node_context)
{
    const SparkResidentDecodeStageNodeContext *reference_node_context;
    const SparkResidentDecodeStageNodeContext *layer_node_context;
    uint32_t layer_index;
    bool stage_slice_plan_is_usable;
    bool requires_builtin_fused_stage_moe;
    SparkStatus status;

    if (slice_node_context == 0 || first_node_context == 0)
    {
        SparkResidentDecodeStageReportValidationFailure(
            0,
            "slice_null",
            SPARK_STATUS_INVALID_ARGUMENT);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *first_node_context = 0;
    if (slice_node_context->abi_version !=
            SPARK_RESIDENT_DECODE_STAGE_SLICE_NODE_CONTEXT_ABI_VERSION ||
        slice_node_context->descriptor_bytes !=
            SPARK_RESIDENT_DECODE_STAGE_SLICE_NODE_CONTEXT_DESCRIPTOR_BYTES ||
        !SparkResidentDecodeStageSliceLayerRangeIsUsable(
            slice_node_context->first_layer_index,
            slice_node_context->layer_count) ||
        slice_node_context->final_token_stage > 1u ||
        slice_node_context->reserved != 0u ||
        slice_node_context->layer_node_contexts == 0)
    {
        SparkResidentDecodeStageReportValidationFailure(
            0,
            "slice_base",
            SPARK_STATUS_INVALID_ARGUMENT);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    reference_node_context = slice_node_context->layer_node_contexts[0];
    if (reference_node_context == 0)
    {
        SparkResidentDecodeStageReportValidationFailure(
            0,
            "slice_reference",
            SPARK_STATUS_INVALID_ARGUMENT);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    stage_slice_plan_is_usable =
        SparkResidentDecodeStageStageSlicePlanIsUsable(
            slice_node_context->stage_slice_plan,
            reference_node_context->max_active_sequence_count,
            slice_node_context->layer_count,
            slice_node_context->first_layer_index,
            slice_node_context->final_token_stage);
    if (slice_node_context->stage_slice_plan != 0 &&
        !stage_slice_plan_is_usable)
    {
        SparkResidentDecodeStageReportValidationFailure(
            reference_node_context,
            "slice_plan",
            SPARK_STATUS_INVALID_ARGUMENT);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    requires_builtin_fused_stage_moe =
        SparkResidentDecodeStageStageSlicePlanRequiresBuiltInFusedStageMoe(
            slice_node_context->stage_slice_plan);
    for (layer_index = 0u;
         layer_index < slice_node_context->layer_count;
         ++layer_index)
    {
        layer_node_context = slice_node_context->layer_node_contexts[layer_index];
        status = SparkValidateGlm52ResidentDecodeStageNodeContext(
            layer_node_context);
        if (status != SPARK_STATUS_OK)
        {
            SparkResidentDecodeStageReportValidationFailure(
                layer_node_context,
                "slice_layer",
                status);
            return status;
        }
        if (requires_builtin_fused_stage_moe &&
            !SparkResidentDecodeStageLayerSupportsBuiltInFusedStageMoe(
                layer_node_context))
        {
            SparkResidentDecodeStageReportValidationFailure(
                layer_node_context,
                "slice_fused_moe",
                SPARK_STATUS_INVALID_ARGUMENT);
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        if (layer_node_context->pipeline_slot_count !=
                reference_node_context->pipeline_slot_count ||
            layer_node_context->max_active_sequence_count <
                reference_node_context->max_active_sequence_count ||
            layer_node_context->enable_cuda_graph_replay !=
                reference_node_context->enable_cuda_graph_replay)
        {
            SparkResidentDecodeStageReportValidationFailure(
                layer_node_context,
                "slice_layer_shape",
                SPARK_STATUS_INVALID_ARGUMENT);
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        if (!stage_slice_plan_is_usable &&
            (layer_node_context->full_stage_plan != 0 ||
             SparkResidentDecodeStageExecutionFlagIsSet(
                layer_node_context,
                SPARK_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_FULL_STAGE_PLAN)))
        {
            SparkResidentDecodeStageReportValidationFailure(
                layer_node_context,
                "slice_full_stage_plan",
                SPARK_STATUS_INVALID_ARGUMENT);
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        if (SparkResidentDecodeStageExecutionFlagIsSet(
                layer_node_context,
                SPARK_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_STAGE_SLICE_PLAN) &&
            !stage_slice_plan_is_usable)
        {
            SparkResidentDecodeStageReportValidationFailure(
                layer_node_context,
                "slice_required_plan",
                SPARK_STATUS_INVALID_ARGUMENT);
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
    }
    *first_node_context = reference_node_context;
    return SPARK_STATUS_OK;
}

SparkStatus SparkResidentDecodeStageModelValidateFrameTaps(const SparkResidentDecodeStageFrameContext *frame_context)
{
	const SparkGlm52DsparkHiddenTapPlan *plan;
	uint32_t tap_index,output_count;
	plan = (const SparkGlm52DsparkHiddenTapPlan *)frame_context->model_hidden_tap_plan;
	if ( frame_context->model_hidden_tap_lane_stride_bytes < SPARK_GLM52_MODEL_HIDDEN_BF16_BYTES || SparkResidentDecodeStageValidateDsparkHiddenTapPlanInline(plan) != SPARK_STATUS_OK )
		return SPARK_STATUS_INVALID_ARGUMENT;
	output_count = 0u;
	for (tap_index = 0u; tap_index < SPARK_GLM52_DSPARK_AUX_LAYER_COUNT; ++tap_index)
	{
		if ( frame_context->model_hidden_tap_output_bf16[tap_index] != 0 )
			output_count += 1u;
	}
	if ( output_count != 0u && output_count != SPARK_GLM52_DSPARK_AUX_LAYER_COUNT )
		return SPARK_STATUS_INVALID_ARGUMENT;
	return SPARK_STATUS_OK;
}
