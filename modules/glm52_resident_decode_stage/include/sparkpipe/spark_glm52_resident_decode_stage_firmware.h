#ifndef SPARKPIPE_SPARK_GLM52_RESIDENT_DECODE_STAGE_FIRMWARE_H
#define SPARKPIPE_SPARK_GLM52_RESIDENT_DECODE_STAGE_FIRMWARE_H

#include "sparkpipe/spark_resident_decode_stage.h"
#include "sparkpipe/spark_glm52_model.h"
#include <stdbool.h>

#define SPARK_RESIDENT_DECODE_STAGE_MODULE_ID \
    "spark.glm52.resident_decode_stage.bf16.h6144.h64.d512.r64.k2048.b1024.rv256.mtp6.v1"
#define SPARK_RESIDENT_DECODE_STAGE_TARGET \
    "cuda.sm121.glm52.resident_decode_stage.bf16"

#include "sparkpipe/spark_glm52_kv_cache.h"

// What remains after A4 part one: the glm DIMENSION TIER. The stage's node
// context, frame context, slots, flags, validation and completion machinery
// are include/sparkpipe/spark_resident_decode_stage.h now - 245 symbols
// whose definitions never mentioned a model dimension. These aliases are
// glm's numbers in the stage's vocabulary, and the quantized MoE weight
// plans stay in their own headers beside this one: a weight layout is
// model content.

#define SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION SPARK_GLM52_MODEL_HIDDEN_DIMENSION
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT SPARK_GLM52_MODEL_HEAD_COUNT
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION SPARK_GLM52_MODEL_LATENT_DIMENSION
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_ROPE_DIMENSION SPARK_GLM52_MODEL_ROPE_DIMENSION
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION SPARK_GLM52_MODEL_QUERY_A_DIMENSION
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_B_DIMENSION SPARK_GLM52_MODEL_QUERY_B_DIMENSION
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_KV_A_DIMENSION SPARK_GLM52_MODEL_KV_A_DIMENSION
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_KV_B_DIMENSION SPARK_GLM52_MODEL_KV_B_DIMENSION
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION SPARK_GLM52_MODEL_QK_NOPE_HEAD_DIMENSION
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_HEAD_DIMENSION SPARK_GLM52_MODEL_QK_HEAD_DIMENSION
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_VALUE_HEAD_DIMENSION SPARK_GLM52_MODEL_VALUE_HEAD_DIMENSION
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT SPARK_GLM52_MODEL_MOE_EXPERT_COUNT
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_TOP_K SPARK_GLM52_MODEL_MOE_TOP_K
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION SPARK_GLM52_MODEL_MOE_INTERMEDIATE_DIMENSION
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_DENSE_INTERMEDIATE_DIMENSION SPARK_GLM52_MODEL_DENSE_INTERMEDIATE_DIMENSION
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_BLOCK SPARK_GLM52_MODEL_FP8_SCALE_BLOCK
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_CACHE_TOKEN_ELEMENTS SPARK_GLM52_MODEL_CACHE_TOKEN_ELEMENTS
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_COUNT SPARK_GLM52_MODEL_LAYER_COUNT

// THE PLAN-USABILITY HELPERS. Shared by the validation tier (glm's own
// source file now) and the execution paths that check a plan before
// dispatching on it. They read glm's plan families, so they live with
// glm's plan headers, as static inline: two translation units, one
// definition, zero link surface.

static inline bool SparkGlm52ResidentDecodeStageB12xMoeDispatchPlanIsRequiredForLayer(
    const SparkResidentDecodeStageNodeContext *node_context)
{
    return node_context->layer_progression_mode ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_ROUTED_NVFP4_TOPK;
}

static inline bool SparkGlm52ResidentDecodeStageFp8MoePlanIsUsable(
    const SparkResidentDecodeStageNodeContext *node_context)
{
    const SparkGlm52ResidentDecodeStageFp8MoePlan *fp8_moe_plan;
    uint32_t required_capabilities;

    if (node_context == 0 || node_context->fp8_moe_plan == 0)
    {
        return false;
    }

    fp8_moe_plan = node_context->fp8_moe_plan;
    required_capabilities =
        SPARK_RESIDENT_DECODE_STAGE_FP8_MOE_REQUIRED_CAPABILITIES;
    if (fp8_moe_plan->abi_version !=
            SPARK_RESIDENT_DECODE_STAGE_FP8_MOE_PLAN_ABI_VERSION ||
        fp8_moe_plan->reserved0 != 0u ||
        fp8_moe_plan->reserved1 != 0u ||
        fp8_moe_plan->maximum_active_sequence_count <
            node_context->max_active_sequence_count ||
        fp8_moe_plan->maximum_token_count <
            node_context->max_active_sequence_count ||
        fp8_moe_plan->expert_count !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT ||
        fp8_moe_plan->top_k !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_TOP_K ||
        fp8_moe_plan->hidden_dimension !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION ||
        fp8_moe_plan->intermediate_dimension !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION ||
        fp8_moe_plan->output_dtype !=
            SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_OUTPUT_DTYPE_BF16 ||
        fp8_moe_plan->cuda_architecture != 121u ||
        fp8_moe_plan->gate_up_order !=
            SPARK_RESIDENT_DECODE_STAGE_FP8_MOE_GATE_UP_ORDER_UP_GATE ||
        fp8_moe_plan->weight_layout !=
            SPARK_RESIDENT_DECODE_STAGE_FP8_MOE_WEIGHT_LAYOUT_EXPERT_MAJOR_ROW_MAJOR ||
        fp8_moe_plan->scale_layout !=
            SPARK_RESIDENT_DECODE_STAGE_FP8_MOE_SCALE_LAYOUT_EXPERT_MAJOR_ROW_BLOCK_MAJOR ||
        fp8_moe_plan->quant_mode !=
            SPARK_RESIDENT_DECODE_STAGE_FP8_MOE_QUANT_MODE_E4M3 ||
        fp8_moe_plan->scale_block_size !=
            SPARK_RESIDENT_DECODE_STAGE_FP8_MOE_SCALE_BLOCK_SIZE ||
        fp8_moe_plan->launch_function == 0 ||
        fp8_moe_plan->w1_weight_fp8_e4m3 == 0 ||
        fp8_moe_plan->w1_scale_inv_f32 == 0 ||
        fp8_moe_plan->w2_weight_fp8_e4m3 == 0 ||
        fp8_moe_plan->w2_scale_inv_f32 == 0 ||
        (fp8_moe_plan->capability_flags & required_capabilities) !=
            required_capabilities ||
        !SparkResidentDecodeStagePointerIsAligned(
            fp8_moe_plan->w1_weight_fp8_e4m3,
            SPARK_RESIDENT_DECODE_STAGE_FP8_MOE_WEIGHT_ALIGNMENT_BYTES) ||
        !SparkResidentDecodeStagePointerIsAligned(
            fp8_moe_plan->w2_weight_fp8_e4m3,
            SPARK_RESIDENT_DECODE_STAGE_FP8_MOE_WEIGHT_ALIGNMENT_BYTES) ||
        !SparkResidentDecodeStagePointerIsAligned(
            fp8_moe_plan->w1_scale_inv_f32,
            SPARK_RESIDENT_DECODE_STAGE_FP8_MOE_SCALE_ALIGNMENT_BYTES) ||
        !SparkResidentDecodeStagePointerIsAligned(
            fp8_moe_plan->w2_scale_inv_f32,
            SPARK_RESIDENT_DECODE_STAGE_FP8_MOE_SCALE_ALIGNMENT_BYTES) ||
        (fp8_moe_plan->workspace_bytes != 0u &&
         !SparkResidentDecodeStagePointerIsAligned(
             fp8_moe_plan->workspace,
             SPARK_RESIDENT_DECODE_STAGE_FP8_MOE_WORKSPACE_ALIGNMENT_BYTES)))
    {
        return false;
    }
    return true;
}

static inline bool SparkGlm52ResidentDecodeStageLinearPlanIsUsable(
    const SparkResidentDecodeStageNodeContext *node_context,
    uint32_t plan_index,
    uint32_t input_dimension,
    uint32_t output_dimension)
{
    const SparkGlm52ResidentDecodeStageLinearPlan *linear_plan;

    if (node_context->linear_plans == 0 ||
        plan_index >= node_context->linear_plan_count)
    {
        return false;
    }
    linear_plan = &node_context->linear_plans[plan_index];
    return linear_plan->abi_version ==
            SPARK_RESIDENT_DECODE_STAGE_LINEAR_PLAN_ABI_VERSION &&
        linear_plan->plan_kind !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_UNUSED &&
        linear_plan->input_dimension == input_dimension &&
        linear_plan->output_dimension == output_dimension &&
        linear_plan->maximum_active_sequence_count >=
            node_context->max_active_sequence_count;
}

static inline bool SparkGlm52ResidentDecodeStageW8lutMoePlanIsUsable(
    const SparkResidentDecodeStageNodeContext *node_context)
{
    const SparkGlm52ResidentDecodeStageW8lutMoePlan *plan;
    uint32_t required_capabilities;
    if (node_context == 0 || node_context->w8lut_moe_plan == 0)
    {
        return false;
    }
    plan = node_context->w8lut_moe_plan;
    required_capabilities =
        SPARK_RESIDENT_DECODE_STAGE_W8LUT_MOE_REQUIRED_CAPABILITIES;
    return plan->abi_version ==
            SPARK_RESIDENT_DECODE_STAGE_W8LUT_MOE_PLAN_ABI_VERSION &&
        plan->reserved0 == 0u && plan->reserved1 == 0u &&
        plan->maximum_active_sequence_count >=
            node_context->max_active_sequence_count &&
        plan->maximum_token_count >= node_context->max_active_sequence_count &&
        plan->expert_count == SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT &&
        plan->top_k == SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_TOP_K &&
        plan->hidden_dimension == SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION &&
        plan->intermediate_dimension ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION &&
        plan->output_dtype ==
            SPARK_RESIDENT_DECODE_STAGE_W8LUT_MOE_OUTPUT_DTYPE_BF16 &&
        plan->cuda_architecture == 121u &&
        plan->gate_up_order ==
            SPARK_RESIDENT_DECODE_STAGE_W8LUT_MOE_GATE_UP_ORDER_UP_GATE &&
        plan->weight_layout ==
            SPARK_RESIDENT_DECODE_STAGE_W8LUT_MOE_WEIGHT_LAYOUT_EXPERT_MAJOR_ROW_MAJOR &&
        plan->scale_layout ==
            SPARK_RESIDENT_DECODE_STAGE_W8LUT_MOE_SCALE_LAYOUT_EXPERT_COMPONENT_E0 &&
        plan->quant_mode == SPARK_RESIDENT_DECODE_STAGE_W8LUT_MOE_QUANT_MODE &&
        plan->launch_function != 0 && plan->w1_weight_codes != 0 &&
        plan->w1_exponent_base != 0 && plan->w2_weight_codes != 0 &&
        plan->w2_exponent_base != 0 && plan->workspace != 0 &&
        plan->workspace_bytes != 0u &&
        (plan->capability_flags & required_capabilities) == required_capabilities &&
        SparkResidentDecodeStagePointerIsAligned(
            plan->w1_weight_codes,
            SPARK_RESIDENT_DECODE_STAGE_W8LUT_MOE_WEIGHT_ALIGNMENT_BYTES) &&
        SparkResidentDecodeStagePointerIsAligned(
            plan->w2_weight_codes,
            SPARK_RESIDENT_DECODE_STAGE_W8LUT_MOE_WEIGHT_ALIGNMENT_BYTES) &&
        SparkResidentDecodeStagePointerIsAligned(
            plan->w1_exponent_base,
            SPARK_RESIDENT_DECODE_STAGE_W8LUT_MOE_EXPONENT_ALIGNMENT_BYTES) &&
        SparkResidentDecodeStagePointerIsAligned(
            plan->w2_exponent_base,
            SPARK_RESIDENT_DECODE_STAGE_W8LUT_MOE_EXPONENT_ALIGNMENT_BYTES) &&
        SparkResidentDecodeStagePointerIsAligned(
            plan->workspace,
            SPARK_RESIDENT_DECODE_STAGE_W8LUT_MOE_WORKSPACE_ALIGNMENT_BYTES);
}

static inline bool SparkGlm52ResidentDecodeStageB12xMoePlanIsUsable(
    const SparkResidentDecodeStageNodeContext *node_context,
    const SparkGlm52ResidentDecodeStageB12xMoeDispatchPlan *b12x_moe_dispatch_plan)
{
    const SparkGlm52ResidentDecodeStageB12xMoePlan *b12x_plan;
    uint32_t required_capabilities;

    if (node_context == 0 || b12x_moe_dispatch_plan == 0 ||
        b12x_moe_dispatch_plan->opaque_state == 0)
    {
        return false;
    }

    b12x_plan = (const SparkGlm52ResidentDecodeStageB12xMoePlan *)
        b12x_moe_dispatch_plan->opaque_state;
    required_capabilities =
        SPARK_RESIDENT_DECODE_STAGE_B12X_MOE_REQUIRED_CAPABILITIES;

    if (b12x_plan->abi_version !=
            SPARK_RESIDENT_DECODE_STAGE_B12X_MOE_PLAN_ABI_VERSION ||
        b12x_plan->reserved0 != 0u ||
        b12x_plan->reserved1 != 0u ||
        b12x_plan->maximum_active_sequence_count <
            node_context->max_active_sequence_count ||
        b12x_plan->maximum_token_count <
            node_context->max_active_sequence_count ||
        b12x_plan->expert_count !=
            SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_EXPERT_COUNT ||
        b12x_plan->top_k !=
            SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_TOP_K ||
        b12x_plan->hidden_dimension !=
            SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_HIDDEN_DIMENSION ||
        b12x_plan->intermediate_dimension !=
            SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_INTERMEDIATE_DIMENSION ||
        b12x_plan->gate_up_order !=
            SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_GATE_UP_ORDER_UP_GATE ||
        b12x_plan->weight_layout !=
            SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_WEIGHT_LAYOUT_FLASHINFER_STATIC_VIEW ||
        b12x_plan->scale_layout !=
            SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_SCALE_LAYOUT_FLASHINFER_STATIC_STORAGE ||
        b12x_plan->quant_mode !=
            SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_QUANT_MODE_NVFP4 ||
        b12x_plan->output_dtype !=
            SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_OUTPUT_DTYPE_BF16 ||
        b12x_plan->cuda_architecture != 121u ||
        b12x_plan->state_cell == 0 ||
        b12x_plan->w1_weight_fp4_static_view == 0 ||
        b12x_plan->w1_scale_static_storage_ue4m3 == 0 ||
        b12x_plan->w1_alpha_fp32_by_expert == 0 ||
        b12x_plan->fc2_input_scale_fp32_by_expert == 0 ||
        b12x_plan->w2_weight_fp4_static_view == 0 ||
        b12x_plan->w2_scale_static_storage_ue4m3 == 0 ||
        b12x_plan->w2_alpha_fp32_by_expert == 0 ||
        (b12x_plan->capability_flags & required_capabilities) !=
            required_capabilities)
    {
        return false;
    }

    if (b12x_plan->recipe.abi_version !=
            SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_ABI_VERSION ||
        b12x_plan->recipe.hidden_dimension != b12x_plan->hidden_dimension ||
        b12x_plan->recipe.intermediate_dimension !=
            b12x_plan->intermediate_dimension ||
        b12x_plan->recipe.expert_count != b12x_plan->expert_count ||
        b12x_plan->recipe.top_k != b12x_plan->top_k ||
        b12x_plan->recipe.maximum_token_count <
            node_context->max_active_sequence_count ||
        b12x_plan->recipe.gate_up_order != b12x_plan->gate_up_order ||
        b12x_plan->recipe.weight_layout != b12x_plan->weight_layout ||
        b12x_plan->recipe.scale_layout != b12x_plan->scale_layout ||
        b12x_plan->recipe.quant_mode != b12x_plan->quant_mode ||
        b12x_plan->recipe.output_dtype != b12x_plan->output_dtype ||
        b12x_plan->recipe.cuda_architecture != b12x_plan->cuda_architecture ||
        b12x_plan->recipe.qualified_maximum_microseconds == 0u ||
        b12x_plan->recipe.qualification_record_hash_low64 == 0u ||
        b12x_plan->recipe.kernel_manifest_hash_low64 == 0u)
    {
        return false;
    }

    return true;
}

static inline bool SparkGlm52ResidentDecodeStageLinearPlanKindIsProductionFast(
    uint32_t plan_kind)
{
    switch (plan_kind)
    {
    case SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_CUBLASLT_BF16_ROW_MAJOR:
    case SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_CUBLASLT_FP8_E4M3_ROW_MAJOR:
    case SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_DRIVER_CUSTOM:
    case SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_TENSOR_CORE_FP8_E4M3_ROW_MAJOR:
    case SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_TENSOR_CORE_NVFP4_E2M1_ROW_MAJOR:
    case SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_TENSOR_CORE_MXFP4_E2M1_ROW_MAJOR:
        return true;
    default:
        return false;
    }
}

static inline bool SparkGlm52ResidentDecodeStageRouterLinearPlanIsUsable(
    const SparkResidentDecodeStageNodeContext *node_context)
{
    return SparkGlm52ResidentDecodeStageLinearPlanIsUsable(
        node_context,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_ROUTER_LOGITS,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT);
}

static inline uint32_t SparkGlm52ResidentDecodeStageLinearPlanExpectedScaleBlockSize(
    uint32_t weight_format)
{
    if (weight_format ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_FP8_E4M3)
    {
        return SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_BLOCK;
    }
    if (weight_format ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_NVFP4_E2M1)
    {
        return SPARK_RESIDENT_DECODE_STAGE_NVFP4_GROUP_SIZE;
    }
    if (weight_format ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_MXFP4_E2M1)
    {
        return SPARK_RESIDENT_DECODE_STAGE_MXFP4_GROUP_SIZE;
    }
    return 0u;
}

static inline uint32_t SparkGlm52ResidentDecodeStageLinearPlanExpectedWeightFormat(
    uint32_t plan_kind)
{
    if (plan_kind ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_TENSOR_CORE_FP8_E4M3_ROW_MAJOR)
    {
        return SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_FP8_E4M3;
    }
    if (plan_kind ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_TENSOR_CORE_NVFP4_E2M1_ROW_MAJOR)
    {
        return SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_NVFP4_E2M1;
    }
    if (plan_kind ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_TENSOR_CORE_MXFP4_E2M1_ROW_MAJOR)
    {
        return SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_MXFP4_E2M1;
    }
    return SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_BF16;
}

static inline bool SparkGlm52ResidentDecodeStageB12xMoeDispatchPlanIsUsable(
    const SparkResidentDecodeStageNodeContext *node_context)
{
    const SparkGlm52ResidentDecodeStageB12xMoeDispatchPlan *b12x_moe_dispatch_plan;
    uint64_t required_route_count;

    if (node_context == 0 || node_context->b12x_moe_dispatch_plan == 0 ||
        node_context->moe_router_score_bias_f32 == 0 ||
        !isfinite(node_context->moe_routed_scaling_factor) ||
        node_context->moe_routed_scaling_factor == 0.0f)
    {
        return false;
    }
    b12x_moe_dispatch_plan = node_context->b12x_moe_dispatch_plan;
    required_route_count =
        (uint64_t)node_context->max_active_sequence_count *
        (uint64_t)node_context->moe_top_k;
    if (required_route_count > UINT32_MAX ||
        b12x_moe_dispatch_plan->abi_version !=
            SPARK_RESIDENT_DECODE_STAGE_B12X_MOE_DISPATCH_PLAN_ABI_VERSION ||
        b12x_moe_dispatch_plan->plan_kind !=
            SPARK_RESIDENT_DECODE_STAGE_B12X_MOE_DISPATCH_PLAN_KIND_FLASHINFER_B12X ||
        b12x_moe_dispatch_plan->reserved != 0u ||
        b12x_moe_dispatch_plan->maximum_active_sequence_count <
            node_context->max_active_sequence_count ||
        b12x_moe_dispatch_plan->maximum_route_count < required_route_count ||
        b12x_moe_dispatch_plan->expert_count != node_context->moe_expert_count ||
        b12x_moe_dispatch_plan->top_k != node_context->moe_top_k ||
        b12x_moe_dispatch_plan->intermediate_dimension !=
            node_context->moe_intermediate_dimension)
    {
        return false;
    }
    return SparkGlm52ResidentDecodeStageB12xMoePlanIsUsable(
        node_context,
        b12x_moe_dispatch_plan);
}

static inline bool SparkGlm52ResidentDecodeStageRouterLinearPlanIsProductionFast(
    const SparkResidentDecodeStageNodeContext *node_context)
{
    const SparkGlm52ResidentDecodeStageLinearPlan *router_plan;

    if (!SparkGlm52ResidentDecodeStageRouterLinearPlanIsUsable(node_context) ||
        node_context->linear_plans == 0 ||
        node_context->linear_plan_count <=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_ROUTER_LOGITS)
    {
        return false;
    }

    router_plan = &node_context->linear_plans[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_ROUTER_LOGITS];
    return router_plan->output_is_f32 != 0u &&
        SparkGlm52ResidentDecodeStageLinearPlanKindIsProductionFast(
            router_plan->plan_kind);
}

static inline bool SparkGlm52ResidentDecodeStageRouterWeightOrPlanIsUsable(
    const SparkResidentDecodeStageNodeContext *node_context)
{
    return SparkResidentDecodeStagePointerIsAligned(
            node_context->moe_router_weight_bf16,
            2u) ||
        SparkGlm52ResidentDecodeStageRouterLinearPlanIsUsable(node_context);
}

static inline bool SparkGlm52ResidentDecodeStageLinearPlanHasBuiltInQuantizedTensorCoreState(
    const SparkGlm52ResidentDecodeStageLinearPlan *linear_plan)
{
    const SparkResidentDecodeStageQuantizedLinearView *view;
    uint32_t expected_weight_format;
    uint32_t expected_scale_block_size;
    uint64_t weight_element_count;
    uint64_t input_scale_block_count;
    uint64_t output_scale_block_count;
    uint64_t scale_element_count;
    uint64_t required_payload_bytes;
    uint64_t required_scale_bytes;
    uint64_t required_output_workspace_bytes;
    uint64_t output_element_bytes;

    if (linear_plan == 0 || linear_plan->custom_state == 0 ||
        linear_plan->input_dimension == 0u ||
        linear_plan->output_dimension == 0u ||
        (linear_plan->input_dimension & 15u) != 0u ||
        (linear_plan->output_dimension & 15u) != 0u ||
        linear_plan->maximum_active_sequence_count == 0u ||
        linear_plan->maximum_active_sequence_count >
            SPARK_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT)
    {
        return false;
    }

    expected_weight_format =
        SparkGlm52ResidentDecodeStageLinearPlanExpectedWeightFormat(
            linear_plan->plan_kind);
    expected_scale_block_size =
        SparkGlm52ResidentDecodeStageLinearPlanExpectedScaleBlockSize(
            expected_weight_format);
    if (expected_scale_block_size == 0u)
    {
        return false;
    }

    view = (const SparkResidentDecodeStageQuantizedLinearView *)
        linear_plan->custom_state;
    if (view->abi_version !=
            SPARK_RESIDENT_DECODE_STAGE_QUANTIZED_LINEAR_VIEW_ABI_VERSION ||
        view->weight_format != expected_weight_format ||
        view->input_dimension != linear_plan->input_dimension ||
        view->output_dimension != linear_plan->output_dimension ||
        view->storage_output_dimension < view->output_dimension ||
        (view->storage_output_dimension & 15u) != 0u ||
        view->scale_block_size != expected_scale_block_size ||
        view->output_is_f32 != linear_plan->output_is_f32 ||
        view->reserved0 != 0u ||
        view->weight_payload == 0 ||
        view->weight_scale == 0)
    {
        return false;
    }
    if (view->weight_format ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_FP8_E4M3)
    {
        if ((view->storage_output_dimension %
                SPARK_RESIDENT_DECODE_STAGE_FP8_SCALED_GEMM_OUTPUT_ALIGNMENT) != 0u)
        {
            return false;
        }
    }
    else if (view->storage_output_dimension != view->output_dimension)
    {
        return false;
    }

    weight_element_count =
        (uint64_t)view->input_dimension *
        (uint64_t)view->storage_output_dimension;
    required_payload_bytes = view->weight_format ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_FP8_E4M3
        ? weight_element_count
        : SparkResidentDecodeStageDivideRoundUpU64(
            weight_element_count,
            2u);
    input_scale_block_count = SparkResidentDecodeStageDivideRoundUpU64(
        view->input_dimension,
        view->scale_block_size);
    output_scale_block_count = SparkResidentDecodeStageDivideRoundUpU64(
        view->storage_output_dimension,
        view->scale_block_size);
    scale_element_count = input_scale_block_count * output_scale_block_count;
    required_scale_bytes = view->weight_format ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_FP8_E4M3
        ? scale_element_count * (uint64_t)sizeof(float)
        : scale_element_count;

    output_element_bytes = view->output_is_f32 != 0u
        ? (uint64_t)sizeof(float)
        : (uint64_t)sizeof(uint16_t);
    required_output_workspace_bytes =
        (uint64_t)linear_plan->maximum_active_sequence_count *
        (uint64_t)view->storage_output_dimension * output_element_bytes;
    return view->weight_payload_bytes >= required_payload_bytes &&
        view->weight_scale_bytes >= required_scale_bytes &&
        (view->storage_output_dimension == view->output_dimension
            ? view->output_workspace == 0 && view->output_workspace_bytes == 0u
            : view->output_workspace != 0 &&
                view->output_workspace_bytes >= required_output_workspace_bytes);
}

static inline bool SparkGlm52ResidentDecodeStageLinearPlanHasQuantizedProjectionKind(
    const SparkResidentDecodeStageNodeContext *node_context,
    uint32_t plan_index)
{
    const SparkGlm52ResidentDecodeStageLinearPlan *linear_plan;
    uint32_t required_tensor_core_plan_kind;

    if (node_context->linear_plans == 0 ||
        plan_index >= node_context->linear_plan_count)
    {
        return false;
    }

    linear_plan = &node_context->linear_plans[plan_index];
    required_tensor_core_plan_kind =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_UNUSED;
    if (node_context->projection_mode ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PROJECTION_RAW_GLM_FP8_E4M3)
    {
        if (node_context->projection_backend_mode ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_PROJECTION_BACKEND_PREBOUND_CUBLASLT)
        {
            return linear_plan->plan_kind ==
                SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_CUBLASLT_FP8_E4M3_ROW_MAJOR;
        }
        required_tensor_core_plan_kind =
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_TENSOR_CORE_FP8_E4M3_ROW_MAJOR;
    }
    else if (node_context->projection_mode ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PROJECTION_RAW_GLM_NVFP4_E2M1)
    {
        required_tensor_core_plan_kind =
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_TENSOR_CORE_NVFP4_E2M1_ROW_MAJOR;
    }
    else if (node_context->projection_mode ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PROJECTION_RAW_GLM_MXFP4_E2M1)
    {
        required_tensor_core_plan_kind =
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_TENSOR_CORE_MXFP4_E2M1_ROW_MAJOR;
    }
    else
    {
        return false;
    }

    if (linear_plan->plan_kind != required_tensor_core_plan_kind)
    {
        return false;
    }
    return linear_plan->custom_launch_function != 0 ||
        SparkGlm52ResidentDecodeStageLinearPlanHasBuiltInQuantizedTensorCoreState(
            linear_plan);
}
#endif
