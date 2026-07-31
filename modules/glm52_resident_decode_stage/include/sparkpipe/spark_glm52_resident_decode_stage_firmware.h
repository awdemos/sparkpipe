#ifndef SPARKPIPE_SPARK_GLM52_RESIDENT_DECODE_STAGE_FIRMWARE_H
#define SPARKPIPE_SPARK_GLM52_RESIDENT_DECODE_STAGE_FIRMWARE_H

#include <math.h>
#include "sparkpipe/spark_resident_decode_stage.h"
#include "sparkpipe/spark_glm52_model.h"
#include <stdbool.h>

#define SPARK_RESIDENT_DECODE_STAGE_MODULE_ID \
    "spark.glm52.resident_decode_stage.bf16.h6144.h64.d512.r64.k2048.b1024.rv256.mtp6.v1"
#define SPARK_RESIDENT_DECODE_STAGE_TARGET \
    "cuda.sm121.glm52.resident_decode_stage.bf16"

#include "sparkpipe/spark_glm52_kv_cache.h"

#include "sparkpipe/spark_glm52_sm121_flashinfer_b12x_moe.h"
#include "sparkpipe/spark_glm52_dspark.h"

// Relocated from the common stage header: glm plan vocabulary that no
// common path names. The frame's tap members are opaque there; the
// concrete plan type and the tap-count mirror live here.
typedef char SparkGlm52StageMtpDraftMirrorAssert[
    (SPARK_GLM52_MODEL_MTP_DRAFT_TOKEN_COUNT <= SPARK_RESIDENT_DECODE_STAGE_MTP_MAX_DRAFT_TOKENS) ? 1 : -1];
typedef char SparkGlm52StageTapCountMirrorAssert[
    (SPARK_GLM52_MODEL_DSPARK_AUX_LAYER_COUNT == SPARK_STAGE_MODEL_MAX_HIDDEN_TAPS) ? 1 : -1];
#define SPARK_RESIDENT_DECODE_STAGE_DSA_KV_FRAGMENT_TRANSPORT_PLAN_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52ResidentDecodeStageDsaKvFragmentTransportPlan))
#define SPARK_RESIDENT_DECODE_STAGE_DSA_KV_FRAGMENT_TRANSPORT_PAYLOAD_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52ResidentDecodeStageDsaKvFragmentTransportPayload))
typedef struct SparkGlm52ResidentDecodeStageB12xMoeDispatchPlan
{
    uint32_t abi_version;
    uint32_t plan_kind;
    uint32_t maximum_active_sequence_count;
    uint32_t maximum_route_count;
    uint32_t expert_count;
    uint32_t top_k;
    uint32_t intermediate_dimension;
    uint32_t reserved;
    void *opaque_state;
    uint64_t validated_maximum_latency_ns;
} SparkGlm52ResidentDecodeStageB12xMoeDispatchPlan;
typedef struct SparkGlm52ResidentDecodeStageB12xMoePlan
{
    uint32_t abi_version;
    uint32_t capability_flags;
    uint32_t maximum_active_sequence_count;
    uint32_t maximum_token_count;
    uint32_t expert_count;
    uint32_t top_k;
    uint32_t hidden_dimension;
    uint32_t intermediate_dimension;
    uint32_t gate_up_order;
    uint32_t weight_layout;
    uint32_t scale_layout;
    uint32_t quant_mode;
    uint32_t output_dtype;
    uint32_t cuda_architecture;
    uint32_t reserved0;
    uint32_t reserved1;
    SparkGlm52Sm121FlashInferB12xMoeRecipe recipe;
    void **state_cell;
    const void *w1_weight_fp4_static_view;
    const void *w1_scale_static_storage_ue4m3;
    const float *w1_alpha_fp32_by_expert;
    const float *fc2_input_scale_fp32_by_expert;
    const void *w2_weight_fp4_static_view;
    const void *w2_scale_static_storage_ue4m3;
    const float *w2_alpha_fp32_by_expert;
    void *workspace;
    uint64_t workspace_bytes;
    uint64_t validated_maximum_latency_ns;
} SparkGlm52ResidentDecodeStageB12xMoePlan;
typedef struct SparkGlm52ResidentDecodeStageDsaKvFragmentTransportPayload
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t flags;
    uint32_t reserved0;
    uint64_t source_block_stride_bytes;
    uint64_t destination_block_stride_bytes;
    uint64_t transfer_bytes;
    const void *source_base;
    void *destination_base;
} SparkGlm52ResidentDecodeStageDsaKvFragmentTransportPayload;
typedef struct SparkGlm52ResidentDecodeStageDsaKvFragmentTransportPlan
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t capability_flags;
    uint32_t payload_count;
    uint32_t physical_block_count;
    uint32_t maximum_active_sequence_count;
    uint32_t selected_block_stride;
    uint32_t selected_block_capacity;
    uint64_t transport_epoch;
    const uint32_t *source_physical_block_indices_by_destination;
    const uint32_t *destination_physical_block_indices_by_source;
    uint64_t *requested_epoch_by_physical_block;
    uint64_t *ready_epoch_by_physical_block;
    uint32_t *written_logical_block_indices;
    uint32_t *written_logical_block_counts;
    uint32_t written_logical_block_stride;
    uint32_t reserved1;
    const uint64_t *source_fragment_keys_by_physical_block;
    const uint64_t *expected_fragment_keys_by_destination;
    uint32_t *copied_block_count_device;
    uint32_t *duplicate_block_count_device;
    uint32_t *invalid_block_count_device;
    uint32_t *key_mismatch_count_device;
    void *selection_ready_event;
    void *transport_ready_event;
    void *transport_stream;
    uint64_t validated_maximum_latency_ns;
    SparkGlm52ResidentDecodeStageDsaKvFragmentTransportPayload payloads[
        SPARK_RESIDENT_DECODE_STAGE_DSA_KV_FRAGMENT_TRANSPORT_MAX_PAYLOADS];
} SparkGlm52ResidentDecodeStageDsaKvFragmentTransportPlan;
typedef struct SparkGlm52ResidentDecodeStageFp8MoePlan
{
    uint32_t abi_version;
    uint32_t capability_flags;
    uint32_t maximum_active_sequence_count;
    uint32_t maximum_token_count;
    uint32_t expert_count;
    uint32_t top_k;
    uint32_t hidden_dimension;
    uint32_t intermediate_dimension;
    uint32_t output_dtype;
    uint32_t cuda_architecture;
    uint32_t gate_up_order;
    uint32_t weight_layout;
    uint32_t scale_layout;
    uint32_t quant_mode;
    uint32_t scale_block_size;
    uint32_t reserved0;
    uint32_t reserved1;
    void *launch_function;
    void *opaque_state;
    const uint8_t *w1_weight_fp8_e4m3;
    const float *w1_scale_inv_f32;
    const uint8_t *w2_weight_fp8_e4m3;
    const float *w2_scale_inv_f32;
    void *workspace;
    uint64_t workspace_bytes;
    uint64_t validated_maximum_latency_ns;
} SparkGlm52ResidentDecodeStageFp8MoePlan;
typedef enum SparkGlm52ResidentDecodeStageLinearPlanIndex
{
    SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_QUERY_LATENT = 0,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_QUERY_ROPE = 1,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_KEY_ROPE = 2,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_KV_LATENT = 3,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_RAW_QUERY_A = 4,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_RAW_QUERY_B = 5,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_RAW_KV_A = 6,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_RAW_KV_B = 7,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_ATTENTION_OUTPUT = 8,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_DENSE_GATE = 9,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_DENSE_UP = 10,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_DENSE_DOWN = 11,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_MOE_GATE = 12,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_MOE_UP = 13,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_MOE_DOWN = 14,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_RESTRICTED_LOGITS = 15,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_ROUTER_LOGITS = 16,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_DSA_QUERY = 17,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_DSA_KEY = 18,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_DSA_WEIGHTS = 19,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_RESERVED_1 = 20
} SparkGlm52ResidentDecodeStageLinearPlanIndex;
typedef enum SparkGlm52ResidentDecodeStageLinearPlanKind
{
    SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_UNUSED = 0,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_CUBLASLT_BF16_ROW_MAJOR = 1,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_CUBLASLT_FP8_E4M3_ROW_MAJOR = 2,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_DRIVER_CUSTOM = 3,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_TENSOR_CORE_FP8_E4M3_ROW_MAJOR = 4,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_TENSOR_CORE_NVFP4_E2M1_ROW_MAJOR = 5,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_TENSOR_CORE_MXFP4_E2M1_ROW_MAJOR = 6
} SparkGlm52ResidentDecodeStageLinearPlanKind;
typedef enum SparkGlm52ResidentDecodeStageLinearWeightFormat
{
    SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_BF16 = 0,
    SPARK_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_FP8_E4M3 = 1,
    SPARK_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_NVFP4_E2M1 = 2,
    SPARK_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_MXFP4_E2M1 = 3
} SparkGlm52ResidentDecodeStageLinearWeightFormat;

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
#define SPARK_RESIDENT_DECODE_STAGE_CACHE_TOKEN_ELEMENTS SPARK_GLM52_MODEL_CACHE_TOKEN_ELEMENTS
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
        SPARK_RESIDENT_DECODE_STAGE_LAYER_ROUTED_NVFP4_TOPK;
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

    fp8_moe_plan = (const SparkGlm52ResidentDecodeStageFp8MoePlan *)node_context->fp8_moe_plan;
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
    const SparkResidentDecodeStageLinearPlan *linear_plan;

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
        SPARK_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_FP8_E4M3)
    {
        return SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_BLOCK;
    }
    if (weight_format ==
        SPARK_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_NVFP4_E2M1)
    {
        return SPARK_RESIDENT_DECODE_STAGE_NVFP4_GROUP_SIZE;
    }
    if (weight_format ==
        SPARK_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_MXFP4_E2M1)
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
        return SPARK_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_FP8_E4M3;
    }
    if (plan_kind ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_TENSOR_CORE_NVFP4_E2M1_ROW_MAJOR)
    {
        return SPARK_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_NVFP4_E2M1;
    }
    if (plan_kind ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_TENSOR_CORE_MXFP4_E2M1_ROW_MAJOR)
    {
        return SPARK_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_MXFP4_E2M1;
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
    b12x_moe_dispatch_plan = (const SparkGlm52ResidentDecodeStageB12xMoeDispatchPlan *)node_context->b12x_moe_dispatch_plan;
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
    const SparkResidentDecodeStageLinearPlan *router_plan;

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
    const SparkResidentDecodeStageLinearPlan *linear_plan)
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
            SPARK_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_FP8_E4M3)
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
            SPARK_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_FP8_E4M3
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
            SPARK_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_FP8_E4M3
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
    const SparkResidentDecodeStageLinearPlan *linear_plan;
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
        SPARK_RESIDENT_DECODE_STAGE_PROJECTION_RAW_FP8_E4M3)
    {
        if (node_context->projection_backend_mode ==
            SPARK_RESIDENT_DECODE_STAGE_PROJECTION_BACKEND_PREBOUND_CUBLASLT)
        {
            return linear_plan->plan_kind ==
                SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_CUBLASLT_FP8_E4M3_ROW_MAJOR;
        }
        required_tensor_core_plan_kind =
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_TENSOR_CORE_FP8_E4M3_ROW_MAJOR;
    }
    else if (node_context->projection_mode ==
        SPARK_RESIDENT_DECODE_STAGE_PROJECTION_RAW_NVFP4_E2M1)
    {
        required_tensor_core_plan_kind =
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_TENSOR_CORE_NVFP4_E2M1_ROW_MAJOR;
    }
    else if (node_context->projection_mode ==
        SPARK_RESIDENT_DECODE_STAGE_PROJECTION_RAW_MXFP4_E2M1)
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

static inline uint32_t SparkResidentDecodeStageEffectiveKvBlockTokenCount(
    const SparkResidentDecodeStageNodeContext *node_context)
{
    if (node_context->kv_block_token_count != 0u)
    {
        return node_context->kv_block_token_count;
    }
    return SPARK_RESIDENT_DECODE_STAGE_BLOCK_TOKENS;
}

#endif
