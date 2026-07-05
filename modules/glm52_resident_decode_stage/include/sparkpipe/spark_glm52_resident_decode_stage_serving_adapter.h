#pragma once

#include <stdint.h>

#include "sparkpipe/spark_glm52_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_glm52_serving_engine.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_GLM52_RESIDENT_DECODE_STAGE_SERVING_ADAPTER_ABI_VERSION 1u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_SERVING_ADAPTER_CONFIGURATION_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52ResidentDecodeStageServingAdapterConfiguration))
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_SERVING_ADAPTER_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52ResidentDecodeStageServingAdapter))

#define SPARK_GLM52_RESIDENT_DECODE_STAGE_SERVING_ADAPTER_FLAG_SYNCHRONIZE_AFTER_LAUNCH \
    0x00000001u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_SERVING_ADAPTER_FLAG_KNOWN_FLAGS \
    SPARK_GLM52_RESIDENT_DECODE_STAGE_SERVING_ADAPTER_FLAG_SYNCHRONIZE_AFTER_LAUNCH

typedef struct SparkGlm52ResidentDecodeStageServingAdapterConfiguration
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t flags;
    uint32_t pipeline_slot_index;
    uint32_t maximum_active_sequence_count;
    uint32_t maximum_prompt_token_count;
    uint32_t vocabulary_size;
    uint32_t hidden_dimension;
    uint32_t final_token_stage;
    uint32_t reserved0;
    const SparkGlm52ResidentDecodeStageStageSlicePlan *stage_slice_plan;
    const SparkGlm52ResidentDecodeStageNodeContext *const *layer_node_contexts;
    uint32_t layer_count;
    const void *embedding_weight_bf16;
    void *cuda_stream;
} SparkGlm52ResidentDecodeStageServingAdapterConfiguration;

typedef struct SparkGlm52ResidentDecodeStageServingAdapter
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t flags;
    uint32_t pipeline_slot_index;
    uint32_t maximum_active_sequence_count;
    uint32_t maximum_prompt_token_count;
    uint32_t vocabulary_size;
    uint32_t hidden_dimension;
    uint32_t final_token_stage;
    uint32_t layer_count;
    uint32_t owns_cuda_stream;
    uint32_t reserved0;
    const SparkGlm52ResidentDecodeStageStageSlicePlan *stage_slice_plan;
    const SparkGlm52ResidentDecodeStageNodeContext *const *layer_node_contexts;
    const void *embedding_weight_bf16;
    void *cuda_stream;
    uint32_t *host_lane_offsets;
    uint32_t *host_lane_counts;
    uint32_t *host_decode_positions;
    uint32_t *host_decode_token_ids;
    uint32_t *host_mtp_draft_token_budgets;
    uint32_t *host_mtp_committed_token_ids;
    uint32_t *device_prefill_token_ids;
    uint32_t *device_lane_offsets;
    uint32_t *device_lane_counts;
    uint32_t *device_decode_positions;
    uint32_t *device_decode_token_ids;
    uint32_t *device_mtp_draft_token_budgets;
    uint32_t *device_prompt_positions;
    uint32_t *device_prompt_slot_mapping;
    uint32_t *device_prompt_context_lengths;
    uint32_t *device_prompt_first_block_token_offsets;
    uint32_t *device_prompt_token_counts;
    void *device_prompt_hidden_bf16;
    void *device_prompt_output_hidden_bf16;
} SparkGlm52ResidentDecodeStageServingAdapter;

SparkStatus SparkGlm52ResidentDecodeStageServingAdapterInitialize(
    SparkGlm52ResidentDecodeStageServingAdapter *adapter,
    const SparkGlm52ResidentDecodeStageServingAdapterConfiguration *configuration);

void SparkGlm52ResidentDecodeStageServingAdapterDestroy(
    SparkGlm52ResidentDecodeStageServingAdapter *adapter);

SparkStatus SparkGlm52ResidentDecodeStageServingAdapterPrefill(
    void *context,
    const SparkGlm52PromptPipelinePrefillDispatch *prefill_dispatch);

SparkStatus SparkGlm52ResidentDecodeStageServingAdapterDecode(
    void *context,
    const SparkGlm52ServingDecodeDispatch *decode_dispatch,
    SparkGlm52ServingDecodeResult *decode_result);

#ifdef __cplusplus
}
#endif
