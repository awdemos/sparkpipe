#pragma once

#include <stdint.h>

#include "sparkpipe/spark_glm52_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_serving_engine.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_RESIDENT_DECODE_STAGE_SERVING_ADAPTER_ABI_VERSION 1u
#define SPARK_RESIDENT_DECODE_STAGE_SERVING_ADAPTER_CONFIGURATION_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkResidentDecodeStageServingAdapterConfiguration))
#define SPARK_RESIDENT_DECODE_STAGE_SERVING_ADAPTER_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkResidentDecodeStageServingAdapter))

#define SPARK_RESIDENT_DECODE_STAGE_SERVING_ADAPTER_FLAG_SYNCHRONIZE_AFTER_LAUNCH \
    0x00000001u
#define SPARK_RESIDENT_DECODE_STAGE_SERVING_ADAPTER_FLAG_KNOWN_FLAGS \
    SPARK_RESIDENT_DECODE_STAGE_SERVING_ADAPTER_FLAG_SYNCHRONIZE_AFTER_LAUNCH

typedef struct SparkResidentDecodeStageServingAdapterConfiguration
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t flags;
    uint32_t pipeline_slot_index;
    /* First-party layer state. The buffer struct inference/llms/glm5_2/layer.cuh
       reads, and the absolute index of this rank's first layer - absolute
       because the DSA index-share group is computed from it, and a group
       boundary does not align with a rank boundary. Unused by the legacy path
       and zero there. */
    void *first_party_buffers;
    uint32_t first_layer_index;
    uint32_t host_max_context_length;
    uint32_t maximum_active_sequence_count;
    uint32_t maximum_prompt_token_count;
    uint32_t vocabulary_size;
    uint32_t hidden_dimension;
    uint32_t final_token_stage;
    uint32_t reserved0;
    const SparkResidentDecodeStageStageSlicePlan *stage_slice_plan;
    const SparkResidentDecodeStageNodeContext *const *layer_node_contexts;
    uint32_t layer_count;
    const void *embedding_weight_bf16;
    void *cuda_stream;
} SparkResidentDecodeStageServingAdapterConfiguration;

typedef struct SparkResidentDecodeStageServingAdapter
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t flags;
    uint32_t pipeline_slot_index;
    /* First-party layer state. The buffer struct inference/llms/glm5_2/layer.cuh
       reads, and the absolute index of this rank's first layer - absolute
       because the DSA index-share group is computed from it, and a group
       boundary does not align with a rank boundary. Unused by the legacy path
       and zero there. */
    void *first_party_buffers;
    uint32_t first_layer_index;
    uint32_t host_max_context_length;
    uint32_t maximum_active_sequence_count;
    uint32_t maximum_prompt_token_count;
    uint32_t vocabulary_size;
    uint32_t hidden_dimension;
    uint32_t final_token_stage;
    uint32_t layer_count;
    uint32_t owns_cuda_stream;
    uint32_t reserved0;
    const SparkResidentDecodeStageStageSlicePlan *stage_slice_plan;
    const SparkResidentDecodeStageNodeContext *const *layer_node_contexts;
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
} SparkResidentDecodeStageServingAdapter;

SparkStatus SparkResidentDecodeStageServingAdapterInitialize(
    SparkResidentDecodeStageServingAdapter *adapter,
    const SparkResidentDecodeStageServingAdapterConfiguration *configuration);

void SparkResidentDecodeStageServingAdapterDestroy(
    SparkResidentDecodeStageServingAdapter *adapter);

SparkStatus SparkResidentDecodeStageServingAdapterPrefill(
    void *context,
    const SparkPromptPipelinePrefillDispatch *prefill_dispatch);

SparkStatus SparkResidentDecodeStageServingAdapterDecode(
    void *context,
    const SparkServingDecodeDispatch *decode_dispatch,
    SparkServingDecodeResult *decode_result);

#ifdef __cplusplus
}
#endif
