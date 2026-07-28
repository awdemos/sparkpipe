#ifndef SPARKPIPE_SPARK_GLM52_RESIDENT_DECODE_STAGE_BACKEND_H
#define SPARKPIPE_SPARK_GLM52_RESIDENT_DECODE_STAGE_BACKEND_H

#include <stdint.h>

#include "sparkpipe/spark_glm52_resident_decode_stage_firmware.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*SparkResidentDecodeStageBackendCompletionFunction)(
    void *completion_context);

typedef struct SparkResidentDecodeStageBackendCompletion
{
    SparkResidentDecodeStageBackendCompletionFunction function;
    void *context;
    uint32_t requested_token_count;
    uint32_t token_count;
    uint32_t token_ids[SPARK_MODEL_DRIVER_COMPLETION_TOKEN_CAPACITY];
} SparkResidentDecodeStageBackendCompletion;

SparkStatus SparkResidentDecodeStageBackendSubmit(
    const SparkResidentDecodeStageNodeContext *node_context,
    uint32_t pipeline_slot_index,
    uint32_t active_sequence_count,
    const SparkKvBlockTableView *runtime_kv_block_table,
    SparkResidentDecodeStageBackendCompletion *completion);

SparkStatus SparkResidentDecodeStageBackendSubmitStageSlice(
    const SparkResidentDecodeStageStageSlicePlan *stage_slice_plan,
    const SparkResidentDecodeStageNodeContext *const *layer_node_contexts,
    uint32_t layer_count,
    uint32_t pipeline_slot_index,
    uint32_t active_sequence_count,
    uint32_t final_token_stage,
    const SparkKvBlockTableView *runtime_kv_block_table,
    const SparkResidentDecodeStageFrameContext *frame_context,
    SparkResidentDecodeStageBackendCompletion *completion);

SparkStatus SparkResidentDecodeStageBackendSubmitBulkPrefill(
    const SparkResidentDecodeStageNodeContext *node_context,
    uint32_t pipeline_slot_index,
    uint32_t active_sequence_count,
    uint32_t prompt_token_offset,
    uint32_t prompt_token_count,
    const SparkKvBlockTableView *runtime_kv_block_table,
    const SparkResidentDecodeStagePrefillFrameView *prefill_frame_view,
    SparkResidentDecodeStageBackendCompletion *completion);

SparkStatus SparkResidentDecodeStageBackendSubmitStageSliceBulkPrefill(
    const SparkResidentDecodeStageNodeContext *const *layer_node_contexts,
    uint32_t layer_count,
    uint32_t pipeline_slot_index,
    uint32_t active_sequence_count,
    uint32_t prompt_token_offset,
    uint32_t prompt_token_count,
    const SparkKvBlockTableView *runtime_kv_block_table,
    const SparkResidentDecodeStagePrefillFrameView *prefill_frame_view,
    SparkResidentDecodeStageBackendCompletion *completion);

void SparkResidentDecodeStageBackendQuiesce(
    const SparkResidentDecodeStageNodeContext *node_context);

#ifdef __cplusplus
}
#endif

#endif
