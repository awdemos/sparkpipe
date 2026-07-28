#include "spark_glm52_resident_decode_stage_backend.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime_api.h>

#include "sparkpipe/spark_glm52_resident_decode_stage_required_cuda.h"

static void CUDART_CB SparkResidentDecodeStageCudaCompletion(
    void *completion_context)
{
    SparkResidentDecodeStageBackendCompletion *completion;

    completion =
        (SparkResidentDecodeStageBackendCompletion *)completion_context;
    if (completion != 0 && completion->function != 0)
    {
        completion->function(completion->context);
    }
}

static SparkStatus SparkResidentDecodeStageCudaCopyFinalTokens(
    const SparkResidentDecodeStagePipelineSlot *pipeline_slot,
    uint32_t final_token_stage,
    uint32_t active_sequence_count,
    SparkResidentDecodeStageBackendCompletion *completion,
    cudaStream_t cuda_stream)
{
    uint32_t token_count;
    cudaError_t cuda_status;

    if (final_token_stage == 0u)
    {
        return SPARK_STATUS_OK;
    }
    if (pipeline_slot == 0 || completion == 0 ||
        active_sequence_count == 0u ||
        pipeline_slot->restricted_selected_token_ids == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    token_count = completion->requested_token_count;
    if (token_count == 0u ||
        token_count > SPARK_MODEL_DRIVER_COMPLETION_TOKEN_CAPACITY ||
        token_count >
            (SPARK_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT + 1u))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    cuda_status = cudaMemcpyAsync(
        &completion->token_ids[0u],
        pipeline_slot->restricted_selected_token_ids,
        sizeof(uint32_t),
        cudaMemcpyDeviceToHost,
        cuda_stream);
    if (cuda_status != cudaSuccess)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    if (token_count > 1u)
    {
        if (pipeline_slot->mtp_committed_token_ids == 0)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        cuda_status = cudaMemcpyAsync(
            &completion->token_ids[1u],
            pipeline_slot->mtp_committed_token_ids,
            (size_t)(token_count - 1u) * sizeof(uint32_t),
            cudaMemcpyDeviceToHost,
            cuda_stream);
        if (cuda_status != cudaSuccess)
        {
            return SPARK_STATUS_INTERNAL_ERROR;
        }
    }
    completion->token_count = token_count;
    return SPARK_STATUS_OK;
}

extern "C" SparkStatus SparkResidentDecodeStageBackendVerifyRequiredCudaModules(
    const SparkResidentDecodeStageNodeContext *node_context)
{
    if (node_context == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SparkGlm52Sm121RequiredDecodeStageInitialize(node_context);
}

extern "C" SparkStatus SparkResidentDecodeStageBackendSubmit(
    const SparkResidentDecodeStageNodeContext *node_context,
    uint32_t pipeline_slot_index,
    uint32_t active_sequence_count,
    const SparkKvBlockTableView *runtime_kv_block_table,
    SparkResidentDecodeStageBackendCompletion *completion)
{
    const SparkResidentDecodeStagePipelineSlot *pipeline_slot;
    void *cuda_stream;
    SparkStatus status;
    cudaError_t cuda_status;

    if (node_context == 0 || completion == 0 || completion->function == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (node_context->pipeline_slots == 0 ||
        pipeline_slot_index >= node_context->pipeline_slot_count ||
        active_sequence_count == 0u ||
        active_sequence_count > node_context->max_active_sequence_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    pipeline_slot = &node_context->pipeline_slots[pipeline_slot_index];
    cuda_stream = pipeline_slot->cuda_stream;
    if (cuda_stream == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    status = SparkGlm52Sm121RequiredDecodeStageLaunch(
        node_context,
        pipeline_slot,
        pipeline_slot_index,
        active_sequence_count,
        runtime_kv_block_table,
        cuda_stream);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    if (getenv("GLM52_STAGE_SLICE_DEBUG_SYNC") != 0)
    {
        cuda_status = cudaStreamSynchronize((cudaStream_t)cuda_stream);
        if (cuda_status != cudaSuccess)
        {
            fprintf(
                stderr,
                "spark_glm52_stage_slice_debug_sync_failed error=%d message=%s\n",
                (int)cuda_status,
                cudaGetErrorString(cuda_status));
            return SPARK_STATUS_INTERNAL_ERROR;
        }
    }

    cuda_status = cudaLaunchHostFunc(
        (cudaStream_t)cuda_stream,
        SparkResidentDecodeStageCudaCompletion,
        completion);
    if (cuda_status != cudaSuccess)
    {
        cudaStreamSynchronize((cudaStream_t)cuda_stream);
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    return SPARK_STATUS_OK;
}


extern "C" SparkStatus SparkResidentDecodeStageBackendSubmitStageSlice(
    const SparkResidentDecodeStageStageSlicePlan *stage_slice_plan,
    const SparkResidentDecodeStageNodeContext *const *layer_node_contexts,
    uint32_t layer_count,
    uint32_t pipeline_slot_index,
    uint32_t active_sequence_count,
    uint32_t final_token_stage,
    const SparkKvBlockTableView *runtime_kv_block_table,
    const SparkResidentDecodeStageFrameContext *frame_context,
    SparkResidentDecodeStageBackendCompletion *completion)
{
    const SparkResidentDecodeStageNodeContext *first_node_context;
    const SparkResidentDecodeStageNodeContext *completion_node_context;
    const SparkResidentDecodeStagePipelineSlot *pipeline_slot;
    const SparkResidentDecodeStagePipelineSlot *completion_pipeline_slot;
    void *cuda_stream;
    void *launch_completion;
    SparkStatus status;
    cudaError_t cuda_status;

    if (layer_node_contexts == 0 ||
        layer_count == 0u ||
        layer_count >
            SPARK_RESIDENT_DECODE_STAGE_MAX_STAGE_SLICE_LAYER_COUNT ||
        completion == 0 ||
        completion->function == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    first_node_context = layer_node_contexts[0];
    if (first_node_context == 0 ||
        first_node_context->pipeline_slots == 0 ||
        pipeline_slot_index >= first_node_context->pipeline_slot_count ||
        active_sequence_count == 0u ||
        active_sequence_count > first_node_context->max_active_sequence_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    pipeline_slot = &first_node_context->pipeline_slots[pipeline_slot_index];
    completion_node_context = final_token_stage != 0u
        ? layer_node_contexts[layer_count - 1u]
        : first_node_context;
    if (completion_node_context == 0 ||
        completion_node_context->pipeline_slots == 0 ||
        pipeline_slot_index >= completion_node_context->pipeline_slot_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    completion_pipeline_slot =
        &completion_node_context->pipeline_slots[pipeline_slot_index];
    cuda_stream = pipeline_slot->cuda_stream;
    if (cuda_stream == 0 ||
        completion_pipeline_slot->cuda_stream != cuda_stream)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    launch_completion = 0;

    status = SparkGlm52Sm121RequiredDecodeStageLaunchStageSlice(
        stage_slice_plan,
        layer_node_contexts,
        layer_count,
        pipeline_slot_index,
        active_sequence_count,
        final_token_stage,
        runtime_kv_block_table,
        frame_context,
        cuda_stream,
        launch_completion);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    if (getenv("GLM52_STAGE_SLICE_DEBUG_SYNC") != 0)
    {
        cuda_status = cudaStreamSynchronize((cudaStream_t)cuda_stream);
        if (cuda_status != cudaSuccess)
        {
            fprintf(
                stderr,
                "spark_glm52_stage_slice_debug_sync_failed error=%d message=%s\n",
                (int)cuda_status,
                cudaGetErrorString(cuda_status));
            return SPARK_STATUS_INTERNAL_ERROR;
        }
    }

    if (final_token_stage != 0u)
    {
        status = SparkResidentDecodeStageCudaCopyFinalTokens(
            completion_pipeline_slot,
            final_token_stage,
            active_sequence_count,
            completion,
            (cudaStream_t)cuda_stream);
        if (status != SPARK_STATUS_OK)
        {
            cudaStreamSynchronize((cudaStream_t)cuda_stream);
            return status;
        }
    }

    cuda_status = cudaLaunchHostFunc(
        (cudaStream_t)cuda_stream,
        SparkResidentDecodeStageCudaCompletion,
        completion);
    if (cuda_status != cudaSuccess)
    {
        cudaStreamSynchronize((cudaStream_t)cuda_stream);
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    return SPARK_STATUS_OK;
}

extern "C" SparkStatus SparkResidentDecodeStageBackendSubmitBulkPrefill(
    const SparkResidentDecodeStageNodeContext *node_context,
    uint32_t pipeline_slot_index,
    uint32_t active_sequence_count,
    uint32_t prompt_token_offset,
    uint32_t prompt_token_count,
    const SparkKvBlockTableView *runtime_kv_block_table,
    const SparkResidentDecodeStagePrefillFrameView *prefill_frame_view,
    SparkResidentDecodeStageBackendCompletion *completion)
{
    const SparkResidentDecodeStagePipelineSlot *pipeline_slot;
    void *cuda_stream;
    SparkStatus status;
    cudaError_t cuda_status;

    if (node_context == 0 || completion == 0 || completion->function == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (node_context->pipeline_slots == 0 ||
        pipeline_slot_index >= node_context->pipeline_slot_count ||
        active_sequence_count == 0u ||
        active_sequence_count > node_context->max_active_sequence_count ||
        prompt_token_count == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    pipeline_slot = &node_context->pipeline_slots[pipeline_slot_index];
    cuda_stream = pipeline_slot->cuda_stream;
    if (cuda_stream == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    status = SparkGlm52Sm121RequiredDecodeStageLaunchBulkPrefill(
        node_context,
        pipeline_slot,
        pipeline_slot_index,
        active_sequence_count,
        prompt_token_offset,
        prompt_token_count,
        runtime_kv_block_table,
        prefill_frame_view,
        cuda_stream);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    cuda_status = cudaLaunchHostFunc(
        (cudaStream_t)cuda_stream,
        SparkResidentDecodeStageCudaCompletion,
        completion);
    if (cuda_status != cudaSuccess)
    {
        cudaStreamSynchronize((cudaStream_t)cuda_stream);
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    return SPARK_STATUS_OK;
}

extern "C" SparkStatus SparkResidentDecodeStageBackendSubmitStageSliceBulkPrefill(
    const SparkResidentDecodeStageNodeContext *const *layer_node_contexts,
    uint32_t layer_count,
    uint32_t pipeline_slot_index,
    uint32_t active_sequence_count,
    uint32_t prompt_token_offset,
    uint32_t prompt_token_count,
    const SparkKvBlockTableView *runtime_kv_block_table,
    const SparkResidentDecodeStagePrefillFrameView *prefill_frame_view,
    SparkResidentDecodeStageBackendCompletion *completion)
{
    const SparkResidentDecodeStageNodeContext *first_node_context;
    const SparkResidentDecodeStageNodeContext *layer_node_context;
    const SparkResidentDecodeStagePipelineSlot *pipeline_slot;
    void *cuda_stream;
    uint32_t layer_index;
    SparkStatus status;
    cudaError_t cuda_status;

    if (layer_node_contexts == 0 ||
        layer_count == 0u ||
        layer_count >
            SPARK_RESIDENT_DECODE_STAGE_MAX_STAGE_SLICE_LAYER_COUNT ||
        completion == 0 ||
        completion->function == 0 ||
        prompt_token_count == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    first_node_context = layer_node_contexts[0];
    if (first_node_context == 0 ||
        first_node_context->pipeline_slots == 0 ||
        pipeline_slot_index >= first_node_context->pipeline_slot_count ||
        active_sequence_count == 0u ||
        active_sequence_count > first_node_context->max_active_sequence_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    pipeline_slot = &first_node_context->pipeline_slots[pipeline_slot_index];
    cuda_stream = pipeline_slot->cuda_stream;
    if (cuda_stream == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    for (layer_index = 0u; layer_index < layer_count; ++layer_index)
    {
        layer_node_context = layer_node_contexts[layer_index];
        if (layer_node_context == 0 ||
            layer_node_context->pipeline_slots == 0 ||
            pipeline_slot_index >= layer_node_context->pipeline_slot_count ||
            active_sequence_count > layer_node_context->max_active_sequence_count ||
            layer_node_context->pipeline_slots[pipeline_slot_index].cuda_stream !=
                cuda_stream ||
            layer_node_context->bulk_prefill_plan == 0 ||
            prompt_token_count >
                layer_node_context->bulk_prefill_plan->maximum_prompt_token_count)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
    }

    (void)layer_node_context;
    status = SparkGlm52Sm121RequiredDecodeStageLaunchStageSliceBulkPrefill(
        layer_node_contexts,
        layer_count,
        pipeline_slot_index,
        active_sequence_count,
        prompt_token_offset,
        prompt_token_count,
        runtime_kv_block_table,
        prefill_frame_view,
        cuda_stream);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    cuda_status = cudaLaunchHostFunc(
        (cudaStream_t)cuda_stream,
        SparkResidentDecodeStageCudaCompletion,
        completion);
    if (cuda_status != cudaSuccess)
    {
        cudaStreamSynchronize((cudaStream_t)cuda_stream);
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    return SPARK_STATUS_OK;
}

extern "C" void SparkResidentDecodeStageBackendQuiesce(
    const SparkResidentDecodeStageNodeContext *node_context)
{
    uint32_t pipeline_slot_index;

    if (node_context == 0 || node_context->pipeline_slots == 0)
    {
        return;
    }

    SparkGlm52Sm121RequiredDecodeStageQuiesce(node_context);
    for (pipeline_slot_index = 0u;
         pipeline_slot_index < node_context->pipeline_slot_count;
         ++pipeline_slot_index)
    {
        if (node_context->pipeline_slots[pipeline_slot_index].cuda_stream != 0)
        {
            cudaStreamSynchronize((cudaStream_t)
                node_context->pipeline_slots[pipeline_slot_index].cuda_stream);
        }
    }
}
