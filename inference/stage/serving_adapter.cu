#include "sparkpipe/spark_glm52_resident_decode_stage_serving_adapter.h"

#include "sparkpipe/spark_glm52_resident_decode_stage_required_cuda.h"

#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

#define SPARK_SERVING_ADAPTER_THREADS 256u
#define SPARK_SERVING_ADAPTER_INVALID_SLOT UINT32_MAX

/* Defined in inference/llms/glm5_2/bind.cu. Declared rather than included
   because bind.cu pulls the whole kernel library and this file needs two
   symbols from it - an include here would make every edit to a kernel recompile
   the serving adapter. */
extern "C" int32_t Glm52StageSlicePrefill(const void *node_context, void *layer_buffers,
    uint32_t first_layer, uint32_t layer_count, uint32_t rows, uint32_t packed_rows,
    uint32_t context, uint32_t multiprocessors, const uint32_t *row_positions, void *stream);

extern "C" int32_t Glm52StageSlice(const void *node_context, void *layer_buffers,
    uint32_t first_layer, uint32_t layer_count, uint32_t rows,
    uint32_t packed_rows, uint32_t context, uint32_t multiprocessors,
    void *stream);

/* Queried once from the device rather than assumed. The ring size changes. */
static uint32_t SparkResidentDecodeStageMultiprocessorCount(void)
{
    static uint32_t cached = 0u;
    int count = 0;
    int device = 0;

    if (cached != 0u)
    {
        return cached;
    }
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device) !=
            cudaSuccess ||
        count <= 0)
    {
        return 0u;
    }
    cached = (uint32_t)count;
    return cached;
}

static SparkStatus SparkServingAdapterCudaStatus(cudaError_t status)
{
    if (status == cudaSuccess)
    {
        return SPARK_STATUS_OK;
    }
    return SPARK_STATUS_DRIVER_LOAD_ERROR;
}

static SparkStatus SparkServingAdapterDeviceAlloc(void **pointer,uint64_t bytes)
{
    if (pointer == 0 || bytes == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *pointer = 0;
    return SparkServingAdapterCudaStatus(cudaMalloc(pointer,(size_t)bytes));
}

static SparkStatus SparkServingAdapterHostAlloc(uint32_t **pointer,uint64_t count)
{
    if (pointer == 0 || count == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *pointer = 0;
    return SparkServingAdapterCudaStatus(
        cudaHostAlloc((void **)pointer,(size_t)(count * sizeof(uint32_t)),cudaHostAllocDefault));
}

static void SparkServingAdapterFreeDevice(void *pointer)
{
    if (pointer != 0)
    {
        (void)cudaFree(pointer);
    }
}

static void SparkServingAdapterFreeHost(void *pointer)
{
    if (pointer != 0)
    {
        (void)cudaFreeHost(pointer);
    }
}

static const SparkResidentDecodeStagePipelineSlot *SparkServingAdapterPipelineSlot(
    const SparkResidentDecodeStageServingAdapter *adapter)
{
    const SparkResidentDecodeStageNodeContext *node_context;

    if (adapter == 0 || adapter->layer_node_contexts == 0 ||
        adapter->layer_count == 0u)
    {
        return 0;
    }
    node_context = adapter->layer_node_contexts[0u];
    if (node_context == 0 ||
        node_context->pipeline_slots == 0 ||
        adapter->pipeline_slot_index >= node_context->pipeline_slot_count)
    {
        return 0;
    }
    return &node_context->pipeline_slots[adapter->pipeline_slot_index];
}

__global__ static void SparkGlm52ServingAdapterBuildPrefillMetadataKernel(
    const uint32_t *__restrict__ lane_offsets,
    const uint32_t *__restrict__ lane_counts,
    const uint32_t *__restrict__ block_table,
    const uint32_t *__restrict__ lane_block_counts,
    uint32_t lane_stride,
    uint32_t block_token_count,
    uint32_t lane_count,
    uint32_t token_stride,
    uint32_t *__restrict__ positions,
    uint32_t *__restrict__ slot_mapping,
    uint32_t *__restrict__ context_lengths,
    uint32_t *__restrict__ first_block_token_offsets,
    uint32_t *__restrict__ token_counts)
{
    uint32_t global_index;
    uint32_t lane_index;
    uint32_t token_index;
    uint32_t position;
    uint32_t block_index;
    uint32_t in_block_index;
    uint32_t physical_block_index;

    global_index = (uint32_t)(blockIdx.x * blockDim.x + threadIdx.x);
    if (global_index >= lane_count * token_stride)
    {
        return;
    }
    lane_index = global_index / token_stride;
    token_index = global_index - (lane_index * token_stride);
    if (token_index == 0u)
    {
        context_lengths[lane_index] =
            lane_offsets[lane_index] + lane_counts[lane_index];
        first_block_token_offsets[lane_index] =
            lane_offsets[lane_index] % block_token_count;
        token_counts[lane_index] = lane_counts[lane_index];
    }
    if (token_index >= lane_counts[lane_index])
    {
        return;
    }
    position = lane_offsets[lane_index] + token_index;
    block_index = position / block_token_count;
    in_block_index = position - (block_index * block_token_count);
    if (block_index >= lane_block_counts[lane_index])
    {
        positions[global_index] = position;
        slot_mapping[global_index] = SPARK_SERVING_ADAPTER_INVALID_SLOT;
        return;
    }
    physical_block_index = block_table[(lane_index * lane_stride) + block_index];
    positions[global_index] = position;
    slot_mapping[global_index] =
        (physical_block_index * block_token_count) + in_block_index;
}

__global__ static void SparkGlm52ServingAdapterGatherPrefillEmbeddingKernel(
    const uint32_t *__restrict__ token_ids,
    const uint32_t *__restrict__ lane_counts,
    const uint32_t *__restrict__ embedding_bf16_words,
    uint32_t *__restrict__ output_bf16_words,
    uint32_t lane_count,
    uint32_t token_stride,
    uint32_t hidden_words)
{
    uint64_t word_index;
    uint64_t route_index;
    uint32_t lane_index;
    uint32_t token_index;
    uint32_t hidden_word_index;
    uint32_t token_id;

    word_index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (word_index >= (uint64_t)lane_count * token_stride * hidden_words)
    {
        return;
    }
    route_index = word_index / hidden_words;
    hidden_word_index = (uint32_t)(word_index - (route_index * hidden_words));
    lane_index = (uint32_t)(route_index / token_stride);
    token_index = (uint32_t)(route_index - ((uint64_t)lane_index * token_stride));
    if (token_index >= lane_counts[lane_index])
    {
        return;
    }
    token_id = token_ids[(lane_index * token_stride) + token_index];
    output_bf16_words[word_index] =
        embedding_bf16_words[((uint64_t)token_id * hidden_words) + hidden_word_index];
}

__global__ static void SparkGlm52ServingAdapterBuildDecodeMetadataKernel(
    const uint32_t *__restrict__ decode_positions,
    const uint32_t *__restrict__ block_table,
    const uint32_t *__restrict__ lane_block_counts,
    uint32_t lane_stride,
    uint32_t block_token_count,
    uint32_t lane_count,
    uint32_t *__restrict__ positions,
    uint32_t *__restrict__ slot_mapping,
    uint32_t *__restrict__ context_lengths,
    uint32_t *__restrict__ first_block_token_offsets)
{
    uint32_t lane_index;
    uint32_t position;
    uint32_t block_index;
    uint32_t in_block_index;
    uint32_t physical_block_index;

    lane_index = (uint32_t)(blockIdx.x * blockDim.x + threadIdx.x);
    if (lane_index >= lane_count)
    {
        return;
    }
    position = decode_positions[lane_index];
    block_index = position / block_token_count;
    in_block_index = position - (block_index * block_token_count);
    positions[lane_index] = position;
    context_lengths[lane_index] = position + 1u;
    first_block_token_offsets[lane_index] = position % block_token_count;
    if (block_index >= lane_block_counts[lane_index])
    {
        slot_mapping[lane_index] = SPARK_SERVING_ADAPTER_INVALID_SLOT;
        return;
    }
    physical_block_index = block_table[(lane_index * lane_stride) + block_index];
    slot_mapping[lane_index] =
        (physical_block_index * block_token_count) + in_block_index;
}

__global__ static void SparkGlm52ServingAdapterGatherDecodeEmbeddingKernel(
    const uint32_t *__restrict__ token_ids,
    const uint32_t *__restrict__ embedding_bf16_words,
    uint32_t *__restrict__ output_bf16_words,
    uint32_t lane_count,
    uint32_t hidden_words)
{
    uint64_t word_index;
    uint32_t lane_index;
    uint32_t hidden_word_index;
    uint32_t token_id;

    word_index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (word_index >= (uint64_t)lane_count * hidden_words)
    {
        return;
    }
    lane_index = (uint32_t)(word_index / hidden_words);
    hidden_word_index = (uint32_t)(word_index - ((uint64_t)lane_index * hidden_words));
    token_id = token_ids[lane_index];
    output_bf16_words[word_index] =
        embedding_bf16_words[((uint64_t)token_id * hidden_words) + hidden_word_index];
}

static SparkStatus SparkGlm52ServingAdapterValidateKvCoverage(
    const SparkKvBlockTableView *kv_block_table,
    uint32_t lane_index,
    uint32_t position)
{
    uint32_t block_index;
    uint32_t physical_block_index;

    if (kv_block_table == 0 ||
        kv_block_table->abi_version != SPARK_KV_CACHE_ABI_VERSION ||
        kv_block_table->descriptor_bytes !=
            SPARK_KV_BLOCK_TABLE_VIEW_DESCRIPTOR_BYTES ||
        kv_block_table->block_token_count == 0u ||
        kv_block_table->physical_block_indices == 0 ||
        kv_block_table->host_physical_block_indices == 0 ||
        kv_block_table->lane_physical_block_counts == 0 ||
        kv_block_table->host_lane_physical_block_counts == 0 ||
        lane_index >= kv_block_table->lane_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    block_index = position / kv_block_table->block_token_count;
    if (block_index >= kv_block_table->host_lane_physical_block_counts[lane_index] ||
        block_index >= kv_block_table->lane_stride)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    physical_block_index =
        kv_block_table->host_physical_block_indices[
            ((uint64_t)lane_index * kv_block_table->lane_stride) + block_index];
    if (physical_block_index == SPARK_SERVING_ADAPTER_INVALID_SLOT)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkResidentDecodeStageServingAdapterInitialize(
    SparkResidentDecodeStageServingAdapter *adapter,
    const SparkResidentDecodeStageServingAdapterConfiguration *configuration)
{
    uint64_t metadata_token_count;
    uint64_t hidden_bytes;
    SparkStatus status;

    if (adapter == 0 || configuration == 0 ||
        configuration->abi_version !=
            SPARK_RESIDENT_DECODE_STAGE_SERVING_ADAPTER_ABI_VERSION ||
        configuration->descriptor_bytes !=
            SPARK_RESIDENT_DECODE_STAGE_SERVING_ADAPTER_CONFIGURATION_DESCRIPTOR_BYTES ||
        (configuration->flags &
            ~SPARK_RESIDENT_DECODE_STAGE_SERVING_ADAPTER_FLAG_KNOWN_FLAGS) != 0u ||
        configuration->maximum_active_sequence_count == 0u ||
        configuration->maximum_prompt_token_count == 0u ||
        configuration->vocabulary_size == 0u ||
        configuration->hidden_dimension !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION ||
        (configuration->hidden_dimension & 1u) != 0u ||
        configuration->stage_slice_plan == 0 ||
        configuration->layer_node_contexts == 0 ||
        configuration->layer_count == 0u ||
        configuration->embedding_weight_bf16 == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    memset(adapter,0,sizeof(*adapter));
    adapter->abi_version =
        SPARK_RESIDENT_DECODE_STAGE_SERVING_ADAPTER_ABI_VERSION;
    adapter->descriptor_bytes =
        SPARK_RESIDENT_DECODE_STAGE_SERVING_ADAPTER_DESCRIPTOR_BYTES;
    adapter->flags = configuration->flags;
    adapter->pipeline_slot_index = configuration->pipeline_slot_index;
    adapter->maximum_active_sequence_count =
        configuration->maximum_active_sequence_count;
    adapter->maximum_prompt_token_count =
        configuration->maximum_prompt_token_count;
    adapter->vocabulary_size = configuration->vocabulary_size;
    adapter->hidden_dimension = configuration->hidden_dimension;
    adapter->final_token_stage = configuration->final_token_stage;
    adapter->layer_count = configuration->layer_count;
    adapter->stage_slice_plan = configuration->stage_slice_plan;
    adapter->layer_node_contexts = configuration->layer_node_contexts;
    adapter->embedding_weight_bf16 = configuration->embedding_weight_bf16;
    adapter->cuda_stream = configuration->cuda_stream;
    if (adapter->cuda_stream == 0)
    {
        status = SparkServingAdapterCudaStatus(
            cudaStreamCreateWithFlags(
                (cudaStream_t *)&adapter->cuda_stream,
                cudaStreamNonBlocking));
        if (status != SPARK_STATUS_OK)
        {
            SparkResidentDecodeStageServingAdapterDestroy(adapter);
            return status;
        }
        adapter->owns_cuda_stream = 1u;
    }

    metadata_token_count =
        (uint64_t)adapter->maximum_active_sequence_count *
        adapter->maximum_prompt_token_count;
    hidden_bytes =
        metadata_token_count *
        adapter->hidden_dimension *
        sizeof(uint16_t);
    status = SparkServingAdapterHostAlloc(
        &adapter->host_lane_offsets,
        adapter->maximum_active_sequence_count);
    if (status == SPARK_STATUS_OK)
        status = SparkServingAdapterHostAlloc(
            &adapter->host_lane_counts,
            adapter->maximum_active_sequence_count);
    if (status == SPARK_STATUS_OK)
        status = SparkServingAdapterHostAlloc(
            &adapter->host_decode_positions,
            adapter->maximum_active_sequence_count);
    if (status == SPARK_STATUS_OK)
        status = SparkServingAdapterHostAlloc(
            &adapter->host_decode_token_ids,
            adapter->maximum_active_sequence_count);
    if (status == SPARK_STATUS_OK)
        status = SparkServingAdapterHostAlloc(
            &adapter->host_mtp_draft_token_budgets,
            adapter->maximum_active_sequence_count);
    if (status == SPARK_STATUS_OK)
        status = SparkServingAdapterHostAlloc(
            &adapter->host_mtp_committed_token_ids,
            adapter->maximum_active_sequence_count *
                SPARK_RESIDENT_DECODE_STAGE_MAX_SPECULATIVE_ROWS_PER_LANE);
    if (status == SPARK_STATUS_OK)
        status = SparkServingAdapterDeviceAlloc(
            (void **)&adapter->device_prefill_token_ids,
            metadata_token_count * sizeof(uint32_t));
    if (status == SPARK_STATUS_OK)
        status = SparkServingAdapterDeviceAlloc(
            (void **)&adapter->device_lane_offsets,
            adapter->maximum_active_sequence_count * sizeof(uint32_t));
    if (status == SPARK_STATUS_OK)
        status = SparkServingAdapterDeviceAlloc(
            (void **)&adapter->device_lane_counts,
            adapter->maximum_active_sequence_count * sizeof(uint32_t));
    if (status == SPARK_STATUS_OK)
        status = SparkServingAdapterDeviceAlloc(
            (void **)&adapter->device_decode_positions,
            adapter->maximum_active_sequence_count * sizeof(uint32_t));
    if (status == SPARK_STATUS_OK)
        status = SparkServingAdapterDeviceAlloc(
            (void **)&adapter->device_decode_token_ids,
            adapter->maximum_active_sequence_count * sizeof(uint32_t));
    if (status == SPARK_STATUS_OK)
        status = SparkServingAdapterDeviceAlloc(
            (void **)&adapter->device_mtp_draft_token_budgets,
            adapter->maximum_active_sequence_count * sizeof(uint32_t));
    if (status == SPARK_STATUS_OK)
        status = SparkServingAdapterDeviceAlloc(
            (void **)&adapter->device_prompt_positions,
            metadata_token_count * sizeof(uint32_t));
    if (status == SPARK_STATUS_OK)
        status = SparkServingAdapterDeviceAlloc(
            (void **)&adapter->device_prompt_slot_mapping,
            metadata_token_count * sizeof(uint32_t));
    if (status == SPARK_STATUS_OK)
        status = SparkServingAdapterDeviceAlloc(
            (void **)&adapter->device_prompt_context_lengths,
            adapter->maximum_active_sequence_count * sizeof(uint32_t));
    if (status == SPARK_STATUS_OK)
        status = SparkServingAdapterDeviceAlloc(
            (void **)&adapter->device_prompt_first_block_token_offsets,
            adapter->maximum_active_sequence_count * sizeof(uint32_t));
    if (status == SPARK_STATUS_OK)
        status = SparkServingAdapterDeviceAlloc(
            (void **)&adapter->device_prompt_token_counts,
            adapter->maximum_active_sequence_count * sizeof(uint32_t));
    if (status == SPARK_STATUS_OK)
        status = SparkServingAdapterDeviceAlloc(
            &adapter->device_prompt_hidden_bf16,
            hidden_bytes);
    if (status == SPARK_STATUS_OK)
        status = SparkServingAdapterDeviceAlloc(
            &adapter->device_prompt_output_hidden_bf16,
            hidden_bytes);
    if (status != SPARK_STATUS_OK)
    {
        SparkResidentDecodeStageServingAdapterDestroy(adapter);
        return status;
    }
    if (SparkServingAdapterPipelineSlot(adapter) == 0)
    {
        SparkResidentDecodeStageServingAdapterDestroy(adapter);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

void SparkResidentDecodeStageServingAdapterDestroy(
    SparkResidentDecodeStageServingAdapter *adapter)
{
    if (adapter == 0)
    {
        return;
    }
    SparkServingAdapterFreeDevice(adapter->device_prefill_token_ids);
    SparkServingAdapterFreeDevice(adapter->device_lane_offsets);
    SparkServingAdapterFreeDevice(adapter->device_lane_counts);
    SparkServingAdapterFreeDevice(adapter->device_decode_positions);
    SparkServingAdapterFreeDevice(adapter->device_decode_token_ids);
    SparkServingAdapterFreeDevice(adapter->device_mtp_draft_token_budgets);
    SparkServingAdapterFreeDevice(adapter->device_prompt_positions);
    SparkServingAdapterFreeDevice(adapter->device_prompt_slot_mapping);
    SparkServingAdapterFreeDevice(adapter->device_prompt_context_lengths);
    SparkServingAdapterFreeDevice(adapter->device_prompt_first_block_token_offsets);
    SparkServingAdapterFreeDevice(adapter->device_prompt_token_counts);
    SparkServingAdapterFreeDevice(adapter->device_prompt_hidden_bf16);
    SparkServingAdapterFreeDevice(adapter->device_prompt_output_hidden_bf16);
    SparkServingAdapterFreeHost(adapter->host_lane_offsets);
    SparkServingAdapterFreeHost(adapter->host_lane_counts);
    SparkServingAdapterFreeHost(adapter->host_decode_positions);
    SparkServingAdapterFreeHost(adapter->host_decode_token_ids);
    SparkServingAdapterFreeHost(adapter->host_mtp_draft_token_budgets);
    SparkServingAdapterFreeHost(adapter->host_mtp_committed_token_ids);
    if (adapter->owns_cuda_stream != 0u && adapter->cuda_stream != 0)
    {
        (void)cudaStreamDestroy((cudaStream_t)adapter->cuda_stream);
    }
    memset(adapter,0,sizeof(*adapter));
}

SparkStatus SparkResidentDecodeStageServingAdapterPrefill(
    void *context,
    const SparkPromptPipelinePrefillDispatch *prefill_dispatch)
{
    SparkResidentDecodeStageServingAdapter *adapter;
    SparkResidentDecodeStagePrefillFrameView prefill_view;
    const SparkRequestApiPrefillDispatchView *request_view;
    cudaStream_t stream;
    uint32_t lane_index;
    uint64_t word_count;
    uint32_t block_count;
    SparkStatus status;

    adapter = (SparkResidentDecodeStageServingAdapter *)context;
    if (adapter == 0 || prefill_dispatch == 0 ||
        prefill_dispatch->abi_version != SPARK_PROMPT_PIPELINE_ABI_VERSION ||
        prefill_dispatch->descriptor_bytes !=
            SPARK_PROMPT_PIPELINE_PREFILL_DISPATCH_DESCRIPTOR_BYTES ||
        prefill_dispatch->prefill_view == 0 ||
        prefill_dispatch->host_token_ids == 0 ||
        prefill_dispatch->kv_block_table_view == 0 ||
        prefill_dispatch->lane_count == 0u ||
        prefill_dispatch->lane_count > adapter->maximum_active_sequence_count ||
        prefill_dispatch->prompt_token_count == 0u ||
        prefill_dispatch->prompt_token_count > adapter->maximum_prompt_token_count ||
        prefill_dispatch->host_token_stride < prefill_dispatch->prompt_token_stride)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    request_view = prefill_dispatch->prefill_view;
    for (lane_index = 0u; lane_index < request_view->lane_count; ++lane_index)
    {
        const SparkRequestApiPrefillDispatchLaneView *lane;
        uint32_t token_index;
        uint32_t last_position;

        lane = &request_view->lanes[lane_index];
        if (lane->prompt_token_count == 0u ||
            lane->prompt_token_count > adapter->maximum_prompt_token_count)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        last_position =
            lane->prompt_token_offset + lane->prompt_token_count - 1u;
        status = SparkGlm52ServingAdapterValidateKvCoverage(
            prefill_dispatch->kv_block_table_view,
            lane_index,
            last_position);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        adapter->host_lane_offsets[lane_index] = lane->prompt_token_offset;
        adapter->host_lane_counts[lane_index] = lane->prompt_token_count;
        for (token_index = 0u; token_index < lane->prompt_token_count; ++token_index)
        {
            uint32_t token_id;

            token_id =
                prefill_dispatch->host_token_ids[
                    ((uint64_t)lane_index * prefill_dispatch->host_token_stride) +
                    token_index];
            if (token_id >= adapter->vocabulary_size)
            {
                return SPARK_STATUS_INVALID_ARGUMENT;
            }
        }
    }

    stream = (cudaStream_t)adapter->cuda_stream;
    status = SparkServingAdapterCudaStatus(
        cudaMemcpy2DAsync(
            adapter->device_prefill_token_ids,
            (size_t)adapter->maximum_prompt_token_count * sizeof(uint32_t),
            prefill_dispatch->host_token_ids,
            (size_t)prefill_dispatch->host_token_stride * sizeof(uint32_t),
            (size_t)prefill_dispatch->prompt_token_stride * sizeof(uint32_t),
            (size_t)prefill_dispatch->lane_count,
            cudaMemcpyHostToDevice,
            stream));
    if (status == SPARK_STATUS_OK)
        status = SparkServingAdapterCudaStatus(
            cudaMemcpyAsync(
                adapter->device_lane_offsets,
                adapter->host_lane_offsets,
                (size_t)prefill_dispatch->lane_count * sizeof(uint32_t),
                cudaMemcpyHostToDevice,
                stream));
    if (status == SPARK_STATUS_OK)
        status = SparkServingAdapterCudaStatus(
            cudaMemcpyAsync(
                adapter->device_lane_counts,
                adapter->host_lane_counts,
                (size_t)prefill_dispatch->lane_count * sizeof(uint32_t),
                cudaMemcpyHostToDevice,
                stream));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    block_count =
        (prefill_dispatch->lane_count * adapter->maximum_prompt_token_count +
         SPARK_SERVING_ADAPTER_THREADS - 1u) /
        SPARK_SERVING_ADAPTER_THREADS;
    SparkGlm52ServingAdapterBuildPrefillMetadataKernel<<<
        block_count,
        SPARK_SERVING_ADAPTER_THREADS,
        0,
        stream>>>(
            adapter->device_lane_offsets,
            adapter->device_lane_counts,
            prefill_dispatch->kv_block_table_view->physical_block_indices,
            prefill_dispatch->kv_block_table_view->lane_physical_block_counts,
            prefill_dispatch->kv_block_table_view->lane_stride,
            prefill_dispatch->kv_block_table_view->block_token_count,
            prefill_dispatch->lane_count,
            adapter->maximum_prompt_token_count,
            adapter->device_prompt_positions,
            adapter->device_prompt_slot_mapping,
            adapter->device_prompt_context_lengths,
            adapter->device_prompt_first_block_token_offsets,
            adapter->device_prompt_token_counts);
    status = SparkServingAdapterCudaStatus(cudaGetLastError());
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    word_count =
        (uint64_t)prefill_dispatch->lane_count *
        adapter->maximum_prompt_token_count *
        (adapter->hidden_dimension / 2u);
    block_count = (uint32_t)(
        (word_count + SPARK_SERVING_ADAPTER_THREADS - 1u) /
        SPARK_SERVING_ADAPTER_THREADS);
    SparkGlm52ServingAdapterGatherPrefillEmbeddingKernel<<<
        block_count,
        SPARK_SERVING_ADAPTER_THREADS,
        0,
        stream>>>(
            adapter->device_prefill_token_ids,
            adapter->device_lane_counts,
            (const uint32_t *)adapter->embedding_weight_bf16,
            (uint32_t *)adapter->device_prompt_hidden_bf16,
            prefill_dispatch->lane_count,
            adapter->maximum_prompt_token_count,
            adapter->hidden_dimension / 2u);
    status = SparkServingAdapterCudaStatus(cudaGetLastError());
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    memset(&prefill_view,0,sizeof(prefill_view));
    prefill_view.abi_version =
        SPARK_RESIDENT_DECODE_STAGE_PREFILL_FRAME_VIEW_ABI_VERSION;
    prefill_view.descriptor_bytes =
        SPARK_RESIDENT_DECODE_STAGE_PREFILL_FRAME_VIEW_DESCRIPTOR_BYTES;
    prefill_view.active_sequence_count = prefill_dispatch->lane_count;
    prefill_view.prompt_token_offset = prefill_dispatch->prompt_token_offset;
    prefill_view.prompt_token_count = prefill_dispatch->prompt_token_count;
    prefill_view.prompt_token_stride = adapter->maximum_prompt_token_count;
    prefill_view.hidden_dimension = adapter->hidden_dimension;
    prefill_view.prompt_positions = adapter->device_prompt_positions;
    prefill_view.prompt_slot_mapping = adapter->device_prompt_slot_mapping;
    prefill_view.prompt_context_lengths = adapter->device_prompt_context_lengths;
    prefill_view.prompt_first_block_token_offsets =
        adapter->device_prompt_first_block_token_offsets;
    prefill_view.prompt_token_counts = adapter->device_prompt_token_counts;
    prefill_view.prompt_hidden_bf16 = adapter->device_prompt_hidden_bf16;
    prefill_view.prompt_output_hidden_bf16 =
        adapter->device_prompt_output_hidden_bf16;

    status = SparkGlm52Sm121RequiredDecodeStageLaunchStageSliceBulkPrefill(
        adapter->layer_node_contexts,
        adapter->layer_count,
        adapter->pipeline_slot_index,
        prefill_dispatch->lane_count,
        prefill_dispatch->prompt_token_offset,
        prefill_dispatch->prompt_token_count,
        prefill_dispatch->kv_block_table_view,
        &prefill_view,
        stream);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if ((adapter->flags &
            SPARK_RESIDENT_DECODE_STAGE_SERVING_ADAPTER_FLAG_SYNCHRONIZE_AFTER_LAUNCH) != 0u)
    {
        return SparkServingAdapterCudaStatus(cudaStreamSynchronize(stream));
    }
    return SPARK_STATUS_OK;
}


static uint32_t SparkServingAdapterDecodeDispatchIsMtpVerify(
    const SparkServingDecodeDispatch *decode_dispatch)
{
    return decode_dispatch != 0 &&
        decode_dispatch->dispatch_kind ==
            SPARK_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH &&
        decode_dispatch->request_dispatch != 0 &&
        (decode_dispatch->request_dispatch->flags &
            SPARK_REQUEST_API_DISPATCH_FLAG_MTP_SPECULATIVE_VERIFY) != 0u;
}

static SparkStatus SparkServingAdapterUploadMtpDraftBudgets(
    SparkResidentDecodeStageServingAdapter *adapter,
    uint32_t active_sequence_count,
    uint32_t mtp_budget,
    cudaStream_t stream)
{
    uint32_t lane_index;

    if (adapter == 0 || active_sequence_count == 0u ||
        active_sequence_count > adapter->maximum_active_sequence_count ||
        mtp_budget > SPARK_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    for (lane_index = 0u;
         lane_index < active_sequence_count;
         ++lane_index)
    {
        adapter->host_mtp_draft_token_budgets[lane_index] = mtp_budget;
    }

    return SparkServingAdapterCudaStatus(
        cudaMemcpyAsync(
            adapter->device_mtp_draft_token_budgets,
            adapter->host_mtp_draft_token_budgets,
            (size_t)active_sequence_count * sizeof(uint32_t),
            cudaMemcpyHostToDevice,
            stream));
}

static SparkStatus SparkGlm52ServingAdapterLaunchDecodeStep(
    SparkResidentDecodeStageServingAdapter *adapter,
    const SparkServingDecodeDispatch *decode_dispatch,
    const SparkResidentDecodeStagePipelineSlot *pipeline_slot,
    const SparkResidentDecodeStageFrameContext *frame_context,
    uint32_t step_index,
    cudaStream_t stream)
{
    uint32_t lane_index;
    uint64_t word_count;
    uint32_t block_count;
    SparkStatus status;

    if (adapter == 0 || decode_dispatch == 0 || pipeline_slot == 0 ||
        frame_context == 0 || decode_dispatch->decode_view == 0 ||
        decode_dispatch->kv_block_table_view == 0 ||
        decode_dispatch->active_sequence_count == 0u ||
        decode_dispatch->active_sequence_count > adapter->maximum_active_sequence_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    for (lane_index = 0u;
         lane_index < decode_dispatch->active_sequence_count;
         ++lane_index)
    {
        const SparkRequestApiDecodeDispatchLaneView *lane;
        uint32_t token_id;
        uint32_t sequence_position;

        lane = &decode_dispatch->decode_view->lanes[lane_index];
        if (lane->sequence_position > UINT32_MAX - step_index)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        sequence_position = lane->sequence_position + step_index;
        status = SparkGlm52ServingAdapterValidateKvCoverage(
            decode_dispatch->kv_block_table_view,
            lane_index,
            sequence_position);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }

        if (step_index == 0u)
        {
            token_id = decode_dispatch->input_token_ids[lane_index];
        }
        else
        {
            if (step_index > decode_dispatch->speculative_token_count)
            {
                return SPARK_STATUS_INVALID_ARGUMENT;
            }
            token_id =
                decode_dispatch->speculative_draft_token_ids[lane_index][step_index - 1u];
        }
        if (token_id >= adapter->vocabulary_size)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        adapter->host_decode_positions[lane_index] = sequence_position;
        adapter->host_decode_token_ids[lane_index] = token_id;
    }

    status = SparkServingAdapterCudaStatus(
        cudaMemcpyAsync(
            adapter->device_decode_positions,
            adapter->host_decode_positions,
            (size_t)decode_dispatch->active_sequence_count * sizeof(uint32_t),
            cudaMemcpyHostToDevice,
            stream));
    if (status == SPARK_STATUS_OK)
    {
        status = SparkServingAdapterCudaStatus(
            cudaMemcpyAsync(
                adapter->device_decode_token_ids,
                adapter->host_decode_token_ids,
                (size_t)decode_dispatch->active_sequence_count * sizeof(uint32_t),
                cudaMemcpyHostToDevice,
                stream));
    }
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    block_count =
        (decode_dispatch->active_sequence_count +
         SPARK_SERVING_ADAPTER_THREADS - 1u) /
        SPARK_SERVING_ADAPTER_THREADS;
    SparkGlm52ServingAdapterBuildDecodeMetadataKernel<<<
        block_count,
        SPARK_SERVING_ADAPTER_THREADS,
        0,
        stream>>>(
            adapter->device_decode_positions,
            decode_dispatch->kv_block_table_view->physical_block_indices,
            decode_dispatch->kv_block_table_view->lane_physical_block_counts,
            decode_dispatch->kv_block_table_view->lane_stride,
            decode_dispatch->kv_block_table_view->block_token_count,
            decode_dispatch->active_sequence_count,
            (uint32_t *)pipeline_slot->positions,
            (uint32_t *)pipeline_slot->slot_mapping,
            (uint32_t *)pipeline_slot->context_lengths,
            (uint32_t *)pipeline_slot->first_block_token_offsets);
    status = SparkServingAdapterCudaStatus(cudaGetLastError());
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    word_count =
        (uint64_t)decode_dispatch->active_sequence_count *
        (adapter->hidden_dimension / 2u);
    block_count = (uint32_t)(
        (word_count + SPARK_SERVING_ADAPTER_THREADS - 1u) /
        SPARK_SERVING_ADAPTER_THREADS);
    SparkGlm52ServingAdapterGatherDecodeEmbeddingKernel<<<
        block_count,
        SPARK_SERVING_ADAPTER_THREADS,
        0,
        stream>>>(
            adapter->device_decode_token_ids,
            (const uint32_t *)adapter->embedding_weight_bf16,
            (uint32_t *)pipeline_slot->input_hidden_bf16,
            decode_dispatch->active_sequence_count,
            adapter->hidden_dimension / 2u);
    status = SparkServingAdapterCudaStatus(cudaGetLastError());
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    /* The first-party slice: inference/llms/glm5_2/bind.cu maps the node context
       to layer buffers and loops. Behind the same build flag as the layer body
       it calls, because the two must be selected together - a first-party slice
       driving the legacy layer, or the reverse, is a configuration nobody
       tested and the compiler cannot see. */
    return Glm52StageSlice(
               adapter->layer_node_contexts[0],
               adapter->first_party_buffers,
               adapter->first_layer_index,
               adapter->layer_count,
               decode_dispatch->active_sequence_count,
               /* Packed rows: every token expands into one row per expert it
                  routes to. The dispatch does not carry the product because the
                  legacy path never needed it. */
               decode_dispatch->active_sequence_count * SPARK_GLM52_MODEL_MOE_TOP_K,
               /* The MAXIMUM context across active sequences, not any one
                  sequence's. It decides the sparse-versus-dense branch, which is
                  a per-launch decision - a batch containing one long sequence
                  must take the sparse path for all of them. Using a single
                  sequence's length would run the dense path over a context the
                  budget cannot cover. */
               adapter->host_max_context_length,
               SparkResidentDecodeStageMultiprocessorCount(),
               stream) == 0
           ? SPARK_STATUS_OK
           : SPARK_STATUS_INTERNAL_ERROR;
}

static SparkStatus SparkGlm52ServingAdapterDecodeMtpVerify(
    SparkResidentDecodeStageServingAdapter *adapter,
    const SparkServingDecodeDispatch *decode_dispatch,
    SparkServingDecodeResult *decode_result,
    const SparkResidentDecodeStagePipelineSlot *pipeline_slot,
    cudaStream_t stream)
{
    SparkResidentDecodeStageFrameContext frame_context;
    uint32_t step_index;
    uint32_t lane_index;
    uint32_t verifier_token_count;
    SparkStatus status;

    if (adapter == 0 || decode_dispatch == 0 || decode_result == 0 ||
        pipeline_slot == 0 || decode_dispatch->speculative_token_count == 0u ||
        decode_dispatch->speculative_token_count >
            SPARK_GLM52_MODEL_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    verifier_token_count = decode_dispatch->speculative_token_count;
    if (verifier_token_count > SPARK_SERVING_MAX_DECODE_TOKENS_PER_LANE)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    status = SparkServingAdapterUploadMtpDraftBudgets(
        adapter,
        decode_dispatch->active_sequence_count,
        0u,
        stream);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    memset(&frame_context, 0, sizeof(frame_context));
    frame_context.abi_version =
        SPARK_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION;
    frame_context.descriptor_bytes =
        SPARK_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_DESCRIPTOR_BYTES;
    frame_context.flags =
        SPARK_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_KV_BLOCK_TABLE |
        SPARK_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_MTP_DRAFT_BUDGETS;
    frame_context.kv_block_table = decode_dispatch->kv_block_table_view;
    frame_context.mtp_draft_token_budgets =
        adapter->device_mtp_draft_token_budgets;

    for (step_index = 0u;
         step_index < verifier_token_count;
         ++step_index)
    {
        status = SparkGlm52ServingAdapterLaunchDecodeStep(
            adapter,
            decode_dispatch,
            pipeline_slot,
            &frame_context,
            step_index,
            stream);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        status = SparkServingAdapterCudaStatus(
            cudaMemcpyAsync(
                &adapter->host_mtp_committed_token_ids[
                    (uint64_t)step_index * adapter->maximum_active_sequence_count],
                pipeline_slot->restricted_selected_token_ids,
                (size_t)decode_dispatch->active_sequence_count * sizeof(uint32_t),
                cudaMemcpyDeviceToHost,
                stream));
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }

    status = SparkServingAdapterCudaStatus(cudaStreamSynchronize(stream));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    for (lane_index = 0u;
         lane_index < decode_dispatch->active_sequence_count;
         ++lane_index)
    {
        decode_result->token_counts[lane_index] = verifier_token_count;
        for (step_index = 0u;
             step_index < verifier_token_count;
             ++step_index)
        {
            decode_result->token_ids[lane_index][step_index] =
                adapter->host_mtp_committed_token_ids[
                    (uint64_t)step_index * adapter->maximum_active_sequence_count +
                    lane_index];
        }
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkResidentDecodeStageServingAdapterDecode(
    void *context,
    const SparkServingDecodeDispatch *decode_dispatch,
    SparkServingDecodeResult *decode_result)
{
    SparkResidentDecodeStageServingAdapter *adapter;
    const SparkResidentDecodeStagePipelineSlot *pipeline_slot;
    SparkResidentDecodeStageFrameContext frame_context;
    cudaStream_t stream;
    uint32_t lane_index;
    uint32_t draft_index;
    uint32_t mtp_commit_active;
    uint32_t mtp_budget;
    SparkStatus status;

    adapter = (SparkResidentDecodeStageServingAdapter *)context;
    if (adapter == 0 || decode_dispatch == 0 || decode_result == 0 ||
        decode_dispatch->abi_version != SPARK_SERVING_ENGINE_ABI_VERSION ||
        decode_dispatch->descriptor_bytes !=
            SPARK_SERVING_DECODE_DISPATCH_DESCRIPTOR_BYTES ||
        decode_dispatch->decode_view == 0 ||
        decode_dispatch->kv_block_table_view == 0 ||
        decode_dispatch->active_sequence_count == 0u ||
        decode_dispatch->active_sequence_count > adapter->maximum_active_sequence_count ||
        adapter->final_token_stage == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (decode_dispatch->dispatch_kind !=
            SPARK_REQUEST_API_DISPATCH_KIND_DECODE_BATCH &&
        decode_dispatch->dispatch_kind !=
            SPARK_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    pipeline_slot = SparkServingAdapterPipelineSlot(adapter);
    if (pipeline_slot == 0 ||
        pipeline_slot->input_hidden_bf16 == 0 ||
        pipeline_slot->positions == 0 ||
        pipeline_slot->slot_mapping == 0 ||
        pipeline_slot->context_lengths == 0 ||
        pipeline_slot->first_block_token_offsets == 0 ||
        pipeline_slot->restricted_selected_token_ids == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    stream = (cudaStream_t)adapter->cuda_stream;
    if (SparkServingAdapterDecodeDispatchIsMtpVerify(decode_dispatch) != 0u)
    {
        return SparkGlm52ServingAdapterDecodeMtpVerify(
            adapter,
            decode_dispatch,
            decode_result,
            pipeline_slot,
            stream);
    }
    if (decode_dispatch->dispatch_kind !=
        SPARK_REQUEST_API_DISPATCH_KIND_DECODE_BATCH)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    mtp_commit_active =
        (decode_dispatch->request_dispatch->flags &
            SPARK_REQUEST_API_DISPATCH_FLAG_MTP_COMMIT) != 0u;
    mtp_budget = 0u;
    if (mtp_commit_active != 0u)
    {
        static_assert(
            SPARK_REQUEST_API_MTP_MAX_DRAFT_TOKEN_COUNT ==
                SPARK_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT,
            "request api and firmware MTP draft token counts must match");
        mtp_budget = decode_dispatch->request_dispatch->mtp_draft_token_budget;
        if (mtp_budget == 0u ||
            mtp_budget > SPARK_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT ||
            pipeline_slot->mtp_draft_token_ids == 0)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
    }

    status = SparkServingAdapterUploadMtpDraftBudgets(
        adapter,
        decode_dispatch->active_sequence_count,
        mtp_budget,
        stream);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    memset(&frame_context, 0, sizeof(frame_context));
    frame_context.abi_version =
        SPARK_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION;
    frame_context.descriptor_bytes =
        SPARK_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_DESCRIPTOR_BYTES;
    frame_context.flags =
        SPARK_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_KV_BLOCK_TABLE |
        SPARK_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_MTP_DRAFT_BUDGETS;
    frame_context.kv_block_table = decode_dispatch->kv_block_table_view;
    frame_context.mtp_draft_token_budgets =
        adapter->device_mtp_draft_token_budgets;

    status = SparkGlm52ServingAdapterLaunchDecodeStep(
        adapter,
        decode_dispatch,
        pipeline_slot,
        &frame_context,
        0u,
        stream);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    status = SparkServingAdapterCudaStatus(
        cudaMemcpyAsync(
            adapter->host_decode_token_ids,
            pipeline_slot->restricted_selected_token_ids,
            (size_t)decode_dispatch->active_sequence_count * sizeof(uint32_t),
            cudaMemcpyDeviceToHost,
            stream));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    if (mtp_commit_active != 0u)
    {
        status = SparkServingAdapterCudaStatus(
            cudaMemcpyAsync(
                adapter->host_mtp_committed_token_ids,
                pipeline_slot->mtp_draft_token_ids,
                (size_t)decode_dispatch->active_sequence_count *
                    SPARK_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT *
                    sizeof(uint32_t),
                cudaMemcpyDeviceToHost,
                stream));
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }

    status = SparkServingAdapterCudaStatus(cudaStreamSynchronize(stream));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    for (lane_index = 0u;
         lane_index < decode_dispatch->active_sequence_count;
         ++lane_index)
    {
        decode_result->token_ids[lane_index][0u] =
            adapter->host_decode_token_ids[lane_index];
        decode_result->token_counts[lane_index] = 1u;
        if (mtp_commit_active != 0u)
        {
            for (draft_index = 0u;
                 draft_index < mtp_budget;
                 ++draft_index)
            {
                decode_result->token_ids[lane_index][draft_index + 1u] =
                    adapter->host_mtp_committed_token_ids[
                        ((uint64_t)lane_index *
                         SPARK_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT) +
                        draft_index];
            }
            decode_result->token_counts[lane_index] = mtp_budget + 1u;
        }
    }
    return SPARK_STATUS_OK;
}
