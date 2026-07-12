#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_glm52_cuda_resident_ipc.h"

static SparkGlm52CudaResidentIpcSubmitDecode *SparkTestBuildWideDecode(
    uint32_t lane_count,
    uint32_t blocks_per_lane,
    uint32_t *payload_bytes_out)
{
    SparkGlm52CudaResidentIpcSubmitDecode *message;
    uint32_t block_index_count;
    uint32_t payload_bytes;
    uint32_t lane_index;
    uint32_t block_index;
    block_index_count = lane_count * blocks_per_lane;
    assert(SparkGlm52CudaResidentIpcDecodePayloadBytes(
        block_index_count,&payload_bytes) == SPARK_STATUS_OK);
    message = (SparkGlm52CudaResidentIpcSubmitDecode *)calloc(1u,payload_bytes);
    assert(message != 0);
    message->descriptor_bytes =
        SPARK_GLM52_CUDA_RESIDENT_IPC_SUBMIT_DECODE_HEADER_BYTES;
    message->dispatch_kind =
        SPARK_GLM52_REQUEST_API_DISPATCH_KIND_DECODE_BATCH;
    message->lane_count = lane_count;
    message->active_sequence_count = lane_count;
    message->kv_block_token_count = SPARK_GLM52_KV_BLOCK_TOKENS;
    message->kv_block_index_count = block_index_count;
    for (lane_index = 0u; lane_index < lane_count; ++lane_index)
    {
        SparkGlm52CudaResidentIpcDecodeLane *lane;
        lane = &message->lanes[lane_index];
        lane->request_id = 1000u + lane_index;
        lane->sequence_id = 2000u + lane_index;
        lane->sequence_position = 8192u + lane_index;
        lane->request_slot_index = lane_index;
        lane->context_token_count = 8192u;
        lane->input_token_id = 17u + lane_index;
        lane->kv_block_offset = lane_index * blocks_per_lane;
        lane->kv_block_count = blocks_per_lane;
        for (block_index = 0u; block_index < blocks_per_lane; ++block_index)
            message->kv_physical_block_indices[
                lane->kv_block_offset + block_index] =
                lane_index * 4096u + block_index;
    }
    *payload_bytes_out = payload_bytes;
    return message;
}

static void SparkTestWideDecodePayload(void)
{
    SparkGlm52CudaResidentIpcSubmitDecode *message;
    uint32_t payload_bytes;
    message = SparkTestBuildWideDecode(
        SPARK_GLM52_PP13_WORK_CONTROL_MAX_LANE_COUNT,17u,&payload_bytes);
    assert(SparkGlm52CudaResidentIpcValidateSubmitDecode(
        message,payload_bytes,
        SPARK_GLM52_PP13_WORK_CONTROL_MAX_LANE_COUNT) == SPARK_STATUS_OK);
    assert(message->lanes[1023u].kv_block_offset == 1023u * 17u);
    assert(message->kv_physical_block_indices[
        message->lanes[1023u].kv_block_offset + 16u] ==
        1023u * 4096u + 16u);

    message->lanes[511u].kv_block_offset += 1u;
    assert(SparkGlm52CudaResidentIpcValidateSubmitDecode(
        message,payload_bytes,
        SPARK_GLM52_PP13_WORK_CONTROL_MAX_LANE_COUNT) ==
        SPARK_STATUS_INVALID_ARGUMENT);
    message->lanes[511u].kv_block_offset -= 1u;

    message->dispatch_kind =
        SPARK_GLM52_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH;
    message->speculative_token_count = 6u;
    for (uint32_t lane_index = 0u; lane_index < message->lane_count;
         ++lane_index)
        message->lanes[lane_index].speculative_token_count = 6u;
    assert(SparkGlm52CudaResidentIpcValidateSubmitDecode(
        message,payload_bytes,
        SPARK_GLM52_PP13_WORK_CONTROL_MAX_LANE_COUNT) == SPARK_STATUS_OK);
    assert(SparkGlm52CudaResidentIpcValidateSubmitDecode(
        message,payload_bytes,
        SPARK_GLM52_PP13_WORK_CONTROL_MAX_LANE_COUNT - 1u) ==
        SPARK_STATUS_INVALID_ARGUMENT);
    free(message);
}

static void SparkTestDecodePayloadCapacity(void)
{
    uint32_t maximum_block_index_count;
    uint32_t payload_bytes;
    maximum_block_index_count =
        SPARK_GLM52_PP13_WORK_CONTROL_MAX_LANE_COUNT *
        SPARK_GLM52_CUDA_RESIDENT_IPC_MAX_LANE_BLOCKS;
    assert(SparkGlm52CudaResidentIpcDecodePayloadBytes(
        maximum_block_index_count,&payload_bytes) == SPARK_STATUS_OK);
    assert(payload_bytes ==
        SPARK_GLM52_CUDA_RESIDENT_IPC_MAX_DECODE_PAYLOAD_BYTES);
    assert(SparkGlm52CudaResidentIpcDecodePayloadBytes(
        maximum_block_index_count + 1u,&payload_bytes) ==
        SPARK_STATUS_CAPACITY_EXCEEDED);
}

static void SparkTestInternalDirectoryDecodeHasNoBlockPayload(void)
{
    SparkGlm52CudaResidentIpcSubmitDecode *message;
    uint32_t payload_bytes;
    uint32_t lane_index;
    assert(SparkGlm52CudaResidentIpcDecodePayloadBytes(
        0u,&payload_bytes) == SPARK_STATUS_OK);
    assert(payload_bytes ==
        SPARK_GLM52_CUDA_RESIDENT_IPC_SUBMIT_DECODE_HEADER_BYTES);
    message = (SparkGlm52CudaResidentIpcSubmitDecode *)calloc(
        1u,payload_bytes);
    assert(message != 0);
    message->descriptor_bytes =
        SPARK_GLM52_CUDA_RESIDENT_IPC_SUBMIT_DECODE_HEADER_BYTES;
    message->dispatch_kind =
        SPARK_GLM52_REQUEST_API_DISPATCH_KIND_DECODE_BATCH;
    message->lane_count = SPARK_GLM52_PP13_WORK_CONTROL_MAX_LANE_COUNT;
    message->active_sequence_count = message->lane_count;
    message->kv_block_token_count = SPARK_GLM52_KV_BLOCK_TOKENS;
    message->resident_flags =
        SPARK_GLM52_CUDA_RESIDENT_IPC_SUBMIT_FLAG_INTERNAL_KV_DIRECTORY;
    for (lane_index = 0u; lane_index < message->lane_count; ++lane_index)
    {
        message->lanes[lane_index].request_id = 1000u + lane_index;
        message->lanes[lane_index].sequence_id = 2000u + lane_index;
        message->lanes[lane_index].request_slot_index = lane_index;
        message->lanes[lane_index].context_token_count = 4096u;
        message->lanes[lane_index].input_token_id = 123u;
    }
    assert(SparkGlm52CudaResidentIpcValidateSubmitDecode(
        message,payload_bytes,
        SPARK_GLM52_PP13_WORK_CONTROL_MAX_LANE_COUNT) == SPARK_STATUS_OK);
    message->lanes[9u].kv_block_count = 1u;
    assert(SparkGlm52CudaResidentIpcValidateSubmitDecode(
        message,payload_bytes,
        SPARK_GLM52_PP13_WORK_CONTROL_MAX_LANE_COUNT) ==
        SPARK_STATUS_INVALID_ARGUMENT);
    free(message);
}

static void SparkTestWideSubmitWorkUsesVariablePayload(void)
{
    SparkGlm52Pp13WorkControlPacket *packet;
    uint32_t lane_index;
    uint32_t packet_bytes;
    uint32_t submit_bytes;

    packet = (SparkGlm52Pp13WorkControlPacket *)calloc(1u,sizeof(*packet));
    assert(packet != 0);
    packet->magic = SPARK_GLM52_PP13_WORK_CONTROL_PACKET_MAGIC;
    packet->abi_version = SPARK_GLM52_PP13_WORK_CONTROL_ABI_VERSION;
    packet->flags =
        SPARK_GLM52_PP13_WORK_CONTROL_FLAG_MTP_SPECULATIVE_VERIFY;
    packet->lane_count = SPARK_GLM52_PP13_WORK_CONTROL_MAX_LANE_COUNT;
    packet->active_sequence_count = packet->lane_count;
    packet->rows_per_lane = SPARK_GLM52_MODEL_MTP_DRAFT_TOKEN_COUNT + 1u;
    packet->execution_row_count = packet->lane_count * packet->rows_per_lane;
    packet->new_token_count = packet->rows_per_lane;
    packet->speculative_token_count =
        SPARK_GLM52_MODEL_MTP_DRAFT_TOKEN_COUNT;
    packet_bytes = SparkGlm52Pp13WorkControlCalculatePacketBytes(
        packet->lane_count);
    assert(packet_bytes == SPARK_GLM52_PP13_WORK_CONTROL_PACKET_BYTES);
    packet->descriptor_bytes = packet_bytes;
    for (lane_index = 0u; lane_index < packet->lane_count; ++lane_index)
    {
        packet->lanes[lane_index].request_id = 1000u + lane_index;
        packet->lanes[lane_index].sequence_id = 2000u + lane_index;
        packet->lanes[lane_index].request_slot_index = lane_index;
        packet->lanes[lane_index].speculative_token_count =
            packet->speculative_token_count;
    }
    submit_bytes = SparkGlm52CudaResidentIpcCalculateSubmitWorkBytes(packet);
    assert(submit_bytes ==
        SPARK_GLM52_CUDA_RESIDENT_IPC_SUBMIT_WORK_PREFIX_BYTES + packet_bytes);
    assert(submit_bytes == SPARK_GLM52_CUDA_RESIDENT_IPC_SUBMIT_WORK_BYTES);

    packet->descriptor_bytes =
        SPARK_GLM52_PP13_WORK_CONTROL_PACKET_PREFIX_BYTES - 1u;
    assert(SparkGlm52CudaResidentIpcCalculateSubmitWorkBytes(packet) == 0u);
    packet->descriptor_bytes =
        SPARK_GLM52_PP13_WORK_CONTROL_PACKET_BYTES + 1u;
    assert(SparkGlm52CudaResidentIpcCalculateSubmitWorkBytes(packet) == 0u);
    free(packet);
}

int main(void)
{
    SparkTestWideDecodePayload();
    SparkTestDecodePayloadCapacity();
    SparkTestInternalDirectoryDecodeHasNoBlockPayload();
    SparkTestWideSubmitWorkUsesVariablePayload();
    return 0;
}
