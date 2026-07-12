#include "sparkpipe/spark_glm52_cuda_resident_ipc.h"

#include <string.h>

void SparkGlm52CudaResidentIpcInitializeHeader(
    SparkGlm52CudaResidentIpcHeader *header,
    uint32_t kind,
    uint32_t rank_index,
    uint64_t sequence_number,
    uint32_t payload_bytes)
{
    if (header == 0)
        return;
    memset(header, 0, sizeof(*header));
    header->magic = SPARK_GLM52_CUDA_RESIDENT_IPC_MAGIC;
    header->abi_version = SPARK_GLM52_CUDA_RESIDENT_IPC_ABI_VERSION;
    header->descriptor_bytes = SPARK_GLM52_CUDA_RESIDENT_IPC_HEADER_BYTES;
    header->kind = kind;
    header->payload_bytes = payload_bytes;
    header->rank_index = rank_index;
    header->sequence_number = sequence_number;
}

SparkStatus SparkGlm52CudaResidentIpcValidateHeader(
    const SparkGlm52CudaResidentIpcHeader *header,
    uint32_t expected_kind,
    uint32_t maximum_payload_bytes)
{
    if (header == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (header->magic != SPARK_GLM52_CUDA_RESIDENT_IPC_MAGIC)
        return SPARK_STATUS_PARSE_ERROR;
    if (header->abi_version != SPARK_GLM52_CUDA_RESIDENT_IPC_ABI_VERSION ||
        header->descriptor_bytes != SPARK_GLM52_CUDA_RESIDENT_IPC_HEADER_BYTES)
        return SPARK_STATUS_ABI_MISMATCH;
    if (expected_kind != 0u && header->kind != expected_kind)
        return SPARK_STATUS_SCHEMA_ERROR;
    if (header->payload_bytes > maximum_payload_bytes)
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    return SPARK_STATUS_OK;
}

static uint32_t SparkGlm52CudaResidentIpcCalculateWorkMessageBytes(
	const SparkGlm52Pp13WorkControlPacket *work_packet,
	uint32_t prefix_bytes,
	uint32_t maximum_bytes)
{
	uint64_t message_bytes;

	if (work_packet == 0 ||
		work_packet->descriptor_bytes <
			SPARK_GLM52_PP13_WORK_CONTROL_PACKET_PREFIX_BYTES ||
		work_packet->descriptor_bytes > SPARK_GLM52_PP13_WORK_CONTROL_PACKET_BYTES)
		return 0u;
	message_bytes = (uint64_t)prefix_bytes + work_packet->descriptor_bytes;
	return message_bytes <= maximum_bytes ? (uint32_t)message_bytes : 0u;
}

uint32_t SparkGlm52CudaResidentIpcCalculateSubmitWorkBytes(
	const SparkGlm52Pp13WorkControlPacket *work_packet)
{
	return SparkGlm52CudaResidentIpcCalculateWorkMessageBytes(
		work_packet,
		SPARK_GLM52_CUDA_RESIDENT_IPC_SUBMIT_WORK_PREFIX_BYTES,
		SPARK_GLM52_CUDA_RESIDENT_IPC_SUBMIT_WORK_BYTES);
}

uint32_t SparkGlm52CudaResidentIpcCalculateSubmitDecodeBytes(
	const SparkGlm52Pp13WorkControlPacket *work_packet)
{
	return SparkGlm52CudaResidentIpcCalculateWorkMessageBytes(
		work_packet,
		SPARK_GLM52_CUDA_RESIDENT_IPC_SUBMIT_DECODE_PREFIX_BYTES,
		SPARK_GLM52_CUDA_RESIDENT_IPC_SUBMIT_DECODE_BYTES);
}
