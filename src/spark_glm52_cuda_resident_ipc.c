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
