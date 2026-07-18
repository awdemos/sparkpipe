#pragma once

#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_GLM52_KV_STORE_ABI_VERSION 2u
#define SPARK_GLM52_KV_STORE_INTERFACE_SYMBOL \
	"SparkGlm52KvStoreGetInterface"
#define SPARK_GLM52_KV_STORE_INTERFACE_BYTES \
	((uint32_t)sizeof(SparkGlm52KvStoreInterface))
#define SPARK_GLM52_KV_STORE_CONFIGURATION_BYTES \
	((uint32_t)sizeof(SparkGlm52KvStoreConfiguration))
#define SPARK_GLM52_KV_STORE_BATCH_BYTES \
	((uint32_t)sizeof(SparkGlm52KvStoreBatch))
#define SPARK_GLM52_KV_STORE_COMPLETION_BYTES \
	((uint32_t)sizeof(SparkGlm52KvStoreCompletion))
#define SPARK_GLM52_KV_STORE_MAX_KEY_BYTES 192u
#define SPARK_GLM52_KV_STORE_MAX_BATCH_BLOCKS 128u
#define SPARK_GLM52_KV_STORE_MAX_INFLIGHT_BATCHES 8u
#define SPARK_GLM52_KV_STORE_DEFAULT_LOOKAHEAD_PACKETS 3u
#define SPARK_GLM52_KV_STORE_MAX_LOOKAHEAD_PACKETS 8u

#define SPARK_GLM52_KV_STORE_CAP_BATCH_GET 0x00000001u
#define SPARK_GLM52_KV_STORE_CAP_BATCH_PUT 0x00000002u
#define SPARK_GLM52_KV_STORE_CAP_PERSISTENT_SERVICE 0x00000004u
#define SPARK_GLM52_KV_STORE_CAP_PROVIDER_BUFFERS 0x00000008u
#define SPARK_GLM52_KV_STORE_REQUIRED_CAPS \
	(SPARK_GLM52_KV_STORE_CAP_BATCH_GET | \
	 SPARK_GLM52_KV_STORE_CAP_BATCH_PUT | \
	 SPARK_GLM52_KV_STORE_CAP_PERSISTENT_SERVICE | \
	 SPARK_GLM52_KV_STORE_CAP_PROVIDER_BUFFERS)

#define SPARK_GLM52_KV_STORE_OPERATION_GET 1u
#define SPARK_GLM52_KV_STORE_OPERATION_PUT 2u

typedef struct SparkGlm52KvStoreConfiguration
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t rank_index;
	uint32_t first_layer_index;
	uint32_t layer_count;
	uint32_t worker_count;
	uint32_t maximum_inflight_batch_count;
	uint32_t maximum_batch_block_count;
	uint64_t model_fingerprint;
	uint64_t cache_layout_fingerprint;
	uint64_t client_memory_pool_bytes;
	uint64_t local_buffer_bytes;
	const char *service_address;
	const char *ipc_socket_path;
} SparkGlm52KvStoreConfiguration;

typedef struct SparkGlm52KvStoreBlock
{
	uint32_t operation;
	uint32_t key_bytes;
	uint32_t payload_bytes;
	uint32_t reserved0;
	void *payload;
	char key[SPARK_GLM52_KV_STORE_MAX_KEY_BYTES];
} SparkGlm52KvStoreBlock;

typedef struct SparkGlm52KvStoreBatch
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t block_count;
	uint32_t priority;
	uint64_t batch_id;
	SparkGlm52KvStoreBlock blocks[SPARK_GLM52_KV_STORE_MAX_BATCH_BLOCKS];
} SparkGlm52KvStoreBatch;

typedef struct SparkGlm52KvStoreCompletion
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	SparkStatus status;
	uint32_t completed_block_count;
	uint64_t batch_id;
} SparkGlm52KvStoreCompletion;

typedef SparkStatus (*SparkGlm52KvStoreInitializeFunction)(
	const SparkGlm52KvStoreConfiguration *configuration,
	void **store_state_out);
typedef void (*SparkGlm52KvStoreDestroyFunction)(void *store_state);
typedef SparkStatus (*SparkGlm52KvStoreSubmitFunction)(
	void *store_state,
	const SparkGlm52KvStoreBatch *batch);
typedef SparkStatus (*SparkGlm52KvStorePollFunction)(
	void *store_state,
	uint64_t batch_id,
	SparkGlm52KvStoreCompletion *completion);
typedef SparkStatus (*SparkGlm52KvStoreAllocateBufferFunction)(
	void *store_state,
	uint64_t buffer_bytes,
	void **buffer_out);
typedef SparkStatus (*SparkGlm52KvStoreReleaseBufferFunction)(
	void *store_state,
	void *buffer);

typedef struct SparkGlm52KvStoreInterface
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t capability_flags;
	uint32_t reserved0;
	SparkGlm52KvStoreInitializeFunction initialize;
	SparkGlm52KvStoreDestroyFunction destroy;
	SparkGlm52KvStoreSubmitFunction submit;
	SparkGlm52KvStorePollFunction poll;
	SparkGlm52KvStoreAllocateBufferFunction allocate_buffer;
	SparkGlm52KvStoreReleaseBufferFunction release_buffer;
} SparkGlm52KvStoreInterface;

typedef const SparkGlm52KvStoreInterface *(
	*SparkGlm52KvStoreGetInterfaceFunction)(void);

typedef struct SparkGlm52KvStoreDynamicLibrary
{
	void *dynamic_library;
	SparkGlm52KvStoreInterface store_interface;
} SparkGlm52KvStoreDynamicLibrary;

uint32_t SparkGlm52KvStoreNormalizeLookaheadPacketCount(
	uint32_t lookahead_packet_count,
	uint32_t queue_depth);
uint32_t SparkGlm52KvStoreSelectPressureLimitedLookaheadPacketCount(
	uint32_t lookahead_packet_count,
	uint32_t queue_depth,
	uint32_t physical_block_capacity,
	uint32_t allocated_physical_block_count,
	uint32_t staging_block_capacity,
	const uint32_t *cumulative_nonresident_block_counts);
SparkStatus SparkGlm52KvStoreValidateConfiguration(
	const SparkGlm52KvStoreConfiguration *configuration);
SparkStatus SparkGlm52KvStoreValidateBatch(
	const SparkGlm52KvStoreBatch *batch);
SparkStatus SparkGlm52KvStoreValidateInterface(
	const SparkGlm52KvStoreInterface *store_interface,
	uint32_t required_capability_flags);
SparkStatus SparkGlm52KvStoreLoadInterfaceFromSharedObject(
	const char *shared_object_path,
	uint32_t required_capability_flags,
	SparkGlm52KvStoreDynamicLibrary *library);
void SparkGlm52KvStoreUnloadInterface(
	SparkGlm52KvStoreDynamicLibrary *library);

#ifdef __cplusplus
}
#endif
