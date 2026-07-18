#include "sparkpipe/spark_glm52_kv_store.h"

#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <thread>

extern "C" const SparkGlm52KvStoreInterface *SparkGlm52KvStoreGetInterface(void);

static SparkStatus SparkTestMooncakeWait(
	const SparkGlm52KvStoreInterface *store_interface,
	void *store_state,
	uint64_t batch_id,
	SparkGlm52KvStoreCompletion *completion)
{
	uint32_t poll_count;
	SparkStatus status;
	for (poll_count = 0u; poll_count < 1000u; ++poll_count)
	{
		status = store_interface->poll(store_state,batch_id,completion);
		if (status != SPARK_STATUS_BUSY)
			return status;
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
	return SPARK_STATUS_BUSY;
}

static void SparkTestMooncakeBlock(
	SparkGlm52KvStoreBlock *block,
	uint32_t operation,
	const char *key,
	void *payload,
	uint32_t payload_bytes)
{
	block->operation = operation;
	block->key_bytes = (uint32_t)std::strlen(key);
	block->payload_bytes = payload_bytes;
	block->payload = payload;
	std::memcpy(block->key,key,block->key_bytes + 1u);
}

int main()
{
	const SparkGlm52KvStoreInterface *store_interface;
	SparkGlm52KvStoreConfiguration configuration;
	SparkGlm52KvStoreCompletion completion;
	SparkGlm52KvStoreBatch batch;
	uint8_t expected[128u];
	uint8_t *staging;
	void *store_state;
	uint32_t index;

	store_interface = SparkGlm52KvStoreGetInterface();
	assert(SparkGlm52KvStoreValidateInterface(
		store_interface,SPARK_GLM52_KV_STORE_REQUIRED_CAPS) == SPARK_STATUS_OK);
	std::memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_GLM52_KV_STORE_ABI_VERSION;
	configuration.descriptor_bytes = SPARK_GLM52_KV_STORE_CONFIGURATION_BYTES;
	configuration.layer_count = 4u;
	configuration.worker_count = 2u;
	configuration.maximum_inflight_batch_count = 2u;
	configuration.maximum_batch_block_count = 2u;
	configuration.model_fingerprint = 1u;
	configuration.cache_layout_fingerprint = 2u;
	configuration.client_memory_pool_bytes = 4096u;
	configuration.local_buffer_bytes = 4096u;
	configuration.service_address = "127.0.0.1:50052";
	configuration.ipc_socket_path = "/tmp/mooncake-test.sock";
	store_state = nullptr;
	assert(store_interface->initialize(&configuration,&store_state) ==
		SPARK_STATUS_OK);
	staging = nullptr;
	assert(store_interface->allocate_buffer(store_state,sizeof(expected),
		reinterpret_cast<void **>(&staging)) ==
		SPARK_STATUS_OK);
	for (index = 0u; index < sizeof(expected); ++index)
		staging[index] = expected[index] = (uint8_t)(index ^ 0x5au);
	std::memset(&batch,0,sizeof(batch));
	batch.abi_version = SPARK_GLM52_KV_STORE_ABI_VERSION;
	batch.descriptor_bytes = SPARK_GLM52_KV_STORE_BATCH_BYTES;
	batch.block_count = 2u;
	batch.batch_id = 11u;
	SparkTestMooncakeBlock(
		&batch.blocks[0u],SPARK_GLM52_KV_STORE_OPERATION_PUT,
		"glm52/test/a",staging,64u);
	SparkTestMooncakeBlock(
		&batch.blocks[1u],SPARK_GLM52_KV_STORE_OPERATION_PUT,
		"glm52/test/b",staging + 64u,64u);
	assert(store_interface->submit(store_state,&batch) == SPARK_STATUS_OK);
	assert(store_interface->submit(store_state,&batch) == SPARK_STATUS_DUPLICATE);
	assert(SparkTestMooncakeWait(
		store_interface,store_state,11u,&completion) == SPARK_STATUS_OK);
	assert(completion.completed_block_count == 2u);
	std::memset(staging,0,sizeof(expected));
	batch.batch_id = 12u;
	batch.blocks[0u].operation = SPARK_GLM52_KV_STORE_OPERATION_GET;
	batch.blocks[1u].operation = SPARK_GLM52_KV_STORE_OPERATION_GET;
	assert(store_interface->submit(store_state,&batch) == SPARK_STATUS_OK);
	assert(SparkTestMooncakeWait(
		store_interface,store_state,12u,&completion) == SPARK_STATUS_OK);
	assert(completion.completed_block_count == 2u);
	assert(std::memcmp(staging,expected,sizeof(expected)) == 0);
	assert(store_interface->release_buffer(store_state,staging) ==
		SPARK_STATUS_OK);
	store_interface->destroy(store_state);
	return 0;
}
