// Hybrid KV wiring arithmetic, locked. The cache tier is layout-
// parameterized; what a hybrid family wires is (a) the KV-bearing
// layer count - only the full-attention layers have growing cache -
// and (b) its layout request. This test pins the bytes-per-token
// numbers for K3 (MLA compressed latent) and qwen 3.6 (full GQA
// key-value) to the values the speed models and the family headers
// agree on, so a wiring mistake fails a gate instead of a bring-up.
// Recurrent state (KDA, GDN) is deliberately absent here: it is
// non-growing and lane-indexed at the execute rung, not a cache-tier
// concept.
#include <assert.h>
#include <stdint.h>
#include <string.h>
#include "sparkpipe/spark_kv_cache.h"
#include "sparkpipe/spark_k3_kv_geometry.h"
#include "sparkpipe/spark_qwen36_model.h"

int main(void)
{
	SparkKvCacheCapacityRequest request;
	SparkKvCacheCapacityEstimate estimate;
	memset(&request, 0, sizeof(request));
	request.abi_version = SPARK_KV_CACHE_ABI_VERSION;
	request.descriptor_bytes = SPARK_KV_CACHE_CAPACITY_REQUEST_DESCRIPTOR_BYTES;
	request.layout = SPARK_K3_KV_LAYOUT;
	request.context_token_count = 4096u;
	request.block_token_count = 64u;
	request.layer_count = SPARK_K3_KV_MLA_LAYER_COUNT;
	request.latent_dimension = SPARK_K3_KV_LATENT_DIMENSION;
	request.rope_dimension = SPARK_K3_KV_ROPE_DIMENSION;
	request.bytes_per_scalar = SPARK_K3_KV_BYTES_PER_SCALAR;
	request.cache_bytes_per_rank = 8ull << 30;
	memset(&estimate, 0, sizeof(estimate));
	assert(SparkKvCacheEstimateCapacity(&request, &estimate) ==
		SPARK_STATUS_OK);
	assert(estimate.attention_bytes_per_token_per_layer == 1152u);
	assert(SPARK_K3_KV_MLA_LAYER_COUNT == 24u);
	assert(SPARK_K3_KV_KDA_LAYER_COUNT == 69u);
	memset(&request, 0, sizeof(request));
	request.abi_version = SPARK_KV_CACHE_ABI_VERSION;
	request.descriptor_bytes = SPARK_KV_CACHE_CAPACITY_REQUEST_DESCRIPTOR_BYTES;
	request.layout = SPARK_KV_CACHE_LAYOUT_FULL_KEY_VALUE;
	request.context_token_count = 4096u;
	request.block_token_count = 64u;
	request.layer_count = 16u;
	request.head_count = SPARK_QWEN36_MODEL_KV_HEAD_COUNT;
	request.qk_nope_head_dimension = SPARK_QWEN36_MODEL_HEAD_DIMENSION;
	request.value_head_dimension = SPARK_QWEN36_MODEL_HEAD_DIMENSION;
	request.bytes_per_scalar = 2u;
	request.cache_bytes_per_rank = 8ull << 30;
	memset(&estimate, 0, sizeof(estimate));
	assert(SparkKvCacheEstimateCapacity(&request, &estimate) ==
		SPARK_STATUS_OK);
	assert(estimate.attention_bytes_per_token_per_layer == 4096u);
	return 0;
}
