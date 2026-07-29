// K3's answer to the stage's two validation questions - the second model
// family through the linker seam, which is the proof the seam exists. The
// common module asks whether a node context is runnable; K3 answers in its
// own dimensions: hidden 7168, the 512 + 64 MLA latent cache, 96 KDA heads
// with 128-wide keys and values, 69 + 24 layers. Where glm's validator
// walks quantized projection catalogs, K3's begins with the geometry
// contract and grows plan checks as the K3 execute path lands (see
// techdebt.md: K3 stage execute).
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include "sparkpipe/spark_resident_decode_stage.h"
#include "sparkpipe/spark_k3_kv_geometry.h"

#define SPARK_K3_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION 7168u

SparkStatus SparkResidentDecodeStageModelValidateNodeContext(const SparkResidentDecodeStageNodeContext *node_context)
{
	if (node_context == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (node_context->abi_version != SPARK_RESIDENT_DECODE_STAGE_NODE_CONTEXT_ABI_VERSION)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (node_context->pipeline_slot_count == 0u ||
		node_context->max_active_sequence_count == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (node_context->cache_token_capacity == 0u ||
		node_context->kv_block_count == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	// The MLA latent arena must be present and MLA-shaped; K3's cache is
	// the 512 + 64 compressed latent, never a full key/value pair.
	if (node_context->mla_cache_bf16 == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	return SPARK_STATUS_OK;
}

SparkStatus SparkResidentDecodeStageModelValidateSliceNodeContext(const SparkResidentDecodeStageSliceNodeContext *slice_node_context, const SparkResidentDecodeStageNodeContext **first_node_context)
{
	if (slice_node_context == 0 || first_node_context == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (slice_node_context->layer_count == 0u ||
		slice_node_context->layer_count >
			SPARK_K3_KV_KDA_LAYER_COUNT + SPARK_K3_KV_MLA_LAYER_COUNT)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (slice_node_context->layer_node_contexts == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	*first_node_context = slice_node_context->layer_node_contexts[0u];
	return SparkResidentDecodeStageModelValidateNodeContext(*first_node_context);
}

SparkStatus SparkResidentDecodeStageModelValidateFrameTaps(const SparkResidentDecodeStageFrameContext *frame_context)
{
	// K3 has no drafter hidden taps; a frame claiming them is malformed.
	if ( frame_context == 0 )
		return SPARK_STATUS_INVALID_ARGUMENT;
	if ( (frame_context->flags & SPARK_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_MODEL_HIDDEN_TAPS) != 0u )
		return SPARK_STATUS_INVALID_ARGUMENT;
	return SPARK_STATUS_OK;
}
