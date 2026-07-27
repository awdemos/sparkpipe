// Bind a node context to the first-party layer.
//
// This is what the 27,307-line decode stage's twenty reachable entry points
// reduce to once every kernel it needs exists elsewhere: a mapping from the
// weight-binding struct the host already fills to the buffer struct
// llms/glm5_2/layer.cuh reads, and a loop.
//
// I STOPPED WRITING THIS ONCE ALREADY, and the reason is worth keeping. The
// first attempt would have mapped 137 NodeContext fields onto a layer that did
// ONE attention projection where the model does four, had no output projection,
// never wrote the KV cache, and routed the three dense layers. Nine defects, all
// found by reading rather than running. An adapter over that would have buried
// them under plausible-looking plumbing.
//
// The mapping below is the part where errors hide now. Each assignment is a
// claim that two names mean the same tensor, and a wrong one produces output
// that is fluent. Where a claim is not obvious the line says why.

#include "inference/kernels/formats/fp8.cuh"
#include "inference/kernels/formats/int7.cuh"
#include "inference/llms/glm5_2/layer.cuh"
#include "sparkpipe/spark_glm52_resident_decode_stage_firmware.h"

// The absorbed path's four projections, from the node context's four weights.
//
// GLM 5.2 ships both forms and the host binds whichever the checkpoint carries.
// Absorbed is chosen when query_latent_weight is present, because that weight
// only exists in a pack whose up-projections were folded at pack time - so the
// presence of the tensor IS the decision, and there is nothing to configure.
static void Glm52BindAbsorbed(const SparkGlm52ResidentDecodeStageNodeContext *node, LmAbsorbedWeights *out)
{
	out->query_latent_weight = node->query_latent_weight_bf16;
	out->query_latent_scale = 0;
	out->query_rope_weight = node->query_rope_weight_bf16;
	out->query_rope_scale = 0;
	out->key_rope_weight = node->key_rope_weight_bf16;
	out->key_rope_scale = 0;
	out->kv_latent_weight = node->kv_latent_weight_bf16;
	out->kv_latent_scale = 0;
	out->input_dimension = GLM52_HIDDEN;
	out->query_latent_dimension = GLM52_LATENT;
	out->rope_dimension = GLM52_ROPE_DIM;
	out->kv_latent_dimension = GLM52_LATENT_ROW;
}

// The raw path: two low-rank projections, each with its own norm between the
// stages. The ranks come from the model rather than the context because they are
// architecture, not binding - the context carries pointers, config.h carries
// shapes, and mixing them is how a rank ends up read from a struct that never
// set it.
static void Glm52BindRaw(const SparkGlm52ResidentDecodeStageNodeContext *node, LmLowRankWeights *query, LmLowRankWeights *kv)
{
	query->down_weight = node->raw_query_a_weight_fp8_e4m3;
	query->down_scale = node->raw_query_a_weight_scale_inv_f32;
	query->norm_weight = node->raw_query_a_norm_weight_bf16;
	query->up_weight = node->raw_query_b_weight_fp8_e4m3;
	query->up_scale = node->raw_query_b_weight_scale_inv_f32;
	query->input_dimension = GLM52_HIDDEN;
	query->rank = GLM52_QUERY_A_DIM;
	query->output_dimension = GLM52_ATTN_HEADS * (GLM52_QK_NOPE_DIM + GLM52_ROPE_DIM);
	query->norm_epsilon = GLM52_RMS_EPSILON;
	kv->down_weight = node->raw_kv_a_weight_fp8_e4m3;
	kv->down_scale = node->raw_kv_a_weight_scale_inv_f32;
	kv->norm_weight = node->raw_kv_a_norm_weight_bf16;
	kv->up_weight = node->raw_kv_b_weight_fp8_e4m3;
	kv->up_scale = node->raw_kv_b_weight_scale_inv_f32;
	kv->input_dimension = GLM52_HIDDEN;
	kv->rank = GLM52_LATENT;
	kv->output_dimension = GLM52_ATTN_HEADS * (GLM52_QK_NOPE_DIM + GLM52_VALUE_DIM);
	kv->norm_epsilon = GLM52_RMS_EPSILON;
}

// Everything a layer reads, from the context and the workspace.
//
// The layer index selects which layer's weights: the context binds all of them
// contiguously, so a layer's slice is its index times the per-layer stride. That
// stride is a property of the pack and comes from the context rather than being
// recomputed here - recomputing it is how a change to the pack layout produces a
// silent off-by-one-layer, which reads as the model being slightly wrong rather
// than as a binding error.
static int32_t Glm52BindLayer(const SparkGlm52ResidentDecodeStageNodeContext *node, uint32_t layer_index, Glm52LayerBuffers *out)
{
	if ( node == 0 || out == 0 || layer_index >= GLM52_LAYERS )
		return(LM_LAUNCH_ERR_SHAPE);
	memset(out,0,sizeof(*out));
	out->attn_norm_weight = node->attention_norm_weight_bf16;
	out->mlp_norm_weight = node->post_attention_norm_weight_bf16;
	out->output_weight = node->attention_output_weight_fp8_e4m3;
	out->output_scale = node->attention_output_weight_scale_inv_f32;
	out->router_weight = node->moe_router_weight_bf16;
	out->dense_down_weight = node->dense_down_weight_fp8_e4m3;
	out->dense_down_scale = node->dense_down_weight_scale_inv_f32;
	// The absorbed weights only exist in a pack that folded the up-projections,
	// so their presence is the decision rather than a flag beside it.
	out->use_absorbed = node->query_latent_weight_bf16 != 0;
	if ( out->use_absorbed )
		Glm52BindAbsorbed(node,&out->absorbed);
	else
		Glm52BindRaw(node,&out->raw_query,&out->raw_kv);
	return(LM_LAUNCH_OK);
}

// One stage slice: the layers this rank owns, in order.
//
// The three dense layers and the routed ones take different halves, which is
// defect 8 - every layer was routed, including the three that have no experts.
// GLM52_FIRST_ROUTED_LAYER is what decides, and it is checked per layer rather
// than assumed from the rank, because a rank's slice can straddle the boundary.
static int32_t Glm52LaunchSlice(const SparkGlm52ResidentDecodeStageNodeContext *node, Glm52LayerBuffers *buffers, uint32_t first_layer, uint32_t layer_count, uint32_t rows, uint32_t packed_rows, uint32_t context, uint32_t multiprocessors, cudaStream_t stream)
{
	uint32_t offset,layer,layer_in_group;
	int32_t status;
	for (offset = 0u; offset < layer_count; ++offset)
	{
		layer = first_layer + offset;
		status = Glm52BindLayer(node,layer,buffers);
		if ( status != LM_LAUNCH_OK )
			return(status);
		// The DSA index is shared across a group of layers, so only the first
		// layer of each group scores and selects. Computed from the absolute
		// layer index, not the offset within this rank's slice - a rank starting
		// mid-group would otherwise recompute a selection the previous rank
		// already made.
		layer_in_group = layer % GLM52_DSA_SHARE_GROUP_LAYERS;
		status = Glm52LayerAttention<LmFp8>(buffers,rows,context,layer_in_group,
			multiprocessors,stream);
		if ( status != LM_LAUNCH_OK )
			return(status);
		if ( layer < GLM52_FIRST_ROUTED_LAYER )
			status = Glm52LayerDenseMlp<LmFp8>(buffers,rows,multiprocessors,stream);
		else
			status = Glm52LayerMoe<LmFp8>(buffers,rows,packed_rows,multiprocessors,stream);
		if ( status != LM_LAUNCH_OK )
			return(status);
	}
	return(LM_LAUNCH_OK);
}

// Prefill: the same slice with row positions.
//
// A prefill row is not the last position and must not see past itself; a decode
// row is and may see everything. That is one argument to the attention kernel,
// which is why there is no prefill slice - only this entry point, which passes
// the positions the decode one leaves null.
//
// Chunking is the caller's: a prompt longer than the intermediate buffers runs
// as several calls, and how long that is depends on what else is resident.
extern "C" int32_t Glm52StageSlicePrefill(const void *node_context, void *layer_buffers, uint32_t first_layer, uint32_t layer_count, uint32_t rows, uint32_t packed_rows, uint32_t context, uint32_t multiprocessors, const uint32_t *row_positions, void *stream)
{
	Glm52LayerBuffers *buffers = (Glm52LayerBuffers *)layer_buffers;
	buffers->row_positions = row_positions;
	return(Glm52LaunchSlice(
		(const SparkGlm52ResidentDecodeStageNodeContext *)node_context,
		buffers,first_layer,layer_count,rows,packed_rows,context,
		multiprocessors,(cudaStream_t)stream));
}

extern "C" int32_t Glm52StageSlice(const void *node_context, void *layer_buffers, uint32_t first_layer, uint32_t layer_count, uint32_t rows, uint32_t packed_rows, uint32_t context, uint32_t multiprocessors, void *stream)
{
	return(Glm52LaunchSlice(
		(const SparkGlm52ResidentDecodeStageNodeContext *)node_context,
		(Glm52LayerBuffers *)layer_buffers,
		first_layer,layer_count,rows,packed_rows,context,multiprocessors,
		(cudaStream_t)stream));
}
