// Bind weights to the layer and run a rank's slice of Kimi K3.
//
// The fourth driver, after glm5_2, qwen_3_6 and mimo_2_5, and the first where
// both halves of a layer branch: the attention kind by K3_LAYER_KIND and the
// feed-forward by whether the layer is past K3_FIRST_ROUTED_LAYER.
//
// Weights arrive as an explicit per-layer table for the same reason as the
// other three: no K3 pack format exists, and inventing one from the layer's
// requirements is the mistake glm5_2/bind.cu records its author stopping to
// avoid. What a packer produces is its business; this file needs the pointers.

#include "inference/kernels/formats/int7.cuh"
#include "inference/kernels/formats/mxfp4.cuh"
#include "inference/llms/kimi_k3/layer.cuh"

struct K3LayerWeights
{
	const void *attn_norm_weight;
	const void *mlp_norm_weight;

	const void *kda_q_weight;
	const void *kda_q_scale;
	const void *kda_k_weight;
	const void *kda_k_scale;
	const void *kda_v_weight;
	const void *kda_v_scale;
	const void *kda_q_conv_weight;
	const void *kda_k_conv_weight;
	const void *kda_v_conv_weight;
	const void *kda_decay_down_weight;
	const void *kda_decay_up_weight;
	const float *kda_decay_bias;
	const float *kda_head_log_scale;
	const void *kda_beta_weight;
	const void *kda_gate_weight;
	const void *kda_gate_scale;
	const void *kda_out_norm_weight;
	const void *kda_out_weight;
	const void *kda_out_scale;

	const void *mla_q_down_weight;
	const void *mla_q_down_scale;
	const void *mla_q_up_weight;
	const void *mla_q_up_scale;
	const void *mla_kv_a_weight;
	const void *mla_kv_a_scale;
	const void *mla_kv_a_norm_weight;
	const void *mla_kv_b_weight;
	const void *mla_kv_b_scale;
	const void *mla_gate_weight;
	const void *mla_gate_scale;
	const void *mla_out_weight;
	const void *mla_out_scale;

	const void *router_weight;
	const void *routed_down_weight;
	const void *routed_down_scale;
	const void *routed_up_weight;
	const void *routed_up_scale;
	const void *routed_norm_weight;
	const void *expert_w1_weight;
	const void *expert_w1_scale;
	const void *expert_w2_weight;
	const void *expert_w2_scale;
	const void *shared_w1_weight;
	const void *shared_w1_scale;
	const void *shared_w2_weight;
	const void *shared_w2_scale;
	const void *dense_gate_up_weight;
	const void *dense_gate_up_scale;
	const void *dense_down_weight;
	const void *dense_down_scale;
	const void *attnres_attn_weight;
	const void *attnres_mlp_weight;
};

// Each assignment is a claim that two names mean the same tensor. A KDA layer
// leaves the MLA pointers null and the reverse, and which set is read is decided
// by K3_LAYER_KIND rather than by which pointers happen to be set - a missing
// tensor should fail loudly, not silently select the other path.
static void K3BindLayer(const K3LayerWeights *weights, K3LayerBuffers *buffers)
{
	buffers->attn_norm_weight = weights->attn_norm_weight;
	buffers->mlp_norm_weight = weights->mlp_norm_weight;
	buffers->kda_q_weight = weights->kda_q_weight;
	buffers->kda_q_scale = weights->kda_q_scale;
	buffers->kda_k_weight = weights->kda_k_weight;
	buffers->kda_k_scale = weights->kda_k_scale;
	buffers->kda_v_weight = weights->kda_v_weight;
	buffers->kda_v_scale = weights->kda_v_scale;
	buffers->kda_q_conv_weight = weights->kda_q_conv_weight;
	buffers->kda_k_conv_weight = weights->kda_k_conv_weight;
	buffers->kda_v_conv_weight = weights->kda_v_conv_weight;
	buffers->kda_decay_down_weight = weights->kda_decay_down_weight;
	buffers->kda_decay_up_weight = weights->kda_decay_up_weight;
	buffers->kda_decay_bias = weights->kda_decay_bias;
	buffers->kda_head_log_scale = weights->kda_head_log_scale;
	buffers->kda_beta_weight = weights->kda_beta_weight;
	buffers->kda_gate_weight = weights->kda_gate_weight;
	buffers->kda_gate_scale = weights->kda_gate_scale;
	buffers->kda_out_norm_weight = weights->kda_out_norm_weight;
	buffers->kda_out_weight = weights->kda_out_weight;
	buffers->kda_out_scale = weights->kda_out_scale;
	buffers->mla_q_down_weight = weights->mla_q_down_weight;
	buffers->mla_q_down_scale = weights->mla_q_down_scale;
	buffers->mla_q_up_weight = weights->mla_q_up_weight;
	buffers->mla_q_up_scale = weights->mla_q_up_scale;
	buffers->mla_kv_a_weight = weights->mla_kv_a_weight;
	buffers->mla_kv_a_scale = weights->mla_kv_a_scale;
	buffers->mla_kv_a_norm_weight = weights->mla_kv_a_norm_weight;
	buffers->mla_kv_b_weight = weights->mla_kv_b_weight;
	buffers->mla_kv_b_scale = weights->mla_kv_b_scale;
	buffers->mla_gate_weight = weights->mla_gate_weight;
	buffers->mla_gate_scale = weights->mla_gate_scale;
	buffers->mla_out_weight = weights->mla_out_weight;
	buffers->mla_out_scale = weights->mla_out_scale;
	buffers->router_weight = weights->router_weight;
	buffers->routed_down_weight = weights->routed_down_weight;
	buffers->routed_down_scale = weights->routed_down_scale;
	buffers->routed_up_weight = weights->routed_up_weight;
	buffers->routed_up_scale = weights->routed_up_scale;
	buffers->routed_norm_weight = weights->routed_norm_weight;
	buffers->expert_w1_weight = weights->expert_w1_weight;
	buffers->expert_w1_scale = weights->expert_w1_scale;
	buffers->expert_w2_weight = weights->expert_w2_weight;
	buffers->expert_w2_scale = weights->expert_w2_scale;
	buffers->shared_w1_weight = weights->shared_w1_weight;
	buffers->shared_w1_scale = weights->shared_w1_scale;
	buffers->shared_w2_weight = weights->shared_w2_weight;
	buffers->shared_w2_scale = weights->shared_w2_scale;
	buffers->dense_gate_up_weight = weights->dense_gate_up_weight;
	buffers->dense_gate_up_scale = weights->dense_gate_up_scale;
	buffers->dense_down_weight = weights->dense_down_weight;
	buffers->dense_down_scale = weights->dense_down_scale;
	buffers->attnres_attn_weight = weights->attnres_attn_weight;
	buffers->attnres_mlp_weight = weights->attnres_mlp_weight;
}

// THE DEFAULT RETURNS AN ERROR. A sixth kind added to LmLayerKind without an arm
// here stops the model instead of running the wrong one, and the compiler warns
// about the unhandled enum value before that.
template<class Format, class Geometry>
static int32_t K3LaunchAttentionHalf(const K3LayerBuffers *buffers, uint32_t layer, uint32_t rows, uint32_t context, uint32_t multiprocessors, cudaStream_t stream)
{
	enum LmLayerKind kind = (enum LmLayerKind)K3_LAYER_KIND(layer);
	switch (kind)
	{
	case LM_LAYER_RECURRENT:
		return(K3LayerKda<Format>(buffers,rows,multiprocessors,stream));
	case LM_LAYER_LATENT:
		return(K3LayerMla<Format,Geometry>(buffers,rows,context,multiprocessors,stream));
	case LM_LAYER_FULL:
	case LM_LAYER_WINDOW:
	case LM_LAYER_SPARSE:
	case LM_LAYER_COMPRESSED:
	case LM_LAYER_KIND_COUNT:
	default:
		return(LM_LAUNCH_ERR_SHAPE);
	}
}

// One stage slice: the layers this rank owns, in order.
//
// The kind comes from the ABSOLUTE layer index, and for this model that matters
// twice over. K3's period is four, which no rank count in use divides, so a rank
// starting mid-period would run the wrong attention on every layer it owns. And
// the last layer is an exception the formula alone does not produce - the
// backbone always ends on global attention - so a rank holding the tail must
// know its absolute position to get that one right.
template<class Format, class Geometry>
static int32_t K3LaunchSlice(const K3LayerWeights *weights, K3LayerBuffers *buffers, uint32_t first_layer, uint32_t layer_count, uint32_t rows, uint32_t packed_rows, uint32_t context, uint32_t multiprocessors, cudaStream_t stream)
{
	uint32_t offset,layer;
	int32_t status;
	for (offset = 0u; offset < layer_count; ++offset)
	{
		layer = first_layer + offset;
		if (layer >= K3_LAYERS)
			return(LM_LAUNCH_ERR_SHAPE);
		K3BindLayer(&weights[offset],buffers);
		// AttnRes replaces the stream before attention and again before the MLP,
		// with different pseudo-queries. The reference applies it from layer 1 -
		// layer 0 has only the embedding in the bank and nothing to retrieve
		// from - so every layer above it reads a bank rather than a residual.
		if ( layer > 0u )
			K3AttnRes(buffers,weights[offset].attnres_attn_weight,layer,rows,stream);
		status = K3LaunchAttentionHalf<Format,Geometry>(buffers,layer,rows,context,
			multiprocessors,stream);
		if (status != LM_LAUNCH_OK)
			return(status);
		// first_k_dense_replace is 1: exactly one full-width feed-forward layer
		// before the MoE stack. Running LatentMoE on layer 0 would use the
		// routed intermediate of 3072 where the model has 33792.
		if ( layer > 0u )
			K3AttnRes(buffers,weights[offset].attnres_mlp_weight,layer,rows,stream);
		if (layer < K3_FIRST_ROUTED_LAYER)
			status = K3LayerDenseMlp<Format>(buffers,rows,multiprocessors,stream);
		else
			status = K3LayerLatentMoe<Format>(buffers,rows,packed_rows,multiprocessors,stream);
		if (status != LM_LAUNCH_OK)
			return(status);
		// The block in progress accumulates this layer's output, and closes if
		// this was its twelfth.
		K3AttnResAccumulate(buffers,layer,rows,stream);
	}
	return(LM_LAUNCH_OK);
}

extern "C" int32_t K3StageSlice(const void *layer_weights, void *layer_buffers, uint32_t first_layer, uint32_t layer_count, uint32_t rows, uint32_t packed_rows, uint32_t context, uint32_t multiprocessors, void *stream)
{
	// THE FORMAT FOLLOWS THE CHECKPOINT'S RECIPE, NOT A GLOBAL CHOICE.
	//
	// This ran LmInt7 across the whole model. K3's quantization_config quantises
	// the routed experts to MXFP4 group 32 and its ignore list excludes
	// attention, latent projections, shared experts, routers and lm_head - and
	// the report says the quantisation-aware training ran from SFT onward, so
	// the routed experts were trained INTO that grid and nothing else was.
	//
	// Requantising attention to INT7 is off-recipe in the same way that storing
	// derived factors at MXFP4 would be. The grid is not the protection; the
	// training into the grid is.
	return(K3LaunchSlice<LmMxfp4,K3GlobalKv>(
		(const K3LayerWeights *)layer_weights,
		(K3LayerBuffers *)layer_buffers,
		first_layer,layer_count,rows,packed_rows,context,multiprocessors,
		(cudaStream_t)stream));
}
