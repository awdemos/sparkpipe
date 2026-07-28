#pragma once
// Qwen 3.6, one layer.
//
// Nothing sequenced this model's kernels. unity.cu instantiated every one it
// needs - delta rule, causal convolution, KV store, attention, rope, quantise,
// SiLU-mul, head - and no function called them in order, so all seventeen
// constants in config.h were declared and unread and the config gate reported
// the model as "--" rather than failing.
//
// Three of every four layers are gated DeltaNet: a recurrent state, no growing
// cache. The fourth is full attention over a paged KV cache. The host picks by
// layer index through QWEN36_LAYER_IS_LINEAR, which is why the two paths are
// separate entry points rather than a branch - the state pool and the KV pool
// are different geometries and a template parameter is where that belongs.
#include "runtime/gemm.cuh"
#include "runtime/launch.h"
#include "inference/kernels/norm.cuh"
#include "inference/kernels/project.cuh"
#include "inference/kernels/attn.cuh"
#include "inference/kernels/linear_attn.cuh"
#include "inference/kernels/head.cuh"
#include "inference/llms/qwen_3_6/config.h"
#include "inference/kernels/kv.cuh"

// The two pools this model needs. Declared here rather than in unity.cu because
// Qwen36LayerAttention takes the geometry as a template argument, so any file
// that calls a layer needs the alias - bind.cu did, and could not see it.
using Qwen36FullKv = LmKvHeads<QWEN36_KV_BITS, QWEN36_KV_HEADS, QWEN36_HEAD_DIM, QWEN36_KV_PAGE_SLOTS>;
using Qwen36GdnState = LmKvState<QWEN36_GDN_STATE_BYTES>;

#define QWEN36_LAYER_THREADS 256u
#define QWEN36_LAYER_TILE_N 128u
#define QWEN36_LAYER_STAGES 2u
#define QWEN36_LAYER_WARPS 8u
#define QWEN36_HEAD_TILE 1024u

// Full attention widths. Query heads and KV heads differ, so the fused
// projection is query + key + value at the KV count, not three equal thirds.
#define QWEN36_Q_DIM (QWEN36_ATTN_HEADS * QWEN36_HEAD_DIM)
#define QWEN36_KV_DIM (QWEN36_KV_HEADS * QWEN36_HEAD_DIM)
#define QWEN36_ATTN_QKV_DIM (QWEN36_Q_DIM + (2u * QWEN36_KV_DIM))

// Gated DeltaNet widths. 48 value heads over 16 key heads is three value heads
// sharing each key head, which is the ratio the delta rule kernel takes rather
// than a second head count.
#define QWEN36_GDN_QK_DIM (QWEN36_GDN_KEY_HEADS * QWEN36_GDN_KEY_DIM)
#define QWEN36_GDN_V_DIM (QWEN36_GDN_VALUE_HEADS * QWEN36_GDN_VALUE_DIM)
#define QWEN36_GDN_QKV_DIM ((2u * QWEN36_GDN_QK_DIM) + QWEN36_GDN_V_DIM)
#define QWEN36_GDN_VALUE_PER_KEY (QWEN36_GDN_VALUE_HEADS / QWEN36_GDN_KEY_HEADS)

static_assert(QWEN36_GDN_VALUE_HEADS % QWEN36_GDN_KEY_HEADS == 0u,
	"value heads share key heads in whole groups");
static_assert(QWEN36_ATTN_QKV_DIM == QWEN36_QKV_DIM,
	"config and layer must agree on the fused projection width");

struct Qwen36LayerBuffers
{
	const void *attn_norm_weight;
	const void *qkv_weight;
	const void *qkv_scale;
	const void *output_weight;
	const void *output_scale;
	const void *gdn_in_weight;
	const void *gdn_in_scale;
	const void *gdn_conv_weight;
	const void *gdn_out_weight;
	const void *gdn_out_scale;
	const void *mlp_norm_weight;
	const void *dense_gate_up_weight;
	const void *dense_gate_up_scale;
	const void *dense_down_weight;
	const void *dense_down_scale;

	uint16_t *hidden_bf16;
	uint16_t *residual_bf16;
	uint16_t *normed_bf16;
	uint16_t *fused_qkv_bf16;
	uint16_t *query_bf16;
	uint16_t *key_bf16;
	uint16_t *value_bf16;
	uint16_t *kv_slot_bf16;
	uint16_t *attention_out_bf16;
	uint8_t *packed_activation;
	uint8_t *packed_scale;
	uint16_t *gate_up_bf16;
	uint16_t *intermediate_bf16;

	// The recurrent half. The state pool never grows, and the convolution
	// window lives in the same slot - QWEN36_GDN_STATE_BYTES is their sum.
	uint8_t *gdn_state_pool;
	uint16_t *gdn_conv_window;
	const uint32_t *gdn_state_index;
	const float *gdn_forget_gate;
	const float *gdn_write_gate;

	LmKvView cache;
	const uint32_t *sequence_of_row;
	const uint32_t *context_length;
	const uint32_t *positions;
	const uint32_t *dense_row_offset;
	const uint32_t *dense_tile_prefix;
	float *head_candidate_score;
	uint32_t *head_candidate_token;
	uint32_t *output_token;
	float *output_score;
};

// Quantise a row block and point the GEMM at it. Every projection below does
// this same four-line dance, and writing it four times is how a scale pointer
// ends up describing the wrong buffer.
template<class Format>
static void Qwen36Quantise(const Qwen36LayerBuffers *b, const uint16_t *source, uint32_t rows, uint32_t width, cudaStream_t stream)
{
	LmQuantiseRowsKernel<Format,QWEN36_LAYER_THREADS>
		<<<dim3(rows,width / Format::kScaleGroup),QWEN36_LAYER_THREADS,
		   (Format::kScaleGroup + 8u) * sizeof(float),stream>>>(
		source,0,b->packed_activation,b->packed_scale,rows,width);
}

// Full attention, one layer in four.
template<class Format, class Geometry>
static int32_t Qwen36LayerAttention(const Qwen36LayerBuffers *b, uint32_t rows, uint32_t context, uint32_t sms, cudaStream_t stream)
{
	LmGemmArguments gemm;
	LmQkvLayout layout;
	int32_t status;
	LmFusedResidualRmsNormKernel<QWEN36_LAYER_THREADS>
		<<<rows,QWEN36_LAYER_THREADS,(QWEN36_HIDDEN + 8u) * sizeof(float),stream>>>(
		b->hidden_bf16,b->residual_bf16,(const uint16_t *)b->attn_norm_weight,
		b->residual_bf16,b->normed_bf16,QWEN36_HIDDEN,QWEN36_RMS_EPSILON);
	Qwen36Quantise<Format>(b,b->normed_bf16,rows,QWEN36_HIDDEN,stream);
	memset(&gemm,0,sizeof(gemm));
	gemm.scale_a = (const float *)b->packed_scale;
	gemm.scale_b = (const float *)b->qkv_scale;
	gemm.group_row_offset = b->dense_row_offset;
	gemm.group_tile_prefix = b->dense_tile_prefix;
	gemm.output_bf16 = b->fused_qkv_bf16;
	status = LmGemmLaunch<Format,QWEN36_LAYER_TILE_N,Format::kTileK,QWEN36_LAYER_STAGES,QWEN36_LAYER_WARPS>(
		&gemm,b->packed_activation,b->qkv_weight,rows,rows,1u,1u,
		QWEN36_HIDDEN,QWEN36_ATTN_QKV_DIM,sms,false,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	layout.query_dimension = QWEN36_Q_DIM;
	layout.key_dimension = QWEN36_KV_DIM;
	layout.value_dimension = QWEN36_KV_DIM;
	layout.rope_dimension = QWEN36_ROPE_DIM;
	layout.head_dimension = QWEN36_HEAD_DIM;
	LmSplitQkvKernel<QWEN36_LAYER_THREADS><<<rows,QWEN36_LAYER_THREADS,0,stream>>>(
		b->fused_qkv_bf16,layout,b->query_bf16,b->key_bf16,b->value_bf16,rows,1.0f);
	// Partial rotary: rope covers the first 64 of each 256-wide head, so this
	// is per head. Rotating the row tail would rotate one head and leave 23.
	LmRopePerHeadKernel<QWEN36_LAYER_THREADS>
		<<<dim3(rows,QWEN36_ATTN_HEADS),QWEN36_LAYER_THREADS,0,stream>>>(
		b->query_bf16,b->positions,QWEN36_ATTN_HEADS,QWEN36_HEAD_DIM,
		QWEN36_ROPE_DIM,QWEN36_ROPE_THETA);
	LmRopePerHeadKernel<QWEN36_LAYER_THREADS>
		<<<dim3(rows,QWEN36_KV_HEADS),QWEN36_LAYER_THREADS,0,stream>>>(
		b->key_bf16,b->positions,QWEN36_KV_HEADS,QWEN36_HEAD_DIM,
		QWEN36_ROPE_DIM,QWEN36_ROPE_THETA);
	LmKvStoreKernel<Geometry,QWEN36_LAYER_THREADS><<<rows,QWEN36_LAYER_THREADS,0,stream>>>(
		b->cache,b->kv_slot_bf16,b->sequence_of_row,b->positions,rows,
		Geometry::kSlotBytes / 2u);
	LmAttentionDecodeKernel<Geometry,QWEN36_LAYER_THREADS,QWEN36_NOPE_DIM,QWEN36_ROPE_DIM>
		<<<dim3(rows,QWEN36_ATTN_HEADS),QWEN36_LAYER_THREADS,0,stream>>>(
		b->query_bf16,b->query_bf16,b->cache,b->sequence_of_row,b->context_length,
		0,0u,QWEN36_ATTN_HEADS,QWEN36_QK_SCALE,b->attention_out_bf16,0);
	Qwen36Quantise<Format>(b,b->attention_out_bf16,rows,QWEN36_Q_DIM,stream);
	gemm.scale_a = (const float *)b->packed_scale;
	gemm.scale_b = (const float *)b->output_scale;
	gemm.output_bf16 = b->attention_out_bf16;
	(void)context;
	return(LmGemmLaunch<Format,QWEN36_LAYER_TILE_N,Format::kTileK,QWEN36_LAYER_STAGES,QWEN36_LAYER_WARPS>(
		&gemm,b->packed_activation,b->output_weight,rows,rows,1u,1u,
		QWEN36_Q_DIM,QWEN36_HIDDEN,sms,false,stream));
}

// Gated DeltaNet, three layers in four. No cache read: the whole history is in
// a fixed state, which is the property that makes 48 of 64 layers cost the same
// at context 1 and context 256K.
template<class Format>
static int32_t Qwen36LayerLinear(const Qwen36LayerBuffers *b, uint32_t rows, uint32_t sms, cudaStream_t stream)
{
	LmGemmArguments gemm;
	LmQkvLayout layout;
	int32_t status;
	LmFusedResidualRmsNormKernel<QWEN36_LAYER_THREADS>
		<<<rows,QWEN36_LAYER_THREADS,(QWEN36_HIDDEN + 8u) * sizeof(float),stream>>>(
		b->hidden_bf16,b->residual_bf16,(const uint16_t *)b->attn_norm_weight,
		b->residual_bf16,b->normed_bf16,QWEN36_HIDDEN,QWEN36_RMS_EPSILON);
	Qwen36Quantise<Format>(b,b->normed_bf16,rows,QWEN36_HIDDEN,stream);
	memset(&gemm,0,sizeof(gemm));
	gemm.scale_a = (const float *)b->packed_scale;
	gemm.scale_b = (const float *)b->gdn_in_scale;
	gemm.group_row_offset = b->dense_row_offset;
	gemm.group_tile_prefix = b->dense_tile_prefix;
	gemm.output_bf16 = b->fused_qkv_bf16;
	status = LmGemmLaunch<Format,QWEN36_LAYER_TILE_N,Format::kTileK,QWEN36_LAYER_STAGES,QWEN36_LAYER_WARPS>(
		&gemm,b->packed_activation,b->gdn_in_weight,rows,rows,1u,1u,
		QWEN36_HIDDEN,QWEN36_GDN_QKV_DIM,sms,false,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	// No rope on this path: position enters through the recurrence, not a
	// rotation, so rope_dimension is zero rather than unset.
	layout.query_dimension = QWEN36_GDN_QK_DIM;
	layout.key_dimension = QWEN36_GDN_QK_DIM;
	layout.value_dimension = QWEN36_GDN_V_DIM;
	layout.rope_dimension = 0u;
	layout.head_dimension = QWEN36_GDN_KEY_DIM;
	LmSplitQkvKernel<QWEN36_LAYER_THREADS><<<rows,QWEN36_LAYER_THREADS,0,stream>>>(
		b->fused_qkv_bf16,layout,b->query_bf16,b->key_bf16,b->value_bf16,rows,1.0f);
	// The short causal convolution runs before the recurrence and carries its
	// own window in the same non-growing slot as the state.
	// SWISH, INFERRED NOT READ. Gated DeltaNet applies Swish after the short
	// convolution - the K3 report cites GDN for exactly that when describing
	// KDA's q/k/v projections - and Qwen 3.6's linear path is GDN. I have not
	// seen Qwen's modelling file, so this follows the architecture rather than
	// the checkpoint, which is a weaker claim than the K3 side of this tree.
	LmCausalConvDecodeKernel<QWEN36_LAYER_THREADS,QWEN36_GDN_CONV_KERNEL,LM_CONV_SWISH>
		<<<rows,QWEN36_LAYER_THREADS,0,stream>>>(
		b->gdn_conv_window,b->gdn_state_index,b->key_bf16,
		(const uint16_t *)b->gdn_conv_weight,b->key_bf16,QWEN36_GDN_QK_DIM,rows);
	LmDeltaRuleDecodeKernel<QWEN36_LAYER_THREADS,QWEN36_GDN_KEY_DIM,QWEN36_GDN_VALUE_DIM>
		<<<dim3(rows,QWEN36_GDN_KEY_HEADS),QWEN36_LAYER_THREADS,0,stream>>>(
		b->gdn_state_pool,b->gdn_state_index,b->query_bf16,b->key_bf16,b->value_bf16,
		b->gdn_forget_gate,b->gdn_write_gate,b->attention_out_bf16,
		QWEN36_GDN_KEY_HEADS,QWEN36_GDN_VALUE_PER_KEY,rows);
	Qwen36Quantise<Format>(b,b->attention_out_bf16,rows,QWEN36_GDN_V_DIM,stream);
	gemm.scale_a = (const float *)b->packed_scale;
	gemm.scale_b = (const float *)b->gdn_out_scale;
	gemm.output_bf16 = b->attention_out_bf16;
	return(LmGemmLaunch<Format,QWEN36_LAYER_TILE_N,Format::kTileK,QWEN36_LAYER_STAGES,QWEN36_LAYER_WARPS>(
		&gemm,b->packed_activation,b->gdn_out_weight,rows,rows,1u,1u,
		QWEN36_GDN_V_DIM,QWEN36_HIDDEN,sms,false,stream));
}

// Dense SwiGLU on every layer. Qwen 3.6 has no routed experts in this
// configuration, so there is one MLP and no router.
template<class Format>
static int32_t Qwen36LayerDenseMlp(const Qwen36LayerBuffers *b, uint32_t rows, uint32_t sms, cudaStream_t stream)
{
	LmGemmArguments gemm;
	int32_t status;
	LmFusedResidualRmsNormKernel<QWEN36_LAYER_THREADS>
		<<<rows,QWEN36_LAYER_THREADS,(QWEN36_HIDDEN + 8u) * sizeof(float),stream>>>(
		b->attention_out_bf16,b->residual_bf16,(const uint16_t *)b->mlp_norm_weight,
		b->residual_bf16,b->normed_bf16,QWEN36_HIDDEN,QWEN36_RMS_EPSILON);
	Qwen36Quantise<Format>(b,b->normed_bf16,rows,QWEN36_HIDDEN,stream);
	memset(&gemm,0,sizeof(gemm));
	gemm.scale_a = (const float *)b->packed_scale;
	gemm.scale_b = (const float *)b->dense_gate_up_scale;
	gemm.group_row_offset = b->dense_row_offset;
	gemm.group_tile_prefix = b->dense_tile_prefix;
	gemm.output_bf16 = b->gate_up_bf16;
	status = LmGemmLaunch<Format,QWEN36_LAYER_TILE_N,Format::kTileK,QWEN36_LAYER_STAGES,QWEN36_LAYER_WARPS>(
		&gemm,b->packed_activation,b->dense_gate_up_weight,rows,rows,1u,1u,
		QWEN36_HIDDEN,QWEN36_FFN_INTERMEDIATE * 2u,sms,false,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	LmSiluMulKernel<QWEN36_LAYER_THREADS><<<rows,QWEN36_LAYER_THREADS,0,stream>>>(
		b->gate_up_bf16,b->intermediate_bf16,QWEN36_FFN_INTERMEDIATE,true);
	Qwen36Quantise<Format>(b,b->intermediate_bf16,rows,QWEN36_FFN_INTERMEDIATE,stream);
	gemm.scale_b = (const float *)b->dense_down_scale;
	gemm.output_bf16 = b->hidden_bf16;
	return(LmGemmLaunch<Format,QWEN36_LAYER_TILE_N,Format::kTileK,QWEN36_LAYER_STAGES,QWEN36_LAYER_WARPS>(
		&gemm,b->packed_activation,b->dense_down_weight,rows,rows,1u,1u,
		QWEN36_FFN_INTERMEDIATE,QWEN36_HIDDEN,sms,false,stream));
}

static int32_t Qwen36Head(const Qwen36LayerBuffers *b, const void *head_norm_weight, const void *head_weight, const uint32_t *token_ids, uint32_t vocabulary, uint32_t rows, cudaStream_t stream)
{
	uint32_t tiles = (vocabulary + QWEN36_HEAD_TILE - 1u) / QWEN36_HEAD_TILE;
	LmFusedResidualRmsNormKernel<QWEN36_LAYER_THREADS>
		<<<rows,QWEN36_LAYER_THREADS,(QWEN36_HIDDEN + 8u) * sizeof(float),stream>>>(
		b->hidden_bf16,0,(const uint16_t *)head_norm_weight,
		0,b->normed_bf16,QWEN36_HIDDEN,QWEN36_RMS_EPSILON);
	LmHeadCandidateKernel<QWEN36_LAYER_THREADS,QWEN36_HEAD_TILE>
		<<<dim3(tiles,rows),QWEN36_LAYER_THREADS,0,stream>>>(
		b->normed_bf16,(const uint16_t *)head_weight,token_ids,
		b->head_candidate_score,b->head_candidate_token,rows,QWEN36_HIDDEN,vocabulary);
	LmHeadCommitKernel<QWEN36_LAYER_THREADS><<<rows,QWEN36_LAYER_THREADS,0,stream>>>(
		b->head_candidate_score,b->head_candidate_token,tiles,
		b->output_token,b->output_score,rows);
	return(cudaPeekAtLastError() == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
}
