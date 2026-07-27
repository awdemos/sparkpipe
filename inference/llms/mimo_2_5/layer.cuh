#pragma once

// One MiMo 2.5 decode layer.
//
// Written second, after llms/glm5_2/layer.cuh, and deliberately not by copying
// it. The two models differ in ways that would have made a copy wrong in exactly
// the manner the first one was wrong nine times:
//
//     glm5_2                        mimo_2_5
//     four separate projections     one fused QKV
//     latent cache, 1152 B/slot     per-head K and V, 3072 or 6144
//     one attention kind            two, chosen per layer
//     one rope theta                two, chosen per layer
//     rope on its own buffers       rope on a suffix of every head
//
// What is shared is every kernel. kernels/ did not change to accommodate this
// file; project.cuh gained two primitives that GLM 5.2 does not use and MiMo 2.5
// does, which is the difference between a shared library and a common one.

#include "runtime/gemm.cuh"
#include "inference/kernels/norm.cuh"
#include "inference/kernels/attn.cuh"
#include "inference/kernels/topk.cuh"
#include "inference/kernels/project.cuh"
#include "inference/kernels/head.cuh"
#include "inference/llms/mimo_2_5/config.h"

using Mimo25FullKv = LmKvHeads<MIMO25_KV_BITS, MIMO25_FULL_KV_HEADS, MIMO25_HEAD_DIM, MIMO25_KV_PAGE_SLOTS>;
using Mimo25SwaKv  = LmKvHeads<MIMO25_KV_BITS, MIMO25_SWA_KV_HEADS, MIMO25_HEAD_DIM, MIMO25_KV_PAGE_SLOTS>;

#define MIMO25_LAYER_THREADS 256u
#define MIMO25_LAYER_TILE_N 128u
#define MIMO25_LAYER_STAGES 2u
#define MIMO25_LAYER_WARPS 8u

struct Mimo25LayerBuffers
{
	const void *attn_norm_weight;
	const void *qkv_weight;
	const void *qkv_scale;
	const void *output_weight;
	const void *output_scale;
	const void *mlp_norm_weight;
	const void *router_weight;
	const void *dense_gate_up_weight;
	const void *dense_gate_up_scale;
	const void *dense_down_weight;
	const void *dense_down_scale;
	const void *expert_w1_weight;
	const void *expert_w1_scale;
	const void *expert_w2_weight;
	const void *expert_w2_scale;

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
	uint16_t *expert_out_bf16;
	uint16_t *router_logits;
	uint32_t *route_expert;
	float *route_weight;
	uint32_t *route_source_token;
	uint32_t *route_packed_row;
	const uint32_t *dense_row_offset;
	const uint32_t *dense_tile_prefix;
	const uint32_t *group_row_offset;
	const uint32_t *group_tile_prefix;

	LmKvView cache;
	const uint32_t *sequence_of_row;
	const uint32_t *context_length;
	const uint32_t *positions;
	uint32_t *window_positions;
	float *head_candidate_score;
	uint32_t *head_candidate_token;
	uint32_t *output_token;
	float *output_score;
};

// Attention. The layer kind selects the projection width, the rope theta, the
// KV geometry and whether the window applies - four things from one flag, which
// is why it is a template parameter rather than an argument.
template<class Format, class Geometry, uint32_t KV_HEADS, uint32_t QKV_DIM>
static int32_t Mimo25LayerAttention(const Mimo25LayerBuffers *b, uint32_t rows, uint32_t context, uint32_t window, float theta, uint32_t sms, cudaStream_t stream)
{
	LmGemmArguments gemm;
	LmQkvLayout layout;
	int32_t status;
	LmFusedResidualRmsNormKernel<MIMO25_LAYER_THREADS>
		<<<rows,MIMO25_LAYER_THREADS,(MIMO25_HIDDEN + 8u) * sizeof(float),stream>>>(
		b->hidden_bf16,b->residual_bf16,(const uint16_t *)b->attn_norm_weight,
		b->residual_bf16,b->normed_bf16,MIMO25_HIDDEN,MIMO25_RMS_EPSILON);
	LmQuantiseRowsKernel<Format,MIMO25_LAYER_THREADS>
		<<<dim3(rows,MIMO25_HIDDEN / Format::kScaleGroup),MIMO25_LAYER_THREADS,
		   (Format::kScaleGroup + 8u) * sizeof(float),stream>>>(
		b->normed_bf16,0,b->packed_activation,b->packed_scale,rows,MIMO25_HIDDEN);
	memset(&gemm,0,sizeof(gemm));
	gemm.scale_a = (const float *)b->packed_scale;
	gemm.scale_b = (const float *)b->qkv_scale;
	gemm.group_row_offset = b->dense_row_offset;
	gemm.group_tile_prefix = b->dense_tile_prefix;
	gemm.output_bf16 = b->fused_qkv_bf16;
	status = LmGemmLaunch<Format,MIMO25_LAYER_TILE_N,Format::kTileK,MIMO25_LAYER_STAGES,MIMO25_LAYER_WARPS>(
		&gemm,b->packed_activation,b->qkv_weight,rows,rows,MIMO25_TOP_K,1u,
		MIMO25_HIDDEN,QKV_DIM,sms,false,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	layout.query_dimension = MIMO25_Q_DIM;
	layout.key_dimension = KV_HEADS * MIMO25_HEAD_DIM;
	layout.value_dimension = KV_HEADS * MIMO25_VALUE_DIM;
	layout.rope_dimension = MIMO25_ROPE_DIM;
	layout.head_dimension = MIMO25_HEAD_DIM;
	LmSplitQkvKernel<MIMO25_LAYER_THREADS><<<rows,MIMO25_LAYER_THREADS,0,stream>>>(
		b->fused_qkv_bf16,layout,b->query_bf16,b->key_bf16,b->value_bf16,
		rows,MIMO25_VALUE_SCALE);
	// Per head, not per row: the rope part is the suffix of every head, so
	// rotating the row's tail would rotate the last head and leave 63 unrotated.
	LmRopePerHeadKernel<MIMO25_LAYER_THREADS>
		<<<dim3(rows,MIMO25_ATTN_HEADS),MIMO25_LAYER_THREADS,0,stream>>>(
		b->query_bf16,b->positions,MIMO25_ATTN_HEADS,MIMO25_HEAD_DIM,MIMO25_ROPE_DIM,theta);
	LmRopePerHeadKernel<MIMO25_LAYER_THREADS>
		<<<dim3(rows,KV_HEADS),MIMO25_LAYER_THREADS,0,stream>>>(
		b->key_bf16,b->positions,KV_HEADS,MIMO25_HEAD_DIM,MIMO25_ROPE_DIM,theta);
	// The slot holds key then value for every KV head, which is what
	// LmKvHeads sizes it for.
	LmKvStoreKernel<Geometry,MIMO25_LAYER_THREADS><<<rows,MIMO25_LAYER_THREADS,0,stream>>>(
		b->cache,b->kv_slot_bf16,b->sequence_of_row,b->positions,rows,
		Geometry::kSlotBytes / 2u);
	// A sliding window is a position list, the same argument the sparse path
	// takes. window == 0 means attend over everything.
	LmAttentionDecodeKernel<Geometry,MIMO25_LAYER_THREADS,MIMO25_HEAD_DIM,MIMO25_ROPE_DIM>
		<<<dim3(rows,MIMO25_ATTN_HEADS),MIMO25_LAYER_THREADS,0,stream>>>(
		b->query_bf16,b->query_bf16,b->cache,b->sequence_of_row,b->context_length,
		window != 0u ? b->window_positions : 0,window,MIMO25_ATTN_HEADS,
		rsqrtf((float)MIMO25_HEAD_DIM),b->attention_out_bf16,
		0);
	LmQuantiseRowsKernel<Format,MIMO25_LAYER_THREADS>
		<<<dim3(rows,MIMO25_O_INPUT_DIM / Format::kScaleGroup),MIMO25_LAYER_THREADS,
		   (Format::kScaleGroup + 8u) * sizeof(float),stream>>>(
		b->attention_out_bf16,0,b->packed_activation,b->packed_scale,rows,MIMO25_O_INPUT_DIM);
	gemm.scale_a = (const float *)b->packed_scale;
	gemm.scale_b = (const float *)b->output_scale;
	gemm.output_bf16 = b->attention_out_bf16;
	(void)context;
	return(LmGemmLaunch<Format,MIMO25_LAYER_TILE_N,Format::kTileK,MIMO25_LAYER_STAGES,MIMO25_LAYER_WARPS>(
		&gemm,b->packed_activation,b->output_weight,rows,rows,MIMO25_TOP_K,1u,
		MIMO25_O_INPUT_DIM,MIMO25_HIDDEN,sms,false,stream));
}

// The MLP half. Routed and dense, the same pair GLM 5.2 has, at MiMo 2.5's
// widths - 2048 per expert against a dense 16384.
//
// Identical in shape to glm5_2's because the MoE really is the same computation:
// route, expand, two GEMMs with a gated activation between, fold back. The
// models differ in attention and agree here, which is why both call the same
// kernels with different constants rather than sharing a function that takes
// both models' constants as arguments.
template<class Format>
static int32_t Mimo25LayerMoe(const Mimo25LayerBuffers *b, uint32_t rows, uint32_t packed_rows, uint32_t sms, cudaStream_t stream)
{
	LmGemmArguments gemm;
	int32_t status;
	LmFusedResidualRmsNormKernel<MIMO25_LAYER_THREADS>
		<<<rows,MIMO25_LAYER_THREADS,(MIMO25_HIDDEN + 8u) * sizeof(float),stream>>>(
		b->attention_out_bf16,b->residual_bf16,(const uint16_t *)b->mlp_norm_weight,
		b->residual_bf16,b->normed_bf16,MIMO25_HIDDEN,MIMO25_RMS_EPSILON);
	memset(&gemm,0,sizeof(gemm));
	gemm.group_row_offset = b->dense_row_offset;
	gemm.group_tile_prefix = b->dense_tile_prefix;
	gemm.output_bf16 = b->router_logits;
	status = LmGemmLaunch<LmBf16Format,MIMO25_LAYER_TILE_N,LmBf16Format::kTileK,MIMO25_LAYER_STAGES,MIMO25_LAYER_WARPS>(
		&gemm,b->normed_bf16,b->router_weight,rows,rows,MIMO25_TOP_K,1u,
		MIMO25_HIDDEN,MIMO25_EXPERTS,sms,false,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	LmTopkSmallKernel<MIMO25_LAYER_THREADS,MIMO25_TOP_K>
		<<<rows,MIMO25_LAYER_THREADS,2u * LM_TOPK_SMALL_LIMIT * sizeof(uint32_t),stream>>>(
		(const float *)b->router_logits,MIMO25_EXPERTS,b->route_expert,b->route_weight,0.0f);
	LmQuantiseRowsKernel<Format,MIMO25_LAYER_THREADS>
		<<<dim3(packed_rows,MIMO25_HIDDEN / Format::kScaleGroup),MIMO25_LAYER_THREADS,
		   (Format::kScaleGroup + 8u) * sizeof(float),stream>>>(
		b->normed_bf16,b->route_source_token,b->packed_activation,b->packed_scale,
		packed_rows,MIMO25_HIDDEN);
	gemm.scale_a = (const float *)b->packed_scale;
	gemm.scale_b = (const float *)b->expert_w1_scale;
	gemm.group_row_offset = b->group_row_offset;
	gemm.group_tile_prefix = b->group_tile_prefix;
	gemm.output_bf16 = b->gate_up_bf16;
	status = LmGemmLaunch<Format,MIMO25_LAYER_TILE_N,Format::kTileK,MIMO25_LAYER_STAGES,MIMO25_LAYER_WARPS>(
		&gemm,b->packed_activation,b->expert_w1_weight,packed_rows,rows,MIMO25_TOP_K,
		MIMO25_EXPERTS,MIMO25_HIDDEN,MIMO25_GATE_UP_DIM,sms,true,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	LmSiluMulKernel<MIMO25_LAYER_THREADS><<<packed_rows,MIMO25_LAYER_THREADS,0,stream>>>(
		b->gate_up_bf16,b->intermediate_bf16,MIMO25_EXPERT_INTERMEDIATE,true);
	LmQuantiseRowsKernel<Format,MIMO25_LAYER_THREADS>
		<<<dim3(packed_rows,MIMO25_EXPERT_INTERMEDIATE / Format::kScaleGroup),MIMO25_LAYER_THREADS,
		   (Format::kScaleGroup + 8u) * sizeof(float),stream>>>(
		b->intermediate_bf16,0,b->packed_activation,b->packed_scale,
		packed_rows,MIMO25_EXPERT_INTERMEDIATE);
	gemm.scale_b = (const float *)b->expert_w2_scale;
	gemm.output_bf16 = b->expert_out_bf16;
	status = LmGemmLaunch<Format,MIMO25_LAYER_TILE_N,Format::kTileK,MIMO25_LAYER_STAGES,MIMO25_LAYER_WARPS>(
		&gemm,b->packed_activation,b->expert_w2_weight,packed_rows,rows,MIMO25_TOP_K,
		MIMO25_EXPERTS,MIMO25_EXPERT_INTERMEDIATE,MIMO25_HIDDEN,sms,true,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	LmMoeFinalizeKernel<MIMO25_LAYER_THREADS>
		<<<dim3((MIMO25_HIDDEN + MIMO25_LAYER_THREADS - 1u) / MIMO25_LAYER_THREADS,rows),
		   MIMO25_LAYER_THREADS,0,stream>>>(
		b->expert_out_bf16,b->route_packed_row,b->route_weight,b->hidden_bf16,
		rows,MIMO25_TOP_K,MIMO25_HIDDEN);
	return(cudaPeekAtLastError() == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
}

template<class Format>
static int32_t Mimo25LayerDenseMlp(const Mimo25LayerBuffers *b, uint32_t rows, uint32_t sms, cudaStream_t stream)
{
	LmGemmArguments gemm;
	int32_t status;
	LmFusedResidualRmsNormKernel<MIMO25_LAYER_THREADS>
		<<<rows,MIMO25_LAYER_THREADS,(MIMO25_HIDDEN + 8u) * sizeof(float),stream>>>(
		b->attention_out_bf16,b->residual_bf16,(const uint16_t *)b->mlp_norm_weight,
		b->residual_bf16,b->normed_bf16,MIMO25_HIDDEN,MIMO25_RMS_EPSILON);
	LmQuantiseRowsKernel<Format,MIMO25_LAYER_THREADS>
		<<<dim3(rows,MIMO25_HIDDEN / Format::kScaleGroup),MIMO25_LAYER_THREADS,
		   (Format::kScaleGroup + 8u) * sizeof(float),stream>>>(
		b->normed_bf16,0,b->packed_activation,b->packed_scale,rows,MIMO25_HIDDEN);
	memset(&gemm,0,sizeof(gemm));
	gemm.scale_a = (const float *)b->packed_scale;
	gemm.scale_b = (const float *)b->dense_gate_up_scale;
	gemm.group_row_offset = b->dense_row_offset;
	gemm.group_tile_prefix = b->dense_tile_prefix;
	gemm.output_bf16 = b->gate_up_bf16;
	status = LmGemmLaunch<Format,MIMO25_LAYER_TILE_N,Format::kTileK,MIMO25_LAYER_STAGES,MIMO25_LAYER_WARPS>(
		&gemm,b->packed_activation,b->dense_gate_up_weight,rows,rows,MIMO25_TOP_K,1u,
		MIMO25_HIDDEN,MIMO25_DENSE_INTERMEDIATE * 2u,sms,false,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	LmSiluMulKernel<MIMO25_LAYER_THREADS><<<rows,MIMO25_LAYER_THREADS,0,stream>>>(
		b->gate_up_bf16,b->intermediate_bf16,MIMO25_DENSE_INTERMEDIATE,true);
	LmQuantiseRowsKernel<Format,MIMO25_LAYER_THREADS>
		<<<dim3(rows,MIMO25_DENSE_INTERMEDIATE / Format::kScaleGroup),MIMO25_LAYER_THREADS,
		   (Format::kScaleGroup + 8u) * sizeof(float),stream>>>(
		b->intermediate_bf16,0,b->packed_activation,b->packed_scale,
		rows,MIMO25_DENSE_INTERMEDIATE);
	gemm.scale_b = (const float *)b->dense_down_scale;
	gemm.output_bf16 = b->hidden_bf16;
	return(LmGemmLaunch<Format,MIMO25_LAYER_TILE_N,Format::kTileK,MIMO25_LAYER_STAGES,MIMO25_LAYER_WARPS>(
		&gemm,b->packed_activation,b->dense_down_weight,rows,rows,MIMO25_TOP_K,1u,
		MIMO25_DENSE_INTERMEDIATE,MIMO25_HIDDEN,sms,false,stream));
}
