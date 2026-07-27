#pragma once

// One DeepSeek V4 decode layer.
//
// The third sequence, and the one that shares most with GLM 5.2 - a latent cache
// and sparse selection over it. What differs is what makes it worth writing
// separately rather than parameterising glm5_2's:
//
//     six experts per token, not eight
//     a SHARED expert every token passes through, added not weighted
//     a sliding window on top of the sparse selection
//     YaRN rope, because the context was extended past training
//     a low-rank query path at rank 1024
//
// The first of those I got wrong in config.h before writing any of this, by
// carrying GLM 5.2's eight across. A constant that looks like another model's
// and is not is the failure this whole llms/ arrangement exists to make visible.

#include "runtime/gemm.cuh"
#include "inference/kernels/norm.cuh"
#include "inference/kernels/attn.cuh"
#include "inference/kernels/topk.cuh"
#include "inference/kernels/project.cuh"
#include "inference/kernels/head.cuh"
#include "inference/llms/deepseek_v4/config.h"

using Dsv4Kv = LmKvLatent<DSV4_KV_BITS, DSV4_HEAD_DIM, DSV4_ROPE_DIM, DSV4_KV_PAGE_SLOTS>;

#define DSV4_LAYER_THREADS 256u
#define DSV4_LAYER_TILE_N 128u
#define DSV4_LAYER_STAGES 2u
#define DSV4_LAYER_WARPS 8u

struct Dsv4LayerBuffers
{
	const void *attn_norm_weight;
	LmLowRankWeights query;
	LmLowRankScratch query_scratch;
	const void *kv_latent_weight;
	const void *kv_latent_scale;
	const void *output_weight;
	const void *output_scale;
	const void *mlp_norm_weight;
	const void *router_weight;
	const void *expert_w1_weight;
	const void *expert_w1_scale;
	const void *expert_w2_weight;
	const void *expert_w2_scale;
	// The shared expert is a dense MLP with its own weights, not expert zero of
	// the routed set.
	const void *shared_w1_weight;
	const void *shared_w1_scale;
	const void *shared_w2_weight;
	const void *shared_w2_scale;

	uint16_t *hidden_bf16;
	uint16_t *residual_bf16;
	uint16_t *normed_bf16;
	uint16_t *query_bf16;
	uint16_t *kv_slot_bf16;
	uint16_t *attention_latent_bf16;
	uint16_t *attention_out_bf16;
	uint8_t *packed_activation;
	uint8_t *packed_scale;
	uint16_t *gate_up_bf16;
	uint16_t *intermediate_bf16;
	uint16_t *expert_out_bf16;
	uint16_t *shared_out_bf16;
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
	uint32_t *selected_positions;
	float *selection_scores;
	float *head_candidate_score;
	uint32_t *head_candidate_token;
	uint32_t *output_token;
	float *output_score;
};

template<class Format>
static int32_t Dsv4LayerAttention(const Dsv4LayerBuffers *b, uint32_t rows, uint32_t context, uint32_t sms, cudaStream_t stream)
{
	LmGemmArguments gemm;
	int32_t status;
	// The sliding window bounds the selection rather than replacing it: DeepSeek
	// V4 selects 512 positions AND restricts to a 128-token window, so the
	// effective set is whichever is smaller. Treating them as alternatives would
	// attend outside the window on any context longer than 512.
	uint32_t budget = context < DSV4_SLIDING_WINDOW ? context : DSV4_SLIDING_WINDOW;
	bool sparse = context > DSV4_INDEX_TOP_K;
	LmFusedResidualRmsNormKernel<DSV4_LAYER_THREADS>
		<<<rows,DSV4_LAYER_THREADS,(DSV4_HIDDEN + 8u) * sizeof(float),stream>>>(
		b->hidden_bf16,b->residual_bf16,(const uint16_t *)b->attn_norm_weight,
		b->residual_bf16,b->normed_bf16,DSV4_HIDDEN,DSV4_RMS_EPSILON);
	// Query through the low-rank path: hidden -> 1024 -> norm -> heads.
	status = LmLowRankProject<Format>(&b->query,&b->query_scratch,b->normed_bf16,
		b->query_bf16,rows,DSV4_LAYER_THREADS,sms,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	// THE QUERY'S ROPE. Omitted in a first version of this file, which the
	// coverage gate caught by noticing DSV4_ROPE_THETA was declared and unused.
	// Query and key must be rotated into the same frame or their dot product
	// measures the angle between frames rather than between contents - the
	// scores stay finite and plausible and the attention is meaningless.
	//
	// The query uses DSV4_ROPE_THETA and the compressed KV path uses
	// DSV4_COMPRESS_ROPE_THETA. Two thetas is not a mistake in the config: the
	// model trains them separately because the compressed path sees a different
	// position distribution.
	LmRopeYarnKernel<DSV4_LAYER_THREADS><<<rows,DSV4_LAYER_THREADS,0,stream>>>(
		b->query_bf16,b->positions,DSV4_ATTN_HEADS * DSV4_HEAD_DIM,
		(DSV4_ATTN_HEADS * DSV4_HEAD_DIM) - DSV4_ROPE_DIM,DSV4_ROPE_DIM,
		DSV4_ROPE_THETA,(float)DSV4_YARN_FACTOR,
		(float)DSV4_YARN_ORIGINAL_POSITIONS,1.0f,32.0f);
	LmQuantiseRowsKernel<Format,DSV4_LAYER_THREADS>
		<<<dim3(rows,DSV4_HIDDEN / Format::kScaleGroup),DSV4_LAYER_THREADS,
		   (Format::kScaleGroup + 8u) * sizeof(float),stream>>>(
		b->normed_bf16,0,b->packed_activation,b->packed_scale,rows,DSV4_HIDDEN);
	memset(&gemm,0,sizeof(gemm));
	gemm.scale_a = (const float *)b->packed_scale;
	gemm.scale_b = (const float *)b->kv_latent_scale;
	gemm.group_row_offset = b->dense_row_offset;
	gemm.group_tile_prefix = b->dense_tile_prefix;
	gemm.output_bf16 = b->kv_slot_bf16;
	status = LmGemmLaunch<Format,DSV4_LAYER_TILE_N,Format::kTileK,DSV4_LAYER_STAGES,DSV4_LAYER_WARPS>(
		&gemm,b->packed_activation,b->kv_latent_weight,rows,rows,DSV4_TOP_K,1u,
		DSV4_HIDDEN,DSV4_HEAD_DIM + DSV4_ROPE_DIM,sms,false,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	// YaRN, not plain rope. The compressed path uses its own theta, which is why
	// the constant is separate rather than shared with the query's.
	LmRopeYarnKernel<DSV4_LAYER_THREADS><<<rows,DSV4_LAYER_THREADS,0,stream>>>(
		b->kv_slot_bf16,b->positions,DSV4_HEAD_DIM + DSV4_ROPE_DIM,DSV4_HEAD_DIM,
		DSV4_ROPE_DIM,DSV4_COMPRESS_ROPE_THETA,(float)DSV4_YARN_FACTOR,
		(float)DSV4_YARN_ORIGINAL_POSITIONS,1.0f,32.0f);
	LmKvStoreKernel<Dsv4Kv,DSV4_LAYER_THREADS><<<rows,DSV4_LAYER_THREADS,0,stream>>>(
		b->cache,b->kv_slot_bf16,b->sequence_of_row,b->positions,rows,
		DSV4_HEAD_DIM + DSV4_ROPE_DIM);
	if ( sparse )
	{
		LmSparseScoreKernel<Dsv4Kv,DSV4_LAYER_THREADS,DSV4_INDEX_DIM>
			<<<dim3(rows,context),DSV4_LAYER_THREADS,0,stream>>>(
			b->query_bf16,b->cache,b->sequence_of_row,b->context_length,
			DSV4_INDEX_HEADS,b->selection_scores);
		LmTopkHistogramKernel<DSV4_LAYER_THREADS><<<rows,DSV4_LAYER_THREADS,0,stream>>>(
			b->selection_scores,context,DSV4_INDEX_TOP_K,(uint32_t *)b->head_candidate_score);
		LmTopkGatherKernel<DSV4_LAYER_THREADS><<<rows,DSV4_LAYER_THREADS,0,stream>>>(
			b->selection_scores,context,DSV4_INDEX_TOP_K,
			(const uint32_t *)b->head_candidate_score,b->selected_positions,0);
	}
	LmAttentionDecodeKernel<Dsv4Kv,DSV4_LAYER_THREADS,DSV4_HEAD_DIM,DSV4_ROPE_DIM>
		<<<dim3(rows,DSV4_ATTN_HEADS),DSV4_LAYER_THREADS,0,stream>>>(
		b->query_bf16,b->query_bf16,b->cache,b->sequence_of_row,b->context_length,
		sparse ? b->selected_positions : 0,sparse ? DSV4_INDEX_TOP_K : budget,
		DSV4_ATTN_HEADS,rsqrtf((float)DSV4_HEAD_DIM),b->attention_latent_bf16);
	LmQuantiseRowsKernel<Format,DSV4_LAYER_THREADS>
		<<<dim3(rows,(DSV4_ATTN_HEADS * DSV4_HEAD_DIM) / Format::kScaleGroup),DSV4_LAYER_THREADS,
		   (Format::kScaleGroup + 8u) * sizeof(float),stream>>>(
		b->attention_latent_bf16,0,b->packed_activation,b->packed_scale,
		rows,DSV4_ATTN_HEADS * DSV4_HEAD_DIM);
	gemm.scale_a = (const float *)b->packed_scale;
	gemm.scale_b = (const float *)b->output_scale;
	gemm.output_bf16 = b->attention_out_bf16;
	return(LmGemmLaunch<Format,DSV4_LAYER_TILE_N,Format::kTileK,DSV4_LAYER_STAGES,DSV4_LAYER_WARPS>(
		&gemm,b->packed_activation,b->output_weight,rows,rows,DSV4_TOP_K,1u,
		DSV4_ATTN_HEADS * DSV4_HEAD_DIM,DSV4_HIDDEN,sms,false,stream));
}

// The MLP half, with the shared expert.
//
// Every token passes through the shared expert IN ADDITION to its six routed
// ones, and its output is added rather than weighted - it has no gate and is not
// part of the top-k. Omitting it drops a dense contribution from every token,
// which degrades quality uniformly rather than visibly, and is the kind of thing
// that reads as "the port is a bit worse" rather than as a missing term.
//
// It is computed on the same normed rows as the routed path, so the norm happens
// once and both read it.
template<class Format>
static int32_t Dsv4LayerMoe(const Dsv4LayerBuffers *b, uint32_t rows, uint32_t packed_rows, uint32_t sms, cudaStream_t stream)
{
	LmGemmArguments gemm;
	int32_t status;
	LmFusedResidualRmsNormKernel<DSV4_LAYER_THREADS>
		<<<rows,DSV4_LAYER_THREADS,(DSV4_HIDDEN + 8u) * sizeof(float),stream>>>(
		b->attention_out_bf16,b->residual_bf16,(const uint16_t *)b->mlp_norm_weight,
		b->residual_bf16,b->normed_bf16,DSV4_HIDDEN,DSV4_RMS_EPSILON);
	memset(&gemm,0,sizeof(gemm));
	gemm.group_row_offset = b->dense_row_offset;
	gemm.group_tile_prefix = b->dense_tile_prefix;
	gemm.output_bf16 = b->router_logits;
	status = LmGemmLaunch<LmBf16Format,DSV4_LAYER_TILE_N,LmBf16Format::kTileK,DSV4_LAYER_STAGES,DSV4_LAYER_WARPS>(
		&gemm,b->normed_bf16,b->router_weight,rows,rows,DSV4_TOP_K,1u,
		DSV4_HIDDEN,DSV4_EXPERTS,sms,false,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	LmTopkSmallKernel<DSV4_LAYER_THREADS,DSV4_TOP_K>
		<<<rows,DSV4_LAYER_THREADS,2u * LM_TOPK_SMALL_LIMIT * sizeof(uint32_t),stream>>>(
		(const float *)b->router_logits,DSV4_EXPERTS,b->route_expert,b->route_weight,0.0f);
	// The shared expert, on the same normed rows, before the routed path
	// overwrites the packed activation buffer with the expanded copy.
	LmQuantiseRowsKernel<Format,DSV4_LAYER_THREADS>
		<<<dim3(rows,DSV4_HIDDEN / Format::kScaleGroup),DSV4_LAYER_THREADS,
		   (Format::kScaleGroup + 8u) * sizeof(float),stream>>>(
		b->normed_bf16,0,b->packed_activation,b->packed_scale,rows,DSV4_HIDDEN);
	gemm.scale_a = (const float *)b->packed_scale;
	gemm.scale_b = (const float *)b->shared_w1_scale;
	gemm.output_bf16 = b->gate_up_bf16;
	status = LmGemmLaunch<Format,DSV4_LAYER_TILE_N,Format::kTileK,DSV4_LAYER_STAGES,DSV4_LAYER_WARPS>(
		&gemm,b->packed_activation,b->shared_w1_weight,rows,rows,DSV4_TOP_K,1u,
		DSV4_HIDDEN,DSV4_SHARED_INTERMEDIATE * 2u,sms,false,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	LmSiluMulKernel<DSV4_LAYER_THREADS><<<rows,DSV4_LAYER_THREADS,0,stream>>>(
		b->gate_up_bf16,b->intermediate_bf16,DSV4_SHARED_INTERMEDIATE,true);
	LmQuantiseRowsKernel<Format,DSV4_LAYER_THREADS>
		<<<dim3(rows,DSV4_SHARED_INTERMEDIATE / Format::kScaleGroup),DSV4_LAYER_THREADS,
		   (Format::kScaleGroup + 8u) * sizeof(float),stream>>>(
		b->intermediate_bf16,0,b->packed_activation,b->packed_scale,
		rows,DSV4_SHARED_INTERMEDIATE);
	gemm.scale_b = (const float *)b->shared_w2_scale;
	gemm.output_bf16 = b->shared_out_bf16;
	status = LmGemmLaunch<Format,DSV4_LAYER_TILE_N,Format::kTileK,DSV4_LAYER_STAGES,DSV4_LAYER_WARPS>(
		&gemm,b->packed_activation,b->shared_w2_weight,rows,rows,DSV4_TOP_K,1u,
		DSV4_SHARED_INTERMEDIATE,DSV4_HIDDEN,sms,false,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	// The routed path. Expands into packed_rows, so it must come after the
	// shared expert has finished with the un-expanded buffer.
	LmQuantiseRowsKernel<Format,DSV4_LAYER_THREADS>
		<<<dim3(packed_rows,DSV4_HIDDEN / Format::kScaleGroup),DSV4_LAYER_THREADS,
		   (Format::kScaleGroup + 8u) * sizeof(float),stream>>>(
		b->normed_bf16,b->route_source_token,b->packed_activation,b->packed_scale,
		packed_rows,DSV4_HIDDEN);
	gemm.scale_a = (const float *)b->packed_scale;
	gemm.scale_b = (const float *)b->expert_w1_scale;
	gemm.group_row_offset = b->group_row_offset;
	gemm.group_tile_prefix = b->group_tile_prefix;
	gemm.output_bf16 = b->gate_up_bf16;
	status = LmGemmLaunch<Format,DSV4_LAYER_TILE_N,Format::kTileK,DSV4_LAYER_STAGES,DSV4_LAYER_WARPS>(
		&gemm,b->packed_activation,b->expert_w1_weight,packed_rows,rows,DSV4_TOP_K,
		DSV4_EXPERTS,DSV4_HIDDEN,DSV4_EXPERT_INTERMEDIATE * 2u,sms,true,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	LmSiluMulKernel<DSV4_LAYER_THREADS><<<packed_rows,DSV4_LAYER_THREADS,0,stream>>>(
		b->gate_up_bf16,b->intermediate_bf16,DSV4_EXPERT_INTERMEDIATE,true);
	LmQuantiseRowsKernel<Format,DSV4_LAYER_THREADS>
		<<<dim3(packed_rows,DSV4_EXPERT_INTERMEDIATE / Format::kScaleGroup),DSV4_LAYER_THREADS,
		   (Format::kScaleGroup + 8u) * sizeof(float),stream>>>(
		b->intermediate_bf16,0,b->packed_activation,b->packed_scale,
		packed_rows,DSV4_EXPERT_INTERMEDIATE);
	gemm.scale_b = (const float *)b->expert_w2_scale;
	gemm.output_bf16 = b->expert_out_bf16;
	status = LmGemmLaunch<Format,DSV4_LAYER_TILE_N,Format::kTileK,DSV4_LAYER_STAGES,DSV4_LAYER_WARPS>(
		&gemm,b->packed_activation,b->expert_w2_weight,packed_rows,rows,DSV4_TOP_K,
		DSV4_EXPERTS,DSV4_EXPERT_INTERMEDIATE,DSV4_HIDDEN,sms,true,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	// Fold the routed outputs, then add the shared one. DSV4_ROUTED_SCALE
	// multiplies the gates, which the host applies to route_weight - scaling
	// after the sum is the same number only if the gates sum to one.
	LmMoeFinalizeKernel<DSV4_LAYER_THREADS>
		<<<dim3((DSV4_HIDDEN + DSV4_LAYER_THREADS - 1u) / DSV4_LAYER_THREADS,rows),
		   DSV4_LAYER_THREADS,0,stream>>>(
		b->expert_out_bf16,b->route_packed_row,b->route_weight,b->hidden_bf16,
		rows,DSV4_TOP_K,DSV4_HIDDEN);
	LmAddRowsKernel<DSV4_LAYER_THREADS>
		<<<dim3((DSV4_HIDDEN + DSV4_LAYER_THREADS - 1u) / DSV4_LAYER_THREADS,rows),
		   DSV4_LAYER_THREADS,0,stream>>>(
		b->hidden_bf16,b->shared_out_bf16,b->hidden_bf16,rows,DSV4_HIDDEN);
	return(cudaPeekAtLastError() == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
}
