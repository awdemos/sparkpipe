#pragma once

// One decode layer, as a sequence of calls into kernels/.
//
// WHAT THIS REPLACES, CONCRETELY. The old stage's LaunchLayerBody is 736 lines
// dispatching thirteen launches and six kernels defined in the same file. Read
// against the sequence below, the correspondence is one to one:
//
//     LaunchRmsNormDimension + ResidualKernel   -> LmFusedResidualRmsNormKernel
//     LaunchRawLinear / LaunchLoweredProjection -> LmGemmKernel
//     LaunchSparseIndexSelection                -> LmSparseScoreKernel + topk
//     LaunchAbsorbedLatentAttention             -> LmAttentionDecodeKernel
//     AttentionKernel + AttentionFp8KvKernel    -> the same, cache format is a
//                                                  trait rather than a kernel
//     LaunchPostAttentionMlp                    -> LmSiluMul + two LmGemmKernel
//     LaunchRestrictedArgmax / Logits           -> LmTopk*
//     LaunchMtpDraft                            -> LmSpeculativeVerify*
//
// Two of those pairs are the interesting ones. AttentionKernel and
// AttentionFp8KvKernel are one kernel here because the cache element width is a
// template parameter, not a separate implementation. And LaunchRawLinear versus
// LaunchLoweredProjection were a dense GEMM and a grouped one; here a dense
// linear is a grouped GEMM with one group.
//
// This is the file that makes the old 27,268-line decode stage deletable. That
// file defines 74 kernels; this one defines none. Every step below is a launch
// of something in kernels/, and what remains here is the order they go in and
// which buffer feeds which - which is the only part that is actually about
// GLM 5.2.
//
// THE ORDER IS THE MODEL. Two models with the same kernels and a different
// sequence are different models; two models with the same sequence and different
// constants are the same model at different sizes. That is why the sequence
// lives in llms/ and the kernels do not.
//
// WHAT IS NOT HERE. No fallbacks, no runtime backend selection, no dispatch on
// anything that could be a template parameter. The old stage carried a
// mlp_execution_mode with six values and a layer_progression_mode with four,
// which between them selected among implementations that mostly did not exist.
// A model that needs a different kernel gets a different instantiation, and the
// choice happens once at build time where a compiler can check it.

#include "runtime/gemm.cuh"
#include "kernels/norm.cuh"
#include "kernels/attn.cuh"
#include "kernels/topk.cuh"
#include "llms/glm5_2/config.h"

using Glm52Kv = LmKvLatent<GLM52_KV_BITS, GLM52_LATENT, GLM52_ROPE_DIM, GLM52_KV_PAGE_SLOTS>;

#define GLM52_LAYER_THREADS 256u
#define GLM52_LAYER_TILE_N 128u
#define GLM52_LAYER_STAGES 2u
#define GLM52_LAYER_WARPS 8u

// Everything a layer reads or writes. One struct rather than forty arguments,
// because a forty-argument call is where a swapped pair of pointers hides.
struct Glm52LayerBuffers
{
	// weights, bound once by the host
	const void *attn_norm_weight;
	const void *qkv_weight;
	const void *qkv_scale;
	const void *output_weight;
	const void *output_scale;
	const void *mlp_norm_weight;
	const void *router_weight;
	const void *expert_w1_weight;
	const void *expert_w1_scale;
	const void *expert_w2_weight;
	const void *expert_w2_scale;
	// per-step state
	uint16_t *hidden_bf16;
	uint16_t *residual_bf16;
	uint16_t *normed_bf16;
	uint16_t *qkv_bf16;
	uint16_t *attention_out_bf16;
	uint8_t *packed_activation;
	uint8_t *packed_scale;
	uint16_t *gate_up_bf16;
	uint16_t *intermediate_bf16;
	uint16_t *expert_out_bf16;
	float *router_logits;
	uint32_t *route_expert;
	float *route_weight;
	uint32_t *route_source_token;
	uint32_t *group_row_offset;
	uint32_t *group_tile_prefix;
	// cache
	LmKvView cache;
	const uint32_t *sequence_of_row;
	const uint32_t *context_length;
	const uint32_t *positions;
	uint32_t *selected_positions;
	float *selection_scores;
};

// Attention half of a layer.
//
// The DSA selection is skipped when the context is shorter than the selection
// budget: picking 2048 of 900 positions is a sort that returns everything, and
// the dense path is the same kernel with a null index. That check is here rather
// than in the kernel because it is a scheduling decision, not an arithmetic one.
template<class Format>
static int32_t Glm52LayerAttention(const Glm52LayerBuffers *b, uint32_t rows, uint32_t context, uint32_t sms, cudaStream_t stream)
{
	LmGemmArguments gemm;
	int32_t status;
	bool sparse = context > GLM52_DSA_SELECTED;
	LmFusedResidualRmsNormKernel<GLM52_LAYER_THREADS>
		<<<rows,GLM52_LAYER_THREADS,(GLM52_HIDDEN + 8u) * sizeof(float),stream>>>(
		b->hidden_bf16,b->residual_bf16,(const uint16_t *)b->attn_norm_weight,
		b->residual_bf16,b->normed_bf16,GLM52_HIDDEN,GLM52_RMS_EPSILON);
	LmQuantiseRowsKernel<Format,GLM52_LAYER_THREADS>
		<<<dim3(rows,GLM52_HIDDEN / Format::kScaleGroup),GLM52_LAYER_THREADS,
		   (Format::kScaleGroup + 8u) * sizeof(float),stream>>>(
		b->normed_bf16,0,b->packed_activation,b->packed_scale,rows,GLM52_HIDDEN);
	memset(&gemm,0,sizeof(gemm));
	gemm.scale_a = (const float *)b->packed_scale;
	gemm.scale_b = (const float *)b->qkv_scale;
	gemm.group_row_offset = b->group_row_offset;
	gemm.group_tile_prefix = b->group_tile_prefix;
	gemm.output_bf16 = b->qkv_bf16;
	status = LmGemmLaunch<Format,GLM52_LAYER_TILE_N,Format::kTileK,GLM52_LAYER_STAGES,GLM52_LAYER_WARPS>(
		&gemm,b->packed_activation,b->qkv_weight,rows,rows,GLM52_TOP_K,1u,
		GLM52_HIDDEN,GLM52_LATENT_ROW,sms,false,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	LmRopeKernel<GLM52_LAYER_THREADS><<<rows,GLM52_LAYER_THREADS,0,stream>>>(
		b->qkv_bf16,b->positions,GLM52_LATENT_ROW,GLM52_LATENT,GLM52_ROPE_DIM,GLM52_ROPE_THETA);
	if ( sparse )
	{
		LmSparseScoreKernel<Glm52Kv,GLM52_LAYER_THREADS,GLM52_DSA_INDEX_DIM>
			<<<dim3(rows,context),GLM52_LAYER_THREADS,0,stream>>>(
			b->qkv_bf16,b->cache,b->sequence_of_row,b->context_length,
			GLM52_DSA_INDEX_HEADS,b->selection_scores);
		LmTopkHistogramKernel<GLM52_LAYER_THREADS><<<rows,GLM52_LAYER_THREADS,0,stream>>>(
			b->selection_scores,context,GLM52_DSA_SELECTED,b->group_tile_prefix);
		LmTopkGatherKernel<GLM52_LAYER_THREADS><<<rows,GLM52_LAYER_THREADS,0,stream>>>(
			b->selection_scores,context,GLM52_DSA_SELECTED,b->group_tile_prefix,
			b->selected_positions,0);
	}
	LmAttentionDecodeKernel<Glm52Kv,GLM52_LAYER_THREADS,GLM52_LATENT,GLM52_ROPE_DIM>
		<<<dim3(rows,GLM52_ATTN_HEADS),GLM52_LAYER_THREADS,0,stream>>>(
		b->qkv_bf16,b->qkv_bf16,b->cache,b->sequence_of_row,b->context_length,
		sparse ? b->selected_positions : 0,GLM52_DSA_SELECTED,GLM52_ATTN_HEADS,
		rsqrtf((float)GLM52_LATENT),b->attention_out_bf16);
	return(cudaPeekAtLastError() == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
}

// Routed MoE half of a layer.
//
// The quantiser takes the route's source-token map, so every packed row is
// written directly. The old stage quantised once per token and replicated the
// row top_k-1 times with a separate gather kernel - a copy that was bit-identical
// to its source and cost 787 MB per pass at B128.
template<class Format>
static int32_t Glm52LayerMoe(const Glm52LayerBuffers *b, uint32_t rows, uint32_t packed_rows, uint32_t sms, cudaStream_t stream)
{
	LmGemmArguments gemm;
	int32_t status;
	LmFusedResidualRmsNormKernel<GLM52_LAYER_THREADS>
		<<<rows,GLM52_LAYER_THREADS,(GLM52_HIDDEN + 8u) * sizeof(float),stream>>>(
		b->attention_out_bf16,b->residual_bf16,(const uint16_t *)b->mlp_norm_weight,
		b->residual_bf16,b->normed_bf16,GLM52_HIDDEN,GLM52_RMS_EPSILON);
	LmTopkSmallKernel<GLM52_LAYER_THREADS,GLM52_TOP_K>
		<<<rows,GLM52_LAYER_THREADS,2u * LM_TOPK_SMALL_LIMIT * sizeof(uint32_t),stream>>>(
		b->router_logits,GLM52_EXPERTS,b->route_expert,b->route_weight,0.0f);
	LmQuantiseRowsKernel<Format,GLM52_LAYER_THREADS>
		<<<dim3(packed_rows,GLM52_HIDDEN / Format::kScaleGroup),GLM52_LAYER_THREADS,
		   (Format::kScaleGroup + 8u) * sizeof(float),stream>>>(
		b->normed_bf16,b->route_source_token,b->packed_activation,b->packed_scale,
		packed_rows,GLM52_HIDDEN);
	memset(&gemm,0,sizeof(gemm));
	gemm.scale_a = (const float *)b->packed_scale;
	gemm.scale_b = (const float *)b->expert_w1_scale;
	gemm.group_row_offset = b->group_row_offset;
	gemm.group_tile_prefix = b->group_tile_prefix;
	gemm.output_bf16 = b->gate_up_bf16;
	status = LmGemmLaunch<Format,GLM52_LAYER_TILE_N,Format::kTileK,GLM52_LAYER_STAGES,GLM52_LAYER_WARPS>(
		&gemm,b->packed_activation,b->expert_w1_weight,packed_rows,rows,GLM52_TOP_K,
		GLM52_EXPERTS,GLM52_HIDDEN,GLM52_GATE_UP_DIM,sms,true,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	LmSiluMulKernel<GLM52_LAYER_THREADS><<<packed_rows,GLM52_LAYER_THREADS,0,stream>>>(
		b->gate_up_bf16,b->intermediate_bf16,GLM52_EXPERT_INTERMEDIATE,true);
	LmQuantiseRowsKernel<Format,GLM52_LAYER_THREADS>
		<<<dim3(packed_rows,GLM52_EXPERT_INTERMEDIATE / Format::kScaleGroup),GLM52_LAYER_THREADS,
		   (Format::kScaleGroup + 8u) * sizeof(float),stream>>>(
		b->intermediate_bf16,0,b->packed_activation,b->packed_scale,
		packed_rows,GLM52_EXPERT_INTERMEDIATE);
	gemm.scale_b = (const float *)b->expert_w2_scale;
	gemm.output_bf16 = b->expert_out_bf16;
	status = LmGemmLaunch<Format,GLM52_LAYER_TILE_N,Format::kTileK,GLM52_LAYER_STAGES,GLM52_LAYER_WARPS>(
		&gemm,b->packed_activation,b->expert_w2_weight,packed_rows,rows,GLM52_TOP_K,
		GLM52_EXPERTS,GLM52_EXPERT_INTERMEDIATE,GLM52_HIDDEN,sms,true,stream);
	return(status);
}
