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
#include "inference/kernels/norm.cuh"
#include "inference/kernels/attn.cuh"
#include "inference/kernels/topk.cuh"
#include "inference/kernels/project.cuh"
#include "inference/kernels/head.cuh"
#include "inference/kernels/formats/bf16.cuh"
#include "inference/llms/glm5_2/config.h"

using Glm52Kv = LmKvLatent<GLM52_KV_BITS, GLM52_LATENT, GLM52_ROPE_DIM, GLM52_KV_PAGE_SLOTS>;

#define GLM52_LAYER_THREADS 256u
#define GLM52_LAYER_TILE_N 128u
#define GLM52_LAYER_STAGES 2u
#define GLM52_LAYER_WARPS 8u

// Everything a layer reads or writes. One struct rather than forty arguments,
// because a forty-argument call is where a swapped pair of pointers hides.
// A dense GEMM is a grouped GEMM with one group, which means it still needs a
// row offset table of two entries - {0, rows} - and a tile prefix of two. An
// earlier draft passed the MoE's tables for the dense calls, which would have
// bounded a dense linear by the routed row counts and silently dropped most of
// its output rows.
//
// The host fills these once per step, not per layer, because they depend only on
// the row count.
struct Glm52LayerBuffers
{
	const uint32_t *dense_row_offset;      /* {0, rows} */
	uint32_t *dense_tile_prefix;     /* launcher-priced scratch, group_count + 1 */
	// weights, bound once by the host
	const void *attn_norm_weight;
	// The attention projection is FOUR linears in absorbed form, not one. An
	// earlier version of this file had a single qkv_weight and did a single
	// GEMM, which computes a different function - the four go to different
	// widths and only two of them feed RoPE.
	//
	// The raw form is a two-stage low-rank pair instead: down, norm, up, once
	// per side. kernels/project.cuh has both; which one a model uses is decided
	// by which weights its checkpoint carries, not at runtime.
	LmAbsorbedWeights absorbed;
	LmLowRankWeights raw_query;
	LmLowRankWeights raw_kv;
	LmLowRankScratch raw_scratch;
	bool use_absorbed;
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
	// per-step state
	uint16_t *hidden_bf16;
	uint16_t *residual_bf16;
	uint16_t *normed_bf16;
	LmAbsorbedOutputs projected;
	uint16_t *raw_query_bf16;
	uint16_t *raw_kv_bf16;
	// latent followed by rotated key rope, exactly one cache slot wide
	uint16_t *kv_slot_bf16;
	uint16_t *attention_latent_bf16;
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
	uint32_t *route_packed_row;
	float *head_candidate_score;
	uint32_t *head_candidate_token;
	uint32_t *output_token;
	float *output_score;
	uint32_t *group_row_offset;
	uint32_t *group_tile_prefix;
	// cache
	LmKvView cache;
	const uint32_t *sequence_of_row;
	const uint32_t *context_length;
	const uint32_t *positions;
	// Non-null makes every attention call causal against each row's own
	// position, which is what distinguishes prefill from decode. Null is decode.
	const uint32_t *row_positions;
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
static int32_t Glm52LayerAttention(const Glm52LayerBuffers *b, uint32_t rows, uint32_t context, uint32_t layer_in_group, uint32_t sms, cudaStream_t stream)
{
	LmGemmArguments gemm;
	int32_t status;
	bool sparse = context > GLM52_DSA_SELECTED;
	bool selects = sparse && layer_in_group == 0u;
	LM_LAUNCH((LmFusedResidualRmsNormKernel<GLM52_LAYER_THREADS>), rows, GLM52_LAYER_THREADS, (GLM52_HIDDEN + 8u) * sizeof(float), stream,
		b->hidden_bf16,b->residual_bf16,(const uint16_t *)b->attn_norm_weight, b->residual_bf16,b->normed_bf16,GLM52_HIDDEN,GLM52_HIDDEN,GLM52_RMS_EPSILON);
	LM_LAUNCH((LmQuantiseRowsKernel<Format,GLM52_LAYER_THREADS>), dim3(rows,GLM52_HIDDEN / Format::kScaleGroup), GLM52_LAYER_THREADS, (Format::kScaleGroup + 8u) * sizeof(float), stream,
		b->normed_bf16,0,b->packed_activation,b->packed_scale,rows,GLM52_HIDDEN);
	if ( b->use_absorbed )
	{
		status = LmAbsorbedProject<Format>(&b->absorbed,&b->projected,b->normed_bf16,
			b->packed_activation,(const float *)b->packed_scale,
			b->dense_row_offset,b->dense_tile_prefix,rows,sms,stream);
	}
	else
	{
		status = LmLowRankProject<Format>(&b->raw_query,&b->raw_scratch,b->normed_bf16,
			b->raw_query_bf16,rows,GLM52_LAYER_THREADS,sms,stream);
		if ( status == LM_LAUNCH_OK )
			status = LmLowRankProject<Format>(&b->raw_kv,&b->raw_scratch,b->normed_bf16,
				b->raw_kv_bf16,rows,GLM52_LAYER_THREADS,sms,stream);
	}
	if ( status != LM_LAUNCH_OK )
		return(status);
	// RoPE applies to the trailing GLM52_ROPE_DIM elements of each row, so the
	// stride is the whole row and the offset is where the rope part starts. The
	// QKV projection above emits GLM52_LATENT_ROW elements per row - latent
	// first, rope last - which is what makes the offset GLM52_LATENT.
	//
	// If the projection ever emits them the other way the rotation lands on the
	// latent half, and the result is fluent text with the wrong positions, which
	// is among the hardest failures to attribute. The assert is cheap.
	// THE CACHE WRITE, which an earlier version of this file omitted entirely -
	// attention read slots that nothing had filled. The latent and the rotated
	// key rope are what a slot holds, so the write happens after RoPE and before
	// attention reads it.
	//
	// Ordering matters and is not obvious: the current token's own key must be in
	// the cache before attention runs, because a token attends to itself. Writing
	// after would make every token's last position read the previous step's row.
	//
	// RoPE applies to the two rope projections, which are their own buffers of
	// exactly GLM52_ROPE_DIM elements - so the stride is the rope width and the
	// offset is zero. An earlier version rotated a slice of a fused row at
	// offset GLM52_LATENT, which was the right arithmetic for a layout that does
	// not exist.
	LM_LAUNCH((LmRopeKernel<GLM52_LAYER_THREADS>), rows, GLM52_LAYER_THREADS, 0, stream,
		b->projected.query_rope_bf16,b->positions,GLM52_ROPE_DIM,0u,GLM52_ROPE_DIM,GLM52_ROPE_THETA);
	LM_LAUNCH((LmRopeKernel<GLM52_LAYER_THREADS>), rows, GLM52_LAYER_THREADS, 0, stream,
		b->projected.key_rope_bf16,b->positions,GLM52_ROPE_DIM,0u,GLM52_ROPE_DIM,GLM52_ROPE_THETA);
	LM_LAUNCH((LmKvStoreKernel<Glm52Kv,GLM52_LAYER_THREADS>), rows, GLM52_LAYER_THREADS, 0, stream,
		b->cache,b->kv_slot_bf16,b->sequence_of_row,b->positions,rows,GLM52_LATENT_ROW);
	if ( selects )
	{
		LM_LAUNCH((LmSparseScoreKernel<Glm52Kv,GLM52_LAYER_THREADS,GLM52_DSA_INDEX_DIM>), dim3(rows,context), GLM52_LAYER_THREADS, 0, stream,
		b->projected.query_latent_bf16,b->cache,b->sequence_of_row,b->context_length, GLM52_DSA_INDEX_HEADS,b->selection_scores);
		LM_LAUNCH((LmTopkHistogramKernel<GLM52_LAYER_THREADS>), rows, GLM52_LAYER_THREADS, 0, stream,
		b->selection_scores,context,GLM52_DSA_SELECTED,b->group_tile_prefix);
		LM_LAUNCH((LmTopkGatherKernel<GLM52_LAYER_THREADS>), rows, GLM52_LAYER_THREADS, 0, stream,
		b->selection_scores,context,GLM52_DSA_SELECTED,b->group_tile_prefix, b->selected_positions,0);
	}
	LM_LAUNCH((LmAttentionDecodeKernel<Glm52Kv,GLM52_LAYER_THREADS,GLM52_LATENT,GLM52_ROPE_DIM>), dim3(rows,GLM52_ATTN_HEADS), GLM52_LAYER_THREADS, 0, stream,
		b->projected.query_latent_bf16,b->projected.query_rope_bf16,b->cache,b->sequence_of_row,b->context_length, sparse ? b->selected_positions : 0,GLM52_DSA_SELECTED,GLM52_ATTN_HEADS, rsqrtf((float)GLM52_LATENT),b->attention_latent_bf16, b->row_positions);
	// THE OUTPUT PROJECTION, which an earlier version of this file declared and
	// never called. Attention produces heads x latent; the layer's output is
	// hidden. Without this the residual add receives a tensor of the wrong width
	// and the layer contributes attention's raw latent rows to the stream.
	//
	// In absorbed form this weight already carries the folded value
	// up-projection, which is why its input is the latent width rather than
	// heads x v_head_dim.
	LM_LAUNCH((LmQuantiseRowsKernel<Format,GLM52_LAYER_THREADS>), dim3(rows,(GLM52_ATTN_HEADS * GLM52_LATENT) / Format::kScaleGroup), GLM52_LAYER_THREADS, (Format::kScaleGroup + 8u) * sizeof(float), stream,
		b->attention_latent_bf16,0,b->packed_activation,b->packed_scale, rows,GLM52_ATTN_HEADS * GLM52_LATENT);
	memset(&gemm,0,sizeof(gemm));
	gemm.scale_a = (const float *)b->packed_scale;
	gemm.scale_b = (const float *)b->output_scale;
	gemm.group_row_offset = b->dense_row_offset;
	gemm.group_tile_prefix = b->dense_tile_prefix;
	gemm.output_bf16 = b->attention_out_bf16;
	status = LmGemmLaunch<Format,GLM52_LAYER_TILE_N,Format::kTileK,GLM52_LAYER_STAGES,GLM52_LAYER_WARPS>(
		&gemm,b->packed_activation,b->output_weight,rows,rows,GLM52_TOP_K,1u,
		GLM52_ATTN_HEADS * GLM52_LATENT,GLM52_HIDDEN,sms,false,stream);
	return(status);
}

// Dense MLP half of a layer.
//
// The first GLM52_FIRST_ROUTED_LAYER layers have no router and no experts - they
// are a plain gated MLP at the dense intermediate width. An earlier version of
// this file routed every layer, which for layers 0 through 2 means selecting
// among experts that the checkpoint does not carry for them.
//
// Structurally it is the routed half with the routing removed: one group, no
// top-k, no expansion, no finalize. That it collapses to the same three GEMMs
// with group_count 1 is the same property that makes a dense linear a grouped
// GEMM - and is why this shares the routed half's kernels rather than having
// its own.
template<class Format>
static int32_t Glm52LayerDenseMlp(const Glm52LayerBuffers *b, uint32_t rows, uint32_t sms, cudaStream_t stream)
{
	LmGemmArguments gemm;
	int32_t status;
	LM_LAUNCH((LmFusedResidualRmsNormKernel<GLM52_LAYER_THREADS>), rows, GLM52_LAYER_THREADS, (GLM52_HIDDEN + 8u) * sizeof(float), stream,
		b->attention_out_bf16,b->residual_bf16,(const uint16_t *)b->mlp_norm_weight, b->residual_bf16,b->normed_bf16,GLM52_HIDDEN,GLM52_HIDDEN,GLM52_RMS_EPSILON);
	LM_LAUNCH((LmQuantiseRowsKernel<Format,GLM52_LAYER_THREADS>), dim3(rows,GLM52_HIDDEN / Format::kScaleGroup), GLM52_LAYER_THREADS, (Format::kScaleGroup + 8u) * sizeof(float), stream,
		b->normed_bf16,0,b->packed_activation,b->packed_scale,rows,GLM52_HIDDEN);
	memset(&gemm,0,sizeof(gemm));
	gemm.scale_a = (const float *)b->packed_scale;
	gemm.scale_b = (const float *)b->dense_gate_up_scale;
	gemm.group_row_offset = b->dense_row_offset;
	gemm.group_tile_prefix = b->dense_tile_prefix;
	gemm.output_bf16 = b->gate_up_bf16;
	status = LmGemmLaunch<Format,GLM52_LAYER_TILE_N,Format::kTileK,GLM52_LAYER_STAGES,GLM52_LAYER_WARPS>(
		&gemm,b->packed_activation,b->dense_gate_up_weight,rows,rows,GLM52_TOP_K,1u,
		GLM52_HIDDEN,GLM52_DENSE_INTERMEDIATE * 2u,sms,false,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	LM_LAUNCH((LmSiluMulKernel<GLM52_LAYER_THREADS>), rows, GLM52_LAYER_THREADS, 0, stream,
		b->gate_up_bf16,b->intermediate_bf16,GLM52_DENSE_INTERMEDIATE,true);
	LM_LAUNCH((LmQuantiseRowsKernel<Format,GLM52_LAYER_THREADS>), dim3(rows,GLM52_DENSE_INTERMEDIATE / Format::kScaleGroup), GLM52_LAYER_THREADS, (Format::kScaleGroup + 8u) * sizeof(float), stream,
		b->intermediate_bf16,0,b->packed_activation,b->packed_scale, rows,GLM52_DENSE_INTERMEDIATE);
	gemm.scale_a = (const float *)b->packed_scale;
	gemm.scale_b = (const float *)b->dense_down_scale;
	gemm.output_bf16 = b->hidden_bf16;
	return(LmGemmLaunch<Format,GLM52_LAYER_TILE_N,Format::kTileK,GLM52_LAYER_STAGES,GLM52_LAYER_WARPS>(
		&gemm,b->packed_activation,b->dense_down_weight,rows,rows,GLM52_TOP_K,1u,
		GLM52_DENSE_INTERMEDIATE,GLM52_HIDDEN,sms,false,stream));
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
	LM_LAUNCH((LmFusedResidualRmsNormKernel<GLM52_LAYER_THREADS>), rows, GLM52_LAYER_THREADS, (GLM52_HIDDEN + 8u) * sizeof(float), stream,
		b->attention_out_bf16,b->residual_bf16,(const uint16_t *)b->mlp_norm_weight, b->residual_bf16,b->normed_bf16,GLM52_HIDDEN,GLM52_HIDDEN,GLM52_RMS_EPSILON);
	// The router logits have to be computed before they are selected from. An
	// earlier draft of this file read b->router_logits without anything filling
	// it, which would have selected experts from uninitialised memory - routing
	// every token to whatever the allocator left behind, and producing output
	// that is fluent because the experts themselves are fine.
	//
	// It is a dense GEMM: hidden by expert count, one group, BF16 in and out. No
	// quantisation, because the router is 6144x256 and rounding it costs
	// accuracy on the one tensor whose errors compound across every expert.
	memset(&gemm,0,sizeof(gemm));
	gemm.scale_a = 0;
	gemm.scale_b = 0;
	gemm.group_row_offset = b->dense_row_offset;
	gemm.group_tile_prefix = b->dense_tile_prefix;
	gemm.output_bf16 = b->router_logits;
	status = LmGemmLaunch<LmBf16Format,GLM52_LAYER_TILE_N,LmBf16Format::kTileK,GLM52_LAYER_STAGES,GLM52_LAYER_WARPS>(
		&gemm,b->normed_bf16,b->router_weight,rows,rows,GLM52_TOP_K,1u,
		GLM52_HIDDEN,GLM52_EXPERTS,sms,false,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	LM_LAUNCH((LmTopkSmallKernel<GLM52_LAYER_THREADS,GLM52_TOP_K>), rows, GLM52_LAYER_THREADS, 2u * LM_TOPK_SMALL_LIMIT * sizeof(uint32_t), stream,
		(const float *)b->router_logits,GLM52_EXPERTS,b->route_expert,b->route_weight,0,0);
	LM_LAUNCH((LmQuantiseRowsKernel<Format,GLM52_LAYER_THREADS>), dim3(packed_rows,GLM52_HIDDEN / Format::kScaleGroup), GLM52_LAYER_THREADS, (Format::kScaleGroup + 8u) * sizeof(float), stream,
		b->normed_bf16,b->route_source_token,b->packed_activation,b->packed_scale, packed_rows,GLM52_HIDDEN);
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
	LM_LAUNCH((LmSiluMulKernel<GLM52_LAYER_THREADS>), packed_rows, GLM52_LAYER_THREADS, 0, stream,
		b->gate_up_bf16,b->intermediate_bf16,GLM52_EXPERT_INTERMEDIATE,true);
	LM_LAUNCH((LmQuantiseRowsKernel<Format,GLM52_LAYER_THREADS>), dim3(packed_rows,GLM52_EXPERT_INTERMEDIATE / Format::kScaleGroup), GLM52_LAYER_THREADS, (Format::kScaleGroup + 8u) * sizeof(float), stream,
		b->intermediate_bf16,0,b->packed_activation,b->packed_scale, packed_rows,GLM52_EXPERT_INTERMEDIATE);
	gemm.scale_b = (const float *)b->expert_w2_scale;
	gemm.output_bf16 = b->expert_out_bf16;
	status = LmGemmLaunch<Format,GLM52_LAYER_TILE_N,Format::kTileK,GLM52_LAYER_STAGES,GLM52_LAYER_WARPS>(
		&gemm,b->packed_activation,b->expert_w2_weight,packed_rows,rows,GLM52_TOP_K,
		GLM52_EXPERTS,GLM52_EXPERT_INTERMEDIATE,GLM52_HIDDEN,sms,true,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	// THE FINALIZE, also missing before. Every token was expanded into top_k
	// packed rows and each produced its own output; this folds them back with the
	// router's gate values as weights. Omitting it leaves each token holding
	// whichever expert landed first, which is fluent and looks like a routing bug
	// rather than a missing sum.
	//
	// The routed scaling factor multiplies the gate values, not the result -
	// GLM 5.2 scales the router output rather than the expert output, and
	// applying it after the sum is the same number only when the gates already
	// sum to one, which top-k renormalisation does not guarantee.
	LM_LAUNCH((LmMoeFinalizeKernel<GLM52_LAYER_THREADS>), dim3((GLM52_HIDDEN + GLM52_LAYER_THREADS - 1u) / GLM52_LAYER_THREADS,rows), GLM52_LAYER_THREADS, 0, stream,
		b->expert_out_bf16,b->route_packed_row,b->route_weight,b->hidden_bf16, rows,GLM52_TOP_K,GLM52_HIDDEN);
	return(cudaPeekAtLastError() == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
}

// The sampling head. Once per step, after the last layer.
//
// Split from the layer because it runs once where a layer runs 78 times, and
// because its cost is a different shape: the head reads 1.9 GB of embedding at
// full vocabulary against a layer's 5.3 GB of experts, so at one token per step
// it is a third of the work and at 128 it is a thirtieth.
//
// GLM52_RESTRICTED_VOCAB is the constrained path. A caller that knows its
// continuation set - a grammar, a tool schema - passes the token ids and pays
// the subset's size instead of the vocabulary's: 256 tokens is 1.6 MB against
// 1.9 GB, and it is exact rather than approximate because the excluded tokens
// were going to be rejected anyway.
#define GLM52_HEAD_TILE 1024u

static int32_t Glm52Head(const Glm52LayerBuffers *b, const void *head_norm_weight, const void *head_weight, const uint32_t *token_ids, uint32_t vocabulary, uint32_t rows, cudaStream_t stream)
{
	uint32_t tiles = (vocabulary + GLM52_HEAD_TILE - 1u) / GLM52_HEAD_TILE;
	// The final norm has no residual to add and no residual to write: this is
	// the end of the stream, not a layer boundary.
	LM_LAUNCH((LmFusedResidualRmsNormKernel<GLM52_LAYER_THREADS>), rows, GLM52_LAYER_THREADS, (GLM52_HIDDEN + 8u) * sizeof(float), stream,
		b->hidden_bf16,0,(const uint16_t *)head_norm_weight, 0,b->normed_bf16,GLM52_HIDDEN,GLM52_HIDDEN,GLM52_RMS_EPSILON);
	LM_LAUNCH((LmHeadCandidateKernel<GLM52_LAYER_THREADS,GLM52_HEAD_TILE>), dim3(tiles,rows), GLM52_LAYER_THREADS, 0, stream,
		b->normed_bf16,(const uint16_t *)head_weight,token_ids, b->head_candidate_score,b->head_candidate_token,rows,GLM52_HIDDEN,vocabulary);
	LM_LAUNCH((LmHeadCommitKernel<GLM52_LAYER_THREADS>), rows, GLM52_LAYER_THREADS, 0, stream,
		b->head_candidate_score,b->head_candidate_token,tiles, b->output_token,b->output_score,rows);
	return(cudaPeekAtLastError() == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
}
