#pragma once

// Projection primitives. Low-rank with an intermediate norm, and its absorbed
// form.
//
// This is a separate module from attention because the pattern outlives any one
// model's attention. A projection that goes down to a low rank, normalises
// there, and comes back up is how every MLA-family model compresses its query
// and KV paths - DeepSeek V2 through V4, GLM 5.2, Kimi K3 - and the only thing
// that differs between them is the ranks.
//
// TWO FORMS OF THE SAME PROJECTION, AND WHY BOTH EXIST.
//
// RAW: hidden -> down -> norm -> up -> per-head keys and values. This is what
// the checkpoint contains and what the arithmetic says.
//
// ABSORBED: the up-projection is folded into the query and output weights at
// pack time, so attention happens directly in the compressed space and per-head
// K and V are never materialised. Four plain linears replace the two-stage path.
//
// The absorbed form is strictly better at decode and strictly worse at prefill,
// which is why a model ships both sets of weights rather than choosing. At
// decode one row attends over a whole cache, so not materialising per-head K and
// V saves the dominant read. At prefill many rows share the cache, the
// materialisation amortises, and the raw form's smaller GEMMs win.
//
// A model that ships only raw weights uses LmLowRankProject. One that ships
// absorbed weights uses four LmGemmLaunch calls and needs nothing from this
// file. GLM 5.2 ships both and selects at bind time, which is a property of the
// checkpoint rather than a runtime mode.

#include "runtime/gemm.cuh"
#include "inference/kernels/norm.cuh"
#include "inference/kernels/attn.cuh"
#include <stdint.h>

// One side of a low-rank projection: the two weights and the norm between them.
//
// Quantisation is per-weight rather than per-projection because the down and up
// matrices have very different shapes - hidden-by-rank against rank-by-output -
// and a rank of 2048 against a hidden of 6144 means the down matrix is three
// times the size. They earn different formats.
struct LmLowRankWeights
{
	const void *down_weight;
	const void *down_scale;
	const void *norm_weight;
	const void *up_weight;
	const void *up_scale;
	uint32_t input_dimension;
	uint32_t rank;
	uint32_t output_dimension;
	float norm_epsilon;
};

// Scratch the projection needs: the compressed rows, and their quantised form.
//
// Sized by rank rather than by output, which is the point of the compression -
// at GLM 5.2's 6144 hidden and 2048 rank this is a third of what a fused
// projection would need, and the second GEMM reads it instead of the hidden.
struct LmLowRankScratch
{
	uint16_t *compressed_bf16;
	uint8_t *compressed_codes;
	uint8_t *compressed_scales;
	const uint32_t *dense_row_offset;
	const uint32_t *dense_tile_prefix;
};

// hidden -> down -> norm -> up.
//
// The norm is INSIDE this primitive rather than the caller's business, because
// it operates on the compressed representation and nothing outside sees that.
// A caller that had to sequence it would need the rank, the scratch and the
// epsilon, which is the whole argument list back again.
//
// The norm is a plain RMS norm with no residual: there is nothing to add at this
// point, the compressed row is not a hidden state. Passing a residual here would
// be adding a 2048-wide vector to something that is not the same tensor.
template<class Format>
static int32_t LmLowRankProject(const LmLowRankWeights *weights, const LmLowRankScratch *scratch, const uint16_t *input_bf16, uint16_t *output_bf16, uint32_t rows, uint32_t threads, uint32_t multiprocessors, cudaStream_t stream)
{
	LmGemmArguments gemm;
	int32_t status;
	memset(&gemm,0,sizeof(gemm));
	gemm.scale_a = 0;
	gemm.scale_b = (const float *)weights->down_scale;
	gemm.group_row_offset = scratch->dense_row_offset;
	gemm.group_tile_prefix = scratch->dense_tile_prefix;
	gemm.output_bf16 = scratch->compressed_bf16;
	status = LmGemmLaunch<Format,128u,Format::kTileK,LM_PIPELINE_STAGES,8u>(
		&gemm,input_bf16,weights->down_weight,rows,rows,1u,1u,
		weights->input_dimension,weights->rank,multiprocessors,false,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	// No residual: the compressed row is not a hidden state and has nothing to
	// add. The output is written back over the same buffer, which is safe
	// because the norm reads a row before it writes it.
	LmFusedResidualRmsNormKernel<256u><<<rows,threads,(weights->rank + 8u) * sizeof(float),stream>>>(
		scratch->compressed_bf16,0,(const uint16_t *)weights->norm_weight,
		0,scratch->compressed_bf16,weights->rank,weights->norm_epsilon);
	LmQuantiseRowsKernel<Format,256u><<<dim3(rows,weights->rank / Format::kScaleGroup),threads,
		(Format::kScaleGroup + 8u) * sizeof(float),stream>>>(
		scratch->compressed_bf16,0,scratch->compressed_codes,scratch->compressed_scales,
		rows,weights->rank);
	gemm.scale_a = (const float *)scratch->compressed_scales;
	gemm.scale_b = (const float *)weights->up_scale;
	gemm.output_bf16 = output_bf16;
	return(LmGemmLaunch<Format,128u,Format::kTileK,LM_PIPELINE_STAGES,8u>(
		&gemm,scratch->compressed_codes,weights->up_weight,rows,rows,1u,1u,
		weights->rank,weights->output_dimension,multiprocessors,false,stream));
}

// The absorbed form: four plain projections from the hidden state.
//
// query-latent and kv-latent go into the compressed space directly; the two rope
// projections produce the positional halves. There is no norm and no
// intermediate, because the folding happened at pack time.
//
// Grouped as one call rather than four at the call site because the four share
// an input and a row count, and because getting one of the four pointed at the
// wrong weight is the kind of mistake a four-field struct prevents and four
// separate calls invite.
struct LmAbsorbedWeights
{
	const void *query_latent_weight;
	const void *query_latent_scale;
	const void *query_rope_weight;
	const void *query_rope_scale;
	const void *key_rope_weight;
	const void *key_rope_scale;
	const void *kv_latent_weight;
	const void *kv_latent_scale;
	uint32_t input_dimension;
	uint32_t query_latent_dimension;
	uint32_t rope_dimension;
	uint32_t kv_latent_dimension;
};

struct LmAbsorbedOutputs
{
	uint16_t *query_latent_bf16;
	uint16_t *query_rope_bf16;
	uint16_t *key_rope_bf16;
	uint16_t *kv_latent_bf16;
};

template<class Format>
static int32_t LmAbsorbedProject(const LmAbsorbedWeights *weights, const LmAbsorbedOutputs *out, const uint16_t *input_bf16, const uint8_t *input_codes, const float *input_scales, const uint32_t *dense_row_offset, const uint32_t *dense_tile_prefix, uint32_t rows, uint32_t multiprocessors, cudaStream_t stream)
{
	struct { const void *weight; const void *scale; uint16_t *out; uint32_t width; } pass[4] = {
		{ weights->query_latent_weight, weights->query_latent_scale, out->query_latent_bf16, weights->query_latent_dimension },
		{ weights->query_rope_weight,   weights->query_rope_scale,   out->query_rope_bf16,   weights->rope_dimension },
		{ weights->key_rope_weight,     weights->key_rope_scale,     out->key_rope_bf16,     weights->rope_dimension },
		{ weights->kv_latent_weight,    weights->kv_latent_scale,    out->kv_latent_bf16,    weights->kv_latent_dimension },
	};
	LmGemmArguments gemm;
	int32_t status;
	uint32_t index;
	(void)input_bf16;
	for (index = 0u; index < 4u; ++index)
	{
		memset(&gemm,0,sizeof(gemm));
		gemm.scale_a = input_scales;
		gemm.scale_b = (const float *)pass[index].scale;
		gemm.group_row_offset = dense_row_offset;
		gemm.group_tile_prefix = dense_tile_prefix;
		gemm.output_bf16 = pass[index].out;
		status = LmGemmLaunch<Format,128u,Format::kTileK,LM_PIPELINE_STAGES,8u>(
			&gemm,input_codes,pass[index].weight,rows,rows,1u,1u,
			weights->input_dimension,pass[index].width,multiprocessors,false,stream);
		if ( status != LM_LAUNCH_OK )
			return(status);
	}
	return(LM_LAUNCH_OK);
}

// -- fused QKV ------------------------------------------------------------------
//
// The other shape a model's attention projection takes: one GEMM producing
// query, key and value concatenated per row, split afterwards.
//
// MiMo 2.5 does this where GLM 5.2 does four separate projections, and neither
// is a variant of the other. A fused projection is one large GEMM with better
// arithmetic intensity; four separate ones let each output have its own
// quantisation and let a latent-absorbed model skip materialising K and V at
// all. Which a model uses is in its checkpoint, not a choice at run time.
//
// The split is a copy rather than a view because the three parts go to different
// places - query to RoPE and then attention, key and value into a cache slot -
// and a view would make every consumer carry the row stride and the offset. One
// copy of a decode row is 27 KB at MiMo 2.5's widths, which is nothing against
// the projection that produced it.
struct LmQkvLayout
{
	uint32_t query_dimension;      /* heads * qk_head_dim */
	uint32_t key_dimension;        /* kv_heads * qk_head_dim */
	uint32_t value_dimension;      /* kv_heads * v_head_dim */
	uint32_t rope_dimension;       /* rotated suffix of each qk head */
	uint32_t head_dimension;       /* qk dim per head, for locating the rope part */
};

// Query, key and value out of a fused row.
//
// The value scale is applied here rather than after attention because it belongs
// to the value tensor: MiMo 2.5 carries a 0.707 factor on V, and folding it into
// the attention output instead is the same number only when the softmax weights
// sum to one - which they do, but the equality stops holding the moment anything
// masks a position after the softmax. Scaling the tensor it belongs to survives
// that.
template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmSplitQkvKernel(const uint16_t *__restrict__ fused_bf16, LmQkvLayout layout, uint16_t *__restrict__ query_bf16, uint16_t *__restrict__ key_bf16, uint16_t *__restrict__ value_bf16, uint32_t rows, float value_scale)
{
	uint32_t row = blockIdx.x,index;
	uint32_t total = layout.query_dimension + layout.key_dimension + layout.value_dimension;
	uint64_t base = (uint64_t)row * total;
	if ( row >= rows )
		return;
	for (index = threadIdx.x; index < layout.query_dimension; index += THREADS)
		query_bf16[((uint64_t)row * layout.query_dimension) + index] = fused_bf16[base + index];
	for (index = threadIdx.x; index < layout.key_dimension; index += THREADS)
		key_bf16[((uint64_t)row * layout.key_dimension) + index] =
			fused_bf16[base + layout.query_dimension + index];
	for (index = threadIdx.x; index < layout.value_dimension; index += THREADS)
		value_bf16[((uint64_t)row * layout.value_dimension) + index] =
			LmFloatToBf16(LmBf16ToFloat(
				fused_bf16[base + layout.query_dimension + layout.key_dimension + index])
				* value_scale);
}

// RoPE over the rotated suffix of every head in a packed multi-head row.
//
// A fused query row is heads x head_dimension with the rope part at the end of
// each head, not at the end of the row. Rotating the row's tail would rotate the
// last head only and leave the other sixty-three unrotated - which produces
// fluent text whose attention ignores position for all but one head.
template<uint32_t THREADS, LmRopePairing PAIRING = LM_ROPE_HALF_SPLIT>
__global__ __launch_bounds__(THREADS, 1)
void LmRopePerHeadKernel(uint16_t *__restrict__ rows_bf16, const uint32_t *__restrict__ positions, uint32_t heads, uint32_t head_dimension, uint32_t rope_dimension, float theta)
{
	uint32_t row = blockIdx.x,head = blockIdx.y,index;
	uint32_t half = rope_dimension / 2u;
	uint64_t base = (((uint64_t)row * heads) + head) * head_dimension
		+ (head_dimension - rope_dimension);
	float position = (float)positions[row];
	for (index = threadIdx.x; index < half; index += THREADS)
		LmRopeRotate<PAIRING>(rows_bf16,base,index,half,
			position * __powf(theta,-2.0f * (float)index / (float)rope_dimension));
}
