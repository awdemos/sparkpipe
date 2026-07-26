#pragma once

// Decode attention. RoPE, latent-absorbed attention, and sparse selection.
//
// None of this is model-specific, and the version it replaces was named as
// though it were: SparkGlm52ResidentDecodeStageAbsorbedAttentionKernel, 299
// lines carrying 52 SPARK_GLM52_* references. Every one of the 52 is a
// dimension, a head count or a cache stride - none changes what is computed. The
// model belongs in the arguments.
//
// WHY LATENT-ABSORBED. Storing per-head keys and values costs
// heads * (qk_dim + v_dim) * 2 bytes per position. Storing one shared latent row
// costs (latent + rope) * 2. For a 64-head model that is 57 KB against 1152
// bytes - a 50x reduction on the only tensor that grows with context, bought by
// folding the up-projections into the query and output weights so the attention
// happens in latent space and per-head K and V are never materialised.
//
// The compute that buys it is free here. Decode attention is bound by cache
// bytes, not arithmetic: the calibration found three structurally different
// attention kernels producing identical time, which is the signature of a path
// limited only by the bytes all three share.
//
// THE CACHE FORMAT IS A TRAIT, INDEPENDENTLY OF THE WEIGHTS. A slot is read once
// per (sequence, position) and shared with nothing, where a weight tile is read
// once and shared across every row in its tile - different points on the
// precision-versus-bytes curve, so BF16 weights with an FP8 cache is a normal
// combination and either can change without the other.

#include "kernels/kv.cuh"
#include "kernels/norm.cuh"
#include <stdint.h>

// -- RoPE ---------------------------------------------------------------------
//
// Rotary position embedding over the trailing rope_dim elements of a row.
// theta and the dimension are arguments because they are the only things that
// differ between models - four families in the old tree had their own copy of
// this and the difference between them was two constants.
//
// The half-split pairing, not the interleaved one: element i pairs with
// i + rope_dim/2. Getting that wrong produces output that is fluent and subtly
// positionally wrong, which is the hardest kind of bug to attribute.
static __device__ __forceinline__ void LmRopePair(float *low, float *high, float angle)
{
	float c = __cosf(angle),s = __sinf(angle);
	float a = *low,b = *high;
	*low = (a * c) - (b * s);
	*high = (a * s) + (b * c);
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmRopeKernel(uint16_t *__restrict__ rows_bf16, const uint32_t *__restrict__ positions, uint32_t row_stride, uint32_t rope_offset, uint32_t rope_dim, float theta)
{
	uint64_t base = ((uint64_t)blockIdx.x * row_stride) + rope_offset;
	uint32_t half = rope_dim / 2u,index;
	float position = (float)positions[blockIdx.x];
	for (index = threadIdx.x; index < half; index += THREADS)
	{
		float low = LmBf16ToFloat(rows_bf16[base + index]);
		float high = LmBf16ToFloat(rows_bf16[base + half + index]);
		LmRopePair(&low,&high,position * __powf(theta,-2.0f * (float)index / (float)rope_dim));
		rows_bf16[base + index] = LmFloatToBf16(low);
		rows_bf16[base + half + index] = LmFloatToBf16(high);
	}
}

// -- latent-absorbed decode attention ------------------------------------------
//
// One block per (sequence, head group). Online softmax in a single pass over the
// cache: running maximum and denominator are rescaled as a larger score appears,
// so the cache is read once. Reading it twice - once to find the maximum, once
// to accumulate - would double the only traffic that matters.
//
// SPARSE SELECTION IS AN INDEX ARRAY, NOT A SEPARATE KERNEL. Passing a list of
// selected positions makes this the sparse path; passing null makes it dense.
// The old tree had these as different kernels, and the difference was which
// positions the loop visited.
template<class Geometry, uint32_t THREADS, uint32_t LATENT, uint32_t ROPE>
__global__ __launch_bounds__(THREADS, 1)
void LmAttentionDecodeKernel(const uint16_t *__restrict__ query_latent_bf16, const uint16_t *__restrict__ query_rope_bf16, LmKvView cache, const uint32_t *__restrict__ sequence_of_row, const uint32_t *__restrict__ context_length, const uint32_t *__restrict__ selected_positions, uint32_t selected_count, uint32_t heads, float qk_scale, uint16_t *__restrict__ output_bf16)
{
	__shared__ float reduction[THREADS / LM_WARP_LANES];
	__shared__ float shared_query[LATENT + ROPE];
	float accumulator[8];
	uint32_t row = blockIdx.x,head = blockIdx.y,index,step,positions;
	uint32_t sequence = sequence_of_row[row];
	uint64_t query_base = ((uint64_t)row * heads + head) * (LATENT + ROPE);
	float running_max = -INFINITY,running_sum = 0.0f;
	for (index = 0u; index < 8u; ++index)
		accumulator[index] = 0.0f;
	for (index = threadIdx.x; index < LATENT + ROPE; index += THREADS)
		shared_query[index] = LmBf16ToFloat(query_latent_bf16[query_base + index]);
	__syncthreads();
	positions = selected_positions != 0 ? selected_count : context_length[sequence];
	for (step = 0u; step < positions; ++step)
	{
		uint32_t position = selected_positions != 0
			? selected_positions[(row * selected_count) + step] : step;
		const uint8_t *slot = LmKvSlot<Geometry>(cache,sequence,position);
		float score = 0.0f,scaled,previous;
		// An unmapped page is not page zero. Reading it as page zero returns
		// another sequence's keys and produces output that is fluent and wrong.
		if ( slot == 0 )
			continue;
		for (index = threadIdx.x; index < LATENT + ROPE; index += THREADS)
			score += shared_query[index] * LmBf16ToFloat(((const uint16_t *)slot)[index]);
		score = LmBlockSum<THREADS>(score,reduction) * qk_scale;
		// Online softmax: rescale what is already accumulated rather than
		// revisit the cache once the maximum is known.
		previous = running_max;
		running_max = fmaxf(running_max,score);
		scaled = __expf(previous - running_max);
		running_sum = (running_sum * scaled) + __expf(score - running_max);
		for (index = 0u; index < 8u; ++index)
		{
			uint32_t element = (index * THREADS) + threadIdx.x;
			if ( element < LATENT )
				accumulator[index] = (accumulator[index] * scaled)
					+ (__expf(score - running_max)
						* LmBf16ToFloat(((const uint16_t *)slot)[element]));
		}
	}
	for (index = 0u; index < 8u; ++index)
	{
		uint32_t element = (index * THREADS) + threadIdx.x;
		if ( element < LATENT )
			output_bf16[(((uint64_t)row * heads) + head) * LATENT + element] =
				LmFloatToBf16(accumulator[index] / fmaxf(running_sum,1.0e-20f));
	}
}

// -- sparse selection ----------------------------------------------------------
//
// Score every cached position with a low-rank index head and keep the top K.
// The scoring is a dot product against a narrower query, so it costs
// index_dim/latent of a full attention pass - about a fifth here - and the full
// pass then reads a fixed number of positions instead of the whole context.
//
// That is the entire point: it turns attention from linear in context length
// into constant, which for a growing cache is the difference between a model
// that degrades with conversation length and one that does not.
//
// The selection is shared across a group of layers, so the score is computed
// once per group rather than once per layer.
template<class Geometry, uint32_t THREADS, uint32_t INDEX_DIM>
__global__ __launch_bounds__(THREADS, 1)
void LmSparseScoreKernel(const uint16_t *__restrict__ index_query_bf16, LmKvView cache, const uint32_t *__restrict__ sequence_of_row, const uint32_t *__restrict__ context_length, uint32_t index_heads, float *__restrict__ scores)
{
	__shared__ float reduction[THREADS / LM_WARP_LANES];
	uint32_t row = blockIdx.x,position = blockIdx.y;
	uint32_t sequence = sequence_of_row[row],head,index;
	const uint8_t *slot;
	float total = 0.0f;
	if ( position >= context_length[sequence] )
		return;
	slot = LmKvSlot<Geometry>(cache,sequence,position);
	if ( slot == 0 )
	{
		if ( threadIdx.x == 0u )
			scores[((uint64_t)row * gridDim.y) + position] = -INFINITY;
		return;
	}
	for (head = 0u; head < index_heads; ++head)
	{
		float partial = 0.0f;
		for (index = threadIdx.x; index < INDEX_DIM; index += THREADS)
			partial += LmBf16ToFloat(index_query_bf16[(((uint64_t)row * index_heads) + head) * INDEX_DIM + index])
				* LmBf16ToFloat(((const uint16_t *)slot)[index]);
		total += LmBlockSum<THREADS>(partial,reduction);
	}
	if ( threadIdx.x == 0u )
		scores[((uint64_t)row * gridDim.y) + position] = total;
}
