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

#include "inference/kernels/kv.cuh"
#include "inference/kernels/norm.cuh"
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

// WHICH TWO ELEMENTS FORM A PAIR. Both conventions are called "rope" and they
// are different rotations, so a checkpoint trained under one and served under
// the other is fluent and positionally wrong - the failure the comment above
// describes, now selectable instead of assumed.
//
// Half-split pairs i with i + rope_dim/2. GLM 5.2, Qwen 3.6 and MiMo 2.5 want
// this and it stays the default, so no existing call site changes.
//
// Interleaved pairs 2i with 2i+1, the view_as_complex layout. DeepSeek V4
// encodes its released checkpoint this way, for both the main attention and the
// compressor sub-module.
enum LmRopePairing
{
	LM_ROPE_HALF_SPLIT = 0,
	LM_ROPE_INTERLEAVED = 1
};

// The three rope kernels below had this body character for character. It is one
// function now, because the pairing is the thing that varies and a convention
// duplicated three times is a convention that will only be fixed twice.
template<LmRopePairing PAIRING>
static __device__ __forceinline__ void LmRopeRotate(uint16_t *rows_bf16, uint64_t base, uint32_t index, uint32_t half, float angle)
{
	uint32_t low_offset,high_offset;
	float low,high;
	low_offset = (PAIRING == LM_ROPE_INTERLEAVED) ? (index * 2u) : index;
	high_offset = (PAIRING == LM_ROPE_INTERLEAVED) ? ((index * 2u) + 1u) : (half + index);
	low = LmBf16ToFloat(rows_bf16[base + low_offset]);
	high = LmBf16ToFloat(rows_bf16[base + high_offset]);
	LmRopePair(&low,&high,angle);
	rows_bf16[base + low_offset] = LmFloatToBf16(low);
	rows_bf16[base + high_offset] = LmFloatToBf16(high);
}

template<uint32_t THREADS, LmRopePairing PAIRING = LM_ROPE_HALF_SPLIT>
__global__ __launch_bounds__(THREADS, 1)
void LmRopeKernel(uint16_t *__restrict__ rows_bf16, const uint32_t *__restrict__ positions, uint32_t row_stride, uint32_t rope_offset, uint32_t rope_dim, float theta)
{
	uint64_t base = ((uint64_t)blockIdx.x * row_stride) + rope_offset;
	uint32_t half = rope_dim / 2u,index;
	float position = (float)positions[blockIdx.x];
	for (index = threadIdx.x; index < half; index += THREADS)
		LmRopeRotate<PAIRING>(rows_bf16,base,index,half,
			position * __powf(theta,-2.0f * (float)index / (float)rope_dim));
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
// PREFILL IS THIS KERNEL WITH MORE ROWS AND A MASK.
//
// The old tree had DsaPrefillRowSetupKernel, DsaSparsePrefillAttentionKernel,
// DsaPrefillIndexerPass and PagedChunkPrefill as separate implementations. They
// compute the same thing decode does: a row attends over cached positions. What
// differs is that a prefill row is not the last position, so it must not see
// positions after its own, and that there are many rows rather than one.
//
// Both are one argument. row_position, when non-null, gives each row its own
// position and the loop stops there; when null every row attends to everything
// cached, which is decode. That is the whole difference, and it is a comparison
// rather than a kernel.
//
// Chunking is the caller's business and stays there. A prefill of 8,000 tokens
// runs as chunks of a few hundred rows because the intermediate buffers are
// sized for that, and which chunk size is a scheduling decision that depends on
// what else is resident - not something a kernel should decide.
template<class Geometry, uint32_t THREADS, uint32_t LATENT, uint32_t ROPE>
__global__ __launch_bounds__(THREADS, 1)
void LmAttentionDecodeKernel(const uint16_t *__restrict__ query_latent_bf16, const uint16_t *__restrict__ query_rope_bf16, LmKvView cache, const uint32_t *__restrict__ sequence_of_row, const uint32_t *__restrict__ context_length, const uint32_t *__restrict__ selected_positions, uint32_t selected_count, uint32_t heads, float qk_scale, uint16_t *__restrict__ output_bf16, const uint32_t *__restrict__ row_position)
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
		// Causal: a prefill row must not see past itself. Skipping rather than
		// masking the score keeps the online softmax's running maximum honest -
		// a masked-to-negative-infinity score still participates in the rescale
		// and costs the cache read that made it worthless.
		if ( row_position != 0 && position > row_position[row] )
			continue;
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

// -- YaRN rope ------------------------------------------------------------------
//
// Rope for a model whose context was extended past its training length.
//
// Plain rope extrapolates: a position beyond anything seen in training gets a
// frequency the model has no calibration for, and quality degrades with
// distance. YaRN interpolates instead, and does it PER FREQUENCY BAND - high
// frequencies encode local order and are left alone, low frequencies encode
// long-range position and are scaled by the extension factor, with a ramp
// between.
//
// The band edges come from how many full rotations a wavelength completes within
// the original context. A dimension whose wavelength is shorter than the
// original length has seen every phase and needs no correction; one longer than
// it has not, and is interpolated.
//
// Getting this wrong is quiet: applying plain rope to a YaRN model produces text
// that is fine for a few thousand tokens and drifts after, which reads as the
// model being bad at long context rather than as a missing transform.
static __device__ __forceinline__ float LmYarnFrequency(uint32_t index, uint32_t rope_dimension, float theta, float scale_factor, float original_positions, float low_band, float high_band)
{
	float exponent = 2.0f * (float)index / (float)rope_dimension;
	float inverse = __powf(theta,-exponent);
	// Wavelength in positions, and where it falls between the two band edges.
	float wavelength = 6.2831853f / inverse;
	float rotations = original_positions / wavelength;
	float ramp = (rotations - low_band) / fmaxf(high_band - low_band,1e-3f);
	float blend = fminf(fmaxf(ramp,0.0f),1.0f);
	// blend 1 is a short wavelength, left at the extrapolated frequency;
	// blend 0 is a long one, fully interpolated by the scale factor.
	return((inverse * blend) + ((inverse / scale_factor) * (1.0f - blend)));
}

template<uint32_t THREADS, LmRopePairing PAIRING = LM_ROPE_HALF_SPLIT>
__global__ __launch_bounds__(THREADS, 1)
void LmRopeYarnKernel(uint16_t *__restrict__ rows_bf16, const uint32_t *__restrict__ positions, uint32_t row_stride, uint32_t rope_offset, uint32_t rope_dim, float theta, float scale_factor, float original_positions, float low_band, float high_band)
{
	uint64_t base = ((uint64_t)blockIdx.x * row_stride) + rope_offset;
	uint32_t half = rope_dim / 2u,index;
	float position = (float)positions[blockIdx.x];
	for (index = threadIdx.x; index < half; index += THREADS)
		LmRopeRotate<PAIRING>(rows_bf16,base,index,half,position *
			LmYarnFrequency(index,rope_dim,theta,scale_factor,
				original_positions,low_band,high_band));
}

// -- hierarchical sparse selection ------------------------------------------------
//
// Harvested from the old decode stage's DsaKeyIndexBlockSummaryBuildKernel and
// DsaScoreSelectHierarchicalKernel, which my first version did not have and
// badly needed.
//
// LmSparseScoreKernel above scores every cached position, so the selection pass
// is linear in context - which defeats the point of sparse attention, whose
// whole job is to make attention constant. The hierarchical form scores one
// SUMMARY per block first, keeps the promising blocks, and scores positions only
// inside those:
//
//     context    flat dots   hierarchical   speedup
//        8,192       8,192          2,176      3.8x
//      131,072     131,072          4,096     32.0x
//    1,048,576   1,048,576         18,432     56.9x
//
// At a million tokens the flat version spends more on deciding what to attend to
// than on attending.
//
// THE SUMMARY IS A MAX, NOT A MEAN. A block is worth visiting if it contains ANY
// position that scores well, and a mean over 64 positions buries one high score
// under sixty-three low ones. Max over the element-wise absolute value bounds
// the best possible dot product in the block, so a block the summary rejects
// cannot contain a winner - which is what makes the pruning safe rather than
// heuristic.
//
// Summaries are maintained incrementally: only the block a new token lands in is
// dirty, so a decode step rebuilds one summary rather than all of them.
template<class Geometry, uint32_t THREADS, uint32_t INDEX_DIM>
__global__ __launch_bounds__(THREADS, 1)
void LmSparseSummaryBuildKernel(LmKvView cache, const uint32_t *__restrict__ sequence_of_row, const uint32_t *__restrict__ dirty_block, uint32_t block_positions, uint16_t *__restrict__ summary_bf16, uint32_t blocks_per_sequence)
{
	__shared__ float reduction[THREADS / LM_WARP_LANES];
	uint32_t row = blockIdx.x,block = dirty_block != 0 ? dirty_block[row] : blockIdx.y;
	uint32_t sequence = sequence_of_row[row],position,index;
	for (index = threadIdx.x; index < INDEX_DIM; index += THREADS)
	{
		float best = 0.0f;
		for (position = 0u; position < block_positions; ++position)
		{
			const uint8_t *slot = LmKvSlot<Geometry>(cache,sequence,(block * block_positions) + position);
			if ( slot == 0 )
				continue;
			best = fmaxf(best,fabsf(LmBf16ToFloat(((const uint16_t *)slot)[index])));
		}
		summary_bf16[(((uint64_t)sequence * blocks_per_sequence) + block) * INDEX_DIM + index] =
			LmFloatToBf16(best);
	}
	(void)reduction;
}

// Score block summaries. One block per (row, summary block), so the grid is the
// context divided by the block size rather than the context.
//
// The score is an upper bound on any position's score within the block, because
// the summary is an element-wise max of absolute values and the query is dotted
// against it with absolute values too. A block whose bound is below the current
// threshold cannot contain a better position than one already selected.
template<uint32_t THREADS, uint32_t INDEX_DIM>
__global__ __launch_bounds__(THREADS, 1)
void LmSparseSummaryScoreKernel(const uint16_t *__restrict__ index_query_bf16, const uint16_t *__restrict__ summary_bf16, const uint32_t *__restrict__ sequence_of_row, const uint32_t *__restrict__ block_count, uint32_t index_heads, uint32_t blocks_per_sequence, float *__restrict__ block_score)
{
	__shared__ float reduction[THREADS / LM_WARP_LANES];
	uint32_t row = blockIdx.x,block = blockIdx.y;
	uint32_t sequence = sequence_of_row[row],head,index;
	float total = 0.0f;
	if ( block >= block_count[sequence] )
	{
		if ( threadIdx.x == 0u )
			block_score[(row * blocks_per_sequence) + block] = -INFINITY;
		return;
	}
	for (head = 0u; head < index_heads; ++head)
	{
		float partial = 0.0f;
		for (index = threadIdx.x; index < INDEX_DIM; index += THREADS)
			partial += fabsf(LmBf16ToFloat(index_query_bf16[
				(((uint64_t)row * index_heads) + head) * INDEX_DIM + index]))
				* LmBf16ToFloat(summary_bf16[
				(((uint64_t)sequence * blocks_per_sequence) + block) * INDEX_DIM + index]);
		total += LmBlockSum<THREADS>(partial,reduction);
	}
	if ( threadIdx.x == 0u )
		block_score[(row * blocks_per_sequence) + block] = total;
}

// Score positions inside the selected blocks only.
//
// The grid is (row, selected block, position-in-block), so its size is the
// selection budget rather than the context. This is where the speedup lands: the
// expensive per-position dot runs on a fixed number of positions no matter how
// long the conversation is.
template<class Geometry, uint32_t THREADS, uint32_t INDEX_DIM>
__global__ __launch_bounds__(THREADS, 1)
void LmSparseRefineKernel(const uint16_t *__restrict__ index_query_bf16, LmKvView cache, const uint32_t *__restrict__ sequence_of_row, const uint32_t *__restrict__ selected_block, uint32_t block_positions, uint32_t index_heads, float *__restrict__ scores, uint32_t *__restrict__ positions_out)
{
	__shared__ float reduction[THREADS / LM_WARP_LANES];
	uint32_t row = blockIdx.x,slot_index = blockIdx.y;
	uint32_t block = selected_block[(row * gridDim.y) + (slot_index / block_positions)];
	uint32_t position = (block * block_positions) + (slot_index % block_positions);
	uint32_t sequence = sequence_of_row[row],head,index;
	const uint8_t *slot = LmKvSlot<Geometry>(cache,sequence,position);
	float total = 0.0f;
	if ( slot == 0 )
	{
		if ( threadIdx.x == 0u )
		{
			scores[(row * gridDim.y) + slot_index] = -INFINITY;
			positions_out[(row * gridDim.y) + slot_index] = position;
		}
		return;
	}
	for (head = 0u; head < index_heads; ++head)
	{
		float partial = 0.0f;
		for (index = threadIdx.x; index < INDEX_DIM; index += THREADS)
			partial += LmBf16ToFloat(index_query_bf16[
				(((uint64_t)row * index_heads) + head) * INDEX_DIM + index])
				* LmBf16ToFloat(((const uint16_t *)slot)[index]);
		total += LmBlockSum<THREADS>(partial,reduction);
	}
	if ( threadIdx.x == 0u )
	{
		scores[(row * gridDim.y) + slot_index] = total;
		positions_out[(row * gridDim.y) + slot_index] = position;
	}
}
