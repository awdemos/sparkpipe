#pragma once

// Top-k selection. One implementation, three callers.
//
// This tree selects the k largest of n in three places and wrote it three times:
// DSA picks 2048 positions from a context, the MoE router picks 8 experts from
// 256, and the sampling head picks candidates from a 154,880-entry vocabulary.
// The old code had DsaSelectRadixTopkKernel, DsaMergeHierarchicalTopkKernel,
// MoeRouterTopKFromLogitsKernel and RestrictedArgmaxKernel, which is four
// implementations of one algorithm at four scales.
//
// The scales differ by five orders of magnitude and the right algorithm does not
// change, but the right SHAPE does:
//
//     k=8 from 256        one block, a sorting network in registers
//     k=2048 from 128k    radix over the value bits, two passes
//     k=1 from 154,880    a max reduction, which is radix with one bucket
//
// Both are here because choosing between them is a function of k and n, and a
// caller should not have to know which it wants.
//
// RADIX ON FLOAT BITS. A float's IEEE bit pattern is monotonic in the value for
// non-negatives, and flipping the sign bit plus inverting negatives makes it
// monotonic everywhere. So a radix pass over the top bits partitions by
// magnitude with no comparisons, and one pass over 8 bits narrows 128k
// candidates to a bucket that a second pass finishes. That is why this beats a
// sort at large n: it never orders anything it does not have to.

#include "inference/kernels/norm.cuh"
#include <stdint.h>

// Monotonic unsigned key from a float. Ordering by this key orders by value,
// which is what lets a radix pass replace a comparison sort.
static __device__ __forceinline__ uint32_t LmTopkKey(float value)
{
	uint32_t bits = __float_as_uint(value);
	return(bits ^ ((bits >> 31u) ? 0xffffffffu : 0x80000000u));
}

static __device__ __forceinline__ float LmTopkValue(uint32_t key)
{
	uint32_t bits = key ^ ((key >> 31u) ? 0x80000000u : 0xffffffffu);
	return(__uint_as_float(bits));
}

// -- small k, small n: bitonic in registers ------------------------------------
//
// For the router - 8 of 256 - a full sort of 256 in shared beats any radix pass,
// because the radix's two passes each cost more than the whole sort at this size.
// The threshold is not tuned; it is where a block can hold n in shared at all.
#define LM_TOPK_SMALL_LIMIT 1024u

// A checkpoint with more expert groups than this needs the bound raised; it is
// a shared-memory array, not a limit of the algorithm.
#define LM_TOPK_MAX_GROUPS 16u

template<uint32_t THREADS, uint32_t K, bool RENORMALISE = false, uint32_t GROUPS = 1u, uint32_t TOP_GROUPS = 1u>
__global__ __launch_bounds__(THREADS, 1)
void LmTopkSmallKernel(const float *__restrict__ scores, uint32_t n, uint32_t *__restrict__ out_indices, float *__restrict__ out_values, const float *__restrict__ selection_bias)
{
	extern __shared__ uint32_t lm_topk_shared[];
	uint32_t *keys = lm_topk_shared;
	uint32_t *slots = lm_topk_shared + LM_TOPK_SMALL_LIMIT;
	uint64_t base = (uint64_t)blockIdx.x * n;
	uint32_t index,size,stride;
	for (index = threadIdx.x; index < LM_TOPK_SMALL_LIMIT; index += THREADS)
	{
		// THE BIAS SELECTS; IT DOES NOT WEIGH. Kimi K3's router adds a per-expert
		// correction bias to pick the top-k and then gathers the mixture weights
		// from the UNBIASED scores - the report is explicit that omitting b from
		// p_i,j is what lets it "regulate dispatch without altering the mixture
		// weights". This kernel took a scalar bias, folded it into the sorted key,
		// and emitted that key as the weight, so a frozen load-balancing bias
		// would have leaked into every mixture weight as a fixed per-expert
		// distortion. Every caller passed 0.0f, so nothing was wrong yet.
		keys[index] = index < n ? LmTopkKey(scores[base + index]
			+ (selection_bias != 0 ? selection_bias[index] : 0.0f)) : 0u;
		slots[index] = index;
	}
	__syncthreads();
	// GROUPED SELECTION, when a checkpoint asks for it.
	//
	// The reference scores each group by the SUM OF ITS TOP TWO experts - not
	// its best and not its mean - keeps the top TOP_GROUPS groups, and masks
	// everything else out of the top-k. The bias is already in keys[] because
	// the reference groups scores_for_choice rather than scores, so the
	// select-versus-weigh split holds one level up as well.
	//
	// GROUPS == 1 is the whole of Kimi K3, and the compiler removes this.
	if ( GROUPS > 1u && GROUPS > TOP_GROUPS )
	{
		__shared__ uint32_t group_key[LM_TOPK_MAX_GROUPS];
		uint32_t group,per_group = n / GROUPS,cut;
		for (group = threadIdx.x; group < GROUPS; group += THREADS)
		{
			uint32_t best = 0u,second = 0u,member;
			for (member = 0u; member < per_group; ++member)
			{
				uint32_t value = keys[(group * per_group) + member];
				if ( value > best ) { second = best; best = value; }
				else if ( value > second ) second = value;
			}
			// The keys are order-preserving monotone images of the scores, so
			// their sum is not the sum of the scores. Ranking groups by it
			// ranks by a different function, so recover the two values first.
			group_key[group] = LmTopkKey(LmTopkValue(best) + LmTopkValue(second));
		}
		__syncthreads();
		// TOP_GROUPS is small - one thread finds the cutoff rather than a second
		// bitonic sort over a handful of entries.
		if ( threadIdx.x == 0u )
		{
			uint32_t taken,highest,index2;
			cut = 0u;
			for (taken = 0u; taken < TOP_GROUPS; ++taken)
			{
				highest = 0u;
				for (index2 = 0u; index2 < GROUPS; ++index2)
					if ( group_key[index2] > highest && group_key[index2] != 0xffffffffu )
						highest = group_key[index2];
				for (index2 = 0u; index2 < GROUPS; ++index2)
					if ( group_key[index2] == highest )
					{
						group_key[index2] = 0xffffffffu;
						break;
					}
			}
			(void)cut;
		}
		__syncthreads();
		for (index = threadIdx.x; index < LM_TOPK_SMALL_LIMIT; index += THREADS)
			if ( index < n && group_key[index / per_group] != 0xffffffffu )
				keys[index] = 0u;
		__syncthreads();
	}
	// Bitonic sort, descending. Every exchange is between a fixed pair, so there
	// is no divergence beyond the direction test.
	for (size = 2u; size <= LM_TOPK_SMALL_LIMIT; size <<= 1u)
		for (stride = size >> 1u; stride > 0u; stride >>= 1u)
		{
			for (index = threadIdx.x; index < LM_TOPK_SMALL_LIMIT; index += THREADS)
			{
				uint32_t partner = index ^ stride;
				if ( partner > index )
				{
					bool descending = ((index & size) == 0u);
					bool swap = descending ? (keys[index] < keys[partner])
						: (keys[index] > keys[partner]);
					if ( swap )
					{
						uint32_t tk = keys[index],ts = slots[index];
						keys[index] = keys[partner];
						slots[index] = slots[partner];
						keys[partner] = tk;
						slots[partner] = ts;
					}
				}
			}
			__syncthreads();
		}
	// The weight is re-read from the unbiased scores at the chosen slot rather
	// than recovered from the sort key, which carries the bias.
	for (index = threadIdx.x; index < K; index += THREADS)
	{
		out_indices[((uint64_t)blockIdx.x * K) + index] = slots[index];
		if ( out_values != 0 )
			out_values[((uint64_t)blockIdx.x * K) + index] = scores[base + slots[index]];
	}
	if ( RENORMALISE && out_values != 0 )
	{
		// moe_renormalize: divide the k gates by their sum so they sum to one.
		// K3 sets it; the report's routed_scaling_factor then multiplies a
		// normalised mixture, which is why that factor being 1.0 makes the
		// multiply a no-op rather than merely a small number.
		__syncthreads();
		if ( threadIdx.x == 0u )
		{
			float total = 0.0f;
			for (index = 0u; index < K; ++index)
				total += out_values[((uint64_t)blockIdx.x * K) + index];
			total += 1e-20f;
			for (index = 0u; index < K; ++index)
				out_values[((uint64_t)blockIdx.x * K) + index] /= total;
		}
	}
}

// -- large n: radix histogram --------------------------------------------------
//
// Pass one counts the top BITS of every key into 2^BITS buckets and finds the
// bucket where the k-th largest falls. Pass two emits every candidate above that
// bucket plus enough from within it. Nothing is sorted.
//
// Two kernels rather than one because the threshold has to be known before the
// second pass can run, and a grid-wide barrier inside one kernel would need
// cooperative launch for no gain.
#define LM_TOPK_RADIX_BITS 8u
#define LM_TOPK_BUCKETS (1u << LM_TOPK_RADIX_BITS)

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmTopkHistogramKernel(const float *__restrict__ scores, uint32_t n, uint32_t k, uint32_t *__restrict__ threshold_out)
{
	__shared__ uint32_t histogram[LM_TOPK_BUCKETS];
	uint64_t base = (uint64_t)blockIdx.x * n;
	uint32_t index,running;
	for (index = threadIdx.x; index < LM_TOPK_BUCKETS; index += THREADS)
		histogram[index] = 0u;
	__syncthreads();
	for (index = threadIdx.x; index < n; index += THREADS)
		atomicAdd(&histogram[LmTopkKey(scores[base + index]) >> (32u - LM_TOPK_RADIX_BITS)],1u);
	__syncthreads();
	// One thread walks the buckets from the top; 256 iterations is cheaper than
	// a parallel scan plus the barrier it would need.
	if ( threadIdx.x == 0u )
	{
		running = 0u;
		for (index = LM_TOPK_BUCKETS; index > 0u; --index)
		{
			running += histogram[index - 1u];
			if ( running >= k )
			{
				threshold_out[blockIdx.x] = index - 1u;
				return;
			}
		}
		threshold_out[blockIdx.x] = 0u;
	}
}

// Emit indices whose bucket is at or above the threshold, capped at k. The cap
// is why the count is atomic: the threshold bucket may hold more candidates than
// remain, and which of them are taken does not matter because they are
// indistinguishable at this radix.
template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmTopkGatherKernel(const float *__restrict__ scores, uint32_t n, uint32_t k, const uint32_t *__restrict__ threshold, uint32_t *__restrict__ out_indices, uint32_t *__restrict__ out_count)
{
	__shared__ uint32_t emitted;
	uint64_t base = (uint64_t)blockIdx.x * n;
	uint32_t bucket = threshold[blockIdx.x],index;
	if ( threadIdx.x == 0u )
		emitted = 0u;
	__syncthreads();
	for (index = threadIdx.x; index < n; index += THREADS)
	{
		if ( (LmTopkKey(scores[base + index]) >> (32u - LM_TOPK_RADIX_BITS)) >= bucket )
		{
			uint32_t slot = atomicAdd(&emitted,1u);
			if ( slot < k )
				out_indices[((uint64_t)blockIdx.x * k) + slot] = index;
		}
	}
	__syncthreads();
	if ( threadIdx.x == 0u && out_count != 0 )
		out_count[blockIdx.x] = emitted < k ? emitted : k;
}
