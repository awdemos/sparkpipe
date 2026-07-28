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

template<uint32_t THREADS, uint32_t K, bool RENORMALISE = false>
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
