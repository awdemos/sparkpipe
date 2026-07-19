#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <stdint.h>

/*
 * Shared device kernels for the sparkpipe model-driver family.
 *
 * Extracted verbatim from the audited K3 resident decode stage (2026-07-18
 * production audit) and renamed SparkLm*: every driver module includes this
 * header inside its own translation unit, so each module gets its own
 * internal-linkage instantiation. One source, zero ABI coupling between
 * modules, zero runtime cost. The MXFP4 group size is a template parameter,
 * never a defaulted macro, so a module that disagrees with the pack format
 * fails to build instead of silently decoding garbage.
 *
 * Consumers today: qwen36, dsv4, mimo25 (from first line). The K3 module
 * retrofits onto this header at its PP v2 pass; until then the K3 copies of
 * these functions remain the audited originals this file was taken from.
 */

#define SPARK_LM_WARP_LANES 32u
#define SPARK_LM_CTA_THREADS 256u
#define SPARK_LM_CTA_WARPS (SPARK_LM_CTA_THREADS / SPARK_LM_WARP_LANES)

#define SPARK_LM_WEIGHT_FORMAT_BF16 0u
#define SPARK_LM_WEIGHT_FORMAT_F32 1u
#define SPARK_LM_WEIGHT_FORMAT_U32 2u
#define SPARK_LM_WEIGHT_FORMAT_MXFP4_E2M1 3u

static __device__ __forceinline__ float SparkLmBf16ToFloat(const void *source, uint64_t index)
{
	return(__bfloat162float(((const __nv_bfloat16 *)source)[index]));
}

static __device__ __forceinline__ void SparkLmFloatToBf16(void *destination, uint64_t index, float value)
{
	((__nv_bfloat16 *)destination)[index] = __float2bfloat16(value);
}

// E2M1 nibble to float: sign, 2 exponent bits (bias 1), 1 mantissa bit.
static __device__ __forceinline__ float SparkLmDecodeE2m1(uint32_t nibble)
{
	const float magnitude[8] = {0.0f,0.5f,1.0f,1.5f,2.0f,3.0f,4.0f,6.0f};
	float value = magnitude[nibble & 7u];
	return((nibble & 8u) != 0u ? -value : value);
}

// E8M0 scale byte: pure power of two, bias 127, 0xff reserved as NaN -> zero.
static __device__ __forceinline__ float SparkLmDecodeE8m0(uint32_t byte_value)
{
	if ( byte_value == 0xffu )
		return(0.0f);
	return(exp2f((float)(int32_t)byte_value - 127.0f));
}

static __device__ __forceinline__ float SparkLmSigmoid(float value)
{
	return(1.0f / (1.0f + __expf(-value)));
}

static __device__ __forceinline__ float SparkLmSoftplus(float value)
{
	if ( value > 20.0f )
		return(value);
	return(log1pf(__expf(value)));
}

static __device__ __forceinline__ float SparkLmSwish(float value)
{
	return(value * SparkLmSigmoid(value));
}

static __device__ __forceinline__ float SparkLmWarpReduceSum(float value)
{
	uint32_t offset;
	for (offset = SPARK_LM_WARP_LANES / 2u; offset != 0u; offset >>= 1u)
		value += __shfl_down_sync(0xffffffffu,value,offset);
	return(value);
}

// Block sum over blockDim.x threads; scratch must hold SPARK_LM_CTA_WARPS floats.
static __device__ float SparkLmBlockReduceSum(float value, float *scratch)
{
	uint32_t lane = threadIdx.x % SPARK_LM_WARP_LANES,warp = threadIdx.x / SPARK_LM_WARP_LANES;
	uint32_t warp_count = (blockDim.x + SPARK_LM_WARP_LANES - 1u) / SPARK_LM_WARP_LANES;
	value = SparkLmWarpReduceSum(value);
	if ( lane == 0u )
		scratch[warp] = value;
	__syncthreads();
	value = (threadIdx.x < warp_count) ? scratch[threadIdx.x] : 0.0f;
	if ( warp == 0u )
		value = SparkLmWarpReduceSum(value);
	if ( threadIdx.x == 0u )
		scratch[0] = value;
	__syncthreads();
	value = scratch[0];
	__syncthreads();
	return(value);
}

// Thread-0 softmax over a small shared scalar table; every thread leaves with
// the normalized weights visible.
static __device__ void SparkLmSharedSoftmax(const float *logits, float *weights, uint32_t count)
{
	uint32_t candidate;
	float maximum,total;
	if ( threadIdx.x == 0u )
	{
		maximum = logits[0];
		for (candidate = 1; candidate < count; candidate++)
			if ( logits[candidate] > maximum )
				maximum = logits[candidate];
		total = 0.0f;
		for (candidate = 0; candidate < count; candidate++)
		{
			weights[candidate] = __expf(logits[candidate] - maximum);
			total += weights[candidate];
		}
		for (candidate = 0; candidate < count; candidate++)
			weights[candidate] /= total;
	}
	__syncthreads();
}

static __global__ void SparkLmRmsNormKernel(const void *input_bf16, const void *gain_bf16, void *output_bf16, uint32_t row_count, uint32_t dimension, float epsilon)
{
	__shared__ float reduce_scratch[SPARK_LM_CTA_WARPS];
	uint32_t row = blockIdx.x;
	uint64_t row_offset;
	uint32_t element;
	float value,sum_squares,inverse_rms;
	if ( row >= row_count )
		return;
	row_offset = (uint64_t)row * (uint64_t)dimension;
	sum_squares = 0.0f;
	for (element = threadIdx.x; element < dimension; element += blockDim.x)
	{
		value = SparkLmBf16ToFloat(input_bf16,row_offset + element);
		sum_squares += (value * value);
	}
	sum_squares = SparkLmBlockReduceSum(sum_squares,reduce_scratch);
	inverse_rms = rsqrtf((sum_squares / (float)dimension) + epsilon);
	for (element = threadIdx.x; element < dimension; element += blockDim.x)
	{
		value = SparkLmBf16ToFloat(input_bf16,row_offset + element);
		SparkLmFloatToBf16(output_bf16,row_offset + element,(value * inverse_rms) * SparkLmBf16ToFloat(gain_bf16,element));
	}
}

/*
 * Row-major linear, one warp per output neuron, eight neurons in flight per
 * block, activations staged once in shared memory. The weight branch is
 * launch-uniform (bf16 or MXFP4 nibble pairs with one E8M0 scale per group),
 * so both formats share the loop. Weight fetch is the bound at decode batch
 * sizes, which this layout keeps fully coalesced. GROUP_SIZE is the MXFP4
 * scale group and must match the stage pack; it is a template parameter so a
 * mismatch is a compile error at the launch site, never a silent default.
 */
template <uint32_t GROUP_SIZE>
static __global__ void SparkLmLinearKernel(uint32_t weight_format, const void *weight_payload, const uint8_t *weight_scale_e8m0, const void *input_bf16, void *output_bf16, uint32_t row_count, uint32_t input_dimension, uint32_t output_dimension)
{
	extern __shared__ float shared_input[];
	uint32_t row = blockIdx.x,neuron_base = blockIdx.y * SPARK_LM_CTA_WARPS;
	uint32_t warp = threadIdx.x / SPARK_LM_WARP_LANES,lane = threadIdx.x % SPARK_LM_WARP_LANES;
	uint32_t neuron,element,packed_index,pair_base;
	uint64_t weight_row_offset;
	float accumulator,scale_value;
	uint32_t packed;
	if ( row >= row_count )
		return;
	for (element = threadIdx.x; element < input_dimension; element += blockDim.x)
		shared_input[element] = SparkLmBf16ToFloat(input_bf16,((uint64_t)row * input_dimension) + element);
	__syncthreads();
	neuron = neuron_base + warp;
	if ( neuron >= output_dimension )
		return;
	accumulator = 0.0f;
	if ( weight_format == SPARK_LM_WEIGHT_FORMAT_BF16 )
	{
		weight_row_offset = (uint64_t)neuron * input_dimension;
		for (element = lane; element < input_dimension; element += SPARK_LM_WARP_LANES)
			accumulator += (shared_input[element] * SparkLmBf16ToFloat(weight_payload,weight_row_offset + element));
	}
	else
	{
		weight_row_offset = (uint64_t)neuron * (input_dimension / 2u);
		for (packed_index = lane; packed_index < (input_dimension / 2u); packed_index += SPARK_LM_WARP_LANES)
		{
			pair_base = packed_index << 1u;
			packed = ((const uint8_t *)weight_payload)[weight_row_offset + packed_index];
			scale_value = SparkLmDecodeE8m0(weight_scale_e8m0[((uint64_t)neuron * (input_dimension / GROUP_SIZE)) + (pair_base / GROUP_SIZE)]);
			accumulator += (shared_input[pair_base] * (SparkLmDecodeE2m1(packed & 0x0fu) * scale_value));
			accumulator += (shared_input[pair_base + 1u] * (SparkLmDecodeE2m1(packed >> 4u) * scale_value));
		}
	}
	accumulator = SparkLmWarpReduceSum(accumulator);
	if ( lane == 0u )
		SparkLmFloatToBf16(output_bf16,((uint64_t)row * output_dimension) + neuron,accumulator);
}

/*
 * Fused LM head: matvec against the row's hidden and a running argmax, no
 * logits tensor ever materialized. One block per row; each warp owns a
 * stripe of candidates, keeps its running best in registers, and the block
 * reduces bests through shared memory. token_ids may be null, in which case
 * the winning candidate INDEX is written (dense-vocab head); non-null maps
 * through a restricted-vocabulary id table.
 */
static __global__ void SparkLmHeadArgmaxKernel(const void *hidden_bf16, const void *head_weight_bf16, const uint32_t *token_ids, uint32_t *output_token_ids, uint32_t row_count, uint32_t hidden_dimension, uint32_t candidate_count)
{
	__shared__ float best_score[SPARK_LM_CTA_WARPS];
	__shared__ uint32_t best_candidate[SPARK_LM_CTA_WARPS];
	uint32_t row = blockIdx.x;
	uint32_t warp = threadIdx.x / SPARK_LM_WARP_LANES,lane = threadIdx.x % SPARK_LM_WARP_LANES;
	uint64_t row_offset;
	uint32_t candidate,element,winner;
	float accumulator,warp_best_score;
	uint32_t warp_best_candidate;
	if ( row >= row_count )
		return;
	row_offset = (uint64_t)row * (uint64_t)hidden_dimension;
	warp_best_score = -3.0e38f;
	warp_best_candidate = 0u;
	for (candidate = warp; candidate < candidate_count; candidate += SPARK_LM_CTA_WARPS)
	{
		accumulator = 0.0f;
		for (element = lane; element < hidden_dimension; element += SPARK_LM_WARP_LANES)
			accumulator += (SparkLmBf16ToFloat(hidden_bf16,row_offset + element) * SparkLmBf16ToFloat(head_weight_bf16,((uint64_t)candidate * hidden_dimension) + element));
		accumulator = SparkLmWarpReduceSum(accumulator);
		accumulator = __shfl_sync(0xffffffffu,accumulator,0);
		if ( accumulator > warp_best_score || (accumulator == warp_best_score && candidate < warp_best_candidate) )
		{
			warp_best_score = accumulator;
			warp_best_candidate = candidate;
		}
	}
	if ( lane == 0u )
	{
		best_score[warp] = warp_best_score;
		best_candidate[warp] = warp_best_candidate;
	}
	__syncthreads();
	if ( threadIdx.x == 0u )
	{
		winner = 0u;
		for (candidate = 1; candidate < SPARK_LM_CTA_WARPS; candidate++)
			if ( best_score[candidate] > best_score[winner] || (best_score[candidate] == best_score[winner] && best_candidate[candidate] < best_candidate[winner]) )
				winner = candidate;
		output_token_ids[row] = token_ids != 0 ? token_ids[best_candidate[winner]] : best_candidate[winner];
	}
}
