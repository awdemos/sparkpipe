#pragma once

// RMS norm, residual, gated activation. The pieces every model reimplements.
//
// These were written five times in this tree: once per family, plus the shared
// library nobody calls. Every copy is the same reduction with the model's
// dimension baked in as a constant - glm52's RmsNormKernel is 62 lines carrying
// seven SPARK_GLM52_MODEL_* references, none of which changes what it computes.
//
// The dimension is a runtime argument here rather than a template parameter, and
// that is deliberate: unlike a tile shape it sizes nothing at compile time, so
// making it static would multiply instantiations for no benefit. The block width
// is the template parameter, because that one does size the reduction buffer.
//
// EVERY OUTPUT PATH IS A FORMAT TRAIT. A norm feeding a BF16 GEMM writes BF16; a
// norm feeding a quantised GEMM writes packed codes and a block scale. Those are
// the same kernel with a different Format, which is why there is no separate
// "RmsNormFp8Quantize" here - the old tree had four such fusions and they
// differed only in the store.

#include "kernels/dtype.cuh"
#include "kernels/formats/bf16.cuh"
#include <stdint.h>

// Block reduction over one row. Warp shuffles first, then one round through
// shared - the shared traffic is warps, not threads.
template<uint32_t THREADS>
static __device__ float LmBlockSum(float value, float *shared)
{
	const uint32_t warps = THREADS / LM_WARP_LANES;
	uint32_t lane = threadIdx.x % LM_WARP_LANES,warp = threadIdx.x / LM_WARP_LANES;
	uint32_t offset;
	for (offset = LM_WARP_LANES / 2u; offset > 0u; offset >>= 1u)
		value += __shfl_down_sync(0xffffffffu,value,offset);
	if ( lane == 0u )
		shared[warp] = value;
	__syncthreads();
	value = threadIdx.x < warps ? shared[threadIdx.x] : 0.0f;
	if ( warp == 0u )
		for (offset = warps / 2u; offset > 0u; offset >>= 1u)
			value += __shfl_down_sync(0xffffffffu,value,offset);
	if ( threadIdx.x == 0u )
		shared[0] = value;
	__syncthreads();
	return(shared[0]);
}

template<uint32_t THREADS>
static __device__ float LmBlockMax(float value, float *shared)
{
	const uint32_t warps = THREADS / LM_WARP_LANES;
	uint32_t lane = threadIdx.x % LM_WARP_LANES,warp = threadIdx.x / LM_WARP_LANES;
	uint32_t offset;
	for (offset = LM_WARP_LANES / 2u; offset > 0u; offset >>= 1u)
		value = fmaxf(value,__shfl_down_sync(0xffffffffu,value,offset));
	if ( lane == 0u )
		shared[warp] = value;
	__syncthreads();
	value = threadIdx.x < warps ? shared[threadIdx.x] : 0.0f;
	if ( warp == 0u )
		for (offset = warps / 2u; offset > 0u; offset >>= 1u)
			value = fmaxf(value,__shfl_down_sync(0xffffffffu,value,offset));
	if ( threadIdx.x == 0u )
		shared[0] = value;
	__syncthreads();
	return(shared[0]);
}

// Residual add and RMS norm, fused.
//
// Fused because the residual has to be read anyway and the norm needs the same
// row: splitting them costs a full extra pass over the hidden state per layer.
// The residual output is written as well as consumed, because the next layer
// needs it.
//
// One pass, not two. The sum of squares and the normalised store both need the
// row, and staging it in shared costs hidden*4 bytes against a second global
// read of hidden*2 - which is why the row is staged rather than re-read.
template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmFusedResidualRmsNormKernel(const uint16_t *__restrict__ input_bf16, const uint16_t *__restrict__ residual_bf16, const uint16_t *__restrict__ weight_bf16, uint16_t *__restrict__ residual_out_bf16, uint16_t *__restrict__ output_bf16, uint32_t dimension, float epsilon)
{
	extern __shared__ float lm_norm_shared[];
	float *row = lm_norm_shared;
	float *reduction = lm_norm_shared + dimension;
	uint64_t base = (uint64_t)blockIdx.x * dimension;
	uint32_t index;
	float total = 0.0f,scale;
	for (index = threadIdx.x; index < dimension; index += THREADS)
	{
		float value = LmBf16ToFloat(input_bf16[base + index]);
		if ( residual_bf16 != 0 )
			value += LmBf16ToFloat(residual_bf16[base + index]);
		row[index] = value;
		total += value * value;
		if ( residual_out_bf16 != 0 )
			residual_out_bf16[base + index] = LmFloatToBf16(value);
	}
	total = LmBlockSum<THREADS>(total,reduction);
	scale = rsqrtf((total / (float)dimension) + epsilon);
	for (index = threadIdx.x; index < dimension; index += THREADS)
		output_bf16[base + index] =
			LmFloatToBf16(row[index] * scale * LmBf16ToFloat(weight_bf16[index]));
}

// SiLU(gate) * up.
//
// gate and up arrive interleaved per row, gate first, because that is how a
// fused w1 emits them. A model whose pack orders them the other way passes
// gate_first false rather than getting its own kernel - the ordering is a
// property of the checkpoint, not of the computation, and a mismatch silently
// swaps SiLU's argument.
template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmSiluMulKernel(const uint16_t *__restrict__ gate_up_bf16, uint16_t *__restrict__ output_bf16, uint32_t dimension, bool gate_first)
{
	uint64_t base = (uint64_t)blockIdx.x * dimension * 2u;
	uint64_t out_base = (uint64_t)blockIdx.x * dimension;
	uint32_t index;
	for (index = threadIdx.x; index < dimension; index += THREADS)
	{
		float gate = LmBf16ToFloat(gate_up_bf16[base + (gate_first ? index : dimension + index)]);
		float up = LmBf16ToFloat(gate_up_bf16[base + (gate_first ? dimension + index : index)]);
		output_bf16[out_base + index] =
			LmFloatToBf16((gate / (1.0f + __expf(-gate))) * up);
	}
}

// Quantise a row into a format's packed layout with per-group block scales.
//
// The absmax and the encode both need the row, so it is staged once rather than
// read twice - the old tree's FP8 quantiser read hidden_bf16 twice and that cost
// 900 MB per pass at B128.
//
// SOURCE ROWS ARE INDIRECT. A routed MoE needs each token's row written once per
// expert it was routed to; taking the row index through a map writes every packed
// row directly and removes the replication pass entirely. Passing a null map is
// the identity, which is the dense case.
template<class Format, uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmQuantiseRowsKernel(const uint16_t *__restrict__ input_bf16, const uint32_t *__restrict__ source_row_map, uint8_t *__restrict__ output_codes, uint8_t *__restrict__ output_scales, uint32_t row_count, uint32_t dimension)
{
	extern __shared__ float lm_quant_shared[];
	float *row = lm_quant_shared;
	float *reduction = lm_quant_shared + Format::kScaleGroup;
	const uint32_t groups = dimension / Format::kScaleGroup;
	uint32_t destination = blockIdx.x,group = blockIdx.y,index;
	uint64_t source;
	float absmax = 0.0f,scale,inverse;
	if ( destination >= row_count || group >= groups )
		return;
	source = (uint64_t)(source_row_map != 0 ? source_row_map[destination] : destination)
		* (uint64_t)dimension + (group * Format::kScaleGroup);
	for (index = threadIdx.x; index < Format::kScaleGroup; index += THREADS)
	{
		row[index] = LmBf16ToFloat(input_bf16[source + index]);
		absmax = fmaxf(absmax,fabsf(row[index]));
	}
	absmax = LmBlockMax<THREADS>(absmax,reduction);
	scale = fmaxf(absmax / Format::kMax,1.0e-8f);
	inverse = 1.0f / scale;
	if ( threadIdx.x == 0u )
		output_scales[((uint64_t)destination * groups) + group] = LmFloatToUe4m3(scale);
	__syncthreads();
	for (index = threadIdx.x * 2u; index < Format::kScaleGroup; index += THREADS * 2u)
	{
		uint64_t bit = (((uint64_t)destination * dimension) + (group * Format::kScaleGroup) + index)
			* Format::kStoredBits;
		LmStoreCodePair<Format>(output_codes,bit,row[index] * inverse,row[index + 1u] * inverse);
	}
}
