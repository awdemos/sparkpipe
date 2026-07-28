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

#include "inference/kernels/dtype.cuh"
#include "inference/kernels/formats/bf16.cuh"
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

// SiTU, the activation Kimi K3 runs on all 93 layers. From the released
// modeling_kimi_linear.py:
//
//     situ_a = beta * tanh(gate / beta) * sigmoid(gate)
//     up     = linear_beta * tanh(up / linear_beta)      when linear_beta is set
//     out    = situ_a * up
//
// Compare LmSiluMulKernel directly above: SiLU-mul is gate * sigmoid(gate) * up.
// SiTU replaces the bare gate with beta * tanh(gate / beta) and does the same to
// up with its own beta. Both branches are soft-clamped to +/- their beta, which
// is what Moonshot means by "activation control" - the function is SiLU with
// both arms bounded, not a new shape.
//
// The betas are 4.0 and 25.0 and they are NOT interchangeable: swapping them
// clamps the gate at 25 and the linear arm at 4, which runs and is wrong. They
// are separate arguments rather than an array for that reason.
//
// GATE IS THE FIRST HALF. The reference splits x at d = width/2 and takes
// gate = x[..., :d]. LmSiluMulKernel carries a gate_first flag because that
// convention varies between checkpoints; this one does not vary, so it is fixed
// here rather than made a parameter nobody can set correctly.
template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmSituMulKernel(const uint16_t *__restrict__ gate_up_bf16, uint16_t *__restrict__ output_bf16, uint32_t dimension, float beta, float linear_beta)
{
	uint64_t base = (uint64_t)blockIdx.x * dimension * 2u;
	uint64_t out_base = (uint64_t)blockIdx.x * dimension;
	uint32_t index;
	for (index = threadIdx.x; index < dimension; index += THREADS)
	{
		float gate = LmBf16ToFloat(gate_up_bf16[base + index]);
		float up = LmBf16ToFloat(gate_up_bf16[base + dimension + index]);
		float activated = beta * tanhf(gate / beta) *
			(1.0f / (1.0f + __expf(-gate)));
		if (linear_beta > 0.0f)
			up = linear_beta * tanhf(up / linear_beta);
		output_bf16[out_base + index] = LmFloatToBf16(activated * up);
	}
}

// Sigmoid a row of router logits in place, bf16 in and float out.
//
// K3's router activation is sigmoid, not softmax: config.json sets
// moe_router_activation_func and modeling_kimi_linear.py branches on it. The
// difference matters beyond the curve - sigmoid scores are independent per
// expert, so the top-k weights do not sum to one before renormalisation, which
// is why moe_renormalize exists at all.
template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmSigmoidRowsKernel(const uint16_t *__restrict__ logits_bf16, float *__restrict__ scores, uint32_t width)
{
	uint64_t base = (uint64_t)blockIdx.x * width;
	uint32_t index;
	for (index = threadIdx.x; index < width; index += THREADS)
	{
		float value = LmBf16ToFloat(logits_bf16[base + index]);
		scores[base + index] = 1.0f / (1.0f + __expf(-value));
	}
}

// Gate an attention output elementwise by a sigmoid of its own projection.
//
// TWO MODELS NEED THIS AND NEITHER COULD HAVE IT. Qwen 3.6's reference calls its
// full-attention path GATED attention and sets attn_output_gate; Kimi K3 sets
// mla_use_output_gate. Both were recorded as unimplemented gaps against a kernel
// library that had no gate at all.
//
// The gate is applied AFTER attention and BEFORE the output projection, so it
// scales the attended values rather than the logits. A gate on the logits would
// also run, also produce text, and be a different model.
template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmOutputGateKernel(uint16_t *__restrict__ output_bf16, const uint16_t *__restrict__ gate_bf16, uint32_t dimension)
{
	uint64_t base = (uint64_t)blockIdx.x * dimension;
	uint32_t index;
	for (index = threadIdx.x; index < dimension; index += THREADS)
	{
		float gate = LmBf16ToFloat(gate_bf16[base + index]);
		float value = LmBf16ToFloat(output_bf16[base + index]);
		output_bf16[base + index] =
			LmFloatToBf16(value * (1.0f / (1.0f + __expf(-gate))));
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

// Reduce a routed MoE's packed rows back to token-major, weighted.
//
// Every token was expanded into top_k packed rows, one per expert it routed to,
// and each produced its own output. This folds them back: a token's result is
// the weighted sum of its routes, with the router's gate values as weights.
//
// The step is easy to omit because the shapes almost work without it - the
// packed output is the right width and the wrong number of rows - and omitting
// it gives every token whichever expert happened to land first, which is fluent
// and wrong in a way that looks like a routing bug rather than a missing sum.
//
// The route row map is the same one the quantiser used to expand, read backwards.
template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmMoeFinalizeKernel(const uint16_t *__restrict__ packed_bf16, const uint32_t *__restrict__ packed_row_of_token_route, const float *__restrict__ route_weight, uint16_t *__restrict__ output_bf16, uint32_t tokens, uint32_t top_k, uint32_t dimension)
{
	uint32_t token = blockIdx.y;
	uint32_t element = (blockIdx.x * THREADS) + threadIdx.x;
	uint32_t route;
	float total = 0.0f;
	if ( token >= tokens || element >= dimension )
		return;
	for (route = 0u; route < top_k; ++route)
	{
		uint32_t packed_row = packed_row_of_token_route[(token * top_k) + route];
		total += LmBf16ToFloat(packed_bf16[((uint64_t)packed_row * dimension) + element])
			* route_weight[(token * top_k) + route];
	}
	output_bf16[((uint64_t)token * dimension) + element] = LmFloatToBf16(total);
}

// a + b, elementwise, BF16. For a shared expert's contribution, which is added
// rather than weighted because it has no gate.
template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmAddRowsKernel(const uint16_t *__restrict__ a_bf16, const uint16_t *__restrict__ b_bf16, uint16_t *__restrict__ out_bf16, uint32_t rows, uint32_t dimension)
{
	uint32_t row = blockIdx.y,element = (blockIdx.x * THREADS) + threadIdx.x;
	uint64_t index;
	if ( row >= rows || element >= dimension )
		return;
	index = ((uint64_t)row * dimension) + element;
	out_bf16[index] = LmFloatToBf16(LmBf16ToFloat(a_bf16[index]) + LmBf16ToFloat(b_bf16[index]));
}

// Norm and quantise, fused.
//
// Harvested from the old decode stage's RmsNormFp8E4m3QuantizeKernel, which had
// this and my first version did not. The norm already holds the row in registers
// and the quantiser needs exactly that row; running them as two kernels writes
// the normed hidden to global and reads it straight back.
//
// At 6144 hidden and B128 that is 1.5 MB written and 1.5 MB read per layer, 225
// MB per token across 75 routed layers - on a path where the entire weight
// stream is 5.3 GB. Three percent, for a fusion that costs nothing but saying so.
//
// The absmax needs the whole group before any code can be written, so the row is
// staged in shared rather than kept in registers: a thread cannot know its own
// scale until every thread's maximum is in. That is why this is one block per
// (row, group) rather than one per row.
template<class Format, uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmFusedNormQuantiseKernel(const uint16_t *__restrict__ input_bf16, const uint16_t *__restrict__ residual_bf16, const uint16_t *__restrict__ weight_bf16, const uint32_t *__restrict__ source_row_map, uint16_t *__restrict__ residual_out_bf16, uint8_t *__restrict__ output_codes, uint8_t *__restrict__ output_scales, uint32_t rows, uint32_t dimension, float epsilon)
{
	extern __shared__ float lm_fused_shared[];
	float *row = lm_fused_shared;
	float *reduction = lm_fused_shared + dimension;
	const uint32_t groups = dimension / Format::kScaleGroup;
	uint32_t destination = blockIdx.x,index,group;
	uint64_t source;
	float total = 0.0f,scale;
	if ( destination >= rows )
		return;
	source = (uint64_t)(source_row_map != 0 ? source_row_map[destination] : destination) * dimension;
	for (index = threadIdx.x; index < dimension; index += THREADS)
	{
		float value = LmBf16ToFloat(input_bf16[source + index]);
		if ( residual_bf16 != 0 )
			value += LmBf16ToFloat(residual_bf16[source + index]);
		row[index] = value;
		total += value * value;
		if ( residual_out_bf16 != 0 )
			residual_out_bf16[source + index] = LmFloatToBf16(value);
	}
	total = LmBlockSum<THREADS>(total,reduction);
	scale = rsqrtf((total / (float)dimension) + epsilon);
	for (index = threadIdx.x; index < dimension; index += THREADS)
		row[index] = row[index] * scale * LmBf16ToFloat(weight_bf16[index]);
	__syncthreads();
	// One group at a time: the absmax is per group and every thread needs it
	// before any of them can encode.
	for (group = 0u; group < groups; ++group)
	{
		float absmax = 0.0f,inverse;
		for (index = threadIdx.x; index < Format::kScaleGroup; index += THREADS)
			absmax = fmaxf(absmax,fabsf(row[(group * Format::kScaleGroup) + index]));
		absmax = LmBlockMax<THREADS>(absmax,reduction);
		inverse = 1.0f / fmaxf(absmax / Format::kMax,1.0e-8f);
		if ( threadIdx.x == 0u )
			output_scales[((uint64_t)destination * groups) + group] =
				LmFloatToUe4m3(fmaxf(absmax / Format::kMax,1.0e-8f));
		for (index = threadIdx.x * 2u; index < Format::kScaleGroup; index += THREADS * 2u)
		{
			uint64_t bit = (((uint64_t)destination * dimension)
				+ (group * Format::kScaleGroup) + index) * Format::kStoredBits;
			LmStoreCodePair<Format>(output_codes,bit,
				row[(group * Format::kScaleGroup) + index] * inverse,
				row[(group * Format::kScaleGroup) + index + 1u] * inverse);
		}
		__syncthreads();
	}
}
