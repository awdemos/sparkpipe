#include "sparkpipe/spark_lm_kernels.cuh"
#include "sparkpipe/spark_mimo25_resident_decode_stage_firmware.h"

#include <math.h>

/*
 * MiMo-V2.5 device kernels. The variant model header arrives via the
 * build's -include; nothing below names a variant. Shared machinery
 * (linear over bf16/f32-block fp8, rms norm, head argmax, embedding
 * gather) comes from spark_lm_kernels.cuh; this file holds what MiMo
 * adds: half-split rotate_half rope with a row stride and offset so it
 * runs in place on the fused qkv slices, in-place slice scaling for the
 * pre-cache value fold, the two-pass GQA decode attention that streams
 * the lane's k/v history (full range or the 128-ring with the sink in
 * the denominator), the f32 sigmoid gate, the biased ties-lower top-k
 * select with sum + 1e-20 weight normalization, the plain silu-mul, and
 * the weighted accumulate that lands each expert's OUTPUT times its
 * routing weight. Every kernel stays within fifty lines; the attention
 * decomposes through dot and probability helpers.
 */

// Half-split pairing on the FIRST rope_dim dims of each head: element i
// pairs with i + rope_dim/2. Row stride and offset address a head slice
// inside a wider fused row; the inverse conjugates.
static __global__ void SparkMimo25RopeKernel(void *data_bf16, uint64_t row_stride, uint32_t row_offset, const float *freqs_f32, const uint64_t *row_positions, uint32_t row_count, uint32_t head_count, uint32_t head_dim, uint32_t rope_dim, uint32_t inverse)
{
	uint32_t row = blockIdx.x,head = blockIdx.y,pair = threadIdx.x,half = rope_dim / 2u;
	uint64_t base;
	float angle,cosine,sine,a,b;
	if ( row >= row_count || head >= head_count || pair >= half )
		return;
	base = ((uint64_t)row * row_stride) + row_offset + ((uint64_t)head * head_dim);
	angle = (float)row_positions[row] * __ldg(freqs_f32 + pair);
	__sincosf(angle,&sine,&cosine);
	sine = inverse != 0u ? -sine : sine;
	a = SparkLmBf16ToFloat(data_bf16,base + pair);
	b = SparkLmBf16ToFloat(data_bf16,base + pair + half);
	SparkLmFloatToBf16(data_bf16,base + pair,a * cosine - b * sine);
	SparkLmFloatToBf16(data_bf16,base + pair + half,b * cosine + a * sine);
}

// In-place scale of a column slice of every row - the value fold before
// the cache write.
static __global__ void SparkMimo25ScaleSliceKernel(void *data_bf16, uint64_t row_stride, uint32_t column_offset, uint32_t width, float scale, uint32_t row_count)
{
	uint64_t base = ((uint64_t)blockIdx.x * row_stride) + column_offset;
	uint32_t row = blockIdx.x,element,pair_count = ((base & 1u) == 0u) ? (width >> 1u) : 0u;
	float2 pair_value;
	if ( row >= row_count )
		return;
	for (element = threadIdx.x; element < pair_count; element += blockDim.x)
	{
		pair_value = SparkLmLoadBf16Pair(data_bf16,(base >> 1u) + element);
		SparkLmStoreBf16Pair(data_bf16,(base >> 1u) + element,pair_value.x * scale,pair_value.y * scale);
	}
	for (element = (pair_count << 1u) + threadIdx.x; element < width; element += blockDim.x)
		SparkLmFloatToBf16(data_bf16,base + element,SparkLmBf16ToFloat(data_bf16,base + element) * scale);
}

// Block max into a shared scalar: warp shuffles, per-warp scratch, a
// thread-zero scan, one barrier each side.
// The gate: f32 router weights against bf16 activations, sigmoid scores
// out - one warp per expert, activations staged shared.
static __global__ void SparkMimo25GateScoresKernel(const float *weight_f32, const void *input_bf16, float *scores_f32, uint32_t row_count, uint32_t input_dimension, uint32_t expert_count)
{
	extern __shared__ float gate_shared[];
	uint32_t row = blockIdx.x,expert_base = blockIdx.y * SPARK_LM_CTA_WARPS;
	uint32_t warp = threadIdx.x / SPARK_LM_WARP_LANES,lane = threadIdx.x % SPARK_LM_WARP_LANES,expert,element;
	float accumulator = 0.0f;
	float2 stage_pair,weight_pair;
	if ( row >= row_count )
		return;
	for (element = threadIdx.x; element < (input_dimension >> 1u); element += blockDim.x)
	{
		stage_pair = SparkLmLoadBf16Pair(input_bf16,(((uint64_t)row * input_dimension) >> 1u) + element);
		gate_shared[element << 1u] = stage_pair.x;
		gate_shared[(element << 1u) + 1u] = stage_pair.y;
	}
	__syncthreads();
	expert = expert_base + warp;
	if ( expert >= expert_count )
		return;
	for (element = lane; element < (input_dimension >> 1u); element += SPARK_LM_WARP_LANES)
	{
		weight_pair = __ldg(((const float2 *)weight_f32) + (((uint64_t)expert * input_dimension) >> 1u) + element);
		accumulator = fmaf(gate_shared[element << 1u],weight_pair.x,accumulator);
		accumulator = fmaf(gate_shared[(element << 1u) + 1u],weight_pair.y,accumulator);
	}
	accumulator = SparkLmWarpReduceSum(accumulator);
	if ( lane == 0u )
		scores_f32[((uint64_t)row * expert_count) + expert] = SparkLmSigmoid(accumulator);
}

/*
 * noaux_tc with degenerate groups: plain top-k on scores + bias, ties to
 * the lower index by strict-greater scanning, weights from the ORIGINAL
 * sigmoid scores normalized by sum + epsilon and scaled - the routing
 * weight is applied downstream on the expert OUTPUT.
 */
static __global__ void SparkMimo25GateSelectKernel(const float *scores_f32, const float *bias_f32, uint32_t row_count, uint32_t expert_count, uint32_t topk, float epsilon, float route_scale, uint32_t *indices_u32, float *weights_f32)
{
	uint32_t row = blockIdx.x * blockDim.x + threadIdx.x,rank,expert,best,chosen;
	const float *scores;
	uint32_t *indices;
	float best_score,shifted,total = 0.0f;
	if ( row >= row_count )
		return;
	scores = scores_f32 + ((uint64_t)row * expert_count);
	indices = indices_u32 + ((uint64_t)row * topk);
	for (rank = 0; rank < topk; rank++)
	{
		best = 0xffffffffu;
		best_score = -3.0e38f;
		for (expert = 0; expert < expert_count; expert++)
		{
			for (chosen = 0; chosen < rank; chosen++)
				if ( indices[chosen] == expert )
					break;
			if ( chosen < rank )
				continue;
			shifted = scores[expert] + bias_f32[expert];
			if ( shifted > best_score )
			{
				best_score = shifted;
				best = expert;
			}
		}
		indices[rank] = best;
		total += scores[best];
	}
	for (rank = 0; rank < topk; rank++)
		weights_f32[((uint64_t)row * topk) + rank] = scores[indices[rank]] / (total + epsilon) * route_scale;
}

// Plain silu(gate) * up, no clamp anywhere in this model.
static __global__ void SparkMimo25SiluMulKernel(const void *gate_bf16, void *up_bf16, uint32_t row_count, uint32_t width)
{
	uint32_t row = blockIdx.x,element;
	uint64_t offset = ((uint64_t)row * width) >> 1u;
	float2 gate_pair,up_pair;
	if ( row >= row_count )
		return;
	for (element = threadIdx.x; element < (width >> 1u); element += blockDim.x)
	{
		gate_pair = SparkLmLoadBf16Pair(gate_bf16,offset + element);
		up_pair = SparkLmLoadBf16Pair(up_bf16,offset + element);
		SparkLmStoreBf16Pair(up_bf16,offset + element,SparkLmSwish(gate_pair.x) * up_pair.x,SparkLmSwish(gate_pair.y) * up_pair.y);
	}
	for (element = ((width >> 1u) << 1u) + threadIdx.x; element < width; element += blockDim.x)
		SparkLmFloatToBf16(up_bf16,((uint64_t)row * width) + element,SparkLmSwish(SparkLmBf16ToFloat(gate_bf16,((uint64_t)row * width) + element)) * SparkLmBf16ToFloat(up_bf16,((uint64_t)row * width) + element));
}

// destination += source * weight, the weight read from a device f32 - the
// routed expert's OUTPUT scaled at accumulation, per the reference.
static __global__ void SparkMimo25AccumScaledAddKernel(void *destination_bf16, const void *source_bf16, const float *weight_f32, uint32_t row_count, uint32_t width)
{
	uint32_t row = blockIdx.x,element;
	uint64_t offset = ((uint64_t)row * width) >> 1u;
	float weight = weight_f32 != 0 ? weight_f32[0] : 1.0f;
	float2 destination_pair,source_pair;
	if ( row >= row_count )
		return;
	for (element = threadIdx.x; element < (width >> 1u); element += blockDim.x)
	{
		destination_pair = SparkLmLoadBf16Pair(destination_bf16,offset + element);
		source_pair = SparkLmLoadBf16Pair(source_bf16,offset + element);
		SparkLmStoreBf16Pair(destination_bf16,offset + element,fmaf(source_pair.x,weight,destination_pair.x),fmaf(source_pair.y,weight,destination_pair.y));
	}
	for (element = ((width >> 1u) << 1u) + threadIdx.x; element < width; element += blockDim.x)
		SparkLmFloatToBf16(destination_bf16,((uint64_t)row * width) + element,SparkLmBf16ToFloat(destination_bf16,((uint64_t)row * width) + element) + SparkLmBf16ToFloat(source_bf16,((uint64_t)row * width) + element) * weight);
}

static __global__ void SparkMimo25ResidualAddKernel(void *hidden_bf16, const void *delta_bf16, uint32_t row_count, uint32_t width)
{
	uint32_t row = blockIdx.x,element;
	uint64_t offset = ((uint64_t)row * width) >> 1u;
	float2 hidden_pair,delta_pair;
	if ( row >= row_count )
		return;
	for (element = threadIdx.x; element < (width >> 1u); element += blockDim.x)
	{
		hidden_pair = SparkLmLoadBf16Pair(hidden_bf16,offset + element);
		delta_pair = SparkLmLoadBf16Pair(delta_bf16,offset + element);
		SparkLmStoreBf16Pair(hidden_bf16,offset + element,hidden_pair.x + delta_pair.x,hidden_pair.y + delta_pair.y);
	}
	for (element = ((width >> 1u) << 1u) + threadIdx.x; element < width; element += blockDim.x)
		SparkLmFloatToBf16(hidden_bf16,((uint64_t)row * width) + element,SparkLmBf16ToFloat(hidden_bf16,((uint64_t)row * width) + element) + SparkLmBf16ToFloat(delta_bf16,((uint64_t)row * width) + element));
}

extern "C" cudaError_t SparkMimo25LaunchRmsNorm(cudaStream_t stream, const void *input_bf16, const void *gain_bf16, void *output_bf16, uint32_t row_count, uint32_t dimension, float epsilon)
{
	SparkLmRmsNormKernel<<<row_count,SPARK_LM_CTA_THREADS,0,stream>>>(input_bf16,gain_bf16,output_bf16,row_count,dimension,epsilon);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkMimo25LaunchLinear(cudaStream_t stream, const SparkMimo25LinearView *view, const void *payload, const void *scale, const void *input_bf16, void *output_bf16, uint32_t row_count)
{
	dim3 grid(row_count,(view->rows + SPARK_LM_CTA_WARPS - 1u) / SPARK_LM_CTA_WARPS);
	uint32_t shared_bytes = view->columns * (uint32_t)sizeof(float);
	SparkLmLinearKernel<128u><<<grid,SPARK_LM_CTA_THREADS,shared_bytes,stream>>>(view->weight_format,payload != 0 ? payload : view->payload,scale != 0 ? scale : view->scale,input_bf16,output_bf16,row_count,view->columns,view->rows);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkMimo25LaunchEmbeddingGather(cudaStream_t stream, const uint32_t *token_ids, const void *embedding_bf16, void *hidden_bf16, uint32_t row_count, uint32_t hidden_dimension)
{
	SparkLmEmbeddingGatherKernel<<<row_count,SPARK_LM_CTA_THREADS,0,stream>>>(token_ids,embedding_bf16,hidden_bf16,row_count,hidden_dimension);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkMimo25LaunchHeadArgmax(cudaStream_t stream, const void *hidden_bf16, const void *head_weight_bf16, uint32_t *output_token_ids, uint32_t row_count, uint32_t candidate_count, uint32_t hidden_dimension)
{
	SparkLmHeadArgmaxKernel<<<row_count,SPARK_LM_CTA_THREADS,0,stream>>>(hidden_bf16,head_weight_bf16,0,output_token_ids,row_count,hidden_dimension,candidate_count);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkMimo25LaunchRope(cudaStream_t stream, void *data_bf16, uint64_t row_stride, uint32_t row_offset, const float *freqs_f32, const uint64_t *row_positions, uint32_t row_count, uint32_t head_count, uint32_t head_dim, uint32_t rope_dim, uint32_t inverse)
{
	dim3 grid(row_count,head_count);
	SparkMimo25RopeKernel<<<grid,rope_dim / 2u,0,stream>>>(data_bf16,row_stride,row_offset,freqs_f32,row_positions,row_count,head_count,head_dim,rope_dim,inverse);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkMimo25LaunchScaleSlice(cudaStream_t stream, void *data_bf16, uint64_t row_stride, uint32_t column_offset, uint32_t width, float scale, uint32_t row_count)
{
	SparkMimo25ScaleSliceKernel<<<row_count,SPARK_LM_CTA_THREADS,0,stream>>>(data_bf16,row_stride,column_offset,width,scale,row_count);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkMimo25LaunchAttnDecode(cudaStream_t stream, const void *q_bf16, uint64_t q_row_stride, const void *k_cache_bf16, const void *v_cache_bf16, uint64_t k_lane_stride, uint64_t v_lane_stride, uint64_t k_slot_stride, uint64_t v_slot_stride, const uint32_t *row_lane_indices, const uint64_t *row_positions, const float *sink_f32, float scale, void *out_bf16, uint32_t row_count, uint32_t head_count, uint32_t group_size, uint32_t head_dim, uint32_t value_dim, uint32_t window_slots)
{
	dim3 grid(row_count,head_count);
	uint32_t shared_bytes = (head_dim + (SPARK_LM_CTA_WARPS * value_dim)) * (uint32_t)sizeof(float);
	SparkLmAttnDecodeKernel<<<grid,SPARK_LM_CTA_THREADS,shared_bytes,stream>>>(q_bf16,q_row_stride,k_cache_bf16,v_cache_bf16,k_lane_stride,v_lane_stride,k_slot_stride,v_slot_stride,row_lane_indices,row_positions,sink_f32,scale,out_bf16,row_count,head_count,group_size,head_dim,value_dim,window_slots);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkMimo25LaunchGateScores(cudaStream_t stream, const SparkMimo25LinearView *gate, const void *input_bf16, float *scores_f32, uint32_t row_count)
{
	dim3 grid(row_count,(gate->rows + SPARK_LM_CTA_WARPS - 1u) / SPARK_LM_CTA_WARPS);
	SparkMimo25GateScoresKernel<<<grid,SPARK_LM_CTA_THREADS,gate->columns * (uint32_t)sizeof(float),stream>>>((const float *)gate->payload,input_bf16,scores_f32,row_count,gate->columns,gate->rows);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkMimo25LaunchGateSelect(cudaStream_t stream, const float *scores_f32, const float *bias_f32, uint32_t row_count, uint32_t expert_count, uint32_t topk, float epsilon, float route_scale, uint32_t *indices_u32, float *weights_f32)
{
	SparkMimo25GateSelectKernel<<<(row_count + 63u) / 64u,64u,0,stream>>>(scores_f32,bias_f32,row_count,expert_count,topk,epsilon,route_scale,indices_u32,weights_f32);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkMimo25LaunchSiluMul(cudaStream_t stream, const void *gate_bf16, void *up_bf16, uint32_t row_count, uint32_t width)
{
	SparkMimo25SiluMulKernel<<<row_count,SPARK_LM_CTA_THREADS,0,stream>>>(gate_bf16,up_bf16,row_count,width);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkMimo25LaunchGatherLinear(cudaStream_t stream, const SparkMimo25LinearView *view, const void *payload, const void *scale, const void *input_bf16, const uint32_t *input_row_map, void *output_bf16, uint32_t slot_count)
{
	dim3 grid(slot_count,(view->rows + SPARK_LM_CTA_WARPS - 1u) / SPARK_LM_CTA_WARPS);
	uint32_t shared_bytes = view->columns * (uint32_t)sizeof(float);
	SparkLmGatherLinearKernel<128u><<<grid,SPARK_LM_CTA_THREADS,shared_bytes,stream>>>(view->weight_format,payload != 0 ? payload : view->payload,scale != 0 ? scale : view->scale,input_bf16,input_row_map,output_bf16,slot_count,view->columns,view->rows);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkMimo25LaunchExpertTile(cudaStream_t stream, const SparkMimo25LinearView *view, const void *payload, const void *scale, const void *input_bf16, const uint32_t *input_row_map, void *output_bf16, uint32_t slot_count)
{
	dim3 grid((slot_count + SPARK_LM_TILE - 1u) / SPARK_LM_TILE,(view->rows + SPARK_LM_TILE_N - 1u) / SPARK_LM_TILE_N);
	SparkLmExpertTileKernel<128u><<<grid,SPARK_LM_CTA_THREADS,0,stream>>>(view->weight_format,payload != 0 ? payload : view->payload,scale != 0 ? scale : view->scale,input_bf16,input_row_map,output_bf16,slot_count,view->columns,view->rows);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkMimo25LaunchScatterScaledAdd(cudaStream_t stream, void *destination_bf16, const void *source_bf16, const uint32_t *row_map, const float *weights_f32, const uint32_t *weight_map, uint32_t slot_count, uint32_t width)
{
	SparkLmScatterScaledAddKernel<<<slot_count,SPARK_LM_CTA_THREADS,0,stream>>>(destination_bf16,source_bf16,row_map,weights_f32,weight_map,slot_count,width);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkMimo25LaunchAccumScaledAdd(cudaStream_t stream, void *destination_bf16, const void *source_bf16, const float *weight_f32, uint32_t row_count, uint32_t width)
{
	SparkMimo25AccumScaledAddKernel<<<row_count,SPARK_LM_CTA_THREADS,0,stream>>>(destination_bf16,source_bf16,weight_f32,row_count,width);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkMimo25LaunchResidualAdd(cudaStream_t stream, void *hidden_bf16, const void *delta_bf16, uint32_t row_count, uint32_t width)
{
	SparkMimo25ResidualAddKernel<<<row_count,SPARK_LM_CTA_THREADS,0,stream>>>(hidden_bf16,delta_bf16,row_count,width);
	return(cudaGetLastError());
}
