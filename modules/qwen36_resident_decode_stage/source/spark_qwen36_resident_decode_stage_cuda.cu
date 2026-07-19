#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "sparkpipe/spark_qwen36_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_lm_kernels.cuh"

/*
 * Qwen 3.6 27B decode-path device code. Every kernel here serves the
 * production hot path: a decode microbatch of up to 512 rows, one next token
 * per distinct lane, walked through this stage's layer slice. Prefill in v1
 * is the same path applied token by token, which the carry oracle proves
 * BITWISE equal to any chunked formulation; the chunked prefill kernel is a
 * later throughput commit, not a correctness requirement.
 *
 * Shared machinery (RmsNorm, dual-format Linear, fused argmax head, reduces)
 * comes from spark_lm_kernels.cuh; this file holds only what is Qwen:
 * the depthwise conv state update, the recurrent gated delta step operating
 * in-place on the resident state pool, the fp32 gated head norm, and paged
 * GQA attention with per-head query|gate fusion, q/k head norms and partial
 * RoPE. All forms are the PINNED modeling_qwen3_5 forms, oracle-matched.
 *
 * Grid conventions: one block per (row, head) for head-shaped work, one
 * block per row for row-shaped work; row order everywhere follows the frame
 * decode batch view. Launchers are extern "C" and stream-ordered; nothing
 * here synchronizes.
 */

#define SPARK_QWEN36_CUDA_DK SPARK_QWEN36_MODEL_GDN_HEAD_KEY_DIMENSION
#define SPARK_QWEN36_CUDA_DV SPARK_QWEN36_MODEL_GDN_HEAD_VALUE_DIMENSION
#define SPARK_QWEN36_CUDA_GVA_GROUP (SPARK_QWEN36_MODEL_GDN_VALUE_HEAD_COUNT / SPARK_QWEN36_MODEL_GDN_KEY_HEAD_COUNT)
#define SPARK_QWEN36_CUDA_ATTN_GROUP (SPARK_QWEN36_MODEL_ATTN_QUERY_HEAD_COUNT / SPARK_QWEN36_MODEL_ATTN_KV_HEAD_COUNT)

static __device__ __forceinline__ float SparkQwen36RopeFrequency(uint32_t pair)
{
	return(exp2f(-((float)(2u * pair) / (float)SPARK_QWEN36_MODEL_ATTN_ROPE_DIMENSION) * log2f((float)SPARK_QWEN36_MODEL_ATTN_ROPE_THETA)));
}

// Depthwise causal conv update for one decode token per row: one thread per
// (row, channel), window = carried tail (3) plus the fresh projection, dot
// with the 4-tap weight, silu, then rotate the tail in place. Cold rows read
// a zero tail. Matches causal_conv1d_update with bias absent.
static __global__ void SparkQwen36ConvUpdateKernel(const void *qkv_bf16, const void *conv_weight_bf16, void *conv_out_bf16, void *conv_tail_bf16, const uint32_t *row_lane_indices, const uint32_t *state_cold_by_row, uint32_t row_count, uint32_t gdn_layer_ordinal, uint64_t tail_lane_stride, uint64_t tail_layer_stride)
{
	uint32_t row = blockIdx.y,channel = (blockIdx.x * blockDim.x) + threadIdx.x;
	uint64_t tail_base;
	float window[4],accumulator;
	uint32_t tap;
	if ( row >= row_count || channel >= SPARK_QWEN36_MODEL_GDN_CONV_CHANNELS )
		return;
	tail_base = ((uint64_t)row_lane_indices[row] * tail_lane_stride) + ((uint64_t)gdn_layer_ordinal * tail_layer_stride) + ((uint64_t)channel * 3u);
	if ( state_cold_by_row[row] != 0u )
	{
		window[0] = 0.0f;
		window[1] = 0.0f;
		window[2] = 0.0f;
	}
	else
	{
		window[0] = SparkLmBf16ToFloat(conv_tail_bf16,tail_base + 0u);
		window[1] = SparkLmBf16ToFloat(conv_tail_bf16,tail_base + 1u);
		window[2] = SparkLmBf16ToFloat(conv_tail_bf16,tail_base + 2u);
	}
	window[3] = SparkLmBf16ToFloat(qkv_bf16,((uint64_t)row * SPARK_QWEN36_MODEL_GDN_CONV_CHANNELS) + channel);
	accumulator = 0.0f;
	for (tap = 0; tap < SPARK_QWEN36_MODEL_GDN_CONV_KERNEL; tap++)
		accumulator += (window[tap] * SparkLmBf16ToFloat(conv_weight_bf16,((uint64_t)channel * SPARK_QWEN36_MODEL_GDN_CONV_KERNEL) + tap));
	SparkLmFloatToBf16(conv_out_bf16,((uint64_t)row * SPARK_QWEN36_MODEL_GDN_CONV_CHANNELS) + channel,SparkLmSwish(accumulator));
	SparkLmFloatToBf16(conv_tail_bf16,tail_base + 0u,window[1]);
	SparkLmFloatToBf16(conv_tail_bf16,tail_base + 1u,window[2]);
	SparkLmFloatToBf16(conv_tail_bf16,tail_base + 2u,window[3]);
}

// Per-head log decay and beta from the two 48-row projections plus the fp32
// decay parameters: g = -exp(a_log) * softplus(a + dt_bias), beta =
// sigmoid(b). One thread per (row, value head).
static __global__ void SparkQwen36DecayBetaKernel(const void *decay_pre_bf16, const void *beta_pre_bf16, const float *a_log_f32, const float *dt_bias_f32, float *log_decay_f32, float *beta_f32, uint32_t row_count)
{
	uint32_t row = blockIdx.x,head = threadIdx.x;
	uint64_t index;
	if ( row >= row_count || head >= SPARK_QWEN36_MODEL_GDN_VALUE_HEAD_COUNT )
		return;
	index = ((uint64_t)row * SPARK_QWEN36_MODEL_GDN_VALUE_HEAD_COUNT) + head;
	log_decay_f32[index] = -expf(a_log_f32[head]) * SparkLmSoftplus(SparkLmBf16ToFloat(decay_pre_bf16,index) + dt_bias_f32[head]);
	beta_f32[index] = SparkLmSigmoid(SparkLmBf16ToFloat(beta_pre_bf16,index));
}

/*
 * Recurrent gated delta step, one block per (row, value head), 128 threads,
 * thread j owning state column j so every state access is coalesced. The
 * conv output is channel order query(2048) | key(2048) | value(6144); the
 * key head for value head h is h / 3 (GVA). q and k are L2-normalized per
 * head with eps 1e-6 and q is scaled 1/sqrt(dk), matching the oracle
 * recurrence bitwise in structure: decay, predict, delta, rank-one update,
 * read-out. State is fp32 in the resident pool and updated in place; cold
 * rows start from zero without a separate memset pass.
 */
static __global__ void SparkQwen36GdnStepKernel(const void *conv_out_bf16, const float *log_decay_f32, const float *beta_f32, float *state_f32, void *core_out_bf16, const uint32_t *row_lane_indices, const uint32_t *state_cold_by_row, uint32_t row_count, uint32_t gdn_layer_ordinal, uint64_t state_lane_stride, uint64_t state_layer_stride)
{
	__shared__ float qn[SPARK_QWEN36_CUDA_DK],kn[SPARK_QWEN36_CUDA_DK],reduce_scratch[SPARK_LM_CTA_WARPS];
	uint32_t row = blockIdx.y,head = blockIdx.x,column = threadIdx.x,key_head = head / SPARK_QWEN36_CUDA_GVA_GROUP,element;
	uint64_t conv_row = (uint64_t)row * SPARK_QWEN36_MODEL_GDN_CONV_CHANNELS,state_base;
	float value,q_norm,k_norm,decay,beta,kv_mem,delta,output;
	if ( row >= row_count )
		return;
	value = SparkLmBf16ToFloat(conv_out_bf16,conv_row + ((uint64_t)key_head * SPARK_QWEN36_CUDA_DK) + column);
	q_norm = SparkLmBlockReduceSum(value * value,reduce_scratch);
	qn[column] = value * rsqrtf(q_norm + 1e-6f) * rsqrtf((float)SPARK_QWEN36_CUDA_DK);
	value = SparkLmBf16ToFloat(conv_out_bf16,conv_row + SPARK_QWEN36_MODEL_GDN_QK_DIMENSION + ((uint64_t)key_head * SPARK_QWEN36_CUDA_DK) + column);
	k_norm = SparkLmBlockReduceSum(value * value,reduce_scratch);
	kn[column] = value * rsqrtf(k_norm + 1e-6f);
	__syncthreads();
	state_base = ((uint64_t)row_lane_indices[row] * state_lane_stride) + ((uint64_t)gdn_layer_ordinal * state_layer_stride) + ((uint64_t)head * SPARK_QWEN36_CUDA_DK * SPARK_QWEN36_CUDA_DV);
	decay = state_cold_by_row[row] != 0u ? 0.0f : expf(log_decay_f32[((uint64_t)row * SPARK_QWEN36_MODEL_GDN_VALUE_HEAD_COUNT) + head]);
	beta = beta_f32[((uint64_t)row * SPARK_QWEN36_MODEL_GDN_VALUE_HEAD_COUNT) + head];
	kv_mem = 0.0f;
	for (element = 0; element < SPARK_QWEN36_CUDA_DK; element++)
	{
		value = state_f32[state_base + ((uint64_t)element * SPARK_QWEN36_CUDA_DV) + column] * decay;
		state_f32[state_base + ((uint64_t)element * SPARK_QWEN36_CUDA_DV) + column] = value;
		kv_mem += (value * kn[element]);
	}
	delta = (SparkLmBf16ToFloat(conv_out_bf16,conv_row + (2u * SPARK_QWEN36_MODEL_GDN_QK_DIMENSION) + ((uint64_t)head * SPARK_QWEN36_CUDA_DV) + column) - kv_mem) * beta;
	output = 0.0f;
	for (element = 0; element < SPARK_QWEN36_CUDA_DK; element++)
	{
		value = state_f32[state_base + ((uint64_t)element * SPARK_QWEN36_CUDA_DV) + column] + (kn[element] * delta);
		state_f32[state_base + ((uint64_t)element * SPARK_QWEN36_CUDA_DV) + column] = value;
		output += (value * qn[element]);
	}
	SparkLmFloatToBf16(core_out_bf16,((uint64_t)row * SPARK_QWEN36_MODEL_GDN_VALUE_DIMENSION) + ((uint64_t)head * SPARK_QWEN36_CUDA_DV) + column,output);
}

// Gated head norm: fp32 RMSNorm over one value head, times weight, times
// silu(z). One block per (row, head), 128 threads. Norm before gate.
static __global__ void SparkQwen36GatedNormKernel(const void *core_bf16, const void *z_bf16, const void *norm_weight_bf16, void *output_bf16, uint32_t row_count, float epsilon)
{
	__shared__ float reduce_scratch[SPARK_LM_CTA_WARPS];
	uint32_t row = blockIdx.y,head = blockIdx.x,column = threadIdx.x;
	uint64_t index = ((uint64_t)row * SPARK_QWEN36_MODEL_GDN_VALUE_DIMENSION) + ((uint64_t)head * SPARK_QWEN36_CUDA_DV) + column;
	float value,variance;
	if ( row >= row_count )
		return;
	value = SparkLmBf16ToFloat(core_bf16,index);
	variance = SparkLmBlockReduceSum(value * value,reduce_scratch) / (float)SPARK_QWEN36_CUDA_DV;
	value = value * rsqrtf(variance + epsilon) * SparkLmBf16ToFloat(norm_weight_bf16,column) * SparkLmSwish(SparkLmBf16ToFloat(z_bf16,index));
	SparkLmFloatToBf16(output_bf16,index,value);
}

/*
 * Attention pre-pass for one decode token per row: per-head RMSNorm on the
 * query half of the fused query|gate projection and on the key projection,
 * partial RoPE on the first 64 dims of both, then the K and V rows land in
 * the paged cache at the row's slot. Fused layout: head h occupies columns
 * [h*512, h*512+256) query and [h*512+256, h*512+512) gate; the gate half is
 * left untouched here for the decode kernel to consume. One block per
 * (row, query head); key/value heads are written by the blocks whose query
 * head is the group leader so each cache row is written exactly once.
 */
static __global__ void SparkQwen36AttnPrepareKernel(void *q_fused_bf16, const void *k_bf16, const void *v_bf16, const void *q_norm_weight_bf16, const void *k_norm_weight_bf16, void *kv_cache_bf16, const uint32_t *slot_mapping, const uint64_t *row_positions, uint32_t row_count, uint32_t attn_layer_ordinal, uint64_t cache_layer_stride, uint64_t cache_block_stride, float epsilon)
{
	__shared__ float reduce_scratch[SPARK_LM_CTA_WARPS];
	uint32_t row = blockIdx.y,head = blockIdx.x,column = threadIdx.x,kv_head = head / SPARK_QWEN36_CUDA_ATTN_GROUP,pair;
	uint32_t slot,block,offset;
	uint64_t q_index = ((uint64_t)row * 2u * SPARK_QWEN36_MODEL_ATTN_QUERY_DIMENSION) + ((uint64_t)head * 2u * SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION) + column;
	uint64_t k_index = ((uint64_t)row * SPARK_QWEN36_MODEL_ATTN_KV_DIMENSION) + ((uint64_t)kv_head * SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION) + column;
	uint64_t cache_base;
	float value,variance,angle,cosine,sine,partner;
	if ( row >= row_count )
		return;
	value = SparkLmBf16ToFloat(q_fused_bf16,q_index);
	variance = SparkLmBlockReduceSum(value * value,reduce_scratch) / (float)SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION;
	value = value * rsqrtf(variance + epsilon) * SparkLmBf16ToFloat(q_norm_weight_bf16,column);
	SparkLmFloatToBf16(q_fused_bf16,q_index,value);
	__syncthreads();
	if ( column < SPARK_QWEN36_MODEL_ATTN_ROPE_DIMENSION / 2u )
	{
		pair = column;
		angle = (float)row_positions[row] * SparkQwen36RopeFrequency(pair);
		cosine = cosf(angle);
		sine = sinf(angle);
		value = SparkLmBf16ToFloat(q_fused_bf16,q_index - column + pair);
		partner = SparkLmBf16ToFloat(q_fused_bf16,q_index - column + pair + (SPARK_QWEN36_MODEL_ATTN_ROPE_DIMENSION / 2u));
		SparkLmFloatToBf16(q_fused_bf16,q_index - column + pair,(value * cosine) - (partner * sine));
		SparkLmFloatToBf16(q_fused_bf16,q_index - column + pair + (SPARK_QWEN36_MODEL_ATTN_ROPE_DIMENSION / 2u),(partner * cosine) + (value * sine));
	}
	if ( (head % SPARK_QWEN36_CUDA_ATTN_GROUP) != 0u )
		return;
	__syncthreads();
	slot = slot_mapping[row];
	block = slot / SPARK_QWEN36_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS;
	offset = slot % SPARK_QWEN36_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS;
	cache_base = ((uint64_t)block * cache_block_stride) + ((uint64_t)attn_layer_ordinal * cache_layer_stride) + ((uint64_t)offset * SPARK_QWEN36_MODEL_ATTN_CACHE_TOKEN_ELEMENTS) + ((uint64_t)kv_head * SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION) + column;
	value = SparkLmBf16ToFloat(k_bf16,k_index);
	variance = SparkLmBlockReduceSum(value * value,reduce_scratch) / (float)SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION;
	value = value * rsqrtf(variance + epsilon) * SparkLmBf16ToFloat(k_norm_weight_bf16,column);
	SparkLmFloatToBf16(kv_cache_bf16,cache_base,value);
	__syncthreads();
	if ( column < SPARK_QWEN36_MODEL_ATTN_ROPE_DIMENSION / 2u )
	{
		pair = column;
		angle = (float)row_positions[row] * SparkQwen36RopeFrequency(pair);
		cosine = cosf(angle);
		sine = sinf(angle);
		value = SparkLmBf16ToFloat(kv_cache_bf16,cache_base - column + pair);
		partner = SparkLmBf16ToFloat(kv_cache_bf16,cache_base - column + pair + (SPARK_QWEN36_MODEL_ATTN_ROPE_DIMENSION / 2u));
		SparkLmFloatToBf16(kv_cache_bf16,cache_base - column + pair,(value * cosine) - (partner * sine));
		SparkLmFloatToBf16(kv_cache_bf16,cache_base - column + pair + (SPARK_QWEN36_MODEL_ATTN_ROPE_DIMENSION / 2u),(partner * cosine) + (value * sine));
	}
	SparkLmFloatToBf16(kv_cache_bf16,cache_base + SPARK_QWEN36_MODEL_ATTN_KV_DIMENSION,SparkLmBf16ToFloat(v_bf16,k_index));
}

/*
 * Paged GQA decode attention with online softmax and the fused sigmoid
 * output gate. One block per (row, query head), one warp lane per cache
 * position stripe; the value accumulation runs per thread over the head dim
 * in registers with an online max/denominator rescale. This is the simple
 * correct kernel: the tensor-core version is a later throughput commit.
 */
static __global__ void SparkQwen36AttnDecodeKernel(const void *q_fused_bf16, const void *kv_cache_bf16, const uint32_t *block_indices, const uint32_t *block_counts, const uint32_t *row_lane_indices, const uint32_t *context_lengths, void *head_out_bf16, uint32_t row_count, uint32_t lane_stride, uint32_t attn_layer_ordinal, uint64_t cache_layer_stride, uint64_t cache_block_stride)
{
	__shared__ float reduce_scratch[SPARK_LM_CTA_WARPS],running_max,running_denominator,rescale;
	uint32_t row = blockIdx.y,head = blockIdx.x,column = threadIdx.x,kv_head = head / SPARK_QWEN36_CUDA_ATTN_GROUP;
	uint64_t q_base = ((uint64_t)row * 2u * SPARK_QWEN36_MODEL_ATTN_QUERY_DIMENSION) + ((uint64_t)head * 2u * SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION);
	float query = 0.0f,accumulator = 0.0f,score,probability,gate;
	uint32_t token,block,offset,context;
	if ( row >= row_count )
		return;
	query = SparkLmBf16ToFloat(q_fused_bf16,q_base + column);
	if ( threadIdx.x == 0u )
	{
		running_max = -3.0e38f;
		running_denominator = 0.0f;
	}
	__syncthreads();
	context = context_lengths[row];
	for (token = 0; token < context; token++)
	{
		block = block_indices[((uint64_t)row_lane_indices[row] * lane_stride) + (token / SPARK_QWEN36_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS)];
		offset = token % SPARK_QWEN36_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS;
		{
			uint64_t cache_base = ((uint64_t)block * cache_block_stride) + ((uint64_t)attn_layer_ordinal * cache_layer_stride) + ((uint64_t)offset * SPARK_QWEN36_MODEL_ATTN_CACHE_TOKEN_ELEMENTS) + ((uint64_t)kv_head * SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION);
			score = SparkLmBlockReduceSum(query * SparkLmBf16ToFloat(kv_cache_bf16,cache_base + column),reduce_scratch) * (1.0f / 16.0f);
			if ( threadIdx.x == 0u )
			{
				if ( score > running_max )
				{
					rescale = __expf(running_max - score);
					running_max = score;
				}
				else
					rescale = 1.0f;
				running_denominator = (running_denominator * rescale) + __expf(score - running_max);
				reduce_scratch[0] = __expf(score - running_max);
				reduce_scratch[1] = rescale;
			}
			__syncthreads();
			probability = reduce_scratch[0];
			accumulator = (accumulator * reduce_scratch[1]) + (probability * SparkLmBf16ToFloat(kv_cache_bf16,cache_base + SPARK_QWEN36_MODEL_ATTN_KV_DIMENSION + column));
			__syncthreads();
		}
	}
	gate = SparkLmSigmoid(SparkLmBf16ToFloat(q_fused_bf16,q_base + SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION + column));
	SparkLmFloatToBf16(head_out_bf16,((uint64_t)row * SPARK_QWEN36_MODEL_ATTN_QUERY_DIMENSION) + ((uint64_t)head * SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION) + column,(accumulator / running_denominator) * gate);
}

// Embedding gather: one thread per (row, element); token ids are validated
// against the vocabulary on the host before upload, so the kernel trusts.
static __global__ void SparkQwen36EmbeddingGatherKernel(const uint32_t *token_ids, const void *embedding_bf16, void *hidden_bf16, uint32_t row_count)
{
	uint64_t index = ((uint64_t)blockIdx.x * blockDim.x) + threadIdx.x;
	uint32_t row = (uint32_t)(index / SPARK_QWEN36_MODEL_HIDDEN_DIMENSION),element = (uint32_t)(index % SPARK_QWEN36_MODEL_HIDDEN_DIMENSION);
	if ( row >= row_count )
		return;
	SparkLmFloatToBf16(hidden_bf16,index,SparkLmBf16ToFloat(embedding_bf16,((uint64_t)token_ids[row] * SPARK_QWEN36_MODEL_HIDDEN_DIMENSION) + element));
}

// Residual add and SwiGLU combine, both row-shaped elementwise.
static __global__ void SparkQwen36ResidualAddKernel(void *hidden_bf16, const void *delta_bf16, uint32_t row_count, uint32_t dimension)
{
	uint64_t index = ((uint64_t)blockIdx.x * blockDim.x) + threadIdx.x;
	if ( index >= ((uint64_t)row_count * dimension) )
		return;
	SparkLmFloatToBf16(hidden_bf16,index,SparkLmBf16ToFloat(hidden_bf16,index) + SparkLmBf16ToFloat(delta_bf16,index));
}

static __global__ void SparkQwen36SwiGluKernel(const void *gate_bf16, void *up_bf16, uint32_t row_count, uint32_t dimension)
{
	uint64_t index = ((uint64_t)blockIdx.x * blockDim.x) + threadIdx.x;
	if ( index >= ((uint64_t)row_count * dimension) )
		return;
	SparkLmFloatToBf16(up_bf16,index,SparkLmSwish(SparkLmBf16ToFloat(gate_bf16,index)) * SparkLmBf16ToFloat(up_bf16,index));
}

extern "C" cudaError_t SparkQwen36LaunchRmsNorm(cudaStream_t stream, const void *input_bf16, const void *gain_bf16, void *output_bf16, uint32_t row_count, uint32_t dimension, float epsilon)
{
	SparkLmRmsNormKernel<<<row_count,SPARK_LM_CTA_THREADS,0,stream>>>(input_bf16,gain_bf16,output_bf16,row_count,dimension,epsilon);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkQwen36LaunchLinear(cudaStream_t stream, const SparkQwen36LinearView *view, const void *input_bf16, void *output_bf16, uint32_t row_count)
{
	dim3 grid(row_count,(view->output_dimension + SPARK_LM_CTA_WARPS - 1u) / SPARK_LM_CTA_WARPS,1u);
	uint32_t shared_bytes = view->input_dimension * (uint32_t)sizeof(float);
	SparkLmLinearKernel<32u><<<grid,SPARK_LM_CTA_THREADS,shared_bytes,stream>>>(view->weight_format,view->weight_payload,view->weight_scale_e8m0,input_bf16,output_bf16,row_count,view->input_dimension,view->output_dimension);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkQwen36LaunchConvUpdate(cudaStream_t stream, const void *qkv_bf16, const SparkQwen36GdnLayerWeights *weights, void *conv_out_bf16, const SparkQwen36GdnStatePool *pool, const uint32_t *row_lane_indices, uint32_t row_count, uint32_t gdn_layer_ordinal)
{
	dim3 grid((SPARK_QWEN36_MODEL_GDN_CONV_CHANNELS + SPARK_LM_CTA_THREADS - 1u) / SPARK_LM_CTA_THREADS,row_count,1u);
	SparkQwen36ConvUpdateKernel<<<grid,SPARK_LM_CTA_THREADS,0,stream>>>(qkv_bf16,weights->conv_weight_bf16,conv_out_bf16,pool->conv_tail_bf16,row_lane_indices,pool->state_cold_by_row,row_count,gdn_layer_ordinal,pool->conv_tail_lane_stride_elements,pool->conv_tail_layer_stride_elements);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkQwen36LaunchDecayBeta(cudaStream_t stream, const void *decay_pre_bf16, const void *beta_pre_bf16, const SparkQwen36GdnLayerWeights *weights, float *log_decay_f32, float *beta_f32, uint32_t row_count)
{
	SparkQwen36DecayBetaKernel<<<row_count,SPARK_QWEN36_MODEL_GDN_VALUE_HEAD_COUNT,0,stream>>>(decay_pre_bf16,beta_pre_bf16,weights->a_log_f32,weights->dt_bias_f32,log_decay_f32,beta_f32,row_count);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkQwen36LaunchGdnStep(cudaStream_t stream, const void *conv_out_bf16, const float *log_decay_f32, const float *beta_f32, const SparkQwen36GdnStatePool *pool, void *core_out_bf16, const uint32_t *row_lane_indices, uint32_t row_count, uint32_t gdn_layer_ordinal)
{
	dim3 grid(SPARK_QWEN36_MODEL_GDN_VALUE_HEAD_COUNT,row_count,1u);
	SparkQwen36GdnStepKernel<<<grid,SPARK_QWEN36_CUDA_DV,0,stream>>>(conv_out_bf16,log_decay_f32,beta_f32,pool->state_f32,core_out_bf16,row_lane_indices,pool->state_cold_by_row,row_count,gdn_layer_ordinal,pool->state_lane_stride_elements,pool->state_layer_stride_elements);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkQwen36LaunchGatedNorm(cudaStream_t stream, const void *core_bf16, const void *z_bf16, const SparkQwen36GdnLayerWeights *weights, void *output_bf16, uint32_t row_count, float epsilon)
{
	dim3 grid(SPARK_QWEN36_MODEL_GDN_VALUE_HEAD_COUNT,row_count,1u);
	SparkQwen36GatedNormKernel<<<grid,SPARK_QWEN36_CUDA_DV,0,stream>>>(core_bf16,z_bf16,weights->gdn_norm_weight_bf16,output_bf16,row_count,epsilon);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkQwen36LaunchAttnPrepare(cudaStream_t stream, void *q_fused_bf16, const void *k_bf16, const void *v_bf16, const SparkQwen36AttnLayerWeights *weights, void *kv_cache_bf16, const uint32_t *slot_mapping, const uint64_t *row_positions, uint32_t row_count, uint32_t attn_layer_ordinal, uint64_t cache_layer_stride, uint64_t cache_block_stride, float epsilon)
{
	dim3 grid(SPARK_QWEN36_MODEL_ATTN_QUERY_HEAD_COUNT,row_count,1u);
	SparkQwen36AttnPrepareKernel<<<grid,SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION,0,stream>>>(q_fused_bf16,k_bf16,v_bf16,weights->query_norm_weight_bf16,weights->key_norm_weight_bf16,kv_cache_bf16,slot_mapping,row_positions,row_count,attn_layer_ordinal,cache_layer_stride,cache_block_stride,epsilon);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkQwen36LaunchAttnDecode(cudaStream_t stream, const void *q_fused_bf16, const void *kv_cache_bf16, const SparkQwen36KvBlockTableView *table, const uint32_t *row_lane_indices, const uint32_t *context_lengths, void *head_out_bf16, uint32_t row_count, uint32_t attn_layer_ordinal, uint64_t cache_layer_stride, uint64_t cache_block_stride)
{
	dim3 grid(SPARK_QWEN36_MODEL_ATTN_QUERY_HEAD_COUNT,row_count,1u);
	SparkQwen36AttnDecodeKernel<<<grid,SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION,0,stream>>>(q_fused_bf16,kv_cache_bf16,table->physical_block_indices,table->lane_physical_block_counts,row_lane_indices,context_lengths,head_out_bf16,row_count,table->lane_stride,attn_layer_ordinal,cache_layer_stride,cache_block_stride);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkQwen36LaunchEmbeddingGather(cudaStream_t stream, const uint32_t *token_ids, const void *embedding_bf16, void *hidden_bf16, uint32_t row_count)
{
	uint64_t elements = (uint64_t)row_count * SPARK_QWEN36_MODEL_HIDDEN_DIMENSION;
	SparkQwen36EmbeddingGatherKernel<<<(uint32_t)((elements + SPARK_LM_CTA_THREADS - 1u) / SPARK_LM_CTA_THREADS),SPARK_LM_CTA_THREADS,0,stream>>>(token_ids,embedding_bf16,hidden_bf16,row_count);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkQwen36LaunchResidualAdd(cudaStream_t stream, void *hidden_bf16, const void *delta_bf16, uint32_t row_count, uint32_t dimension)
{
	uint64_t elements = (uint64_t)row_count * dimension;
	SparkQwen36ResidualAddKernel<<<(uint32_t)((elements + SPARK_LM_CTA_THREADS - 1u) / SPARK_LM_CTA_THREADS),SPARK_LM_CTA_THREADS,0,stream>>>(hidden_bf16,delta_bf16,row_count,dimension);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkQwen36LaunchSwiGlu(cudaStream_t stream, const void *gate_bf16, void *up_bf16, uint32_t row_count, uint32_t dimension)
{
	uint64_t elements = (uint64_t)row_count * dimension;
	SparkQwen36SwiGluKernel<<<(uint32_t)((elements + SPARK_LM_CTA_THREADS - 1u) / SPARK_LM_CTA_THREADS),SPARK_LM_CTA_THREADS,0,stream>>>(gate_bf16,up_bf16,row_count,dimension);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkQwen36LaunchHeadArgmax(cudaStream_t stream, const void *hidden_bf16, const void *head_weight_bf16, const uint32_t *token_ids, uint32_t *output_token_ids, uint32_t row_count, uint32_t candidate_count)
{
	SparkLmHeadArgmaxKernel<<<row_count,SPARK_LM_CTA_THREADS,0,stream>>>(hidden_bf16,head_weight_bf16,token_ids,output_token_ids,row_count,SPARK_QWEN36_MODEL_HIDDEN_DIMENSION,candidate_count);
	return(cudaGetLastError());
}
