#include "sparkpipe/spark_lm_kernels.cuh"
#include "sparkpipe/spark_dsv4_resident_decode_stage_firmware.h"

#include <math.h>

/*
 * DeepSeek V4 device kernels. The variant model header arrives via the
 * build's -include ahead of everything here; nothing below names a
 * variant. Shared machinery (linear over bf16/fp8/mxfp4, rms norm, head
 * argmax, embedding gather, reductions) comes from spark_lm_kernels.cuh;
 * this file holds only what DeepSeek V4 adds: adjacent-pair rope and its
 * inverse, the unweighted query-head rms, the fp8/fp4 quantize-dequantize
 * cache sims with power-of-two scales, Hadamard rotation, sink-in-
 * denominator sparse attention over gathered cache slots, the gated
 * softmax compressor in both prefill and decode-state forms, indexer
 * scoring and iterative top-k, the two router gates, the swiglu clamp,
 * and the full mHC split with inference Sinkhorn. Every kernel body stays
 * within fifty lines; the sparse-attention two-pass and the compressor
 * pooling decompose through device helpers.
 */

static __device__ __forceinline__ float SparkDsv4EncodeE4m3(float value)
{
	float magnitude = fabsf(value),sign = value < 0.0f ? -1.0f : 1.0f,scaled,snapped;
	int32_t exponent;
	if ( magnitude < 0.0009765625f )
		return(sign * rintf(magnitude * 512.0f) / 512.0f);
	if ( magnitude > 448.0f )
		return(sign * 448.0f);
	frexpf(magnitude,&exponent);
	scaled = ldexpf(magnitude,4 - exponent);
	snapped = rintf(scaled);
	if ( snapped >= 16.0f )
	{
		snapped = 8.0f;
		exponent += 1;
	}
	return(sign * ldexpf(snapped,exponent - 4));
}

// e2m1 snap with RN-even at every midpoint: the mantissa-zero neighbour
// wins ties, matching the reference cast exactly.
static __device__ __forceinline__ float SparkDsv4EncodeE2m1(float value)
{
	const float points[8] = {0.0f,0.5f,1.0f,1.5f,2.0f,3.0f,4.0f,6.0f};
	const float ties[7] = {0.0f,1.0f,1.0f,2.0f,2.0f,4.0f,4.0f};
	float magnitude = fabsf(value),sign = value < 0.0f ? -1.0f : 1.0f,midpoint;
	uint32_t index;
	if ( magnitude >= 6.0f )
		return(sign * 6.0f);
	for (index = 0; index < 7u; index++)
	{
		midpoint = (points[index] + points[index + 1u]) * 0.5f;
		if ( magnitude < midpoint )
			return(sign * points[index]);
		if ( magnitude == midpoint )
			return(sign * ties[index]);
	}
	return(sign * 6.0f);
}

static __device__ __forceinline__ float SparkDsv4Pow2CeilScale(float amax, float format_max)
{
	return(exp2f(ceilf(log2f(amax / format_max))));
}

/*
 * In-place block quantize-dequantize sim over the trailing width of each
 * row: per block amax (floored 1e-4), power-of-two scale, snap, rescale -
 * fp8 with 448 or fp4 with 6 by format_max. One warp per (row, block).
 */
static __global__ void SparkDsv4QuantSimKernel(void *data_bf16, uint32_t row_count, uint32_t row_stride, uint32_t width, uint32_t block, float format_max, uint32_t fp4)
{
	uint32_t row = blockIdx.x,group = blockIdx.y,lane = threadIdx.x;
	uint32_t base = group * block,limit = base + block < width ? base + block : width,element;
	uint64_t offset = (uint64_t)row * row_stride;
	float value,amax = 1e-4f,scale;
	if ( row >= row_count || base >= width )
		return;
	for (element = base + lane; element < limit; element += SPARK_LM_WARP_LANES)
	{
		value = fabsf(SparkLmBf16ToFloat(data_bf16,offset + element));
		if ( value > amax )
			amax = value;
	}
	for (element = SPARK_LM_WARP_LANES / 2u; element != 0u; element >>= 1u)
	{
		value = __shfl_down_sync(0xffffffffu,amax,element);
		if ( value > amax )
			amax = value;
	}
	amax = __shfl_sync(0xffffffffu,amax,0);
	scale = SparkDsv4Pow2CeilScale(amax,format_max);
	for (element = base + lane; element < limit; element += SPARK_LM_WARP_LANES)
	{
		value = SparkLmBf16ToFloat(data_bf16,offset + element) / scale;
		if ( value > format_max )
			value = format_max;
		if ( value < -format_max )
			value = -format_max;
		value = (fp4 != 0u ? SparkDsv4EncodeE2m1(value) : SparkDsv4EncodeE4m3(value)) * scale;
		SparkLmFloatToBf16(data_bf16,offset + element,value);
	}
}

// Adjacent-pair rotation on the LAST rope_dim entries of every head; the
// inverse conjugates - the attention output's de-rotation. One block per
// (row, head), threads over pairs; freqs are the layer's YaRN table.
static __global__ void SparkDsv4RopeKernel(void *data_bf16, const float *freqs_f32, const uint64_t *row_positions, uint32_t row_count, uint32_t head_count, uint32_t head_dim, uint32_t rope_dim, uint32_t inverse)
{
	uint32_t row = blockIdx.x,head = blockIdx.y,pair = threadIdx.x;
	uint64_t base;
	float angle,cosine,sine,real,imaginary;
	if ( row >= row_count || head >= head_count || pair >= rope_dim / 2u )
		return;
	base = (((uint64_t)row * head_count) + head) * head_dim + (head_dim - rope_dim) + 2u * pair;
	angle = (float)row_positions[row] * freqs_f32[pair];
	cosine = cosf(angle);
	sine = inverse != 0u ? -sinf(angle) : sinf(angle);
	real = SparkLmBf16ToFloat(data_bf16,base);
	imaginary = SparkLmBf16ToFloat(data_bf16,base + 1u);
	SparkLmFloatToBf16(data_bf16,base,real * cosine - imaginary * sine);
	SparkLmFloatToBf16(data_bf16,base + 1u,real * sine + imaginary * cosine);
}

// The unweighted per-head query rms the reference applies before rope.
static __global__ void SparkDsv4QueryHeadRmsKernel(void *data_bf16, uint32_t row_count, uint32_t head_count, uint32_t head_dim, float epsilon)
{
	__shared__ float reduce_scratch[SPARK_LM_CTA_WARPS];
	uint32_t row = blockIdx.x,head = blockIdx.y,element;
	uint64_t base = (((uint64_t)row * head_count) + head) * head_dim;
	float value,total = 0.0f,inverse;
	if ( row >= row_count || head >= head_count )
		return;
	for (element = threadIdx.x; element < head_dim; element += blockDim.x)
	{
		value = SparkLmBf16ToFloat(data_bf16,base + element);
		total += value * value;
	}
	total = SparkLmBlockReduceSum(total,reduce_scratch);
	inverse = rsqrtf(total / (float)head_dim + epsilon);
	for (element = threadIdx.x; element < head_dim; element += blockDim.x)
		SparkLmFloatToBf16(data_bf16,base + element,SparkLmBf16ToFloat(data_bf16,base + element) * inverse);
}

// In-place Hadamard rotation scaled n^-0.5 on power-of-two vectors; one
// block per vector, the whole vector staged in shared memory.
static __global__ void SparkDsv4HadamardKernel(void *data_bf16, uint32_t vector_count, uint32_t width)
{
	extern __shared__ float hadamard_shared[];
	uint32_t vector = blockIdx.x,element,half,partner;
	uint64_t base = (uint64_t)vector * width;
	float scale = rsqrtf((float)width),a,b;
	if ( vector >= vector_count )
		return;
	for (element = threadIdx.x; element < width; element += blockDim.x)
		hadamard_shared[element] = SparkLmBf16ToFloat(data_bf16,base + element);
	__syncthreads();
	for (half = 1; half < width; half <<= 1u)
	{
		for (element = threadIdx.x; element < width / 2u; element += blockDim.x)
		{
			partner = ((element / half) * half * 2u) + (element % half);
			a = hadamard_shared[partner];
			b = hadamard_shared[partner + half];
			hadamard_shared[partner] = a + b;
			hadamard_shared[partner + half] = a - b;
		}
		__syncthreads();
	}
	for (element = threadIdx.x; element < width; element += blockDim.x)
		SparkLmFloatToBf16(data_bf16,base + element,hadamard_shared[element] * scale);
}

static __device__ __forceinline__ float SparkDsv4SlotDot(const float *q_shared, const void *kv_bf16, uint64_t slot_base, uint32_t head_dim)
{
	uint32_t element;
	float accumulator = 0.0f;
	for (element = 0; element < head_dim; element++)
		accumulator += q_shared[element] * SparkLmBf16ToFloat(kv_bf16,slot_base + element);
	return(accumulator);
}

/*
 * Sparse attention, one block per (row, head): pass one computes every
 * gathered slot's logit into shared memory (thread-per-slot stripes, -1
 * indices at -inf) and the block max; pass two accumulates the output
 * dims stripe-wise with the sink joining the denominator only. topk is
 * bounded by the launcher at the shared-memory logit capacity.
 */
static __global__ void SparkDsv4SparseAttnKernel(const void *q_bf16, const void *kv_cache_bf16, uint64_t lane_stride_elements, const uint32_t *row_lane_indices, const int32_t *topk_idxs, uint32_t topk, const float *sink_f32, float scale, void *out_bf16, uint32_t row_count, uint32_t head_count, uint32_t head_dim)
{
	extern __shared__ float attn_shared[];
	float *logits = attn_shared,*q_shared = attn_shared + topk;
	__shared__ float reduce_scratch[SPARK_LM_CTA_WARPS];
	uint32_t row = blockIdx.x,head = blockIdx.y,slot,element;
	uint64_t q_base = (((uint64_t)row * head_count) + head) * head_dim,kv_base;
	int32_t index;
	float maximum = -3.0e38f,value,denominator,weight;
	if ( row >= row_count || head >= head_count )
		return;
	kv_base = (uint64_t)row_lane_indices[row] * lane_stride_elements;
	for (element = threadIdx.x; element < head_dim; element += blockDim.x)
		q_shared[element] = SparkLmBf16ToFloat(q_bf16,q_base + element);
	__syncthreads();
	for (slot = threadIdx.x; slot < topk; slot += blockDim.x)
	{
		index = topk_idxs[((uint64_t)row * topk) + slot];
		logits[slot] = index < 0 ? -3.0e38f : SparkDsv4SlotDot(q_shared,kv_cache_bf16,kv_base + ((uint64_t)(uint32_t)index * head_dim),head_dim) * scale;
		if ( logits[slot] > maximum )
			maximum = logits[slot];
	}
	for (element = SPARK_LM_WARP_LANES / 2u; element != 0u; element >>= 1u)
	{
		value = __shfl_down_sync(0xffffffffu,maximum,element);
		if ( value > maximum )
			maximum = value;
	}
	if ( threadIdx.x % SPARK_LM_WARP_LANES == 0u )
		reduce_scratch[threadIdx.x / SPARK_LM_WARP_LANES] = maximum;
	__syncthreads();
	maximum = reduce_scratch[0];
	for (element = 1; element < SPARK_LM_CTA_WARPS; element++)
		if ( reduce_scratch[element] > maximum )
			maximum = reduce_scratch[element];
	__syncthreads();
	denominator = __expf(sink_f32[head] - maximum);
	for (slot = 0; slot < topk; slot++)
		denominator += logits[slot] <= -3.0e38f ? 0.0f : __expf(logits[slot] - maximum);
	for (element = threadIdx.x; element < head_dim; element += blockDim.x)
	{
		value = 0.0f;
		for (slot = 0; slot < topk; slot++)
		{
			index = topk_idxs[((uint64_t)row * topk) + slot];
			if ( index < 0 )
				continue;
			weight = __expf(logits[slot] - maximum);
			value += weight * SparkLmBf16ToFloat(kv_cache_bf16,kv_base + ((uint64_t)(uint32_t)index * head_dim) + element);
		}
		SparkLmFloatToBf16(out_bf16,q_base + element,value / denominator);
	}
}

// Softmax pooling of one output channel over the pool slots: -inf scores
// drop out; shared by decode and prefill compressor forms.
static __device__ __forceinline__ float SparkDsv4PoolChannel(const float *kv, const float *score, uint32_t slots, uint32_t stride, uint32_t channel)
{
	uint32_t slot;
	float maximum = -3.0e38f,total = 0.0f,value = 0.0f,weight;
	for (slot = 0; slot < slots; slot++)
		if ( score[slot * stride + channel] > maximum )
			maximum = score[slot * stride + channel];
	for (slot = 0; slot < slots; slot++)
	{
		if ( score[slot * stride + channel] <= -3.0e38f )
			continue;
		weight = __expf(score[slot * stride + channel] - maximum);
		total += weight;
		value += weight * kv[slot * stride + channel];
	}
	return(value / total);
}

// The overlap gather pool: 2*ratio slots where slot i < ratio reads the
// previous group's FIRST channel half and slot i >= ratio the current
// group's SECOND half - the concatenation the reference builds before its
// single softmax pool.
static __device__ __forceinline__ float SparkDsv4PoolOverlapChannel(const float *kv_state, const float *score_state, uint32_t ratio, uint32_t channels, uint32_t width, uint32_t channel)
{
	uint32_t slot,source;
	float maximum = -3.0e38f,total = 0.0f,value = 0.0f,weight,score;
	for (slot = 0; slot < 2u * ratio; slot++)
	{
		source = slot < ratio ? slot * channels + channel : slot * channels + width + channel;
		if ( score_state[source] > maximum )
			maximum = score_state[source];
	}
	for (slot = 0; slot < 2u * ratio; slot++)
	{
		source = slot < ratio ? slot * channels + channel : slot * channels + width + channel;
		score = score_state[source];
		if ( score <= -3.0e38f )
			continue;
		weight = __expf(score - maximum);
		total += weight;
		value += weight * kv_state[source];
	}
	return(value / total);
}

/*
 * Compressor decode step for one token per row: the token's coff*d kv and
 * gate channels (ape already added by the caller's kernel) land in the
 * lane's state slot position%ratio; on a ratio boundary the pooled d-wide
 * entry emits per the overlap rule - previous group through the first
 * channel half - and the overlap state shifts down. emitted[row] reports
 * the boundary. State layout per lane: [coff*ratio slots][coff*d ch] f32.
 */
static __global__ void SparkDsv4CompressStepKernel(const float *kv_f32, const float *score_f32, float *kv_state_f32, float *score_state_f32, uint64_t state_lane_stride, const uint32_t *row_lane_indices, const uint64_t *row_positions, uint32_t row_count, uint32_t ratio, uint32_t overlap, uint32_t width, void *emit_bf16, uint32_t *emitted)
{
	uint32_t row = blockIdx.x,coff = overlap != 0u ? 2u : 1u,channels = coff * width,channel;
	uint32_t slot = (uint32_t)(row_positions[row] % ratio),boundary = (row_positions[row] + 1u) % ratio == 0u ? 1u : 0u;
	uint64_t state_base = (uint64_t)row_lane_indices[row] * state_lane_stride;
	float *kv_state = kv_state_f32 + state_base,*score_state = score_state_f32 + state_base;
	float pooled;
	if ( row >= row_count )
		return;
	for (channel = threadIdx.x; channel < channels; channel += blockDim.x)
	{
		kv_state[((overlap != 0u ? ratio : 0u) + slot) * channels + channel] = kv_f32[(uint64_t)row * channels + channel];
		score_state[((overlap != 0u ? ratio : 0u) + slot) * channels + channel] = score_f32[(uint64_t)row * channels + channel];
	}
	__syncthreads();
	if ( threadIdx.x == 0u )
		emitted[row] = boundary;
	if ( boundary == 0u )
		return;
	for (channel = threadIdx.x; channel < width; channel += blockDim.x)
	{
		if ( overlap != 0u )
			pooled = SparkDsv4PoolOverlapChannel(kv_state,score_state,ratio,channels,width,channel);
		else
			pooled = SparkDsv4PoolChannel(kv_state,score_state,ratio,channels,channel);
		SparkLmFloatToBf16(emit_bf16,(uint64_t)row * width + channel,pooled);
	}
	__syncthreads();
	if ( overlap != 0u )
		for (channel = threadIdx.x; channel < ratio * channels; channel += blockDim.x)
		{
			kv_state[channel] = kv_state[ratio * channels + channel];
			score_state[channel] = score_state[ratio * channels + channel];
		}
}

// The gate scores: linear against the router weight in fp32 with
// sqrtsoftplus applied - one warp per expert, activations shared.
static __global__ void SparkDsv4GateScoresKernel(const void *weight_bf16, const void *input_bf16, float *scores_f32, uint32_t row_count, uint32_t input_dimension, uint32_t expert_count)
{
	extern __shared__ float gate_shared[];
	uint32_t row = blockIdx.x,expert_base = blockIdx.y * SPARK_LM_CTA_WARPS;
	uint32_t warp = threadIdx.x / SPARK_LM_WARP_LANES,lane = threadIdx.x % SPARK_LM_WARP_LANES,expert,element;
	float accumulator;
	if ( row >= row_count )
		return;
	for (element = threadIdx.x; element < input_dimension; element += blockDim.x)
		gate_shared[element] = SparkLmBf16ToFloat(input_bf16,((uint64_t)row * input_dimension) + element);
	__syncthreads();
	expert = expert_base + warp;
	if ( expert >= expert_count )
		return;
	accumulator = SparkLmDotRowBf16(gate_shared,weight_bf16,expert,input_dimension,lane);
	accumulator = SparkLmWarpReduceSum(accumulator);
	if ( lane == 0u )
		scores_f32[((uint64_t)row * expert_count) + expert] = sqrtf(SparkLmSoftplus(accumulator));
}

/*
 * noaux_tc selection and the hash path in one kernel, one thread per row:
 * with tid2eid the indices come from the table row of the token id; else
 * the top-k runs on scores+bias. Weights gather ORIGINAL scores, sum-
 * normalize, and scale - the pinned routing arithmetic.
 */
static __global__ void SparkDsv4GateSelectKernel(const float *scores_f32, const float *bias_f32, const uint32_t *tid2eid_u32, const uint32_t *token_ids, uint32_t row_count, uint32_t expert_count, uint32_t topk, float route_scale, uint32_t *indices_u32, float *weights_f32)
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
		if ( tid2eid_u32 != 0 )
			best = tid2eid_u32[((uint64_t)token_ids[row] * topk) + rank];
		else
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
				shifted = scores[expert] + (bias_f32 != 0 ? bias_f32[expert] : 0.0f);
				if ( shifted > best_score )
				{
					best_score = shifted;
					best = expert;
				}
			}
		}
		indices[rank] = best;
		total += scores[best];
	}
	for (rank = 0; rank < topk; rank++)
		weights_f32[((uint64_t)row * topk) + rank] = scores[indices[rank]] / total * route_scale;
}

// The swiglu clamp on gathered gate/up rows, routing weight folded in:
// up two-sided, gate max-only, silu(gate)*up in fp32.
static __global__ void SparkDsv4SwigluClampKernel(const void *gate_bf16, void *up_bf16, uint32_t row_count, uint32_t width, float limit, const float *row_weights_f32, const uint32_t *weight_map)
{
	uint32_t row = blockIdx.x,element;
	uint64_t offset = (uint64_t)row * width;
	float gate,up,weight;
	if ( row >= row_count )
		return;
	weight = row_weights_f32 != 0 ? row_weights_f32[weight_map != 0 ? weight_map[row] : row] : 1.0f;
	for (element = threadIdx.x; element < width; element += blockDim.x)
	{
		gate = SparkLmBf16ToFloat(gate_bf16,offset + element);
		up = SparkLmBf16ToFloat(up_bf16,offset + element);
		if ( limit > 0.0f )
		{
			if ( up > limit )
				up = limit;
			if ( up < -limit )
				up = -limit;
			if ( gate > limit )
				gate = limit;
		}
		SparkLmFloatToBf16(up_bf16,offset + element,SparkLmSwish(gate) * up * weight);
	}
}

static __global__ void SparkDsv4AccumAddKernel(void *destination_bf16, const void *source_bf16, uint32_t row_count, uint32_t width)
{
	uint32_t row = blockIdx.x,element;
	uint64_t offset = (uint64_t)row * width;
	if ( row >= row_count )
		return;
	for (element = threadIdx.x; element < width; element += blockDim.x)
		SparkLmFloatToBf16(destination_bf16,offset + element,SparkLmBf16ToFloat(destination_bf16,offset + element) + SparkLmBf16ToFloat(source_bf16,offset + element));
}

// The indexer score: relu(q_h . kv) per head times the projected head
// weight, summed over heads - one warp per slot, lanes over dims.
static __global__ void SparkDsv4IndexerScoreKernel(const void *q_bf16, const void *kv_cache_bf16, uint64_t lane_stride_elements, const uint32_t *row_lane_indices, const uint32_t *slot_counts, const float *head_weights_f32, float *scores_f32, uint32_t row_count, uint32_t max_slots, uint32_t head_count, uint32_t head_dim)
{
	uint32_t row = blockIdx.x,slot = blockIdx.y * SPARK_LM_CTA_WARPS + threadIdx.x / SPARK_LM_WARP_LANES;
	uint32_t lane = threadIdx.x % SPARK_LM_WARP_LANES,head,element;
	uint64_t q_base = (uint64_t)row * head_count * head_dim,kv_base;
	float accumulator,total = 0.0f;
	if ( row >= row_count || slot >= max_slots )
		return;
	if ( slot >= slot_counts[row] )
	{
		if ( lane == 0u )
			scores_f32[((uint64_t)row * max_slots) + slot] = -3.0e38f;
		return;
	}
	kv_base = ((uint64_t)row_lane_indices[row] * lane_stride_elements) + ((uint64_t)slot * head_dim);
	for (head = 0; head < head_count; head++)
	{
		accumulator = 0.0f;
		for (element = lane; element < head_dim; element += SPARK_LM_WARP_LANES)
			accumulator += SparkLmBf16ToFloat(q_bf16,q_base + ((uint64_t)head * head_dim) + element) * SparkLmBf16ToFloat(kv_cache_bf16,kv_base + element);
		accumulator = SparkLmWarpReduceSum(accumulator);
		accumulator = __shfl_sync(0xffffffffu,accumulator,0);
		if ( accumulator > 0.0f )
			total += accumulator * head_weights_f32[((uint64_t)row * head_count) + head];
	}
	if ( lane == 0u )
		scores_f32[((uint64_t)row * max_slots) + slot] = total;
}

/*
 * Iterative top-k, one block per row: k passes of a block-parallel masked
 * argmax over the slot scores, ties to the lower index, -1 padding past
 * the valid count, +offset on emission. Correct and simple; the fast
 * selection is scheduled with the wmma pass.
 */
static __global__ void SparkDsv4TopKKernel(float *scores_f32, const uint32_t *slot_counts, uint32_t max_slots, uint32_t topk, int32_t offset, int32_t *indices_out, uint64_t out_row_stride, uint32_t row_count)
{
	__shared__ float best_score[SPARK_LM_CTA_WARPS];
	__shared__ int32_t best_index[SPARK_LM_CTA_WARPS];
	uint32_t row = blockIdx.x,rank,slot,warp = threadIdx.x / SPARK_LM_WARP_LANES,lane = threadIdx.x % SPARK_LM_WARP_LANES,candidate;
	float *scores = scores_f32 + ((uint64_t)row * max_slots),value,lane_score;
	int32_t lane_index,winner;
	if ( row >= row_count )
		return;
	for (rank = 0; rank < topk; rank++)
	{
		lane_score = -3.0e38f;
		lane_index = -1;
		for (slot = threadIdx.x; slot < slot_counts[row]; slot += blockDim.x)
			if ( scores[slot] > lane_score )
			{
				lane_score = scores[slot];
				lane_index = (int32_t)slot;
			}
		for (candidate = SPARK_LM_WARP_LANES / 2u; candidate != 0u; candidate >>= 1u)
		{
			value = __shfl_down_sync(0xffffffffu,lane_score,candidate);
			winner = __shfl_down_sync(0xffffffffu,lane_index,candidate);
			if ( value > lane_score || (value == lane_score && winner >= 0 && (lane_index < 0 || winner < lane_index)) )
			{
				lane_score = value;
				lane_index = winner;
			}
		}
		if ( lane == 0u )
		{
			best_score[warp] = lane_score;
			best_index[warp] = lane_index;
		}
		__syncthreads();
		if ( threadIdx.x == 0u )
		{
			winner = 0;
			for (candidate = 1; candidate < SPARK_LM_CTA_WARPS; candidate++)
				if ( best_score[candidate] > best_score[winner] || (best_score[candidate] == best_score[winner] && best_index[candidate] >= 0 && (best_index[winner] < 0 || best_index[candidate] < best_index[winner])) )
					winner = (int32_t)candidate;
			indices_out[((uint64_t)row * out_row_stride) + rank] = best_index[winner] < 0 ? -1 : best_index[winner] + offset;
			if ( best_index[winner] >= 0 )
				scores[best_index[winner]] = -3.0e38f;
		}
		__syncthreads();
	}
}

/*
 * mHC mixes for one row: 24 (or hc for the head) fp32 dot products of the
 * fn rows against the flattened streams, scaled by the rsqrt of the
 * flattened mean square - the norm applied to the mix, exactly the
 * reference order. One block per row, one warp per mix row.
 */
static __global__ void SparkDsv4HcMixKernel(const void *streams_bf16, const float *fn_f32, float *mixes_f32, uint32_t row_count, uint32_t flat_dimension, uint32_t mix_rows, float epsilon)
{
	extern __shared__ float hc_shared[];
	__shared__ float reduce_scratch[SPARK_LM_CTA_WARPS];
	uint32_t row = blockIdx.x,warp = threadIdx.x / SPARK_LM_WARP_LANES,lane = threadIdx.x % SPARK_LM_WARP_LANES,mix,element;
	float value,total = 0.0f,inverse,accumulator;
	if ( row >= row_count )
		return;
	for (element = threadIdx.x; element < flat_dimension; element += blockDim.x)
	{
		value = SparkLmBf16ToFloat(streams_bf16,((uint64_t)row * flat_dimension) + element);
		hc_shared[element] = value;
		total += value * value;
	}
	total = SparkLmBlockReduceSum(total,reduce_scratch);
	inverse = rsqrtf(total / (float)flat_dimension + epsilon);
	for (mix = warp; mix < mix_rows; mix += SPARK_LM_CTA_WARPS)
	{
		accumulator = 0.0f;
		for (element = lane; element < flat_dimension; element += SPARK_LM_WARP_LANES)
			accumulator += hc_shared[element] * fn_f32[((uint64_t)mix * flat_dimension) + element];
		accumulator = SparkLmWarpReduceSum(accumulator);
		if ( lane == 0u )
			mixes_f32[((uint64_t)row * mix_rows) + mix] = accumulator * inverse;
	}
}

/*
 * The split with inference Sinkhorn, one thread per row: sigmoid pre
 * (+eps), doubled sigmoid post, comb row-softmax +eps then the iteration
 * count of alternating row and column normalizations with +eps inside
 * every division - the first row pass is the softmax itself, matching the
 * reference kernel step for step at hc = 4.
 */
static __global__ void SparkDsv4HcSplitSinkhornKernel(const float *mixes_f32, const float *scale3_f32, const float *base_f32, uint32_t row_count, uint32_t hc, uint32_t iterations, float epsilon, float *pre_f32, float *post_f32, float *comb_f32)
{
	uint32_t row = blockIdx.x * blockDim.x + threadIdx.x,i,j,iteration,mix_rows = (2u + hc) * hc;
	const float *mixes;
	float comb[16],maximum,total;
	if ( row >= row_count || hc > 4u )
		return;
	mixes = mixes_f32 + ((uint64_t)row * mix_rows);
	for (i = 0; i < hc; i++)
	{
		pre_f32[((uint64_t)row * hc) + i] = SparkLmSigmoid(mixes[i] * scale3_f32[0] + base_f32[i]) + epsilon;
		post_f32[((uint64_t)row * hc) + i] = 2.0f * SparkLmSigmoid(mixes[hc + i] * scale3_f32[1] + base_f32[hc + i]);
	}
	for (i = 0; i < hc; i++)
	{
		maximum = -3.0e38f;
		for (j = 0; j < hc; j++)
		{
			comb[i * hc + j] = mixes[2u * hc + i * hc + j] * scale3_f32[2] + base_f32[2u * hc + i * hc + j];
			maximum = comb[i * hc + j] > maximum ? comb[i * hc + j] : maximum;
		}
		total = 0.0f;
		for (j = 0; j < hc; j++)
			total += (comb[i * hc + j] = __expf(comb[i * hc + j] - maximum));
		for (j = 0; j < hc; j++)
			comb[i * hc + j] = comb[i * hc + j] / total + epsilon;
	}
	for (iteration = 0; iteration < iterations; iteration++)
	{
		if ( iteration != 0u )
			for (i = 0; i < hc; i++)
			{
				total = 0.0f;
				for (j = 0; j < hc; j++)
					total += comb[i * hc + j];
				for (j = 0; j < hc; j++)
					comb[i * hc + j] /= total + epsilon;
			}
		for (j = 0; j < hc; j++)
		{
			total = 0.0f;
			for (i = 0; i < hc; i++)
				total += comb[i * hc + j];
			for (i = 0; i < hc; i++)
				comb[i * hc + j] /= total + epsilon;
		}
	}
	for (i = 0; i < hc * hc; i++)
		comb_f32[((uint64_t)row * hc * hc) + i] = comb[i];
}

// Stream reduction by pre, expansion by post + transposed comb, and the
// sigmoid head reduction - three small element kernels.
static __global__ void SparkDsv4HcPreReduceKernel(const void *streams_bf16, const float *pre_f32, void *reduced_bf16, uint32_t row_count, uint32_t hc, uint32_t dimension)
{
	uint32_t row = blockIdx.x,element,stream;
	float value;
	if ( row >= row_count )
		return;
	for (element = threadIdx.x; element < dimension; element += blockDim.x)
	{
		value = 0.0f;
		for (stream = 0; stream < hc; stream++)
			value += pre_f32[((uint64_t)row * hc) + stream] * SparkLmBf16ToFloat(streams_bf16,(((uint64_t)row * hc) + stream) * dimension + element);
		SparkLmFloatToBf16(reduced_bf16,((uint64_t)row * dimension) + element,value);
	}
}

static __global__ void SparkDsv4HcPostKernel(const void *out_bf16, const void *residual_bf16, const float *post_f32, const float *comb_f32, void *streams_bf16, uint32_t row_count, uint32_t hc, uint32_t dimension)
{
	uint32_t row = blockIdx.x,element,stream,source;
	float value;
	if ( row >= row_count )
		return;
	for (element = threadIdx.x; element < hc * dimension; element += blockDim.x)
	{
		stream = element / dimension;
		value = post_f32[((uint64_t)row * hc) + stream] * SparkLmBf16ToFloat(out_bf16,((uint64_t)row * dimension) + (element % dimension));
		for (source = 0; source < hc; source++)
			value += comb_f32[((uint64_t)row * hc * hc) + (source * hc) + stream] * SparkLmBf16ToFloat(residual_bf16,(((uint64_t)row * hc) + source) * dimension + (element % dimension));
		SparkLmFloatToBf16(streams_bf16,((uint64_t)row * hc * dimension) + element,value);
	}
}

static __global__ void SparkDsv4HcHeadReduceKernel(const void *streams_bf16, const float *mixes_f32, float scale, const float *base_f32, float epsilon, void *reduced_bf16, uint32_t row_count, uint32_t hc, uint32_t dimension)
{
	uint32_t row = blockIdx.x,element,stream;
	float value,pre;
	if ( row >= row_count )
		return;
	for (element = threadIdx.x; element < dimension; element += blockDim.x)
	{
		value = 0.0f;
		for (stream = 0; stream < hc; stream++)
		{
			pre = SparkLmSigmoid(mixes_f32[((uint64_t)row * hc) + stream] * scale + base_f32[stream]) + epsilon;
			value += pre * SparkLmBf16ToFloat(streams_bf16,(((uint64_t)row * hc) + stream) * dimension + element);
		}
		SparkLmFloatToBf16(reduced_bf16,((uint64_t)row * dimension) + element,value);
	}
}

/*
 * Grouped linear with strided input rows: the o composition reads group g
 * of each row's heads*head_dim output (row stride the full width, input
 * slice group_dim wide at g*group_dim) against wo_a's group block. Same
 * dot helpers as the flat linear; the stride is the only difference.
 */
static __global__ void SparkDsv4StridedLinearKernel(uint32_t weight_format, const void *weight_payload, const uint8_t *weight_scale_e8m0, const void *input_bf16, uint64_t input_row_stride, uint32_t input_offset, void *output_bf16, uint64_t output_row_stride, uint32_t output_offset, uint32_t row_count, uint32_t input_dimension, uint32_t output_dimension)
{
	extern __shared__ float strided_shared[];
	uint32_t row = blockIdx.x,neuron_base = blockIdx.y * SPARK_LM_CTA_WARPS;
	uint32_t warp = threadIdx.x / SPARK_LM_WARP_LANES,lane = threadIdx.x % SPARK_LM_WARP_LANES,neuron,element;
	float accumulator;
	if ( row >= row_count )
		return;
	for (element = threadIdx.x; element < input_dimension; element += blockDim.x)
		strided_shared[element] = SparkLmBf16ToFloat(input_bf16,((uint64_t)row * input_row_stride) + input_offset + element);
	__syncthreads();
	neuron = neuron_base + warp;
	if ( neuron >= output_dimension )
		return;
	if ( weight_format == SPARK_LM_WEIGHT_FORMAT_BF16 )
		accumulator = SparkLmDotRowBf16(strided_shared,weight_payload,neuron,input_dimension,lane);
	else if ( weight_format == SPARK_LM_WEIGHT_FORMAT_FP8_E4M3 )
		accumulator = SparkLmDotRowFp8<128u>(strided_shared,weight_payload,weight_scale_e8m0,neuron,input_dimension,lane);
	else
		accumulator = SparkLmDotRowMxfp4<32u>(strided_shared,weight_payload,weight_scale_e8m0,neuron,input_dimension,lane);
	accumulator = SparkLmWarpReduceSum(accumulator);
	if ( lane == 0u )
		SparkLmFloatToBf16(output_bf16,((uint64_t)row * output_row_stride) + output_offset + neuron,accumulator);
}

// Score = wgate output plus the in-group ape row - the ape add the
// reference folds before pooling.
static __global__ void SparkDsv4ApeAddKernel(float *score_f32, const float *ape_f32, const uint64_t *row_positions, uint32_t row_count, uint32_t ratio, uint32_t channels)
{
	uint32_t row = blockIdx.x,channel;
	if ( row >= row_count )
		return;
	for (channel = threadIdx.x; channel < channels; channel += blockDim.x)
		score_f32[((uint64_t)row * channels) + channel] += ape_f32[((uint64_t)(row_positions[row] % ratio) * channels) + channel];
}

// bf16 rows widened to f32 (times a scalar) for the compressor's fp32
// pooling and the indexer's pre-scaled head weights.
static __global__ void SparkDsv4WidenKernel(const void *input_bf16, float *output_f32, uint32_t row_count, uint32_t width, float scale)
{
	uint32_t row = blockIdx.x,element;
	if ( row >= row_count )
		return;
	for (element = threadIdx.x; element < width; element += blockDim.x)
		output_f32[((uint64_t)row * width) + element] = SparkLmBf16ToFloat(input_bf16,((uint64_t)row * width) + element) * scale;
}

extern "C" cudaError_t SparkDsv4LaunchRmsNorm(cudaStream_t stream, const void *input_bf16, const void *gain_bf16, void *output_bf16, uint32_t row_count, uint32_t dimension, float epsilon)
{
	SparkLmRmsNormKernel<<<row_count,SPARK_LM_CTA_THREADS,0,stream>>>(input_bf16,gain_bf16,output_bf16,row_count,dimension,epsilon);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkDsv4LaunchLinear(cudaStream_t stream, const SparkDsv4LinearView *view, const void *input_bf16, void *output_bf16, uint32_t row_count)
{
	dim3 grid(row_count,(view->rows + SPARK_LM_CTA_WARPS - 1u) / SPARK_LM_CTA_WARPS);
	uint32_t shared_bytes = view->columns * (uint32_t)sizeof(float);
	if ( view->weight_format == SPARK_LM_WEIGHT_FORMAT_FP8_E4M3 )
		SparkLmLinearKernel<128u><<<grid,SPARK_LM_CTA_THREADS,shared_bytes,stream>>>(view->weight_format,view->payload,view->scale_e8m0,input_bf16,output_bf16,row_count,view->columns,view->rows);
	else
		SparkLmLinearKernel<32u><<<grid,SPARK_LM_CTA_THREADS,shared_bytes,stream>>>(view->weight_format,view->payload,view->scale_e8m0,input_bf16,output_bf16,row_count,view->columns,view->rows);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkDsv4LaunchStridedLinear(cudaStream_t stream, const SparkDsv4LinearView *view, const void *payload, const uint8_t *scale, const void *input_bf16, uint64_t input_row_stride, uint32_t input_offset, void *output_bf16, uint64_t output_row_stride, uint32_t output_offset, uint32_t row_count)
{
	dim3 grid(row_count,(view->rows + SPARK_LM_CTA_WARPS - 1u) / SPARK_LM_CTA_WARPS);
	uint32_t shared_bytes = view->columns * (uint32_t)sizeof(float);
	SparkDsv4StridedLinearKernel<<<grid,SPARK_LM_CTA_THREADS,shared_bytes,stream>>>(view->weight_format,payload,scale,input_bf16,input_row_stride,input_offset,output_bf16,output_row_stride,output_offset,row_count,view->columns,view->rows);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkDsv4LaunchEmbeddingGather(cudaStream_t stream, const uint32_t *token_ids, const void *embedding_bf16, void *hidden_bf16, uint32_t row_count, uint32_t hidden_dimension)
{
	SparkLmEmbeddingGatherKernel<<<row_count,SPARK_LM_CTA_THREADS,0,stream>>>(token_ids,embedding_bf16,hidden_bf16,row_count,hidden_dimension);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkDsv4LaunchHeadArgmax(cudaStream_t stream, const void *hidden_bf16, const void *head_weight_bf16, const uint32_t *token_ids, uint32_t *output_token_ids, uint32_t row_count, uint32_t candidate_count, uint32_t hidden_dimension)
{
	SparkLmHeadArgmaxKernel<<<row_count,SPARK_LM_CTA_THREADS,0,stream>>>(hidden_bf16,head_weight_bf16,token_ids,output_token_ids,row_count,hidden_dimension,candidate_count);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkDsv4LaunchQuantSim(cudaStream_t stream, void *data_bf16, uint32_t row_count, uint32_t row_stride, uint32_t width, uint32_t block, uint32_t fp4)
{
	dim3 grid(row_count,(width + block - 1u) / block);
	SparkDsv4QuantSimKernel<<<grid,SPARK_LM_WARP_LANES,0,stream>>>(data_bf16,row_count,row_stride,width,block,fp4 != 0u ? 6.0f : 448.0f,fp4);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkDsv4LaunchRope(cudaStream_t stream, void *data_bf16, const float *freqs_f32, const uint64_t *row_positions, uint32_t row_count, uint32_t head_count, uint32_t head_dim, uint32_t rope_dim, uint32_t inverse)
{
	dim3 grid(row_count,head_count);
	SparkDsv4RopeKernel<<<grid,rope_dim / 2u,0,stream>>>(data_bf16,freqs_f32,row_positions,row_count,head_count,head_dim,rope_dim,inverse);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkDsv4LaunchQueryHeadRms(cudaStream_t stream, void *data_bf16, uint32_t row_count, uint32_t head_count, uint32_t head_dim, float epsilon)
{
	dim3 grid(row_count,head_count);
	SparkDsv4QueryHeadRmsKernel<<<grid,SPARK_LM_CTA_THREADS,0,stream>>>(data_bf16,row_count,head_count,head_dim,epsilon);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkDsv4LaunchHadamard(cudaStream_t stream, void *data_bf16, uint32_t vector_count, uint32_t width)
{
	SparkDsv4HadamardKernel<<<vector_count,SPARK_LM_CTA_THREADS,width * (uint32_t)sizeof(float),stream>>>(data_bf16,vector_count,width);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkDsv4LaunchSparseAttn(cudaStream_t stream, const void *q_bf16, const void *kv_cache_bf16, uint64_t lane_stride_elements, const uint32_t *row_lane_indices, const int32_t *topk_idxs, uint32_t topk, const float *sink_f32, float scale, void *out_bf16, uint32_t row_count, uint32_t head_count, uint32_t head_dim)
{
	dim3 grid(row_count,head_count);
	uint32_t shared_bytes = (topk + head_dim) * (uint32_t)sizeof(float);
	SparkDsv4SparseAttnKernel<<<grid,SPARK_LM_CTA_THREADS,shared_bytes,stream>>>(q_bf16,kv_cache_bf16,lane_stride_elements,row_lane_indices,topk_idxs,topk,sink_f32,scale,out_bf16,row_count,head_count,head_dim);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkDsv4LaunchWiden(cudaStream_t stream, const void *input_bf16, float *output_f32, uint32_t row_count, uint32_t width, float scale)
{
	SparkDsv4WidenKernel<<<row_count,SPARK_LM_CTA_THREADS,0,stream>>>(input_bf16,output_f32,row_count,width,scale);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkDsv4LaunchApeAdd(cudaStream_t stream, float *score_f32, const float *ape_f32, const uint64_t *row_positions, uint32_t row_count, uint32_t ratio, uint32_t channels)
{
	SparkDsv4ApeAddKernel<<<row_count,SPARK_LM_CTA_THREADS,0,stream>>>(score_f32,ape_f32,row_positions,row_count,ratio,channels);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkDsv4LaunchCompressStep(cudaStream_t stream, const float *kv_f32, const float *score_f32, float *kv_state_f32, float *score_state_f32, uint64_t state_lane_stride, const uint32_t *row_lane_indices, const uint64_t *row_positions, uint32_t row_count, uint32_t ratio, uint32_t overlap, uint32_t width, void *emit_bf16, uint32_t *emitted)
{
	SparkDsv4CompressStepKernel<<<row_count,SPARK_LM_CTA_THREADS,0,stream>>>(kv_f32,score_f32,kv_state_f32,score_state_f32,state_lane_stride,row_lane_indices,row_positions,row_count,ratio,overlap,width,emit_bf16,emitted);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkDsv4LaunchGateScores(cudaStream_t stream, const SparkDsv4LinearView *gate, const void *input_bf16, float *scores_f32, uint32_t row_count)
{
	dim3 grid(row_count,(gate->rows + SPARK_LM_CTA_WARPS - 1u) / SPARK_LM_CTA_WARPS);
	SparkDsv4GateScoresKernel<<<grid,SPARK_LM_CTA_THREADS,gate->columns * (uint32_t)sizeof(float),stream>>>(gate->payload,input_bf16,scores_f32,row_count,gate->columns,gate->rows);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkDsv4LaunchGateSelect(cudaStream_t stream, const float *scores_f32, const float *bias_f32, const uint32_t *tid2eid_u32, const uint32_t *token_ids, uint32_t row_count, uint32_t expert_count, uint32_t topk, float route_scale, uint32_t *indices_u32, float *weights_f32)
{
	SparkDsv4GateSelectKernel<<<(row_count + 63u) / 64u,64u,0,stream>>>(scores_f32,bias_f32,tid2eid_u32,token_ids,row_count,expert_count,topk,route_scale,indices_u32,weights_f32);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkDsv4LaunchSwigluClamp(cudaStream_t stream, const void *gate_bf16, void *up_bf16, uint32_t row_count, uint32_t width, float limit, const float *row_weights_f32, const uint32_t *weight_map)
{
	SparkDsv4SwigluClampKernel<<<row_count,SPARK_LM_CTA_THREADS,0,stream>>>(gate_bf16,up_bf16,row_count,width,limit,row_weights_f32,weight_map);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkDsv4LaunchGatherLinear(cudaStream_t stream, const SparkDsv4LinearView *view, const void *input_bf16, const uint32_t *input_row_map, void *output_bf16, uint32_t slot_count)
{
	dim3 grid(slot_count,(view->rows + SPARK_LM_CTA_WARPS - 1u) / SPARK_LM_CTA_WARPS);
	uint32_t shared_bytes = view->columns * (uint32_t)sizeof(float);
	if ( view->weight_format == SPARK_LM_WEIGHT_FORMAT_FP8_E4M3 )
		SparkLmGatherLinearKernel<128u><<<grid,SPARK_LM_CTA_THREADS,shared_bytes,stream>>>(view->weight_format,view->payload,view->scale_e8m0,input_bf16,input_row_map,output_bf16,slot_count,view->columns,view->rows);
	else
		SparkLmGatherLinearKernel<32u><<<grid,SPARK_LM_CTA_THREADS,shared_bytes,stream>>>(view->weight_format,view->payload,view->scale_e8m0,input_bf16,input_row_map,output_bf16,slot_count,view->columns,view->rows);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkDsv4LaunchExpertTile(cudaStream_t stream, const SparkDsv4LinearView *view, const void *input_bf16, const uint32_t *input_row_map, void *output_bf16, uint32_t slot_count)
{
	dim3 grid((slot_count + SPARK_LM_TILE - 1u) / SPARK_LM_TILE,(view->rows + SPARK_LM_TILE - 1u) / SPARK_LM_TILE);
	if ( view->weight_format == SPARK_LM_WEIGHT_FORMAT_FP8_E4M3 )
		SparkLmExpertTileKernel<128u><<<grid,SPARK_LM_CTA_THREADS,0,stream>>>(view->weight_format,view->payload,view->scale_e8m0,input_bf16,input_row_map,output_bf16,slot_count,view->columns,view->rows);
	else
		SparkLmExpertTileKernel<32u><<<grid,SPARK_LM_CTA_THREADS,0,stream>>>(view->weight_format,view->payload,view->scale_e8m0,input_bf16,input_row_map,output_bf16,slot_count,view->columns,view->rows);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkDsv4LaunchScatterAdd(cudaStream_t stream, void *destination_bf16, const void *source_bf16, const uint32_t *row_map, uint32_t slot_count, uint32_t width)
{
	SparkLmScatterScaledAddKernel<<<slot_count,SPARK_LM_CTA_THREADS,0,stream>>>(destination_bf16,source_bf16,row_map,0,0,slot_count,width);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkDsv4LaunchAccumAdd(cudaStream_t stream, void *destination_bf16, const void *source_bf16, uint32_t row_count, uint32_t width)
{
	SparkDsv4AccumAddKernel<<<row_count,SPARK_LM_CTA_THREADS,0,stream>>>(destination_bf16,source_bf16,row_count,width);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkDsv4LaunchIndexerScore(cudaStream_t stream, const void *q_bf16, const void *kv_cache_bf16, uint64_t lane_stride_elements, const uint32_t *row_lane_indices, const uint32_t *slot_counts, const float *head_weights_f32, float *scores_f32, uint32_t row_count, uint32_t max_slots, uint32_t head_count, uint32_t head_dim)
{
	dim3 grid(row_count,(max_slots + SPARK_LM_CTA_WARPS - 1u) / SPARK_LM_CTA_WARPS);
	SparkDsv4IndexerScoreKernel<<<grid,SPARK_LM_CTA_THREADS,0,stream>>>(q_bf16,kv_cache_bf16,lane_stride_elements,row_lane_indices,slot_counts,head_weights_f32,scores_f32,row_count,max_slots,head_count,head_dim);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkDsv4LaunchTopK(cudaStream_t stream, float *scores_f32, const uint32_t *slot_counts, uint32_t max_slots, uint32_t topk, int32_t offset, int32_t *indices_out, uint64_t out_row_stride, uint32_t row_count)
{
	SparkDsv4TopKKernel<<<row_count,SPARK_LM_CTA_THREADS,0,stream>>>(scores_f32,slot_counts,max_slots,topk,offset,indices_out,out_row_stride,row_count);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkDsv4LaunchHcMix(cudaStream_t stream, const void *streams_bf16, const float *fn_f32, float *mixes_f32, uint32_t row_count, uint32_t flat_dimension, uint32_t mix_rows, float epsilon)
{
	SparkDsv4HcMixKernel<<<row_count,SPARK_LM_CTA_THREADS,flat_dimension * (uint32_t)sizeof(float),stream>>>(streams_bf16,fn_f32,mixes_f32,row_count,flat_dimension,mix_rows,epsilon);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkDsv4LaunchHcSplitSinkhorn(cudaStream_t stream, const float *mixes_f32, const float *scale3_f32, const float *base_f32, uint32_t row_count, uint32_t hc, uint32_t iterations, float epsilon, float *pre_f32, float *post_f32, float *comb_f32)
{
	SparkDsv4HcSplitSinkhornKernel<<<(row_count + 63u) / 64u,64u,0,stream>>>(mixes_f32,scale3_f32,base_f32,row_count,hc,iterations,epsilon,pre_f32,post_f32,comb_f32);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkDsv4LaunchHcPreReduce(cudaStream_t stream, const void *streams_bf16, const float *pre_f32, void *reduced_bf16, uint32_t row_count, uint32_t hc, uint32_t dimension)
{
	SparkDsv4HcPreReduceKernel<<<row_count,SPARK_LM_CTA_THREADS,0,stream>>>(streams_bf16,pre_f32,reduced_bf16,row_count,hc,dimension);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkDsv4LaunchHcPost(cudaStream_t stream, const void *out_bf16, const void *residual_bf16, const float *post_f32, const float *comb_f32, void *streams_bf16, uint32_t row_count, uint32_t hc, uint32_t dimension)
{
	SparkDsv4HcPostKernel<<<row_count,SPARK_LM_CTA_THREADS,0,stream>>>(out_bf16,residual_bf16,post_f32,comb_f32,streams_bf16,row_count,hc,dimension);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkDsv4LaunchHcHeadReduce(cudaStream_t stream, const void *streams_bf16, const float *mixes_f32, float scale, const float *base_f32, float epsilon, void *reduced_bf16, uint32_t row_count, uint32_t hc, uint32_t dimension)
{
	SparkDsv4HcHeadReduceKernel<<<row_count,SPARK_LM_CTA_THREADS,0,stream>>>(streams_bf16,mixes_f32,scale,base_f32,epsilon,reduced_bf16,row_count,hc,dimension);
	return(cudaGetLastError());
}
