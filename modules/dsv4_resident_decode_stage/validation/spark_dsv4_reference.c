#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * CPU reference oracles for DeepSeek V4, encoding the PINNED forms from the
 * repo's inference/model.py + kernel.py (fetched 2026-07-19, one shared
 * reference for Flash and Pro) in plain fp32 C:
 *
 * - fp8 e4m3 and fp4 e2m1 quantize-dequantize sims with power-of-two
 *   (e8m0) scales, amax floored at 1e-4, RN-even snapping
 * - YaRN rope with ADJACENT-pair complex rotation on the last rope dims,
 *   and the inverse (conjugate) rotation the attention output takes
 * - sparse attention: gathered rows, -1 masking, softmax with the per-head
 *   sink joining the DENOMINATOR only
 * - the compressor's gated softmax pooling, both the one-shot prefill form
 *   and the incremental decode-state form, with the ratio-4 OVERLAP
 *   doubling (previous group through the first channel half)
 * - the indexer score: relu(q . kv) per head times projected weights,
 *   summed over heads, after Hadamard rotation and fp4 sim
 * - sqrtsoftplus routing with the noaux_tc bias (selection-only) and the
 *   hash tid2eid path; the swiglu clamp (up two-sided, gate max-only)
 * - mHC: mixes = fn @ flat(x) scaled by rsqrt(mean(flat^2)+eps); pre =
 *   sigmoid(.)+eps, post = 2 sigmoid(.), comb = row-softmax +eps then 20
 *   Sinkhorn normalizations (+eps inside every division) - Sinkhorn RUNS
 *   AT INFERENCE; hc_post writes stream k as post[k] out + sum_j
 *   comb[j][k] residual[j]; the head reduction is the sigmoid pre only
 *
 * Tests are self-contained on synthetic data and assert closed forms,
 * invariances, and the state-carry contract: a sequence compressed in one
 * prefill pass must equal the same sequence fed token-by-token through the
 * decode state - exactly the boundary the resident driver crosses.
 */

static uint64_t SparkDsv4RefNext(uint64_t *state)
{
	uint64_t value = *state;
	value ^= value >> 12;
	value ^= value << 25;
	value ^= value >> 27;
	*state = value;
	return(value * 2685821657736338717ull);
}

static float SparkDsv4RefUniform(uint64_t *state)
{
	return((float)((SparkDsv4RefNext(state) >> 40) * (1.0 / 16777216.0)) * 2.0f - 1.0f);
}

// e4m3fn round-to-nearest-even via the float bit pattern: values are
// finite, max 448, subnormals at 2^-9 granularity.
static float SparkDsv4RefE4m3(float value)
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
		snapped = 8.0f,exponent += 1;
	return(sign * ldexpf(snapped,exponent - 4));
}

static float SparkDsv4RefPow2CeilScale(float amax, float format_max)
{
	float ratio = amax / format_max;
	return(exp2f(ceilf(log2f(ratio))));
}

// Block quantize-dequantize sim, fp8: amax floored, power-of-two scale,
// clamp to +-448, e4m3 snap, rescale - the reference's act_quant inplace.
static void SparkDsv4RefFp8Sim(float *values, uint32_t count, uint32_t block)
{
	uint32_t start,index,limit;
	float amax,scale,clamped;
	for (start = 0; start < count; start += block)
	{
		limit = start + block < count ? start + block : count;
		amax = 1e-4f;
		for (index = start; index < limit; index++)
			if ( fabsf(values[index]) > amax )
				amax = fabsf(values[index]);
		scale = SparkDsv4RefPow2CeilScale(amax,448.0f);
		for (index = start; index < limit; index++)
		{
			clamped = values[index] / scale;
			if ( clamped > 448.0f )
				clamped = 448.0f;
			if ( clamped < -448.0f )
				clamped = -448.0f;
			values[index] = SparkDsv4RefE4m3(clamped) * scale;
		}
	}
}

// e2m1 snap, RN-even between representables: at every midpoint the value
// with mantissa bit zero wins (0.25->0, 0.75->1, 1.25->1, 1.75->2,
// 2.5->2, 3.5->4, 5->4).
static float SparkDsv4RefE2m1(float value)
{
	static const float points[8] = {0.0f,0.5f,1.0f,1.5f,2.0f,3.0f,4.0f,6.0f};
	static const float ties[7] = {0.0f,1.0f,1.0f,2.0f,2.0f,4.0f,4.0f};
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

static void SparkDsv4RefFp4Sim(float *values, uint32_t count, uint32_t block)
{
	uint32_t start,index,limit;
	float amax,scale;
	for (start = 0; start < count; start += block)
	{
		limit = start + block < count ? start + block : count;
		amax = 1e-4f;
		for (index = start; index < limit; index++)
			if ( fabsf(values[index]) > amax )
				amax = fabsf(values[index]);
		scale = SparkDsv4RefPow2CeilScale(amax,6.0f);
		for (index = start; index < limit; index++)
			values[index] = SparkDsv4RefE2m1(values[index] / scale) * scale;
	}
}

// Fast Walsh-Hadamard transform scaled by n^-0.5, the reference's
// rotate_activation on power-of-two widths.
static void SparkDsv4RefHadamard(float *values, uint32_t count)
{
	uint32_t half,start,index;
	float a,b,scale = 1.0f / sqrtf((float)count);
	for (half = 1; half < count; half <<= 1)
		for (start = 0; start < count; start += half << 1)
			for (index = start; index < start + half; index++)
			{
				a = values[index];
				b = values[index + half];
				values[index] = a + b;
				values[index + half] = a - b;
			}
	for (index = 0; index < count; index++)
		values[index] *= scale;
}

// YaRN frequency table for one position: base rope frequencies, and when
// original_seq_len > 0 the interpolation ramp between the beta correction
// dims - the exact precompute_freqs_cis arithmetic.
static void SparkDsv4RefYarnFreqs(uint32_t rope_dim, float base, float factor, uint32_t original, float beta_fast, float beta_slow, float *freqs)
{
	uint32_t pair,pairs = rope_dim / 2u;
	float low,high,ramp,smooth,frequency;
	low = floorf((float)rope_dim * logf((float)original / (beta_fast * 2.0f * 3.14159265f)) / (2.0f * logf(base)));
	high = ceilf((float)rope_dim * logf((float)original / (beta_slow * 2.0f * 3.14159265f)) / (2.0f * logf(base)));
	if ( low < 0.0f )
		low = 0.0f;
	if ( high > (float)(rope_dim - 1u) )
		high = (float)(rope_dim - 1u);
	if ( low == high )
		high += 0.001f;
	for (pair = 0; pair < pairs; pair++)
	{
		frequency = 1.0f / powf(base,(float)(2u * pair) / (float)rope_dim);
		if ( original != 0u )
		{
			ramp = ((float)pair - low) / (high - low);
			if ( ramp < 0.0f )
				ramp = 0.0f;
			if ( ramp > 1.0f )
				ramp = 1.0f;
			smooth = 1.0f - ramp;
			frequency = frequency / factor * (1.0f - smooth) + frequency * smooth;
		}
		freqs[pair] = frequency;
	}
}

// Adjacent-pair complex rotation on the LAST rope_dim entries of a head
// vector; inverse conjugates - the output de-rotation.
static void SparkDsv4RefRope(float *head, uint32_t head_dim, uint32_t rope_dim, const float *freqs, float position, uint32_t inverse)
{
	uint32_t pair,base = head_dim - rope_dim;
	float angle,cosine,sine,real,imaginary;
	for (pair = 0; pair < rope_dim / 2u; pair++)
	{
		angle = position * freqs[pair];
		cosine = cosf(angle);
		sine = inverse != 0u ? -sinf(angle) : sinf(angle);
		real = head[base + 2u * pair];
		imaginary = head[base + 2u * pair + 1u];
		head[base + 2u * pair] = real * cosine - imaginary * sine;
		head[base + 2u * pair + 1u] = real * sine + imaginary * cosine;
	}
}

// Sparse attention over gathered rows: -1 indices contribute nothing, the
// sink joins the softmax denominator only, output stays un-normalized by
// any value from the sink.
static void SparkDsv4RefSparseAttn(const float *q, uint32_t heads, uint32_t head_dim, const float *kv, const int32_t *idxs, uint32_t topk, const float *sink, float scale, float *out)
{
	uint32_t head,slot,dim;
	float maximum,logit,weight,denominator;
	for (head = 0; head < heads; head++)
	{
		maximum = -INFINITY;
		for (slot = 0; slot < topk; slot++)
		{
			if ( idxs[slot] < 0 )
				continue;
			logit = 0.0f;
			for (dim = 0; dim < head_dim; dim++)
				logit += q[head * head_dim + dim] * kv[(uint32_t)idxs[slot] * head_dim + dim];
			logit *= scale;
			if ( logit > maximum )
				maximum = logit;
		}
		denominator = expf(sink[head] - maximum);
		for (dim = 0; dim < head_dim; dim++)
			out[head * head_dim + dim] = 0.0f;
		for (slot = 0; slot < topk; slot++)
		{
			if ( idxs[slot] < 0 )
				continue;
			logit = 0.0f;
			for (dim = 0; dim < head_dim; dim++)
				logit += q[head * head_dim + dim] * kv[(uint32_t)idxs[slot] * head_dim + dim];
			weight = expf(logit * scale - maximum);
			denominator += weight;
			for (dim = 0; dim < head_dim; dim++)
				out[head * head_dim + dim] += weight * kv[(uint32_t)idxs[slot] * head_dim + dim];
		}
		for (dim = 0; dim < head_dim; dim++)
			out[head * head_dim + dim] /= denominator;
	}
}

// One compressed slot from raw per-token channels: score softmax over the
// pooling axis (slots carrying -inf drop out), weighted sum - the shared
// core of prefill and decode paths.
static void SparkDsv4RefCompressPool(const float *kv, const float *score, uint32_t slots, uint32_t width, float *out)
{
	uint32_t slot,channel;
	float maximum,weight,denominator;
	for (channel = 0; channel < width; channel++)
	{
		maximum = -INFINITY;
		for (slot = 0; slot < slots; slot++)
			if ( score[slot * width + channel] > maximum )
				maximum = score[slot * width + channel];
		denominator = 0.0f;
		out[channel] = 0.0f;
		for (slot = 0; slot < slots; slot++)
		{
			if ( score[slot * width + channel] == -INFINITY )
				continue;
			weight = expf(score[slot * width + channel] - maximum);
			denominator += weight;
			out[channel] += weight * kv[slot * width + channel];
		}
		out[channel] /= denominator;
	}
}

/*
 * Compressor decode state: the reference's kv_state/score_state buffers.
 * With overlap the state holds 2*ratio slots of 2*d channels; a compress
 * event concatenates [previous group's first-half channels | current
 * group's second-half channels] into a 2*ratio x d pool and shifts the
 * current group down. Without overlap it is a plain ratio-slot pool.
 */
typedef struct SparkDsv4RefCompressState
{
	uint32_t ratio;
	uint32_t overlap;
	uint32_t width;
	float *kv_state;
	float *score_state;
} SparkDsv4RefCompressState;

static void SparkDsv4RefCompressStateInit(SparkDsv4RefCompressState *state, uint32_t ratio, uint32_t overlap, uint32_t width, float *kv_storage, float *score_storage)
{
	uint32_t slots = (overlap != 0u ? 2u : 1u) * ratio,channels = (overlap != 0u ? 2u : 1u) * width,index;
	state->ratio = ratio;
	state->overlap = overlap;
	state->width = width;
	state->kv_state = kv_storage;
	state->score_state = score_storage;
	for (index = 0; index < slots * channels; index++)
	{
		state->kv_state[index] = 0.0f;
		state->score_state[index] = -INFINITY;
	}
}

// One decode token into the state; returns 1 with the pooled d-wide slot
// in out when (position+1) crosses a ratio boundary. kv/score are the
// token's coff*d channels with the layer's ape row already added to score.
static uint32_t SparkDsv4RefCompressStateStep(SparkDsv4RefCompressState *state, const float *kv, const float *score, uint64_t position, float *out)
{
	uint32_t ratio = state->ratio,width = state->width,channels = (state->overlap != 0u ? 2u : 1u) * width;
	uint32_t slot = (uint32_t)(position % ratio),index,channel;
	float *pool_kv,*pool_score;
	if ( state->overlap != 0u )
	{
		memcpy(state->kv_state + (ratio + slot) * channels,kv,channels * sizeof(float));
		memcpy(state->score_state + (ratio + slot) * channels,score,channels * sizeof(float));
		if ( (position + 1u) % ratio != 0u )
			return(0u);
		pool_kv = (float *)malloc(2u * ratio * width * sizeof(float));
		pool_score = (float *)malloc(2u * ratio * width * sizeof(float));
		for (index = 0; index < ratio; index++)
			for (channel = 0; channel < width; channel++)
			{
				pool_kv[index * width + channel] = state->kv_state[index * channels + channel];
				pool_score[index * width + channel] = state->score_state[index * channels + channel];
				pool_kv[(ratio + index) * width + channel] = state->kv_state[(ratio + index) * channels + width + channel];
				pool_score[(ratio + index) * width + channel] = state->score_state[(ratio + index) * channels + width + channel];
			}
		SparkDsv4RefCompressPool(pool_kv,pool_score,2u * ratio,width,out);
		memcpy(state->kv_state,state->kv_state + ratio * channels,ratio * channels * sizeof(float));
		memcpy(state->score_state,state->score_state + ratio * channels,ratio * channels * sizeof(float));
		free(pool_kv);
		free(pool_score);
		return(1u);
	}
	memcpy(state->kv_state + slot * channels,kv,channels * sizeof(float));
	memcpy(state->score_state + slot * channels,score,channels * sizeof(float));
	if ( (position + 1u) % ratio != 0u )
		return(0u);
	SparkDsv4RefCompressPool(state->kv_state,state->score_state,ratio,width,out);
	return(1u);
}

/*
 * One-shot prefill compression of tokens [0, count) - the start_pos == 0
 * branch: whole groups pool directly; with overlap, group g pools its own
 * second-half channels together with group g-1's first-half channels
 * (group 0's overlap half carries zero kv at -inf score, dropping out).
 * kv/score are count x channels with ape already added per in-group slot.
 */
static void SparkDsv4RefCompressPrefill(const float *kv, const float *score, uint32_t count, uint32_t ratio, uint32_t overlap, uint32_t width, float *out, uint32_t *out_count)
{
	uint32_t channels = (overlap != 0u ? 2u : 1u) * width,groups = count / ratio,group,slot,channel;
	float *pool_kv = (float *)malloc(2u * ratio * width * sizeof(float));
	float *pool_score = (float *)malloc(2u * ratio * width * sizeof(float));
	for (group = 0; group < groups; group++)
	{
		for (slot = 0; slot < ratio; slot++)
			for (channel = 0; channel < width; channel++)
			{
				pool_kv[(ratio + slot) * width + channel] = kv[(group * ratio + slot) * channels + (overlap != 0u ? width : 0u) + channel];
				pool_score[(ratio + slot) * width + channel] = score[(group * ratio + slot) * channels + (overlap != 0u ? width : 0u) + channel];
				pool_kv[slot * width + channel] = overlap != 0u && group != 0u ? kv[((group - 1u) * ratio + slot) * channels + channel] : 0.0f;
				pool_score[slot * width + channel] = overlap != 0u && group != 0u ? score[((group - 1u) * ratio + slot) * channels + channel] : -INFINITY;
			}
		SparkDsv4RefCompressPool(pool_kv + (overlap != 0u ? 0u : ratio * width),pool_score + (overlap != 0u ? 0u : ratio * width),(overlap != 0u ? 2u : 1u) * ratio,width,out + group * width);
	}
	free(pool_kv);
	free(pool_score);
	*out_count = groups;
}

// Indexer score for one query over the compressed index cache: per head
// relu(q_h . kv) times the projected head weight, summed over heads.
static void SparkDsv4RefIndexerScore(const float *q, uint32_t heads, uint32_t head_dim, const float *kv, uint32_t slots, const float *weights, float *scores)
{
	uint32_t slot,head,dim;
	float dot;
	for (slot = 0; slot < slots; slot++)
	{
		scores[slot] = 0.0f;
		for (head = 0; head < heads; head++)
		{
			dot = 0.0f;
			for (dim = 0; dim < head_dim; dim++)
				dot += q[head * head_dim + dim] * kv[slot * head_dim + dim];
			if ( dot > 0.0f )
				scores[slot] += dot * weights[head];
		}
	}
}

// Top-k by score, ties resolved to the lower slot index; -1 padding past
// the available slot count - torch.topk order semantics.
static void SparkDsv4RefTopK(const float *scores, uint32_t slots, uint32_t topk, int32_t *indices)
{
	uint32_t rank,slot;
	int32_t best;
	uint8_t taken[4096];
	memset(taken,0,slots);
	for (rank = 0; rank < topk; rank++)
	{
		best = -1;
		for (slot = 0; slot < slots; slot++)
			if ( taken[slot] == 0u && (best < 0 || scores[slot] > scores[best]) )
				best = (int32_t)slot;
		indices[rank] = best;
		if ( best >= 0 )
			taken[best] = 1u;
	}
}

// sqrtsoftplus scores, noaux_tc selection (bias shifts the top-k, never
// the weights), sum-normalized gathered weights scaled by route_scale.
static void SparkDsv4RefRouter(const float *logits, const float *bias, uint32_t experts, uint32_t topk, float route_scale, int32_t *indices, float *weights)
{
	float scores[512],shifted[512],total = 0.0f;
	uint32_t expert,rank;
	for (expert = 0; expert < experts; expert++)
	{
		scores[expert] = sqrtf(logf(1.0f + expf(logits[expert])));
		shifted[expert] = scores[expert] + (bias != 0 ? bias[expert] : 0.0f);
	}
	SparkDsv4RefTopK(shifted,experts,topk,indices);
	for (rank = 0; rank < topk; rank++)
		total += scores[indices[rank]];
	for (rank = 0; rank < topk; rank++)
		weights[rank] = scores[indices[rank]] / total * route_scale;
}

// The swiglu clamp: up two-sided, gate max-only, silu(gate)*up, routing
// weight on the fp32 intermediate.
static float SparkDsv4RefSwigluClamp(float gate, float up, float limit, float weight)
{
	if ( limit > 0.0f )
	{
		if ( up > limit )
			up = limit;
		if ( up < -limit )
			up = -limit;
		if ( gate > limit )
			gate = limit;
	}
	return(gate / (1.0f + expf(-gate)) * up * weight);
}

// The mHC split, the exact kernel arithmetic: sigmoid pre (+eps), doubled
// sigmoid post, comb row-softmax +eps then 2*iters alternating row/col
// normalizations with +eps inside every division (the first row pass is
// the softmax itself).
static void SparkDsv4RefHcSplitSinkhorn(const float *mixes, const float *hc_scale, const float *hc_base, uint32_t hc, uint32_t iterations, float eps, float *pre, float *post, float *comb)
{
	uint32_t row,column,iteration;
	float maximum,total;
	for (row = 0; row < hc; row++)
	{
		pre[row] = 1.0f / (1.0f + expf(-(mixes[row] * hc_scale[0] + hc_base[row]))) + eps;
		post[row] = 2.0f / (1.0f + expf(-(mixes[hc + row] * hc_scale[1] + hc_base[hc + row])));
	}
	for (row = 0; row < hc; row++)
	{
		maximum = -INFINITY;
		for (column = 0; column < hc; column++)
		{
			comb[row * hc + column] = mixes[2u * hc + row * hc + column] * hc_scale[2] + hc_base[2u * hc + row * hc + column];
			if ( comb[row * hc + column] > maximum )
				maximum = comb[row * hc + column];
		}
		total = 0.0f;
		for (column = 0; column < hc; column++)
		{
			comb[row * hc + column] = expf(comb[row * hc + column] - maximum);
			total += comb[row * hc + column];
		}
		for (column = 0; column < hc; column++)
			comb[row * hc + column] = comb[row * hc + column] / total + eps;
	}
	for (iteration = 0; iteration < iterations; iteration++)
	{
		if ( iteration != 0u )
			for (row = 0; row < hc; row++)
			{
				total = 0.0f;
				for (column = 0; column < hc; column++)
					total += comb[row * hc + column];
				for (column = 0; column < hc; column++)
					comb[row * hc + column] /= total + eps;
			}
		for (column = 0; column < hc; column++)
		{
			total = 0.0f;
			for (row = 0; row < hc; row++)
				total += comb[row * hc + column];
			for (row = 0; row < hc; row++)
				comb[row * hc + column] /= total + eps;
		}
	}
}

// hc_pre: mixes = fn @ flat(x) scaled by rsqrt(mean(flat^2)+eps), split,
// then the pre-weighted stream reduction to one d-wide vector.
static void SparkDsv4RefHcPre(const float *x, uint32_t hc, uint32_t dim, const float *fn, const float *scale3, const float *base, uint32_t iterations, float eps, float *reduced, float *post, float *comb)
{
	uint32_t mix_rows = (2u + hc) * hc,row,element,stream;
	float mixes[64],pre[8],sum = 0.0f,rsqrt;
	for (element = 0; element < hc * dim; element++)
		sum += x[element] * x[element];
	rsqrt = 1.0f / sqrtf(sum / (float)(hc * dim) + eps);
	for (row = 0; row < mix_rows; row++)
	{
		mixes[row] = 0.0f;
		for (element = 0; element < hc * dim; element++)
			mixes[row] += fn[row * hc * dim + element] * x[element];
		mixes[row] *= rsqrt;
	}
	SparkDsv4RefHcSplitSinkhorn(mixes,scale3,base,hc,iterations,eps,pre,post,comb);
	for (element = 0; element < dim; element++)
	{
		reduced[element] = 0.0f;
		for (stream = 0; stream < hc; stream++)
			reduced[element] += pre[stream] * x[stream * dim + element];
	}
}

// hc_post: stream k = post[k] * out + sum_j comb[j][k] * residual[j] - the
// comb applies TRANSPOSED, pinned from the reference's einsum axes.
static void SparkDsv4RefHcPost(const float *out, const float *residual, uint32_t hc, uint32_t dim, const float *post, const float *comb, float *streams)
{
	uint32_t stream,source,element;
	for (stream = 0; stream < hc; stream++)
		for (element = 0; element < dim; element++)
		{
			streams[stream * dim + element] = post[stream] * out[element];
			for (source = 0; source < hc; source++)
				streams[stream * dim + element] += comb[source * hc + stream] * residual[source * dim + element];
		}
}

// The head reduction: sigmoid pre only, no post, no comb - shared by the
// final head and the MTP head.
static void SparkDsv4RefHcHead(const float *x, uint32_t hc, uint32_t dim, const float *fn, float scale, const float *base, float eps, float *reduced)
{
	uint32_t row,element,stream;
	float mixes[8],sum = 0.0f,rsqrt,pre;
	for (element = 0; element < hc * dim; element++)
		sum += x[element] * x[element];
	rsqrt = 1.0f / sqrtf(sum / (float)(hc * dim) + eps);
	for (row = 0; row < hc; row++)
	{
		mixes[row] = 0.0f;
		for (element = 0; element < hc * dim; element++)
			mixes[row] += fn[row * hc * dim + element] * x[element];
		mixes[row] *= rsqrt;
	}
	for (element = 0; element < dim; element++)
		reduced[element] = 0.0f;
	for (stream = 0; stream < hc; stream++)
	{
		pre = 1.0f / (1.0f + expf(-(mixes[stream] * scale + base[stream]))) + eps;
		for (element = 0; element < dim; element++)
			reduced[element] += pre * x[stream * dim + element];
	}
}

static void SparkDsv4RefTestQuant(void)
{
	float values[8] = {0.30f,-0.74f,1.24f,-1.26f,2.49f,5.10f,-7.00f,0.0f};
	float block[4] = {448.0f,224.0f,-112.0f,3.5f};
	SparkDsv4RefFp4Sim(values,8u,8u);
	// amax 7 -> scale 2: representables x2 are 0,1,2,3,4,6,8,12; the -7
	// input lands on the (6,8) midpoint and RN-even takes the 8.
	if ( values[0] != 0.0f || values[1] != -1.0f || values[2] != 1.0f || values[3] != -1.0f )
		abort();
	if ( values[4] != 2.0f || values[5] != 6.0f || values[6] != -8.0f || values[7] != 0.0f )
		abort();
	SparkDsv4RefFp8Sim(block,4u,4u);
	// amax exactly 448 -> scale 1: e4m3-representable values survive.
	if ( block[0] != 448.0f || block[1] != 224.0f || block[2] != -112.0f || block[3] != 3.5f )
		abort();
	printf("quant sims PASS\n");
}

static void SparkDsv4RefTestRope(void)
{
	float freqs[32],head[64],copy[64];
	uint32_t index;
	uint64_t seed = 11u;
	SparkDsv4RefYarnFreqs(64u,160000.0f,16.0f,65536u,32.0f,1.0f,freqs);
	for (index = 0; index < 64u; index++)
		copy[index] = head[index] = SparkDsv4RefUniform(&seed);
	SparkDsv4RefRope(head,64u,64u,freqs,12345.0f,0u);
	SparkDsv4RefRope(head,64u,64u,freqs,12345.0f,1u);
	for (index = 0; index < 64u; index++)
		if ( fabsf(head[index] - copy[index]) > 1e-4f )
			abort();
	// YaRN interpolates: every scaled frequency sits within [f/16, f].
	for (index = 0; index < 32u; index++)
	{
		float base_frequency = 1.0f / powf(160000.0f,(float)(2u * index) / 64.0f);
		if ( freqs[index] > base_frequency * 1.0001f || freqs[index] < base_frequency / 16.0f * 0.9999f )
			abort();
	}
	printf("rope + inverse PASS\n");
}

static void SparkDsv4RefTestSparseAttn(void)
{
	float q[2 * 8],kv[6 * 8],sink[2] = {-INFINITY,3.0f},out[2 * 8],logit,weight,denominator,expected;
	int32_t idxs[4] = {0,3,-1,5};
	uint32_t head,slot,dim,active[3] = {0u,3u,5u};
	uint64_t seed = 22u;
	for (dim = 0; dim < 16u; dim++)
		q[dim] = SparkDsv4RefUniform(&seed);
	for (dim = 0; dim < 48u; dim++)
		kv[dim] = SparkDsv4RefUniform(&seed);
	SparkDsv4RefSparseAttn(q,2u,8u,kv,idxs,4u,sink,0.35f,out);
	for (head = 0; head < 2u; head++)
	{
		denominator = sink[head] == -INFINITY ? 0.0f : expf(sink[head]);
		expected = 0.0f;
		for (slot = 0; slot < 3u; slot++)
		{
			logit = 0.0f;
			for (dim = 0; dim < 8u; dim++)
				logit += q[head * 8u + dim] * kv[active[slot] * 8u + dim];
			weight = expf(logit * 0.35f);
			denominator += weight;
			if ( slot == 1u )
				expected += weight * kv[active[1] * 8u + 0u];
			else
				expected += weight * kv[active[slot] * 8u + 0u];
		}
		if ( fabsf(out[head * 8u + 0u] - expected / denominator) > 1e-4f * fabsf(expected / denominator) + 1e-5f )
			abort();
	}
	printf("sparse attention + sink PASS\n");
}

// The state-carry contract: a sequence compressed one-shot must equal the
// same tokens stepped through the decode state, overlap and non-overlap.
static void SparkDsv4RefTestCompressor(uint32_t ratio, uint32_t overlap, uint32_t count)
{
	uint32_t width = 6u,channels = (overlap != 0u ? 2u : 1u) * width,groups,token,index,emitted = 0u;
	float *kv = (float *)malloc(count * channels * sizeof(float));
	float *score = (float *)malloc(count * channels * sizeof(float));
	float *prefill_out = (float *)malloc((count / ratio) * width * sizeof(float));
	float *decode_out = (float *)malloc((count / ratio) * width * sizeof(float));
	float kv_storage[2 * 128 * 12],score_storage[2 * 128 * 12];
	SparkDsv4RefCompressState state;
	uint64_t seed = 33u + ratio;
	for (index = 0; index < count * channels; index++)
	{
		kv[index] = SparkDsv4RefUniform(&seed);
		score[index] = SparkDsv4RefUniform(&seed) * 2.0f;
	}
	SparkDsv4RefCompressPrefill(kv,score,count,ratio,overlap,width,prefill_out,&groups);
	SparkDsv4RefCompressStateInit(&state,ratio,overlap,width,kv_storage,score_storage);
	for (token = 0; token < count; token++)
		emitted += SparkDsv4RefCompressStateStep(&state,kv + token * channels,score + token * channels,token,decode_out + emitted * width);
	if ( emitted != groups || groups != count / ratio )
		abort();
	for (index = 0; index < groups * width; index++)
		if ( fabsf(prefill_out[index] - decode_out[index]) > 1e-4f )
			abort();
	free(kv);
	free(score);
	free(prefill_out);
	free(decode_out);
	printf("compressor prefill==decode ratio=%u overlap=%u groups=%u PASS\n",ratio,overlap,groups);
}

static void SparkDsv4RefTestIndexerAndRouter(void)
{
	float q[3 * 8],kv[10 * 8],weights[3] = {0.5f,-0.25f,1.5f},scores[10],naive,dot;
	int32_t indices[6];
	uint32_t head,slot,dim,rank;
	uint64_t seed = 44u;
	for (dim = 0; dim < 24u; dim++)
		q[dim] = SparkDsv4RefUniform(&seed);
	SparkDsv4RefHadamard(q,8u);
	SparkDsv4RefFp4Sim(q,24u,8u);
	for (dim = 0; dim < 80u; dim++)
		kv[dim] = SparkDsv4RefUniform(&seed);
	SparkDsv4RefIndexerScore(q,3u,8u,kv,10u,weights,scores);
	for (slot = 0; slot < 10u; slot++)
	{
		naive = 0.0f;
		for (head = 0; head < 3u; head++)
		{
			dot = 0.0f;
			for (dim = 0; dim < 8u; dim++)
				dot += q[head * 8u + dim] * kv[slot * 8u + dim];
			naive += (dot > 0.0f ? dot : 0.0f) * weights[head];
		}
		if ( fabsf(scores[slot] - naive) > 1e-5f )
			abort();
	}
	SparkDsv4RefTopK(scores,10u,6u,indices);
	for (rank = 1; rank < 6u; rank++)
		if ( scores[indices[rank]] > scores[indices[rank - 1u]] )
			abort();
	SparkDsv4RefTopK(scores,4u,6u,indices);
	if ( indices[4] != -1 || indices[5] != -1 )
		abort();
	printf("indexer score + topk PASS\n");
}

static void SparkDsv4RefTestRouterAndSwiglu(void)
{
	float logits[16],bias[16],route_weights[4],total;
	int32_t route_indices[4];
	uint32_t dim,rank;
	uint64_t seed = 66u;
	for (dim = 0; dim < 16u; dim++)
	{
		logits[dim] = SparkDsv4RefUniform(&seed) * 3.0f;
		bias[dim] = SparkDsv4RefUniform(&seed) * 0.1f;
	}
	SparkDsv4RefRouter(logits,bias,16u,4u,1.5f,route_indices,route_weights);
	total = 0.0f;
	for (rank = 0; rank < 4u; rank++)
		total += route_weights[rank];
	if ( fabsf(total - 1.5f) > 1e-5f )
		abort();
	// Hash path: same weight arithmetic on externally chosen indices.
	route_indices[0] = 2;
	route_indices[1] = 7;
	route_indices[2] = 11;
	route_indices[3] = 3;
	total = 0.0f;
	for (rank = 0; rank < 4u; rank++)
		total += sqrtf(logf(1.0f + expf(logits[route_indices[rank]])));
	for (rank = 0; rank < 4u; rank++)
		if ( fabsf(sqrtf(logf(1.0f + expf(logits[route_indices[rank]]))) / total * 1.5f) > 1.5f )
			abort();
	if ( fabsf(SparkDsv4RefSwigluClamp(25.0f,-14.0f,10.0f,2.0f) - (10.0f / (1.0f + expf(-10.0f)) * -10.0f * 2.0f)) > 1e-4f )
		abort();
	if ( fabsf(SparkDsv4RefSwigluClamp(-25.0f,4.0f,10.0f,1.0f) - (-25.0f / (1.0f + expf(25.0f)) * 4.0f)) > 1e-6f )
		abort();
	printf("router + swiglu clamp PASS\n");
}

static void SparkDsv4RefTestHc(void)
{
	float fn[24 * 4 * 5],base[24],scale3[3] = {0.7f,0.9f,1.1f},x[4 * 5],residual[4 * 5],reduced[5],post[4],comb[16],streams[4 * 5],row_total,column_total,check;
	float head_fn[4 * 4 * 5],head_base[4],head_out[5];
	uint32_t row,column,element,stream;
	uint64_t seed = 55u;
	for (element = 0; element < 24u * 20u; element++)
		fn[element] = SparkDsv4RefUniform(&seed) * 0.2f;
	for (element = 0; element < 24u; element++)
		base[element] = SparkDsv4RefUniform(&seed) * 0.3f;
	for (element = 0; element < 20u; element++)
	{
		x[element] = SparkDsv4RefUniform(&seed);
		residual[element] = SparkDsv4RefUniform(&seed);
	}
	SparkDsv4RefHcPre(x,4u,5u,fn,scale3,base,20u,1e-6f,reduced,post,comb);
	// Sinkhorn leaves comb near doubly stochastic; the trailing column
	// pass makes columns exact to fp32, rows within the +eps drift.
	for (row = 0; row < 4u; row++)
	{
		row_total = 0.0f;
		column_total = 0.0f;
		for (column = 0; column < 4u; column++)
		{
			row_total += comb[row * 4u + column];
			column_total += comb[column * 4u + row];
		}
		if ( fabsf(row_total - 1.0f) > 1e-3f || fabsf(column_total - 1.0f) > 1e-4f )
			abort();
	}
	for (stream = 0; stream < 4u; stream++)
		if ( post[stream] <= 0.0f || post[stream] >= 2.0f )
			abort();
	SparkDsv4RefHcPost(reduced,residual,4u,5u,post,comb,streams);
	for (stream = 0; stream < 4u; stream++)
		for (element = 0; element < 5u; element++)
		{
			check = post[stream] * reduced[element];
			for (row = 0; row < 4u; row++)
				check += comb[row * 4u + stream] * residual[row * 5u + element];
			if ( fabsf(streams[stream * 5u + element] - check) > 1e-5f )
				abort();
		}
	for (element = 0; element < 80u; element++)
		head_fn[element] = SparkDsv4RefUniform(&seed) * 0.2f;
	for (element = 0; element < 4u; element++)
		head_base[element] = SparkDsv4RefUniform(&seed);
	SparkDsv4RefHcHead(x,4u,5u,head_fn,0.8f,head_base,1e-6f,head_out);
	for (element = 0; element < 5u; element++)
		if ( !isfinite(head_out[element]) )
			abort();
	printf("mHC pre/post/sinkhorn/head PASS\n");
}

int32_t main(void)
{
	SparkDsv4RefTestQuant();
	SparkDsv4RefTestRope();
	SparkDsv4RefTestSparseAttn();
	SparkDsv4RefTestCompressor(4u,1u,29u);
	SparkDsv4RefTestCompressor(4u,0u,29u);
	SparkDsv4RefTestCompressor(5u,0u,23u);
	SparkDsv4RefTestCompressor(128u,0u,384u);
	SparkDsv4RefTestIndexerAndRouter();
	SparkDsv4RefTestRouterAndSwiglu();
	SparkDsv4RefTestHc();
	printf("spark_dsv4_reference PASS\n");
	return(0);
}
