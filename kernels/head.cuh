#pragma once

// The sampling head. Final norm, logits, token selection.
//
// The last thing a decode step does and the only part that produces a token.
// Every model needs it and none had it; the exemption list in
// tests/test_config_coverage.py named it for five models at once, which is what
// that list is for.
//
// TWO SHAPES, AND THE CHOICE IS NOT A PREFERENCE. Full-vocabulary decoding
// projects the hidden state against the whole embedding matrix: at GLM 5.2's
// 6144 hidden and 154,880 vocabulary that is 952 M parameters, 1.9 GB in BF16,
// read once per step. That is a third of a routed layer's weight stream for one
// token, which is why it dominates at small batch and why the restricted form
// exists.
//
// Restricted decoding projects against a subset - a grammar, a tool schema, a
// set of legal continuations - and costs the subset's size. A 256-token
// restriction is 1.6 MB against 1.9 GB, three orders of magnitude, and it is
// exact rather than approximate: the tokens outside the set were going to be
// rejected anyway.
//
// So the head is not one kernel with a flag. It is two, and a caller that knows
// its continuation set is constrained should say so.
//
// WHY THE ARGMAX IS SEPARATE FROM THE PROJECTION. Fusing them looks free - the
// logits are in registers, take the max there - but the projection is one block
// per (row, vocabulary tile) and the max is over the whole row. Fusing forces
// either one block per row, which leaves the machine idle at small batch, or a
// grid-wide reduction inside the kernel, which needs cooperative launch. Two
// kernels and a small intermediate is the cheaper arrangement, and the
// intermediate is candidates rather than full logits: each projection block
// emits its own best, and the second pass picks among those.

#include "kernels/gemm.cuh"
#include "kernels/norm.cuh"
#include "kernels/topk.cuh"
#include <stdint.h>

// One block per (row, vocabulary tile). Each emits the best token in its tile,
// so the second pass reduces over tiles rather than over the vocabulary - 152
// candidates instead of 154,880 at a 1024-wide tile.
template<uint32_t THREADS, uint32_t TILE>
__global__ __launch_bounds__(THREADS, 1)
void LmHeadCandidateKernel(const uint16_t *__restrict__ normed_bf16, const uint16_t *__restrict__ head_weight_bf16, const uint32_t *__restrict__ token_ids, float *__restrict__ candidate_score, uint32_t *__restrict__ candidate_token, uint32_t rows, uint32_t hidden, uint32_t vocabulary)
{
	__shared__ float shared_score[THREADS];
	__shared__ uint32_t shared_token[THREADS];
	uint32_t row = blockIdx.y,tile = blockIdx.x,index,element,stride;
	float best = -INFINITY;
	uint32_t best_token = 0u;
	for (index = tile * TILE + threadIdx.x; index < (tile + 1u) * TILE && index < vocabulary; index += THREADS)
	{
		// token_ids null means the full vocabulary and the row index IS the
		// token; non-null means a restricted set and the row is an index into it.
		uint32_t token = token_ids != 0 ? token_ids[index] : index;
		float total = 0.0f;
		for (element = 0u; element < hidden; ++element)
			total += LmBf16ToFloat(normed_bf16[((uint64_t)row * hidden) + element])
				* LmBf16ToFloat(head_weight_bf16[((uint64_t)token * hidden) + element]);
		if ( total > best )
		{
			best = total;
			best_token = token;
		}
	}
	shared_score[threadIdx.x] = best;
	shared_token[threadIdx.x] = best_token;
	__syncthreads();
	for (stride = THREADS / 2u; stride > 0u; stride >>= 1u)
	{
		if ( threadIdx.x < stride && shared_score[threadIdx.x + stride] > shared_score[threadIdx.x] )
		{
			shared_score[threadIdx.x] = shared_score[threadIdx.x + stride];
			shared_token[threadIdx.x] = shared_token[threadIdx.x + stride];
		}
		__syncthreads();
	}
	if ( threadIdx.x == 0u )
	{
		candidate_score[(row * gridDim.x) + tile] = shared_score[0];
		candidate_token[(row * gridDim.x) + tile] = shared_token[0];
	}
}

// Reduce the per-tile candidates to one token per row.
//
// Ties go to the lower token id, deterministically. An argmax that breaks ties
// by whichever thread arrived first is not reproducible across launches, and a
// decoder that is not reproducible cannot be compared against a reference - which
// is the whole reason a reference exists.
template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmHeadCommitKernel(const float *__restrict__ candidate_score, const uint32_t *__restrict__ candidate_token, uint32_t tiles, uint32_t *__restrict__ token_out, float *__restrict__ score_out, uint32_t rows)
{
	__shared__ float shared_score[THREADS];
	__shared__ uint32_t shared_token[THREADS];
	uint32_t row = blockIdx.x,index,stride;
	float best = -INFINITY;
	uint32_t best_token = 0xffffffffu;
	if ( row >= rows )
		return;
	for (index = threadIdx.x; index < tiles; index += THREADS)
	{
		float score = candidate_score[(row * tiles) + index];
		uint32_t token = candidate_token[(row * tiles) + index];
		if ( score > best || (score == best && token < best_token) )
		{
			best = score;
			best_token = token;
		}
	}
	shared_score[threadIdx.x] = best;
	shared_token[threadIdx.x] = best_token;
	__syncthreads();
	for (stride = THREADS / 2u; stride > 0u; stride >>= 1u)
	{
		if ( threadIdx.x < stride )
		{
			float other = shared_score[threadIdx.x + stride];
			uint32_t other_token = shared_token[threadIdx.x + stride];
			if ( other > shared_score[threadIdx.x]
				|| (other == shared_score[threadIdx.x] && other_token < shared_token[threadIdx.x]) )
			{
				shared_score[threadIdx.x] = other;
				shared_token[threadIdx.x] = other_token;
			}
		}
		__syncthreads();
	}
	if ( threadIdx.x == 0u )
	{
		token_out[row] = shared_token[0];
		if ( score_out != 0 )
			score_out[row] = shared_score[0];
	}
}

// Softmax over a row of logits, in place, for sampled decoding.
//
// Subtracts the maximum before exponentiating. Without that a logit of 90 -
// ordinary for a confident model at this vocabulary size - overflows a float
// exponential and the whole row becomes NaN, which propagates into every
// subsequent token rather than producing one bad one.
template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmHeadSoftmaxKernel(float *__restrict__ logits, uint32_t rows, uint32_t vocabulary, float temperature)
{
	__shared__ float reduction[THREADS / LM_WARP_LANES];
	uint64_t base = (uint64_t)blockIdx.x * vocabulary;
	uint32_t index;
	float local = -INFINITY,maximum,total = 0.0f,inverse_temperature;
	if ( blockIdx.x >= rows )
		return;
	inverse_temperature = 1.0f / fmaxf(temperature,1.0e-4f);
	for (index = threadIdx.x; index < vocabulary; index += THREADS)
		local = fmaxf(local,logits[base + index] * inverse_temperature);
	maximum = LmBlockMax<THREADS>(local,reduction);
	for (index = threadIdx.x; index < vocabulary; index += THREADS)
	{
		float value = __expf((logits[base + index] * inverse_temperature) - maximum);
		logits[base + index] = value;
		total += value;
	}
	total = LmBlockSum<THREADS>(total,reduction);
	for (index = threadIdx.x; index < vocabulary; index += THREADS)
		logits[base + index] /= fmaxf(total,1.0e-20f);
}
