#pragma once

// Linear attention with a recurrent state. The delta rule at decode.
//
// The other half of this tree's architecture space. Qwen 3.6 runs Gated DeltaNet
// on 48 of 64 layers and Kimi K3 runs Kimi Delta Attention on 3 of every 4, and
// both are the same recurrence with different gate parameterisations.
//
// WHAT MAKES IT DIFFERENT FROM EVERYTHING ELSE HERE. Softmax attention keeps
// every past key and value and re-reads them each step, so its cost grows with
// context. This keeps a fixed matrix per head - key_dim by value_dim - and
// updates it in place, so its cost per token is constant no matter how long the
// conversation is. At Qwen 3.6's widths that is 512 KB per sequence against a KV
// cache that passes 512 KB at about 170 tokens and keeps growing.
//
// THE DELTA RULE, and why it is not just an accumulation. A naive linear
// attention sets S += v k^T, which writes every value into the state and never
// removes anything - keys that recur just pile up. The delta rule instead
// computes what the state ALREADY predicts for this key, and writes only the
// difference:
//
//     predicted = S^T k
//     S = alpha * S + beta * (v - predicted) k^T
//
// So a key the state already handles correctly changes nothing, and a key it
// gets wrong is corrected in proportion to the error. That is what makes the
// fixed-size state usable for long context rather than saturating.
//
// alpha is the forget gate, beta the write strength. Both are per-token and
// per-head, produced by the layer's projections. Setting alpha to 1 and beta to
// 1 recovers the ungated rule; setting beta to 1 and dropping the prediction
// recovers naive linear attention. Neither is what these models do.
//
// STATE LAYOUT. [head][key_dim][value_dim], key-major, so the k^T outer product
// writes contiguous value_dim runs and the S^T q read is a strided gather that
// every thread does once. The alternative layout makes the update contiguous and
// the read strided, and the update happens once per token where the read happens
// once per token too - so the tie is broken by the outer product being the
// larger of the two.

#include "kernels/dtype.cuh"
#include "kernels/norm.cuh"
#include <stdint.h>

// One decode step of the gated delta rule, one block per (row, head).
//
// GROUPED VALUE HEADS. Qwen 3.6 has 16 key heads and 48 value heads, so three
// value heads share each key head's state slice. The state is indexed by key
// head and the value offset selects within it, which is why value_heads_per_key
// is a parameter rather than assumed to be one.
template<uint32_t THREADS, uint32_t KEY_DIM, uint32_t VALUE_DIM>
__global__ __launch_bounds__(THREADS, 1)
void LmDeltaRuleDecodeKernel(uint8_t *__restrict__ state_pool, const uint32_t *__restrict__ state_index, const uint16_t *__restrict__ query_bf16, const uint16_t *__restrict__ key_bf16, const uint16_t *__restrict__ value_bf16, const float *__restrict__ forget_gate, const float *__restrict__ write_gate, uint16_t *__restrict__ output_bf16, uint32_t key_heads, uint32_t value_heads_per_key, uint32_t rows)
{
	__shared__ float shared_key[KEY_DIM];
	__shared__ float shared_predicted[VALUE_DIM];
	uint32_t row = blockIdx.x,head = blockIdx.y,index,element;
	uint16_t *state;
	float alpha,beta;
	if ( row >= rows || head >= key_heads )
		return;
	// One slot per sequence, never paged: the state does not grow, so its
	// address is the sequence's slot base plus this head's slice.
	state = (uint16_t *)(state_pool
		+ ((uint64_t)state_index[row] * key_heads * KEY_DIM * VALUE_DIM * 2u)
		+ ((uint64_t)head * KEY_DIM * VALUE_DIM * 2u));
	alpha = forget_gate[(row * key_heads) + head];
	beta = write_gate[(row * key_heads) + head];
	for (index = threadIdx.x; index < KEY_DIM; index += THREADS)
		shared_key[index] = LmBf16ToFloat(key_bf16[(((uint64_t)row * key_heads) + head) * KEY_DIM + index]);
	for (index = threadIdx.x; index < VALUE_DIM; index += THREADS)
		shared_predicted[index] = 0.0f;
	__syncthreads();
	// predicted = S^T k. Each thread owns a set of value columns and walks the
	// key dimension, so the state read is contiguous per thread.
	for (element = threadIdx.x; element < VALUE_DIM; element += THREADS)
	{
		float total = 0.0f;
		for (index = 0u; index < KEY_DIM; ++index)
			total += LmBf16ToFloat(state[(index * VALUE_DIM) + element]) * shared_key[index];
		shared_predicted[element] = total;
	}
	__syncthreads();
	// S = alpha*S + beta*(v - predicted) k^T, and o = S^T q with the UPDATED
	// state. Updating first is not an optimisation detail: the token must attend
	// to its own value, exactly as a softmax token attends to its own key.
	{
		uint32_t value_head = head * value_heads_per_key;
		for (index = 0u; index < KEY_DIM; ++index)
		{
			float k = shared_key[index];
			for (element = threadIdx.x; element < VALUE_DIM; element += THREADS)
			{
				float v = LmBf16ToFloat(value_bf16[
					(((uint64_t)row * key_heads * value_heads_per_key) + value_head) * VALUE_DIM + element]);
				float previous = LmBf16ToFloat(state[(index * VALUE_DIM) + element]);
				state[(index * VALUE_DIM) + element] =
					LmFloatToBf16((alpha * previous) + (beta * (v - shared_predicted[element]) * k));
			}
		}
	}
	__syncthreads();
	for (element = threadIdx.x; element < VALUE_DIM; element += THREADS)
	{
		float total = 0.0f;
		for (index = 0u; index < KEY_DIM; ++index)
			total += LmBf16ToFloat(state[(index * VALUE_DIM) + element])
				* LmBf16ToFloat(query_bf16[(((uint64_t)row * key_heads) + head) * KEY_DIM + index]);
		output_bf16[(((uint64_t)row * key_heads) + head) * VALUE_DIM + element] =
			LmFloatToBf16(total);
	}
}

// Short causal convolution over the last KERNEL tokens, per channel.
//
// Both GDN and KDA put one of these before the delta rule. It gives the
// recurrence a few tokens of exact local memory, which the state alone provides
// only approximately - the delta rule compresses everything into a fixed matrix
// and a 4-tap convolution keeps the immediate past verbatim.
//
// The window lives in the same non-growing slot as the state, after it, because
// both are per-sequence and neither grows. That is why kernels/kv.cuh sizes the
// slot from the sum rather than from the state alone.
template<uint32_t THREADS, uint32_t KERNEL>
__global__ __launch_bounds__(THREADS, 1)
void LmCausalConvDecodeKernel(uint16_t *__restrict__ window, const uint32_t *__restrict__ state_index, const uint16_t *__restrict__ input_bf16, const uint16_t *__restrict__ weight_bf16, uint16_t *__restrict__ output_bf16, uint32_t channels, uint32_t rows)
{
	uint32_t row = blockIdx.x,channel = (blockIdx.y * THREADS) + threadIdx.x,tap;
	uint16_t *slot;
	float total = 0.0f;
	if ( row >= rows || channel >= channels )
		return;
	slot = window + ((uint64_t)state_index[row] * channels * KERNEL);
	// Shift the window and admit the new token. A ring buffer would avoid the
	// shift, but KERNEL is 4 and a ring needs a per-sequence head index that
	// every reader has to agree on - the shift is three moves and no state.
	for (tap = 0u; tap + 1u < KERNEL; ++tap)
		slot[(channel * KERNEL) + tap] = slot[(channel * KERNEL) + tap + 1u];
	slot[(channel * KERNEL) + KERNEL - 1u] =
		input_bf16[((uint64_t)row * channels) + channel];
	for (tap = 0u; tap < KERNEL; ++tap)
		total += LmBf16ToFloat(slot[(channel * KERNEL) + tap])
			* LmBf16ToFloat(weight_bf16[(channel * KERNEL) + tap]);
	output_bf16[((uint64_t)row * channels) + channel] = LmFloatToBf16(total);
}
