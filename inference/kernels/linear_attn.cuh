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

#include "inference/kernels/dtype.cuh"
#include "inference/kernels/norm.cuh"
#include <stdint.h>

// The decay logit to a per-channel retention factor. Kimi K3 technical report
// equation 5:
//
//     g = g_min * Sigmoid(exp(A_h) * z)      in (g_min, 0)
//     alpha = exp(g)                          in (exp(g_min), 1)
//
// A_h is a learnable per-head log-scale initialised to 0; g_min is fixed at -5.
//
// THE LOWER BOUND IS THE POINT, and it is a numerical argument rather than a
// modelling one. Gated DeltaNet and Mamba-2 use an UNBOUNDED negative-softplus
// mapping, g = -exp(A) * Softplus(z), which lets the cumulative decay over a
// chunk go to zero and its reciprocal - which the chunkwise form divides by -
// grow without bound. Bounding g below at -5 keeps every retention factor above
// exp(-5), so the cumulative log-decay over a 16-token tile stays within
// (-80, 0) and the reciprocal stays inside BF16 range. That is what lets the
// diagonal tiles use dense Tensor Core matrix multiplications instead of an
// explicit position-pair path.
//
// So a plausible-looking substitution here - softplus for sigmoid, or dropping
// the bound - does not merely change the model. It reintroduces an overflow the
// architecture was designed to remove.
// THE BIAS IS ADDED IN FP32, BEFORE THE SCALE. FlashKDA's reference keeps
// dt_bias as a separate fp32 tensor and adds it after upcasting the bf16 logit:
// folding a bf16 bias into a bf16 logit and upcasting afterwards is a different
// number. The report writes z = W_up W_down x + b_alpha and hides that ordering.
//
// FlashKDA also works in LOG2: it computes g_min * log2(e) * sigmoid(...) and
// exponentiates with ex2, because the chunkwise form takes a cumulative sum of
// these logs and ex2 is the hardware instruction. exp(g_min * s) and
// exp2(g_min * log2(e) * s) are the same number; anyone writing the chunkwise
// path should carry log2 rather than convert per element.
// IF SPECULATION EVER REPLAYS THIS RECURRENCE, IT MUST NOT RECOMPUTE THE GATE.
//
// SGLang's ReplaySSM handles rejected drafts by storing each draft step's raw
// inputs - v, k, the gate value and beta, about 1 KB - instead of snapshotting
// the whole state after every step, which at 96 heads by 128 by 128 would be
// 64 KB per request per layer per head. After the sampler fixes the accepted
// length, one fold kernel replays only the accepted prefix.
//
// Their note on getting it wrong is the part to keep: an early version
// recomputed the gate during the fold with a subtly different formula, "which
// left every output looking correct while the state quietly drifted
// underneath". The fold consumes the gate values the verify kernel stored, not
// a second call to this function.
//
// So LmBoundedDecay is the wrong thing to call twice. A replay path stores what
// this returns and reads it back; it does not evaluate the mapping again, even
// with identical inputs, because __expf is an approximation and two call sites
// that agree today are two call sites that can be edited apart.
static __device__ __forceinline__ float LmBoundedDecay(float logit, float bias, float head_log_scale, float minimum_log_decay)
{
	float scaled = __expf(head_log_scale) * (logit + bias);
	float log_decay = minimum_log_decay * (1.0f / (1.0f + __expf(-scaled)));
	return(__expf(log_decay));
}

// Map a row of decay logits to retention factors, one block per (row, head).
// The logits arrive from a low-rank projection - W_up(W_down(x)) plus a
// per-head bias - so they are PER HEAD PER CHANNEL, not one scalar per head.
template<uint32_t THREADS, uint32_t KEY_DIM>
__global__ __launch_bounds__(THREADS, 1)
void LmBoundedDecayKernel(const uint16_t *__restrict__ logit_bf16, const float *__restrict__ channel_bias, const float *__restrict__ head_log_scale, float *__restrict__ retention, uint32_t heads, float minimum_log_decay, uint32_t rows)
{
	uint32_t row = blockIdx.x,head = blockIdx.y,index;
	uint64_t base;
	if ( row >= rows || head >= heads )
		return;
	base = (((uint64_t)row * heads) + head) * KEY_DIM;
	for (index = threadIdx.x; index < KEY_DIM; index += THREADS)
		retention[base + index] = LmBoundedDecay(LmBf16ToFloat(logit_bf16[base + index]),
			channel_bias[(head * KEY_DIM) + index],head_log_scale[head],minimum_log_decay);
}

// L2-normalise each head's vector. KDA normalises q and k after the convolution
// and before the recurrence; without it the delta rule's k k^T term is not a
// projection and the state grows without bound.
//
// FlashKDA reduces with a warp-shuffle tree and FMA, and its torch reference
// reproduces that exact order rather than calling torch.norm - the accumulation
// order is part of the contract when the result feeds a recurrence. This is a
// plain block reduction, so it will not be bit-identical; that is a difference
// worth knowing about before comparing outputs element-wise.
template<uint32_t THREADS, uint32_t HEAD_DIM>
__global__ __launch_bounds__(THREADS, 1)
void LmL2NormalisePerHeadKernel(uint16_t *__restrict__ rows_bf16, uint32_t heads, uint32_t rows, float epsilon)
{
	// LmBlockSum, not a hand-rolled tree. The first version of this kernel wrote
	// its own shared-memory reduction with a barrier per step - log2(THREADS)
	// barriers where the shuffle path needs one - and duplicated a reduction the
	// tree already had. Same defect twice over: slower and a second copy.
	__shared__ float reduction[THREADS / LM_WARP_LANES];
	uint32_t row = blockIdx.x,head = blockIdx.y,index;
	uint64_t base;
	float total = 0.0f,inverse;
	if ( row >= rows || head >= heads )
		return;
	base = (((uint64_t)row * heads) + head) * HEAD_DIM;
	for (index = threadIdx.x; index < HEAD_DIM; index += THREADS)
	{
		float value = LmBf16ToFloat(rows_bf16[base + index]);
		total += value * value;
	}
	inverse = rsqrtf(LmBlockSum<THREADS>(total,reduction) + epsilon);
	for (index = threadIdx.x; index < HEAD_DIM; index += THREADS)
		rows_bf16[base + index] =
			LmFloatToBf16(LmBf16ToFloat(rows_bf16[base + index]) * inverse);
}

// -- ReplaySSM ------------------------------------------------------------------
//
// Speculation over a recurrent state is the reason K3's decode can go from ~113
// to ~423 tok/s, and the reason it is hard. Verification accepts a prefix of the
// draft, so the state must be rewindable - and a KDA state overwrites itself
// every token. The obvious fix, snapshotting after each draft step, costs
// heads * KEY_DIM * VALUE_DIM per step per layer: at K3's shape 96 * 128 * 128
// bf16 is 3 MB, times gamma+1 steps, times 69 layers, per running request. It
// outgrows the state pool it competes with and caps concurrency.
//
// ReplaySSM stores the INPUTS instead. Verify reads the committed checkpoint and
// never writes it, recording each step's (v, k, gate, beta) - about a kilobyte
// against three megabytes. Once the sampler fixes the accepted length one fold
// replays only the accepted prefix. Rejected drafts are never replayed, so
// rollback is free.
//
// THE FOLD MUST NOT RECOMPUTE THE GATE. SGLang's account of getting this wrong
// is the whole reason this comment is long: an early version recomputed it with
// a subtly different formula, "which left every output looking correct while the
// state quietly drifted underneath". The retention factor is stored by verify
// and read back here. LmBoundedDecay is not called on this path.
struct LmReplayStep
{
	const uint16_t *key_bf16;
	const uint16_t *value_bf16;
	const float *retention;
	const float *write_gate;
};

// Advance the committed state through the accepted prefix, in place. One block
// per (row, head); the recurrence over steps is serial by nature and the work
// inside a step is the same flat pass the decode kernel uses.
template<uint32_t THREADS, uint32_t KEY_DIM, uint32_t VALUE_DIM>
__global__ __launch_bounds__(THREADS, 1)
void LmReplayFoldKernel(uint8_t *__restrict__ state_pool, uint32_t slot_bytes, const uint32_t *__restrict__ state_index, const LmReplayStep *__restrict__ steps, const uint32_t *__restrict__ accepted_length, uint32_t key_heads, uint32_t value_heads_per_key, uint32_t rows)
{
	__shared__ float shared_key[KEY_DIM];
	__shared__ float shared_predicted[VALUE_DIM];
	uint32_t row = blockIdx.x,head = blockIdx.y,step,index,flat;
	float *state;
	if ( row >= rows || head >= key_heads )
		return;
	// THE POOL STRIDE IS A PARAMETER, NOT AN ASSUMPTION.
	//
	// This computed its slot from key_heads * KEY_DIM * VALUE_DIM * 2, hard-wiring
	// both the element size and the absence of anything else in the slot. K3's
	// config declares the state fp32 and bundles the three convolution windows
	// into the same per-request block - sized against SGLang's measured 54 MB at
	// TP=8 - so a pool allocated at that stride and addressed at this one aliases
	// sequence 1 into sequence 0 for any batch above one.
	//
	// Neither side was wrong on its own. They were two expressions of the same
	// quantity that nothing forced to agree, which is the third time this branch
	// has found that shape: the MLA latent standing in for the KDA head dim, and
	// qwen's KV heads before it.
	//
	// slot_bytes now comes from the caller, which reads it from the model's
	// config, and the element size follows from the same place.
	state = (float *)(state_pool + ((uint64_t)state_index[row]
		* slot_bytes)) + ((uint64_t)head * KEY_DIM * VALUE_DIM);
	for (step = 0u; step < accepted_length[row]; ++step)
	{
		const LmReplayStep *input = &steps[step];
		uint64_t head_key = (((uint64_t)row * key_heads) + head) * KEY_DIM;
		uint64_t head_value = (((uint64_t)row * key_heads * value_heads_per_key)
			+ (head * value_heads_per_key)) * VALUE_DIM;
		float beta = input->write_gate[(row * key_heads) + head];
		for (index = threadIdx.x; index < KEY_DIM; index += THREADS)
			shared_key[index] = LmBf16ToFloat(input->key_bf16[head_key + index]);
		__syncthreads();
		// the decode path L2-normalizes k before the update; the fold must
		// replay the SAME arithmetic or the states diverge byte by byte
		if ( threadIdx.x == 0u )
		{
			float kk = 0.0f;
			for (index = 0u; index < KEY_DIM; ++index)
				kk += shared_key[index] * shared_key[index];
			shared_predicted[0] = rsqrtf(kk + 1e-6f);
		}
		__syncthreads();
		for (index = threadIdx.x; index < KEY_DIM; index += THREADS)
			shared_key[index] *= shared_predicted[0];
		__syncthreads();
		for (index = threadIdx.x; index < VALUE_DIM; index += THREADS)
		{
			float total = 0.0f;
			uint32_t channel;
			for (channel = 0u; channel < KEY_DIM; ++channel)
				total += state[(channel * VALUE_DIM) + index]
					* shared_key[channel] * input->retention[head_key + channel];
			shared_predicted[index] = total;
		}
		__syncthreads();
		for (flat = threadIdx.x; flat < KEY_DIM * VALUE_DIM; flat += THREADS)
		{
			uint32_t channel = flat / VALUE_DIM,element = flat % VALUE_DIM;
			float v = LmBf16ToFloat(input->value_bf16[head_value + element]);
			state[flat] = (
				(input->retention[head_key + channel] * state[flat])
				+ (beta * (v - shared_predicted[element]) * shared_key[channel]));
		}
		__syncthreads();
	}
}

// One decode step of the gated delta rule, one block per (row, head).
//
// GROUPED VALUE HEADS. Qwen 3.6 has 16 key heads and 48 value heads, so three
// value heads share each key head's state slice. The state is indexed by key
// head and the value offset selects within it, which is why value_heads_per_key
// is a parameter rather than assumed to be one.
template<uint32_t THREADS, uint32_t KEY_DIM, uint32_t VALUE_DIM>
__global__ __launch_bounds__(THREADS, 1)
void LmDeltaRuleKernel(uint8_t *__restrict__ state_pool, uint32_t slot_bytes, const uint32_t *__restrict__ state_index, const uint32_t *__restrict__ sequence_row_begin, const uint32_t *__restrict__ sequence_row_count, const uint16_t *__restrict__ query_bf16, const uint16_t *__restrict__ key_bf16, const uint16_t *__restrict__ value_bf16, const float *__restrict__ forget_gate, const float *__restrict__ write_gate, uint16_t *__restrict__ output_bf16, uint32_t key_heads, uint32_t value_heads_per_key, uint32_t sequences, uint32_t commit)
{
	// ONE KERNEL FOR DECODE, PREFILL AND VERIFY. The recurrence is serial in
	// the token and parallel in nothing else a batch offers, so the run is the
	// unit: this block owns one sequence's rows for one head, loads the state
	// into shared memory once, streams the run through it, and touches the
	// global slot again only if commit says the run really happened. Decode is
	// a run of one - a null sequence_row_begin means row i IS sequence i - and
	// a run of T is bit-identical to T decode calls, because the state lives in
	// shared memory at exactly the BF16 width the slot stores between calls.
	// That equivalence is a gate, not a comment.
	//
	// Verify is the third caller: DSpark scores a drafted block by running it
	// forward, and a draft must not advance what the sequence remembers.
	// commit == 0 computes every output and abandons the state.
	// THE STATE IS FP32, IN SHARED AND IN THE POOL. The contract
	// (K3_KDA_STATE_ELEMENT_BYTES = 4) always said so; this kernel held it
	// bf16, which both halved the precision of a recurrence that compounds
	// per token and mis-strode slabs sized for four-byte elements. 64 KB
	// per head-block exceeds the static shared limit, so the tile is
	// dynamic - the launch passes KEY_DIM * VALUE_DIM * 4 bytes.
	extern __shared__ float state_s[];
	__shared__ float shared_key[KEY_DIM];
	__shared__ float shared_query[KEY_DIM];
	__shared__ float shared_norm[2];
	__shared__ float shared_predicted[VALUE_DIM];
	uint32_t sequence = blockIdx.x,head = blockIdx.y,index,element,row,begin,end,flat;
	float *state;
	float beta;
	if ( sequence >= sequences || head >= key_heads )
		return;
	begin = sequence_row_begin != 0 ? sequence_row_begin[sequence] : sequence;
	end = sequence_row_begin != 0 ? sequence_row_begin[sequence + 1u] : sequence + 1u;
	if ( sequence_row_count != 0 )
		end = begin + sequence_row_count[sequence];
	// One slot per sequence, never paged: the state does not grow, so its
	// address is the sequence's slot base plus this head's slice.
	state = (float *)(state_pool
		+ ((uint64_t)state_index[sequence] * slot_bytes)
		+ ((uint64_t)head * KEY_DIM * VALUE_DIM * 4u));
	for (flat = threadIdx.x; flat < KEY_DIM * VALUE_DIM; flat += THREADS)
		state_s[flat] = state[flat];
	for (row = begin; row < end; ++row)
	{
		// PER HEAD PER CHANNEL. This read was one scalar per head, which cannot
		// express Kimi K3's channel-wise decay - its retention factor is a d_k
		// vector from a low-rank projection. Qwen 3.6's gated DeltaNet is the
		// same shape, so the width serves both; a per-head caller passes a
		// vector with every channel equal.
		const float *forget = forget_gate + ((((uint64_t)row * key_heads) + head) * KEY_DIM);
		beta = write_gate[(row * key_heads) + head];
		for (index = threadIdx.x; index < KEY_DIM; index += THREADS)
		{
			shared_key[index] = LmBf16ToFloat(key_bf16[(((uint64_t)row * key_heads) + head) * KEY_DIM + index]);
			shared_query[index] = LmBf16ToFloat(query_bf16[(((uint64_t)row * key_heads) + head) * KEY_DIM + index]);
		}
		for (index = threadIdx.x; index < VALUE_DIM; index += THREADS)
			shared_predicted[index] = 0.0f;
		__syncthreads();
		// QK L2-NORM, IN KERNEL, per the reference's
		// use_qk_l2norm_in_kernel=True: q and k are unit vectors before the
		// delta rule sees them. Serial reduce by thread 0 - KEY_DIM is 128
		// and this runs once per row, not per element.
		if ( threadIdx.x == 0u )
		{
			float kk = 0.0f,qq = 0.0f;
			for (index = 0u; index < KEY_DIM; ++index)
			{
				kk += shared_key[index] * shared_key[index];
				qq += shared_query[index] * shared_query[index];
			}
			shared_norm[0] = rsqrtf(kk + 1e-6f);
			shared_norm[1] = rsqrtf(qq + 1e-6f);
		}
		__syncthreads();
		for (index = threadIdx.x; index < KEY_DIM; index += THREADS)
		{
			shared_key[index] *= shared_norm[0];
			shared_query[index] *= shared_norm[1];
		}
		__syncthreads();
		// predicted = S^T k, against the DECAYED state. Report eq. 1 expands to
		// S = Diag(a) S + b k (v^T - k^T Diag(a) S): the retention factor
		// applies BEFORE the prediction, not only to the state that survives
		// it. FlashKDA folds it into the key instead - the same product,
		// written the other way round.
		for (element = threadIdx.x; element < VALUE_DIM; element += THREADS)
		{
			float total = 0.0f;
			for (index = 0u; index < KEY_DIM; ++index)
				total += state_s[(index * VALUE_DIM) + element]
					* shared_key[index] * forget[index];
			shared_predicted[element] = total;
		}
		__syncthreads();
		// S = alpha*S + beta*(v - predicted) k^T, and o = S^T q with the
		// UPDATED state. Updating first is not an optimisation detail: the
		// token attends to its own value, exactly as a softmax token attends
		// to its own key. Flat over the whole outer product - every
		// (channel, element) pair independent, consecutive threads on the
		// contiguous axis.
		{
			uint32_t value_head = head * value_heads_per_key;
			uint64_t value_base =
				(((uint64_t)row * key_heads * value_heads_per_key) + value_head) * VALUE_DIM;
			for (flat = threadIdx.x; flat < KEY_DIM * VALUE_DIM; flat += THREADS)
			{
				uint32_t channel = flat / VALUE_DIM,element_index = flat % VALUE_DIM;
				float v = LmBf16ToFloat(value_bf16[value_base + element_index]);
				float previous = state_s[flat];
				state_s[flat] =
					(forget[channel] * previous)
					+ (beta * (v - shared_predicted[element_index]) * shared_key[channel]);
			}
		}
		__syncthreads();
		for (element = threadIdx.x; element < VALUE_DIM; element += THREADS)
		{
			float total = 0.0f;
			for (index = 0u; index < KEY_DIM; ++index)
				total += state_s[(index * VALUE_DIM) + element]
					* shared_query[index];
			output_bf16[(((uint64_t)row * key_heads) + head) * VALUE_DIM + element] =
				LmFloatToBf16(total);
		}
		__syncthreads();
	}
	if ( commit == 0u )
		return;
	for (flat = threadIdx.x; flat < KEY_DIM * VALUE_DIM; flat += THREADS)
		state[flat] = state_s[flat];
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
enum LmConvActivation
{
	LM_CONV_NONE = 0,
	LM_CONV_SWISH = 1
};

template<uint32_t THREADS, uint32_t KERNEL, uint32_t ACTIVATION, class Weight>
__global__ __launch_bounds__(THREADS, 1)
void LmCausalConvKernel(uint16_t *__restrict__ window, const uint32_t *__restrict__ state_index, const uint32_t *__restrict__ sequence_row_begin, const uint32_t *__restrict__ sequence_row_count, const uint16_t *__restrict__ input_bf16, const Weight *__restrict__ weight, uint16_t *__restrict__ output_bf16, uint32_t channels, uint32_t sequences, uint32_t commit)
{
	// ONE KERNEL FOR DECODE, PREFILL AND VERIFY. A sequence's rows are a run -
	// contiguous, positions ascending - and the window walks the run with the
	// taps held in registers, touching the slot once on the way in and, when
	// commit says so, once on the way out. Decode is a run of one: a null
	// sequence_row_begin means row i IS sequence i, which keeps every decode
	// call site's data exactly as it was. Verify is a run that must not
	// advance what the sequence remembers: commit zero computes every output
	// and abandons the window, which is the whole difference between
	// speculating about tokens and having accepted them.
	uint32_t sequence = blockIdx.x,channel = (blockIdx.y * THREADS) + threadIdx.x;
	uint32_t begin,end,row,tap;
	uint16_t taps[KERNEL];
	uint16_t *slot;
	if ( sequence >= sequences || channel >= channels )
		return;
	begin = sequence_row_begin != 0 ? sequence_row_begin[sequence] : sequence;
	end = sequence_row_begin != 0 ? sequence_row_begin[sequence + 1u] : sequence + 1u;
	// A fold replays only the ACCEPTED prefix of a verify run: the slab keeps
	// its stride, the run keeps its begin, and the count says how much of it
	// really happened. Null means the whole run, which is every other caller.
	if ( sequence_row_count != 0 )
		end = begin + sequence_row_count[sequence];
	slot = window + ((uint64_t)state_index[sequence] * channels * KERNEL);
	for (tap = 0u; tap < KERNEL; ++tap)
		taps[tap] = slot[(channel * KERNEL) + tap];
	for (row = begin; row < end; ++row)
	{
		float total = 0.0f;
		// Shift the window and admit the new token. A ring buffer would avoid
		// the shift, but KERNEL is 4 and a ring needs a head index every reader
		// agrees on - the shift is three register moves and no state.
		for (tap = 0u; tap + 1u < KERNEL; ++tap)
			taps[tap] = taps[tap + 1u];
		taps[KERNEL - 1u] = input_bf16[((uint64_t)row * channels) + channel];
		for (tap = 0u; tap < KERNEL; ++tap)
			total += LmBf16ToFloat(taps[tap])
				* LmScalarToFloat(weight[(channel * KERNEL) + tap]);
		// SWISH, NOT NOTHING. FlashKDA's projections are
		// L2Norm(Swish(ShortConv(Wx))) for q and k and Swish(ShortConv(Wx)) for
		// v; the reference builds ShortConvolution with activation='silu'. A
		// convolution that returns its raw sum runs and is a different model, so
		// the activation is a template parameter with no default.
		if ( ACTIVATION == LM_CONV_SWISH )
			total = total * (1.0f / (1.0f + __expf(-total)));
		output_bf16[((uint64_t)row * channels) + channel] = LmFloatToBf16(total);
	}
	if ( commit == 0u )
		return;
	for (tap = 0u; tap < KERNEL; ++tap)
		slot[(channel * KERNEL) + tap] = taps[tap];
}
