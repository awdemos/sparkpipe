// Run the real KDA decode path on a CPU and print what it produced.
//
// This includes inference/kernels/linear_attn.cuh unmodified. The kernels
// compiled here are the kernels compiled for sm_121a; the only difference is the
// shim that gives them a grid. A number that comes out of this file came out of
// the code that will run on hardware, which is the point - a reimplementation
// in the test would only prove the test agrees with itself.
//
// tests/test_kda_host.py drives this, compares against a transcription of
// FlashKDA's recurrence, and is where the tolerances live.

#include "tests/host_cuda/lm_host_cuda.cuh"

#include <stdio.h>
#include <stdlib.h>

LmHostDim3 blockIdx, threadIdx, blockDim, gridDim;

// Backing for the kernels' dynamic shared memory. On a device the launch sizes
// these; here one fixed buffer per declared name is enough, because one block
// runs at a time.
uint32_t lm_topk_shared[LM_HOST_SHARED_BYTES / sizeof(uint32_t)];
float lm_norm_shared[LM_HOST_SHARED_BYTES / sizeof(float)];

#include "inference/kernels/dtype.cuh"
#include "inference/kernels/linear_attn.cuh"

#define HEADS 2u
#define KEY_DIM 4u
#define VALUE_DIM 4u
#define CONV_KERNEL 4u
#define STEPS 6u
#define THREADS 1u

static uint16_t bf16(float value) { return LmFloatToBf16(value); }
static float f32(uint16_t value) { return LmBf16ToFloat(value); }

static float NextRandom(uint32_t *state)
{
	*state = (*state * 1664525u) + 1013904223u;
	return (float)((*state >> 8) & 0xffffu) / 32768.0f - 1.0f;
}

int main(void)
{
	static uint16_t query[STEPS][HEADS * KEY_DIM];
	static uint16_t key[STEPS][HEADS * KEY_DIM];
	static uint16_t value[STEPS][HEADS * VALUE_DIM];
	static uint16_t decay_logit[STEPS][HEADS * KEY_DIM];
	static uint16_t output[STEPS][HEADS * VALUE_DIM];
	static float retention[HEADS * KEY_DIM];
	static float write_gate[HEADS];
	static float channel_bias[HEADS * KEY_DIM];
	static float head_log_scale[HEADS];
	static uint8_t state_pool[HEADS * KEY_DIM * VALUE_DIM * 2u];
	static uint32_t state_index[1] = { 0u };
	uint32_t seed = 12345u, step, index, head;

	memset(state_pool, 0, sizeof(state_pool));
	for (index = 0u; index < HEADS * KEY_DIM; ++index)
		channel_bias[index] = NextRandom(&seed) * 0.5f;
	for (head = 0u; head < HEADS; ++head)
		head_log_scale[head] = NextRandom(&seed) * 0.3f;
	for (step = 0u; step < STEPS; ++step)
	{
		for (index = 0u; index < HEADS * KEY_DIM; ++index)
		{
			query[step][index] = bf16(NextRandom(&seed));
			key[step][index] = bf16(NextRandom(&seed));
			decay_logit[step][index] = bf16(NextRandom(&seed) * 2.0f);
		}
		for (index = 0u; index < HEADS * VALUE_DIM; ++index)
			value[step][index] = bf16(NextRandom(&seed));
		for (head = 0u; head < HEADS; ++head)
			write_gate[head] = 0.5f + 0.25f * NextRandom(&seed);
	}

	// echo the inputs so the Python side scores the same numbers, not its own
	printf("HEADS %u KEY_DIM %u VALUE_DIM %u STEPS %u GMIN %.9g\n",
		HEADS, KEY_DIM, VALUE_DIM, STEPS, (double)-5.0f);
	for (index = 0u; index < HEADS * KEY_DIM; ++index)
		printf("bias %.9g\n", (double)channel_bias[index]);
	for (head = 0u; head < HEADS; ++head)
		printf("scale %.9g\n", (double)head_log_scale[head]);
	for (step = 0u; step < STEPS; ++step)
	{
		for (index = 0u; index < HEADS * KEY_DIM; ++index)
			printf("q %.9g\nk %.9g\nz %.9g\n",
				(double)f32(query[step][index]), (double)f32(key[step][index]),
				(double)f32(decay_logit[step][index]));
		for (index = 0u; index < HEADS * VALUE_DIM; ++index)
			printf("v %.9g\n", (double)f32(value[step][index]));
		for (head = 0u; head < HEADS; ++head)
			printf("beta %.9g\n", (double)write_gate[head]);
	}

	for (step = 0u; step < STEPS; ++step)
	{
		LM_HOST_LAUNCH(dim3(1u, HEADS),
			(LmBoundedDecayKernel<THREADS, KEY_DIM>(
				decay_logit[step], channel_bias, head_log_scale,
				retention, HEADS, -5.0f, 1u)));
		LM_HOST_LAUNCH(dim3(1u, HEADS),
			(LmDeltaRuleDecodeKernel<THREADS, KEY_DIM, VALUE_DIM>(
				state_pool, state_index, query[step], key[step], value[step],
				retention, write_gate, output[step], HEADS, 1u, 1u)));
		for (index = 0u; index < HEADS * VALUE_DIM; ++index)
			printf("out %.9g\n", (double)f32(output[step][index]));
	}
	return 0;
}
