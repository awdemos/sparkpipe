// The engine, driven: two requests through admission, chunked prefill, mixed
// prefill+decode steps, EOS, and slot reuse by a third. Prints every plan so
// the python gate can hold the scheduler to its contract - the same contract
// the slice kernels enforce on their side: rows sorted by sequence, positions
// ascending in a run, context_length counting every stored row.

#include <stdio.h>
#include "inference/llms/kimi_k3/engine.h"

#define SLOTS 2u
#define BUDGET 4u
#define STEP_MAX 16u

static void PrintStep(uint32_t index, const struct K3EngineStep *step)
{
	uint32_t s,r;
	printf("step %u rows %u sequences %u\n", index, step->rows, step->sequences);
	for (s = 0u; s < step->sequences; ++s)
	{
		printf("  seq %u slot %u request %llu run %u..%u context %u logits %d\n",
			s, step->slot[s], (unsigned long long)step->request_id[s],
			step->sequence_row_begin[s], step->sequence_row_begin[s + 1u],
			step->context_length[s],
			step->logits_row[s] == K3_ENGINE_NO_LOGITS
				? -1 : (int)step->logits_row[s]);
		for (r = step->sequence_row_begin[s]; r < step->sequence_row_begin[s + 1u]; ++r)
			printf("    row %u token %u position %u slot %u\n",
				r, step->token[r], step->position[r], step->sequence_of_row[r]);
	}
}

int main(void)
{
	static struct K3Engine engine;
	static struct K3EngineRequest requests[4];
	static uint32_t slots[SLOTS];
	static uint32_t token[STEP_MAX], position[STEP_MAX], seq_of_row[STEP_MAX];
	static uint32_t run_begin[SLOTS + 1u], slot[SLOTS], context[SLOTS];
	static uint32_t logits[SLOTS];
	static uint64_t request_id[SLOTS];
	static uint32_t prompt_a[5] = { 11u, 12u, 13u, 14u, 15u };
	static uint32_t prompt_b[3] = { 21u, 22u, 23u };
	static uint32_t prompt_c[2] = { 31u, 32u };
	static uint32_t out_a[6], out_b[4], out_c[2];
	static uint32_t sampled[SLOTS];
	struct K3EngineStep step;
	uint32_t index,s,next = 100u;
	int32_t rows;
	memset(&step, 0, sizeof(step));
	step.token = token; step.position = position; step.sequence_of_row = seq_of_row;
	step.sequence_row_begin = run_begin; step.slot = slot; step.request_id = request_id;
	step.context_length = context; step.logits_row = logits;
	if ( K3EngineInit(&engine, requests, 4u, slots, SLOTS, BUDGET) != K3_ENGINE_OK )
		return 1;
	printf("submit a %lld\n", (long long)K3EngineSubmit(&engine, prompt_a, 5u, 6u, out_a));
	printf("submit b %lld\n", (long long)K3EngineSubmit(&engine, prompt_b, 3u, 4u, out_b));
	// The third arrives before any slot frees: it must queue, then take the
	// first slot a finished request abandons.
	printf("submit c %lld\n", (long long)K3EngineSubmit(&engine, prompt_c, 2u, 2u, out_c));
	for (index = 0u; index < 12u; ++index)
	{
		// After request a's first sampled token, hand it a three-token draft:
		// the next plan must be a verify-only step - one run of four rows at
		// ascending positions, logits at the run's head - and its resolution
		// (two accepted plus the bonus) must land exactly three tokens.
		if ( index == 3u )
		{
			static uint32_t draft[3] = { 201u, 202u, 203u };
			if ( K3EngineSubmitDraft(&engine, 1u, draft, 3u) != K3_ENGINE_OK )
				return 4;
		}
		rows = K3EnginePlanStep(&engine, &step);
		if ( rows < 0 )
			return 2;
		if ( rows == 0 )
		{
			printf("idle at %u\n", index);
			break;
		}
		PrintStep(index, &step);
		if ( step.verify != 0u )
		{
			static uint32_t accepted[SLOTS], bonus[SLOTS];
			printf("verify step %u\n", index);
			for (s = 0u; s < step.sequences; ++s)
			{
				accepted[s] = 2u;
				bonus[s] = next++;
				// The bonus token occupies the first rejected position and
				// was never forwarded; the next decode row re-runs it there.
				printf("verify_next %llu %u\n",
					(unsigned long long)step.request_id[s],
					step.position[step.sequence_row_begin[s]] + 1u + accepted[s]);
			}
			if ( K3EngineCommitVerify(&engine, &step, accepted, bonus, 7u) != K3_ENGINE_OK )
				return 5;
			continue;
		}
		for (s = 0u; s < step.sequences; ++s)
		{
			sampled[s] = 0u;
			if ( step.logits_row[s] == K3_ENGINE_NO_LOGITS )
				continue;
			// Request b's second token is EOS, ending it under budget; the
			// rest count up so the transcript shows who sampled what.
			sampled[s] = (step.request_id[s] == 2u
				&& engine.requests[engine.slot_request[step.slot[s]]].generated == 1u)
				? 7u : next++;
		}
		if ( K3EngineCommitStep(&engine, &step, sampled, 7u) != K3_ENGINE_OK )
			return 3;
	}
	printf("out_a %u %u %u %u %u %u\n", out_a[0], out_a[1], out_a[2], out_a[3], out_a[4], out_a[5]);
	printf("out_b %u %u\n", out_b[0], out_b[1]);
	printf("out_c %u %u\n", out_c[0], out_c[1]);
	return 0;
}
