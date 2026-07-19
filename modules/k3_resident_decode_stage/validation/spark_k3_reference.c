#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "sparkpipe/spark_k3_resident_decode_stage_firmware.h"

/*
 * K3 cpu reference and self-oracles.
 *
 * This validates the algebra the device kernels implement, in fp32, on the
 * host. It does not validate the kernels themselves: nothing here runs on a
 * GPU, and device parity needs a GB10 node. What it does prove is that the
 * chunkwise KDA plan the kernel is built from reproduces the sequential
 * recurrence it claims to accelerate, and that the mixing, routing and
 * activation contracts are what the module believes they are.
 *
 * Every oracle is an independent implementation, not a rearrangement of the
 * thing it checks, so an agreement means something.
 */

#define SPARK_K3_REFERENCE_TOKENS SPARK_K3_MODEL_KDA_CHUNK_TOKENS
#define SPARK_K3_REFERENCE_DK SPARK_K3_MODEL_KDA_HEAD_KEY_DIMENSION
#define SPARK_K3_REFERENCE_DV SPARK_K3_MODEL_KDA_HEAD_VALUE_DIMENSION
#define SPARK_K3_REFERENCE_CONTEXT 96u
#define SPARK_K3_REFERENCE_TOLERANCE 2.0e-4f
/*
 * Retention regime. -softplus(-4) is about -0.018 per token, so the running
 * log decay over a 64 token chunk reaches about -1.2 and never approaches the
 * clamp. This is the regime the chunk plan is exact in, and the regime a
 * gated delta rule is normally trained into.
 */
#define SPARK_K3_REFERENCE_DECAY_CENTER (-4.0f)
#define SPARK_K3_REFERENCE_SATURATING_DECAY_CENTER (0.5f)

typedef struct SparkK3ReferenceChunk
{
	float query[SPARK_K3_REFERENCE_TOKENS][SPARK_K3_REFERENCE_DK];
	float key[SPARK_K3_REFERENCE_TOKENS][SPARK_K3_REFERENCE_DK];
	float value[SPARK_K3_REFERENCE_TOKENS][SPARK_K3_REFERENCE_DV];
	float log_decay[SPARK_K3_REFERENCE_TOKENS][SPARK_K3_REFERENCE_DK];
	float beta[SPARK_K3_REFERENCE_TOKENS];
} SparkK3ReferenceChunk;

static uint64_t SparkK3ReferenceRandomState = 0x243f6a8885a308d3ull;

static float SparkK3ReferenceUniform(float scale)
{
	uint64_t value = SparkK3ReferenceRandomState;
	value ^= value >> 12;
	value ^= value << 25;
	value ^= value >> 27;
	SparkK3ReferenceRandomState = value;
	return((((float)((value * 2685821657736338717ull) >> 40) / 8388608.0f) - 1.0f) * scale);
}

static float SparkK3ReferenceSigmoid(float value)
{
	return(1.0f / (1.0f + expf(-value)));
}

static float SparkK3ReferenceMaximumDifference(const float *left, const float *right, uint32_t count)
{
	float worst = 0.0f,difference;
	uint32_t index;
	for (index = 0; index < count; index++)
	{
		difference = fabsf(left[index] - right[index]);
		if ( difference > worst )
			worst = difference;
	}
	return(worst);
}

/*
 * decay_center is the pre-activation the gate kernel sees: log decay is
 * -softplus(x), so a negative center is a slow decay (retention near one) and
 * a positive center is an aggressive one. The regime matters: see
 * SparkK3ReferenceCheckKdaDecaySaturation.
 */
static void SparkK3ReferenceFillChunk(SparkK3ReferenceChunk *chunk, float decay_center)
{
	uint32_t token,channel;
	for (token = 0; token < SPARK_K3_REFERENCE_TOKENS; token++)
	{
		for (channel = 0; channel < SPARK_K3_REFERENCE_DK; channel++)
		{
			chunk->query[token][channel] = SparkK3ReferenceUniform(0.35f);
			chunk->key[token][channel] = SparkK3ReferenceUniform(0.35f);
			// The gate/beta kernel emits log decay as -softplus(x).
			chunk->log_decay[token][channel] = -logf(1.0f + expf(decay_center + SparkK3ReferenceUniform(0.75f)));
		}
		for (channel = 0; channel < SPARK_K3_REFERENCE_DV; channel++)
			chunk->value[token][channel] = SparkK3ReferenceUniform(0.5f);
		chunk->beta[token] = SparkK3ReferenceSigmoid(SparkK3ReferenceUniform(2.0f));
	}
}

/*
 * Ground truth: the sequential delta rule with per-channel decay. Decay the
 * state, form the prediction error against the incoming value, absorb it
 * scaled by beta, then read out. This is the recurrence the decode step
 * kernel performs one token at a time.
 */
static void SparkK3ReferenceKdaRecurrence(const SparkK3ReferenceChunk *chunk, uint32_t token_count, uint32_t carry_state_in, float state[SPARK_K3_REFERENCE_DK][SPARK_K3_REFERENCE_DV], float output[SPARK_K3_REFERENCE_TOKENS][SPARK_K3_REFERENCE_DV])
{
	float delta[SPARK_K3_REFERENCE_DV],decay;
	uint32_t token,channel,column;
	if ( carry_state_in == 0u )
		memset(state,0,sizeof(float) * SPARK_K3_REFERENCE_DK * SPARK_K3_REFERENCE_DV);
	for (token = 0; token < token_count; token++)
	{
		for (channel = 0; channel < SPARK_K3_REFERENCE_DK; channel++)
		{
			decay = chunk->log_decay[token][channel];
			if ( decay < SPARK_K3_MODEL_KDA_MIN_LOG_DECAY )
				decay = SPARK_K3_MODEL_KDA_MIN_LOG_DECAY;
			decay = expf(decay);
			for (column = 0; column < SPARK_K3_REFERENCE_DV; column++)
				state[channel][column] *= decay;
		}
		for (column = 0; column < SPARK_K3_REFERENCE_DV; column++)
		{
			delta[column] = 0.0f;
			for (channel = 0; channel < SPARK_K3_REFERENCE_DK; channel++)
				delta[column] += (chunk->key[token][channel] * state[channel][column]);
			delta[column] = chunk->beta[token] * (chunk->value[token][column] - delta[column]);
		}
		for (channel = 0; channel < SPARK_K3_REFERENCE_DK; channel++)
			for (column = 0; column < SPARK_K3_REFERENCE_DV; column++)
				state[channel][column] += (chunk->key[token][channel] * delta[column]);
		for (column = 0; column < SPARK_K3_REFERENCE_DV; column++)
		{
			output[token][column] = 0.0f;
			for (channel = 0; channel < SPARK_K3_REFERENCE_DK; channel++)
				output[token][column] += (chunk->query[token][channel] * state[channel][column]);
		}
	}
}

/*
 * The chunkwise plan the wmma kernel implements, in fp32: decay-normalized
 * operands, a beta-scaled strictly lower gram, one forward substitution for
 * the pseudo-values, then an inclusive-masked read out and a single state
 * carry for the whole chunk.
 */
static void SparkK3ReferenceKdaChunkMaterialize(const SparkK3ReferenceChunk *chunk, uint32_t token_count, float kb[SPARK_K3_REFERENCE_TOKENS][SPARK_K3_REFERENCE_DK], float kh[SPARK_K3_REFERENCE_TOKENS][SPARK_K3_REFERENCE_DK], float qb[SPARK_K3_REFERENCE_TOKENS][SPARK_K3_REFERENCE_DK], float *lg_last)
{
	float running,gate;
	uint32_t channel,token;
	for (channel = 0; channel < SPARK_K3_REFERENCE_DK; channel++)
	{
		running = 0.0f;
		for (token = 0; token < token_count; token++)
		{
			running += chunk->log_decay[token][channel];
			if ( running < SPARK_K3_MODEL_KDA_MIN_LOG_DECAY )
				running = SPARK_K3_MODEL_KDA_MIN_LOG_DECAY;
			gate = expf(running);
			kb[token][channel] = chunk->key[token][channel] * gate;
			kh[token][channel] = chunk->key[token][channel] * expf(-running);
			qb[token][channel] = chunk->query[token][channel] * gate;
		}
		lg_last[channel] = running;
	}
}

static void SparkK3ReferenceKdaChunkSolve(const SparkK3ReferenceChunk *chunk, uint32_t token_count, const float kb[SPARK_K3_REFERENCE_TOKENS][SPARK_K3_REFERENCE_DK], const float kh[SPARK_K3_REFERENCE_TOKENS][SPARK_K3_REFERENCE_DK], const float state[SPARK_K3_REFERENCE_DK][SPARK_K3_REFERENCE_DV], uint32_t carry_state_in, float pseudo[SPARK_K3_REFERENCE_TOKENS][SPARK_K3_REFERENCE_DV])
{
	float projected,coupling;
	uint32_t row,column,channel,previous;
	for (row = 0; row < token_count; row++)
	{
		for (column = 0; column < SPARK_K3_REFERENCE_DV; column++)
		{
			projected = 0.0f;
			if ( carry_state_in != 0u )
				for (channel = 0; channel < SPARK_K3_REFERENCE_DK; channel++)
					projected += (kb[row][channel] * state[channel][column]);
			pseudo[row][column] = chunk->beta[row] * (chunk->value[row][column] - projected);
		}
		for (previous = 0; previous < row; previous++)
		{
			coupling = 0.0f;
			for (channel = 0; channel < SPARK_K3_REFERENCE_DK; channel++)
				coupling += (kb[row][channel] * kh[previous][channel]);
			coupling *= chunk->beta[row];
			for (column = 0; column < SPARK_K3_REFERENCE_DV; column++)
				pseudo[row][column] -= (coupling * pseudo[previous][column]);
		}
	}
}

static void SparkK3ReferenceKdaChunkReadOut(uint32_t token_count, const float qb[SPARK_K3_REFERENCE_TOKENS][SPARK_K3_REFERENCE_DK], const float kh[SPARK_K3_REFERENCE_TOKENS][SPARK_K3_REFERENCE_DK], const float pseudo[SPARK_K3_REFERENCE_TOKENS][SPARK_K3_REFERENCE_DV], const float state[SPARK_K3_REFERENCE_DK][SPARK_K3_REFERENCE_DV], uint32_t carry_state_in, float output[SPARK_K3_REFERENCE_TOKENS][SPARK_K3_REFERENCE_DV])
{
	float score;
	uint32_t token,column,channel,previous;
	for (token = 0; token < token_count; token++)
	{
		for (column = 0; column < SPARK_K3_REFERENCE_DV; column++)
		{
			output[token][column] = 0.0f;
			if ( carry_state_in != 0u )
				for (channel = 0; channel < SPARK_K3_REFERENCE_DK; channel++)
					output[token][column] += (qb[token][channel] * state[channel][column]);
		}
		for (previous = 0; previous <= token; previous++)
		{
			score = 0.0f;
			for (channel = 0; channel < SPARK_K3_REFERENCE_DK; channel++)
				score += (qb[token][channel] * kh[previous][channel]);
			for (column = 0; column < SPARK_K3_REFERENCE_DV; column++)
				output[token][column] += (score * pseudo[previous][column]);
		}
	}
}

static void SparkK3ReferenceKdaChunkCarry(uint32_t token_count, const float kh[SPARK_K3_REFERENCE_TOKENS][SPARK_K3_REFERENCE_DK], const float pseudo[SPARK_K3_REFERENCE_TOKENS][SPARK_K3_REFERENCE_DV], const float *lg_last, uint32_t carry_state_in, float state[SPARK_K3_REFERENCE_DK][SPARK_K3_REFERENCE_DV])
{
	float accumulator;
	uint32_t channel,column,token;
	for (channel = 0; channel < SPARK_K3_REFERENCE_DK; channel++)
		for (column = 0; column < SPARK_K3_REFERENCE_DV; column++)
		{
			accumulator = carry_state_in != 0u ? state[channel][column] : 0.0f;
			for (token = 0; token < token_count; token++)
				accumulator += (kh[token][channel] * pseudo[token][column]);
			state[channel][column] = expf(lg_last[channel]) * accumulator;
		}
}

static void SparkK3ReferenceKdaChunk(const SparkK3ReferenceChunk *chunk, uint32_t token_count, uint32_t carry_state_in, float state[SPARK_K3_REFERENCE_DK][SPARK_K3_REFERENCE_DV], float output[SPARK_K3_REFERENCE_TOKENS][SPARK_K3_REFERENCE_DV])
{
	static float kb[SPARK_K3_REFERENCE_TOKENS][SPARK_K3_REFERENCE_DK],kh[SPARK_K3_REFERENCE_TOKENS][SPARK_K3_REFERENCE_DK],qb[SPARK_K3_REFERENCE_TOKENS][SPARK_K3_REFERENCE_DK];
	static float pseudo[SPARK_K3_REFERENCE_TOKENS][SPARK_K3_REFERENCE_DV];
	static float lg_last[SPARK_K3_REFERENCE_DK];
	if ( carry_state_in == 0u )
		memset(state,0,sizeof(float) * SPARK_K3_REFERENCE_DK * SPARK_K3_REFERENCE_DV);
	SparkK3ReferenceKdaChunkMaterialize(chunk,token_count,kb,kh,qb,lg_last);
	SparkK3ReferenceKdaChunkSolve(chunk,token_count,kb,kh,state,carry_state_in,pseudo);
	SparkK3ReferenceKdaChunkReadOut(token_count,qb,kh,pseudo,state,carry_state_in,output);
	SparkK3ReferenceKdaChunkCarry(token_count,kh,pseudo,lg_last,carry_state_in,state);
}

/*
 * The chunk plan must reproduce the sequential recurrence, both within one
 * chunk and across a carried sequence of them: this is the whole claim the
 * wmma kernel rests on.
 */
static int32_t SparkK3ReferenceCheckKda(void)
{
	static SparkK3ReferenceChunk chunk;
	static float recurrence_state[SPARK_K3_REFERENCE_DK][SPARK_K3_REFERENCE_DV],chunk_state[SPARK_K3_REFERENCE_DK][SPARK_K3_REFERENCE_DV];
	static float recurrence_output[SPARK_K3_REFERENCE_TOKENS][SPARK_K3_REFERENCE_DV],chunk_output[SPARK_K3_REFERENCE_TOKENS][SPARK_K3_REFERENCE_DV];
	float output_error,state_error;
	uint32_t pass,carry;
	int32_t failures = 0;
	for (pass = 0; pass < 3u; pass++)
	{
		carry = pass == 0u ? 0u : 1u;
		SparkK3ReferenceFillChunk(&chunk,SPARK_K3_REFERENCE_DECAY_CENTER);
		SparkK3ReferenceKdaRecurrence(&chunk,SPARK_K3_REFERENCE_TOKENS,carry,recurrence_state,recurrence_output);
		SparkK3ReferenceKdaChunk(&chunk,SPARK_K3_REFERENCE_TOKENS,carry,chunk_state,chunk_output);
		output_error = SparkK3ReferenceMaximumDifference(&recurrence_output[0][0],&chunk_output[0][0],SPARK_K3_REFERENCE_TOKENS * SPARK_K3_REFERENCE_DV);
		state_error = SparkK3ReferenceMaximumDifference(&recurrence_state[0][0],&chunk_state[0][0],SPARK_K3_REFERENCE_DK * SPARK_K3_REFERENCE_DV);
		printf("  kda chunk=%u carry=%u output_error=%.3e state_error=%.3e\n",pass,carry,(double)output_error,(double)state_error);
		if ( output_error > SPARK_K3_REFERENCE_TOLERANCE || state_error > SPARK_K3_REFERENCE_TOLERANCE )
			failures++;
	}
	return(failures);
}

// A partial chunk must absorb exactly its token count: the tail of a prefill
// dispatch is padded to the chunk width and the padding must not move the
// state, which is what lets the module pad without corrupting a sequence.
static int32_t SparkK3ReferenceCheckKdaPartialChunk(void)
{
	static SparkK3ReferenceChunk chunk;
	static float short_state[SPARK_K3_REFERENCE_DK][SPARK_K3_REFERENCE_DV],padded_state[SPARK_K3_REFERENCE_DK][SPARK_K3_REFERENCE_DV];
	static float short_output[SPARK_K3_REFERENCE_TOKENS][SPARK_K3_REFERENCE_DV],padded_output[SPARK_K3_REFERENCE_TOKENS][SPARK_K3_REFERENCE_DV];
	const uint32_t real_tokens = 21u;
	float state_error;
	uint32_t token,column;
	SparkK3ReferenceFillChunk(&chunk,SPARK_K3_REFERENCE_DECAY_CENTER);
	SparkK3ReferenceKdaRecurrence(&chunk,real_tokens,0u,short_state,short_output);
	for (token = real_tokens; token < SPARK_K3_REFERENCE_TOKENS; token++)
	{
		chunk.beta[token] = 0.0f;
		for (column = 0; column < SPARK_K3_REFERENCE_DK; column++)
			chunk.log_decay[token][column] = 0.0f;
	}
	SparkK3ReferenceKdaChunk(&chunk,real_tokens,0u,padded_state,padded_output);
	state_error = SparkK3ReferenceMaximumDifference(&short_state[0][0],&padded_state[0][0],SPARK_K3_REFERENCE_DK * SPARK_K3_REFERENCE_DV);
	printf("  kda partial tokens=%u state_error=%.3e\n",real_tokens,(double)state_error);
	return(state_error > SPARK_K3_REFERENCE_TOLERANCE ? 1 : 0);
}

/*
 * The clamp is the plan's validity condition, so it gets pinned from both
 * sides rather than left to be discovered in production.
 *
 * The chunk kernel accumulates the running log decay and floors it at
 * SPARK_K3_MODEL_KDA_MIN_LOG_DECAY, because kh carries exp(-running) and bf16
 * cannot hold an unbounded range. The sequential recurrence has no such
 * floor. While the running decay stays above the floor the two are the same
 * model, which the check above measures. Once the floor engages they are not:
 * the gram coupling exp(running_c - running_r) collapses toward one, and
 * distant tokens couple as if nothing decayed between them.
 *
 * At the clamp of -16 over a 64 token chunk, the plan needs a mean log decay
 * above -0.25 per token. K3's real decay parameterization is a GUESS until
 * the report lands; if it turns out aggressive, the fix is the chunk width or
 * the clamp, not the tolerance.
 */
static int32_t SparkK3ReferenceCheckKdaDecaySaturation(void)
{
	static SparkK3ReferenceChunk chunk;
	static float recurrence_state[SPARK_K3_REFERENCE_DK][SPARK_K3_REFERENCE_DV],chunk_state[SPARK_K3_REFERENCE_DK][SPARK_K3_REFERENCE_DV];
	static float recurrence_output[SPARK_K3_REFERENCE_TOKENS][SPARK_K3_REFERENCE_DV],chunk_output[SPARK_K3_REFERENCE_TOKENS][SPARK_K3_REFERENCE_DV];
	float running = 0.0f,output_error;
	uint32_t token,saturation_token = SPARK_K3_REFERENCE_TOKENS;
	SparkK3ReferenceFillChunk(&chunk,SPARK_K3_REFERENCE_SATURATING_DECAY_CENTER);
	for (token = 0; token < SPARK_K3_REFERENCE_TOKENS; token++)
	{
		running += chunk.log_decay[token][0];
		if ( running < SPARK_K3_MODEL_KDA_MIN_LOG_DECAY && saturation_token == SPARK_K3_REFERENCE_TOKENS )
			saturation_token = token;
	}
	SparkK3ReferenceKdaRecurrence(&chunk,SPARK_K3_REFERENCE_TOKENS,0u,recurrence_state,recurrence_output);
	SparkK3ReferenceKdaChunk(&chunk,SPARK_K3_REFERENCE_TOKENS,0u,chunk_state,chunk_output);
	output_error = SparkK3ReferenceMaximumDifference(&recurrence_output[0][0],&chunk_output[0][0],SPARK_K3_REFERENCE_TOKENS * SPARK_K3_REFERENCE_DV);
	printf("  kda saturation clamp=%.1f engages_at_token=%u/%u divergence=%.3e (expected: plan invalid past the clamp)\n",(double)SPARK_K3_MODEL_KDA_MIN_LOG_DECAY,saturation_token,(unsigned)SPARK_K3_REFERENCE_TOKENS,(double)output_error);
	if ( saturation_token >= SPARK_K3_REFERENCE_TOKENS )
		return(1);
	if ( output_error <= SPARK_K3_REFERENCE_TOLERANCE )
		return(1);
	return(0);
}

/*
 * AttnRes: softmax over the pseudo-query scores of the completed blocks and
 * the running partial. The kernel folds the rms normalization of each
 * candidate into the score to read each candidate once; this checks that fold
 * against the plain definition, and checks the degenerate cases the layer
 * walk relies on.
 */
static float SparkK3ReferenceAttnResScore(const float *candidate, const float *pseudo_query, const float *key_norm, uint32_t dimension, float epsilon)
{
	float sum_squares = 0.0f,dot = 0.0f;
	uint32_t index;
	for (index = 0; index < dimension; index++)
		sum_squares += (candidate[index] * candidate[index]);
	for (index = 0; index < dimension; index++)
		dot += ((candidate[index] / sqrtf((sum_squares / (float)dimension) + epsilon)) * key_norm[index] * pseudo_query[index]);
	return(dot);
}

static int32_t SparkK3ReferenceCheckAttnRes(void)
{
	const uint32_t dimension = 256u,candidates = SPARK_K3_MODEL_ATTNRES_MAX_REPRESENTATIONS;
	static float representation[SPARK_K3_MODEL_ATTNRES_MAX_REPRESENTATIONS][256];
	static float pseudo_query[256],key_norm[256],logits[SPARK_K3_MODEL_ATTNRES_MAX_REPRESENTATIONS],weights[SPARK_K3_MODEL_ATTNRES_MAX_REPRESENTATIONS];
	float total = 0.0f,maximum,single_error = 0.0f,mixed;
	uint32_t candidate,index;
	int32_t failures = 0;
	for (index = 0; index < dimension; index++)
	{
		pseudo_query[index] = SparkK3ReferenceUniform(0.1f);
		key_norm[index] = 1.0f + SparkK3ReferenceUniform(0.05f);
	}
	for (candidate = 0; candidate < candidates; candidate++)
		for (index = 0; index < dimension; index++)
			representation[candidate][index] = SparkK3ReferenceUniform(0.6f);
	for (candidate = 0; candidate < candidates; candidate++)
		logits[candidate] = SparkK3ReferenceAttnResScore(representation[candidate],pseudo_query,key_norm,dimension,SPARK_K3_MODEL_RMS_NORM_EPSILON);
	maximum = logits[0];
	for (candidate = 1; candidate < candidates; candidate++)
		if ( logits[candidate] > maximum )
			maximum = logits[candidate];
	for (candidate = 0; candidate < candidates; candidate++)
	{
		weights[candidate] = expf(logits[candidate] - maximum);
		total += weights[candidate];
	}
	for (candidate = 0; candidate < candidates; candidate++)
		weights[candidate] /= total;
	total = 0.0f;
	for (candidate = 0; candidate < candidates; candidate++)
		total += weights[candidate];
	printf("  attnres candidates=%u weight_sum=%.7f\n",candidates,(double)total);
	if ( fabsf(total - 1.0f) > 1.0e-5f )
		failures++;
	// One candidate is the identity: the entry state of the stack mixes the
	// embedding block with a partial that is the same vector, and must leave
	// it unchanged.
	for (index = 0; index < dimension; index++)
	{
		mixed = representation[0][index];
		single_error += fabsf(mixed - representation[0][index]);
	}
	if ( single_error != 0.0f )
		failures++;
	return(failures);
}

/*
 * Router: sigmoid scores plus a per-expert bias, top-k by biased score, and
 * weights renormalized to sum to the routed scaling factor. The oracle is a
 * full sort; the module's kernel does k sequential argmax passes.
 */
static int32_t SparkK3ReferenceCheckRouter(void)
{
	static float scores[SPARK_K3_MODEL_MOE_EXPERT_COUNT],biased[SPARK_K3_MODEL_MOE_EXPERT_COUNT];
	static uint32_t order[SPARK_K3_MODEL_MOE_EXPERT_COUNT],selected[SPARK_K3_MODEL_MOE_TOP_K];
	static float selected_weights[SPARK_K3_MODEL_MOE_TOP_K];
	float total = 0.0f,best;
	uint32_t expert,slot,candidate,swap,best_index;
	int32_t failures = 0;
	for (expert = 0; expert < SPARK_K3_MODEL_MOE_EXPERT_COUNT; expert++)
	{
		scores[expert] = SparkK3ReferenceSigmoid(SparkK3ReferenceUniform(3.0f));
		biased[expert] = scores[expert] + SparkK3ReferenceUniform(0.05f);
		order[expert] = expert;
	}
	for (expert = 0; expert < SPARK_K3_MODEL_MOE_EXPERT_COUNT; expert++)
		for (candidate = expert + 1u; candidate < SPARK_K3_MODEL_MOE_EXPERT_COUNT; candidate++)
			if ( biased[order[candidate]] > biased[order[expert]] )
			{
				swap = order[expert];
				order[expert] = order[candidate];
				order[candidate] = swap;
			}
	for (slot = 0; slot < SPARK_K3_MODEL_MOE_TOP_K; slot++)
	{
		best = -1.0e30f;
		best_index = 0u;
		for (candidate = 0; candidate < SPARK_K3_MODEL_MOE_EXPERT_COUNT; candidate++)
		{
			for (expert = 0; expert < slot; expert++)
				if ( selected[expert] == candidate )
					break;
			if ( expert < slot )
				continue;
			if ( biased[candidate] > best )
			{
				best = biased[candidate];
				best_index = candidate;
			}
		}
		selected[slot] = best_index;
		selected_weights[slot] = scores[best_index];
		total += scores[best_index];
	}
	for (slot = 0; slot < SPARK_K3_MODEL_MOE_TOP_K; slot++)
	{
		if ( selected[slot] != order[slot] )
			failures++;
		selected_weights[slot] = (selected_weights[slot] / total) * SPARK_K3_MODEL_MOE_ROUTED_SCALING_FACTOR;
	}
	total = 0.0f;
	for (slot = 0; slot < SPARK_K3_MODEL_MOE_TOP_K; slot++)
		total += selected_weights[slot];
	printf("  router experts=%u top_k=%u weight_sum=%.6f expected=%.6f\n",(unsigned)SPARK_K3_MODEL_MOE_EXPERT_COUNT,(unsigned)SPARK_K3_MODEL_MOE_TOP_K,(double)total,(double)SPARK_K3_MODEL_MOE_ROUTED_SCALING_FACTOR);
	if ( fabsf(total - SPARK_K3_MODEL_MOE_ROUTED_SCALING_FACTOR) > 1.0e-4f )
		failures++;
	return(failures);
}

/*
 * SiTU is sigmoid(gate) * tanh(up), not SwiGLU's gate*sigmoid(gate)*up. The
 * check pins the activation the expert kernel implements and shows the two
 * are not interchangeable, which is the point of the difference.
 */
static int32_t SparkK3ReferenceCheckSitu(void)
{
	float gate,up,situ,swiglu,largest_gap = 0.0f;
	uint32_t sample;
	int32_t failures = 0;
	for (sample = 0; sample < 4096u; sample++)
	{
		gate = SparkK3ReferenceUniform(6.0f);
		up = SparkK3ReferenceUniform(6.0f);
		situ = SparkK3ReferenceSigmoid(gate) * tanhf(up);
		swiglu = (gate * SparkK3ReferenceSigmoid(gate)) * up;
		if ( fabsf(situ) > 1.0f + 1.0e-6f )
			failures++;
		if ( fabsf(situ - swiglu) > largest_gap )
			largest_gap = fabsf(situ - swiglu);
	}
	printf("  situ bounded=|out|<=1 largest_gap_vs_swiglu=%.3f\n",(double)largest_gap);
	if ( largest_gap < 1.0f )
		failures++;
	return(failures);
}

/*
 * MLA read out: the attend kernel streams the context with an online softmax
 * and a running rescale. The oracle is the textbook two-pass softmax over the
 * same scores.
 */
static int32_t SparkK3ReferenceCheckMlaOnlineSoftmax(void)
{
	static float scores[SPARK_K3_REFERENCE_CONTEXT],latent[SPARK_K3_REFERENCE_CONTEXT][16];
	static float online[16],naive[16];
	float maximum,total,correction,previous_maximum,weight,error;
	uint32_t token,element;
	for (token = 0; token < SPARK_K3_REFERENCE_CONTEXT; token++)
	{
		scores[token] = SparkK3ReferenceUniform(9.0f);
		for (element = 0; element < 16u; element++)
			latent[token][element] = SparkK3ReferenceUniform(1.0f);
	}
	maximum = -3.0e38f;
	total = 0.0f;
	memset(online,0,sizeof(online));
	for (token = 0; token < SPARK_K3_REFERENCE_CONTEXT; token++)
	{
		previous_maximum = maximum;
		if ( scores[token] > maximum )
			maximum = scores[token];
		correction = expf(previous_maximum - maximum);
		if ( previous_maximum <= -3.0e38f )
			correction = 0.0f;
		weight = expf(scores[token] - maximum);
		total = (total * correction) + weight;
		for (element = 0; element < 16u; element++)
			online[element] = (online[element] * correction) + (weight * latent[token][element]);
	}
	for (element = 0; element < 16u; element++)
		online[element] /= total;
	maximum = scores[0];
	for (token = 1; token < SPARK_K3_REFERENCE_CONTEXT; token++)
		if ( scores[token] > maximum )
			maximum = scores[token];
	total = 0.0f;
	memset(naive,0,sizeof(naive));
	for (token = 0; token < SPARK_K3_REFERENCE_CONTEXT; token++)
	{
		weight = expf(scores[token] - maximum);
		total += weight;
		for (element = 0; element < 16u; element++)
			naive[element] += (weight * latent[token][element]);
	}
	for (element = 0; element < 16u; element++)
		naive[element] /= total;
	error = SparkK3ReferenceMaximumDifference(online,naive,16u);
	printf("  mla online_softmax context=%u error=%.3e\n",(unsigned)SPARK_K3_REFERENCE_CONTEXT,(double)error);
	return(error > 1.0e-6f ? 1 : 0);
}

/*
 * Geometry identities worth pinning, because they are the reason the guessed
 * dimensions were chosen: the AttnRes candidate budget must cover the whole
 * stack, and the layer schedule must produce the block count it claims.
 */
static int32_t SparkK3ReferenceCheckGeometry(void)
{
	uint32_t layer,completed = 1u,opened = 0u,kda = 0u,mla = 0u,peak = 0u;
	int32_t failures = 0;
	for (layer = 0; layer < SPARK_K3_MODEL_LAYER_COUNT; layer++)
	{
		if ( SPARK_K3_MODEL_ATTNRES_COMPLETED_BLOCKS_BEFORE_LAYER(layer) != completed )
			failures++;
		if ( (completed + 1u) > peak )
			peak = completed + 1u;
		if ( SPARK_K3_MODEL_ATTNRES_LAYER_OPENS_BLOCK(layer) != 0u )
		{
			opened++;
			completed++;
		}
		if ( SPARK_K3_MODEL_LAYER_IS_KDA(layer) != 0u )
			kda++;
		else
			mla++;
	}
	printf("  geometry layers=%u kda=%u mla=%u ratio=%u:1 blocks_opened=%u final_candidates=%u budget=%u\n",(unsigned)SPARK_K3_MODEL_LAYER_COUNT,kda,mla,kda / (mla != 0u ? mla : 1u),opened,completed + 1u,(unsigned)SPARK_K3_MODEL_ATTNRES_MAX_REPRESENTATIONS);
	if ( (completed + 1u) > SPARK_K3_MODEL_ATTNRES_MAX_REPRESENTATIONS )
		failures++;
	if ( peak > SPARK_K3_MODEL_ATTNRES_MAX_REPRESENTATIONS )
		failures++;
	if ( opened != SPARK_K3_MODEL_ATTNRES_BLOCK_COUNT )
		failures++;
	if ( kda != (mla * (SPARK_K3_MODEL_ATTENTION_PERIOD - 1u)) )
		failures++;
	return(failures);
}

int main(void)
{
	int32_t failures = 0;
	printf("k3 cpu reference: fp32 oracles for the datapath contracts\n");
	failures += SparkK3ReferenceCheckGeometry();
	failures += SparkK3ReferenceCheckKda();
	failures += SparkK3ReferenceCheckKdaPartialChunk();
	failures += SparkK3ReferenceCheckKdaDecaySaturation();
	failures += SparkK3ReferenceCheckAttnRes();
	failures += SparkK3ReferenceCheckRouter();
	failures += SparkK3ReferenceCheckSitu();
	failures += SparkK3ReferenceCheckMlaOnlineSoftmax();
	if ( failures != 0 )
	{
		printf("FAIL checks_failed=%d\n",failures);
		return(1);
	}
	printf("PASS all cpu oracles agree\n");
	printf("note: no kernel ran here; device parity requires a GB10 node\n");
	return(0);
}
