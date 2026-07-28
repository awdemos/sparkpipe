#pragma once

// The K3 serving engine: requests in, slice steps out.
//
// This is the host half of actual usage - accept a request, hold it in a
// queue, admit it when a sequence slot frees, cut its prompt into chunks that
// share steps with everyone else's decode, and advance it token by token until
// EOS or its budget. It is pure planning: no CUDA, no allocation, no model.
// One call plans a step as the exact arrays K3StageSlice and the K3 buffers
// consume - rows sorted by sequence, sequence_row_begin a prefix, positions
// ascending within a run, context_length counting every stored row - and one
// call commits the step's sampled tokens and moves the machine forward. The
// driver owns the device copies and the sampler; the engine owns the truth
// about who is where.
//
// CONTINUOUS BATCHING IS THE POLICY, NOT A MODE. Every step carries every
// decoding sequence (one row each) and then spends whatever row budget
// remains on the oldest prefilling sequence's next chunk. A decode-only step
// and a prefill-only step are both just this rule with one side empty.
//
// Storage is the caller's, sized by the two capacity numbers in K3EngineInit,
// because a serving process knows its memory and a header does not. No malloc
// anywhere, per the house rule and because an allocator in the admission path
// is a latency cliff waiting for load.

#include <stdint.h>
#include <string.h>

#define K3_ENGINE_OK               0
#define K3_ENGINE_ERR_NULL       -70
#define K3_ENGINE_ERR_CAPACITY   -71
#define K3_ENGINE_ERR_PROMPT     -72
#define K3_ENGINE_ERR_STATE      -73
#define K3_ENGINE_ERR_SAMPLE     -74

// What a sequence is doing. VERIFY is DSpark's lane: a drafted block planned
// as a run the layer executes with commit off, so the state never learns what
// the sampler later rejects. The planner treats it as prefill-shaped rows
// whose tokens came from the drafter rather than the user.
#define K3_SEQ_FREE      0u
#define K3_SEQ_QUEUED    1u
#define K3_SEQ_PREFILL   2u
#define K3_SEQ_DECODE    3u
#define K3_SEQ_DONE      4u

struct K3EngineRequest
{
	uint64_t id;
	const uint32_t *prompt;
	uint32_t prompt_length;
	uint32_t max_new;
	uint32_t generated;
	uint32_t prefilled;
	uint32_t slot;
	uint32_t state;
	uint32_t *output;
};

struct K3EngineStep
{
	// Per row, in plan order: the token to embed and its position.
	uint32_t *token;
	uint32_t *position;
	uint32_t *sequence_of_row;
	// Per sequence in plan order: the run prefix, the slot, the request, the
	// stored context after this step's rows land, and which row's logits the
	// sampler must read - K3_ENGINE_NO_LOGITS for a chunk that is not yet at
	// the prompt's end and therefore predicts nothing anyone keeps.
	uint32_t *sequence_row_begin;
	uint32_t *slot;
	uint64_t *request_id;
	uint32_t *context_length;
	uint32_t *logits_row;
	uint32_t rows;
	uint32_t sequences;
};

#define K3_ENGINE_NO_LOGITS 0xffffffffu

struct K3Engine
{
	struct K3EngineRequest *requests;
	uint32_t request_capacity;
	uint32_t slot_capacity;
	uint32_t row_budget;
	uint64_t next_id;
	uint32_t *slot_request;
};

static int32_t K3EngineInit(struct K3Engine *engine, struct K3EngineRequest *request_storage, uint32_t request_capacity, uint32_t *slot_storage, uint32_t slot_capacity, uint32_t row_budget)
{
	uint32_t index;
	if ( engine == 0 || request_storage == 0 || slot_storage == 0 )
		return(K3_ENGINE_ERR_NULL);
	if ( request_capacity == 0u || slot_capacity == 0u || row_budget == 0u )
		return(K3_ENGINE_ERR_CAPACITY);
	memset(engine,0,sizeof(*engine));
	memset(request_storage,0,(size_t)request_capacity * sizeof(*request_storage));
	engine->requests = request_storage;
	engine->request_capacity = request_capacity;
	engine->slot_capacity = slot_capacity;
	engine->row_budget = row_budget;
	engine->next_id = 1u;
	engine->slot_request = slot_storage;
	for (index = 0u; index < slot_capacity; ++index)
		slot_storage[index] = 0xffffffffu;
	return(K3_ENGINE_OK);
}

// Accept a request. The prompt pointer must outlive the request; the output
// array must hold max_new tokens. Returns the id, or a negative status.
static int64_t K3EngineSubmit(struct K3Engine *engine, const uint32_t *prompt, uint32_t prompt_length, uint32_t max_new, uint32_t *output)
{
	struct K3EngineRequest *request = 0;
	uint32_t index;
	if ( engine == 0 || prompt == 0 || output == 0 )
		return(K3_ENGINE_ERR_NULL);
	if ( prompt_length == 0u || max_new == 0u )
		return(K3_ENGINE_ERR_PROMPT);
	for (index = 0u; index < engine->request_capacity; ++index)
		if ( engine->requests[index].state == K3_SEQ_FREE )
		{
			request = &engine->requests[index];
			break;
		}
	if ( request == 0 )
		return(K3_ENGINE_ERR_CAPACITY);
	memset(request,0,sizeof(*request));
	request->id = engine->next_id++;
	request->prompt = prompt;
	request->prompt_length = prompt_length;
	request->max_new = max_new;
	request->state = K3_SEQ_QUEUED;
	request->slot = 0xffffffffu;
	request->output = output;
	return((int64_t)request->id);
}

// Admission: the oldest queued request takes the lowest free slot. Called by
// the planner so a slot freed by this step's commit is refilled by the next
// plan, never left idle while the queue waits.
static void K3EngineAdmit(struct K3Engine *engine)
{
	struct K3EngineRequest *oldest;
	uint32_t slot,index;
	for (slot = 0u; slot < engine->slot_capacity; ++slot)
	{
		if ( engine->slot_request[slot] != 0xffffffffu )
			continue;
		oldest = 0;
		for (index = 0u; index < engine->request_capacity; ++index)
			if ( engine->requests[index].state == K3_SEQ_QUEUED
				&& (oldest == 0 || engine->requests[index].id < oldest->id) )
				oldest = &engine->requests[index];
		if ( oldest == 0 )
			return;
		oldest->slot = slot;
		oldest->state = K3_SEQ_PREFILL;
		engine->slot_request[slot] = (uint32_t)(oldest - engine->requests);
	}
}

// One sequence's contribution to the plan: DECODE is one row carrying the
// last sampled (or last prompt) token; PREFILL is the next chunk of the
// prompt. Rows land contiguously, positions ascend, and only a run that
// reaches the prompt's end asks for logits.
static uint32_t K3EnginePlanSequence(const struct K3EngineRequest *request, struct K3EngineStep *step, uint32_t budget)
{
	uint32_t sequence = step->sequences,base = step->rows,count,index,position;
	if ( request->state == K3_SEQ_DECODE )
	{
		count = 1u;
		position = request->prompt_length + request->generated - 1u;
		step->token[base] = request->generated == 0u
			? request->prompt[request->prompt_length - 1u]
			: request->output[request->generated - 1u];
		step->position[base] = position;
		step->logits_row[sequence] = base;
	}
	else
	{
		count = request->prompt_length - request->prefilled;
		// The final prompt token belongs to decode: its forward is the one
		// that predicts the first new token, so prefill stops one short and
		// the decode row carries it. A prompt of one token prefills nothing.
		count = count > 0u ? count - 1u : 0u;
		count = count > budget ? budget : count;
		for (index = 0u; index < count; ++index)
		{
			step->token[base + index] = request->prompt[request->prefilled + index];
			step->position[base + index] = request->prefilled + index;
		}
		step->logits_row[sequence] = K3_ENGINE_NO_LOGITS;
		position = request->prefilled + count - 1u;
	}
	for (index = 0u; index < count; ++index)
		step->sequence_of_row[base + index] = request->slot;
	step->sequence_row_begin[sequence + 1u] = base + count;
	step->slot[sequence] = request->slot;
	step->request_id[sequence] = request->id;
	step->context_length[sequence] = position + 1u;
	step->rows = base + count;
	step->sequences = sequence + 1u;
	return(count);
}

// Plan one step: admit, then every decoding sequence, then the oldest
// prefilling sequence's next chunk into whatever budget remains. Returns the
// row count; zero means nothing to do.
static int32_t K3EnginePlanStep(struct K3Engine *engine, struct K3EngineStep *step)
{
	struct K3EngineRequest *request,*oldest;
	uint32_t index;
	if ( engine == 0 || step == 0 )
		return(K3_ENGINE_ERR_NULL);
	K3EngineAdmit(engine);
	step->rows = 0u;
	step->sequences = 0u;
	step->sequence_row_begin[0] = 0u;
	for (index = 0u; index < engine->request_capacity; ++index)
	{
		request = &engine->requests[index];
		if ( request->state == K3_SEQ_DECODE && step->rows < engine->row_budget )
			K3EnginePlanSequence(request,step,1u);
	}
	oldest = 0;
	for (index = 0u; index < engine->request_capacity; ++index)
	{
		request = &engine->requests[index];
		if ( request->state == K3_SEQ_PREFILL
			&& (oldest == 0 || request->id < oldest->id) )
			oldest = request;
	}
	if ( oldest != 0 && step->rows < engine->row_budget )
		K3EnginePlanSequence(oldest,step,engine->row_budget - step->rows);
	return((int32_t)step->rows);
}

// Commit the step: sampled[sequence] must hold a token for every sequence
// whose logits_row was real, and is ignored for the rest. eos ends a request
// early; the budget ends it on time; either way the slot frees for the next
// admission.
static int32_t K3EngineCommitStep(struct K3Engine *engine, const struct K3EngineStep *step, const uint32_t *sampled, uint32_t eos_token)
{
	struct K3EngineRequest *request;
	uint32_t sequence,token;
	if ( engine == 0 || step == 0 || sampled == 0 )
		return(K3_ENGINE_ERR_NULL);
	for (sequence = 0u; sequence < step->sequences; ++sequence)
	{
		request = &engine->requests[engine->slot_request[step->slot[sequence]]];
		if ( request->state == K3_SEQ_PREFILL )
		{
			request->prefilled += step->sequence_row_begin[sequence + 1u]
				- step->sequence_row_begin[sequence];
			if ( request->prefilled + 1u >= request->prompt_length )
				request->state = K3_SEQ_DECODE;
			continue;
		}
		if ( request->state != K3_SEQ_DECODE )
			return(K3_ENGINE_ERR_STATE);
		if ( step->logits_row[sequence] == K3_ENGINE_NO_LOGITS )
			return(K3_ENGINE_ERR_SAMPLE);
		token = sampled[sequence];
		request->output[request->generated++] = token;
		if ( token == eos_token || request->generated >= request->max_new )
		{
			request->state = K3_SEQ_DONE;
			engine->slot_request[request->slot] = 0xffffffffu;
		}
	}
	return(K3_ENGINE_OK);
}
