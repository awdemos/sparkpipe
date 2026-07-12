#include "sparkpipe/spark_glm52_pp13_work_control.h"

#include <string.h>

uint32_t SparkGlm52Pp13WorkControlCalculatePacketBytes(
	uint32_t active_sequence_count)
{
	uint64_t packet_bytes;

	if (active_sequence_count == 0u ||
		active_sequence_count >
			SPARK_GLM52_PP13_WORK_CONTROL_MAX_ACTIVE_SEQUENCE_COUNT)
		return 0u;
	packet_bytes =
		(uint64_t)SPARK_GLM52_PP13_WORK_CONTROL_PACKET_PREFIX_BYTES +
		((uint64_t)active_sequence_count *
		 (uint64_t)SPARK_GLM52_PP13_WORK_CONTROL_LANE_BYTES);
	return packet_bytes <= UINT32_MAX ? (uint32_t)packet_bytes : 0u;
}

static SparkStatus SparkGlm52Pp13WorkControlCopyLaneKvBlocks(
	const SparkGlm52KvBlockTableView *view,
	uint32_t lane_index,
	uint32_t context_token_count,
	SparkGlm52Pp13WorkControlLane *lane)
{
	const uint32_t *block_counts;
	const uint32_t *block_indices;
	uint32_t block_count;
	uint32_t block_index;
	uint64_t source_base;

	if (view == 0 || lane == 0 || lane_index >= view->lane_count ||
		view->block_token_count != SPARK_GLM52_KV_BLOCK_TOKENS ||
		view->lane_stride == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	block_counts = view->host_lane_physical_block_counts != 0 ?
		view->host_lane_physical_block_counts :
		view->lane_physical_block_counts;
	block_indices = view->host_physical_block_indices != 0 ?
		view->host_physical_block_indices : view->physical_block_indices;
	if (block_counts == 0 || block_indices == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	block_count = SparkGlm52Pp13WorkControlBlockCount(
		context_token_count,view->block_token_count);
	if (block_count == 0u ||
		block_count > SPARK_GLM52_PP13_WORK_CONTROL_KV_BLOCK_CAPACITY ||
		block_count > block_counts[lane_index] ||
		block_count > view->lane_stride)
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	source_base = (uint64_t)lane_index * (uint64_t)view->lane_stride;
	for (block_index = 0u; block_index < block_count; ++block_index)
	{
		lane->kv_physical_block_indices[block_index] =
			block_indices[source_base + block_index];
		if (lane->kv_physical_block_indices[block_index] == UINT32_MAX)
			return SPARK_STATUS_VALIDATION_FAILED;
	}
	lane->kv_block_count = block_count;
	return SPARK_STATUS_OK;
}

static void SparkGlm52Pp13WorkControlSetDecodeFlags(
	const SparkGlm52ServingDecodeDispatch *decode_dispatch,
	uint32_t mtp_budget,
	SparkGlm52Pp13WorkControlPacket *packet)
{
	if (mtp_budget != 0u)
		packet->flags |= SPARK_GLM52_PP13_WORK_CONTROL_FLAG_MTP_DRAFT;
	if ((decode_dispatch->request_dispatch->flags &
			SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_DSPARK_TAP_CAPTURE) != 0u)
		packet->flags |=
			SPARK_GLM52_PP13_WORK_CONTROL_FLAG_DSPARK_TAP_CAPTURE;
	if ((decode_dispatch->request_dispatch->flags &
			SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_DSPARK_SPECULATIVE_VERIFY) != 0u)
		packet->flags |=
			SPARK_GLM52_PP13_WORK_CONTROL_FLAG_DSPARK_SPECULATIVE_VERIFY;
	if ((decode_dispatch->request_dispatch->flags &
			SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_MTP_SPECULATIVE_VERIFY) != 0u)
		packet->flags |=
			SPARK_GLM52_PP13_WORK_CONTROL_FLAG_MTP_SPECULATIVE_VERIFY;
}

static SparkStatus SparkGlm52Pp13WorkControlBuildDecodeLanes(
	const SparkGlm52ServingDecodeDispatch *decode_dispatch,
	uint32_t speculative_token_index,
	SparkGlm52Pp13WorkControlPacket *packet)
{
	const SparkGlm52RequestApiDecodeDispatchLaneView *source_lane;
	SparkGlm52Pp13WorkControlLane *lane;
	uint32_t lane_index;
	SparkStatus status;

	for (lane_index = 0u;
		 lane_index < decode_dispatch->active_sequence_count;
		 ++lane_index)
	{
		source_lane = &decode_dispatch->decode_view->lanes[lane_index];
		lane = &packet->lanes[lane_index];
		lane->request_id =
			decode_dispatch->request_dispatch->request_ids[lane_index];
		lane->sequence_id =
			decode_dispatch->request_dispatch->sequence_ids[lane_index];
		lane->sequence_position =
			(uint64_t)source_lane->sequence_position + speculative_token_index;
		lane->context_token_count =
			source_lane->context_token_count + speculative_token_index;
		lane->input_token_id = speculative_token_index == 0u ?
			decode_dispatch->input_token_ids[lane_index] :
			decode_dispatch->speculative_draft_token_ids[lane_index][
				speculative_token_index - 1u];
		status = SparkGlm52Pp13WorkControlCopyLaneKvBlocks(
			decode_dispatch->kv_block_table_view,
			lane_index,
			lane->context_token_count,
			lane);
		if (status != SPARK_STATUS_OK)
			return status;
		if (lane->context_token_count > packet->kv_block_table_token_count)
			packet->kv_block_table_token_count = lane->context_token_count;
	}
	return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52Pp13WorkControlBuildDecodePacket(
	const SparkGlm52ServingDecodeDispatch *decode_dispatch,
	uint32_t speculative_token_index,
	SparkGlm52Pp13WorkControlPacket *packet)
{
	uint32_t speculative_verify;
	uint32_t mtp_budget;
	SparkStatus status;

	if (decode_dispatch == 0 || packet == 0 ||
		decode_dispatch->request_dispatch == 0 ||
		decode_dispatch->decode_view == 0 ||
		decode_dispatch->kv_block_table_view == 0 ||
		decode_dispatch->request_count == 0u ||
		decode_dispatch->request_count != decode_dispatch->active_sequence_count ||
		decode_dispatch->decode_view->lane_count !=
			decode_dispatch->active_sequence_count ||
		decode_dispatch->active_sequence_count >
			SPARK_GLM52_PP13_WORK_CONTROL_MAX_ACTIVE_SEQUENCE_COUNT)
		return SPARK_STATUS_INVALID_ARGUMENT;
	speculative_verify = decode_dispatch->dispatch_kind ==
		SPARK_GLM52_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH;
	if ((speculative_verify != 0u &&
		 (decode_dispatch->active_sequence_count != 1u ||
		  decode_dispatch->speculative_token_count == 0u ||
		  speculative_token_index > decode_dispatch->speculative_token_count)) ||
		(speculative_verify == 0u && speculative_token_index != 0u))
		return SPARK_STATUS_INVALID_ARGUMENT;
	mtp_budget = 0u;
	if (speculative_verify == 0u &&
		(decode_dispatch->request_dispatch->flags &
			SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_MTP_COMMIT) != 0u)
		mtp_budget = decode_dispatch->request_dispatch->mtp_draft_token_budget;
	if (mtp_budget != 0u && decode_dispatch->active_sequence_count != 1u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	memset(packet,0,sizeof(*packet));
	packet->magic = SPARK_GLM52_PP13_WORK_CONTROL_PACKET_MAGIC;
	packet->abi_version = SPARK_GLM52_PP13_WORK_CONTROL_ABI_VERSION;
	packet->active_sequence_count = decode_dispatch->active_sequence_count;
	packet->descriptor_bytes = SparkGlm52Pp13WorkControlCalculatePacketBytes(
		packet->active_sequence_count);
	packet->new_token_count = speculative_verify != 0u ? 1u : mtp_budget + 1u;
	packet->priority = decode_dispatch->request_dispatch->highest_priority;
	packet->block_token_count = SPARK_GLM52_KV_BLOCK_TOKENS;
	packet->max_blocks_per_sequence =
		SPARK_GLM52_PP13_WORK_CONTROL_KV_BLOCK_CAPACITY;
	packet->mtp_draft_token_count = mtp_budget;
	SparkGlm52Pp13WorkControlSetDecodeFlags(decode_dispatch,mtp_budget,packet);
	status = SparkGlm52Pp13WorkControlBuildDecodeLanes(
		decode_dispatch,speculative_token_index,packet);
	if (status != SPARK_STATUS_OK)
		return status;
	packet->request_id = packet->lanes[0u].request_id;
	packet->sequence_id = packet->lanes[0u].sequence_id;
	packet->sequence_position = packet->lanes[0u].sequence_position;
	packet->input_token_id = packet->lanes[0u].input_token_id;
	if (speculative_verify != 0u)
	{
		packet->speculative_token_count =
			decode_dispatch->speculative_token_count;
		packet->speculative_token_index = speculative_token_index;
		memcpy(packet->speculative_draft_token_ids,
			decode_dispatch->speculative_draft_token_ids[0u],
			sizeof(packet->speculative_draft_token_ids));
	}
	return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52Pp13WorkControlBuildPrefillPacket(
	const SparkGlm52PromptPipelinePrefillDispatch *prefill_dispatch,
	uint32_t token_offset,
	SparkGlm52Pp13WorkControlPacket *packet)
{
	SparkGlm52Pp13WorkControlLane *lane;
	uint32_t position;
	SparkStatus status;

	if (prefill_dispatch == 0 || packet == 0 ||
		prefill_dispatch->request_dispatch == 0 ||
		prefill_dispatch->prefill_view == 0 ||
		prefill_dispatch->kv_block_table_view == 0 ||
		prefill_dispatch->lane_count != 1u ||
		prefill_dispatch->active_sequence_count != 1u ||
		prefill_dispatch->prefill_view->lane_count != 1u ||
		token_offset >= prefill_dispatch->prompt_token_count)
		return SPARK_STATUS_INVALID_ARGUMENT;
	position = prefill_dispatch->prompt_token_offset + token_offset;
	memset(packet,0,sizeof(*packet));
	packet->magic = SPARK_GLM52_PP13_WORK_CONTROL_PACKET_MAGIC;
	packet->abi_version = SPARK_GLM52_PP13_WORK_CONTROL_ABI_VERSION;
	packet->descriptor_bytes =
		SparkGlm52Pp13WorkControlCalculatePacketBytes(1u);
	packet->flags = SPARK_GLM52_PP13_WORK_CONTROL_FLAG_PREFILL;
	if ((prefill_dispatch->request_dispatch->flags &
			SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_DSPARK_TAP_CAPTURE) != 0u)
		packet->flags |=
			SPARK_GLM52_PP13_WORK_CONTROL_FLAG_DSPARK_TAP_CAPTURE;
	packet->active_sequence_count = 1u;
	packet->new_token_count = 1u;
	packet->priority = prefill_dispatch->request_dispatch->highest_priority;
	packet->block_token_count = SPARK_GLM52_KV_BLOCK_TOKENS;
	packet->kv_block_table_token_count = position + 1u;
	packet->max_blocks_per_sequence =
		SPARK_GLM52_PP13_WORK_CONTROL_KV_BLOCK_CAPACITY;
	lane = &packet->lanes[0u];
	lane->request_id = prefill_dispatch->request_dispatch->request_ids[0u];
	lane->sequence_id = prefill_dispatch->request_dispatch->sequence_ids[0u];
	lane->sequence_position = position;
	lane->context_token_count = position + 1u;
	lane->input_token_id = prefill_dispatch->host_token_ids[token_offset];
	status = SparkGlm52Pp13WorkControlCopyLaneKvBlocks(
		prefill_dispatch->kv_block_table_view,0u,lane->context_token_count,lane);
	if (status != SPARK_STATUS_OK)
		return status;
	packet->request_id = lane->request_id;
	packet->sequence_id = lane->sequence_id;
	packet->sequence_position = lane->sequence_position;
	packet->input_token_id = lane->input_token_id;
	return SPARK_STATUS_OK;
}

uint32_t SparkGlm52Pp13WorkControlBlockCount(
	uint32_t token_count,
	uint32_t block_token_count)
{
	if (token_count == 0u || block_token_count == 0u)
		return 0u;
	return (token_count + block_token_count - 1u) / block_token_count;
}

SparkStatus SparkGlm52Pp13WorkControlValidatePacket(
	const SparkGlm52Pp13WorkControlPacket *packet,
	uint32_t max_active_sequence_count,
	uint32_t max_pipeline_slot_count)
{
	const SparkGlm52Pp13WorkControlLane *lane;
	uint32_t block_count;
	uint32_t block_index;
	uint32_t lane_index;
	uint32_t maximum_context_token_count;
	uint32_t token_index;
	uint32_t dspark_verify;
	uint32_t mtp_verify;
	uint32_t speculative_verify;

	if (packet == 0 ||
		packet->magic != SPARK_GLM52_PP13_WORK_CONTROL_PACKET_MAGIC ||
		packet->abi_version != SPARK_GLM52_PP13_WORK_CONTROL_ABI_VERSION)
		return SPARK_STATUS_ABI_MISMATCH;
	if ((packet->flags & ~SPARK_GLM52_PP13_WORK_CONTROL_KNOWN_FLAGS) != 0u ||
		packet->request_id == 0u ||
		packet->sequence_id == 0u ||
		packet->active_sequence_count == 0u ||
		packet->active_sequence_count > max_active_sequence_count ||
		packet->new_token_count == 0u ||
		packet->pipeline_slot >= max_pipeline_slot_count ||
		packet->block_token_count == 0u ||
		packet->kv_block_table_token_count == 0u ||
		packet->max_blocks_per_sequence == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (packet->active_sequence_count >
			SPARK_GLM52_PP13_WORK_CONTROL_MAX_ACTIVE_SEQUENCE_COUNT ||
		packet->descriptor_bytes != SparkGlm52Pp13WorkControlCalculatePacketBytes(
			packet->active_sequence_count) ||
		packet->max_blocks_per_sequence >
			SPARK_GLM52_PP13_WORK_CONTROL_KV_BLOCK_CAPACITY)
		return SPARK_STATUS_ABI_MISMATCH;
	if ((packet->flags & SPARK_GLM52_PP13_WORK_CONTROL_FLAG_PREFILL) == 0u &&
		packet->new_token_count >
			SPARK_GLM52_PP13_WORK_CONTROL_MAX_SPECULATIVE_TOKEN_COUNT + 1u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	dspark_verify = (packet->flags &
		SPARK_GLM52_PP13_WORK_CONTROL_FLAG_DSPARK_SPECULATIVE_VERIFY) != 0u;
	mtp_verify = (packet->flags &
		SPARK_GLM52_PP13_WORK_CONTROL_FLAG_MTP_SPECULATIVE_VERIFY) != 0u;
	speculative_verify = dspark_verify | mtp_verify;
	if (dspark_verify != 0u && mtp_verify != 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if ((packet->active_sequence_count != 1u &&
		 (packet->flags &
			(SPARK_GLM52_PP13_WORK_CONTROL_FLAG_PREFILL |
			 SPARK_GLM52_PP13_WORK_CONTROL_FLAG_MTP_DRAFT |
			 SPARK_GLM52_PP13_WORK_CONTROL_FLAG_DSPARK_TAP_CAPTURE |
			 SPARK_GLM52_PP13_WORK_CONTROL_FLAG_DSPARK_SPECULATIVE_VERIFY |
			 SPARK_GLM52_PP13_WORK_CONTROL_FLAG_MTP_SPECULATIVE_VERIFY)) != 0u) ||
		(packet->mtp_draft_token_count != 0u &&
		 packet->active_sequence_count != 1u))
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (speculative_verify != 0u)
	{
		if ((packet->flags & (SPARK_GLM52_PP13_WORK_CONTROL_FLAG_PREFILL |
				SPARK_GLM52_PP13_WORK_CONTROL_FLAG_MTP_DRAFT)) != 0u ||
			packet->new_token_count != 1u ||
			packet->speculative_token_count == 0u ||
			packet->speculative_token_count >
				SPARK_GLM52_PP13_WORK_CONTROL_MAX_SPECULATIVE_TOKEN_COUNT ||
			packet->speculative_token_index >
				packet->speculative_token_count ||
			(dspark_verify != 0u &&
			 (packet->flags &
				SPARK_GLM52_PP13_WORK_CONTROL_FLAG_DSPARK_TAP_CAPTURE) == 0u) ||
			(mtp_verify != 0u &&
			 (packet->flags &
				SPARK_GLM52_PP13_WORK_CONTROL_FLAG_DSPARK_TAP_CAPTURE) != 0u))
			return SPARK_STATUS_INVALID_ARGUMENT;
		for (token_index = 0u;
			 token_index < packet->speculative_token_count;
			 ++token_index)
		{
			if (packet->speculative_draft_token_ids[token_index] >=
				SPARK_GLM52_MODEL_OUTPUT_VOCAB_COUNT)
				return SPARK_STATUS_INVALID_ARGUMENT;
		}
	}
	else
	{
		if (packet->speculative_token_count != 0u ||
			packet->speculative_token_index != 0u)
			return SPARK_STATUS_INVALID_ARGUMENT;
		for (token_index = 0u;
			 token_index <
				SPARK_GLM52_PP13_WORK_CONTROL_MAX_SPECULATIVE_TOKEN_COUNT;
			 ++token_index)
		{
			if (packet->speculative_draft_token_ids[token_index] != 0u)
				return SPARK_STATUS_INVALID_ARGUMENT;
		}
	}
	maximum_context_token_count = 0u;
	for (lane_index = 0u;
		 lane_index < packet->active_sequence_count;
		 ++lane_index)
	{
		lane = &packet->lanes[lane_index];
		block_count = SparkGlm52Pp13WorkControlBlockCount(
			lane->context_token_count,packet->block_token_count);
		if (lane->request_id == 0u || lane->sequence_id == 0u ||
			lane->context_token_count == 0u ||
			lane->context_token_count >
				SPARK_GLM52_PP13_WORK_CONTROL_KV_CONTEXT_TOKEN_CAPACITY ||
			lane->input_token_id >= SPARK_GLM52_MODEL_OUTPUT_VOCAB_COUNT ||
			lane->kv_block_count != block_count ||
			block_count == 0u ||
			block_count > packet->max_blocks_per_sequence ||
			lane->reserved0 != 0u)
			return SPARK_STATUS_INVALID_ARGUMENT;
		for (block_index = 0u; block_index < block_count; ++block_index)
		{
			if (lane->kv_physical_block_indices[block_index] == UINT32_MAX)
				return SPARK_STATUS_INVALID_ARGUMENT;
		}
		if (lane->context_token_count > maximum_context_token_count)
			maximum_context_token_count = lane->context_token_count;
	}
	if (packet->request_id != packet->lanes[0u].request_id ||
		packet->sequence_id != packet->lanes[0u].sequence_id ||
		packet->sequence_position != packet->lanes[0u].sequence_position ||
		packet->input_token_id != packet->lanes[0u].input_token_id ||
		packet->kv_block_table_token_count != maximum_context_token_count)
		return SPARK_STATUS_INVALID_ARGUMENT;
	return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52Pp13WorkControlInitializeKvState(
	SparkGlm52Pp13WorkControlKvState *state,
	uint32_t lane_capacity,
	uint32_t lane_stride,
	uint32_t block_token_count,
	uint32_t physical_block_capacity,
	uint32_t *physical_block_indices,
	uint32_t *lane_physical_block_counts,
	uint8_t *physical_block_states)
{
	uint64_t table_entry_capacity;

	if (state == 0 ||
		lane_capacity == 0u ||
		lane_stride == 0u ||
		block_token_count == 0u ||
		physical_block_capacity == 0u ||
		physical_block_indices == 0 ||
		lane_physical_block_counts == 0 ||
		physical_block_states == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	table_entry_capacity = (uint64_t)lane_capacity * (uint64_t)lane_stride;
	if (table_entry_capacity > UINT32_MAX)
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	memset(state,0,sizeof(*state));
	state->abi_version = SPARK_GLM52_PP13_WORK_CONTROL_ABI_VERSION;
	state->descriptor_bytes = SPARK_GLM52_PP13_WORK_CONTROL_KV_STATE_BYTES;
	state->lane_capacity = lane_capacity;
	state->lane_stride = lane_stride;
	state->block_token_count = block_token_count;
	state->table_entry_capacity = (uint32_t)table_entry_capacity;
	state->physical_block_capacity = physical_block_capacity;
	state->physical_block_indices = physical_block_indices;
	state->lane_physical_block_counts = lane_physical_block_counts;
	state->physical_block_states = physical_block_states;
	memset(state->physical_block_states,SPARK_GLM52_PP13_KV_ENTRY_MISSING,
		state->physical_block_capacity * sizeof(state->physical_block_states[0]));
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13WorkControlValidateKvState(
	const SparkGlm52Pp13WorkControlPacket *packet,
	SparkGlm52Pp13WorkControlKvState *state)
{
	if (packet == 0 || state == 0 ||
		state->abi_version != SPARK_GLM52_PP13_WORK_CONTROL_ABI_VERSION ||
		state->descriptor_bytes != SPARK_GLM52_PP13_WORK_CONTROL_KV_STATE_BYTES ||
		state->physical_block_indices == 0 ||
		state->lane_physical_block_counts == 0 ||
		state->physical_block_states == 0 ||
		state->lane_capacity == 0u ||
		state->lane_stride == 0u ||
		state->block_token_count == 0u ||
		state->table_entry_capacity == 0u ||
		state->physical_block_capacity == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (SparkGlm52Pp13WorkControlValidatePacket(
			packet,
			state->lane_capacity,
			UINT32_MAX) != SPARK_STATUS_OK)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (packet->block_token_count != state->block_token_count ||
		packet->max_blocks_per_sequence > state->lane_stride ||
		packet->active_sequence_count > state->lane_capacity)
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	return SPARK_STATUS_OK;
}

static void SparkGlm52Pp13WorkControlResetReadinessCounts(
	SparkGlm52Pp13WorkControlKvState *state)
{
	state->missing_block_count = 0u;
	state->in_flight_block_count = 0u;
	state->resident_block_count = 0u;
}

static void SparkGlm52Pp13WorkControlAccountReadiness(
	SparkGlm52Pp13WorkControlKvState *state,
	uint8_t entry_state)
{
	if (entry_state == SPARK_GLM52_PP13_KV_ENTRY_RESIDENT)
		state->resident_block_count += 1u;
	else if (entry_state == SPARK_GLM52_PP13_KV_ENTRY_IN_FLIGHT)
		state->in_flight_block_count += 1u;
	else
		state->missing_block_count += 1u;
}

static SparkStatus SparkGlm52Pp13WorkControlMarkTable(
	const SparkGlm52Pp13WorkControlPacket *packet,
	SparkGlm52Pp13WorkControlKvState *state,
	uint8_t entry_state)
{
	uint32_t lane_index;
	uint32_t block_index;
	uint64_t table_index;
	uint64_t physical_block_index;
	SparkStatus status;

	status = SparkGlm52Pp13WorkControlValidateKvState(packet,state);
	if (status != SPARK_STATUS_OK)
		return status;
	for (lane_index = 0u; lane_index < packet->active_sequence_count; ++lane_index)
	{
		for (block_index = 0u;
			 block_index < packet->lanes[lane_index].kv_block_count;
			 ++block_index)
		{
			table_index =
				((uint64_t)lane_index * (uint64_t)state->lane_stride) +
				(uint64_t)block_index;
			physical_block_index =
				packet->lanes[lane_index].kv_physical_block_indices[block_index];
			if (table_index >= state->table_entry_capacity)
				return SPARK_STATUS_CAPACITY_EXCEEDED;
			if (physical_block_index >= state->physical_block_capacity)
				return SPARK_STATUS_CAPACITY_EXCEEDED;
			state->physical_block_states[physical_block_index] = entry_state;
		}
	}
	return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52Pp13WorkControlBuildHostKvBlockTable(
	const SparkGlm52Pp13WorkControlPacket *packet,
	SparkGlm52Pp13WorkControlKvState *state,
	SparkGlm52KvBlockTableView *view)
{
	uint32_t lane_index;
	uint32_t block_index;
	uint32_t block_count;
	uint8_t entry_state;
	uint64_t table_index;
	uint64_t physical_block_index;
	SparkStatus status;

	if (view == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	status = SparkGlm52Pp13WorkControlValidateKvState(packet,state);
	if (status != SPARK_STATUS_OK)
		return status;
	memset(
		state->physical_block_indices,
		0xff,
		(uint64_t)packet->active_sequence_count * state->lane_stride *
			sizeof(state->physical_block_indices[0]));
	memset(
		state->lane_physical_block_counts,
		0,
		packet->active_sequence_count *
			sizeof(state->lane_physical_block_counts[0]));
	SparkGlm52Pp13WorkControlResetReadinessCounts(state);
	for (lane_index = 0u; lane_index < packet->active_sequence_count; ++lane_index)
	{
		block_count = packet->lanes[lane_index].kv_block_count;
		state->lane_physical_block_counts[lane_index] = block_count;
		for (block_index = 0u; block_index < block_count; ++block_index)
		{
			table_index =
				((uint64_t)lane_index * (uint64_t)state->lane_stride) +
				(uint64_t)block_index;
			physical_block_index =
				packet->lanes[lane_index].kv_physical_block_indices[block_index];
			if (table_index >= state->table_entry_capacity)
				return SPARK_STATUS_CAPACITY_EXCEEDED;
			if (physical_block_index >= state->physical_block_capacity)
				return SPARK_STATUS_CAPACITY_EXCEEDED;
			entry_state = state->physical_block_states[physical_block_index];
			if ((packet->flags & SPARK_GLM52_PP13_WORK_CONTROL_FLAG_PREFILL) != 0u)
			{
				if (entry_state == SPARK_GLM52_PP13_KV_ENTRY_MISSING)
					entry_state = SPARK_GLM52_PP13_KV_ENTRY_IN_FLIGHT;
			}
			else if (block_index + 1u < block_count &&
				entry_state != SPARK_GLM52_PP13_KV_ENTRY_RESIDENT)
			{
				SparkGlm52Pp13WorkControlAccountReadiness(state,entry_state);
				return SPARK_STATUS_BUSY;
			}
			else if (block_index + 1u == block_count &&
				entry_state == SPARK_GLM52_PP13_KV_ENTRY_MISSING)
			{
				entry_state = SPARK_GLM52_PP13_KV_ENTRY_IN_FLIGHT;
			}
			state->physical_block_states[physical_block_index] = entry_state;
			SparkGlm52Pp13WorkControlAccountReadiness(state,entry_state);
			state->physical_block_indices[table_index] =
				(uint32_t)physical_block_index;
		}
	}
	memset(view,0,sizeof(*view));
	view->abi_version = SPARK_GLM52_KV_CACHE_ABI_VERSION;
	view->descriptor_bytes = SPARK_GLM52_KV_BLOCK_TABLE_VIEW_DESCRIPTOR_BYTES;
	view->block_token_count = packet->block_token_count;
	view->lane_count = packet->active_sequence_count;
	view->lane_stride = state->lane_stride;
	view->lane_capacity = state->lane_capacity;
	view->physical_block_indices = state->physical_block_indices;
	view->lane_physical_block_counts = state->lane_physical_block_counts;
	view->host_physical_block_indices = state->physical_block_indices;
	view->host_lane_physical_block_counts = state->lane_physical_block_counts;
	return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52Pp13WorkControlCommitHostKvBlockTable(
	const SparkGlm52Pp13WorkControlPacket *packet,
	SparkGlm52Pp13WorkControlKvState *state)
{
	return SparkGlm52Pp13WorkControlMarkTable(
		packet,
		state,
		SPARK_GLM52_PP13_KV_ENTRY_RESIDENT);
}

SparkStatus SparkGlm52Pp13WorkControlCancelHostKvBlockTable(
	const SparkGlm52Pp13WorkControlPacket *packet,
	SparkGlm52Pp13WorkControlKvState *state)
{
	uint32_t lane_index;
	uint32_t block_index;
	uint64_t physical_block_index;
	SparkStatus status;

	status = SparkGlm52Pp13WorkControlValidateKvState(packet,state);
	if (status != SPARK_STATUS_OK)
		return status;
	for (lane_index = 0u; lane_index < packet->active_sequence_count; ++lane_index)
	{
		for (block_index = 0u;
			 block_index < packet->lanes[lane_index].kv_block_count;
			 ++block_index)
		{
			physical_block_index =
				packet->lanes[lane_index].kv_physical_block_indices[block_index];
			if (physical_block_index >= state->physical_block_capacity)
				return SPARK_STATUS_CAPACITY_EXCEEDED;
			if (state->physical_block_states[physical_block_index] ==
				SPARK_GLM52_PP13_KV_ENTRY_IN_FLIGHT)
				state->physical_block_states[physical_block_index] =
					SPARK_GLM52_PP13_KV_ENTRY_MISSING;
		}
	}
	return SPARK_STATUS_OK;
}
