#include <assert.h>
#include <string.h>

#include "sparkpipe/spark_glm52_pp13_work_control.h"

static void SparkTestInitializeWorkPacket(
	SparkGlm52Pp13WorkControlPacket *packet)
{
	uint32_t block_index;
	uint32_t lane_index;

	memset(packet,0,sizeof(*packet));
	packet->magic = SPARK_GLM52_PP13_WORK_CONTROL_PACKET_MAGIC;
	packet->abi_version = SPARK_GLM52_PP13_WORK_CONTROL_ABI_VERSION;
	packet->active_sequence_count = 4u;
	packet->descriptor_bytes =
		SparkGlm52Pp13WorkControlCalculatePacketBytes(
			packet->active_sequence_count);
	packet->request_id = 7u;
	packet->sequence_id = 11u;
	packet->sequence_position = 128u;
	packet->new_token_count = 1u;
	packet->pipeline_slot = 0u;
	packet->block_token_count = SPARK_GLM52_KV_BLOCK_TOKENS;
	packet->kv_block_table_token_count = 129u;
	packet->max_blocks_per_sequence =
		SPARK_GLM52_PP13_WORK_CONTROL_KV_BLOCK_CAPACITY;
	for (lane_index = 0u; lane_index < packet->active_sequence_count; ++lane_index)
	{
		packet->lanes[lane_index].request_id = 7u + lane_index;
		packet->lanes[lane_index].sequence_id = 11u + lane_index;
		packet->lanes[lane_index].sequence_position = 128u;
		packet->lanes[lane_index].context_token_count = 129u;
		packet->lanes[lane_index].input_token_id = 100u + lane_index;
		packet->lanes[lane_index].kv_block_count = 3u;
		for (block_index = 0u; block_index < 3u; ++block_index)
			packet->lanes[lane_index].kv_physical_block_indices[block_index] =
				(lane_index * 4u) + block_index;
	}
	packet->input_token_id = packet->lanes[0u].input_token_id;
}

static void SparkTestGlm52Pp13WorkControlPacket(void)
{
	SparkGlm52Pp13WorkControlPacket packet;

	SparkTestInitializeWorkPacket(&packet);
	assert(SparkGlm52Pp13WorkControlValidatePacket(&packet,1024u,4u) ==
		SPARK_STATUS_OK);
	packet.new_token_count = 9u;
	assert(SparkGlm52Pp13WorkControlValidatePacket(&packet,1024u,4u) ==
		SPARK_STATUS_INVALID_ARGUMENT);
	SparkTestInitializeWorkPacket(&packet);
	packet.flags = SPARK_GLM52_PP13_WORK_CONTROL_FLAG_PREFILL;
	packet.new_token_count = 512u;
	packet.active_sequence_count = 1u;
	packet.descriptor_bytes =
		SparkGlm52Pp13WorkControlCalculatePacketBytes(1u);
	assert(SparkGlm52Pp13WorkControlValidatePacket(&packet,1024u,4u) ==
		SPARK_STATUS_OK);
	packet.active_sequence_count = 1025u;
	assert(SparkGlm52Pp13WorkControlValidatePacket(&packet,1024u,4u) ==
		SPARK_STATUS_INVALID_ARGUMENT);
}

static void SparkTestGlm52Pp13WorkControlHostBlockTable(void)
{
	SparkGlm52Pp13WorkControlPacket packet;
	SparkGlm52Pp13WorkControlKvState state;
	SparkGlm52KvBlockTableView view;
	uint32_t physical_blocks[4u * 32u];
	uint32_t lane_counts[4u];
	uint8_t block_states[64u];
	uint32_t block_index;

	SparkTestInitializeWorkPacket(&packet);
	assert(SparkGlm52Pp13WorkControlInitializeKvState(
		&state,
		4u,
		32u,
		SPARK_GLM52_KV_BLOCK_TOKENS,
		64u,
		physical_blocks,
		lane_counts,
		block_states) == SPARK_STATUS_OK);
	memset(block_states,SPARK_GLM52_PP13_KV_ENTRY_RESIDENT,
		sizeof(block_states));
	assert(SparkGlm52Pp13WorkControlBuildHostKvBlockTable(
		&packet,
		&state,
		&view) == SPARK_STATUS_OK);
	assert(view.abi_version == SPARK_GLM52_KV_CACHE_ABI_VERSION);
	assert(view.block_token_count == SPARK_GLM52_KV_BLOCK_TOKENS);
	assert(view.lane_count == 4u);
	assert(view.lane_stride == 32u);
	assert(view.physical_block_indices == physical_blocks);
	assert(view.host_physical_block_indices == physical_blocks);
	assert(lane_counts[0] == 3u);
	assert(lane_counts[3] == 3u);
	assert(physical_blocks[0] == 0u);
	assert(physical_blocks[2] == 2u);
	assert(physical_blocks[32] == 4u);
	for (block_index = 0u; block_index < 3u; ++block_index)
		assert(physical_blocks[(3u * 32u) + block_index] ==
			12u + block_index);
}

static void SparkTestGlm52Pp13WorkControlTracksKvReadiness(void)
{
	SparkGlm52Pp13WorkControlPacket packet;
	SparkGlm52Pp13WorkControlKvState state;
	SparkGlm52KvBlockTableView view;
	uint32_t physical_blocks[2u * 32u];
	uint32_t lane_counts[2u];
	uint8_t block_states[64u];

	SparkTestInitializeWorkPacket(&packet);
	packet.active_sequence_count = 1u;
	packet.descriptor_bytes =
		SparkGlm52Pp13WorkControlCalculatePacketBytes(1u);
	assert(SparkGlm52Pp13WorkControlInitializeKvState(
		&state,
		2u,
		32u,
		SPARK_GLM52_KV_BLOCK_TOKENS,
		64u,
		physical_blocks,
		lane_counts,
		block_states) == SPARK_STATUS_OK);
	assert(SparkGlm52Pp13WorkControlBuildHostKvBlockTable(
		&packet,
		&state,
		&view) == SPARK_STATUS_BUSY);
	assert(state.missing_block_count == 1u);
	packet.flags = SPARK_GLM52_PP13_WORK_CONTROL_FLAG_PREFILL;
	assert(SparkGlm52Pp13WorkControlBuildHostKvBlockTable(
		&packet,
		&state,
		&view) == SPARK_STATUS_OK);
	assert(block_states[0] == SPARK_GLM52_PP13_KV_ENTRY_IN_FLIGHT);
	assert(SparkGlm52Pp13WorkControlCommitHostKvBlockTable(
		&packet,
		&state) == SPARK_STATUS_OK);
	assert(block_states[0] == SPARK_GLM52_PP13_KV_ENTRY_RESIDENT);
	packet.flags = 0u;
	assert(SparkGlm52Pp13WorkControlBuildHostKvBlockTable(
		&packet,
		&state,
		&view) == SPARK_STATUS_OK);
	assert(state.resident_block_count != 0u);
	assert(SparkGlm52Pp13WorkControlCancelHostKvBlockTable(
		&packet,
		&state) == SPARK_STATUS_OK);
	assert(block_states[0] == SPARK_GLM52_PP13_KV_ENTRY_RESIDENT);
}

static void SparkTestGlm52Pp13WorkControlDsparkVerify(void)
{
	SparkGlm52Pp13WorkControlPacket packet;
	uint32_t token_index;

	SparkTestInitializeWorkPacket(&packet);
	packet.active_sequence_count = 1u;
	packet.descriptor_bytes =
		SparkGlm52Pp13WorkControlCalculatePacketBytes(1u);
	packet.flags =
		SPARK_GLM52_PP13_WORK_CONTROL_FLAG_DSPARK_TAP_CAPTURE |
		SPARK_GLM52_PP13_WORK_CONTROL_FLAG_DSPARK_SPECULATIVE_VERIFY;
	packet.speculative_token_count =
		SPARK_GLM52_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT;
	packet.speculative_token_index =
		SPARK_GLM52_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT;
	packet.input_token_id = 101u;
	packet.lanes[0u].input_token_id = packet.input_token_id;
	for (token_index = 0u;
		 token_index < SPARK_GLM52_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT;
		 ++token_index)
		packet.speculative_draft_token_ids[token_index] = 200u + token_index;
	assert(SparkGlm52Pp13WorkControlValidatePacket(&packet,1024u,4u) ==
		SPARK_STATUS_OK);
	packet.flags &=
		~SPARK_GLM52_PP13_WORK_CONTROL_FLAG_DSPARK_TAP_CAPTURE;
	assert(SparkGlm52Pp13WorkControlValidatePacket(&packet,1024u,4u) ==
		SPARK_STATUS_INVALID_ARGUMENT);
	packet.flags |=
		SPARK_GLM52_PP13_WORK_CONTROL_FLAG_DSPARK_TAP_CAPTURE;
	packet.speculative_token_index += 1u;
	assert(SparkGlm52Pp13WorkControlValidatePacket(&packet,1024u,4u) ==
		SPARK_STATUS_INVALID_ARGUMENT);
}

static void SparkTestGlm52Pp13WorkControlMtpVerify(void)
{
	SparkGlm52Pp13WorkControlPacket packet;
	uint32_t token_index;

	SparkTestInitializeWorkPacket(&packet);
	packet.active_sequence_count = 1u;
	packet.descriptor_bytes =
		SparkGlm52Pp13WorkControlCalculatePacketBytes(1u);
	packet.flags =
		SPARK_GLM52_PP13_WORK_CONTROL_FLAG_MTP_SPECULATIVE_VERIFY;
	packet.speculative_token_count = SPARK_GLM52_MODEL_MTP_DRAFT_TOKEN_COUNT;
	packet.speculative_token_index = SPARK_GLM52_MODEL_MTP_DRAFT_TOKEN_COUNT;
	packet.input_token_id = 101u;
	packet.lanes[0u].input_token_id = packet.input_token_id;
	for (token_index = 0u;
		 token_index < SPARK_GLM52_MODEL_MTP_DRAFT_TOKEN_COUNT;
		 ++token_index)
		packet.speculative_draft_token_ids[token_index] = 300u + token_index;
	assert(SparkGlm52Pp13WorkControlValidatePacket(&packet,1024u,4u) ==
		SPARK_STATUS_OK);
	packet.flags |= SPARK_GLM52_PP13_WORK_CONTROL_FLAG_DSPARK_TAP_CAPTURE;
	assert(SparkGlm52Pp13WorkControlValidatePacket(&packet,1024u,4u) ==
		SPARK_STATUS_INVALID_ARGUMENT);
	packet.flags =
		SPARK_GLM52_PP13_WORK_CONTROL_FLAG_MTP_SPECULATIVE_VERIFY |
		SPARK_GLM52_PP13_WORK_CONTROL_FLAG_DSPARK_SPECULATIVE_VERIFY;
	assert(SparkGlm52Pp13WorkControlValidatePacket(&packet,1024u,4u) ==
		SPARK_STATUS_INVALID_ARGUMENT);
}

static void SparkTestGlm52Pp13WorkControlBuildDecodeBatch(void)
{
	static SparkGlm52RequestApiDispatch request_dispatch;
	static SparkGlm52RequestApiDecodeDispatchView decode_view;
	static SparkGlm52ServingDecodeDispatch decode_dispatch;
	SparkGlm52Pp13WorkControlPacket packet;
	SparkGlm52KvBlockTableView kv_view;
	uint32_t block_indices[4u][2u];
	uint32_t block_counts[4u];
	uint32_t lane_index;

	memset(&request_dispatch,0,sizeof(request_dispatch));
	request_dispatch.abi_version = SPARK_GLM52_REQUEST_API_ABI_VERSION;
	request_dispatch.descriptor_bytes =
		SPARK_GLM52_REQUEST_API_DISPATCH_DESCRIPTOR_BYTES;
	request_dispatch.kind = SPARK_GLM52_REQUEST_API_DISPATCH_KIND_DECODE_BATCH;
	request_dispatch.request_count = 4u;
	request_dispatch.highest_priority = 77u;
	memset(&decode_view,0,sizeof(decode_view));
	decode_view.abi_version = SPARK_GLM52_REQUEST_API_ABI_VERSION;
	decode_view.descriptor_bytes =
		SPARK_GLM52_REQUEST_API_DECODE_DISPATCH_VIEW_DESCRIPTOR_BYTES;
	decode_view.kind = SPARK_GLM52_REQUEST_API_DISPATCH_KIND_DECODE_BATCH;
	decode_view.active_sequence_count = 4u;
	decode_view.lane_count = 4u;
	memset(&kv_view,0,sizeof(kv_view));
	kv_view.abi_version = SPARK_GLM52_KV_CACHE_ABI_VERSION;
	kv_view.descriptor_bytes = SPARK_GLM52_KV_BLOCK_TABLE_VIEW_DESCRIPTOR_BYTES;
	kv_view.block_token_count = SPARK_GLM52_KV_BLOCK_TOKENS;
	kv_view.lane_count = 4u;
	kv_view.lane_stride = 2u;
	kv_view.lane_capacity = 4u;
	kv_view.physical_block_indices = &block_indices[0u][0u];
	kv_view.lane_physical_block_counts = block_counts;
	kv_view.host_physical_block_indices = &block_indices[0u][0u];
	kv_view.host_lane_physical_block_counts = block_counts;
	memset(&decode_dispatch,0,sizeof(decode_dispatch));
	decode_dispatch.abi_version = SPARK_GLM52_SERVING_ENGINE_ABI_VERSION;
	decode_dispatch.descriptor_bytes =
		SPARK_GLM52_SERVING_DECODE_DISPATCH_DESCRIPTOR_BYTES;
	decode_dispatch.dispatch_kind =
		SPARK_GLM52_REQUEST_API_DISPATCH_KIND_DECODE_BATCH;
	decode_dispatch.request_count = 4u;
	decode_dispatch.active_sequence_count = 4u;
	decode_dispatch.request_dispatch = &request_dispatch;
	decode_dispatch.kv_block_table_view = &kv_view;
	decode_dispatch.decode_view = &decode_view;
	for (lane_index = 0u; lane_index < 4u; ++lane_index)
	{
		request_dispatch.request_ids[lane_index] = 100u + lane_index;
		request_dispatch.sequence_ids[lane_index] = 200u + lane_index;
		decode_view.lanes[lane_index].sequence_position = 31u;
		decode_view.lanes[lane_index].context_token_count = 32u;
		decode_dispatch.input_token_ids[lane_index] = 300u + lane_index;
		block_counts[lane_index] = 1u;
		block_indices[lane_index][0u] = 40u + lane_index;
	}
	assert(SparkGlm52Pp13WorkControlBuildDecodePacket(
		&decode_dispatch,0u,&packet) == SPARK_STATUS_OK);
	assert(packet.active_sequence_count == 4u);
	assert(packet.descriptor_bytes ==
		SparkGlm52Pp13WorkControlCalculatePacketBytes(4u));
	assert(packet.lanes[3u].request_id == 103u);
	assert(packet.lanes[3u].input_token_id == 303u);
	assert(packet.lanes[3u].kv_physical_block_indices[0u] == 43u);
	assert(SparkGlm52Pp13WorkControlValidatePacket(&packet,4u,1u) ==
		SPARK_STATUS_OK);
}

int main(void)
{
	SparkTestGlm52Pp13WorkControlPacket();
	SparkTestGlm52Pp13WorkControlHostBlockTable();
	SparkTestGlm52Pp13WorkControlTracksKvReadiness();
	SparkTestGlm52Pp13WorkControlDsparkVerify();
	SparkTestGlm52Pp13WorkControlMtpVerify();
	SparkTestGlm52Pp13WorkControlBuildDecodeBatch();
	return 0;
}
