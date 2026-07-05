#include <assert.h>
#include <string.h>

#include "sparkpipe/spark_glm52_pp13_work_control.h"

static void SparkTestInitializeWorkPacket(
	SparkGlm52Pp13WorkControlPacket *packet)
{
	memset(packet,0,sizeof(*packet));
	packet->magic = SPARK_GLM52_PP13_WORK_CONTROL_PACKET_MAGIC;
	packet->abi_version = SPARK_GLM52_PP13_WORK_CONTROL_ABI_VERSION;
	packet->descriptor_bytes = SPARK_GLM52_PP13_WORK_CONTROL_PACKET_BYTES;
	packet->request_id = 7u;
	packet->sequence_id = 11u;
	packet->sequence_position = 32768u;
	packet->active_sequence_count = 4u;
	packet->new_token_count = 1u;
	packet->pipeline_slot = 0u;
	packet->block_token_count = 256u;
	packet->kv_block_table_token_count = 32769u;
	packet->max_blocks_per_sequence = 4096u;
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
	uint32_t physical_blocks[4u * 4096u];
	uint32_t lane_counts[4u];
	uint8_t block_states[4u * 4096u];

	SparkTestInitializeWorkPacket(&packet);
	packet.flags = SPARK_GLM52_PP13_WORK_CONTROL_FLAG_PREFILL;
	assert(SparkGlm52Pp13WorkControlInitializeKvState(
		&state,
		4u,
		4096u,
		256u,
		physical_blocks,
		lane_counts,
		block_states) == SPARK_STATUS_OK);
	assert(SparkGlm52Pp13WorkControlBuildHostKvBlockTable(
		&packet,
		&state,
		&view) == SPARK_STATUS_OK);
	assert(view.abi_version == SPARK_GLM52_KV_CACHE_ABI_VERSION);
	assert(view.block_token_count == 256u);
	assert(view.lane_count == 4u);
	assert(view.lane_stride == 4096u);
	assert(view.physical_block_indices == physical_blocks);
	assert(view.host_physical_block_indices == physical_blocks);
	assert(lane_counts[0] == 129u);
	assert(lane_counts[3] == 129u);
	assert(physical_blocks[0] == 0u);
	assert(physical_blocks[128] == 128u);
	assert(physical_blocks[4096] == 4096u);
	assert(physical_blocks[(3u * 4096u) + 128u] == ((3u * 4096u) + 128u));
}

static void SparkTestGlm52Pp13WorkControlTracksKvReadiness(void)
{
	SparkGlm52Pp13WorkControlPacket packet;
	SparkGlm52Pp13WorkControlKvState state;
	SparkGlm52KvBlockTableView view;
	uint32_t physical_blocks[2u * 4096u];
	uint32_t lane_counts[2u];
	uint8_t block_states[2u * 4096u];

	SparkTestInitializeWorkPacket(&packet);
	packet.active_sequence_count = 2u;
	assert(SparkGlm52Pp13WorkControlInitializeKvState(
		&state,
		2u,
		4096u,
		256u,
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

int main(void)
{
	SparkTestGlm52Pp13WorkControlPacket();
	SparkTestGlm52Pp13WorkControlHostBlockTable();
	SparkTestGlm52Pp13WorkControlTracksKvReadiness();
	return 0;
}
