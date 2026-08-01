#include "../node/backend.c"

#include <assert.h>
#include <stdint.h>
#include <string.h>

static void SparkTestBackendBuildPacket(
    SparkRingWorkControlPacket *packet,
    uint64_t request_id,
    uint64_t request_generation,
    uint64_t sequence_id,
    uint64_t sequence_position)
{
    memset(packet,0,sizeof(*packet));
    packet->magic = SPARK_RING_WORK_CONTROL_PACKET_MAGIC;
    packet->abi_version = SPARK_RING_WORK_CONTROL_ABI_VERSION;
    packet->new_token_count = 1u;
    packet->lane_count = 1u;
    packet->active_sequence_count = 1u;
    packet->rows_per_lane = 1u;
    packet->execution_row_count = 1u;
    packet->execution_batch_bucket = 1u;
    packet->block_token_count = SPARK_GLM52_KV_BLOCK_TOKENS;
    packet->max_blocks_per_sequence =
        SPARK_RING_WORK_CONTROL_KV_BLOCK_CAPACITY;
    packet->kv_block_table_token_count =
        (uint32_t)sequence_position + 1u;
    packet->request_id = request_id;
    packet->sequence_id = sequence_id;
    packet->sequence_position = sequence_position;
    packet->input_token_id = 17u;
    packet->lanes[0u].request_id = request_id;
    packet->lanes[0u].request_generation = request_generation;
    packet->lanes[0u].sequence_id = sequence_id;
    packet->lanes[0u].sequence_position = sequence_position;
    packet->lanes[0u].request_slot_index = 0u;
    packet->lanes[0u].context_token_count =
        (uint32_t)sequence_position + 1u;
    packet->lanes[0u].input_token_id = packet->input_token_id;
    packet->descriptor_bytes =
        SparkRingWorkControlCalculatePacketBytes(packet->lane_count);
    assert(SparkRingWorkControlFinalizeTransaction(
        packet,11u,0u,1u) == SPARK_STATUS_OK);
}

static void SparkTestBackendBuildFinalEvent(
    SparkRingRuntimeFinalEvent *event,
    const SparkRingWorkControlPacket *packet)
{
    memset(event,0,sizeof(*event));
    event->magic = SPARK_RING_RUNTIME_FINAL_EVENT_MAGIC;
    event->descriptor_bytes =
        SPARK_RING_RUNTIME_FINAL_EVENT_DESCRIPTOR_BYTES;
    event->status = SPARK_STATUS_OK;
    event->completion_flags =
        SPARK_MODEL_DRIVER_COMPLETION_FLAG_TOKEN_IDS;
    event->token_count = 1u;
    event->token_ids[0u] = 23u;
    event->request_id = packet->lanes[0u].request_id;
    event->control_generation = packet->control_generation;
    event->transaction_id = packet->transaction_id;
    event->dispatch_generation = packet->dispatch_generation;
    event->request_generation =
        packet->lanes[0u].request_generation;
    event->sequence_id = packet->lanes[0u].sequence_id;
    event->sequence_position = packet->lanes[0u].sequence_position;
    event->step_generation = packet->step_generation;
    event->step_chunk_index = packet->step_chunk_index;
    event->step_chunk_count = packet->step_chunk_count;
    event->transaction_phase = packet->transaction_phase;
}

static void SparkTestBackendValidatesFinalEventTransaction(void)
{
    SparkRingRuntimeFinalEvent event;
    SparkRingWorkControlPacket packet;

    SparkTestBackendBuildPacket(&packet,101u,1001u,201u,7u);
    SparkTestBackendBuildFinalEvent(&event,&packet);
    assert(SparkRingServiceBackendValidateFinalEventEnvelope(&event) ==
        SPARK_STATUS_OK);
    event.step_chunk_count = 0u;
    assert(SparkRingServiceBackendValidateFinalEventEnvelope(&event) ==
        SPARK_STATUS_VALIDATION_FAILED);
    SparkTestBackendBuildFinalEvent(&event,&packet);
    event.reserved_transaction = 1u;
    assert(SparkRingServiceBackendValidateFinalEventEnvelope(&event) ==
        SPARK_STATUS_VALIDATION_FAILED);
}

static void SparkTestBackendMatchesPendingLaneGeneration(void)
{
    SparkRingRuntimeFinalEvent event;
    SparkRingServiceBackendPendingDecode pending;
    SparkRingWorkControlPacket packet;
    uint32_t transaction_matches;

    memset(&pending,0,sizeof(pending));
    SparkTestBackendBuildPacket(&packet,102u,1002u,202u,8u);
    SparkTestBackendBuildFinalEvent(&event,&packet);
    pending.dispatch.request_count = 1u;
    pending.dispatch.request_ids[0u] = event.request_id;
    pending.dispatch.sequence_ids[0u] = event.sequence_id;
    pending.lane_transactions[0u].control_generation =
        event.control_generation;
    pending.lane_transactions[0u].transaction_id = event.transaction_id;
    pending.lane_transactions[0u].dispatch_generation =
        event.dispatch_generation;
    pending.lane_transactions[0u].request_generation =
        event.request_generation;
    pending.lane_transactions[0u].sequence_position =
        event.sequence_position;
    pending.lane_transactions[0u].step_generation = event.step_generation;
    pending.lane_transactions[0u].step_chunk_index = event.step_chunk_index;
    pending.lane_transactions[0u].step_chunk_count = event.step_chunk_count;
    pending.lane_transactions[0u].transaction_phase =
        event.transaction_phase;
    transaction_matches = 0u;
    assert(SparkRingServiceBackendFindDecodeLane(
        &pending,&event,&transaction_matches) == 0);
    assert(transaction_matches == 1u);
    event.step_generation += 1u;
    transaction_matches = 1u;
    assert(SparkRingServiceBackendFindDecodeLane(
        &pending,&event,&transaction_matches) == 0);
    assert(transaction_matches == 0u);
}

static void SparkTestBackendSuppressesExactEarlyFinalReplay(void)
{
    static SparkRingServiceBackendState state;
    SparkRingRuntimeFinalEvent event;
    SparkRingWorkControlPacket packet;

    memset(&state,0,sizeof(state));
    SparkTestBackendBuildPacket(&packet,103u,1003u,203u,9u);
    SparkTestBackendBuildFinalEvent(&event,&packet);
    assert(SparkRingServiceBackendStashEarlyFinalEvent(
        &state,&event) == SPARK_STATUS_BUSY);
    assert(state.early_final_event_count == 1u);
    assert(SparkRingServiceBackendStashEarlyFinalEvent(
        &state,&event) == SPARK_STATUS_BUSY);
    assert(state.early_final_event_count == 1u);
    assert(state.final_event_receive_error_count == 0u);
}


static void SparkTestBackendRejectsConflictingEarlyFinalReplay(void)
{
	static SparkRingServiceBackendState state;
	SparkRingRuntimeFinalEvent event;
	SparkRingWorkControlPacket packet;

	memset(&state,0,sizeof(state));
	SparkTestBackendBuildPacket(&packet,106u,1006u,206u,11u);
	SparkTestBackendBuildFinalEvent(&event,&packet);
	assert(SparkRingServiceBackendStashEarlyFinalEvent(
		&state,&event) == SPARK_STATUS_BUSY);
	event.token_ids[0u] += 1u;
	assert(SparkRingServiceBackendStashEarlyFinalEvent(
		&state,&event) == SPARK_STATUS_VALIDATION_FAILED);
	assert(state.early_final_event_count == 1u);
}

static void SparkTestBackendCopiesOnlyWorkDescriptor(void)
{
	static SparkRingServiceBackendState state;
	SparkRingServiceBackendWorkOutputSlot *queue_slot;
	SparkRingWorkControlPacket packet;
	uint8_t *packet_bytes;
	uint32_t probe_offset;

	memset(&state,0,sizeof(state));
	state.rank_plan.flags = SPARK_RING_RUNTIME_RANK_FLAG_HAS_NEXT;
	SparkTestBackendBuildPacket(&packet,104u,1004u,204u,10u);
	packet_bytes = (uint8_t *)&packet;
	probe_offset = packet.descriptor_bytes + 8u;
	assert(probe_offset < sizeof(packet));
	packet_bytes[probe_offset] = 0xa5u;
	assert(SparkRingServiceBackendEnqueueWorkPacket(
		&state,&packet) == SPARK_STATUS_OK);
	assert(state.work_queue_count == 1u);
	queue_slot = &state.work_queue[state.work_queue_head];
	assert(queue_slot->packet_bytes != 0);
	assert(queue_slot->packet_bytes_count == packet.descriptor_bytes);
	assert(memcmp(
		queue_slot->packet_bytes,
		&packet,
		packet.descriptor_bytes) == 0);
	SparkRingServiceBackendPopWorkPacket(&state);
	assert(state.work_queue_count == 0u);
}

static void SparkTestBackendCoalescesGeneratedRelease(void)
{
    static SparkRingServiceBackendState state;
    uint32_t first_token_count;

    memset(&state,0,sizeof(state));
    assert(SparkRingServiceBackendQueueSequenceRelease(
        &state,105u,1005u,205u,40u) == SPARK_STATUS_OK);
    assert(state.release_queue_count == 1u);
    first_token_count = state.release_queue[0u].token_count;
    assert(SparkRingServiceBackendQueueSequenceRelease(
        &state,105u,1005u,205u,20u) == SPARK_STATUS_OK);
    assert(state.release_queue_count == 1u);
    assert(state.release_queue[0u].token_count == first_token_count);
    assert(SparkRingServiceBackendQueueSequenceRelease(
        &state,105u,1005u,205u,80u) == SPARK_STATUS_OK);
    assert(state.release_queue_count == 1u);
    assert(state.release_queue[0u].token_count > first_token_count);
    assert(SparkRingServiceBackendQueueSequenceRelease(
        &state,105u,1006u,205u,80u) == SPARK_STATUS_OK);
    assert(state.release_queue_count == 2u);
}

int main(void)
{
    SparkTestBackendValidatesFinalEventTransaction();
    SparkTestBackendMatchesPendingLaneGeneration();
    SparkTestBackendSuppressesExactEarlyFinalReplay();
    SparkTestBackendRejectsConflictingEarlyFinalReplay();
    SparkTestBackendCopiesOnlyWorkDescriptor();
    SparkTestBackendCoalescesGeneratedRelease();
    return 0;
}
