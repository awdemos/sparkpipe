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

static void SparkTestBackendPrepareInflightPending(
    SparkRingServiceBackendState *state,
    SparkScheduler *scheduler,
    SparkRequestApiSlot *request_slots,
    SparkServingRequestRecord *request_records,
    SparkServingEvent *event_ring,
    uint64_t request_id,
    uint64_t request_handle)
{
    SparkRingServiceBackendPendingDecode *pending;
    uint32_t hash_index;

    memset(state,0,sizeof(*state));
    memset(scheduler,0,sizeof(*scheduler));
    memset(request_slots,0,sizeof(request_slots[0]));
    memset(request_records,0,sizeof(request_records[0]));
    memset(event_ring,0,sizeof(event_ring[0]) * 4u);

    state->request_api.abi_version = SPARK_REQUEST_API_ABI_VERSION;
    state->request_api.descriptor_bytes = SPARK_REQUEST_API_DESCRIPTOR_BYTES;
    state->request_api.request_capacity = 1u;
    state->request_api.running_request_count = 1u;
    state->request_api.decode_batch_target = 1u;
    state->request_api.decode_execution_row_capacity = 1u;
    state->request_api.scheduler = scheduler;
    state->request_api.request_slots = request_slots;
    for (hash_index = 0u;
         hash_index < SPARK_REQUEST_API_SLOT_HASH_SLOTS;
         ++hash_index)
    {
        state->request_api.slot_handle_hash_heads[hash_index] = 0u;
    }
    request_slots[0].state = SPARK_REQUEST_API_STATE_RUNNING_DECODE;
    request_slots[0].handle = request_handle;
    request_slots[0].handle_hash_next = SPARK_REQUEST_API_NO_SLOT;

    state->serving_engine.abi_version = SPARK_SERVING_ENGINE_ABI_VERSION;
    state->serving_engine.descriptor_bytes =
        SPARK_SERVING_ENGINE_DESCRIPTOR_BYTES;
    state->serving_engine.flags =
        SPARK_SERVING_ENGINE_FLAG_DYNAMIC_REQUEST_TOKEN_STORAGE;
    state->serving_engine.request_api = &state->request_api;
    state->serving_engine.request_records = request_records;
    state->serving_engine.request_record_capacity = 1u;
    state->serving_engine.event_ring = event_ring;
    state->serving_engine.event_ring_capacity = 4u;

    pending = &state->pending_decodes[0];
    pending->state = SPARK_RING_SERVICE_BACKEND_PENDING_DECODE_STATE_ACTIVE;
    pending->dispatch.accepted = 1u;
    pending->dispatch.kind = SPARK_REQUEST_API_DISPATCH_KIND_DECODE_BATCH;
    pending->dispatch.request_count = 1u;
    pending->dispatch.request_handles[0u] = request_handle;
    pending->dispatch.request_ids[0u] = request_id;
    pending->dispatch.decode_batch_decision.abi_version =
        SPARK_SCHEDULER_ABI_VERSION;
    pending->dispatch.decode_batch_decision.descriptor_bytes =
        SPARK_SCHEDULER_BATCH_DECISION_DESCRIPTOR_BYTES;
    pending->dispatch.decode_batch_decision.accepted = 1u;
    pending->dispatch.decode_batch_decision.packed_request_count = 1u;
    pending->dispatch.decode_batch_decision.decision_flags =
        SPARK_SCHEDULER_DECISION_FLAG_ADAPTIVE_DECODE_PACK;
}

static void SparkTestBackendHardNegativeAckFailsCohort(void)
{
    static SparkRingServiceBackendState state;
    static SparkScheduler scheduler;
    static SparkRequestApiSlot request_slots[1];
    static SparkServingRequestRecord request_records[1];
    static SparkServingEvent event_ring[4];
    SparkDistributedWorkAcknowledgement acknowledgement;
    SparkDistributedWorkIdentity identity;
    SparkRingWorkControlPacket packet;
    uint64_t packet_hash;
    int32_t sockets[2];

    SparkTestBackendPrepareInflightPending(
        &state,&scheduler,request_slots,request_records,event_ring,
        301u,777ull);
    state.rank_plan.flags = SPARK_RING_RUNTIME_RANK_FLAG_HAS_NEXT;
    SparkTestBackendBuildPacket(&packet,301u,3001u,401u,12u);
    assert(SparkRingServiceBackendEnqueueWorkPacket(
        &state,&packet) == SPARK_STATUS_OK);
    assert(state.work_queue_count == 1u);
    assert(SparkRingWorkControlGetTransactionIdentity(
        &packet,&identity) == SPARK_STATUS_OK);
    packet_hash = SparkDistributedWorkHashBytes(
        &packet,packet.descriptor_bytes);
    assert(packet_hash != 0u);
    SparkDistributedWorkInitializeAcknowledgement(
        &acknowledgement,&identity,packet_hash,
        SPARK_STATUS_MODULE_NOT_VALIDATED);
    assert(socketpair(AF_UNIX,SOCK_STREAM,0,sockets) == 0);
    state.work_output_socket_fd = sockets[0];
    assert(write(sockets[1],&acknowledgement,sizeof(acknowledgement)) ==
        (ssize_t)sizeof(acknowledgement));
    assert(SparkRingServiceBackendFlushWorkOutput(&state) == SPARK_STATUS_OK);
    /* The poisoned packet is dropped instead of retransmitted forever. */
    assert(state.work_queue_count == 0u);
    assert(state.work_output_waiting_for_acknowledgement == 0u);
    /* The rejection targets content, not the link, so the socket stays. */
    assert(state.work_output_socket_fd == sockets[0]);
    /* The owning cohort is failed deterministically. */
    assert(state.pending_decodes[0].state ==
        SPARK_RING_SERVICE_BACKEND_PENDING_DECODE_STATE_FREE);
    assert(request_slots[0].state == SPARK_REQUEST_API_STATE_CANCELLED);
    /* The next pump has nothing to retransmit. */
    assert(SparkRingServiceBackendFlushWorkOutput(&state) == SPARK_STATUS_OK);
    close(sockets[0]);
    close(sockets[1]);
}

static void SparkTestBackendReconnectFailsInflightPendings(void)
{
    static SparkRingServiceBackendState state;
    static SparkScheduler scheduler;
    static SparkRequestApiSlot request_slots[1];
    static SparkServingRequestRecord request_records[1];
    static SparkServingEvent event_ring[4];

    SparkTestBackendPrepareInflightPending(
        &state,&scheduler,request_slots,request_records,event_ring,
        302u,778ull);
    state.cuda_resident_fd = -1;
    /* The connect itself fails, but the in-flight pendings charged to
       the torn-down connection must already be failed deterministically
       before any fresh credit ledger replaces the old accounting. */
    assert(SparkRingServiceBackendConnectCudaResident(
        &state,"/nonexistent/sparkpipe-test-resident.sock") !=
        SPARK_STATUS_OK);
    assert(state.pending_decodes[0].state ==
        SPARK_RING_SERVICE_BACKEND_PENDING_DECODE_STATE_FREE);
    assert(request_slots[0].state == SPARK_REQUEST_API_STATE_CANCELLED);
    assert(state.cuda_resident_fd == -1);
}

static void SparkTestBackendStaleCompletionDoesNotTeardown(void)
{
    static SparkRingServiceBackendState state;
    SparkCudaResidentIpcHeader header;
    uint32_t credit_capacities[SPARK_DISTRIBUTED_WORK_CREDIT_DOMAIN_COUNT];

    memset(&state,0,sizeof(state));
    memset(&header,0,sizeof(header));
    header.kind = SPARK_CUDA_RESIDENT_IPC_KIND_COMPLETION;
    header.payload_bytes = SPARK_CUDA_RESIDENT_IPC_COMPLETION_BYTES;
    state.cuda_resident_payload.completion.descriptor_bytes =
        SPARK_CUDA_RESIDENT_IPC_COMPLETION_BYTES;
    state.cuda_resident_payload.completion.completion.status =
        SPARK_STATUS_OK;
    state.cuda_resident_payload.completion.completion.request_id = 303u;
    state.cuda_resident_payload.completion.completion.sequence_id = 403u;
    memset(credit_capacities,0,sizeof(credit_capacities));
    credit_capacities[
        SPARK_DISTRIBUTED_WORK_CREDIT_DOMAIN_RESIDENT_RESERVATION] = 4u;
    assert(SparkDistributedWorkInitializeCreditLedger(
        &state.credit_ledger,credit_capacities) == SPARK_STATUS_OK);
    /* The fresh ledger has nothing in use: a completion charged to a
       torn-down connection must be tolerated, not error the new one. */
    assert(SparkRingServiceBackendHandleResidentCompletion(
        &state,&header) == SPARK_STATUS_OK);
    assert(state.cuda_resident_completion_count == 1u);
}

int main(void)
{
    SparkTestBackendValidatesFinalEventTransaction();
    SparkTestBackendMatchesPendingLaneGeneration();
    SparkTestBackendSuppressesExactEarlyFinalReplay();
    SparkTestBackendRejectsConflictingEarlyFinalReplay();
    SparkTestBackendCopiesOnlyWorkDescriptor();
    SparkTestBackendCoalescesGeneratedRelease();
    SparkTestBackendHardNegativeAckFailsCohort();
    SparkTestBackendReconnectFailsInflightPendings();
    SparkTestBackendStaleCompletionDoesNotTeardown();
    return 0;
}
