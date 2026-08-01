#define SparkServiceBackendGetInterface \
    SparkTestRingServiceBackendGetInterface
#include "../node/backend.c"
#undef SparkServiceBackendGetInterface

#include <assert.h>

static void SparkTestBackendInitializeState(
    SparkRingServiceBackendState *state)
{
    memset(state,0,sizeof(*state));
    state->session_id_base = UINT64_C(123456789);
    state->work_output_socket_fd = -1;
    state->cuda_resident_fd = -1;
    state->final_event_listen_fd = -1;
    state->final_event_socket_fd = -1;
    state->rank_plan.flags = SPARK_RING_RUNTIME_RANK_FLAG_HAS_NEXT;
    state->rank_plan.execution_row_capacity = SPARK_STAGE_PLAN_BUCKET_B16;
}

static void SparkTestBackendBuildPacket(
    SparkRingServiceBackendState *state,
    SparkRingWorkControlPacket *packet,
    uint64_t request_id)
{
    memset(packet,0,sizeof(*packet));
    packet->magic = SPARK_RING_WORK_CONTROL_PACKET_MAGIC;
    packet->abi_version = SPARK_RING_WORK_CONTROL_ABI_VERSION;
    packet->descriptor_bytes = SparkRingWorkControlCalculatePacketBytes(1u);
    packet->flags = SPARK_RING_WORK_CONTROL_FLAG_RELEASE_SEQUENCES;
    packet->request_id = request_id;
    packet->sequence_id = request_id;
    packet->active_sequence_count = 1u;
    packet->lane_count = 1u;
    packet->block_token_count = SPARK_RING_SERVICE_BACKEND_KV_BLOCK_TOKENS;
    packet->kv_block_table_token_count = 32u;
    packet->max_blocks_per_sequence =
        SPARK_RING_SERVICE_BACKEND_MAX_BLOCKS_PER_SEQUENCE;
    packet->lanes[0u].request_id = request_id;
    packet->lanes[0u].request_generation = request_id + 1000u;
    packet->lanes[0u].sequence_id = request_id;
    packet->lanes[0u].context_token_count = 32u;
    packet->lanes[0u].mtp_resolution_path_id =
        SPARK_MODEL_MTP_TREE_RESOLUTION_NONE;
    assert(SparkRingServiceBackendStampWorkPacket(
        state,
        packet) == SPARK_STATUS_OK);
}

static void SparkTestBackendStampIsDeterministic(void)
{
    static SparkRingServiceBackendState state;
    SparkRingWorkControlPacket first;
    SparkRingWorkControlPacket second;

    SparkTestBackendInitializeState(&state);
    SparkTestBackendBuildPacket(&state,&first,11u);
    second = first;
    assert(SparkRingServiceBackendStampWorkPacket(
        &state,
        &second) == SPARK_STATUS_OK);
    assert(first.control_generation == second.control_generation);
    assert(first.transaction_id == second.transaction_id);
    assert(first.dispatch_generation == second.dispatch_generation);
    assert(first.step_generation == second.step_generation);
    assert(memcmp(&first,&second,first.descriptor_bytes) == 0);
}

static void SparkTestBackendQueueOwnsExactPacketAndDeduplicates(void)
{
    static SparkRingServiceBackendState state;
    SparkRingWorkControlPacket packet;
    SparkRingWorkControlPacket *queued_packet;

    SparkTestBackendInitializeState(&state);
    assert(SparkRingServiceBackendInitializeWorkPacketArena(&state) ==
        SPARK_STATUS_OK);
    SparkTestBackendBuildPacket(&state,&packet,21u);
    assert(SparkRingServiceBackendEnqueueWorkPacket(
        &state,
        &packet) == SPARK_STATUS_OK);
    memset(&packet,0,sizeof(packet));
    queued_packet = SparkRingServiceBackendWorkQueueHeadPacket(&state);
    assert(queued_packet != 0);
    assert(queued_packet->request_id == 21u);
    packet = *queued_packet;
    assert(SparkRingServiceBackendEnqueueWorkPacket(
        &state,
        &packet) == SPARK_STATUS_OK);
    assert(state.work_queue_count == 1u);
    SparkRingServiceBackendFreeStorage(&state);
}

static void SparkTestBackendRequiresAcknowledgementBeforePop(void)
{
    static SparkRingServiceBackendState state;
    SparkDistributedWorkAcknowledgement acknowledgement;
    SparkDistributedWorkIdentity identity;
    SparkRingWorkControlPacket received_packet;
    SparkRingWorkControlPacket packet;
    SparkStatus status;
    int32_t sockets[2];
    uint64_t packet_hash;

    SparkTestBackendInitializeState(&state);
    assert(SparkRingServiceBackendInitializeWorkPacketArena(&state) ==
        SPARK_STATUS_OK);
    assert(socketpair(AF_UNIX,SOCK_STREAM,0,sockets) == 0);
    assert(SparkNetSetNonblocking(sockets[0]) == 0);
    state.work_output_socket_fd = sockets[0];
    SparkTestBackendBuildPacket(&state,&packet,31u);
    assert(SparkRingServiceBackendEnqueueWorkPacket(
        &state,
        &packet) == SPARK_STATUS_OK);
    status = SparkRingServiceBackendFlushWorkOutput(&state);
    assert(status == SPARK_STATUS_BUSY);
    assert(state.work_queue_count == 1u);
    memset(&received_packet,0,sizeof(received_packet));
    {
        uint32_t read_offset;
        ssize_t got;

        read_offset = 0u;
        while (read_offset < packet.descriptor_bytes)
        {
            got = read(
                sockets[1],
                ((uint8_t *)&received_packet) + read_offset,
                packet.descriptor_bytes - read_offset);
            assert(got > 0);
            read_offset += (uint32_t)got;
        }
    }
    assert(memcmp(&received_packet,&packet,packet.descriptor_bytes) == 0);
    assert(SparkRingWorkControlGetTransactionIdentity(
        &packet,
        &identity) == SPARK_STATUS_OK);
    packet_hash = SparkDistributedWorkHashBytes(
        &packet,
        packet.descriptor_bytes);
    SparkDistributedWorkInitializeAcknowledgement(
        &acknowledgement,
        &identity,
        packet_hash,
        SPARK_STATUS_OK);
    assert(write(sockets[1],&acknowledgement,sizeof(acknowledgement)) ==
        (ssize_t)sizeof(acknowledgement));
    assert(SparkRingServiceBackendFlushWorkOutput(&state) ==
        SPARK_STATUS_OK);
    assert(state.work_queue_count == 0u);
    close(sockets[0]);
    close(sockets[1]);
    SparkArenaDestroy(&state.work_packet_arena);
}

static void SparkTestBackendCoalescesSequenceRelease(void)
{
    static SparkRingServiceBackendState state;

    SparkTestBackendInitializeState(&state);
    assert(SparkRingServiceBackendQueueSequenceRelease(
        &state,
        41u,
        141u,
        51u,
        64u) == SPARK_STATUS_OK);
    assert(SparkRingServiceBackendQueueSequenceRelease(
        &state,
        41u,
        141u,
        51u,
        128u) == SPARK_STATUS_OK);
    assert(state.release_queue_count == 1u);
    assert(state.release_queue[0u].token_count ==
        128u + SPARK_RING_WORK_CONTROL_MAX_SPECULATIVE_TOKEN_COUNT);
}

static void SparkTestBackendEarlyFinalQueueBackpressures(void)
{
    static SparkRingServiceBackendState state;
    SparkRingRuntimeFinalEvent event;

    SparkTestBackendInitializeState(&state);
    memset(&event,0,sizeof(event));
    event.magic = SPARK_RING_RUNTIME_FINAL_EVENT_MAGIC;
    event.descriptor_bytes = SPARK_RING_RUNTIME_FINAL_EVENT_DESCRIPTOR_BYTES;
    event.status = SPARK_STATUS_OK;
    event.completion_flags = SPARK_MODEL_DRIVER_COMPLETION_FLAG_TOKEN_IDS;
    event.token_count = 1u;
    event.token_ids[0u] = 1u;
    event.request_id = 1u;
    event.control_generation = 10u;
    event.transaction_id = 11u;
    event.dispatch_generation = 12u;
    event.request_generation = 2u;
    event.sequence_id = 3u;
    event.sequence_position = 4u;
    event.step_generation = 5u;
    event.step_chunk_count = 1u;
    event.transaction_phase = SPARK_DISTRIBUTED_WORK_PHASE_DECODE;
    state.early_final_event_count =
        SPARK_RING_SERVICE_BACKEND_EARLY_FINAL_EVENT_CAPACITY;
    assert(SparkRingServiceBackendStashEarlyFinalEvent(
        &state,
        &event) == SPARK_STATUS_BUSY);
    assert(state.early_final_event_count ==
        SPARK_RING_SERVICE_BACKEND_EARLY_FINAL_EVENT_CAPACITY);
    assert(state.early_final_event_head == 0u);
}

static void SparkTestBackendResidentCreditIsStrict(void)
{
    static SparkRingServiceBackendState state;
    uint32_t capacities[SPARK_DISTRIBUTED_WORK_CREDIT_DOMAIN_COUNT];

    SparkTestBackendInitializeState(&state);
    memset(capacities,0,sizeof(capacities));
    capacities[SPARK_DISTRIBUTED_WORK_CREDIT_DOMAIN_TRANSPORT_WINDOW] = 1u;
    capacities[SPARK_DISTRIBUTED_WORK_CREDIT_DOMAIN_RESIDENT_RESERVATION] = 2u;
    capacities[SPARK_DISTRIBUTED_WORK_CREDIT_DOMAIN_EXECUTION] = 2u;
    capacities[SPARK_DISTRIBUTED_WORK_CREDIT_DOMAIN_COMPLETION_OWNERSHIP] = 2u;
    assert(SparkDistributedWorkInitializeCreditLedger(
        &state.credit_ledger,
        capacities) == SPARK_STATUS_OK);
    assert(SparkDistributedWorkAcquireCredits(
        &state.credit_ledger,
        SPARK_DISTRIBUTED_WORK_CREDIT_DOMAIN_RESIDENT_RESERVATION,
        1u) == SPARK_STATUS_OK);
    assert(SparkRingServiceBackendReleaseResidentSubmitCredit(&state) ==
        SPARK_STATUS_OK);
    assert(SparkRingServiceBackendReleaseResidentSubmitCredit(&state) ==
        SPARK_STATUS_VALIDATION_FAILED);
}

int main(void)
{
    SparkTestBackendStampIsDeterministic();
    SparkTestBackendQueueOwnsExactPacketAndDeduplicates();
    SparkTestBackendRequiresAcknowledgementBeforePop();
    SparkTestBackendCoalescesSequenceRelease();
    SparkTestBackendEarlyFinalQueueBackpressures();
    SparkTestBackendResidentCreditIsStrict();
    return 0;
}
