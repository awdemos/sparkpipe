#define main SparkTestGlm52RingRankDaemonMain
#include "../node/rank_daemon.c"
#undef main

#include <assert.h>

static SparkStatus SparkTestRankDaemonSubmitStatus = SPARK_STATUS_OK;
static uint32_t SparkTestRankDaemonSubmitCompletions = 0u;

static SparkStatus SparkTestRankDaemonSubmitWork(
    void *builder_state,
    const SparkRingWorkControlPacket *work_packet,
    SparkHiddenTransportSession *input_transport_session,
    SparkHiddenTransportSession *output_transport_session,
    SparkModelDriverCompletionFunction completion_function,
    void *completion_context)
{
    SparkModelDriverCompletion completion;
    uint32_t lane_index;

    (void)builder_state;
    (void)input_transport_session;
    (void)output_transport_session;
    if (work_packet == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkTestRankDaemonSubmitStatus != SPARK_STATUS_OK ||
        SparkTestRankDaemonSubmitCompletions == 0u ||
        completion_function == 0)
    {
        return SparkTestRankDaemonSubmitStatus;
    }
    for (lane_index = 0u;
         lane_index < work_packet->lane_count;
         ++lane_index)
    {
        memset(&completion,0,sizeof(completion));
        completion.request_id = work_packet->lanes[lane_index].request_id;
        completion.sequence_id = work_packet->lanes[lane_index].sequence_id;
        completion.sequence_position =
            work_packet->lanes[lane_index].sequence_position;
        completion.program_id = 3u;
        completion.driver_dispatch_slot = lane_index;
        completion.accepted_token_count = 1u;
        completion.completion_flags =
            SPARK_MODEL_DRIVER_COMPLETION_FLAG_TOKEN_IDS;
        completion.token_count = 1u;
        completion.token_ids[0u] = 100u + lane_index;
        completion.status = SPARK_STATUS_OK;
        completion_function(completion_context,&completion);
    }
    return SPARK_STATUS_OK;
}

static uint32_t SparkTestRankDaemonPhaseFromFlags(uint32_t flags)
{
    if ((flags & SPARK_RING_WORK_CONTROL_FLAG_RELEASE_SEQUENCES) != 0u)
    {
        return SPARK_RING_DAEMON_WORK_PHASE_RELEASE;
    }
    if ((flags & SPARK_RING_WORK_CONTROL_FLAG_PREFILL) != 0u)
    {
        return SPARK_RING_DAEMON_WORK_PHASE_PREFILL;
    }
    if ((flags &
            (SPARK_RING_WORK_CONTROL_FLAG_DSPARK_SPECULATIVE_VERIFY |
             SPARK_RING_WORK_CONTROL_FLAG_MTP_SPECULATIVE_VERIFY)) != 0u)
    {
        return SPARK_RING_DAEMON_WORK_PHASE_VERIFY;
    }
    return SPARK_RING_DAEMON_WORK_PHASE_DECODE;
}

static uint64_t SparkTestRankDaemonTransactionId(
    uint64_t sequence_id,
    uint64_t sequence_position,
    uint32_t flags)
{
    return sequence_id * UINT64_C(1000000) +
        sequence_position * UINT64_C(16) +
        (uint64_t)(SparkTestRankDaemonPhaseFromFlags(flags) + 1u);
}

static void SparkTestRankDaemonInitializeRuntime(
    SparkRingDaemonRuntime *runtime)
{
    memset(runtime,0,sizeof(*runtime));
    runtime->cuda_resident_fd = -1;
    runtime->work_listen_fd = -1;
    runtime->work_input_socket_fd = -1;
    runtime->work_output_socket_fd = -1;
    runtime->final_event_listen_fd = -1;
    runtime->final_event_socket_fd = -1;
    runtime->wake_pipe_read_fd = -1;
    runtime->wake_pipe_write_fd = -1;
    SparkRingDaemonInitializeWorkQueue(runtime);
    assert(SparkRingDaemonInitializeCompletionState(runtime) ==
        SPARK_STATUS_OK);
    assert(SparkDistributedWorkInitializeTransactionLedger(
        &runtime->transaction_ledger,
        runtime->transaction_entries,
        runtime->transaction_hash_heads,
        runtime->transaction_hash_next,
        SPARK_RING_DAEMON_TRANSACTION_LEDGER_CAPACITY) == SPARK_STATUS_OK);
}

static void SparkTestRankDaemonObservePacket(
    SparkRingDaemonRuntime *runtime,
    const SparkRingWorkControlPacket *packet)
{
    SparkDistributedWorkIdentity identity;
    SparkStatus terminal_status;
    uint32_t existing_state;
    uint64_t packet_hash;

    assert(SparkRingWorkControlGetTransactionIdentity(
        packet,
        &identity) == SPARK_STATUS_OK);
    packet_hash = SparkDistributedWorkHashBytes(
        packet,
        packet->descriptor_bytes);
    assert(packet_hash != 0u);
    assert(SparkDistributedWorkObserveTransaction(
        &runtime->transaction_ledger,
        &identity,
        packet_hash,
        &existing_state,
        &terminal_status) == SPARK_STATUS_OK);
}

static void SparkTestRankDaemonBuildPacket(
    SparkRingWorkControlPacket *packet,
    uint64_t sequence_id,
    uint64_t sequence_position,
    uint32_t flags)
{
    uint32_t prefill;

    memset(packet,0,sizeof(*packet));
    prefill = (flags & SPARK_RING_WORK_CONTROL_FLAG_PREFILL) != 0u;
    packet->magic = SPARK_RING_WORK_CONTROL_PACKET_MAGIC;
    packet->abi_version = SPARK_RING_WORK_CONTROL_ABI_VERSION;
    packet->descriptor_bytes =
        SparkRingWorkControlCalculatePacketBytes(1u);
    packet->flags = flags;
    packet->control_generation = 7u;
    packet->request_id = sequence_id;
    packet->sequence_id = sequence_id;
    packet->sequence_position = sequence_position;
    packet->active_sequence_count = 1u;
    packet->lane_count = 1u;
    packet->new_token_count = 1u;
    packet->pipeline_slot = 0u;
    packet->block_token_count = 16u;
    packet->kv_block_table_token_count = (uint32_t)sequence_position + 1u;
    packet->max_blocks_per_sequence = 1u;
    packet->rows_per_lane = 1u;
    packet->execution_row_count = 1u;
    packet->execution_batch_bucket = SPARK_STAGE_PLAN_BUCKET_B16;
    packet->input_token_id = 1u;
    packet->prefill_token_ids[0u] = 1u;
    packet->mtp_draft_token_count =
        (flags & SPARK_RING_WORK_CONTROL_FLAG_MTP_DRAFT) != 0u ? 1u : 0u;
    packet->lanes[0u].request_id = sequence_id;
    packet->lanes[0u].request_generation = sequence_id + 1000u;
    packet->lanes[0u].sequence_id = sequence_id;
    packet->lanes[0u].sequence_position = sequence_position;
    packet->lanes[0u].request_slot_index = 0u;
    packet->lanes[0u].context_token_count =
        (uint32_t)sequence_position + 1u;
    packet->lanes[0u].input_token_id = 1u;
    packet->lanes[0u].mtp_draft_token_count =
        packet->mtp_draft_token_count;
    packet->lanes[0u].mtp_resolution_path_id =
        SPARK_MODEL_MTP_TREE_RESOLUTION_NONE;
    if (prefill == 0u)
    {
        packet->prefill_token_ids[0u] = 0u;
    }
    packet->step_chunk_index = 0u;
    packet->step_chunk_count = 1u;
    packet->transaction_phase = SparkRingWorkControlTransactionPhase(packet);
    assert(SparkRingWorkControlSetTransactionIdentity(
        packet,
        packet->control_generation,
        SparkTestRankDaemonTransactionId(
            sequence_id,
            sequence_position,
            flags),
        SparkTestRankDaemonTransactionId(
            sequence_id,
            sequence_position,
            flags),
        sequence_position + 1u) == SPARK_STATUS_OK);
}

static void SparkTestRankDaemonBuildTwoLanePacket(
    SparkRingWorkControlPacket *packet,
    uint64_t first_sequence_id,
    uint64_t sequence_position,
    uint32_t flags)
{
    SparkTestRankDaemonBuildPacket(
        packet,
        first_sequence_id,
        sequence_position,
        flags);
    packet->active_sequence_count = 2u;
    packet->lane_count = 2u;
    packet->execution_row_count = 2u;
    packet->descriptor_bytes = SparkRingWorkControlCalculatePacketBytes(2u);
    packet->lanes[1u] = packet->lanes[0u];
    packet->lanes[1u].request_id = first_sequence_id + 1u;
    packet->lanes[1u].request_generation = first_sequence_id + 2001u;
    packet->lanes[1u].sequence_id = first_sequence_id + 1u;
    packet->lanes[1u].request_slot_index = 1u;
    assert(SparkRingWorkControlSetTransactionIdentity(
        packet,
        packet->control_generation,
        SparkTestRankDaemonTransactionId(
            first_sequence_id,
            sequence_position,
            flags),
        SparkTestRankDaemonTransactionId(
            first_sequence_id,
            sequence_position,
            flags),
        sequence_position + 1u) == SPARK_STATUS_OK);
}

static void SparkTestRankDaemonBuildCompletionRecord(
    const SparkRingWorkControlLane *lane,
    uint32_t token_id,
    SparkRingDaemonDriverCompletionRecord *record)
{
    memset(record,0,sizeof(*record));
    record->completion.request_id = lane->request_id;
    record->completion.sequence_id = lane->sequence_id;
    record->completion.sequence_position = lane->sequence_position;
    record->completion.program_id = 9u;
    record->completion.driver_dispatch_slot = lane->request_slot_index;
    record->completion.accepted_token_count = 1u;
    record->completion.completion_flags =
        SPARK_MODEL_DRIVER_COMPLETION_FLAG_TOKEN_IDS;
    record->completion.token_count = 1u;
    record->completion.token_ids[0u] = token_id;
    record->completion.status = SPARK_STATUS_OK;
}

static uint32_t SparkTestRankDaemonCountInflightCompletionMappings(
    const SparkRingDaemonRuntime *runtime)
{
    uint32_t completion_index;
    uint32_t active_count;

    active_count = 0u;
    for (completion_index = 0u;
         completion_index < SPARK_RING_DAEMON_INFLIGHT_COMPLETION_CAPACITY;
         ++completion_index)
    {
        if (runtime->inflight_completions[completion_index].state ==
            SPARK_RING_DAEMON_INFLIGHT_COMPLETION_STATE_ACTIVE)
        {
            active_count += 1u;
        }
    }
    return active_count;
}

static void SparkTestRankDaemonReadsSplitResidentMessage(void)
{
    static SparkRingDaemonRuntime runtime;
    SparkCudaResidentIpcHeader header;
    SparkCudaResidentIpcHeader received_header;
    SparkCudaResidentIpcSubmitResult payload;
    int32_t sockets[2];
    uint32_t header_split;
    uint32_t payload_split;

    memset(&runtime,0,sizeof(runtime));
    memset(&payload,0,sizeof(payload));
    assert(socketpair(AF_UNIX,SOCK_STREAM,0,sockets) == 0);
    runtime.cuda_resident_fd = sockets[0];
    assert(SparkNetSetNonblocking(sockets[0]) == 0);
    payload.descriptor_bytes = SPARK_CUDA_RESIDENT_IPC_SUBMIT_RESULT_BYTES;
    payload.status = SPARK_STATUS_OK;
    SparkCudaResidentIpcInitializeHeader(
        &header,SPARK_CUDA_RESIDENT_IPC_KIND_SUBMIT_RESULT,
        4u,9u,sizeof(payload));
    header_split = (uint32_t)sizeof(header) / 2u;
    payload_split = (uint32_t)sizeof(payload) / 2u;
    assert(write(sockets[1],&header,header_split) == (ssize_t)header_split);
    assert(SparkRingDaemonReadResidentMessage(
        &runtime,0u,&received_header) == SPARK_STATUS_BUSY);
    assert(runtime.cuda_resident_read_header_offset == header_split);
    assert(write(sockets[1],((const uint8_t *)&header) + header_split,
        sizeof(header) - header_split) ==
        (ssize_t)(sizeof(header) - header_split));
    assert(write(sockets[1],&payload,payload_split) ==
        (ssize_t)payload_split);
    assert(SparkRingDaemonReadResidentMessage(
        &runtime,0u,&received_header) == SPARK_STATUS_BUSY);
    assert(runtime.cuda_resident_read_header_offset == sizeof(header));
    assert(runtime.cuda_resident_read_payload_offset == payload_split);
    assert(write(sockets[1],((const uint8_t *)&payload) + payload_split,
        sizeof(payload) - payload_split) ==
        (ssize_t)(sizeof(payload) - payload_split));
    assert(SparkRingDaemonReadResidentMessage(
        &runtime,0u,&received_header) == SPARK_STATUS_OK);
    assert(received_header.kind ==
        SPARK_CUDA_RESIDENT_IPC_KIND_SUBMIT_RESULT);
    assert(memcmp(runtime.cuda_resident_payload,&payload,sizeof(payload)) == 0);
    assert(runtime.cuda_resident_read_header_offset == 0u);
    assert(runtime.cuda_resident_read_payload_offset == 0u);
    close(sockets[0]);
    close(sockets[1]);
}

static void SparkTestRankDaemonRequestsSubmitResult(void)
{
    static SparkRingDaemonRuntime runtime;
    static SparkCudaResidentIpcSubmitWork submitted;
    SparkCudaResidentIpcSubmitResult result;
    SparkCudaResidentIpcHeader header;
    SparkCudaResidentIpcHeader submitted_header;
    SparkRingWorkControlPacket packet;
    int32_t sockets[2];

    memset(&runtime,0,sizeof(runtime));
    memset(&result,0,sizeof(result));
    memset(&submitted,0,sizeof(submitted));
    SparkTestRankDaemonBuildPacket(&packet,31u,2u,
        SPARK_RING_WORK_CONTROL_FLAG_PREFILL);
    assert(socketpair(AF_UNIX,SOCK_STREAM,0,sockets) == 0);
    runtime.cuda_resident_fd = sockets[0];
    runtime.cuda_resident_socket_path = "attached";
    runtime.rank_plan.rank_index = 1u;
    result.descriptor_bytes =
        SPARK_CUDA_RESIDENT_IPC_SUBMIT_RESULT_BYTES;
    result.status = SPARK_STATUS_OK;
    SparkCudaResidentIpcInitializeHeader(
        &header,SPARK_CUDA_RESIDENT_IPC_KIND_SUBMIT_RESULT,
        runtime.rank_plan.rank_index,1u,sizeof(result));
    assert(write(sockets[1],&header,sizeof(header)) == (ssize_t)sizeof(header));
    assert(write(sockets[1],&result,sizeof(result)) == (ssize_t)sizeof(result));
    assert(SparkRingDaemonSubmitWork(&runtime,&packet) == SPARK_STATUS_OK);
    assert(SparkRingDaemonReadFull(
        sockets[1],&submitted_header,sizeof(submitted_header)) ==
        SPARK_STATUS_OK);
    assert(submitted_header.kind ==
        SPARK_CUDA_RESIDENT_IPC_KIND_SUBMIT_WORK);
    assert(submitted_header.payload_bytes ==
        SPARK_CUDA_RESIDENT_IPC_SUBMIT_WORK_PREFIX_BYTES +
        packet.descriptor_bytes);
    assert(SparkRingDaemonReadFull(
        sockets[1],&submitted,submitted_header.payload_bytes) ==
        SPARK_STATUS_OK);
    assert(submitted.flags ==
        SPARK_CUDA_RESIDENT_IPC_SUBMIT_WORK_FLAG_EXPECT_RESULT);
    assert(SparkCudaResidentIpcValidateSubmitWork(
        &submitted,submitted_header.payload_bytes) == SPARK_STATUS_OK);
    close(sockets[0]);
    close(sockets[1]);
}

static void SparkTestRankDaemonPacketIdentityIncludesPhase(void)
{
    static SparkRingDaemonRuntime runtime;
    SparkRingWorkControlPacket decode;
    SparkRingWorkControlPacket prefill;
    SparkTestRankDaemonInitializeRuntime(&runtime);
    SparkTestRankDaemonBuildPacket(&prefill,41u,8u,
        SPARK_RING_WORK_CONTROL_FLAG_PREFILL);
    decode = prefill;
    decode.flags = SPARK_RING_WORK_CONTROL_FLAG_MTP_DRAFT;
    assert(SparkRingDaemonQueueWork(&runtime,&prefill) == SPARK_STATUS_OK);
    assert(SparkRingDaemonQueueWork(&runtime,&decode) == SPARK_STATUS_OK);
    assert(runtime.work_queue_count == 2u);
    assert(runtime.work_duplicate_count == 0u);
    assert(SparkRingDaemonQueueWork(&runtime,&decode) == SPARK_STATUS_OK);
    assert(runtime.work_queue_count == 2u);
    assert(runtime.work_duplicate_count == 1u);
}

static void SparkTestRankDaemonFindsLaneDependency(void)
{
    static SparkRingDaemonRuntime runtime;
    SparkRingWorkControlPacket current;
    SparkRingWorkControlPacket earlier;
    SparkTestRankDaemonInitializeRuntime(&runtime);
    SparkTestRankDaemonBuildPacket(&earlier,51u,4u,
        SPARK_RING_WORK_CONTROL_FLAG_PREFILL);
    earlier.lane_count = 2u;
    earlier.active_sequence_count = 2u;
    earlier.descriptor_bytes =
        SparkRingWorkControlCalculatePacketBytes(2u);
    earlier.lanes[1u].request_id = 52u;
    earlier.lanes[1u].sequence_id = 52u;
    earlier.lanes[1u].sequence_position = 4u;
    SparkTestRankDaemonBuildPacket(&current,52u,5u,
        SPARK_RING_WORK_CONTROL_FLAG_PREFILL);
    assert(SparkRingDaemonQueueWork(&runtime,&earlier) == SPARK_STATUS_OK);
    assert(SparkRingDaemonQueueWork(&runtime,&current) == SPARK_STATUS_OK);
    assert(SparkRingDaemonHasQueuedDependency(&runtime,
        &runtime.work_queue[1u]) == 1u);
    SparkRingDaemonPopWork(&runtime);
    assert(SparkRingDaemonHasQueuedDependency(&runtime,
        &runtime.work_queue[runtime.work_queue_head]) == 0u);
}

static void SparkTestRankDaemonDecodeWaitsForSamePositionPrefill(void)
{
    static SparkRingDaemonRuntime runtime;
    SparkRingWorkControlPacket decode;
    SparkRingWorkControlPacket prefill;
    SparkTestRankDaemonInitializeRuntime(&runtime);
    SparkTestRankDaemonBuildPacket(&prefill,61u,8u,
        SPARK_RING_WORK_CONTROL_FLAG_PREFILL);
    SparkTestRankDaemonBuildPacket(&decode,61u,8u,
        SPARK_RING_WORK_CONTROL_FLAG_MTP_DRAFT);
    assert(SparkRingDaemonQueueWork(&runtime,&prefill) == SPARK_STATUS_OK);
    assert(SparkRingDaemonQueueWork(&runtime,&decode) == SPARK_STATUS_OK);
    assert(SparkRingDaemonHasQueuedDependency(&runtime,
        &runtime.work_queue[1u]) == 1u);
}

static void SparkTestRankDaemonForwardWaitPreservesFifo(void)
{
    static SparkRingDaemonRuntime runtime;
    SparkRingWorkControlPacket packet;

    SparkTestRankDaemonInitializeRuntime(&runtime);
    runtime.rank_plan.flags = SPARK_RING_RUNTIME_RANK_FLAG_HAS_NEXT;
    SparkTestRankDaemonBuildPacket(
        &packet,
        71u,
        0u,
        SPARK_RING_WORK_CONTROL_FLAG_PREFILL);
    SparkTestRankDaemonObservePacket(&runtime,&packet);
    assert(SparkRingDaemonQueueWork(&runtime,&packet) == SPARK_STATUS_OK);
    packet.sequence_position = 1u;
    packet.lanes[0u].sequence_position = 1u;
    packet.kv_block_table_token_count = 2u;
    packet.lanes[0u].context_token_count = 2u;
    assert(SparkRingWorkControlSetTransactionIdentity(
        &packet,
        packet.control_generation,
        SparkTestRankDaemonTransactionId(
            packet.sequence_id,
            packet.sequence_position,
            packet.flags),
        SparkTestRankDaemonTransactionId(
            packet.sequence_id,
            packet.sequence_position,
            packet.flags),
        packet.sequence_position + 1u) == SPARK_STATUS_OK);
    SparkTestRankDaemonObservePacket(&runtime,&packet);
    assert(SparkRingDaemonQueueWork(&runtime,&packet) == SPARK_STATUS_OK);
    runtime.work_queue_state[runtime.work_queue_head] =
        SPARK_RING_DAEMON_WORK_STATE_WAITING_FORWARD;
    assert(SparkRingDaemonPumpQueuedWork(&runtime) == 0u);
    assert(runtime.work_queue_head == 0u);
    assert(runtime.work_queue_count == 2u);
}

static void SparkTestRankDaemonBackpressuresFullWorkQueue(void)
{
    static SparkRingDaemonRuntime runtime;
    SparkDistributedWorkAcknowledgement acknowledgement;
    SparkRingWorkControlPacket packet;
    int32_t sockets[2];

    SparkTestRankDaemonInitializeRuntime(&runtime);
    assert(socketpair(AF_UNIX,SOCK_STREAM,0,sockets) == 0);
    assert(SparkNetSetNonblocking(sockets[0]) == 0);
    runtime.work_listen_fd = 0;
    runtime.work_input_socket_fd = sockets[0];
    runtime.rank_plan.execution_row_capacity = 1u;
    runtime.work_queue_count = SPARK_RING_DAEMON_WORK_QUEUE_CAPACITY;
    SparkTestRankDaemonBuildPacket(
        &packet,
        81u,
        0u,
        SPARK_RING_WORK_CONTROL_FLAG_PREFILL);
    assert(write(sockets[1],&packet,packet.descriptor_bytes) ==
        (ssize_t)packet.descriptor_bytes);
    assert(SparkRingDaemonWorkInputCanRead(&runtime) == 1u);
    assert(SparkRingDaemonPumpWorkControl(&runtime) == 1u);
    assert(SparkRingDaemonReadFull(
        sockets[1],
        &acknowledgement,
        sizeof(acknowledgement)) == SPARK_STATUS_OK);
    assert(acknowledgement.status ==
        (uint32_t)SPARK_STATUS_CAPACITY_EXCEEDED);
    assert(runtime.work_queue_count == SPARK_RING_DAEMON_WORK_QUEUE_CAPACITY);
    close(sockets[0]);
    close(sockets[1]);
}

static void SparkTestRankDaemonCommittedDuplicateIsAcknowledged(void)
{
    static SparkRingDaemonRuntime runtime;
    SparkDistributedWorkIdentity identity;
    SparkRingWorkControlPacket packet;
    SparkStatus terminal_status;
    uint32_t existing_state;
    uint64_t packet_hash;

    SparkTestRankDaemonInitializeRuntime(&runtime);
    runtime.rank_plan.execution_row_capacity = 1u;
    SparkTestRankDaemonBuildPacket(
        &packet,
        91u,
        0u,
        SPARK_RING_WORK_CONTROL_FLAG_PREFILL);
    packet_hash = SparkDistributedWorkHashBytes(
        &packet,
        packet.descriptor_bytes);
    assert(SparkRingWorkControlGetTransactionIdentity(
        &packet,
        &identity) == SPARK_STATUS_OK);
    assert(SparkDistributedWorkObserveTransaction(
        &runtime.transaction_ledger,
        &identity,
        packet_hash,
        &existing_state,
        &terminal_status) == SPARK_STATUS_OK);
    assert(SparkDistributedWorkTransitionTransaction(
        &runtime.transaction_ledger,
        &identity,
        packet_hash,
        SPARK_DISTRIBUTED_WORK_TRANSACTION_STATE_ACCEPTED,
        SPARK_STATUS_PENDING) == SPARK_STATUS_OK);
    assert(SparkDistributedWorkTransitionTransaction(
        &runtime.transaction_ledger,
        &identity,
        packet_hash,
        SPARK_DISTRIBUTED_WORK_TRANSACTION_STATE_EXECUTING,
        SPARK_STATUS_PENDING) == SPARK_STATUS_OK);
    assert(SparkDistributedWorkTransitionTransaction(
        &runtime.transaction_ledger,
        &identity,
        packet_hash,
        SPARK_DISTRIBUTED_WORK_TRANSACTION_STATE_COMMITTED,
        SPARK_STATUS_OK) == SPARK_STATUS_OK);
    assert(SparkRingDaemonHandleWork(&runtime,&packet) ==
        SPARK_STATUS_DUPLICATE);
    assert(runtime.work_queue_count == 0u);
}

static void SparkTestRankDaemonRejectsIdentityReuseWithDifferentBytes(void)
{
    static SparkRingDaemonRuntime runtime;
    SparkRingWorkControlPacket altered_packet;
    SparkRingWorkControlPacket packet;

    SparkTestRankDaemonInitializeRuntime(&runtime);
    runtime.rank_plan.execution_row_capacity = 1u;
    SparkTestRankDaemonBuildPacket(
        &packet,
        101u,
        0u,
        SPARK_RING_WORK_CONTROL_FLAG_PREFILL);
    assert(SparkRingDaemonHandleWork(&runtime,&packet) == SPARK_STATUS_OK);
    altered_packet = packet;
    altered_packet.pipeline_slot = 1u;
    assert(SparkRingDaemonHandleWork(&runtime,&altered_packet) ==
        SPARK_STATUS_VALIDATION_FAILED);
    assert(runtime.work_queue_count == 1u);
}

static void SparkTestRankDaemonCommitsAfterEveryLaneCompletion(void)
{
    static SparkRingDaemonRuntime runtime;
    SparkDistributedWorkIdentity identity;
    const SparkDistributedWorkTransactionEntry *entry;
    SparkRingDaemonDriverCompletionRecord record;
    SparkRingWorkControlPacket packet;
    uint32_t transaction_index;
    uint64_t packet_hash;

    SparkTestRankDaemonInitializeRuntime(&runtime);
    runtime.rank_plan.flags = SPARK_RING_RUNTIME_RANK_FLAG_FINAL_STAGE;
    runtime.rank_plan.execution_row_capacity = 16u;
    SparkTestRankDaemonBuildTwoLanePacket(
        &packet,
        111u,
        4u,
        SPARK_RING_WORK_CONTROL_FLAG_MTP_DRAFT);
    SparkTestRankDaemonObservePacket(&runtime,&packet);
    assert(SparkRingDaemonTransitionPacket(
        &runtime,
        &packet,
        SPARK_DISTRIBUTED_WORK_TRANSACTION_STATE_ACCEPTED,
        SPARK_STATUS_PENDING) == SPARK_STATUS_OK);
    assert(SparkRingDaemonRegisterInflightTransaction(
        &runtime,
        &packet,
        &transaction_index) == SPARK_STATUS_OK);
    assert(runtime.driver_inflight_count == 2u);
    assert(SparkTestRankDaemonCountInflightCompletionMappings(&runtime) == 2u);
    assert(SparkRingDaemonTransitionPacket(
        &runtime,
        &packet,
        SPARK_DISTRIBUTED_WORK_TRANSACTION_STATE_EXECUTING,
        SPARK_STATUS_PENDING) == SPARK_STATUS_OK);
    assert(SparkRingWorkControlGetTransactionIdentity(
        &packet,
        &identity) == SPARK_STATUS_OK);
    packet_hash = SparkRingWorkControlPacketFingerprint(&packet);
    SparkTestRankDaemonBuildCompletionRecord(
        &packet.lanes[0u],
        701u,
        &record);
    assert(SparkRingDaemonProcessDriverCompletion(&runtime,&record) ==
        SPARK_STATUS_OK);
    assert(runtime.driver_inflight_count == 1u);
    assert(runtime.final_event_queue_count == 1u);
    assert(runtime.final_event_queue[0u].request_generation ==
        packet.lanes[0u].request_generation);
    assert(runtime.final_event_queue[0u].transaction_id ==
        packet.transaction_id);
    assert(SparkDistributedWorkFindTransaction(
        &runtime.transaction_ledger,
        &identity,
        packet_hash,
        &entry) == SPARK_STATUS_OK);
    assert(entry->state ==
        SPARK_DISTRIBUTED_WORK_TRANSACTION_STATE_EXECUTING);
    SparkTestRankDaemonBuildCompletionRecord(
        &packet.lanes[1u],
        702u,
        &record);
    assert(SparkRingDaemonProcessDriverCompletion(&runtime,&record) ==
        SPARK_STATUS_OK);
    assert(runtime.driver_inflight_count == 0u);
    assert(SparkTestRankDaemonCountInflightCompletionMappings(&runtime) == 0u);
    assert(runtime.final_event_queue_count == 2u);
    assert(runtime.final_event_queue[1u].request_generation ==
        packet.lanes[1u].request_generation);
    assert(SparkDistributedWorkFindTransaction(
        &runtime.transaction_ledger,
        &identity,
        packet_hash,
        &entry) == SPARK_STATUS_OK);
    assert(entry->state ==
        SPARK_DISTRIBUTED_WORK_TRANSACTION_STATE_COMMITTED);
}

static void SparkTestRankDaemonRejectsStaleCompletionWithoutLosingOwner(void)
{
    static SparkRingDaemonRuntime runtime;
    SparkRingDaemonDriverCompletionRecord record;
    SparkRingWorkControlPacket packet;
    uint32_t transaction_index;

    SparkTestRankDaemonInitializeRuntime(&runtime);
    SparkTestRankDaemonBuildPacket(
        &packet,
        121u,
        7u,
        SPARK_RING_WORK_CONTROL_FLAG_MTP_DRAFT);
    SparkTestRankDaemonObservePacket(&runtime,&packet);
    assert(SparkRingDaemonTransitionPacket(
        &runtime,
        &packet,
        SPARK_DISTRIBUTED_WORK_TRANSACTION_STATE_ACCEPTED,
        SPARK_STATUS_PENDING) == SPARK_STATUS_OK);
    assert(SparkRingDaemonRegisterInflightTransaction(
        &runtime,
        &packet,
        &transaction_index) == SPARK_STATUS_OK);
    assert(SparkRingDaemonTransitionPacket(
        &runtime,
        &packet,
        SPARK_DISTRIBUTED_WORK_TRANSACTION_STATE_EXECUTING,
        SPARK_STATUS_PENDING) == SPARK_STATUS_OK);
    SparkTestRankDaemonBuildCompletionRecord(
        &packet.lanes[0u],
        703u,
        &record);
    record.completion.sequence_position += 1u;
    assert(SparkRingDaemonProcessDriverCompletion(&runtime,&record) ==
        SPARK_STATUS_VALIDATION_FAILED);
    assert(runtime.driver_inflight_count == 1u);
    assert(SparkTestRankDaemonCountInflightCompletionMappings(&runtime) == 1u);
    assert(SparkRingDaemonCancelInflightTransaction(
        &runtime,
        transaction_index) == SPARK_STATUS_OK);
    assert(runtime.driver_inflight_count == 0u);
    assert(SparkTestRankDaemonCountInflightCompletionMappings(&runtime) == 0u);
}

static void SparkTestRankDaemonSynchronousCompletionWaitsForExecutionState(void)
{
    static SparkRingDaemonRuntime runtime;
    SparkDistributedWorkIdentity identity;
    const SparkDistributedWorkTransactionEntry *entry;
    SparkRingWorkControlPacket packet;
    uint64_t packet_hash;

    SparkTestRankDaemonInitializeRuntime(&runtime);
    runtime.rank_plan.flags = SPARK_RING_RUNTIME_RANK_FLAG_FINAL_STAGE;
    runtime.rank_plan.execution_row_capacity = 16u;
    runtime.builder_state = &runtime;
    runtime.builder_library.builder_interface.submit_work =
        SparkTestRankDaemonSubmitWork;
    SparkTestRankDaemonSubmitStatus = SPARK_STATUS_OK;
    SparkTestRankDaemonSubmitCompletions = 1u;
    SparkTestRankDaemonBuildPacket(
        &packet,
        131u,
        9u,
        SPARK_RING_WORK_CONTROL_FLAG_MTP_DRAFT);
    assert(SparkRingDaemonHandleWork(&runtime,&packet) == SPARK_STATUS_OK);
    assert(SparkRingDaemonPumpQueuedWork(&runtime) == 1u);
    assert(runtime.work_queue_count == 0u);
    assert(runtime.driver_inflight_count == 1u);
    assert(runtime.driver_completion_queue_count == 1u);
    assert(SparkRingWorkControlGetTransactionIdentity(
        &packet,
        &identity) == SPARK_STATUS_OK);
    packet_hash = SparkRingWorkControlPacketFingerprint(&packet);
    assert(SparkDistributedWorkFindTransaction(
        &runtime.transaction_ledger,
        &identity,
        packet_hash,
        &entry) == SPARK_STATUS_OK);
    assert(entry->state ==
        SPARK_DISTRIBUTED_WORK_TRANSACTION_STATE_EXECUTING);
    assert(SparkRingDaemonPumpDriverCompletions(&runtime) == 1u);
    assert(runtime.driver_inflight_count == 0u);
    assert(runtime.driver_completion_queue_count == 0u);
    assert(runtime.final_event_queue_count == 1u);
    assert(runtime.final_event_queue[0u].control_generation ==
        packet.control_generation);
    assert(runtime.final_event_queue[0u].transaction_id ==
        packet.transaction_id);
    assert(runtime.final_event_queue[0u].dispatch_generation ==
        packet.dispatch_generation);
    assert(runtime.final_event_queue[0u].request_generation ==
        packet.lanes[0u].request_generation);
    assert(runtime.final_event_queue[0u].step_generation ==
        packet.step_generation);
    assert(SparkDistributedWorkFindTransaction(
        &runtime.transaction_ledger,
        &identity,
        packet_hash,
        &entry) == SPARK_STATUS_OK);
    assert(entry->state ==
        SPARK_DISTRIBUTED_WORK_TRANSACTION_STATE_COMMITTED);
    SparkTestRankDaemonSubmitCompletions = 0u;
}

static void SparkTestRankDaemonBusySubmitRollsBackCompletionOwnership(void)
{
    static SparkRingDaemonRuntime runtime;
    SparkDistributedWorkIdentity identity;
    const SparkDistributedWorkTransactionEntry *entry;
    SparkRingWorkControlPacket packet;
    uint64_t packet_hash;

    SparkTestRankDaemonInitializeRuntime(&runtime);
    runtime.rank_plan.execution_row_capacity = 16u;
    runtime.builder_state = &runtime;
    runtime.builder_library.builder_interface.submit_work =
        SparkTestRankDaemonSubmitWork;
    SparkTestRankDaemonSubmitStatus = SPARK_STATUS_BUSY;
    SparkTestRankDaemonSubmitCompletions = 0u;
    SparkTestRankDaemonBuildPacket(
        &packet,
        141u,
        3u,
        SPARK_RING_WORK_CONTROL_FLAG_MTP_DRAFT);
    assert(SparkRingDaemonHandleWork(&runtime,&packet) == SPARK_STATUS_OK);
    assert(SparkRingDaemonPumpQueuedWork(&runtime) == 0u);
    assert(runtime.work_queue_count == 1u);
    assert(runtime.driver_inflight_count == 0u);
    assert(SparkTestRankDaemonCountInflightCompletionMappings(&runtime) == 0u);
    assert(SparkRingWorkControlGetTransactionIdentity(
        &packet,
        &identity) == SPARK_STATUS_OK);
    packet_hash = SparkRingWorkControlPacketFingerprint(&packet);
    assert(SparkDistributedWorkFindTransaction(
        &runtime.transaction_ledger,
        &identity,
        packet_hash,
        &entry) == SPARK_STATUS_OK);
    assert(entry->state ==
        SPARK_DISTRIBUTED_WORK_TRANSACTION_STATE_ACCEPTED);
    SparkTestRankDaemonSubmitStatus = SPARK_STATUS_OK;
}

static void SparkTestRankDaemonReleaseCommitsWithoutCompletion(void)
{
    static SparkRingDaemonRuntime runtime;
    SparkDistributedWorkIdentity identity;
    const SparkDistributedWorkTransactionEntry *entry;
    SparkRingWorkControlPacket packet;
    uint64_t packet_hash;

    SparkTestRankDaemonInitializeRuntime(&runtime);
    runtime.rank_plan.execution_row_capacity = 1u;
    runtime.builder_state = &runtime;
    runtime.builder_library.builder_interface.submit_work =
        SparkTestRankDaemonSubmitWork;
    SparkTestRankDaemonSubmitStatus = SPARK_STATUS_OK;
    SparkTestRankDaemonSubmitCompletions = 0u;
    SparkTestRankDaemonBuildPacket(
        &packet,
        151u,
        0u,
        SPARK_RING_WORK_CONTROL_FLAG_RELEASE_SEQUENCES);
    packet.new_token_count = 0u;
    packet.rows_per_lane = 0u;
    packet.execution_row_count = 0u;
    packet.execution_batch_bucket = 0u;
    packet.input_token_id = 0u;
    packet.prefill_token_ids[0u] = 0u;
    packet.mtp_draft_token_count = 0u;
    packet.lanes[0u].input_token_id = 0u;
    packet.lanes[0u].mtp_draft_token_count = 0u;
    packet.lanes[0u].sequence_position = 0u;
    packet.sequence_position = 0u;
    packet.kv_block_table_token_count = 1u;
    packet.lanes[0u].context_token_count = 1u;
    packet.transaction_phase = SparkRingWorkControlTransactionPhase(&packet);
    assert(SparkRingWorkControlSetTransactionIdentity(
        &packet,
        packet.control_generation,
        SparkTestRankDaemonTransactionId(
            packet.sequence_id,
            packet.sequence_position,
            packet.flags),
        SparkTestRankDaemonTransactionId(
            packet.sequence_id,
            packet.sequence_position,
            packet.flags),
        1u) == SPARK_STATUS_OK);
    assert(SparkRingDaemonHandleWork(&runtime,&packet) == SPARK_STATUS_OK);
    assert(SparkRingDaemonPumpQueuedWork(&runtime) == 1u);
    assert(runtime.work_queue_count == 0u);
    assert(runtime.driver_inflight_count == 0u);
    assert(SparkTestRankDaemonCountInflightCompletionMappings(&runtime) == 0u);
    assert(SparkRingWorkControlGetTransactionIdentity(
        &packet,
        &identity) == SPARK_STATUS_OK);
    packet_hash = SparkRingWorkControlPacketFingerprint(&packet);
    assert(SparkDistributedWorkFindTransaction(
        &runtime.transaction_ledger,
        &identity,
        packet_hash,
        &entry) == SPARK_STATUS_OK);
    assert(entry->state ==
        SPARK_DISTRIBUTED_WORK_TRANSACTION_STATE_COMMITTED);
}

int main(void)
{
    SparkTestRankDaemonReadsSplitResidentMessage();
    SparkTestRankDaemonRequestsSubmitResult();
    SparkTestRankDaemonPacketIdentityIncludesPhase();
    SparkTestRankDaemonFindsLaneDependency();
    SparkTestRankDaemonDecodeWaitsForSamePositionPrefill();
    SparkTestRankDaemonForwardWaitPreservesFifo();
    SparkTestRankDaemonBackpressuresFullWorkQueue();
    SparkTestRankDaemonCommittedDuplicateIsAcknowledged();
    SparkTestRankDaemonRejectsIdentityReuseWithDifferentBytes();
    SparkTestRankDaemonCommitsAfterEveryLaneCompletion();
    SparkTestRankDaemonRejectsStaleCompletionWithoutLosingOwner();
    SparkTestRankDaemonSynchronousCompletionWaitsForExecutionState();
    SparkTestRankDaemonBusySubmitRollsBackCompletionOwnership();
    SparkTestRankDaemonReleaseCommitsWithoutCompletion();
    return 0;
}
