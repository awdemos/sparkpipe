// The request surface: session and slot lifecycle, stop conditions, validation,
// and the speculative dispatch policy.
//
// 7,178 lines carrying 34 model constants, which sounds model-specific and is
// not. Measured by concern:
//
//     4,176 lines   6 model constants   session and request plumbing
//     1,603         5                   slot lifecycle
//     1,040        20                   SPECULATIVE DISPATCH POLICY
//       309         2                   stop conditions
//       198         0                   validation
//       137         0                   cost model
//
// Twenty of the thirty-four are in the speculative dispatch policy, and that is
// where they belong: whether drafting beats plain decode depends on the model's
// draft token count, its layer count, and how fast its first layers are. This
// file's own comments record the measurement - plain 3.89 tok/s against MTP 3.47
// at B1, so MTP was LOSING - and note that without hysteresis the scheduler
// oscillates around the break-even point.
//
// WHICH RANK DRAFTS IS A MODEL PROPERTY, NOT A CONSTANT. The policy here assumes
// the last rank, which is where the logits are. For GLM 5.2 that is the wrong
// choice: its first three layers are dense and fast, so the rank holding them
// has slack the last rank does not, and drafting there costs less. A model whose
// early layers are not cheap would want the opposite. That belongs in the model
// description as a list of drafting ranks, and it is not there yet.
#include "sparkpipe/spark_request_api.h"
#include "sparkpipe/spark_mtp_tree.h"
#include "sparkpipe/spark_row_allocator.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// model seam entries - defined in the drafter module, resolved at link
uint32_t SparkRequestModelSlotCanSpeculate(const SparkRequestApi *api, const SparkRequestApiSlot *slot);
uint32_t SparkRequestModelSpeculatorIsValid(const struct SparkRequestModelSpeculator *speculator);
SparkStatus SparkRequestModelGetDraft(SparkRequestApi *api, uint64_t sequence_id, SparkRequestModelDraftResult *result);
SparkStatus SparkRequestModelMarkVerifierTapsReady(SparkRequestApi *api, uint64_t request_id, uint64_t sequence_id, uint64_t sequence_position, uint64_t *tap_generation_out);
uint32_t SparkRequestModelDefaultSpeculativeTokenCount(const SparkRequestApi *api);
SparkStatus SparkRequestModelCompleteVerify(SparkRequestApi *api, uint64_t sequence_id, const SparkRequestModelVerifyResult *verify_result);
SparkStatus SparkRequestModelResolveVerifierTokens(const uint32_t *draft_token_ids, uint32_t draft_token_count, const uint32_t *verifier_token_ids, uint32_t verifier_token_count, SparkRequestModelVerifyResult *verify_result);
SparkStatus SparkRequestModelPrepareDraftForSlot(SparkRequestApi *api, SparkRequestApiSlot *slot);
SparkStatus SparkRequestModelCancelSequence(SparkRequestApi *api, uint64_t sequence_id);
uint32_t SparkRequestModelDsparkSpeculationIsEnabled(
    const SparkRequestApi *api);
SparkStatus SparkRequestModelGetSlotSpeculativeDraft(
    SparkRequestApi *api,
    const SparkRequestApiSlot *slot,
    uint32_t preferred_source,
    SparkRequestModelDraftResult *draft_result,
    uint32_t *source_out);
uint32_t SparkRequestModelMtpOutranksPlainDecode(
    const SparkRequestApi *api,
    uint32_t plain_request_count,
    uint32_t mtp_request_count);
SparkStatus SparkRequestModelReleaseSlotSequence(
    SparkRequestApi *api,
    SparkRequestApiSlot *slot);
SparkStatus SparkRequestModelResolveMtpTreeVerifierTokens(
    const uint32_t *candidate_token_ids,
    const uint32_t *verifier_token_ids,
    SparkRequestModelVerifyResult *verify_result,
    uint32_t *resolution_path_id_out);
void SparkRequestModelRestoreRetriedDecodeCounters(
    SparkRequestApi *api,
    const SparkRequestApiDispatch *dispatch);


static uint32_t SparkRequestApiNormalizeConfigurationFlags(
    uint32_t configuration_flags)
{
    if (configuration_flags == 0u)
    {
        return SPARK_REQUEST_API_CONFIGURATION_DEFAULT_FLAGS;
    }
    return configuration_flags;
}

static uint32_t SparkRequestApiNormalizePrefetchLookaheadRequestCount(
    uint32_t prefetch_lookahead_request_count,
    uint32_t request_capacity)
{
    if (prefetch_lookahead_request_count == 0u)
    {
        prefetch_lookahead_request_count =
            SPARK_REQUEST_API_DEFAULT_PREFETCH_LOOKAHEAD_REQUEST_COUNT;
    }
    if (prefetch_lookahead_request_count > request_capacity)
    {
        return request_capacity;
    }
    return prefetch_lookahead_request_count;
}

static uint32_t SparkRequestApiNormalizePrefetchLaneCount(
    uint32_t prefetch_lane_count)
{
    if (prefetch_lane_count == 0u)
    {
        return SPARK_REQUEST_API_DEFAULT_PREFETCH_LANE_COUNT;
    }
    return prefetch_lane_count;
}

static uint32_t SparkRequestApiNormalizeDecodeBatchTarget(
    uint32_t decode_batch_target)
{
    if (decode_batch_target == 0u)
    {
        return SPARK_REQUEST_API_DEFAULT_DECODE_BATCH_TARGET;
    }
    if (decode_batch_target > SPARK_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT)
    {
        return SPARK_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT;
    }
    return decode_batch_target;
}

static uint32_t SparkGlm52RequestApiNormalizeDecodeExecutionRowCapacity(
    const SparkRequestApiConfiguration *configuration)
{
    uint32_t decode_batch_target;
    uint32_t maximum_speculative_token_count;

    if (configuration == 0)
    {
        return 0u;
    }
    if (configuration->decode_execution_row_capacity != 0u)
    {
        return configuration->decode_execution_row_capacity;
    }
    decode_batch_target = SparkRequestApiNormalizeDecodeBatchTarget(
        configuration->decode_batch_target);
    maximum_speculative_token_count =
        SPARK_REQUEST_MODEL_MAX_SPECULATIVE_TOKENS >
            SPARK_REQUEST_API_MTP_MAX_DRAFT_TOKEN_COUNT
        ? SPARK_REQUEST_MODEL_MAX_SPECULATIVE_TOKENS
        : SPARK_REQUEST_API_MTP_MAX_DRAFT_TOKEN_COUNT;
    return decode_batch_target * (maximum_speculative_token_count + 1u);
}

static uint32_t SparkRequestApiNormalizeMaxResidentKvBlockCount(
    const SparkRequestApiConfiguration *configuration)
{
    uint32_t physical_block_count;

    if (configuration == 0 ||
        configuration->scheduler == 0 ||
        configuration->scheduler->prefix_cache == 0 ||
        configuration->scheduler->prefix_cache->kv_cache_arena == 0)
    {
        return 0u;
    }

    physical_block_count =
        configuration->scheduler->prefix_cache->kv_cache_arena->physical_block_count;
    if (configuration->max_resident_kv_block_count == 0u ||
        configuration->max_resident_kv_block_count >= physical_block_count)
    {
        return 0u;
    }
    return configuration->max_resident_kv_block_count;
}

static uint32_t SparkRequestApiNormalizePriority(
    const SparkRequestApiSubmitRequest *request)
{
    if ((request->flags & SPARK_REQUEST_API_REQUEST_FLAG_REALTIME) != 0u)
    {
        return SPARK_REQUEST_API_REALTIME_PRIORITY;
    }
    if (request->priority == 0u)
    {
        return SPARK_REQUEST_API_DEFAULT_PRIORITY;
    }
    return request->priority;
}

static uint32_t SparkRequestApiConfigurationFlagsAreValid(
    uint32_t configuration_flags)
{
    return (configuration_flags &
        ~SPARK_REQUEST_API_CONFIGURATION_KNOWN_FLAGS) == 0u;
}

static uint32_t SparkRequestApiJitPrefetchIsEnabled(
    const SparkRequestApi *api)
{
    return (api->configuration_flags &
        SPARK_REQUEST_API_CONFIGURATION_FLAG_JIT_KV_PREFETCH) != 0u;
}

static uint32_t SparkRequestApiAsyncJitPrefetchIsEnabled(
    const SparkRequestApi *api)
{
    return (api->configuration_flags &
        SPARK_REQUEST_API_CONFIGURATION_FLAG_ASYNC_JIT_KV_PREFETCH) != 0u;
}

static uint32_t SparkRequestApiQueueAwarePrefixCacheEvictionIsEnabled(
    const SparkRequestApi *api)
{
    return (api->configuration_flags &
        SPARK_REQUEST_API_CONFIGURATION_FLAG_QUEUE_AWARE_PREFIX_CACHE_EVICTION) != 0u;
}

static uint32_t SparkRequestApiMtpCommitIsEnabled(
    const SparkRequestApi *api)
{
    return (api->configuration_flags &
        SPARK_REQUEST_API_CONFIGURATION_FLAG_MTP_COMMIT) != 0u;
}

static uint32_t SparkRequestApiDecodeBatchingIsEnabled(
    const SparkRequestApi *api)
{
    return (api->configuration_flags &
        SPARK_REQUEST_API_CONFIGURATION_FLAG_DECODE_BATCHING) != 0u;
}

static uint32_t SparkRequestApiAdaptivePipelineBatchingIsEnabled(
    const SparkRequestApi *api)
{
    return (api->configuration_flags &
        SPARK_REQUEST_API_CONFIGURATION_FLAG_ADAPTIVE_PIPELINE_BATCHING) != 0u;
}

static uint32_t SparkRequestApiPrefixCohortingIsEnabled(
    const SparkRequestApi *api)
{
    return (api->configuration_flags &
        SPARK_REQUEST_API_CONFIGURATION_FLAG_PREFIX_COHORTING) != 0u;
}

static uint32_t SparkRequestApiCrossSequencePrefixReuseIsEnabled(
    const SparkRequestApi *api)
{
    return api != 0 && api->scheduler != 0 &&
        (api->scheduler->configuration_flags &
            SPARK_SCHEDULER_CONFIGURATION_FLAG_CROSS_SEQUENCE_PREFIX_REUSE) != 0u;
}

static uint32_t SparkRequestApiPrefillBatchingIsEnabled(
    const SparkRequestApi *api)
{
    return (api->configuration_flags &
        SPARK_REQUEST_API_CONFIGURATION_FLAG_PREFILL_BATCHING) != 0u;
}



static SparkStatus SparkRequestApiValidate(
    const SparkRequestApi *api)
{
    if (api == 0 ||
        api->abi_version != SPARK_REQUEST_API_ABI_VERSION ||
        api->descriptor_bytes != SPARK_REQUEST_API_DESCRIPTOR_BYTES ||
        api->request_capacity == 0u ||
        api->decode_batch_target == 0u ||
        api->decode_execution_row_capacity < api->decode_batch_target ||
        api->scheduler == 0 ||
        api->request_slots == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkRequestApiValidateScheduler(
    SparkScheduler *scheduler)
{
    if (scheduler == 0 ||
        scheduler->abi_version != SPARK_SCHEDULER_ABI_VERSION ||
        scheduler->descriptor_bytes != SPARK_SCHEDULER_DESCRIPTOR_BYTES)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52RequestApiValidateConfiguration(
    const SparkRequestApiConfiguration *configuration)
{
    uint32_t configuration_flags;
    uint32_t decode_batch_target;
    uint32_t decode_execution_row_capacity;
    uint32_t prefetch_lane_count;
    SparkStatus status;

    if (configuration == 0 ||
        configuration->abi_version != SPARK_REQUEST_API_ABI_VERSION ||
        configuration->descriptor_bytes !=
            SPARK_REQUEST_API_CONFIGURATION_DESCRIPTOR_BYTES ||
        configuration->request_capacity == 0u ||
        configuration->request_slots == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    configuration_flags = SparkRequestApiNormalizeConfigurationFlags(
        configuration->configuration_flags);
    prefetch_lane_count = SparkRequestApiNormalizePrefetchLaneCount(
        configuration->prefetch_lane_count);
    decode_batch_target = SparkRequestApiNormalizeDecodeBatchTarget(
        configuration->decode_batch_target);
    decode_execution_row_capacity =
        SparkGlm52RequestApiNormalizeDecodeExecutionRowCapacity(configuration);
    if (!SparkRequestApiConfigurationFlagsAreValid(configuration_flags) ||
        prefetch_lane_count == 0u ||
        prefetch_lane_count > SPARK_KV_CACHE_MAX_PREFETCH_LANE_COUNT ||
        decode_execution_row_capacity < decode_batch_target)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    status = SparkRequestApiValidateScheduler(configuration->scheduler);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    if ((configuration_flags &
            SPARK_REQUEST_API_CONFIGURATION_FLAG_ASYNC_JIT_KV_PREFETCH) != 0u &&
        (configuration_flags &
            SPARK_REQUEST_API_CONFIGURATION_FLAG_JIT_KV_PREFETCH) == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if ((configuration_flags &
            SPARK_REQUEST_API_CONFIGURATION_FLAG_QUEUE_AWARE_PREFIX_CACHE_EVICTION) != 0u &&
        configuration->scheduler->prefix_cache == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if ((configuration_flags &
            SPARK_REQUEST_API_CONFIGURATION_FLAG_DSPARK_SPECULATIVE_DECODE) != 0u &&
        (configuration->model_speculator == 0 ||
         SparkRequestModelSpeculatorIsValid(configuration->model_speculator) ==
            SPARK_STATUS_OK))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if ((configuration_flags &
            SPARK_REQUEST_API_CONFIGURATION_FLAG_JIT_KV_PREFETCH) != 0u)
    {
        if (configuration->scheduler->prefix_cache == 0 ||
            configuration->scheduler->prefix_cache->kv_cache_arena == 0)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        if ((configuration_flags &
                SPARK_REQUEST_API_CONFIGURATION_FLAG_ASYNC_JIT_KV_PREFETCH) != 0u)
        {
            if (configuration->kv_prefetch_start_function == 0 ||
                configuration->kv_prefetch_poll_function == 0)
            {
                return SPARK_STATUS_INVALID_ARGUMENT;
            }
        }
        else if (configuration->kv_prefetch_function == 0)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
    }
    return SPARK_STATUS_OK;
}

static void SparkRequestApiInitializeSlot(
    SparkRequestApiSlot *slot)
{
    memset(slot, 0, sizeof(*slot));
    slot->abi_version = SPARK_REQUEST_API_ABI_VERSION;
    slot->descriptor_bytes = SPARK_REQUEST_API_SLOT_DESCRIPTOR_BYTES;
    slot->state = SPARK_REQUEST_API_STATE_FREE;
    slot->handle_hash_next = SPARK_REQUEST_API_NO_SLOT;
    slot->free_slot_next = SPARK_REQUEST_API_NO_SLOT;
    slot->mtp_commit_ema_milli =
        SPARK_REQUEST_API_MTP_COMMIT_EMA_INITIAL_MILLI;
}

static uint32_t SparkRequestApiHashHandle(
    SparkRequestApiHandle handle)
{
    uint64_t hash;

    hash = handle;
    hash ^= (hash >> 33u);
    hash *= 0xff51afd7ed558ccdull;
    hash ^= (hash >> 33u);
    return (uint32_t)(hash % SPARK_REQUEST_API_SLOT_HASH_SLOTS);
}

static uint32_t SparkRequestApiSlotIndex(
    const SparkRequestApi *api,
    const SparkRequestApiSlot *slot)
{
    uint64_t byte_offset;
    uint64_t slot_index;

    if (api == 0 || slot == 0 || api->request_slots == 0 ||
        slot < api->request_slots ||
        slot >= &api->request_slots[api->request_capacity])
    {
        return SPARK_REQUEST_API_NO_SLOT;
    }
    byte_offset = (uint64_t)((uintptr_t)slot - (uintptr_t)api->request_slots);
    slot_index = byte_offset / (uint64_t)sizeof(*slot);
    if (slot_index >= api->request_capacity)
    {
        return SPARK_REQUEST_API_NO_SLOT;
    }
    return (uint32_t)slot_index;
}

static void SparkRequestApiInsertSlotHash(
    SparkRequestApi *api,
    SparkRequestApiSlot *slot)
{
    uint32_t slot_index;
    uint32_t hash_slot;

    slot_index = SparkRequestApiSlotIndex(api, slot);
    if (slot_index == SPARK_REQUEST_API_NO_SLOT ||
        slot->handle == SPARK_REQUEST_API_INVALID_HANDLE)
    {
        return;
    }
    hash_slot = SparkRequestApiHashHandle(slot->handle);
    slot->handle_hash_next = api->slot_handle_hash_heads[hash_slot];
    api->slot_handle_hash_heads[hash_slot] = slot_index;
}

static void SparkRequestApiRemoveSlotHash(
    SparkRequestApi *api,
    SparkRequestApiSlot *slot)
{
    uint32_t slot_index;
    uint32_t hash_slot;
    uint32_t current_slot;
    uint32_t previous_slot;

    slot_index = SparkRequestApiSlotIndex(api, slot);
    if (slot_index == SPARK_REQUEST_API_NO_SLOT ||
        slot->handle == SPARK_REQUEST_API_INVALID_HANDLE)
    {
        return;
    }
    hash_slot = SparkRequestApiHashHandle(slot->handle);
    current_slot = api->slot_handle_hash_heads[hash_slot];
    previous_slot = SPARK_REQUEST_API_NO_SLOT;
    while (current_slot != SPARK_REQUEST_API_NO_SLOT)
    {
        if (current_slot == slot_index)
        {
            if (previous_slot == SPARK_REQUEST_API_NO_SLOT)
            {
                api->slot_handle_hash_heads[hash_slot] =
                    api->request_slots[current_slot].handle_hash_next;
            }
            else
            {
                api->request_slots[previous_slot].handle_hash_next =
                    api->request_slots[current_slot].handle_hash_next;
            }
            api->request_slots[current_slot].handle_hash_next =
                SPARK_REQUEST_API_NO_SLOT;
            return;
        }
        previous_slot = current_slot;
        current_slot = api->request_slots[current_slot].handle_hash_next;
    }
}


SparkStatus SparkRequestApiConfigurationUseAsyncKvCachePrefetchBackend(
    SparkRequestApiConfiguration *configuration,
    SparkKvCacheAsyncPrefetchBackend *backend)
{
    uint32_t configuration_flags;

    if (configuration == 0 || backend == 0 ||
        backend->abi_version != SPARK_KV_CACHE_PREFETCH_BACKEND_ABI_VERSION ||
        backend->descriptor_bytes != SPARK_KV_CACHE_PREFETCH_BACKEND_DESCRIPTOR_BYTES)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    configuration_flags = SparkRequestApiNormalizeConfigurationFlags(
        configuration->configuration_flags);
    configuration_flags |=
        SPARK_REQUEST_API_CONFIGURATION_FLAG_JIT_KV_PREFETCH |
        SPARK_REQUEST_API_CONFIGURATION_FLAG_ASYNC_JIT_KV_PREFETCH;
    configuration->configuration_flags = configuration_flags;
    configuration->kv_prefetch_context = backend;
    configuration->kv_prefetch_function =
        SparkKvCacheAsyncPrefetchBackendSubmitSynchronous;
    configuration->kv_prefetch_start_function =
        SparkKvCacheAsyncPrefetchBackendStart;
    configuration->kv_prefetch_poll_function =
        SparkKvCacheAsyncPrefetchBackendPoll;
    return SPARK_STATUS_OK;
}

SparkStatus SparkRequestApiInitialize(
    SparkRequestApi *api,
    const SparkRequestApiConfiguration *configuration)
{
    uint32_t slot_index;
    uint32_t hash_index;
    uint32_t configuration_flags;
    SparkStatus status;

    if (api == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkGlm52RequestApiValidateConfiguration(configuration);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    configuration_flags = SparkRequestApiNormalizeConfigurationFlags(
        configuration->configuration_flags);
    memset(api, 0, sizeof(*api));
    api->abi_version = SPARK_REQUEST_API_ABI_VERSION;
    api->descriptor_bytes = SPARK_REQUEST_API_DESCRIPTOR_BYTES;
    api->configuration_flags = configuration_flags;
    api->request_capacity = configuration->request_capacity;
    api->prefetch_lookahead_request_count =
        SparkRequestApiNormalizePrefetchLookaheadRequestCount(
            configuration->prefetch_lookahead_request_count,
            configuration->request_capacity);
    api->prefetch_lane_count = SparkRequestApiNormalizePrefetchLaneCount(
        configuration->prefetch_lane_count);
    api->decode_batch_target = SparkRequestApiNormalizeDecodeBatchTarget(
        configuration->decode_batch_target);
    api->max_resident_kv_block_count =
        SparkRequestApiNormalizeMaxResidentKvBlockCount(configuration);
    api->decode_execution_row_capacity =
        SparkGlm52RequestApiNormalizeDecodeExecutionRowCapacity(configuration);
    api->next_handle = 1u;
    api->next_sequence_id = 1u;
    api->next_prefetch_id = 1u;
    api->scheduler = configuration->scheduler;
    api->request_slots = configuration->request_slots;
    api->kv_prefetch_function = configuration->kv_prefetch_function;
    api->kv_prefetch_context = configuration->kv_prefetch_context;
    api->kv_prefetch_start_function =
        configuration->kv_prefetch_start_function;
    api->kv_prefetch_poll_function =
        configuration->kv_prefetch_poll_function;
    api->model_speculator = configuration->model_speculator;

    for (hash_index = 0u;
         hash_index < SPARK_REQUEST_API_SLOT_HASH_SLOTS;
         ++hash_index)
    {
        api->slot_handle_hash_heads[hash_index] =
            SPARK_REQUEST_API_NO_SLOT;
    }
    for (slot_index = 0u; slot_index < api->request_capacity; ++slot_index)
    {
        SparkRequestApiInitializeSlot(&api->request_slots[slot_index]);
        api->request_slots[slot_index].free_slot_next =
            slot_index + 1u < api->request_capacity
                ? slot_index + 1u
                : SPARK_REQUEST_API_NO_SLOT;
    }
    api->free_slot_head = 0u;
    return SPARK_STATUS_OK;
}

static uint32_t SparkRequestApiSlotIsReadyForDispatch(
    const SparkRequestApiSlot *slot)
{
    if (slot->state == SPARK_REQUEST_API_STATE_QUEUED_PREFILL)
    {
        return 1u;
    }
    if ((slot->state == SPARK_REQUEST_API_STATE_READY_DECODE ||
         slot->state == SPARK_REQUEST_API_STATE_READY_SPECULATIVE_VERIFY) &&
        (slot->remaining_thinking_token_budget != 0u ||
         slot->remaining_output_token_budget != 0u))
    {
        return 1u;
    }
    return 0u;
}

static uint32_t SparkRequestApiSlotIsActive(
    const SparkRequestApiSlot *slot)
{
    return slot->state != SPARK_REQUEST_API_STATE_FREE &&
        slot->state != SPARK_REQUEST_API_STATE_COMPLETED &&
        slot->state != SPARK_REQUEST_API_STATE_CANCELLED;
}

uint32_t SparkRequestApiCurrentPipelineBatchWidth(
    const SparkRequestApi *api)
{
    uint32_t highest_ready_priority;
    uint32_t ready_request_count;
    uint32_t slot_index;

    if (api == 0 || api->scheduler == 0 || api->decode_batch_target == 0u)
    {
        return 0u;
    }
    if (!SparkRequestApiAdaptivePipelineBatchingIsEnabled(api))
    {
        return api->decode_batch_target;
    }
    highest_ready_priority = 0u;
    ready_request_count = 0u;
    for (slot_index = 0u; slot_index < api->request_capacity; ++slot_index)
    {
        const SparkRequestApiSlot *slot;

        slot = &api->request_slots[slot_index];
        if (!SparkRequestApiSlotIsReadyForDispatch(slot))
        {
            continue;
        }
        if (slot->priority > highest_ready_priority)
        {
            highest_ready_priority = slot->priority;
        }
    }
    if (highest_ready_priority == 0u)
    {
        return 0u;
    }
    for (slot_index = 0u; slot_index < api->request_capacity; ++slot_index)
    {
        const SparkRequestApiSlot *slot;

        slot = &api->request_slots[slot_index];
        if (SparkRequestApiSlotIsActive(slot) &&
            slot->priority == highest_ready_priority)
        {
            ready_request_count += 1u;
        }
    }
    return SparkSchedulerSelectPipelineBatchWidth(
        api->scheduler,
        ready_request_count,
        api->decode_batch_target);
}

uint32_t SparkRequestApiSlotHasRealtimePriority(
    const SparkRequestApiSlot *slot);

static SparkRequestApiSlot *SparkGlm52RequestApiFindFreeSlot(
    SparkRequestApi *api)
{
    uint32_t slot_index;
    SparkRequestApiSlot *slot;

    if (api == 0 || api->free_slot_head == SPARK_REQUEST_API_NO_SLOT)
    {
        return 0;
    }
    slot_index = api->free_slot_head;
    if (slot_index >= api->request_capacity)
    {
        return 0;
    }
    slot = &api->request_slots[slot_index];
    if (slot->state != SPARK_REQUEST_API_STATE_FREE)
    {
        return 0;
    }
    api->free_slot_head = slot->free_slot_next;
    slot->free_slot_next = SPARK_REQUEST_API_NO_SLOT;
    return slot;
}

static SparkRequestApiSlot *SparkRequestApiFindSlotByHandle(
    SparkRequestApi *api,
    SparkRequestApiHandle handle)
{
    uint32_t hash_slot;
    uint32_t slot_index;

    if (handle == SPARK_REQUEST_API_INVALID_HANDLE)
    {
        return 0;
    }
    hash_slot = SparkRequestApiHashHandle(handle);
    slot_index = api->slot_handle_hash_heads[hash_slot];
    while (slot_index != SPARK_REQUEST_API_NO_SLOT &&
           slot_index < api->request_capacity)
    {
        SparkRequestApiSlot *slot;

        slot = &api->request_slots[slot_index];
        if (slot->state != SPARK_REQUEST_API_STATE_FREE &&
            slot->handle == handle)
        {
            return slot;
        }
        slot_index = slot->handle_hash_next;
    }
    return 0;
}

static SparkStatus SparkRequestApiValidateSubmitRequest(
    const SparkRequestApiSubmitRequest *request)
{
    if (request == 0 ||
        request->abi_version != SPARK_REQUEST_API_ABI_VERSION ||
        request->descriptor_bytes != SPARK_REQUEST_API_SUBMIT_DESCRIPTOR_BYTES ||
        (request->flags & ~SPARK_REQUEST_API_REQUEST_FLAG_KNOWN_FLAGS) != 0u ||
        request->prompt_token_count == 0u ||
        request->prompt_token_ids == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkRequestApiSubmit(
    SparkRequestApi *api,
    const SparkRequestApiSubmitRequest *request,
    SparkRequestApiHandle *handle_out)
{
    SparkRequestApiSlot *slot;
    SparkStatus status;

    status = SparkRequestApiValidate(api);
    if (status != SPARK_STATUS_OK || handle_out == 0)
    {
        return status == SPARK_STATUS_OK ? SPARK_STATUS_INVALID_ARGUMENT : status;
    }
    status = SparkRequestApiValidateSubmitRequest(request);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    slot = SparkGlm52RequestApiFindFreeSlot(api);
    if (slot == 0)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }

    SparkRequestApiInitializeSlot(slot);
    slot->state = SPARK_REQUEST_API_STATE_QUEUED_PREFILL;
    slot->flags = request->flags;
    slot->priority = SparkRequestApiNormalizePriority(request);
    slot->prompt_token_count = request->prompt_token_count;
    slot->thinking_token_budget = request->thinking_token_budget;
    slot->output_token_budget = request->output_token_budget;
    slot->remaining_thinking_token_budget = request->thinking_token_budget;
    slot->remaining_output_token_budget = request->output_token_budget;
    slot->max_prefill_tokens_per_step = request->max_prefill_tokens_per_step;
    slot->mtp_next_draft_token_budget =
        SPARK_REQUEST_API_MTP_INITIAL_DRAFT_TOKEN_COUNT;
    slot->request_id = request->request_id;
    if (request->sequence_id != 0u)
    {
        slot->sequence_id = request->sequence_id;
    }
    else
    {
        slot->sequence_id = api->next_sequence_id;
        api->next_sequence_id += 1u;
    }
    slot->handle = api->next_handle;
    api->next_handle += 1u;
    slot->submission_order = api->submission_counter;
    api->submission_counter += 1u;
    slot->prompt_token_ids = request->prompt_token_ids;
    SparkRequestApiInsertSlotHash(api, slot);

    api->queued_request_count += 1u;
    api->submitted_request_count += 1u;
    *handle_out = slot->handle;
    return SPARK_STATUS_OK;
}

static uint32_t SparkRequestApiSlotIsSchedulablePrefill(
    const SparkRequestApiSlot *slot)
{
    if (slot->state == SPARK_REQUEST_API_STATE_QUEUED_PREFILL)
        return 1u;
    return slot->state == SPARK_REQUEST_API_STATE_RUNNING_PREFILL &&
        slot->dispatched_prompt_token_count < slot->prompt_token_count &&
        slot->inflight_prefill_dispatch_count <
            SPARK_REQUEST_API_PREFILL_INFLIGHT_WAVE_LIMIT;
}

static uint32_t SparkRequestApiSlotIsSchedulableDecode(
    const SparkRequestApiSlot *slot)
{
    return (slot->state == SPARK_REQUEST_API_STATE_READY_DECODE ||
            (slot->state ==
                SPARK_REQUEST_API_STATE_READY_SPECULATIVE_VERIFY &&
             slot->mtp_draft_token_count != 0u)) &&
        (slot->remaining_thinking_token_budget != 0u ||
         slot->remaining_output_token_budget != 0u);
}

static uint32_t SparkRequestApiSlotIsSchedulableSpeculativeVerify(
    const SparkRequestApiSlot *slot)
{
    return slot->state ==
        SPARK_REQUEST_API_STATE_READY_SPECULATIVE_VERIFY &&
        (slot->remaining_thinking_token_budget != 0u ||
         slot->remaining_output_token_budget != 0u);
}



static uint32_t SparkRequestApiMinimumU32(
    uint32_t left,
    uint32_t right);

static uint32_t SparkRequestApiRoundDownToMultiple(
    uint32_t value,
    uint32_t multiple);

static uint32_t SparkRequestApiPrefillCachedBlocksAreResident(
    SparkRequestApi *api,
    const SparkRequestApiSlot *slot,
    uint32_t prompt_token_count);

static uint32_t SparkRequestApiNextPrefillStepTokenCount(
    SparkRequestApi *api,
    const SparkRequestApiSlot *slot,
    uint32_t *computed_prompt_token_count_out);

static uint32_t SparkRequestApiPrefillBlockCountForScheduledTokens(
    const SparkRequestApi *api,
    uint32_t scheduled_prompt_token_count);

static uint32_t SparkRequestApiSlotIsCompatiblePrefillBatchMember(
    SparkRequestApi *api,
    const SparkRequestApiSlot *leader_slot,
    const SparkRequestApiSlot *candidate_slot,
    uint32_t leader_prefill_block_count,
    uint32_t require_resident_cached_blocks,
    uint32_t *candidate_scheduled_prompt_token_count_out);

static uint32_t SparkRequestApiDecodeBlocksAreResident(
    SparkRequestApi *api,
    const SparkRequestApiSlot *slot);

static uint32_t SparkGlm52RequestApiSlotHasHigherSchedulingPriority(
    const SparkRequestApiSlot *candidate,
    const SparkRequestApiSlot *current)
{
    if (current == 0)
    {
        return 1u;
    }
    if (candidate->priority != current->priority)
    {
        return candidate->priority > current->priority;
    }
    return candidate->submission_order < current->submission_order;
}

static uint32_t SparkRequestApiSlotsHaveSameSchedulingPriority(
    const SparkRequestApiSlot *left,
    const SparkRequestApiSlot *right)
{
    return left != 0 && right != 0 && left->priority == right->priority;
}

typedef struct SparkGlm52RequestApiPrefillBatchShape
{
    SparkRequestApiSlot *slot;
    uint32_t scheduled_prompt_token_count;
    uint32_t compatible_request_count;
    uint32_t bucket_capacity;
    uint32_t graph_padding_count;
    uint32_t resident_cached_blocks;
    uint32_t realtime_priority;
    uint32_t prefix_family_shared_token_count;
    uint32_t prefix_family_request_count;
    uint64_t prefix_family_saved_prompt_token_count;
} SparkGlm52RequestApiPrefillBatchShape;

#define SPARK_GLM52_REQUEST_API_PREFIX_FAMILY_GROUP_CAPACITY 256u

typedef struct SparkGlm52RequestApiPrefixFamilyGroup
{
    uint32_t valid;
    uint32_t shared_prefix_token_count;
    uint32_t request_count;
    uint32_t capped_request_count;
    uint32_t highest_priority;
    uint32_t realtime_priority;
    uint64_t prefix_hash;
    SparkRequestApiSlot *leader_slot;
    uint64_t earliest_submission_order;
} SparkGlm52RequestApiPrefixFamilyGroup;

typedef struct SparkGlm52RequestApiPrefixFamilyChoice
{
    SparkRequestApiSlot *leader_slot;
    uint32_t shared_prefix_token_count;
    uint32_t request_count;
    uint32_t realtime_priority;
    uint32_t highest_priority;
    uint64_t saved_prompt_token_count;
} SparkGlm52RequestApiPrefixFamilyChoice;

static uint32_t SparkRequestApiBatchBucketCapacityForSequenceCount(
    uint32_t active_sequence_count)
{
    return SparkStagePlanSelectBatchBucketValue(active_sequence_count);
}

uint32_t SparkRequestApiSlotHasRealtimePriority(
    const SparkRequestApiSlot *slot)
{
    return slot != 0 &&
        ((slot->flags & SPARK_REQUEST_API_REQUEST_FLAG_REALTIME) != 0u ||
         slot->priority >= SPARK_REQUEST_API_REALTIME_PRIORITY);
}

static uint64_t SparkRequestApiPrefixFamilySavedTokenCount(
    uint32_t shared_prefix_token_count,
    uint32_t request_count)
{
    if (shared_prefix_token_count == 0u || request_count < 2u)
    {
        return 0u;
    }
    return (uint64_t)shared_prefix_token_count *
        (uint64_t)(request_count - 1u);
}

static uint32_t SparkRequestApiPrefixFamilyLeaderIsBetter(
    const SparkRequestApiSlot *candidate,
    const SparkRequestApiSlot *current)
{
    if (current == 0)
    {
        return 1u;
    }
    if (candidate->priority != current->priority)
    {
        return candidate->priority > current->priority;
    }
    return candidate->submission_order < current->submission_order;
}

static void SparkRequestApiInitializePrefixFamilyChoice(
    SparkGlm52RequestApiPrefixFamilyChoice *choice)
{
    memset(choice, 0, sizeof(*choice));
}

static uint32_t SparkRequestApiPrefixFamilyGroupIsBetter(
    const SparkGlm52RequestApiPrefixFamilyGroup *candidate,
    const SparkGlm52RequestApiPrefixFamilyGroup *current)
{
    uint64_t candidate_saved_token_count;
    uint64_t current_saved_token_count;

    if (candidate == 0 || candidate->valid == 0u ||
        candidate->capped_request_count < 2u)
    {
        return 0u;
    }
    if (current == 0 || current->valid == 0u ||
        current->capped_request_count < 2u)
    {
        return 1u;
    }

    if (candidate->highest_priority != current->highest_priority)
    {
        return candidate->highest_priority > current->highest_priority;
    }
    if (candidate->realtime_priority != current->realtime_priority)
    {
        return candidate->realtime_priority > current->realtime_priority;
    }

    candidate_saved_token_count = SparkRequestApiPrefixFamilySavedTokenCount(
        candidate->shared_prefix_token_count,
        candidate->capped_request_count);
    current_saved_token_count = SparkRequestApiPrefixFamilySavedTokenCount(
        current->shared_prefix_token_count,
        current->capped_request_count);
    if (candidate_saved_token_count != current_saved_token_count)
    {
        return candidate_saved_token_count > current_saved_token_count;
    }
    if (candidate->capped_request_count != current->capped_request_count)
    {
        return candidate->capped_request_count > current->capped_request_count;
    }
    if (candidate->shared_prefix_token_count != current->shared_prefix_token_count)
    {
        return candidate->shared_prefix_token_count >
            current->shared_prefix_token_count;
    }
    return candidate->earliest_submission_order <
        current->earliest_submission_order;
}

static SparkGlm52RequestApiPrefixFamilyGroup *
SparkRequestApiFindPrefixFamilyGroup(
    SparkGlm52RequestApiPrefixFamilyGroup *groups,
    uint32_t group_count,
    uint64_t prefix_hash,
    uint32_t shared_prefix_token_count,
    uint32_t priority)
{
    uint32_t group_index;

    for (group_index = 0u; group_index < group_count; ++group_index)
    {
        if (groups[group_index].valid != 0u &&
            groups[group_index].prefix_hash == prefix_hash &&
            groups[group_index].shared_prefix_token_count ==
                shared_prefix_token_count &&
            groups[group_index].highest_priority == priority)
        {
            return &groups[group_index];
        }
    }
    return 0;
}

static SparkGlm52RequestApiPrefixFamilyGroup *
SparkRequestApiAcquirePrefixFamilyGroup(
    SparkGlm52RequestApiPrefixFamilyGroup *groups,
    uint32_t *group_count,
    uint64_t prefix_hash,
    uint32_t shared_prefix_token_count,
    uint32_t priority)
{
    SparkGlm52RequestApiPrefixFamilyGroup *group;

    group = SparkRequestApiFindPrefixFamilyGroup(
        groups,
        *group_count,
        prefix_hash,
        shared_prefix_token_count,
        priority);
    if (group != 0)
    {
        return group;
    }
    if (*group_count >= SPARK_GLM52_REQUEST_API_PREFIX_FAMILY_GROUP_CAPACITY)
    {
        return 0;
    }

    group = &groups[*group_count];
    memset(group, 0, sizeof(*group));
    group->valid = 1u;
    group->prefix_hash = prefix_hash;
    group->shared_prefix_token_count = shared_prefix_token_count;
    group->highest_priority = priority;
    group->earliest_submission_order = UINT64_MAX;
    *group_count += 1u;
    return group;
}

static void SparkRequestApiAddSlotToPrefixFamilyGroup(
    SparkGlm52RequestApiPrefixFamilyGroup *group,
    SparkRequestApiSlot *slot)
{
    if (group == 0 || slot == 0)
    {
        return;
    }

    group->request_count += 1u;
    if (group->capped_request_count <
        SPARK_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT)
    {
        group->capped_request_count += 1u;
    }
    if (slot->priority > group->highest_priority)
    {
        group->highest_priority = slot->priority;
    }
    if (SparkRequestApiSlotHasRealtimePriority(slot))
    {
        group->realtime_priority = 1u;
    }
    if (slot->submission_order < group->earliest_submission_order)
    {
        group->earliest_submission_order = slot->submission_order;
    }
    if (SparkRequestApiPrefixFamilyLeaderIsBetter(
            slot,
            group->leader_slot))
    {
        group->leader_slot = slot;
    }
}

static SparkStatus SparkRequestApiExtendPrefixScanHash(
    SparkRequestApiSlot *slot,
    uint32_t block_token_count,
    uint32_t needed_token_count,
    uint64_t *prefix_hash_out)
{
    SparkPrefixCachePromptHash block_hash;
    uint32_t hashed_token_count;
    uint64_t hash_value;
    SparkStatus status;

    hashed_token_count = slot->prefix_scan_hashed_token_count;
    hash_value = slot->prefix_scan_hash;
    if (hashed_token_count == 0u ||
        hashed_token_count > needed_token_count ||
        (hashed_token_count % block_token_count) != 0u)
    {
        hashed_token_count = 0u;
        hash_value = SPARK_PREFIX_CACHE_EMPTY_PARENT_HASH;
    }
    while (hashed_token_count < needed_token_count)
    {
        status = SparkPrefixCacheHashPromptTokens(
            block_token_count,
            hash_value,
            &slot->prompt_token_ids[hashed_token_count],
            block_token_count,
            &block_hash);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        hash_value = block_hash.prompt_hash;
        hashed_token_count += block_token_count;
    }
    slot->prefix_scan_hashed_token_count = hashed_token_count;
    slot->prefix_scan_hash = hash_value;
    *prefix_hash_out = hash_value;
    return SPARK_STATUS_OK;
}

static uint32_t SparkGlm52RequestApiBuildBestPrefixFamilyChoice(
    SparkRequestApi *api,
    SparkGlm52RequestApiPrefixFamilyChoice *choice)
{
    SparkGlm52RequestApiPrefixFamilyGroup groups[
        SPARK_GLM52_REQUEST_API_PREFIX_FAMILY_GROUP_CAPACITY];
    SparkGlm52RequestApiPrefixFamilyGroup *best_group;
    uint32_t group_count;
    uint32_t slot_index;
    uint32_t block_token_count;

    SparkRequestApiInitializePrefixFamilyChoice(choice);
    if (api == 0 ||
        !SparkRequestApiPrefixCohortingIsEnabled(api) ||
        api->scheduler == 0 ||
        api->scheduler->prefix_cache == 0 ||
        api->scheduler->prefix_cache->block_token_count == 0u)
    {
        return 0u;
    }

    memset(groups, 0, sizeof(groups));
    group_count = 0u;
    best_group = 0;
    block_token_count = api->scheduler->prefix_cache->block_token_count;

    for (slot_index = 0u; slot_index < api->request_capacity; ++slot_index)
    {
        SparkRequestApiSlot *slot;
        uint32_t reusable_prefix_token_count;
        uint32_t scheduled_prompt_token_count;
        uint32_t maximum_family_prefix_token_count;
        uint32_t prefix_token_count;

        slot = &api->request_slots[slot_index];
        if (!SparkRequestApiSlotIsSchedulablePrefill(slot) ||
            slot->prompt_token_ids == 0)
        {
            continue;
        }

        scheduled_prompt_token_count = SparkRequestApiNextPrefillStepTokenCount(
            api,
            slot,
            &reusable_prefix_token_count);
        if (scheduled_prompt_token_count == 0u)
        {
            continue;
        }

        maximum_family_prefix_token_count = reusable_prefix_token_count +
            scheduled_prompt_token_count;
        maximum_family_prefix_token_count = SparkRequestApiRoundDownToMultiple(
            SparkRequestApiMinimumU32(
                maximum_family_prefix_token_count,
                slot->prompt_token_count),
            block_token_count);
        prefix_token_count = SparkRequestApiRoundDownToMultiple(
            reusable_prefix_token_count + block_token_count,
            block_token_count);
        if (prefix_token_count <= reusable_prefix_token_count)
        {
            prefix_token_count += block_token_count;
        }

        while (prefix_token_count <= maximum_family_prefix_token_count)
        {
            uint64_t prefix_hash_value;
            SparkGlm52RequestApiPrefixFamilyGroup *group;

            if (SparkRequestApiExtendPrefixScanHash(
                    slot,
                    block_token_count,
                    prefix_token_count,
                    &prefix_hash_value) != SPARK_STATUS_OK)
            {
                break;
            }
            group = SparkRequestApiAcquirePrefixFamilyGroup(
                groups,
                &group_count,
                prefix_hash_value,
                prefix_token_count,
                slot->priority);
            SparkRequestApiAddSlotToPrefixFamilyGroup(group, slot);
            prefix_token_count += block_token_count;
        }
    }

    for (slot_index = 0u; slot_index < group_count; ++slot_index)
    {
        if (SparkRequestApiPrefixFamilyGroupIsBetter(
                &groups[slot_index],
                best_group))
        {
            best_group = &groups[slot_index];
        }
    }

    if (best_group == 0 || best_group->capped_request_count < 2u)
    {
        return 0u;
    }

    choice->leader_slot = best_group->leader_slot;
    choice->shared_prefix_token_count = best_group->shared_prefix_token_count;
    choice->request_count = best_group->capped_request_count;
    choice->realtime_priority = best_group->realtime_priority;
    choice->highest_priority = best_group->highest_priority;
    choice->saved_prompt_token_count =
        SparkRequestApiPrefixFamilySavedTokenCount(
            best_group->shared_prefix_token_count,
            best_group->capped_request_count);
    return choice->leader_slot != 0 && choice->saved_prompt_token_count != 0u;
}

static uint32_t SparkRequestApiPrefixFamilyChoiceBeatsPrefillSlot(
    const SparkGlm52RequestApiPrefixFamilyChoice *choice,
    const SparkRequestApiSlot *slot)
{
    if (choice == 0 || choice->leader_slot == 0 ||
        choice->saved_prompt_token_count == 0u)
    {
        return 0u;
    }
    if (slot == 0)
    {
        return 1u;
    }
    if (choice->leader_slot->priority != slot->priority)
    {
        return choice->leader_slot->priority > slot->priority;
    }
    if (choice->realtime_priority !=
        SparkRequestApiSlotHasRealtimePriority(slot))
    {
        return choice->realtime_priority != 0u;
    }
    return choice->saved_prompt_token_count != 0u;
}

static uint32_t SparkRequestApiEvaluatePrefillBatchShape(
    SparkRequestApi *api,
    SparkRequestApiSlot *slot,
    uint32_t require_resident_cached_blocks,
    SparkGlm52RequestApiPrefillBatchShape *shape)
{
    uint32_t slot_index;
    uint32_t batch_target;
    uint32_t leader_prefill_block_count;

    memset(shape, 0, sizeof(*shape));
    if (!SparkRequestApiSlotIsSchedulablePrefill(slot))
    {
        return 0u;
    }
    if (require_resident_cached_blocks != 0u &&
        !SparkRequestApiPrefillCachedBlocksAreResident(
            api,
            slot,
            slot->prompt_token_count))
    {
        return 0u;
    }

    shape->slot = slot;
    shape->scheduled_prompt_token_count =
        SparkRequestApiNextPrefillStepTokenCount(api, slot, 0);
    if (shape->scheduled_prompt_token_count == 0u)
    {
        return 0u;
    }

    shape->resident_cached_blocks = SparkRequestApiPrefillCachedBlocksAreResident(
        api,
        slot,
        slot->prompt_token_count);
    leader_prefill_block_count =
        SparkRequestApiPrefillBlockCountForScheduledTokens(
            api,
            shape->scheduled_prompt_token_count);
    if (leader_prefill_block_count == 0u)
    {
        return 0u;
    }
    shape->realtime_priority = SparkRequestApiSlotHasRealtimePriority(slot);
    shape->compatible_request_count = 1u;

    if (SparkRequestApiPrefillBatchingIsEnabled(api))
    {
        batch_target = SparkRequestApiCurrentPipelineBatchWidth(api);
        if (batch_target > SPARK_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT)
        {
            batch_target = SPARK_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT;
        }
        for (slot_index = 0u;
             slot_index < api->request_capacity &&
                 shape->compatible_request_count < batch_target;
             ++slot_index)
        {
            SparkRequestApiSlot *candidate;

            candidate = &api->request_slots[slot_index];
            if (SparkRequestApiSlotIsCompatiblePrefillBatchMember(
                    api,
                    slot,
                    candidate,
                    leader_prefill_block_count,
                    1u,
                    0))
            {
                shape->compatible_request_count += 1u;
            }
        }
    }

    shape->bucket_capacity = SparkRequestApiBatchBucketCapacityForSequenceCount(
        shape->compatible_request_count);
    if (shape->bucket_capacity == 0u)
    {
        return 0u;
    }
    shape->graph_padding_count =
        shape->bucket_capacity - shape->compatible_request_count;
    return 1u;
}

static uint32_t SparkRequestApiPrefillShapeIsBetter(
    const SparkGlm52RequestApiPrefillBatchShape *candidate,
    const SparkGlm52RequestApiPrefillBatchShape *current)
{
    if (current->slot == 0)
    {
        return 1u;
    }

    if (candidate->slot->priority != current->slot->priority)
    {
        return candidate->slot->priority > current->slot->priority;
    }
    if (candidate->realtime_priority != current->realtime_priority)
    {
        return candidate->realtime_priority > current->realtime_priority;
    }

    if (candidate->resident_cached_blocks != current->resident_cached_blocks)
    {
        return candidate->resident_cached_blocks > current->resident_cached_blocks;
    }

    if (candidate->compatible_request_count != current->compatible_request_count)
    {
        return candidate->compatible_request_count > current->compatible_request_count;
    }

    if (candidate->graph_padding_count != current->graph_padding_count)
    {
        return candidate->graph_padding_count < current->graph_padding_count;
    }

    if (candidate->scheduled_prompt_token_count !=
        current->scheduled_prompt_token_count)
    {
        return candidate->scheduled_prompt_token_count >
            current->scheduled_prompt_token_count;
    }

    return SparkGlm52RequestApiSlotHasHigherSchedulingPriority(
        candidate->slot,
        current->slot);
}

static SparkRequestApiSlot *SparkRequestApiFindBestPrefillSlot(
    SparkRequestApi *api,
    uint32_t require_resident_cached_blocks)
{
    SparkGlm52RequestApiPrefillBatchShape best_shape;
    uint32_t slot_index;

    memset(&best_shape, 0, sizeof(best_shape));
    for (slot_index = 0u; slot_index < api->request_capacity; ++slot_index)
    {
        SparkGlm52RequestApiPrefillBatchShape candidate_shape;
        SparkRequestApiSlot *slot;

        slot = &api->request_slots[slot_index];
        if (!SparkRequestApiEvaluatePrefillBatchShape(
                api,
                slot,
                require_resident_cached_blocks,
                &candidate_shape))
        {
            continue;
        }
        if (SparkRequestApiPrefillShapeIsBetter(
                &candidate_shape,
                &best_shape))
        {
            best_shape = candidate_shape;
        }
    }
    return best_shape.slot;
}

// The decode and speculative-verify searches differed in one predicate and
// nothing else, so the predicate is the parameter.
static SparkRequestApiSlot *SparkRequestApiFindBestSchedulableSlot(
    SparkRequestApi *api,
    uint32_t (*is_schedulable)(const SparkRequestApiSlot *),
    SparkRequestApiHandle *excluded_handles,
    uint32_t excluded_handle_count,
    uint32_t require_resident_kv)
{
    SparkRequestApiSlot *best_slot;
    uint32_t slot_index;

    best_slot = 0;
    for (slot_index = 0u; slot_index < api->request_capacity; ++slot_index)
    {
        SparkRequestApiSlot *slot;
        uint32_t excluded_index;
        uint32_t is_excluded;

        slot = &api->request_slots[slot_index];
        if (!is_schedulable(slot) ||
            (require_resident_kv != 0u &&
             !SparkRequestApiDecodeBlocksAreResident(api, slot)))
        {
            continue;
        }
        is_excluded = 0u;
        for (excluded_index = 0u;
             excluded_index < excluded_handle_count;
             ++excluded_index)
        {
            if (excluded_handles[excluded_index] == slot->handle)
            {
                is_excluded = 1u;
                break;
            }
        }
        if (is_excluded != 0u)
        {
            continue;
        }
        if (SparkGlm52RequestApiSlotHasHigherSchedulingPriority(slot, best_slot))
        {
            best_slot = slot;
        }
    }
    return best_slot;
}

static void SparkRequestApiInsertBatchMemberByPriority(
    SparkRequestApiSlot **selected_slots,
    uint32_t *selected_count,
    uint32_t selected_capacity,
    SparkRequestApiSlot *slot)
{
    uint32_t insert_index;
    uint32_t shift_index;

    if (*selected_count >= selected_capacity)
    {
        if (!SparkGlm52RequestApiSlotHasHigherSchedulingPriority(
                slot,
                selected_slots[selected_capacity - 1u]))
        {
            return;
        }
        *selected_count = selected_capacity - 1u;
    }
    insert_index = *selected_count;
    while (insert_index > 1u &&
           SparkGlm52RequestApiSlotHasHigherSchedulingPriority(
               slot,
               selected_slots[insert_index - 1u]))
    {
        insert_index -= 1u;
    }
    for (shift_index = *selected_count;
         shift_index > insert_index;
         --shift_index)
    {
        selected_slots[shift_index] = selected_slots[shift_index - 1u];
    }
    selected_slots[insert_index] = slot;
    *selected_count += 1u;
}

static uint32_t SparkRequestApiCollectDecodeBatchMembers(
    SparkRequestApi *api,
    SparkRequestApiSlot *leader_slot,
    uint32_t require_resident_kv,
    SparkRequestApiSlot **selected_slots,
    uint32_t selected_capacity)
{
    uint32_t selected_count;
    uint32_t slot_index;

    selected_slots[0] = leader_slot;
    selected_count = 1u;
    for (slot_index = 0u; slot_index < api->request_capacity; ++slot_index)
    {
        SparkRequestApiSlot *slot;

		slot = &api->request_slots[slot_index];
		if (slot == leader_slot ||
			!SparkRequestApiSlotIsSchedulableDecode(slot) ||
			SparkRequestModelSlotCanSpeculate(api,slot) !=
				SparkRequestModelSlotCanSpeculate(api,leader_slot))
		{
			continue;
		}
        if ((require_resident_kv != 0u ||
             slot->priority < leader_slot->priority) &&
            !SparkRequestApiDecodeBlocksAreResident(api, slot))
        {
            continue;
        }
        SparkRequestApiInsertBatchMemberByPriority(
            selected_slots,
            &selected_count,
            selected_capacity,
            slot);
    }
    return selected_count;
}

static uint32_t SparkRequestApiMinimumU32(
    uint32_t left,
    uint32_t right)
{
    return left < right ? left : right;
}

static uint32_t SparkRequestApiMaximumU32(
    uint32_t left,
    uint32_t right)
{
    return left > right ? left : right;
}

static uint32_t SparkRequestApiRoundDownToMultiple(
    uint32_t value,
    uint32_t multiple)
{
    if (multiple == 0u)
    {
        return 0u;
    }
    return value - (value % multiple);
}

static uint32_t SparkRequestApiCountCommonPrefixTokens(
    const SparkRequestApiSlot *left,
    const SparkRequestApiSlot *right)
{
    uint32_t shared_token_count;
    uint32_t token_index;

    if (left == 0 || right == 0 ||
        left->prompt_token_ids == 0 || right->prompt_token_ids == 0)
    {
        return 0u;
    }
    shared_token_count = SparkRequestApiMinimumU32(
        left->prompt_token_count,
        right->prompt_token_count);
    for (token_index = 0u;
         token_index < shared_token_count;
         ++token_index)
    {
        if (left->prompt_token_ids[token_index] !=
            right->prompt_token_ids[token_index])
        {
            return token_index;
        }
    }
    return shared_token_count;
}

static uint32_t SparkRequestApiSharedCachePrefixTokenCount(
    SparkRequestApi *api,
    const SparkRequestApiSlot *left,
    const SparkRequestApiSlot *right)
{
    uint32_t block_token_count;
    uint32_t common_prefix_token_count;

    if (SparkRequestApiCrossSequencePrefixReuseIsEnabled(api) == 0u ||
        api->scheduler->prefix_cache == 0)
    {
        return 0u;
    }
    block_token_count = api->scheduler->prefix_cache->block_token_count;
    common_prefix_token_count = SparkRequestApiCountCommonPrefixTokens(
        left,
        right);
    common_prefix_token_count = SparkRequestApiRoundDownToMultiple(
        common_prefix_token_count,
        block_token_count);
    if (common_prefix_token_count <= SparkRequestApiMaximumU32(
            left->computed_prompt_token_count,
            right->computed_prompt_token_count))
    {
        return 0u;
    }
    return common_prefix_token_count;
}

static uint32_t SparkRequestApiPrefillCachedBlocksAreResident(
    SparkRequestApi *api,
    const SparkRequestApiSlot *slot,
    uint32_t prompt_token_count)
{
    uint32_t matched_token_count;
    uint32_t resident_block_count;
    uint32_t nonresident_block_count;
    SparkStatus status;

    if (!SparkRequestApiJitPrefetchIsEnabled(api) ||
        api->scheduler == 0 ||
        api->scheduler->prefix_cache == 0 ||
        slot == 0 ||
        prompt_token_count == 0u)
    {
        return 1u;
    }

    status = SparkPrefixCacheProbeReusablePrefixResidency(
        api->scheduler->prefix_cache,
        slot->prompt_token_ids,
        prompt_token_count,
        &matched_token_count,
        &resident_block_count,
        &nonresident_block_count);
    if (status != SPARK_STATUS_OK)
    {
        return 0u;
    }
    (void)matched_token_count;
    (void)resident_block_count;
    return nonresident_block_count == 0u;
}

static SparkStatus SparkGlm52RequestApiPendingSpeculativeTokenCount(
    SparkRequestApi *api,
    const SparkRequestApiSlot *slot,
    uint32_t *speculative_token_count_out)
{
    SparkRequestModelDraftResult draft_result;
    SparkStatus status;

    if (api == 0 || slot == 0 || speculative_token_count_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *speculative_token_count_out = 0u;
    if (!SparkRequestApiSlotIsSchedulableSpeculativeVerify(slot) &&
        slot->state !=
            SPARK_REQUEST_API_STATE_RUNNING_SPECULATIVE_VERIFY)
    {
        return SPARK_STATUS_OK;
    }
    if (slot->mtp_draft_token_count != 0u)
    {
        if (slot->mtp_draft_token_count >
            SPARK_REQUEST_API_MTP_MAX_DRAFT_TOKEN_COUNT)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        *speculative_token_count_out = slot->mtp_draft_token_count;
        return SPARK_STATUS_OK;
    }
    if (!SparkRequestModelDsparkSpeculationIsEnabled(api))
    {
        return SPARK_STATUS_NOT_FOUND;
    }
    status = SparkRequestModelGetDraft(
        api,
        slot->sequence_id,
        &draft_result);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (draft_result.token_count == 0u ||
        draft_result.token_count >
            SPARK_REQUEST_MODEL_MAX_SPECULATIVE_TOKENS)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *speculative_token_count_out = draft_result.token_count;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkRequestApiRequiredDecodeKvTokenCount(
    const SparkRequestApiSlot *slot,
    uint32_t speculative_token_count,
    uint32_t *required_token_count_out)
{
    uint64_t required_token_count;

    if (slot == 0 || required_token_count_out == 0 ||
        slot->computed_prompt_token_count == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    required_token_count =
        (uint64_t)slot->computed_prompt_token_count +
        (uint64_t)slot->completed_decode_token_count + 1u +
        (uint64_t)speculative_token_count;
    if (required_token_count > UINT32_MAX ||
        required_token_count > SPARK_SCHEDULER_MAX_CONTEXT_TOKENS)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    *required_token_count_out = (uint32_t)required_token_count;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkRequestApiApplyActiveKvBlockBudget(
    SparkRequestApi *api,
    SparkRequestApiSlot **selected_slots,
    uint32_t *selected_count,
    uint32_t additional_token_count)
{
    uint64_t selected_block_count;
    uint32_t block_token_count;
    uint32_t input_index;
    uint32_t output_count;

    if (api == 0 || selected_slots == 0 || selected_count == 0 ||
        *selected_count == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (api->max_resident_kv_block_count == 0u)
    {
        return SPARK_STATUS_OK;
    }
    if (api->scheduler == 0 || api->scheduler->prefix_cache == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    block_token_count = api->scheduler->prefix_cache_block_tokens;
    if (block_token_count == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    selected_block_count = 0u;
    output_count = 0u;
    for (input_index = 0u; input_index < *selected_count; ++input_index)
    {
        uint64_t required_block_count;
        uint32_t required_token_count;
        SparkStatus status;

        status = SparkRequestApiRequiredDecodeKvTokenCount(
            selected_slots[input_index],
            additional_token_count,
            &required_token_count);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        required_block_count =
            ((uint64_t)required_token_count + block_token_count - 1u) /
            block_token_count;
        if (required_block_count > api->max_resident_kv_block_count)
        {
            if (input_index == 0u)
            {
                return SPARK_STATUS_CAPACITY_EXCEEDED;
            }
            continue;
        }
        if (selected_block_count >
            api->max_resident_kv_block_count - required_block_count)
        {
            continue;
        }
        selected_slots[output_count++] = selected_slots[input_index];
        selected_block_count += required_block_count;
    }
    if (output_count == 0u)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    *selected_count = output_count;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkRequestApiEnsureDecodeSlotKvCapacity(
    SparkRequestApi *api,
    SparkRequestApiSlot *slot,
    uint32_t speculative_token_count,
    uint32_t *required_token_count_out)
{
    uint32_t required_token_count;
    SparkStatus status;

    if (api == 0 || api->scheduler == 0 ||
        api->scheduler->prefix_cache == 0 || slot == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkRequestApiRequiredDecodeKvTokenCount(
        slot,
        speculative_token_count,
        &required_token_count);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkPrefixCacheEnsureSequenceTokenCapacity(
        api->scheduler->prefix_cache,
        slot->sequence_id,
        required_token_count);
    if (status == SPARK_STATUS_OK && required_token_count_out != 0)
    {
        *required_token_count_out = required_token_count;
    }
    return status;
}

static SparkStatus SparkRequestApiEnsurePendingDecodeSlotKvCapacity(
    SparkRequestApi *api,
    SparkRequestApiSlot *slot,
    uint32_t *required_token_count_out)
{
    uint32_t speculative_token_count;
    SparkStatus status;

    status = SparkGlm52RequestApiPendingSpeculativeTokenCount(
        api,
        slot,
        &speculative_token_count);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    return SparkRequestApiEnsureDecodeSlotKvCapacity(
        api,
        slot,
        speculative_token_count,
        required_token_count_out);
}

static uint32_t SparkRequestApiDecodeBlocksAreResident(
    SparkRequestApi *api,
    const SparkRequestApiSlot *slot)
{
    uint32_t required_token_count;
    uint32_t physical_block_count;
    uint32_t resident_block_count;
    uint32_t nonresident_block_count;
    SparkStatus status;

    if (api->scheduler == 0 ||
        api->scheduler->prefix_cache == 0 ||
        slot == 0 ||
        slot->computed_prompt_token_count == 0u)
    {
        return 1u;
    }

    status = SparkRequestApiEnsurePendingDecodeSlotKvCapacity(
        api,
        (SparkRequestApiSlot *)slot,
        &required_token_count);
    if (status != SPARK_STATUS_OK)
    {
        return 0u;
    }
    if (!SparkRequestApiJitPrefetchIsEnabled(api))
    {
        return 1u;
    }

    status = SparkPrefixCacheProbeSequenceResidency(
        api->scheduler->prefix_cache,
        slot->sequence_id,
        required_token_count,
        &physical_block_count,
        &resident_block_count,
        &nonresident_block_count);
    if (status != SPARK_STATUS_OK)
    {
        return 0u;
    }
    (void)physical_block_count;
    (void)resident_block_count;
    return nonresident_block_count == 0u;
}

static uint32_t SparkRequestApiOlderLowerPrioritySchedulableSlotExists(
    SparkRequestApi *api,
    const SparkRequestApiSlot *chosen_slot)
{
    uint32_t slot_index;

    for (slot_index = 0u; slot_index < api->request_capacity; ++slot_index)
    {
        SparkRequestApiSlot *slot;

        slot = &api->request_slots[slot_index];
        if ((SparkRequestApiSlotIsSchedulablePrefill(slot) ||
             SparkRequestApiSlotIsSchedulableDecode(slot)) &&
            slot->submission_order < chosen_slot->submission_order &&
            chosen_slot->priority > slot->priority)
        {
            return 1u;
        }
    }
    return 0u;
}

static void SparkRequestApiCollectPhysicalBlockIndex(
    uint32_t *physical_block_indices,
    uint32_t *physical_block_count,
    uint32_t physical_block_index)
{
    uint32_t block_index;

    for (block_index = 0u; block_index < *physical_block_count; ++block_index)
    {
        if (physical_block_indices[block_index] == physical_block_index)
        {
            return;
        }
    }
    if (*physical_block_count <
        SPARK_REQUEST_API_MAX_PREFETCH_SOURCE_BLOCK_COUNT)
    {
        physical_block_indices[*physical_block_count] = physical_block_index;
        *physical_block_count += 1u;
    }
}

static SparkStatus SparkRequestApiCollectPrefillSlotBlocks(
    SparkRequestApi *api,
    SparkRequestApiSlot *slot,
    uint32_t *physical_block_indices,
    uint32_t *physical_block_count)
{
    uint32_t matched_token_count;
    uint32_t slot_physical_block_count;
    uint32_t slot_physical_block_indices[
        SPARK_REQUEST_API_MAX_PREFETCH_SOURCE_BLOCK_COUNT];
    uint32_t block_index;
    SparkStatus status;

    status = SparkPrefixCacheProbePhysicalBlockTable(
        api->scheduler->prefix_cache,
        slot->prompt_token_ids,
        slot->prompt_token_count,
        slot_physical_block_indices,
        SPARK_REQUEST_API_MAX_PREFETCH_SOURCE_BLOCK_COUNT,
        &matched_token_count,
        &slot_physical_block_count);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    (void)matched_token_count;
    for (block_index = 0u;
         block_index < slot_physical_block_count &&
             *physical_block_count <
                 SPARK_REQUEST_API_MAX_PREFETCH_SOURCE_BLOCK_COUNT;
         ++block_index)
    {
        SparkRequestApiCollectPhysicalBlockIndex(
            physical_block_indices,
            physical_block_count,
            slot_physical_block_indices[block_index]);
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkRequestApiCollectDecodeSlotBlocks(
    SparkRequestApi *api,
    SparkRequestApiSlot *slot,
    uint32_t *physical_block_indices,
    uint32_t *physical_block_count)
{
    uint32_t required_token_count;
    uint32_t slot_physical_block_count;
    uint32_t slot_physical_block_indices[
        SPARK_REQUEST_API_MAX_PREFETCH_SOURCE_BLOCK_COUNT];
    uint32_t block_index;
    SparkStatus status;

    status = SparkRequestApiEnsurePendingDecodeSlotKvCapacity(
        api,
        slot,
        &required_token_count);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkPrefixCacheBuildPhysicalBlockTable(
        api->scheduler->prefix_cache,
        slot->sequence_id,
        required_token_count,
        slot_physical_block_indices,
        SPARK_REQUEST_API_MAX_PREFETCH_SOURCE_BLOCK_COUNT,
        &slot_physical_block_count);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    for (block_index = 0u;
         block_index < slot_physical_block_count &&
             *physical_block_count <
                 SPARK_REQUEST_API_MAX_PREFETCH_SOURCE_BLOCK_COUNT;
         ++block_index)
    {
        SparkRequestApiCollectPhysicalBlockIndex(
            physical_block_indices,
            physical_block_count,
            slot_physical_block_indices[block_index]);
    }
    return SPARK_STATUS_OK;
}

static void SparkRequestApiCollectPrefetchSourceBlock(
    SparkKvCachePrefetchSourceBlock *source_blocks,
    uint32_t *source_block_count,
    const SparkKvCachePrefetchSourceBlock *source_block)
{
    uint32_t block_index;

    for (block_index = 0u; block_index < *source_block_count; ++block_index)
    {
        if (source_blocks[block_index].physical_block_index ==
                source_block->physical_block_index &&
            source_blocks[block_index].block_hash == source_block->block_hash &&
            source_blocks[block_index].content_hash == source_block->content_hash)
        {
            return;
        }
    }
    if (*source_block_count <
        SPARK_REQUEST_API_MAX_PREFETCH_SOURCE_BLOCK_COUNT)
    {
        source_blocks[*source_block_count] = *source_block;
        *source_block_count += 1u;
    }
}

static SparkStatus SparkRequestApiCollectPrefillSlotPrefetchSources(
    SparkRequestApi *api,
    SparkRequestApiSlot *slot,
    SparkKvCachePrefetchSourceBlock *source_blocks,
    uint32_t *source_block_count)
{
    SparkKvCachePrefetchSourceBlock slot_source_blocks[
        SPARK_REQUEST_API_MAX_PREFETCH_SOURCE_BLOCK_COUNT];
    uint32_t matched_token_count;
    uint32_t slot_source_block_count;
    uint32_t block_index;
    SparkStatus status;

    status = SparkPrefixCacheProbeReusablePrefixPrefetchSources(
        api->scheduler->prefix_cache,
        slot->prompt_token_ids,
        slot->prompt_token_count,
        slot_source_blocks,
        SPARK_REQUEST_API_MAX_PREFETCH_SOURCE_BLOCK_COUNT,
        &matched_token_count,
        &slot_source_block_count);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    (void)matched_token_count;
    for (block_index = 0u;
         block_index < slot_source_block_count &&
             *source_block_count <
                 SPARK_REQUEST_API_MAX_PREFETCH_SOURCE_BLOCK_COUNT;
         ++block_index)
    {
        SparkRequestApiCollectPrefetchSourceBlock(
            source_blocks,
            source_block_count,
            &slot_source_blocks[block_index]);
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkRequestApiCollectDecodeSlotPrefetchSources(
    SparkRequestApi *api,
    SparkRequestApiSlot *slot,
    SparkKvCachePrefetchSourceBlock *source_blocks,
    uint32_t *source_block_count)
{
    uint32_t required_token_count;
    SparkKvCachePrefetchSourceBlock slot_source_blocks[
        SPARK_REQUEST_API_MAX_PREFETCH_SOURCE_BLOCK_COUNT];
    uint32_t slot_source_block_count;
    uint32_t block_index;
    SparkStatus status;

    status = SparkRequestApiEnsurePendingDecodeSlotKvCapacity(
        api,
        slot,
        &required_token_count);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkPrefixCacheBuildSequencePrefetchSources(
        api->scheduler->prefix_cache,
        slot->sequence_id,
        required_token_count,
        slot_source_blocks,
        SPARK_REQUEST_API_MAX_PREFETCH_SOURCE_BLOCK_COUNT,
        &slot_source_block_count);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    for (block_index = 0u;
         block_index < slot_source_block_count &&
             *source_block_count <
                 SPARK_REQUEST_API_MAX_PREFETCH_SOURCE_BLOCK_COUNT;
         ++block_index)
    {
        SparkRequestApiCollectPrefetchSourceBlock(
            source_blocks,
            source_block_count,
            &slot_source_blocks[block_index]);
    }
    return SPARK_STATUS_OK;
}

static SparkRequestApiSlot *SparkRequestApiFindBestPrefetchLookaheadSlot(
    SparkRequestApi *api,
    SparkRequestApiHandle *selected_handles,
    uint32_t selected_handle_count)
{
    SparkRequestApiSlot *best_slot;
    uint32_t slot_index;

    best_slot = 0;
    for (slot_index = 0u; slot_index < api->request_capacity; ++slot_index)
    {
        SparkRequestApiSlot *slot;
        uint32_t selected_index;
        uint32_t is_selected;

        slot = &api->request_slots[slot_index];
        if (!SparkRequestApiSlotIsSchedulablePrefill(slot) &&
            !SparkRequestApiSlotIsSchedulableDecode(slot) &&
            !SparkRequestApiSlotIsSchedulableSpeculativeVerify(slot))
        {
            continue;
        }
        is_selected = 0u;
        for (selected_index = 0u;
             selected_index < selected_handle_count;
             ++selected_index)
        {
            if (selected_handles[selected_index] == slot->handle)
            {
                is_selected = 1u;
                break;
            }
        }
        if (is_selected != 0u)
        {
            continue;
        }
        if (SparkGlm52RequestApiSlotHasHigherSchedulingPriority(slot, best_slot))
        {
            best_slot = slot;
        }
    }
    return best_slot;
}

static SparkStatus SparkRequestApiRefreshLookaheadPrefixProtections(
    SparkRequestApi *api)
{
    SparkRequestApiHandle selected_handles[
        SPARK_REQUEST_API_MAX_PREFETCH_SOURCE_BLOCK_COUNT];
    uint32_t selected_handle_count;
    uint32_t lookahead_index;
    uint32_t total_protected_block_count;
    SparkStatus status;

    if (!SparkRequestApiQueueAwarePrefixCacheEvictionIsEnabled(api))
    {
        return SPARK_STATUS_OK;
    }
    if (api->scheduler == 0 || api->scheduler->prefix_cache == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    status = SparkPrefixCacheResetLookaheadProtection(
        api->scheduler->prefix_cache);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    selected_handle_count = 0u;
    total_protected_block_count = 0u;
    for (lookahead_index = 0u;
         lookahead_index < api->prefetch_lookahead_request_count &&
             selected_handle_count <
                SPARK_REQUEST_API_MAX_PREFETCH_SOURCE_BLOCK_COUNT;
         ++lookahead_index)
    {
        SparkRequestApiSlot *slot;
        uint32_t protected_token_count;
        uint32_t protected_block_count;

        slot = SparkRequestApiFindBestPrefetchLookaheadSlot(
            api,
            selected_handles,
            selected_handle_count);
        if (slot == 0)
        {
            break;
        }
        selected_handles[selected_handle_count] = slot->handle;
        selected_handle_count += 1u;

        if (slot->prompt_token_ids == 0 || slot->prompt_token_count == 0u)
        {
            continue;
        }
        status = SparkPrefixCacheProtectPromptLookahead(
            api->scheduler->prefix_cache,
            slot->prompt_token_ids,
            slot->prompt_token_count,
            slot->priority,
            &protected_token_count,
            &protected_block_count);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        total_protected_block_count += protected_block_count;
    }

    api->lookahead_protection_sweep_count += 1u;
    api->lookahead_protected_block_count += total_protected_block_count;
    return SPARK_STATUS_OK;
}

static uint32_t SparkRequestApiJitResidencyPolicyIsEnabled(
    const SparkRequestApi *api)
{
    return SparkRequestApiJitPrefetchIsEnabled(api) &&
        api->max_resident_kv_block_count != 0u &&
        api->scheduler != 0 &&
        api->scheduler->prefix_cache != 0 &&
        api->scheduler->prefix_cache->kv_cache_arena != 0;
}

static void SparkRequestApiCollectProtectedPrefetchPlanBlocks(
    const SparkKvCachePrefetchPlan *prefetch_plan,
    uint32_t *protected_physical_block_indices,
    uint32_t *protected_physical_block_count)
{
    uint32_t block_index;

    if (prefetch_plan == 0)
    {
        return;
    }
    for (block_index = 0u;
         block_index < prefetch_plan->prefetch_block_count &&
             *protected_physical_block_count <
                SPARK_REQUEST_API_MAX_PREFETCH_SOURCE_BLOCK_COUNT;
         ++block_index)
    {
        SparkRequestApiCollectPhysicalBlockIndex(
            protected_physical_block_indices,
            protected_physical_block_count,
            prefetch_plan->blocks[block_index].physical_block_index);
    }
}

static SparkStatus SparkRequestApiCollectProtectedSlotBlocks(
    SparkRequestApi *api,
    SparkRequestApiSlot *slot,
    uint32_t *protected_physical_block_indices,
    uint32_t *protected_physical_block_count)
{
    if (slot == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkRequestApiSlotIsSchedulablePrefill(slot))
    {
        return SparkRequestApiCollectPrefillSlotBlocks(
            api,
            slot,
            protected_physical_block_indices,
            protected_physical_block_count);
    }
    if ((SparkRequestApiSlotIsSchedulableDecode(slot) ||
         SparkRequestApiSlotIsSchedulableSpeculativeVerify(slot) ||
         slot->state == SPARK_REQUEST_API_STATE_RUNNING_DECODE ||
         slot->state == SPARK_REQUEST_API_STATE_RUNNING_SPECULATIVE_VERIFY ||
         slot->state == SPARK_REQUEST_API_STATE_RUNNING_PREFILL ||
         slot->state == SPARK_REQUEST_API_STATE_WAITING_PREFIX_COHORT) &&
        slot->computed_prompt_token_count != 0u)
    {
        return SparkRequestApiCollectDecodeSlotBlocks(
            api,
            slot,
            protected_physical_block_indices,
            protected_physical_block_count);
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkRequestApiCollectRunningProtectedBlocks(
    SparkRequestApi *api,
    uint32_t *protected_physical_block_indices,
    uint32_t *protected_physical_block_count)
{
    uint32_t slot_index;

    for (slot_index = 0u;
         slot_index < api->request_capacity &&
             *protected_physical_block_count <
                SPARK_REQUEST_API_MAX_PREFETCH_SOURCE_BLOCK_COUNT;
         ++slot_index)
    {
        SparkRequestApiSlot *slot;
        SparkStatus status;

        slot = &api->request_slots[slot_index];
        if (slot->state != SPARK_REQUEST_API_STATE_RUNNING_PREFILL &&
            slot->state != SPARK_REQUEST_API_STATE_RUNNING_DECODE &&
            slot->state != SPARK_REQUEST_API_STATE_WAITING_PREFIX_COHORT)
        {
            continue;
        }
        status = SparkRequestApiCollectProtectedSlotBlocks(
            api,
            slot,
            protected_physical_block_indices,
            protected_physical_block_count);
        if (status != SPARK_STATUS_OK && status != SPARK_STATUS_NOT_FOUND)
        {
            return status;
        }
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkRequestApiApplyJitKvResidencyPolicy(
    SparkRequestApi *api,
    const SparkKvCachePrefetchPlan *protected_prefetch_plan,
    const uint32_t *additional_protected_physical_block_indices,
    uint32_t additional_protected_physical_block_count)
{
    uint32_t protected_physical_block_indices[
        SPARK_REQUEST_API_MAX_PREFETCH_SOURCE_BLOCK_COUNT];
    uint32_t protected_physical_block_count;
    uint32_t pending_index;
    uint32_t evicted_block_count;
    SparkStatus status;

    if (!SparkRequestApiJitResidencyPolicyIsEnabled(api))
    {
        return SPARK_STATUS_OK;
    }

    if (additional_protected_physical_block_count != 0u &&
        additional_protected_physical_block_indices == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    protected_physical_block_count = 0u;
    SparkRequestApiCollectProtectedPrefetchPlanBlocks(
        protected_prefetch_plan,
        protected_physical_block_indices,
        &protected_physical_block_count);
    for (pending_index = 0u;
         pending_index < additional_protected_physical_block_count;
         ++pending_index)
    {
        SparkRequestApiCollectPhysicalBlockIndex(
            protected_physical_block_indices,
            &protected_physical_block_count,
            additional_protected_physical_block_indices[pending_index]);
    }
    for (pending_index = 0u;
         pending_index < SPARK_REQUEST_API_PENDING_PREFETCH_CAPACITY;
         ++pending_index)
    {
        if (api->pending_prefetches[pending_index].active == 0u)
        {
            continue;
        }
        SparkRequestApiCollectProtectedPrefetchPlanBlocks(
            &api->pending_prefetches[pending_index].prefetch_plan,
            protected_physical_block_indices,
            &protected_physical_block_count);
    }

    status = SparkRequestApiCollectRunningProtectedBlocks(
        api,
        protected_physical_block_indices,
        &protected_physical_block_count);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    evicted_block_count = 0u;
    status = SparkPrefixCacheTrimResidentBlocksByReuseScore(
        api->scheduler->prefix_cache,
        api->max_resident_kv_block_count,
        protected_physical_block_indices,
        protected_physical_block_count,
        &evicted_block_count);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    api->jit_residency_eviction_count += evicted_block_count;
    api->jit_residency_protected_block_count +=
        protected_physical_block_count +
        api->scheduler->prefix_cache->lookahead_protected_block_count;
    return SPARK_STATUS_OK;
}

SparkStatus SparkRequestApiBuildJitKvPrefetchPlan(
    SparkRequestApi *api,
    SparkKvCachePrefetchPlan *prefetch_plan)
{
    SparkRequestApiHandle selected_handles[
        SPARK_REQUEST_API_MAX_PREFETCH_SOURCE_BLOCK_COUNT];
    SparkKvCachePrefetchSourceBlock source_blocks[
        SPARK_REQUEST_API_MAX_PREFETCH_SOURCE_BLOCK_COUNT];
    uint32_t selected_handle_count;
    uint32_t source_block_count;
    uint32_t lookahead_index;
    SparkStatus status;

    status = SparkRequestApiValidate(api);
    if (status != SPARK_STATUS_OK || prefetch_plan == 0)
    {
        return status == SPARK_STATUS_OK ? SPARK_STATUS_INVALID_ARGUMENT : status;
    }
    if (!SparkRequestApiJitPrefetchIsEnabled(api))
    {
        return SparkKvCacheArenaBuildPrefetchPlan(
            api->scheduler->prefix_cache->kv_cache_arena,
            0,
            0u,
            api->prefetch_lane_count,
            prefetch_plan);
    }

    selected_handle_count = 0u;
    source_block_count = 0u;
    for (lookahead_index = 0u;
         lookahead_index < api->prefetch_lookahead_request_count &&
             selected_handle_count <
                SPARK_REQUEST_API_MAX_PREFETCH_SOURCE_BLOCK_COUNT;
         ++lookahead_index)
    {
        SparkRequestApiSlot *slot;

        slot = SparkRequestApiFindBestPrefetchLookaheadSlot(
            api,
            selected_handles,
            selected_handle_count);
        if (slot == 0)
        {
            break;
        }
        selected_handles[selected_handle_count] = slot->handle;
        selected_handle_count += 1u;
        if (SparkRequestApiSlotIsSchedulablePrefill(slot))
        {
            status = SparkRequestApiCollectPrefillSlotPrefetchSources(
                api,
                slot,
                source_blocks,
                &source_block_count);
        }
        else
        {
            status = SparkRequestApiCollectDecodeSlotPrefetchSources(
                api,
                slot,
                source_blocks,
                &source_block_count);
        }
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        if (source_block_count >=
            SPARK_REQUEST_API_MAX_PREFETCH_SOURCE_BLOCK_COUNT)
        {
            break;
        }
    }

    return SparkKvCacheArenaBuildPrefetchPlanFromSourceBlocks(
        api->scheduler->prefix_cache->kv_cache_arena,
        source_block_count != 0u ? source_blocks : 0,
        source_block_count,
        api->prefetch_lane_count,
        prefetch_plan);
}

static uint32_t SparkRequestApiPrefetchBlockIsResident(
    const SparkRequestApi *api,
    const SparkKvCachePrefetchBlock *prefetch_block)
{
    const SparkKvCacheArena *arena;
    const SparkKvCacheBlock *block;

    if (SparkRequestApiCrossSequencePrefixReuseIsEnabled(api) == 0u ||
        api->scheduler->prefix_cache == 0 ||
        api->scheduler->prefix_cache->kv_cache_arena == 0 ||
        prefetch_block == 0)
    {
        return 0u;
    }

    arena = api->scheduler->prefix_cache->kv_cache_arena;
    if (prefetch_block->physical_block_index >= arena->physical_block_count)
    {
        return 0u;
    }

    block = &arena->blocks[prefetch_block->physical_block_index];
    return (block->flags & SPARK_KV_CACHE_BLOCK_FLAG_ALLOCATED) != 0u &&
        (block->flags & SPARK_KV_CACHE_BLOCK_FLAG_RESIDENT) != 0u &&
        block->generation == prefetch_block->generation;
}

static uint32_t SparkRequestApiPrefetchPlanIsResident(
    const SparkRequestApi *api,
    const SparkKvCachePrefetchPlan *prefetch_plan)
{
    uint32_t block_index;

    if (prefetch_plan == 0)
    {
        return 0u;
    }
    for (block_index = 0u;
         block_index < prefetch_plan->prefetch_block_count;
         ++block_index)
    {
        if (!SparkRequestApiPrefetchBlockIsResident(
                api,
                &prefetch_plan->blocks[block_index]))
        {
            return 0u;
        }
    }
    return 1u;
}

static uint32_t SparkRequestApiPendingPrefetchContainsBlock(
    const SparkRequestApiPendingPrefetch *pending_prefetch,
    const SparkKvCachePrefetchBlock *prefetch_block)
{
    uint32_t block_index;

    if (pending_prefetch == 0 ||
        pending_prefetch->active == 0u ||
        prefetch_block == 0)
    {
        return 0u;
    }
    for (block_index = 0u;
         block_index < pending_prefetch->prefetch_plan.prefetch_block_count;
         ++block_index)
    {
        const SparkKvCachePrefetchBlock *pending_block;

        pending_block = &pending_prefetch->prefetch_plan.blocks[block_index];
        if (pending_block->physical_block_index ==
                prefetch_block->physical_block_index &&
            pending_block->generation == prefetch_block->generation)
        {
            return 1u;
        }
    }
    return 0u;
}

static uint32_t SparkRequestApiPendingPrefetchesCoverPlan(
    const SparkRequestApi *api,
    const SparkKvCachePrefetchPlan *prefetch_plan)
{
    uint32_t block_index;

    if (api == 0 || prefetch_plan == 0)
    {
        return 0u;
    }
    for (block_index = 0u;
         block_index < prefetch_plan->prefetch_block_count;
         ++block_index)
    {
        uint32_t pending_index;
        uint32_t found_pending_block;

        if (SparkRequestApiPrefetchBlockIsResident(
                api,
                &prefetch_plan->blocks[block_index]))
        {
            continue;
        }

        found_pending_block = 0u;
        for (pending_index = 0u;
             pending_index < SPARK_REQUEST_API_PENDING_PREFETCH_CAPACITY;
             ++pending_index)
        {
            if (SparkRequestApiPendingPrefetchContainsBlock(
                    &api->pending_prefetches[pending_index],
                    &prefetch_plan->blocks[block_index]))
            {
                found_pending_block = 1u;
                break;
            }
        }
        if (found_pending_block == 0u)
        {
            return 0u;
        }
    }
    return 1u;
}

static SparkRequestApiPendingPrefetch *
SparkRequestApiFindFreePendingPrefetch(
    SparkRequestApi *api)
{
    uint32_t pending_index;

    for (pending_index = 0u;
         pending_index < SPARK_REQUEST_API_PENDING_PREFETCH_CAPACITY;
         ++pending_index)
    {
        if (api->pending_prefetches[pending_index].active == 0u)
        {
            return &api->pending_prefetches[pending_index];
        }
    }
    return 0;
}

static void SparkRequestApiClearPendingPrefetch(
    SparkRequestApiPendingPrefetch *pending_prefetch)
{
    if (pending_prefetch == 0)
    {
        return;
    }
    memset(pending_prefetch, 0, sizeof(*pending_prefetch));
}

static SparkStatus SparkRequestApiPollOnePendingPrefetch(
    SparkRequestApi *api,
    SparkRequestApiPendingPrefetch *pending_prefetch)
{
    SparkStatus status;

    if (api == 0 || pending_prefetch == 0 || pending_prefetch->active == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    status = api->kv_prefetch_poll_function(
        api->kv_prefetch_context,
        pending_prefetch->prefetch_id,
        &pending_prefetch->prefetch_plan);
    pending_prefetch->poll_count += 1u;
    api->async_jit_prefetch_poll_count += 1u;
    if (status == SPARK_STATUS_BUSY)
    {
        return SPARK_STATUS_BUSY;
    }
    if (status != SPARK_STATUS_OK)
    {
        SparkRequestApiClearPendingPrefetch(pending_prefetch);
        return status;
    }

    status = SparkKvCacheArenaMarkPrefetchPlanResident(
        api->scheduler->prefix_cache->kv_cache_arena,
        &pending_prefetch->prefetch_plan);
    if (status != SPARK_STATUS_OK)
    {
        SparkRequestApiClearPendingPrefetch(pending_prefetch);
        return status;
    }
    status = SparkRequestApiApplyJitKvResidencyPolicy(
        api,
        &pending_prefetch->prefetch_plan,
        0,
        0u);
    if (status != SPARK_STATUS_OK)
    {
        SparkRequestApiClearPendingPrefetch(pending_prefetch);
        return status;
    }

    api->jit_prefetch_dispatch_count += 1u;
    api->jit_prefetch_block_count +=
        pending_prefetch->prefetch_plan.prefetch_block_count;
    api->async_jit_prefetch_completion_count += 1u;
    SparkRequestApiClearPendingPrefetch(pending_prefetch);
    return SPARK_STATUS_OK;
}

static SparkStatus SparkRequestApiPollPendingJitKvPrefetches(
    SparkRequestApi *api)
{
    uint32_t pending_index;

    for (pending_index = 0u;
         pending_index < SPARK_REQUEST_API_PENDING_PREFETCH_CAPACITY;
         ++pending_index)
    {
        SparkStatus status;

        if (api->pending_prefetches[pending_index].active == 0u)
        {
            continue;
        }
        status = SparkRequestApiPollOnePendingPrefetch(
            api,
            &api->pending_prefetches[pending_index]);
        if (status != SPARK_STATUS_OK && status != SPARK_STATUS_BUSY)
        {
            return status;
        }
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkRequestApiStartAsyncJitKvPrefetch(
    SparkRequestApi *api,
    const SparkKvCachePrefetchPlan *prefetch_plan)
{
    SparkRequestApiPendingPrefetch *pending_prefetch;
    SparkStatus status;
    uint64_t prefetch_id;

    pending_prefetch = SparkRequestApiFindFreePendingPrefetch(api);
    if (pending_prefetch == 0)
    {
        return SPARK_STATUS_BUSY;
    }

    prefetch_id = api->next_prefetch_id;
    api->next_prefetch_id += 1u;
    if (api->next_prefetch_id == 0u)
    {
        api->next_prefetch_id = 1u;
    }

    status = api->kv_prefetch_start_function(
        api->kv_prefetch_context,
        prefetch_id,
        prefetch_plan);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    memset(pending_prefetch, 0, sizeof(*pending_prefetch));
    pending_prefetch->active = 1u;
    pending_prefetch->prefetch_id = prefetch_id;
    pending_prefetch->prefetch_plan = *prefetch_plan;
    api->async_jit_prefetch_start_count += 1u;

    status = SparkRequestApiPollOnePendingPrefetch(
        api,
        pending_prefetch);
    if (status == SPARK_STATUS_OK)
    {
        return SPARK_STATUS_OK;
    }
    if (status == SPARK_STATUS_BUSY)
    {
        return SPARK_STATUS_BUSY;
    }
    return status;
}

static SparkStatus SparkRequestApiDispatchJitKvPrefetchWithProtectedBlocks(
    SparkRequestApi *api,
    SparkKvCachePrefetchPlan *prefetch_plan,
    const uint32_t *additional_protected_physical_block_indices,
    uint32_t additional_protected_physical_block_count)
{
    SparkStatus status;

    status = SparkRequestApiValidate(api);
    if (status != SPARK_STATUS_OK || prefetch_plan == 0)
    {
        return status == SPARK_STATUS_OK ? SPARK_STATUS_INVALID_ARGUMENT : status;
    }
    if (!SparkRequestApiJitPrefetchIsEnabled(api))
    {
        return SPARK_STATUS_OK;
    }
    if (SparkRequestApiAsyncJitPrefetchIsEnabled(api))
    {
        if (prefetch_plan->prefetch_block_count == 0u ||
            SparkRequestApiPrefetchPlanIsResident(api, prefetch_plan))
        {
            return SparkRequestApiApplyJitKvResidencyPolicy(
                api,
                prefetch_plan,
                additional_protected_physical_block_indices,
                additional_protected_physical_block_count);
        }
        status = SparkRequestApiPollPendingJitKvPrefetches(api);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        if (SparkRequestApiPrefetchPlanIsResident(api, prefetch_plan))
        {
            return SparkRequestApiApplyJitKvResidencyPolicy(
                api,
                prefetch_plan,
                additional_protected_physical_block_indices,
                additional_protected_physical_block_count);
        }
        if (SparkRequestApiPendingPrefetchesCoverPlan(
                api,
                prefetch_plan))
        {
            return SPARK_STATUS_BUSY;
        }
        return SparkRequestApiStartAsyncJitKvPrefetch(
            api,
            prefetch_plan);
    }

    if (prefetch_plan->prefetch_block_count == 0u)
    {
        return SparkRequestApiApplyJitKvResidencyPolicy(
            api,
            prefetch_plan,
            additional_protected_physical_block_indices,
            additional_protected_physical_block_count);
    }
    if (api->kv_prefetch_function == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    status = api->kv_prefetch_function(
        api->kv_prefetch_context,
        prefetch_plan);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkKvCacheArenaMarkPrefetchPlanResidentWithProtectedBlocks(
        api->scheduler->prefix_cache->kv_cache_arena,
        prefetch_plan,
        additional_protected_physical_block_indices,
        additional_protected_physical_block_count);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkRequestApiApplyJitKvResidencyPolicy(
        api,
        prefetch_plan,
        additional_protected_physical_block_indices,
        additional_protected_physical_block_count);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    api->jit_prefetch_dispatch_count += 1u;
    api->jit_prefetch_block_count += prefetch_plan->prefetch_block_count;
    return SPARK_STATUS_OK;
}

SparkStatus SparkRequestApiDispatchJitKvPrefetch(
    SparkRequestApi *api,
    SparkKvCachePrefetchPlan *prefetch_plan)
{
    return SparkRequestApiDispatchJitKvPrefetchWithProtectedBlocks(
        api,
        prefetch_plan,
        0,
        0u);
}




static SparkStatus SparkRequestApiBuildSlotArrayJitKvPrefetchPlan(
    SparkRequestApi *api,
    SparkRequestApiSlot **slots,
    uint32_t slot_count,
    SparkKvCachePrefetchPlan *prefetch_plan)
{
    SparkKvCachePrefetchSourceBlock source_blocks[
        SPARK_REQUEST_API_MAX_PREFETCH_SOURCE_BLOCK_COUNT];
    uint32_t source_block_count;
    uint32_t slot_index;
    SparkStatus status;

    if (prefetch_plan == 0 || slots == 0 || slot_count == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (!SparkRequestApiJitPrefetchIsEnabled(api))
    {
        memset(prefetch_plan, 0, sizeof(*prefetch_plan));
        return SPARK_STATUS_OK;
    }

    source_block_count = 0u;
    for (slot_index = 0u;
         slot_index < slot_count &&
             source_block_count <
                SPARK_REQUEST_API_MAX_PREFETCH_SOURCE_BLOCK_COUNT;
         ++slot_index)
    {
        SparkRequestApiSlot *slot;

        slot = slots[slot_index];
        if (slot == 0)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        if (SparkRequestApiSlotIsSchedulablePrefill(slot))
        {
            status = SparkRequestApiCollectPrefillSlotPrefetchSources(
                api,
                slot,
                source_blocks,
                &source_block_count);
        }
        else if (SparkRequestApiSlotIsSchedulableDecode(slot) ||
                 SparkRequestApiSlotIsSchedulableSpeculativeVerify(slot))
        {
            status = SparkRequestApiCollectDecodeSlotPrefetchSources(
                api,
                slot,
                source_blocks,
                &source_block_count);
        }
        else
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }

    return SparkKvCacheArenaBuildPrefetchPlanFromSourceBlocks(
        api->scheduler->prefix_cache->kv_cache_arena,
        source_block_count != 0u ? source_blocks : 0,
        source_block_count,
        api->prefetch_lane_count,
        prefetch_plan);
}

static SparkStatus SparkRequestApiRunSlotArrayCriticalJitKvPrefetch(
    SparkRequestApi *api,
    SparkRequestApiSlot **slots,
    uint32_t slot_count,
    SparkRequestApiDispatch *dispatch)
{
    uint32_t critical_physical_block_indices[
        SPARK_REQUEST_API_MAX_PREFETCH_SOURCE_BLOCK_COUNT];
    uint32_t critical_physical_block_count;
    uint32_t slot_index;
    SparkStatus status;

    if (!SparkRequestApiJitPrefetchIsEnabled(api))
    {
        return SPARK_STATUS_OK;
    }

    critical_physical_block_count = 0u;
    for (slot_index = 0u; slot_index < slot_count; ++slot_index)
    {
        status = SparkRequestApiCollectProtectedSlotBlocks(
            api,
            slots[slot_index],
            critical_physical_block_indices,
            &critical_physical_block_count);
        if (status != SPARK_STATUS_OK && status != SPARK_STATUS_NOT_FOUND)
        {
            return status;
        }
    }

    status = SparkRequestApiBuildSlotArrayJitKvPrefetchPlan(
        api,
        slots,
        slot_count,
        &dispatch->kv_prefetch_plan);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkRequestApiDispatchJitKvPrefetchWithProtectedBlocks(
        api,
        &dispatch->kv_prefetch_plan,
        critical_physical_block_indices,
        critical_physical_block_count);
    if (status == SPARK_STATUS_BUSY)
    {
        dispatch->flags |=
            SPARK_REQUEST_API_DISPATCH_FLAG_JIT_PREFETCH_PENDING;
        return SPARK_STATUS_BUSY;
    }
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (dispatch->kv_prefetch_plan.prefetch_block_count != 0u)
    {
        dispatch->flags |=
            SPARK_REQUEST_API_DISPATCH_FLAG_JIT_PREFETCHED_KV;
    }
    return SPARK_STATUS_OK;
}


static uint32_t SparkRequestApiPrefetchPlanFitsResidentLimit(
    const SparkRequestApi *api,
    const SparkKvCachePrefetchPlan *prefetch_plan,
    const uint32_t *protected_physical_block_indices,
    uint32_t protected_physical_block_count)
{
    uint32_t combined_physical_block_indices[
        SPARK_REQUEST_API_MAX_PREFETCH_SOURCE_BLOCK_COUNT];
    uint32_t combined_physical_block_count;
    uint32_t block_index;

    if (api == 0 || api->max_resident_kv_block_count == 0u)
    {
        return 1u;
    }
    combined_physical_block_count = 0u;
    for (block_index = 0u;
         block_index < protected_physical_block_count;
         ++block_index)
    {
        SparkRequestApiCollectPhysicalBlockIndex(
            combined_physical_block_indices,
            &combined_physical_block_count,
            protected_physical_block_indices[block_index]);
    }
    if (prefetch_plan != 0)
    {
        for (block_index = 0u;
             block_index < prefetch_plan->prefetch_block_count;
             ++block_index)
        {
            SparkRequestApiCollectPhysicalBlockIndex(
                combined_physical_block_indices,
                &combined_physical_block_count,
                prefetch_plan->blocks[block_index].physical_block_index);
        }
    }
    return combined_physical_block_count <= api->max_resident_kv_block_count;
}

static SparkStatus SparkRequestApiRunOpportunisticJitKvPrefetch(
    SparkRequestApi *api,
    SparkRequestApiSlot *protected_slot)
{
    uint32_t protected_physical_block_indices[
        SPARK_REQUEST_API_MAX_PREFETCH_SOURCE_BLOCK_COUNT];
    uint32_t protected_physical_block_count;
    SparkKvCachePrefetchPlan prefetch_plan;
    SparkStatus status;

    if (!SparkRequestApiJitPrefetchIsEnabled(api))
    {
        return SPARK_STATUS_OK;
    }

    protected_physical_block_count = 0u;
    if (protected_slot != 0)
    {
        status = SparkRequestApiCollectProtectedSlotBlocks(
            api,
            protected_slot,
            protected_physical_block_indices,
            &protected_physical_block_count);
        if (status != SPARK_STATUS_OK && status != SPARK_STATUS_NOT_FOUND)
        {
            return status;
        }
    }

    status = SparkRequestApiBuildJitKvPrefetchPlan(
        api,
        &prefetch_plan);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (!SparkRequestApiPrefetchPlanFitsResidentLimit(
            api,
            &prefetch_plan,
            protected_physical_block_indices,
            protected_physical_block_count))
    {
        return SparkRequestApiApplyJitKvResidencyPolicy(
            api,
            0,
            protected_physical_block_indices,
            protected_physical_block_count);
    }
    status = SparkRequestApiDispatchJitKvPrefetchWithProtectedBlocks(
        api,
        &prefetch_plan,
        protected_physical_block_indices,
        protected_physical_block_count);
    if (status == SPARK_STATUS_BUSY || status == SPARK_STATUS_CAPACITY_EXCEEDED)
    {
        return SparkRequestApiApplyJitKvResidencyPolicy(
            api,
            0,
            protected_physical_block_indices,
            protected_physical_block_count);
    }
    return status;
}

static void SparkRequestApiInitializeDispatch(
    SparkRequestApiDispatch *dispatch)
{
    memset(dispatch, 0, sizeof(*dispatch));
    dispatch->abi_version = SPARK_REQUEST_API_ABI_VERSION;
    dispatch->descriptor_bytes = SPARK_REQUEST_API_DISPATCH_DESCRIPTOR_BYTES;
}

static uint32_t SparkRequestApiProbeReusablePrefixTokenCount(
    SparkRequestApi *api,
    const SparkRequestApiSlot *slot)
{
    SparkPrefixCacheLookup lookup;
    SparkStatus status;

    if (api == 0 ||
        api->scheduler == 0 ||
        api->scheduler->prefix_cache == 0 ||
        slot == 0 ||
        slot->prompt_token_ids == 0 ||
        slot->prompt_token_count == 0u)
    {
        return 0u;
    }

    status = SparkPrefixCacheProbePrompt(
        api->scheduler->prefix_cache,
        slot->sequence_id,
        slot->prompt_token_ids,
        slot->prompt_token_count,
        &lookup);
    if (status != SPARK_STATUS_OK)
    {
        return slot->computed_prompt_token_count;
    }
    return SparkRequestApiMaximumU32(
        slot->computed_prompt_token_count,
        lookup.matched_token_count);
}


static uint32_t SparkRequestApiSchedulerMaxPrefillTokensPerStep(
    const SparkRequestApi *api,
    const SparkRequestApiSlot *slot)
{
    uint32_t max_prefill_tokens_per_step;

    max_prefill_tokens_per_step = slot->max_prefill_tokens_per_step;
    if (max_prefill_tokens_per_step == 0u ||
        max_prefill_tokens_per_step > api->scheduler->max_prefill_tokens_per_step)
    {
        max_prefill_tokens_per_step = api->scheduler->max_prefill_tokens_per_step;
    }
    if (max_prefill_tokens_per_step < api->scheduler->prefix_cache_block_tokens)
    {
        max_prefill_tokens_per_step = api->scheduler->prefix_cache_block_tokens;
    }
    return max_prefill_tokens_per_step;
}

static uint32_t SparkRequestApiConfigurationHasChunkedPrefill(
    const SparkRequestApi *api)
{
    return (api->scheduler->configuration_flags &
        SPARK_SCHEDULER_CONFIGURATION_FLAG_CHUNKED_PREFILL) != 0u;
}

static uint32_t SparkRequestApiRoundDownSchedulerBlock(
    const SparkRequestApi *api,
    uint32_t token_count)
{
    uint32_t block_token_count;

    block_token_count = api->scheduler->prefix_cache_block_tokens;
    if (block_token_count == 0u)
    {
        return token_count;
    }
    return token_count - (token_count % block_token_count);
}

static uint32_t SparkRequestApiNextPrefillStepTokenCount(
    SparkRequestApi *api,
    const SparkRequestApiSlot *slot,
    uint32_t *computed_prompt_token_count_out)
{
    uint32_t cached_prefix_token_count;
    uint32_t computed_prompt_token_count;
    uint32_t remaining_prompt_token_count;
    uint32_t max_prefill_tokens_per_step;
    uint32_t scheduled_prompt_token_count;

    cached_prefix_token_count = SparkRequestApiProbeReusablePrefixTokenCount(
        api,
        slot);
    computed_prompt_token_count = SparkRequestApiMaximumU32(
        slot->computed_prompt_token_count,
        cached_prefix_token_count);
    if (computed_prompt_token_count_out != 0)
    {
        *computed_prompt_token_count_out = computed_prompt_token_count;
    }
    if (computed_prompt_token_count >= slot->prompt_token_count)
    {
        return 0u;
    }
    remaining_prompt_token_count =
        slot->prompt_token_count - computed_prompt_token_count;
    if (!SparkRequestApiConfigurationHasChunkedPrefill(api))
    {
        return remaining_prompt_token_count;
    }
    max_prefill_tokens_per_step = SparkRequestApiSchedulerMaxPrefillTokensPerStep(
        api,
        slot);
    if (remaining_prompt_token_count <= max_prefill_tokens_per_step)
    {
        return remaining_prompt_token_count;
    }
    scheduled_prompt_token_count = SparkRequestApiRoundDownSchedulerBlock(
        api,
        max_prefill_tokens_per_step);
    if (scheduled_prompt_token_count == 0u)
    {
        scheduled_prompt_token_count = SparkRequestApiMinimumU32(
            remaining_prompt_token_count,
            api->scheduler->prefix_cache_block_tokens);
    }
    return scheduled_prompt_token_count;
}

static void SparkRequestApiFillPrefillSchedulerRequest(
    const SparkRequestApi *api,
    const SparkRequestApiSlot *slot,
    uint32_t prompt_token_count,
    uint32_t max_scheduled_prompt_token_count,
    SparkSchedulerRequest *scheduler_request)
{
    memset(scheduler_request, 0, sizeof(*scheduler_request));
    scheduler_request->abi_version = SPARK_SCHEDULER_ABI_VERSION;
    scheduler_request->descriptor_bytes =
        SPARK_SCHEDULER_REQUEST_DESCRIPTOR_BYTES;
    scheduler_request->active_sequence_count = 1u;
    scheduler_request->prompt_token_count = prompt_token_count;
    scheduler_request->computed_prompt_token_count =
        SparkRequestApiCrossSequencePrefixReuseIsEnabled(api) != 0u
            ? 0u : SparkRequestApiMaximumU32(
                slot->computed_prompt_token_count,
                slot->dispatched_prompt_token_count);
    scheduler_request->flags = SPARK_SCHEDULER_REQUEST_FLAG_PREFILL;
    scheduler_request->max_scheduled_prompt_token_count =
        max_scheduled_prompt_token_count != 0u
            ? max_scheduled_prompt_token_count
            : slot->max_prefill_tokens_per_step;
    scheduler_request->sequence_id = slot->sequence_id;
    scheduler_request->prompt_token_ids = slot->prompt_token_ids;
}

static uint32_t SparkRequestApiFindBestSharedPrefixTokenCount(
    SparkRequestApi *api,
    const SparkRequestApiSlot *leader_slot)
{
    uint32_t slot_index;
    uint32_t best_shared_prefix_token_count;

    if (!SparkRequestApiPrefixCohortingIsEnabled(api))
    {
        return 0u;
    }

    best_shared_prefix_token_count = 0u;
    for (slot_index = 0u; slot_index < api->request_capacity; ++slot_index)
    {
        SparkRequestApiSlot *candidate;
        uint32_t candidate_shared_prefix_token_count;

        candidate = &api->request_slots[slot_index];
        if (candidate == leader_slot ||
            candidate->state != SPARK_REQUEST_API_STATE_QUEUED_PREFILL ||
            !SparkRequestApiSlotsHaveSameSchedulingPriority(
                candidate,
                leader_slot))
        {
            continue;
        }
        candidate_shared_prefix_token_count =
            SparkRequestApiSharedCachePrefixTokenCount(
                api,
                leader_slot,
                candidate);
        if (candidate_shared_prefix_token_count >
            best_shared_prefix_token_count)
        {
            best_shared_prefix_token_count = candidate_shared_prefix_token_count;
        }
    }
    return best_shared_prefix_token_count;
}

static uint32_t SparkRequestApiPrefillBlockCountForScheduledTokens(
    const SparkRequestApi *api,
    uint32_t scheduled_prompt_token_count)
{
    uint32_t block_token_count;

    if (api == 0 ||
        api->scheduler == 0 ||
        scheduled_prompt_token_count == 0u)
    {
        return 0u;
    }

    block_token_count = api->scheduler->prefix_cache_block_tokens;
    if (block_token_count == 0u)
    {
        return 1u;
    }
    return (scheduled_prompt_token_count + block_token_count - 1u) /
        block_token_count;
}

static uint32_t SparkRequestApiSlotResidentKvBlockCount(
    const SparkRequestApi *api,
    const SparkRequestApiSlot *slot)
{
    uint64_t token_count;
    uint32_t block_token_count;

    if (api == 0 || api->scheduler == 0 || slot == 0 ||
        slot->state == SPARK_REQUEST_API_STATE_FREE ||
        slot->state == SPARK_REQUEST_API_STATE_COMPLETED ||
        slot->state == SPARK_REQUEST_API_STATE_CANCELLED ||
        (slot->computed_prompt_token_count == 0u &&
         (slot->state == SPARK_REQUEST_API_STATE_QUEUED_PREFILL ||
          slot->state == SPARK_REQUEST_API_STATE_WAITING_PREFIX_COHORT)))
    {
        return 0u;
    }
    block_token_count = api->scheduler->prefix_cache_block_tokens;
    if (block_token_count == 0u)
    {
        return 0u;
    }
    token_count = slot->prompt_token_count;
    token_count += slot->scheduled_decode_token_count;
    if (slot->state ==
        SPARK_REQUEST_API_STATE_RUNNING_SPECULATIVE_VERIFY)
    {
        token_count += 1u;
    }
    return token_count > UINT32_MAX
        ? UINT32_MAX
        : (uint32_t)((token_count + block_token_count - 1u) /
            block_token_count);
}

static uint64_t SparkRequestApiResidentKvBlockCount(
    const SparkRequestApi *api)
{
    uint64_t block_count;
    uint32_t slot_index;

    block_count = 0u;
    for (slot_index = 0u; slot_index < api->request_capacity; ++slot_index)
    {
        block_count += SparkRequestApiSlotResidentKvBlockCount(
            api,
            &api->request_slots[slot_index]);
    }
    return block_count;
}

static uint32_t SparkRequestApiReservePrefillResidentKvBlocks(
    const SparkRequestApi *api,
    const SparkRequestApiSlot *slot,
    uint64_t *reserved_block_count)
{
    uint64_t additional_block_count;
    uint32_t current_block_count;
    uint32_t required_block_count;

    if (api->max_resident_kv_block_count == 0u ||
        SparkRequestApiJitPrefetchIsEnabled(api))
    {
        return 1u;
    }
    current_block_count = SparkRequestApiSlotResidentKvBlockCount(
        api,
        slot);
    required_block_count = SparkRequestApiPrefillBlockCountForScheduledTokens(
        api,
        slot->prompt_token_count);
    additional_block_count = required_block_count > current_block_count
        ? required_block_count - current_block_count
        : 0u;
    if (*reserved_block_count > api->max_resident_kv_block_count ||
        additional_block_count >
            api->max_resident_kv_block_count - *reserved_block_count)
    {
        return 0u;
    }
    *reserved_block_count += additional_block_count;
    return 1u;
}

static SparkStatus SparkGlm52RequestApiSchedulePrefill(
    SparkRequestApi *api,
    SparkRequestApiSlot *slot,
    uint32_t selected_shared_prefix_token_count,
    SparkRequestApiDispatch *dispatch)
{
    SparkSchedulerRequest scheduler_request;
    uint32_t slot_index;
    uint32_t scheduler_prompt_token_count;
    uint32_t scheduler_step_token_limit;
    uint32_t shared_prefix_token_count;
    uint32_t reusable_prefix_token_count;
    uint32_t committed_prefix_token_count;
    uint64_t reserved_block_count;
    SparkStatus status;

    reserved_block_count = SparkRequestApiResidentKvBlockCount(api);
    if (!SparkRequestApiReservePrefillResidentKvBlocks(
            api,
            slot,
            &reserved_block_count))
    {
        return SPARK_STATUS_BUSY;
    }

    shared_prefix_token_count = selected_shared_prefix_token_count;
    if (shared_prefix_token_count == 0u)
    {
        shared_prefix_token_count = SparkRequestApiFindBestSharedPrefixTokenCount(
            api,
            slot);
    }
    reusable_prefix_token_count = SparkRequestApiProbeReusablePrefixTokenCount(
        api,
        slot);
    scheduler_prompt_token_count = slot->prompt_token_count;
    scheduler_step_token_limit = 0u;
    if (shared_prefix_token_count > reusable_prefix_token_count)
    {
        scheduler_step_token_limit =
            shared_prefix_token_count - reusable_prefix_token_count;
    }

    if (!SparkRequestApiPrefillCachedBlocksAreResident(
            api,
            slot,
            scheduler_prompt_token_count))
    {
        return SPARK_STATUS_BUSY;
    }

    SparkRequestApiFillPrefillSchedulerRequest(
        api,
        slot,
        scheduler_prompt_token_count,
        scheduler_step_token_limit,
        &scheduler_request);
    status = SparkSchedulerAdmit(
        api->scheduler,
        &scheduler_request,
        &dispatch->prefill_decision);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (dispatch->prefill_decision.accepted == 0u)
    {
        return SPARK_STATUS_OK;
    }

    committed_prefix_token_count =
        dispatch->prefill_decision.cache_commit_token_count_after_step;
    if (committed_prefix_token_count > slot->prompt_token_count)
    {
        committed_prefix_token_count = slot->prompt_token_count;
    }

    if (slot->state == SPARK_REQUEST_API_STATE_QUEUED_PREFILL)
    {
        slot->state = SPARK_REQUEST_API_STATE_RUNNING_PREFILL;
        api->queued_request_count -= 1u;
    }
    if (slot->inflight_prefill_dispatch_count == 0u &&
        slot->dispatched_prompt_token_count <
            slot->computed_prompt_token_count)
        slot->dispatched_prompt_token_count =
            slot->computed_prompt_token_count;
    if (committed_prefix_token_count > slot->dispatched_prompt_token_count)
        slot->dispatched_prompt_token_count = committed_prefix_token_count;
    slot->inflight_prefill_dispatch_count += 1u;
    slot->scheduled_prefill_step_count += 1u;
    api->running_request_count += 1u;
    api->scheduled_prefill_dispatch_count += 1u;

    dispatch->accepted = 1u;
    dispatch->kind = SPARK_REQUEST_API_DISPATCH_KIND_PREFILL;
    dispatch->request_count = 1u;
    dispatch->highest_priority = slot->priority;
    dispatch->shared_prefix_token_count =
        shared_prefix_token_count > committed_prefix_token_count
            ? committed_prefix_token_count
            : shared_prefix_token_count;
    if (api->scheduler != 0 && api->scheduler->prefix_cache_block_tokens != 0u)
    {
        dispatch->shared_prefix_block_count = dispatch->shared_prefix_token_count /
            api->scheduler->prefix_cache_block_tokens;
    }
    dispatch->prefix_cache_parent_hash =
        dispatch->prefill_decision.prefix_cache_parent_hash;
    dispatch->prefix_cache_result_hash =
        dispatch->prefill_decision.prefix_cache_result_hash;
    dispatch->request_handles[0] = slot->handle;
    dispatch->request_slot_indices[0] = SparkRequestApiSlotIndex(api, slot);
    dispatch->request_ids[0] = slot->request_id;
    dispatch->sequence_ids[0] = slot->sequence_id;
    if (SparkRequestModelSlotCanSpeculate(api, slot))
    {
        dispatch->flags |=
            SPARK_REQUEST_API_DISPATCH_FLAG_DSPARK_TAP_CAPTURE;
        api->dspark_tap_capture_dispatch_count += 1u;
    }

    if (dispatch->shared_prefix_token_count == 0u)
    {
        return SPARK_STATUS_OK;
    }

    for (slot_index = 0u;
         slot_index < api->request_capacity &&
             dispatch->request_count <
                SPARK_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT;
         ++slot_index)
    {
        SparkRequestApiSlot *candidate;

		candidate = &api->request_slots[slot_index];
		if (candidate == slot ||
			candidate->state != SPARK_REQUEST_API_STATE_QUEUED_PREFILL ||
			SparkRequestModelSlotCanSpeculate(api,candidate) !=
				SparkRequestModelSlotCanSpeculate(api,slot) ||
			SparkRequestApiSharedCachePrefixTokenCount(
				api,
                slot,
                candidate) < dispatch->shared_prefix_token_count)
        {
            continue;
        }
        candidate->state = SPARK_REQUEST_API_STATE_WAITING_PREFIX_COHORT;
        candidate->scheduled_prefill_step_count += 1u;
        api->queued_request_count -= 1u;
        api->running_request_count += 1u;
        dispatch->request_handles[dispatch->request_count] = candidate->handle;
        dispatch->request_slot_indices[dispatch->request_count] =
            SparkRequestApiSlotIndex(api, candidate);
        dispatch->request_ids[dispatch->request_count] = candidate->request_id;
        dispatch->sequence_ids[dispatch->request_count] = candidate->sequence_id;
        dispatch->request_count += 1u;
    }
    if (dispatch->request_count > 1u)
    {
        dispatch->flags |= SPARK_REQUEST_API_DISPATCH_FLAG_PREFIX_COHORT;
        if (selected_shared_prefix_token_count != 0u)
        {
            dispatch->flags |=
                SPARK_REQUEST_API_DISPATCH_FLAG_PREFIX_FAMILY_SELECTED;
        }
        api->prefix_family_dispatch_count += 1u;
        api->prefix_family_member_count += dispatch->request_count;
        api->prefix_family_saved_prompt_token_count +=
            SparkRequestApiPrefixFamilySavedTokenCount(
                dispatch->shared_prefix_token_count,
                dispatch->request_count);
    }
    return SPARK_STATUS_OK;
}


static uint32_t SparkRequestApiSlotIsCompatiblePrefillBatchMember(
    SparkRequestApi *api,
    const SparkRequestApiSlot *leader_slot,
    const SparkRequestApiSlot *candidate_slot,
    uint32_t leader_prefill_block_count,
    uint32_t require_resident_cached_blocks,
    uint32_t *candidate_scheduled_prompt_token_count_out)
{
    uint32_t candidate_scheduled_prompt_token_count;
    uint32_t candidate_prefill_block_count;

    if (candidate_scheduled_prompt_token_count_out != 0)
    {
        *candidate_scheduled_prompt_token_count_out = 0u;
    }
	if (candidate_slot == leader_slot ||
		!SparkRequestApiSlotIsSchedulablePrefill(candidate_slot) ||
		SparkRequestModelSlotCanSpeculate(api,candidate_slot) !=
			SparkRequestModelSlotCanSpeculate(api,leader_slot))
    {
        return 0u;
    }
    if ((require_resident_cached_blocks != 0u ||
         candidate_slot->priority < leader_slot->priority) &&
        !SparkRequestApiPrefillCachedBlocksAreResident(
            api,
            candidate_slot,
            candidate_slot->prompt_token_count))
    {
        return 0u;
    }

    candidate_scheduled_prompt_token_count =
        SparkRequestApiNextPrefillStepTokenCount(
            api,
            candidate_slot,
            0);
    candidate_prefill_block_count =
        SparkRequestApiPrefillBlockCountForScheduledTokens(
            api,
            candidate_scheduled_prompt_token_count);
    if (candidate_prefill_block_count == 0u ||
        candidate_prefill_block_count != leader_prefill_block_count)
    {
        return 0u;
    }
    if (candidate_scheduled_prompt_token_count_out != 0)
    {
        *candidate_scheduled_prompt_token_count_out =
            candidate_scheduled_prompt_token_count;
    }
    return 1u;
}

static uint32_t SparkRequestApiPrefillBatchCandidateIsBetter(
    const SparkRequestApiSlot *candidate_slot,
    uint32_t candidate_scheduled_prompt_token_count,
    const SparkRequestApiSlot *best_slot,
    uint32_t best_scheduled_prompt_token_count)
{
    if (best_slot == 0)
    {
        return 1u;
    }
    if (candidate_slot->priority != best_slot->priority)
    {
        return candidate_slot->priority > best_slot->priority;
    }
    if (candidate_scheduled_prompt_token_count !=
        best_scheduled_prompt_token_count)
    {
        return candidate_scheduled_prompt_token_count >
            best_scheduled_prompt_token_count;
    }
    return candidate_slot->submission_order < best_slot->submission_order;
}

static SparkRequestApiSlot *SparkRequestApiFindBestPrefillBatchMember(
    SparkRequestApi *api,
    const SparkRequestApiSlot *leader_slot,
    uint32_t leader_prefill_block_count,
    SparkRequestApiHandle *selected_handles,
    uint32_t selected_handle_count,
    uint32_t require_resident_cached_blocks,
    uint32_t *selected_scheduled_prompt_token_count_out)
{
    SparkRequestApiSlot *best_slot;
    uint32_t best_scheduled_prompt_token_count;
    uint32_t slot_index;

    if (selected_scheduled_prompt_token_count_out != 0)
    {
        *selected_scheduled_prompt_token_count_out = 0u;
    }
    best_slot = 0;
    best_scheduled_prompt_token_count = 0u;
    for (slot_index = 0u; slot_index < api->request_capacity; ++slot_index)
    {
        SparkRequestApiSlot *slot;
        uint32_t selected_index;
        uint32_t is_selected;
        uint32_t candidate_scheduled_prompt_token_count;

        slot = &api->request_slots[slot_index];
        if (!SparkRequestApiSlotIsCompatiblePrefillBatchMember(
                api,
                leader_slot,
                slot,
                leader_prefill_block_count,
                require_resident_cached_blocks,
                &candidate_scheduled_prompt_token_count))
        {
            continue;
        }
        is_selected = 0u;
        for (selected_index = 0u;
             selected_index < selected_handle_count;
             ++selected_index)
        {
            if (selected_handles[selected_index] == slot->handle)
            {
                is_selected = 1u;
                break;
            }
        }
        if (is_selected != 0u)
        {
            continue;
        }
        if (SparkRequestApiPrefillBatchCandidateIsBetter(
                slot,
                candidate_scheduled_prompt_token_count,
                best_slot,
                best_scheduled_prompt_token_count))
        {
            best_slot = slot;
            best_scheduled_prompt_token_count =
                candidate_scheduled_prompt_token_count;
        }
    }
    if (selected_scheduled_prompt_token_count_out != 0)
    {
        *selected_scheduled_prompt_token_count_out =
            best_scheduled_prompt_token_count;
    }
    return best_slot;
}

static SparkStatus SparkGlm52RequestApiSchedulePrefillBatch(
    SparkRequestApi *api,
    SparkRequestApiSlot *first_slot,
    SparkRequestApiDispatch *dispatch)
{
    SparkSchedulerRequest scheduler_requests[
        SPARK_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT];
    SparkSchedulerPrefillBatchRequest batch_request;
    SparkRequestApiHandle selected_handles[
        SPARK_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT];
    SparkRequestApiSlot *selected_slots[
        SPARK_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT];
    SparkRequestApiSlot *slot;
    uint32_t request_count;
    uint32_t request_index;
    uint32_t batch_target;
    uint32_t leader_scheduled_prompt_token_count;
    uint32_t leader_prefill_block_count;
    uint32_t require_resident_batch_members;
    uint64_t reserved_block_count;
    uint32_t selected_scheduled_prompt_token_counts[
        SPARK_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT];
    SparkStatus status;

    if (!SparkRequestApiPrefillBatchingIsEnabled(api))
    {
        return SPARK_STATUS_NOT_FOUND;
    }
    if (first_slot->state != SPARK_REQUEST_API_STATE_QUEUED_PREFILL)
    {
        return SPARK_STATUS_NOT_FOUND;
    }
    if (!SparkRequestApiPrefillCachedBlocksAreResident(
            api,
            first_slot,
            first_slot->prompt_token_count))
    {
        return SPARK_STATUS_BUSY;
    }
    leader_scheduled_prompt_token_count =
        SparkRequestApiNextPrefillStepTokenCount(
            api,
            first_slot,
            0);
    leader_prefill_block_count =
        SparkRequestApiPrefillBlockCountForScheduledTokens(
            api,
            leader_scheduled_prompt_token_count);
    if (leader_prefill_block_count == 0u)
    {
        return SPARK_STATUS_NOT_FOUND;
    }

    batch_target = SparkRequestApiCurrentPipelineBatchWidth(api);
    if (batch_target > SPARK_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT)
    {
        batch_target = SPARK_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT;
    }
    require_resident_batch_members =
        SparkRequestApiSlotHasRealtimePriority(first_slot) ||
        SparkRequestApiAsyncJitPrefetchIsEnabled(api);
    reserved_block_count = SparkRequestApiResidentKvBlockCount(api);
    request_count = 0u;
    slot = first_slot;
    while (slot != 0 && request_count < batch_target)
    {
        if (!SparkRequestApiReservePrefillResidentKvBlocks(
                api,
                slot,
                &reserved_block_count))
        {
            if (request_count == 0u)
            {
                return SPARK_STATUS_BUSY;
            }
            break;
        }
        if (request_count == 0u)
        {
            selected_scheduled_prompt_token_counts[request_count] =
                leader_scheduled_prompt_token_count;
        }
        selected_slots[request_count] = slot;
        selected_handles[request_count] = slot->handle;
        SparkRequestApiFillPrefillSchedulerRequest(
            api,
            slot,
            slot->prompt_token_count,
            selected_scheduled_prompt_token_counts[request_count],
            &scheduler_requests[request_count]);
        request_count += 1u;
        slot = SparkRequestApiFindBestPrefillBatchMember(
            api,
            first_slot,
            leader_prefill_block_count,
            selected_handles,
            request_count,
            require_resident_batch_members,
            request_count < batch_target
                ? &selected_scheduled_prompt_token_counts[request_count]
                : 0);
    }
    if (request_count < 2u)
    {
        return SPARK_STATUS_NOT_FOUND;
    }

    status = SparkRequestApiRunSlotArrayCriticalJitKvPrefetch(
        api,
        selected_slots,
        request_count,
        dispatch);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    for (request_index = 0u; request_index < request_count; ++request_index)
    {
        if (!SparkRequestApiPrefillCachedBlocksAreResident(
                api,
                selected_slots[request_index],
                selected_slots[request_index]->prompt_token_count))
        {
            dispatch->flags |=
                SPARK_REQUEST_API_DISPATCH_FLAG_JIT_PREFETCH_PENDING;
            return SPARK_STATUS_BUSY;
        }
    }

    memset(&batch_request, 0, sizeof(batch_request));
    batch_request.abi_version = SPARK_SCHEDULER_ABI_VERSION;
    batch_request.descriptor_bytes =
        SPARK_SCHEDULER_PREFILL_BATCH_REQUEST_DESCRIPTOR_BYTES;
    batch_request.request_count = request_count;
    batch_request.requests = scheduler_requests;
    status = SparkSchedulerAdmitPrefillBatch(
        api->scheduler,
        &batch_request,
        &dispatch->prefill_batch_decision);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (dispatch->prefill_batch_decision.accepted == 0u)
    {
        return SPARK_STATUS_OK;
    }

    dispatch->accepted = 1u;
    dispatch->kind = SPARK_REQUEST_API_DISPATCH_KIND_PREFILL_BATCH;
    dispatch->flags |= SPARK_REQUEST_API_DISPATCH_FLAG_PREFILL_BATCH;
    dispatch->request_count =
        dispatch->prefill_batch_decision.packed_request_count;
    dispatch->highest_priority = first_slot->priority;
	for (request_index = 0u;
		 request_index < dispatch->request_count;
		 ++request_index)
    {
        SparkRequestApiSlot *selected_slot;

        selected_slot = selected_slots[request_index];
        selected_slot->state = SPARK_REQUEST_API_STATE_RUNNING_PREFILL;
        selected_slot->scheduled_prefill_step_count += 1u;
        if (selected_slot->inflight_prefill_dispatch_count == 0u &&
            selected_slot->dispatched_prompt_token_count <
                selected_slot->computed_prompt_token_count)
            selected_slot->dispatched_prompt_token_count =
                selected_slot->computed_prompt_token_count;
        {
            uint32_t lane_committed;

            lane_committed = dispatch->prefill_batch_decision.lanes[
                request_index].cache_commit_token_count_after_step;
            if (lane_committed > selected_slot->prompt_token_count)
                lane_committed = selected_slot->prompt_token_count;
            if (lane_committed > selected_slot->dispatched_prompt_token_count)
                selected_slot->dispatched_prompt_token_count = lane_committed;
        }
        selected_slot->inflight_prefill_dispatch_count += 1u;
        api->queued_request_count -= 1u;
        api->running_request_count += 1u;
        dispatch->request_handles[request_index] = selected_slot->handle;
        dispatch->request_slot_indices[request_index] =
            SparkRequestApiSlotIndex(api, selected_slot);
        dispatch->request_ids[request_index] = selected_slot->request_id;
		dispatch->sequence_ids[request_index] = selected_slot->sequence_id;
	}
	if (SparkRequestModelSlotCanSpeculate(api,first_slot))
	{
		dispatch->flags |=
			SPARK_REQUEST_API_DISPATCH_FLAG_DSPARK_TAP_CAPTURE;
		api->dspark_tap_capture_dispatch_count += 1u;
	}
	api->scheduled_prefill_dispatch_count += 1u;
	return SPARK_STATUS_OK;
}

static void SparkRequestApiFillDecodeSchedulerRequest(
    SparkSchedulerRequest *scheduler_request)
{
    memset(scheduler_request, 0, sizeof(*scheduler_request));
    scheduler_request->abi_version = SPARK_SCHEDULER_ABI_VERSION;
    scheduler_request->descriptor_bytes =
        SPARK_SCHEDULER_REQUEST_DESCRIPTOR_BYTES;
    scheduler_request->active_sequence_count = 1u;
    scheduler_request->flags = SPARK_SCHEDULER_REQUEST_FLAG_DECODE;
}






static uint32_t SparkGlm52RequestApiCollectSpeculativeVerifyBatchMembers(
    SparkRequestApi *api,
    SparkRequestApiSlot *leader_slot,
    uint32_t leader_token_count,
    uint32_t leader_source,
    uint32_t require_resident_kv,
    SparkRequestApiSlot **selected_slots,
    uint32_t selected_capacity)
{
    uint32_t selected_count;
    uint32_t slot_index;

    selected_slots[0] = leader_slot;
    selected_count = 1u;
    for (slot_index = 0u; slot_index < api->request_capacity; ++slot_index)
    {
        SparkRequestApiSlot *slot;
        SparkRequestModelDraftResult draft_result;
        uint32_t draft_source;

		slot = &api->request_slots[slot_index];
		if (slot == leader_slot ||
			!SparkRequestApiSlotIsSchedulableSpeculativeVerify(slot) ||
			SparkRequestModelSlotCanSpeculate(api,slot) !=
				SparkRequestModelSlotCanSpeculate(api,leader_slot) ||
			((require_resident_kv != 0u ||
              slot->priority < leader_slot->priority) &&
             !SparkRequestApiDecodeBlocksAreResident(api, slot)))
        {
            continue;
        }
        if (SparkRequestModelGetSlotSpeculativeDraft(
                api,
                slot,
                leader_source,
                &draft_result,
                &draft_source) != SPARK_STATUS_OK ||
            draft_source != leader_source ||
            draft_result.token_count != leader_token_count)
        {
            continue;
        }
        SparkRequestApiInsertBatchMemberByPriority(
            selected_slots,
            &selected_count,
            selected_capacity,
            slot);
    }
    return selected_count;
}

static void SparkRequestApiFillSpeculativeVerifySchedulerRequest(
    SparkSchedulerRequest *scheduler_request)
{
    memset(scheduler_request, 0, sizeof(*scheduler_request));
    scheduler_request->abi_version = SPARK_SCHEDULER_ABI_VERSION;
    scheduler_request->descriptor_bytes =
        SPARK_SCHEDULER_REQUEST_DESCRIPTOR_BYTES;
    scheduler_request->active_sequence_count = 1u;
    scheduler_request->flags = SPARK_SCHEDULER_REQUEST_FLAG_DECODE;
}

static SparkStatus SparkRequestApiAdmitDecodeBatchMembers(
    SparkRequestApi *api,
    SparkRequestApiSlot **selected_slots,
    SparkSchedulerRequest *scheduler_requests,
    uint32_t *request_count_io,
    uint32_t context_extension,
    void (*fill_scheduler_request)(SparkSchedulerRequest *),
    SparkRequestApiDispatch *dispatch)
{
    SparkSchedulerBatchRequest batch_request;
    uint32_t request_count;
    uint32_t request_index;
    SparkStatus status;

    request_count = *request_count_io;
    status = SparkRequestApiApplyActiveKvBlockBudget(
        api,
        selected_slots,
        &request_count,
        context_extension);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    for (request_index = 0u; request_index < request_count; ++request_index)
    {
        fill_scheduler_request(
            &scheduler_requests[request_index]);
    }
    if (request_count == 0u)
    {
        return SPARK_STATUS_NOT_FOUND;
    }

    for (request_index = 0u; request_index < request_count; ++request_index)
    {
        status = SparkRequestApiEnsureDecodeSlotKvCapacity(
            api,
            selected_slots[request_index],
            context_extension,
            0);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }

    status = SparkRequestApiRunSlotArrayCriticalJitKvPrefetch(
        api,
        selected_slots,
        request_count,
        dispatch);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    for (request_index = 0u; request_index < request_count; ++request_index)
    {
        if (!SparkRequestApiDecodeBlocksAreResident(
                api,
                selected_slots[request_index]))
        {
            dispatch->flags |=
                SPARK_REQUEST_API_DISPATCH_FLAG_JIT_PREFETCH_PENDING;
            return SPARK_STATUS_BUSY;
        }
    }

    memset(&batch_request, 0, sizeof(batch_request));
    batch_request.abi_version = SPARK_SCHEDULER_ABI_VERSION;
    batch_request.descriptor_bytes =
        SPARK_SCHEDULER_BATCH_REQUEST_DESCRIPTOR_BYTES;
    batch_request.request_count = request_count;
    batch_request.requests = scheduler_requests;
    status = SparkSchedulerAdmitDecodeBatch(
        api->scheduler,
        &batch_request,
        &dispatch->decode_batch_decision);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (dispatch->decode_batch_decision.accepted == 0u)
    {
        return SPARK_STATUS_OK;
    }
    *request_count_io = request_count;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52RequestApiScheduleSpeculativeVerifyBatch(
    SparkRequestApi *api,
    SparkRequestApiSlot *first_slot,
    SparkRequestApiDispatch *dispatch)
{
    SparkSchedulerRequest scheduler_requests[
        SPARK_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT];
    SparkRequestApiSlot *selected_slots[
        SPARK_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT];
    SparkRequestModelDraftResult leader_draft;
    uint32_t leader_source;
    uint32_t batch_target;
    uint32_t request_count;
    uint32_t request_index;
    uint32_t token_index;
    uint32_t require_resident_batch_members;
    uint32_t speculative_context_extension;
    SparkStatus status;

    if (first_slot == 0)
    {
        return SPARK_STATUS_NOT_FOUND;
    }
    status = SparkRequestModelGetSlotSpeculativeDraft(
        api,
        first_slot,
        (api->configuration_flags &
            SPARK_REQUEST_API_CONFIGURATION_FLAG_PREFER_DSPARK_SPECULATION) != 0u &&
            SparkRequestModelDsparkSpeculationIsEnabled(api)
            ? SPARK_REQUEST_MODEL_SPECULATIVE_SOURCE_DRAFTER
            : 0u,
        &leader_draft,
        &leader_source);
    if (status != SPARK_STATUS_OK || leader_draft.token_count == 0u)
    {
        return status == SPARK_STATUS_NOT_FOUND ? SPARK_STATUS_NOT_FOUND : status;
    }

    batch_target = SparkRequestApiDecodeBatchingIsEnabled(api)
        ? SparkRequestApiCurrentPipelineBatchWidth(api)
        : 1u;
    if (batch_target > SPARK_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT)
    {
        batch_target = SPARK_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT;
    }
    {
        uint32_t row_limited_batch_target;
        uint32_t verifier_row_count;
        verifier_row_count =
            leader_source == SPARK_REQUEST_MODEL_SPECULATIVE_SOURCE_MTP
            ? SPARK_MODEL_MTP_TREE_VERIFIER_ROW_COUNT
            : leader_draft.token_count + 1u;
        row_limited_batch_target = api->decode_execution_row_capacity /
            verifier_row_count;
        if (row_limited_batch_target == 0u)
        {
            return SPARK_STATUS_CAPACITY_EXCEEDED;
        }
        if (batch_target > row_limited_batch_target)
        {
            batch_target = row_limited_batch_target;
        }
    }
    require_resident_batch_members =
        SparkRequestApiSlotHasRealtimePriority(first_slot) ||
        !SparkRequestApiJitPrefetchIsEnabled(api) ||
        SparkRequestApiAsyncJitPrefetchIsEnabled(api);
    request_count = SparkGlm52RequestApiCollectSpeculativeVerifyBatchMembers(
        api,
        first_slot,
        leader_draft.token_count,
        leader_source,
        require_resident_batch_members,
        selected_slots,
        batch_target);
    speculative_context_extension =
        leader_source == SPARK_REQUEST_MODEL_SPECULATIVE_SOURCE_MTP
            ? SPARK_MODEL_MTP_TREE_CONTEXT_EXTENSION
            : leader_draft.token_count;
    status = SparkRequestApiAdmitDecodeBatchMembers(
        api,
        selected_slots,
        scheduler_requests,
        &request_count,
        speculative_context_extension,
        SparkRequestApiFillSpeculativeVerifySchedulerRequest,
        dispatch);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (dispatch->decode_batch_decision.accepted == 0u)
    {
        return SPARK_STATUS_OK;
    }

    dispatch->accepted = 1u;
    dispatch->kind =
        SPARK_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH;
    if (leader_source == SPARK_REQUEST_MODEL_SPECULATIVE_SOURCE_MTP)
    {
        if (leader_draft.token_count !=
            SPARK_MODEL_MTP_TREE_CANDIDATE_COUNT)
        {
            return SPARK_STATUS_MODULE_NOT_VALIDATED;
        }
        dispatch->flags |=
            SPARK_REQUEST_API_DISPATCH_FLAG_MTP_SPECULATIVE_VERIFY |
            SPARK_REQUEST_API_DISPATCH_FLAG_MTP_TREE_VERIFY;
        dispatch->mtp_draft_token_budget =
            SPARK_MODEL_MTP_TREE_CANDIDATE_COUNT;
        dispatch->speculative_verifier_token_count =
            SPARK_MODEL_MTP_TREE_VERIFIER_ROW_COUNT;
        dispatch->speculative_max_committed_token_count =
            SPARK_MODEL_MTP_TREE_MAX_COMMITTED_TOKEN_COUNT;
    }
    else
    {
        dispatch->flags |=
            SPARK_REQUEST_API_DISPATCH_FLAG_DSPARK_SPECULATIVE_VERIFY |
            SPARK_REQUEST_API_DISPATCH_FLAG_DSPARK_TAP_CAPTURE;
        dispatch->speculative_verifier_token_count =
            leader_draft.token_count + 1u;
        dispatch->speculative_max_committed_token_count =
            leader_draft.token_count + 1u;
    }
    if (leader_source == SPARK_REQUEST_MODEL_SPECULATIVE_SOURCE_MTP &&
        SparkRequestModelSlotCanSpeculate(api,first_slot))
    {
        dispatch->flags |=
            SPARK_REQUEST_API_DISPATCH_FLAG_DSPARK_TAP_CAPTURE;
    }
    if ((dispatch->flags &
            SPARK_REQUEST_API_DISPATCH_FLAG_DSPARK_TAP_CAPTURE) != 0u)
        api->dspark_tap_capture_dispatch_count += 1u;
    dispatch->request_count =
        dispatch->decode_batch_decision.packed_request_count;
    dispatch->highest_priority = first_slot->priority;
    dispatch->speculative_token_count = leader_draft.token_count;

    for (request_index = 0u;
         request_index < dispatch->request_count;
         ++request_index)
    {
        SparkRequestApiSlot *selected_slot;
        SparkRequestModelDraftResult draft_result;
        uint32_t draft_source;

        selected_slot = selected_slots[request_index];
        status = SparkRequestModelGetSlotSpeculativeDraft(
            api,
            selected_slot,
            leader_source,
            &draft_result,
            &draft_source);
        if (status != SPARK_STATUS_OK ||
            draft_source != leader_source ||
            draft_result.token_count != leader_draft.token_count)
        {
            return status == SPARK_STATUS_OK ?
                SPARK_STATUS_INVALID_ARGUMENT : status;
        }
        selected_slot->state =
            SPARK_REQUEST_API_STATE_RUNNING_SPECULATIVE_VERIFY;
        selected_slot->scheduled_decode_token_count +=
            draft_result.token_count;
        api->running_request_count += 1u;
        dispatch->request_handles[request_index] = selected_slot->handle;
        dispatch->request_slot_indices[request_index] =
            SparkRequestApiSlotIndex(api, selected_slot);
        dispatch->request_ids[request_index] = selected_slot->request_id;
        dispatch->sequence_ids[request_index] = selected_slot->sequence_id;
        for (token_index = 0u;
             token_index < draft_result.token_count;
             ++token_index)
        {
            dispatch->speculative_draft_token_ids[request_index][token_index] =
                draft_result.token_ids[token_index];
            dispatch->speculative_confidence_milli[request_index][token_index] =
                draft_result.confidence_milli[token_index];
        }
    }
    api->scheduled_decode_dispatch_count += 1u;
    if (leader_source == SPARK_REQUEST_MODEL_SPECULATIVE_SOURCE_MTP)
    {
        api->mtp_verify_dispatch_count += 1u;
    }
    else
    {
        api->dspark_verify_dispatch_count += 1u;
    }
    return SPARK_STATUS_OK;
}

uint32_t SparkRequestApiSlotRemainingDecodeBudget(
    const SparkRequestApiSlot *slot)
{
    if (slot == 0)
    {
        return 0u;
    }
    return slot->remaining_thinking_token_budget +
        slot->remaining_output_token_budget;
}

uint64_t SparkRequestApiMtpResolvedRequestCount(
    const SparkRequestApi *api,
    uint64_t *committed_token_count_out)
{
    uint64_t proposed_token_count;

    proposed_token_count =
        api->mtp_accepted_draft_token_count +
        api->mtp_rejected_token_count;
    *committed_token_count_out = api->mtp_committed_token_count;
    /* A partially-resolved in-flight cycle leaves a non-zero remainder;
     * floor the completed cycle count instead of discarding every
     * sample, which previously zeroed the utility estimate. */
    return proposed_token_count /
        SPARK_MODEL_MTP_TREE_CANDIDATE_COUNT;
}



static uint32_t SparkRequestApiDecodeBatchMtpBudget(
    const SparkRequestApi *api,
    SparkRequestApiSlot *const *selected_slots,
    uint32_t request_count)
{
    uint32_t budget;
    uint32_t request_index;

    if (!SparkRequestApiMtpCommitIsEnabled(api) ||
        selected_slots == 0 || request_count == 0u ||
        !SparkRequestModelMtpOutranksPlainDecode(
            api,
            request_count,
            request_count))
    {
        return 0u;
    }
    budget = SPARK_REQUEST_API_MTP_MAX_DRAFT_TOKEN_COUNT;
    for (request_index = 0u; request_index < request_count; ++request_index)
    {
        const SparkRequestApiSlot *slot;
        uint32_t lane_budget;

        slot = selected_slots[request_index];
        if (slot == 0 ||
            (slot->flags &
                SPARK_REQUEST_API_REQUEST_FLAG_DISABLE_SPECULATION) != 0u ||
            SparkRequestApiSlotRemainingDecodeBudget(slot) <
                SPARK_MODEL_MTP_TREE_MAX_COMMITTED_TOKEN_COUNT + 1u)
        {
            return 0u;
        }
        lane_budget = SPARK_MODEL_MTP_TREE_CANDIDATE_COUNT;
        if (slot->mtp_next_draft_token_budget != lane_budget)
            return 0u;
        if (lane_budget < budget)
        {
            budget = lane_budget;
        }
    }
    return budget;
}

static void SparkRequestApiDiscardMtpDraft(
    SparkRequestApiSlot *slot)
{
    if (slot->state !=
            SPARK_REQUEST_API_STATE_READY_SPECULATIVE_VERIFY ||
        slot->mtp_draft_token_count == 0u)
    {
        return;
    }
    memset(slot->mtp_draft_token_ids,0,sizeof(slot->mtp_draft_token_ids));
    slot->mtp_draft_token_count = 0u;
    slot->state = SPARK_REQUEST_API_STATE_READY_DECODE;
}

static SparkStatus SparkGlm52RequestApiScheduleDecodeBatch(
    SparkRequestApi *api,
    SparkRequestApiSlot *first_slot,
    SparkRequestApiDispatch *dispatch)
{
    SparkSchedulerRequest scheduler_requests[
        SPARK_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT];
    SparkRequestApiSlot *selected_slots[
        SPARK_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT];
    uint32_t request_count;
    uint32_t request_index;
    uint32_t batch_target;
    uint32_t require_resident_batch_members;
    uint32_t batch_disables_speculation;
    uint32_t mtp_draft_token_budget;
    SparkStatus status;

    batch_disables_speculation = 0u;
    batch_target = SparkRequestApiDecodeBatchingIsEnabled(api)
        ? SparkRequestApiCurrentPipelineBatchWidth(api)
        : 1u;
    require_resident_batch_members =
        SparkRequestApiSlotHasRealtimePriority(first_slot) ||
        !SparkRequestApiJitPrefetchIsEnabled(api) ||
        SparkRequestApiAsyncJitPrefetchIsEnabled(api);
    request_count = SparkRequestApiCollectDecodeBatchMembers(
        api,
        first_slot,
        require_resident_batch_members,
        selected_slots,
        batch_target);
    mtp_draft_token_budget = SparkRequestApiDecodeBatchMtpBudget(
        api,selected_slots,request_count);
    status = SparkRequestApiAdmitDecodeBatchMembers(
        api,
        selected_slots,
        scheduler_requests,
        &request_count,
        mtp_draft_token_budget,
        SparkRequestApiFillDecodeSchedulerRequest,
        dispatch);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (dispatch->decode_batch_decision.accepted == 0u)
    {
        return SPARK_STATUS_OK;
    }

    dispatch->accepted = 1u;
    dispatch->kind = SPARK_REQUEST_API_DISPATCH_KIND_DECODE_BATCH;
    dispatch->request_count =
        dispatch->decode_batch_decision.packed_request_count;
    dispatch->highest_priority = first_slot->priority;
    for (request_index = 0u;
         request_index < dispatch->request_count;
         ++request_index)
    {
        SparkRequestApiSlot *selected_slot;

        selected_slot = selected_slots[request_index];
        SparkRequestApiDiscardMtpDraft(selected_slot);
        selected_slot->state = SPARK_REQUEST_API_STATE_RUNNING_DECODE;
        selected_slot->scheduled_decode_token_count += 1u;
        api->running_request_count += 1u;
        dispatch->request_handles[request_index] = selected_slot->handle;
        dispatch->request_slot_indices[request_index] =
            SparkRequestApiSlotIndex(api, selected_slot);
        dispatch->request_ids[request_index] = selected_slot->request_id;
        dispatch->sequence_ids[request_index] = selected_slot->sequence_id;
        dispatch->decode_committed_token_counts[request_index] = 1u;
		if ((selected_slot->flags &
				SPARK_REQUEST_API_REQUEST_FLAG_DISABLE_SPECULATION) != 0u)
		{
			batch_disables_speculation = 1u;
		}
	}
	if (SparkRequestModelSlotCanSpeculate(api,first_slot))
		dispatch->flags |=
			SPARK_REQUEST_API_DISPATCH_FLAG_DSPARK_TAP_CAPTURE;
    if (mtp_draft_token_budget != 0u &&
        batch_disables_speculation == 0u)
    {
        dispatch->flags |= SPARK_REQUEST_API_DISPATCH_FLAG_MTP_COMMIT;
        dispatch->mtp_draft_token_budget = mtp_draft_token_budget;
    }
    if ((dispatch->flags &
            SPARK_REQUEST_API_DISPATCH_FLAG_DSPARK_TAP_CAPTURE) != 0u)
    {
        api->dspark_tap_capture_dispatch_count += 1u;
    }
    api->scheduled_decode_dispatch_count += 1u;
    return SPARK_STATUS_OK;
}

static uint32_t SparkRequestApiMtpVerifyOutranksDecode(
    SparkRequestApi *api,
    SparkRequestApiSlot *decode_slot,
    SparkRequestApiSlot *speculative_verify_slot)
{
    (void)api;
    (void)decode_slot;
    /* A pending draft is sunk cost: the draft-chain dispatches are
     * already paid, so verifying commits ~E tokens for the price of one
     * verify batch while discarding the draft to run plain decode pays
     * the same batch price for one token and throws the draft work
     * away. Draft utility is therefore gated only where new drafts are
     * budgeted (SparkRequestApiDecodeBatchMtpBudget), never here —
     * gating here oscillated between drafting and discarding. */
    return speculative_verify_slot != 0 &&
        speculative_verify_slot->state ==
            SPARK_REQUEST_API_STATE_READY_SPECULATIVE_VERIFY;
}

static uint32_t SparkRequestApiShouldFillDecodeBatch(
	const SparkRequestApi *api,
	const SparkRequestApiSlot *prefill_slot,
	const SparkRequestApiSlot *decode_slot)
{
	const SparkRequestApiSlot *slot;
	uint32_t batch_target;
	uint32_t ready_decode_count;
	uint32_t slot_index;

	if (!SparkRequestApiDecodeBatchingIsEnabled(api) ||
		prefill_slot == 0 || decode_slot == 0 ||
		prefill_slot->priority < decode_slot->priority)
		return 0u;
	batch_target = SparkRequestApiCurrentPipelineBatchWidth(api);
	ready_decode_count = 0u;
	for (slot_index = 0u;
		 slot_index < api->request_capacity &&
			ready_decode_count < batch_target;
		 ++slot_index)
	{
		slot = &api->request_slots[slot_index];
		if (SparkRequestApiSlotIsSchedulableDecode(slot) &&
			SparkRequestApiSlotsHaveSameSchedulingPriority(
				slot,
				decode_slot))
			ready_decode_count += 1u;
	}
	return ready_decode_count < batch_target;
}

static uint32_t SparkRequestApiPrefillHasResidentKvHeadroom(
	const SparkRequestApi *api,
	const SparkRequestApiSlot *prefill_slot)
{
	uint64_t reserved_block_count;

	reserved_block_count = SparkRequestApiResidentKvBlockCount(api);
	return SparkRequestApiReservePrefillResidentKvBlocks(
		api,prefill_slot,&reserved_block_count);
}

static SparkRequestApiSlot *SparkRequestApiChooseReadySlot(
	SparkRequestApi *api,
	SparkRequestApiSlot *prefill_slot,
	SparkRequestApiSlot *decode_slot,
	SparkRequestApiSlot *speculative_verify_slot,
	uint32_t *chosen_is_prefill)
{
	SparkRequestApiSlot *chosen_slot;

	SparkSchedulerSetPrefillDemand(
		api->scheduler,
		prefill_slot != 0 ? 1u : 0u);
	*chosen_is_prefill = 0u;
	chosen_slot = decode_slot;
	if (prefill_slot != 0 &&
		(chosen_slot == 0 ||
		 SparkGlm52RequestApiSlotHasHigherSchedulingPriority(
			prefill_slot,
			chosen_slot) ||
		 (chosen_slot == decode_slot &&
		  ((SparkRequestApiSlotsHaveSameSchedulingPriority(
			prefill_slot,decode_slot) &&
			SparkRequestApiPrefillHasResidentKvHeadroom(
				api,prefill_slot) != 0u) ||
		   SparkRequestApiShouldFillDecodeBatch(
			api,prefill_slot,decode_slot) != 0u))))
	{
		*chosen_is_prefill = 1u;
		chosen_slot = prefill_slot;
	}
	if (speculative_verify_slot != 0 &&
		(chosen_slot == 0 ||
		 speculative_verify_slot->priority > chosen_slot->priority ||
		 (*chosen_is_prefill != 0u &&
		  SparkRequestApiSlotsHaveSameSchedulingPriority(
			speculative_verify_slot,chosen_slot)) ||
		 (chosen_slot == decode_slot &&
		  SparkRequestApiSlotsHaveSameSchedulingPriority(
			speculative_verify_slot,decode_slot) &&
		  SparkRequestApiMtpVerifyOutranksDecode(
			api,decode_slot,speculative_verify_slot))))
	{
		*chosen_is_prefill = 0u;
		chosen_slot = speculative_verify_slot;
	}
	return chosen_slot;
}

SparkStatus SparkRequestApiScheduleNext(
    SparkRequestApi *api,
    SparkRequestApiDispatch *dispatch)
{
    SparkRequestApiSlot *prefill_slot;
    SparkRequestApiSlot *decode_slot;
    SparkRequestApiSlot *speculative_verify_slot;
    SparkRequestApiSlot *chosen_slot;
    SparkGlm52RequestApiPrefixFamilyChoice prefix_family_choice;
    uint32_t chosen_is_prefill;
    uint32_t overlaps_pending_prefetch;
    uint32_t selected_shared_prefix_token_count;
    SparkStatus status;

    status = SparkRequestApiValidate(api);
    if (status != SPARK_STATUS_OK || dispatch == 0)
    {
        return status == SPARK_STATUS_OK ? SPARK_STATUS_INVALID_ARGUMENT : status;
    }

    SparkRequestApiInitializeDispatch(dispatch);
    if (SparkRequestApiAsyncJitPrefetchIsEnabled(api))
    {
        uint64_t completed_prefetch_count_before_poll;

        completed_prefetch_count_before_poll =
            api->async_jit_prefetch_completion_count;
        status = SparkRequestApiPollPendingJitKvPrefetches(api);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        if (api->async_jit_prefetch_completion_count >
            completed_prefetch_count_before_poll)
        {
            dispatch->flags |=
                SPARK_REQUEST_API_DISPATCH_FLAG_JIT_PREFETCHED_KV;
        }
    }
    status = SparkRequestApiRefreshLookaheadPrefixProtections(api);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    selected_shared_prefix_token_count = 0u;
    overlaps_pending_prefetch = 0u;
    prefill_slot = SparkRequestApiFindBestPrefillSlot(api, 0u);
    if (SparkGlm52RequestApiBuildBestPrefixFamilyChoice(
            api,
            &prefix_family_choice) &&
        SparkRequestApiPrefixFamilyChoiceBeatsPrefillSlot(
            &prefix_family_choice,
            prefill_slot))
    {
        prefill_slot = prefix_family_choice.leader_slot;
        selected_shared_prefix_token_count =
            prefix_family_choice.shared_prefix_token_count;
    }
    speculative_verify_slot = SparkRequestApiFindBestSchedulableSlot(
        api,
        SparkRequestApiSlotIsSchedulableSpeculativeVerify,
        0,
        0u,
        SparkRequestApiJitPrefetchIsEnabled(api) ? 0u : 1u);
    decode_slot = SparkRequestApiFindBestSchedulableSlot(
        api,
        SparkRequestApiSlotIsSchedulableDecode,
        0,
        0u,
        SparkRequestApiJitPrefetchIsEnabled(api) ? 0u : 1u);
    if (prefill_slot == 0 && decode_slot == 0 && speculative_verify_slot == 0)
    {
        return SPARK_STATUS_NOT_FOUND;
    }

    chosen_slot = SparkRequestApiChooseReadySlot(
        api,
        prefill_slot,
        decode_slot,
        speculative_verify_slot,
        &chosen_is_prefill);

    status = SparkRequestApiRunSlotArrayCriticalJitKvPrefetch(
        api,
        &chosen_slot,
        1u,
        dispatch);
    if (status == SPARK_STATUS_BUSY &&
        SparkRequestApiAsyncJitPrefetchIsEnabled(api))
    {
        uint32_t pending_dispatch_flags;

        pending_dispatch_flags = dispatch->flags |
            SPARK_REQUEST_API_DISPATCH_FLAG_JIT_PREFETCH_PENDING;
        overlaps_pending_prefetch = 1u;
        selected_shared_prefix_token_count = 0u;
        prefill_slot = SparkRequestApiFindBestPrefillSlot(api, 1u);
        speculative_verify_slot =
            SparkRequestApiFindBestSchedulableSlot(
        api,
        SparkRequestApiSlotIsSchedulableSpeculativeVerify, 0, 0u, 1u);
        decode_slot = SparkRequestApiFindBestSchedulableSlot(api, SparkRequestApiSlotIsSchedulableDecode, 0, 0u, 1u);
        chosen_slot = SparkRequestApiChooseReadySlot(
            api,
            prefill_slot,
            decode_slot,
            speculative_verify_slot,
            &chosen_is_prefill);
        if (chosen_slot == 0)
        {
            return SPARK_STATUS_BUSY;
        }
        SparkRequestApiInitializeDispatch(dispatch);
        dispatch->flags = pending_dispatch_flags;
        status = SparkRequestApiRunSlotArrayCriticalJitKvPrefetch(
            api,
            &chosen_slot,
            1u,
            dispatch);
    }
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (overlaps_pending_prefetch == 0u)
    {
        status = SparkRequestApiRunOpportunisticJitKvPrefetch(
            api,
            chosen_slot);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }

    if (chosen_is_prefill != 0u)
    {
        if ((decode_slot != 0 &&
             SparkGlm52RequestApiSlotHasHigherSchedulingPriority(
                prefill_slot,
                decode_slot)) ||
            SparkRequestApiOlderLowerPrioritySchedulableSlotExists(
                api,
                prefill_slot))
        {
            dispatch->flags |=
                SPARK_REQUEST_API_DISPATCH_FLAG_PRIORITY_PREEMPTED_QUEUE;
        }
        if ((selected_shared_prefix_token_count != 0u
                ? selected_shared_prefix_token_count
                : SparkRequestApiFindBestSharedPrefixTokenCount(
                    api,
                    prefill_slot)) <=
            SparkRequestApiProbeReusablePrefixTokenCount(
                api,
                prefill_slot))
        {
            status = SparkGlm52RequestApiSchedulePrefillBatch(
                api,
                prefill_slot,
                dispatch);
            if (status == SPARK_STATUS_OK && dispatch->accepted != 0u)
            {
                return SPARK_STATUS_OK;
            }
            if (status != SPARK_STATUS_OK && status != SPARK_STATUS_BUSY &&
                status != SPARK_STATUS_NOT_FOUND)
            {
                return status;
            }
        }
        status = SparkGlm52RequestApiSchedulePrefill(
            api,
            prefill_slot,
            selected_shared_prefix_token_count,
            dispatch);
        if (status == SPARK_STATUS_OK && dispatch->accepted != 0u)
        {
            return SPARK_STATUS_OK;
        }
        if (status != SPARK_STATUS_OK && status != SPARK_STATUS_BUSY)
        {
            return status;
        }
        if (decode_slot == 0 && speculative_verify_slot == 0)
        {
            return status;
        }
        chosen_slot = decode_slot != 0 ? decode_slot : speculative_verify_slot;
        chosen_is_prefill = 0u;
        {
            uint32_t saved_dispatch_flags;

            saved_dispatch_flags = dispatch->flags &
                SPARK_REQUEST_API_DISPATCH_FLAG_JIT_PREFETCH_PENDING;
            SparkRequestApiInitializeDispatch(dispatch);
            dispatch->flags = saved_dispatch_flags;
        }
    }

    if (chosen_slot == speculative_verify_slot && speculative_verify_slot != 0)
    {
        if (!SparkRequestApiDecodeBlocksAreResident(
                api,
                speculative_verify_slot))
        {
            dispatch->flags |=
                SPARK_REQUEST_API_DISPATCH_FLAG_JIT_PREFETCH_PENDING;
            return SPARK_STATUS_BUSY;
        }
        if (SparkRequestApiOlderLowerPrioritySchedulableSlotExists(
                api,
                speculative_verify_slot))
        {
            dispatch->flags |=
                SPARK_REQUEST_API_DISPATCH_FLAG_PRIORITY_PREEMPTED_QUEUE;
        }
        return SparkGlm52RequestApiScheduleSpeculativeVerifyBatch(
            api,
            speculative_verify_slot,
            dispatch);
    }

    if (!SparkRequestApiDecodeBlocksAreResident(api, decode_slot))
    {
        dispatch->flags |=
            SPARK_REQUEST_API_DISPATCH_FLAG_JIT_PREFETCH_PENDING;
        return SPARK_STATUS_BUSY;
    }
    if (SparkRequestApiOlderLowerPrioritySchedulableSlotExists(
            api,
            decode_slot))
    {
        dispatch->flags |=
            SPARK_REQUEST_API_DISPATCH_FLAG_PRIORITY_PREEMPTED_QUEUE;
    }
    return SparkGlm52RequestApiScheduleDecodeBatch(api, decode_slot, dispatch);
}

static void SparkRequestApiFinishSlotAfterPrefill(
    SparkRequestApi *api,
    SparkRequestApiSlot *slot,
    const SparkSchedulerDecision *decision)
{
    uint32_t committed_prompt_token_count;

    if (slot->inflight_prefill_dispatch_count != 0u)
        slot->inflight_prefill_dispatch_count -= 1u;
    api->running_request_count -= 1u;
    slot->completed_prefill_step_count += 1u;
    committed_prompt_token_count = decision->cache_commit_token_count_after_step;
    if (committed_prompt_token_count > slot->prompt_token_count)
    {
        committed_prompt_token_count = slot->prompt_token_count;
    }
    if (committed_prompt_token_count > slot->computed_prompt_token_count)
    {
        slot->computed_prompt_token_count = committed_prompt_token_count;
        slot->last_committed_prefix_token_count = committed_prompt_token_count;
        slot->last_committed_prefix_hash = decision->prefix_cache_result_hash;
    }
    if (slot->computed_prompt_token_count < slot->prompt_token_count)
    {
        if (slot->inflight_prefill_dispatch_count != 0u)
            return;
        if (slot->dispatched_prompt_token_count >
            slot->computed_prompt_token_count)
            slot->dispatched_prompt_token_count =
                slot->computed_prompt_token_count;
        slot->state = SPARK_REQUEST_API_STATE_QUEUED_PREFILL;
        api->queued_request_count += 1u;
        return;
    }
    if (slot->inflight_prefill_dispatch_count != 0u)
        return;

    slot->state = SPARK_REQUEST_API_STATE_READY_DECODE;
    if (slot->remaining_thinking_token_budget == 0u &&
        slot->remaining_output_token_budget == 0u)
    {
        slot->state = SPARK_REQUEST_API_STATE_COMPLETED;
        api->completed_request_count += 1u;
    }
}


static void SparkRequestApiFinishSlotAfterPrefillBatchLane(
    SparkRequestApi *api,
    SparkRequestApiSlot *slot,
    const SparkSchedulerPrefillBatchLane *lane)
{
    uint32_t committed_prompt_token_count;

    if (slot->inflight_prefill_dispatch_count != 0u)
        slot->inflight_prefill_dispatch_count -= 1u;
    committed_prompt_token_count = lane->cache_commit_token_count_after_step;
    if (committed_prompt_token_count > slot->prompt_token_count)
    {
        committed_prompt_token_count = slot->prompt_token_count;
    }
    if (committed_prompt_token_count > slot->computed_prompt_token_count)
    {
        slot->computed_prompt_token_count = committed_prompt_token_count;
        slot->last_committed_prefix_token_count = committed_prompt_token_count;
        slot->last_committed_prefix_hash = lane->prefix_cache_result_hash;
    }

    slot->completed_prefill_step_count += 1u;
    api->running_request_count -= 1u;
    if (slot->computed_prompt_token_count < slot->prompt_token_count)
    {
        if (slot->dispatched_prompt_token_count >
            slot->computed_prompt_token_count)
            slot->dispatched_prompt_token_count =
                slot->computed_prompt_token_count;
        slot->state = SPARK_REQUEST_API_STATE_QUEUED_PREFILL;
        api->queued_request_count += 1u;
        return;
    }

    slot->state = SPARK_REQUEST_API_STATE_READY_DECODE;
    if (slot->remaining_thinking_token_budget == 0u &&
        slot->remaining_output_token_budget == 0u)
    {
        slot->state = SPARK_REQUEST_API_STATE_COMPLETED;
        api->completed_request_count += 1u;
    }
}

static void SparkRequestApiConsumeDecodeBudget(
    SparkRequestApiSlot *slot,
    uint32_t committed_token_count)
{
    uint32_t consumed_token_count;

    consumed_token_count = 0u;
    while (consumed_token_count < committed_token_count &&
           slot->remaining_thinking_token_budget != 0u)
    {
        slot->remaining_thinking_token_budget -= 1u;
        consumed_token_count += 1u;
    }
    while (consumed_token_count < committed_token_count &&
           slot->remaining_output_token_budget != 0u)
    {
        slot->remaining_output_token_budget -= 1u;
        consumed_token_count += 1u;
    }
}


static void SparkRequestApiFinishSlotAfterDecode(
    SparkRequestApi *api,
    SparkRequestApiSlot *slot,
    uint32_t committed_token_count)
{
    SparkStatus status;

    slot->mtp_resolution_base_position = 0u;
    slot->mtp_resolution_proposed_token_count = 0u;
    slot->mtp_resolution_accepted_token_count = 0u;
    slot->mtp_resolution_committed_token_count = 0u;
    slot->mtp_resolution_path_id = SPARK_MODEL_MTP_TREE_RESOLUTION_NONE;
    if (slot->mtp_next_draft_token_budget == 0u &&
        slot->mtp_probe_countdown != 0u)
    {
        if (slot->mtp_probe_countdown > committed_token_count)
        {
            slot->mtp_probe_countdown -= committed_token_count;
        }
        else
        {
            slot->mtp_probe_countdown = 0u;
            slot->mtp_next_draft_token_budget =
                SPARK_MODEL_MTP_TREE_CANDIDATE_COUNT;
        }
    }
    SparkRequestApiConsumeDecodeBudget(slot, committed_token_count);
    slot->completed_decode_token_count += committed_token_count;
    api->running_request_count -= 1u;
    if (slot->remaining_thinking_token_budget == 0u &&
        slot->remaining_output_token_budget == 0u)
    {
        slot->state = SPARK_REQUEST_API_STATE_COMPLETED;
        api->completed_request_count += 1u;
        return;
    }

    slot->state = SPARK_REQUEST_API_STATE_READY_DECODE;
    status = SparkRequestModelPrepareDraftForSlot(api, slot);
    if (status != SPARK_STATUS_OK && status != SPARK_STATUS_NOT_FOUND)
    {
        slot->state = SPARK_REQUEST_API_STATE_READY_DECODE;
    }
}

static SparkStatus SparkGlm52RequestApiFinishSlotAfterSpeculativeVerify(
    SparkRequestApi *api,
    SparkRequestApiSlot *slot,
    uint32_t proposed_token_count,
    uint32_t accepted_draft_token_count,
    uint32_t committed_token_count,
    uint32_t fallback_token_id,
    uint32_t resolution_path_id,
    uint32_t mtp_verify)
{
    SparkRequestModelVerifyResult verify_result;
    uint64_t resolution_base_position;
    SparkStatus status;

    if (proposed_token_count == 0u ||
        proposed_token_count > SPARK_REQUEST_MODEL_MAX_SPECULATIVE_TOKENS ||
        accepted_draft_token_count > proposed_token_count ||
        committed_token_count == 0u ||
        committed_token_count > proposed_token_count + 1u ||
        committed_token_count > SparkRequestApiSlotRemainingDecodeBudget(slot))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    resolution_base_position = 0u;
    if (mtp_verify != 0u)
    {
        if (proposed_token_count !=
                SPARK_MODEL_MTP_TREE_CANDIDATE_COUNT ||
            accepted_draft_token_count >
                SPARK_MODEL_MTP_TREE_CONTEXT_EXTENSION ||
            committed_token_count != accepted_draft_token_count + 1u ||
            committed_token_count >
                SPARK_MODEL_MTP_TREE_MAX_COMMITTED_TOKEN_COUNT ||
            resolution_path_id >=
                SPARK_MODEL_MTP_TREE_RESOLUTION_COUNT ||
            SparkMtpTreeAcceptedTokenCount(
                resolution_path_id) != accepted_draft_token_count ||
            slot->computed_prompt_token_count == 0u)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        resolution_base_position =
            (uint64_t)slot->computed_prompt_token_count +
            (uint64_t)slot->completed_decode_token_count - 1u;
    }

    memset(&verify_result, 0, sizeof(verify_result));
    verify_result.abi_version = SPARK_REQUEST_MODEL_ABI_VERSION;
    verify_result.descriptor_bytes =
        SPARK_REQUEST_MODEL_VERIFY_RESULT_DESCRIPTOR_BYTES;
    verify_result.proposed_token_count = proposed_token_count;
    verify_result.accepted_draft_token_count = accepted_draft_token_count;
    verify_result.committed_token_count = committed_token_count;
    verify_result.fallback_token_id = fallback_token_id;
    if (accepted_draft_token_count == proposed_token_count)
    {
        verify_result.flags |= SPARK_REQUEST_MODEL_VERIFY_RESULT_FLAG_ACCEPTED_ALL;
    }
    else
    {
        verify_result.flags |= SPARK_REQUEST_MODEL_VERIFY_RESULT_FLAG_REJECTED;
    }

    if (mtp_verify != 0u)
    {
        int32_t ema_delta;
        if (slot->mtp_draft_token_count != proposed_token_count)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        ema_delta = (int32_t)(committed_token_count * 1000u) -
            (int32_t)slot->mtp_commit_ema_milli;
        slot->mtp_commit_ema_milli = (uint32_t)(
            (int32_t)slot->mtp_commit_ema_milli +
            ema_delta / SPARK_REQUEST_API_MTP_COMMIT_EMA_DIVISOR);
        if (slot->mtp_commit_ema_milli <
            SPARK_REQUEST_API_MTP_SUPPRESS_THRESHOLD_MILLI)
        {
            slot->mtp_next_draft_token_budget = 0u;
            slot->mtp_probe_countdown =
                SPARK_REQUEST_API_MTP_REPROBE_INTERVAL;
        }
        else
        {
            slot->mtp_next_draft_token_budget =
                SPARK_MODEL_MTP_TREE_CANDIDATE_COUNT;
            slot->mtp_probe_countdown = 0u;
        }
        slot->mtp_resolution_base_position = resolution_base_position;
        slot->mtp_resolution_proposed_token_count = proposed_token_count;
        slot->mtp_resolution_accepted_token_count = accepted_draft_token_count;
        slot->mtp_resolution_committed_token_count = committed_token_count;
        slot->mtp_resolution_path_id = resolution_path_id;
        memset(slot->mtp_draft_token_ids, 0, sizeof(slot->mtp_draft_token_ids));
        slot->mtp_draft_token_count = 0u;
        api->mtp_accepted_draft_token_count += accepted_draft_token_count;
        api->mtp_committed_token_count += committed_token_count;
        if (accepted_draft_token_count < proposed_token_count)
        {
            api->mtp_rejected_token_count +=
                proposed_token_count - accepted_draft_token_count;
        }
    }
    else
    {
        slot->mtp_resolution_base_position = 0u;
        slot->mtp_resolution_proposed_token_count = 0u;
        slot->mtp_resolution_accepted_token_count = 0u;
        slot->mtp_resolution_committed_token_count = 0u;
        if (resolution_path_id !=
            SPARK_MODEL_MTP_TREE_RESOLUTION_NONE)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        slot->mtp_resolution_path_id =
            SPARK_MODEL_MTP_TREE_RESOLUTION_NONE;
        if (slot->mtp_draft_token_count != 0u)
        {
            memset(slot->mtp_draft_token_ids,0,sizeof(slot->mtp_draft_token_ids));
            slot->mtp_draft_token_count = 0u;
            if (api->mtp_draft_ready_count != 0u)
                api->mtp_draft_ready_count -= 1u;
        }
        status = SparkRequestModelCompleteVerify(
            api,
            slot->sequence_id,
            &verify_result);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        api->dspark_accepted_draft_token_count += accepted_draft_token_count;
        api->dspark_committed_token_count += committed_token_count;
        if (accepted_draft_token_count < proposed_token_count)
        {
            api->dspark_rejected_token_count +=
                proposed_token_count - accepted_draft_token_count;
        }
    }

    SparkRequestApiConsumeDecodeBudget(slot, committed_token_count);
    slot->completed_decode_token_count += committed_token_count;
    api->running_request_count -= 1u;

    if (slot->remaining_thinking_token_budget == 0u &&
        slot->remaining_output_token_budget == 0u)
    {
        slot->state = SPARK_REQUEST_API_STATE_COMPLETED;
        api->completed_request_count += 1u;
        return SPARK_STATUS_OK;
    }

    slot->state = SPARK_REQUEST_API_STATE_READY_DECODE;
    status = SparkRequestModelPrepareDraftForSlot(api, slot);
    if (status != SPARK_STATUS_OK && status != SPARK_STATUS_NOT_FOUND)
    {
        slot->state = SPARK_REQUEST_API_STATE_READY_DECODE;
    }
    return SPARK_STATUS_OK;
}


SparkStatus SparkRequestApiArmMtpVerifyDispatch(
    SparkRequestApi *api,
    const SparkRequestApiDispatch *completed_decode_dispatch,
    const uint32_t *draft_token_ids,
    uint32_t lane_stride,
    uint32_t draft_token_count)
{
	uint32_t arm_draft_token_count;
	uint32_t dispatch_is_mtp_producer;
	uint32_t dispatch_is_mtp_verify;
    uint32_t request_index;
    uint32_t token_index;
    SparkStatus status;

    status = SparkRequestApiValidate(api);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    dispatch_is_mtp_producer = completed_decode_dispatch != 0 &&
        completed_decode_dispatch->kind ==
            SPARK_REQUEST_API_DISPATCH_KIND_DECODE_BATCH &&
        (completed_decode_dispatch->flags &
            SPARK_REQUEST_API_DISPATCH_FLAG_MTP_COMMIT) != 0u;
    dispatch_is_mtp_verify = completed_decode_dispatch != 0 &&
        completed_decode_dispatch->kind ==
            SPARK_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH &&
        (completed_decode_dispatch->flags &
            SPARK_REQUEST_API_DISPATCH_FLAG_MTP_SPECULATIVE_VERIFY) != 0u;
    if (completed_decode_dispatch == 0 ||
        completed_decode_dispatch->abi_version !=
            SPARK_REQUEST_API_ABI_VERSION ||
        completed_decode_dispatch->descriptor_bytes !=
            SPARK_REQUEST_API_DISPATCH_DESCRIPTOR_BYTES ||
        completed_decode_dispatch->accepted == 0u ||
        (dispatch_is_mtp_producer == 0u && dispatch_is_mtp_verify == 0u) ||
        completed_decode_dispatch->request_count == 0u ||
        completed_decode_dispatch->request_count >
            SPARK_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT ||
        draft_token_ids == 0 ||
        draft_token_count == 0u ||
        draft_token_count >
            SPARK_REQUEST_API_MTP_MAX_DRAFT_TOKEN_COUNT ||
        (dispatch_is_mtp_producer != 0u &&
         draft_token_count > completed_decode_dispatch->mtp_draft_token_budget) ||
        lane_stride < draft_token_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if (draft_token_count != SPARK_MODEL_MTP_TREE_CANDIDATE_COUNT ||
        (dispatch_is_mtp_producer != 0u &&
         completed_decode_dispatch->mtp_draft_token_budget !=
            SPARK_MODEL_MTP_TREE_CANDIDATE_COUNT) ||
        (dispatch_is_mtp_verify != 0u &&
         ((completed_decode_dispatch->flags &
            SPARK_REQUEST_API_DISPATCH_FLAG_MTP_TREE_VERIFY) == 0u ||
          completed_decode_dispatch->mtp_draft_token_budget !=
            SPARK_MODEL_MTP_TREE_CANDIDATE_COUNT)))
    {
        return SPARK_STATUS_MODULE_NOT_VALIDATED;
    }
    arm_draft_token_count = SPARK_MODEL_MTP_TREE_CANDIDATE_COUNT;
    for (request_index = 0u;
         request_index < completed_decode_dispatch->request_count;
         ++request_index)
    {
        SparkRequestApiSlot *slot;

        slot = SparkRequestApiFindSlotByHandle(
            api,
            completed_decode_dispatch->request_handles[request_index]);
        if (slot == 0)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        if (slot->state == SPARK_REQUEST_API_STATE_COMPLETED ||
            SparkRequestApiSlotRemainingDecodeBudget(slot) <
                SPARK_MODEL_MTP_TREE_MAX_COMMITTED_TOKEN_COUNT)
            return SPARK_STATUS_NOT_FOUND;
        if ((slot->state != SPARK_REQUEST_API_STATE_READY_DECODE &&
             slot->state !=
                SPARK_REQUEST_API_STATE_READY_SPECULATIVE_VERIFY) ||
            slot->mtp_draft_token_count != 0u)
            return SPARK_STATUS_INVALID_ARGUMENT;
        if (slot->mtp_next_draft_token_budget == 0u)
            return SPARK_STATUS_NOT_FOUND;
        if (slot->mtp_next_draft_token_budget !=
            SPARK_MODEL_MTP_TREE_CANDIDATE_COUNT)
            return SPARK_STATUS_MODULE_NOT_VALIDATED;
    }
    if (arm_draft_token_count == 0u)
        return SPARK_STATUS_NOT_FOUND;
    for (request_index = 0u;
         request_index < completed_decode_dispatch->request_count;
         ++request_index)
    {
        SparkRequestApiSlot *slot;

        slot = SparkRequestApiFindSlotByHandle(
            api,
            completed_decode_dispatch->request_handles[request_index]);
        for (token_index = 0u;
             token_index < arm_draft_token_count;
             ++token_index)
        {
            slot->mtp_draft_token_ids[token_index] =
                draft_token_ids[(uint64_t)request_index * lane_stride +
                    token_index];
        }
        for (; token_index < SPARK_REQUEST_API_MTP_MAX_DRAFT_TOKEN_COUNT;
             ++token_index)
        {
            slot->mtp_draft_token_ids[token_index] = 0u;
        }
        slot->mtp_draft_token_count = arm_draft_token_count;
        if (slot->state ==
                SPARK_REQUEST_API_STATE_READY_SPECULATIVE_VERIFY &&
            api->dspark_draft_ready_count != 0u)
            api->dspark_draft_ready_count -= 1u;
        slot->state = SPARK_REQUEST_API_STATE_READY_SPECULATIVE_VERIFY;
        api->mtp_draft_ready_count += 1u;
    }
    return SPARK_STATUS_OK;
}


static const char *SparkRequestApiSpeculativeTraceSource(
    const SparkRequestApiDispatch *dispatch,
    uint32_t *trace_confidence)
{
    if (dispatch == 0 || trace_confidence == 0)
    {
        return 0;
    }
    if ((dispatch->flags &
            SPARK_REQUEST_API_DISPATCH_FLAG_DSPARK_SPECULATIVE_VERIFY) != 0u)
    {
        if (getenv("SPARKPIPE_DSPARK_TRACE") == 0)
        {
            return 0;
        }
        *trace_confidence = 1u;
        return "dspark";
    }
    if ((dispatch->flags &
            SPARK_REQUEST_API_DISPATCH_FLAG_MTP_SPECULATIVE_VERIFY) != 0u)
    {
        if (getenv("SPARKPIPE_RING_TRACE") == 0)
        {
            return 0;
        }
        *trace_confidence = 0u;
        return "mtp";
    }
    return 0;
}

static void SparkRequestApiTraceTokenIds(
    const char *label,
    const uint32_t *token_ids,
    uint32_t token_count)
{
    uint32_t token_index;

    fprintf(stderr," %s=",label);
    for (token_index = 0u; token_index < token_count; ++token_index)
    {
        fprintf(stderr,"%s%u",token_index == 0u ? "" : ",",
            token_ids[token_index]);
    }
}

static void SparkRequestApiTraceSpeculativeVerify(
    const SparkRequestApiDispatch *dispatch,
    const uint32_t *verifier_token_ids,
    uint32_t lane_stride,
    uint32_t verifier_token_count,
    uint32_t request_index,
    const SparkRequestModelVerifyResult *verify_result)
{
    const char *source;
    uint32_t trace_confidence;

    if (verifier_token_ids == 0 || verify_result == 0)
    {
        return;
    }
    source = SparkRequestApiSpeculativeTraceSource(
        dispatch,&trace_confidence);
    if (source == 0)
    {
        return;
    }
    fprintf(stderr,
        "%s_trace verify request=%llu sequence=%llu proposed=%u accepted=%u committed=%u fallback=%u",
        source,
        (unsigned long long)dispatch->request_ids[request_index],
        (unsigned long long)dispatch->sequence_ids[request_index],
        dispatch->speculative_token_count,
        verify_result->accepted_draft_token_count,
        verify_result->committed_token_count,
        verify_result->fallback_token_id);
    SparkRequestApiTraceTokenIds(
        "draft_ids",
        dispatch->speculative_draft_token_ids[request_index],
        dispatch->speculative_token_count);
    if (trace_confidence != 0u)
        SparkRequestApiTraceTokenIds(
            "confidence_milli",
            dispatch->speculative_confidence_milli[request_index],
            dispatch->speculative_token_count);
    SparkRequestApiTraceTokenIds(
        "verifier_ids",
        &verifier_token_ids[(uint64_t)request_index * lane_stride],
        verifier_token_count);
    fprintf(stderr,"\n");
}

SparkStatus SparkRequestApiResolveSpeculativeVerifyDispatch(
    SparkRequestApi *api,
    SparkRequestApiDispatch *dispatch,
    const uint32_t *verifier_token_ids,
    uint32_t lane_stride,
    uint32_t verifier_token_count)
{
    uint32_t request_index;
    SparkStatus status;

    status = SparkRequestApiValidate(api);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (dispatch == 0 ||
        dispatch->abi_version != SPARK_REQUEST_API_ABI_VERSION ||
        dispatch->descriptor_bytes !=
            SPARK_REQUEST_API_DISPATCH_DESCRIPTOR_BYTES ||
        dispatch->accepted == 0u ||
        dispatch->kind !=
            SPARK_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH ||
        dispatch->request_count == 0u ||
        dispatch->request_count >
            SPARK_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT ||
        dispatch->speculative_token_count == 0u ||
        dispatch->speculative_token_count >
            SPARK_REQUEST_MODEL_MAX_SPECULATIVE_TOKENS ||
        verifier_token_ids == 0 ||
        dispatch->speculative_verifier_token_count == 0u ||
        verifier_token_count != dispatch->speculative_verifier_token_count ||
        lane_stride < verifier_token_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    for (request_index = 0u;
         request_index < dispatch->request_count;
         ++request_index)
    {
        SparkRequestModelVerifyResult verify_result;
        uint32_t resolution_path_id;

        resolution_path_id = SPARK_MODEL_MTP_TREE_RESOLUTION_NONE;
        if ((dispatch->flags &
            SPARK_REQUEST_API_DISPATCH_FLAG_MTP_TREE_VERIFY) != 0u)
        {
            if ((dispatch->flags &
                    SPARK_REQUEST_API_DISPATCH_FLAG_MTP_SPECULATIVE_VERIFY) == 0u ||
                dispatch->speculative_token_count !=
                    SPARK_MODEL_MTP_TREE_CANDIDATE_COUNT ||
                dispatch->speculative_verifier_token_count !=
                    SPARK_MODEL_MTP_TREE_VERIFIER_ROW_COUNT ||
                dispatch->speculative_max_committed_token_count !=
                    SPARK_MODEL_MTP_TREE_MAX_COMMITTED_TOKEN_COUNT)
            {
                return SPARK_STATUS_INVALID_ARGUMENT;
            }
            status = SparkRequestModelResolveMtpTreeVerifierTokens(
                dispatch->speculative_draft_token_ids[request_index],
                &verifier_token_ids[(uint64_t)request_index * lane_stride],
                &verify_result,
                &resolution_path_id);
        }
        else
        {
            status = SparkRequestModelResolveVerifierTokens(
                dispatch->speculative_draft_token_ids[request_index],
                dispatch->speculative_token_count,
                &verifier_token_ids[(uint64_t)request_index * lane_stride],
                verifier_token_count,
                &verify_result);
        }
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        SparkRequestApiTraceSpeculativeVerify(
            dispatch,
            verifier_token_ids,
            lane_stride,
            verifier_token_count,
            request_index,
            &verify_result);

        dispatch->speculative_accepted_token_counts[request_index] =
            verify_result.accepted_draft_token_count;
        dispatch->speculative_committed_token_counts[request_index] =
            verify_result.committed_token_count;
        dispatch->speculative_fallback_token_ids[request_index] =
            verify_result.fallback_token_id;
        dispatch->speculative_resolution_path_ids[request_index] =
            resolution_path_id;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkRequestApiCompleteDispatch(
    SparkRequestApi *api,
    const SparkRequestApiDispatch *dispatch)
{
    uint32_t request_index;
    SparkStatus status;

    status = SparkRequestApiValidate(api);
    if (status != SPARK_STATUS_OK || dispatch == 0 ||
        dispatch->abi_version != SPARK_REQUEST_API_ABI_VERSION ||
        dispatch->descriptor_bytes !=
            SPARK_REQUEST_API_DISPATCH_DESCRIPTOR_BYTES ||
        dispatch->accepted == 0u || dispatch->request_count == 0u)
    {
        return status == SPARK_STATUS_OK ? SPARK_STATUS_INVALID_ARGUMENT : status;
    }

    if (dispatch->kind == SPARK_REQUEST_API_DISPATCH_KIND_PREFILL)
    {
        status = SparkSchedulerComplete(
            api->scheduler,
            &dispatch->prefill_decision);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        for (request_index = 0u;
             request_index < dispatch->request_count;
             ++request_index)
        {
            SparkRequestApiSlot *slot;

            slot = SparkRequestApiFindSlotByHandle(
                api,
                dispatch->request_handles[request_index]);
            if (slot != 0 && request_index == 0u &&
                slot->state == SPARK_REQUEST_API_STATE_CANCELLED)
            {
                api->stale_prefill_completion_count += 1u;
                continue;
            }
            if (slot == 0 ||
                (request_index == 0u &&
                 slot->state != SPARK_REQUEST_API_STATE_RUNNING_PREFILL) ||
                (request_index != 0u &&
                 slot->state !=
                    SPARK_REQUEST_API_STATE_WAITING_PREFIX_COHORT))
            {
                return SPARK_STATUS_INVALID_ARGUMENT;
            }
            if (request_index != 0u &&
                dispatch->prefill_decision.cache_commit_token_count_after_step != 0u)
            {
                status = SparkPrefixCacheBindCommittedPrefixFromSequence(
                    api->scheduler->prefix_cache,
                    dispatch->sequence_ids[0],
                    slot->sequence_id,
                    dispatch->prefill_decision.cache_commit_token_count_after_step);
                if (status != SPARK_STATUS_OK)
                {
                    return status;
                }
            }
            SparkRequestApiFinishSlotAfterPrefill(
                api,
                slot,
                &dispatch->prefill_decision);
        }
        return SPARK_STATUS_OK;
    }
    if (dispatch->kind == SPARK_REQUEST_API_DISPATCH_KIND_PREFILL_BATCH)
    {
        status = SparkSchedulerCompletePrefillBatch(
            api->scheduler,
            &dispatch->prefill_batch_decision);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        for (request_index = 0u;
             request_index < dispatch->request_count;
             ++request_index)
        {
            SparkRequestApiSlot *slot;

            slot = SparkRequestApiFindSlotByHandle(
                api,
                dispatch->request_handles[request_index]);
            if (slot == 0 ||
                slot->state != SPARK_REQUEST_API_STATE_RUNNING_PREFILL)
            {
                return SPARK_STATUS_INVALID_ARGUMENT;
            }
            SparkRequestApiFinishSlotAfterPrefillBatchLane(
                api,
                slot,
                &dispatch->prefill_batch_decision.lanes[request_index]);
        }
        return SPARK_STATUS_OK;
    }
    if (dispatch->kind == SPARK_REQUEST_API_DISPATCH_KIND_DECODE_BATCH)
    {
        status = SparkSchedulerCompleteDecodeBatch(
            api->scheduler,
            &dispatch->decode_batch_decision);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        for (request_index = 0u;
             request_index < dispatch->request_count;
             ++request_index)
        {
            SparkRequestApiSlot *slot;
            uint32_t committed_token_count;

            slot = SparkRequestApiFindSlotByHandle(
                api,
                dispatch->request_handles[request_index]);
            if (slot == 0 || slot->state !=
                SPARK_REQUEST_API_STATE_RUNNING_DECODE)
            {
                return SPARK_STATUS_INVALID_ARGUMENT;
            }
            committed_token_count = 1u;
            if ((dispatch->flags &
                    SPARK_REQUEST_API_DISPATCH_FLAG_MTP_COMMIT) != 0u)
            {
                committed_token_count =
                    dispatch->decode_committed_token_counts[request_index];
                if (committed_token_count != 1u ||
                    committed_token_count >
                        SparkRequestApiSlotRemainingDecodeBudget(slot))
                {
                    return SPARK_STATUS_INVALID_ARGUMENT;
                }
            }
            SparkRequestApiFinishSlotAfterDecode(
                api,
                slot,
                committed_token_count);
        }
        return SPARK_STATUS_OK;
    }
    if (dispatch->kind ==
        SPARK_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH)
    {
        status = SparkSchedulerCompleteDecodeBatch(
            api->scheduler,
            &dispatch->decode_batch_decision);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        if (dispatch->speculative_token_count == 0u ||
            dispatch->speculative_token_count >
                SPARK_REQUEST_MODEL_MAX_SPECULATIVE_TOKENS)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        for (request_index = 0u;
             request_index < dispatch->request_count;
             ++request_index)
        {
            SparkRequestApiSlot *slot;
            uint32_t committed_token_count;
            uint32_t accepted_token_count;

            slot = SparkRequestApiFindSlotByHandle(
                api,
                dispatch->request_handles[request_index]);
            if (slot == 0 ||
                slot->state !=
                    SPARK_REQUEST_API_STATE_RUNNING_SPECULATIVE_VERIFY)
            {
                return SPARK_STATUS_INVALID_ARGUMENT;
            }
            committed_token_count =
                dispatch->speculative_committed_token_counts[request_index];
            accepted_token_count =
                dispatch->speculative_accepted_token_counts[request_index];
            status = SparkGlm52RequestApiFinishSlotAfterSpeculativeVerify(
                api,
                slot,
                dispatch->speculative_token_count,
                accepted_token_count,
                committed_token_count,
                dispatch->speculative_fallback_token_ids[request_index],
                dispatch->speculative_resolution_path_ids[request_index],
                (dispatch->flags &
                    SPARK_REQUEST_API_DISPATCH_FLAG_MTP_SPECULATIVE_VERIFY)
                        != 0u ? 1u : 0u);
            if (status != SPARK_STATUS_OK)
            {
                return status;
            }
        }
        return SPARK_STATUS_OK;
    }
    return SPARK_STATUS_INVALID_ARGUMENT;
}

static uint32_t SparkRequestApiDispatchLaneCount(
    const SparkRequestApiDispatch *dispatch)
{
    if (dispatch->kind == SPARK_REQUEST_API_DISPATCH_KIND_PREFILL)
    {
        return dispatch->prefill_decision.active_sequence_count;
    }
    if (dispatch->kind == SPARK_REQUEST_API_DISPATCH_KIND_PREFILL_BATCH)
    {
        return dispatch->prefill_batch_decision.active_sequence_count;
    }
    if (dispatch->kind == SPARK_REQUEST_API_DISPATCH_KIND_DECODE_BATCH ||
        dispatch->kind ==
            SPARK_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH)
    {
        return dispatch->decode_batch_decision.active_sequence_count;
    }
    return 0u;
}


SparkStatus SparkRequestApiDescribePrefillDispatch(
    const SparkRequestApiDispatch *dispatch,
    SparkRequestApiPrefillDispatchView *prefill_view)
{
    uint32_t lane_index;
    uint32_t lane_count;
    uint32_t prompt_token_stride;

    if (prefill_view == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    memset(prefill_view, 0, sizeof(*prefill_view));

    if (dispatch == 0 ||
        dispatch->abi_version != SPARK_REQUEST_API_ABI_VERSION ||
        dispatch->descriptor_bytes !=
            SPARK_REQUEST_API_DISPATCH_DESCRIPTOR_BYTES ||
        dispatch->accepted == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if (dispatch->kind == SPARK_REQUEST_API_DISPATCH_KIND_PREFILL)
    {
        const SparkSchedulerDecision *decision;

        decision = &dispatch->prefill_decision;
        if (decision->abi_version != SPARK_SCHEDULER_ABI_VERSION ||
            decision->descriptor_bytes !=
                SPARK_SCHEDULER_DECISION_DESCRIPTOR_BYTES ||
            decision->accepted == 0u ||
            decision->active_sequence_count == 0u ||
            decision->active_sequence_count > 1u ||
            decision->scheduled_prompt_token_count == 0u ||
            decision->prompt_token_ids == 0)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }

        prefill_view->abi_version = SPARK_REQUEST_API_ABI_VERSION;
        prefill_view->descriptor_bytes =
            SPARK_REQUEST_API_PREFILL_DISPATCH_VIEW_DESCRIPTOR_BYTES;
        prefill_view->kind = dispatch->kind;
        prefill_view->active_sequence_count = decision->active_sequence_count;
        prefill_view->lane_count = 1u;
        prefill_view->prompt_token_offset =
            decision->scheduled_prompt_token_offset;
        prefill_view->prompt_token_count =
            decision->scheduled_prompt_token_count;
        prefill_view->prompt_token_stride =
            decision->scheduled_prompt_token_count;
        prefill_view->lanes[0u].request_index = 0u;
        prefill_view->lanes[0u].prompt_token_offset =
            decision->scheduled_prompt_token_offset;
        prefill_view->lanes[0u].prompt_token_count =
            decision->scheduled_prompt_token_count;
        prefill_view->lanes[0u].request_slot_index =
            dispatch->request_slot_indices[0u];
        prefill_view->lanes[0u].request_id = dispatch->request_ids[0u];
        prefill_view->lanes[0u].sequence_id = dispatch->sequence_ids[0u];
        prefill_view->lanes[0u].request_handle = dispatch->request_handles[0u];
        prefill_view->lanes[0u].prompt_token_ids = decision->prompt_token_ids;
        return SPARK_STATUS_OK;
    }

    if (dispatch->kind == SPARK_REQUEST_API_DISPATCH_KIND_PREFILL_BATCH)
    {
        const SparkSchedulerPrefillBatchDecision *batch_decision;

        batch_decision = &dispatch->prefill_batch_decision;
        lane_count = batch_decision->active_sequence_count;
        prompt_token_stride = batch_decision->maximum_scheduled_prompt_token_count;
        if (batch_decision->abi_version != SPARK_SCHEDULER_ABI_VERSION ||
            batch_decision->descriptor_bytes !=
                SPARK_SCHEDULER_PREFILL_BATCH_DECISION_DESCRIPTOR_BYTES ||
            batch_decision->accepted == 0u ||
            lane_count == 0u ||
            lane_count > SPARK_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT ||
            prompt_token_stride == 0u)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }

        prefill_view->abi_version = SPARK_REQUEST_API_ABI_VERSION;
        prefill_view->descriptor_bytes =
            SPARK_REQUEST_API_PREFILL_DISPATCH_VIEW_DESCRIPTOR_BYTES;
        prefill_view->kind = dispatch->kind;
        prefill_view->active_sequence_count = lane_count;
        prefill_view->lane_count = lane_count;
        prefill_view->prompt_token_stride = prompt_token_stride;
        for (lane_index = 0u; lane_index < lane_count; ++lane_index)
        {
            const SparkSchedulerPrefillBatchLane *lane;

            lane = &batch_decision->lanes[lane_index];
            if (lane->abi_version != SPARK_SCHEDULER_ABI_VERSION ||
                lane->descriptor_bytes !=
                    SPARK_SCHEDULER_PREFILL_BATCH_LANE_DESCRIPTOR_BYTES ||
                lane->active_sequence_count == 0u ||
                lane->scheduled_prompt_token_count == 0u ||
                lane->scheduled_prompt_token_count > prompt_token_stride ||
                lane->prompt_token_ids == 0)
            {
                return SPARK_STATUS_INVALID_ARGUMENT;
            }
            if (lane_index == 0u)
            {
                prefill_view->prompt_token_offset =
                    lane->scheduled_prompt_token_offset;
            }
            else if (prefill_view->prompt_token_offset !=
                lane->scheduled_prompt_token_offset)
            {
                prefill_view->prompt_token_offset = 0u;
            }
            prefill_view->prompt_token_count = SparkRequestApiMaximumU32(
                prefill_view->prompt_token_count,
                lane->scheduled_prompt_token_count);
            prefill_view->lanes[lane_index].request_index = lane->request_index;
            prefill_view->lanes[lane_index].prompt_token_offset =
                lane->scheduled_prompt_token_offset;
            prefill_view->lanes[lane_index].prompt_token_count =
                lane->scheduled_prompt_token_count;
            prefill_view->lanes[lane_index].request_slot_index =
                dispatch->request_slot_indices[lane_index];
            prefill_view->lanes[lane_index].request_id =
                dispatch->request_ids[lane_index];
            prefill_view->lanes[lane_index].sequence_id =
                dispatch->sequence_ids[lane_index];
            prefill_view->lanes[lane_index].request_handle =
                dispatch->request_handles[lane_index];
            prefill_view->lanes[lane_index].prompt_token_ids =
                lane->prompt_token_ids;
        }
        return SPARK_STATUS_OK;
    }

    return SPARK_STATUS_INVALID_ARGUMENT;
}

SparkStatus SparkRequestApiCopyPrefillDispatchTokenIds(
    const SparkRequestApiDispatch *dispatch,
    uint32_t *destination_token_ids,
    uint32_t destination_token_stride,
    uint32_t destination_lane_capacity)
{
    SparkRequestApiPrefillDispatchView prefill_view;
    uint32_t lane_index;
    SparkStatus status;

    if (destination_token_ids == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    status = SparkRequestApiDescribePrefillDispatch(
        dispatch,
        &prefill_view);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (destination_lane_capacity < prefill_view.lane_count ||
        destination_token_stride < prefill_view.prompt_token_stride)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    for (lane_index = 0u; lane_index < prefill_view.lane_count; ++lane_index)
    {
        const SparkRequestApiPrefillDispatchLaneView *lane;
        uint32_t token_index;
        uint32_t *destination_lane;

        lane = &prefill_view.lanes[lane_index];
        destination_lane =
            &destination_token_ids[(uint64_t)lane_index * destination_token_stride];
        for (token_index = 0u;
             token_index < lane->prompt_token_count;
             ++token_index)
        {
            destination_lane[token_index] =
                lane->prompt_token_ids[lane->prompt_token_offset + token_index];
        }
        for (token_index = lane->prompt_token_count;
             token_index < destination_token_stride;
             ++token_index)
        {
            destination_lane[token_index] = 0u;
        }
    }

    return SPARK_STATUS_OK;
}

SparkStatus SparkRequestApiDescribeDecodeDispatch(
    SparkRequestApi *api,
    const SparkRequestApiDispatch *dispatch,
    SparkRequestApiDecodeDispatchView *decode_view)
{
    uint32_t lane_index;
    SparkStatus status;

    if (decode_view == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    memset(decode_view, 0, sizeof(*decode_view));

    status = SparkRequestApiValidate(api);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    if (dispatch == 0 ||
        dispatch->abi_version != SPARK_REQUEST_API_ABI_VERSION ||
        dispatch->descriptor_bytes !=
            SPARK_REQUEST_API_DISPATCH_DESCRIPTOR_BYTES ||
        dispatch->accepted == 0u ||
        (dispatch->kind != SPARK_REQUEST_API_DISPATCH_KIND_DECODE_BATCH &&
         dispatch->kind !=
            SPARK_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH) ||
        dispatch->request_count == 0u ||
        dispatch->request_count > SPARK_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    decode_view->abi_version = SPARK_REQUEST_API_ABI_VERSION;
    decode_view->descriptor_bytes =
        SPARK_REQUEST_API_DECODE_DISPATCH_VIEW_DESCRIPTOR_BYTES;
    decode_view->kind = dispatch->kind;
    decode_view->active_sequence_count = dispatch->request_count;
    decode_view->lane_count = dispatch->request_count;
    decode_view->speculative_token_count = dispatch->speculative_token_count;
    for (lane_index = 0u;
         lane_index < dispatch->request_count;
         ++lane_index)
    {
        SparkRequestApiSlot *slot;
        SparkRequestApiDecodeDispatchLaneView *lane;
        uint64_t sequence_position;

        slot = SparkRequestApiFindSlotByHandle(
            api,
            dispatch->request_handles[lane_index]);
        if (slot == 0 ||
            dispatch->request_slot_indices[lane_index] !=
                SparkRequestApiSlotIndex(api, slot) ||
            slot->computed_prompt_token_count == 0u ||
            (slot->state != SPARK_REQUEST_API_STATE_RUNNING_DECODE &&
             slot->state !=
                SPARK_REQUEST_API_STATE_RUNNING_SPECULATIVE_VERIFY))
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }

        sequence_position =
            (uint64_t)slot->computed_prompt_token_count +
            (uint64_t)slot->completed_decode_token_count - 1u;
        if (sequence_position > UINT32_MAX - 1u)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }

        lane = &decode_view->lanes[lane_index];
        lane->request_index = lane_index;
        lane->sequence_position = (uint32_t)sequence_position;
        lane->context_token_count = (uint32_t)(sequence_position + 1u);
        lane->request_slot_index = SparkRequestApiSlotIndex(api, slot);
        if (lane->request_slot_index == SPARK_REQUEST_API_NO_SLOT)
        {
            return SPARK_STATUS_INTERNAL_ERROR;
        }
        lane->request_id = slot->request_id;
        lane->sequence_id = slot->sequence_id;
        lane->request_handle = slot->handle;
        if (slot->mtp_resolution_proposed_token_count == 0u)
        {
            if (slot->mtp_resolution_base_position != 0u ||
                slot->mtp_resolution_accepted_token_count != 0u ||
                slot->mtp_resolution_committed_token_count != 0u ||
                slot->mtp_resolution_path_id !=
                    SPARK_MODEL_MTP_TREE_RESOLUTION_NONE)
            {
                return SPARK_STATUS_INTERNAL_ERROR;
            }
        }
        else if (slot->mtp_resolution_proposed_token_count >
                SPARK_REQUEST_API_MTP_MAX_DRAFT_TOKEN_COUNT ||
            slot->mtp_resolution_committed_token_count !=
                slot->mtp_resolution_accepted_token_count + 1u ||
            slot->mtp_resolution_base_position >
                UINT64_MAX - slot->mtp_resolution_committed_token_count ||
            slot->mtp_resolution_base_position +
                slot->mtp_resolution_committed_token_count != sequence_position ||
            SparkMtpTreeResolutionIsValid(
                slot->mtp_resolution_proposed_token_count,
                slot->mtp_resolution_accepted_token_count,
                slot->mtp_resolution_path_id) == 0u)
        {
            return SPARK_STATUS_INTERNAL_ERROR;
        }
        lane->mtp_resolution_base_position =
            slot->mtp_resolution_base_position;
        lane->mtp_resolution_proposed_token_count =
            slot->mtp_resolution_proposed_token_count;
        lane->mtp_resolution_accepted_token_count =
            slot->mtp_resolution_accepted_token_count;
        lane->mtp_resolution_committed_token_count =
            slot->mtp_resolution_committed_token_count;
        lane->mtp_resolution_path_id = slot->mtp_resolution_path_id;
    }

    return SPARK_STATUS_OK;
}

SparkStatus SparkRequestApiBuildDispatchKvBlockTables(
    SparkRequestApi *api,
    const SparkRequestApiDispatch *dispatch,
    uint32_t *physical_block_indices,
    uint32_t lane_stride,
    uint32_t lane_capacity,
    uint32_t *lane_physical_block_counts,
    uint32_t lane_count_capacity)
{
    uint32_t lane_index;
    SparkStatus status;

    status = SparkRequestApiValidate(api);
    if (status != SPARK_STATUS_OK || dispatch == 0 ||
        dispatch->abi_version != SPARK_REQUEST_API_ABI_VERSION ||
        dispatch->descriptor_bytes !=
            SPARK_REQUEST_API_DISPATCH_DESCRIPTOR_BYTES ||
        dispatch->accepted == 0u ||
        physical_block_indices == 0 ||
        lane_physical_block_counts == 0 ||
        lane_capacity == 0u ||
        lane_stride < lane_capacity)
    {
        return status == SPARK_STATUS_OK ? SPARK_STATUS_INVALID_ARGUMENT : status;
    }

    if (dispatch->kind == SPARK_REQUEST_API_DISPATCH_KIND_PREFILL)
    {
        if (lane_count_capacity < 1u)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        for (lane_index = 0u; lane_index < lane_count_capacity; ++lane_index)
        {
            lane_physical_block_counts[lane_index] = 0u;
        }
        return SparkSchedulerBuildKvBlockTable(
            api->scheduler,
            &dispatch->prefill_decision,
            physical_block_indices,
            lane_capacity,
            &lane_physical_block_counts[0]);
    }

    if (dispatch->kind == SPARK_REQUEST_API_DISPATCH_KIND_PREFILL_BATCH)
    {
        return SparkSchedulerBuildPrefillBatchKvBlockTables(
            api->scheduler,
            &dispatch->prefill_batch_decision,
            physical_block_indices,
            lane_stride,
            lane_capacity,
            lane_physical_block_counts,
            lane_count_capacity);
    }

    if (dispatch->kind == SPARK_REQUEST_API_DISPATCH_KIND_DECODE_BATCH ||
        dispatch->kind ==
            SPARK_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH)
    {
        if (lane_count_capacity < dispatch->request_count)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        for (lane_index = 0u; lane_index < lane_count_capacity; ++lane_index)
        {
            lane_physical_block_counts[lane_index] = 0u;
        }
        for (lane_index = 0u; lane_index < dispatch->request_count; ++lane_index)
        {
            SparkRequestApiSlot *slot;
            uint32_t required_token_count;

            slot = SparkRequestApiFindSlotByHandle(
                api,
                dispatch->request_handles[lane_index]);
            if (slot == 0 ||
                (slot->state != SPARK_REQUEST_API_STATE_RUNNING_DECODE &&
                 slot->state !=
                    SPARK_REQUEST_API_STATE_RUNNING_SPECULATIVE_VERIFY) ||
                slot->computed_prompt_token_count == 0u)
            {
                return SPARK_STATUS_INVALID_ARGUMENT;
            }
            status = SparkRequestApiEnsureDecodeSlotKvCapacity(
                api,
                slot,
                dispatch->kind ==
                    SPARK_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH
                        ? (dispatch->flags &
                            SPARK_REQUEST_API_DISPATCH_FLAG_MTP_TREE_VERIFY) != 0u
                            ? SPARK_MODEL_MTP_TREE_CONTEXT_EXTENSION
                            : dispatch->speculative_token_count
                        : (dispatch->flags &
                            SPARK_REQUEST_API_DISPATCH_FLAG_MTP_COMMIT) != 0u
                            ? dispatch->mtp_draft_token_budget : 0u,
                &required_token_count);
            if (status != SPARK_STATUS_OK)
            {
                return status;
            }
            status = SparkPrefixCacheBuildPhysicalBlockTable(
                api->scheduler->prefix_cache,
                slot->sequence_id,
                required_token_count,
                &physical_block_indices[(uint64_t)lane_index * lane_stride],
                lane_capacity,
                &lane_physical_block_counts[lane_index]);
            if (status != SPARK_STATUS_OK)
            {
                return status;
            }
        }
        return SPARK_STATUS_OK;
    }

    return SPARK_STATUS_INVALID_ARGUMENT;
}

SparkStatus SparkRequestApiBuildDispatchKvBlockTableView(
    SparkRequestApi *api,
    const SparkRequestApiDispatch *dispatch,
    uint32_t *host_physical_block_indices,
    const uint32_t *execution_physical_block_indices,
    uint32_t lane_stride,
    uint32_t lane_capacity,
    uint32_t *lane_physical_block_counts,
    uint32_t lane_count_capacity,
    SparkKvBlockTableView *block_table_view)
{
    uint32_t lane_count;
    SparkStatus status;

    if (block_table_view == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    memset(block_table_view, 0, sizeof(*block_table_view));

    status = SparkRequestApiValidate(api);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    if (dispatch == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    lane_count = SparkRequestApiDispatchLaneCount(dispatch);
    if (dispatch->abi_version != SPARK_REQUEST_API_ABI_VERSION ||
        dispatch->descriptor_bytes !=
            SPARK_REQUEST_API_DISPATCH_DESCRIPTOR_BYTES ||
        dispatch->accepted == 0u ||
        lane_count == 0u ||
        lane_count > lane_count_capacity ||
        host_physical_block_indices == 0 ||
        lane_physical_block_counts == 0 ||
        lane_capacity == 0u ||
        lane_stride < lane_capacity)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    status = SparkRequestApiBuildDispatchKvBlockTables(
        api,
        dispatch,
        host_physical_block_indices,
        lane_stride,
        lane_capacity,
        lane_physical_block_counts,
        lane_count_capacity);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    block_table_view->abi_version = SPARK_KV_CACHE_ABI_VERSION;
    block_table_view->descriptor_bytes =
        SPARK_KV_BLOCK_TABLE_VIEW_DESCRIPTOR_BYTES;
    block_table_view->block_token_count = api->scheduler->prefix_cache_block_tokens;
    block_table_view->lane_count = lane_count;
    block_table_view->lane_stride = lane_stride;
    block_table_view->lane_capacity = lane_capacity;
    block_table_view->physical_block_indices = execution_physical_block_indices != 0 ?
        execution_physical_block_indices : host_physical_block_indices;
    block_table_view->lane_physical_block_counts = lane_physical_block_counts;
    block_table_view->host_physical_block_indices = host_physical_block_indices;
    block_table_view->host_lane_physical_block_counts = lane_physical_block_counts;

    return SPARK_STATUS_OK;
}


static uint32_t SparkRequestApiRetryDecodeTokenCount(
    const SparkRequestApiDispatch *dispatch)
{
    if (dispatch != 0 &&
        dispatch->kind ==
            SPARK_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH)
    {
        return dispatch->speculative_token_count;
    }
    return 1u;
}

static SparkStatus SparkGlm52RequestApiValidateRetryDecodeCounters(
    SparkRequestApi *api,
    const SparkRequestApiDispatch *dispatch)
{
    uint32_t retry_token_count;

    if (api == 0 || dispatch == 0 || dispatch->accepted == 0u ||
        (dispatch->kind !=
            SPARK_REQUEST_API_DISPATCH_KIND_DECODE_BATCH &&
         dispatch->kind !=
            SPARK_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH) ||
        dispatch->request_count == 0u ||
        api->running_request_count < dispatch->request_count ||
        api->scheduled_decode_dispatch_count == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    retry_token_count = SparkRequestApiRetryDecodeTokenCount(dispatch);
    if (retry_token_count == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (dispatch->kind ==
        SPARK_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH)
    {
        if ((dispatch->flags &
                SPARK_REQUEST_API_DISPATCH_FLAG_MTP_SPECULATIVE_VERIFY) != 0u)
        {
            if (api->mtp_verify_dispatch_count == 0u)
            {
                return SPARK_STATUS_INVALID_ARGUMENT;
            }
        }
        else if (api->dspark_verify_dispatch_count == 0u)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkRequestApiValidateRetryDecodeSlots(
    SparkRequestApi *api,
    const SparkRequestApiDispatch *dispatch)
{
    uint32_t request_index;
    uint32_t retry_token_count;

    retry_token_count = SparkRequestApiRetryDecodeTokenCount(dispatch);
    for (request_index = 0u;
         request_index < dispatch->request_count;
         ++request_index)
    {
        SparkRequestApiSlot *slot;
        uint32_t expected_state;

        slot = SparkRequestApiFindSlotByHandle(
            api,
            dispatch->request_handles[request_index]);
        expected_state =
            dispatch->kind ==
                SPARK_REQUEST_API_DISPATCH_KIND_DECODE_BATCH
            ? SPARK_REQUEST_API_STATE_RUNNING_DECODE
            : SPARK_REQUEST_API_STATE_RUNNING_SPECULATIVE_VERIFY;
        if (slot == 0 || slot->state != expected_state ||
            slot->scheduled_decode_token_count < retry_token_count)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkRequestApiValidateRetryDecodeDispatch(
    SparkRequestApi *api,
    const SparkRequestApiDispatch *dispatch)
{
    SparkStatus status;

    status = SparkGlm52RequestApiValidateRetryDecodeCounters(api,dispatch);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    return SparkRequestApiValidateRetryDecodeSlots(api,dispatch);
}

static void SparkRequestApiRestoreRetriedDecodeSlots(
    SparkRequestApi *api,
    const SparkRequestApiDispatch *dispatch)
{
    uint32_t request_index;
    uint32_t retry_token_count;

    retry_token_count = SparkRequestApiRetryDecodeTokenCount(dispatch);
    for (request_index = 0u;
         request_index < dispatch->request_count;
         ++request_index)
    {
        SparkRequestApiSlot *slot;

        slot = SparkRequestApiFindSlotByHandle(
            api,
            dispatch->request_handles[request_index]);
        slot->scheduled_decode_token_count -= retry_token_count;
        slot->state =
            dispatch->kind ==
                SPARK_REQUEST_API_DISPATCH_KIND_DECODE_BATCH
            ? SPARK_REQUEST_API_STATE_READY_DECODE
            : SPARK_REQUEST_API_STATE_READY_SPECULATIVE_VERIFY;
    }
}


SparkStatus SparkRequestApiRetryDecodeDispatch(
    SparkRequestApi *api,
    const SparkRequestApiDispatch *dispatch)
{
    SparkStatus status;

    status = SparkRequestApiValidate(api);
    if (status == SPARK_STATUS_OK)
    {
        status = SparkRequestApiValidateRetryDecodeDispatch(
            api,
            dispatch);
    }
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkSchedulerCancelDecodeBatch(
        api->scheduler,
        &dispatch->decode_batch_decision);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    SparkRequestApiRestoreRetriedDecodeSlots(api,dispatch);
    SparkRequestModelRestoreRetriedDecodeCounters(api,dispatch);
    return SPARK_STATUS_OK;
}

SparkStatus SparkRequestApiCancelDispatch(
    SparkRequestApi *api,
    const SparkRequestApiDispatch *dispatch)
{
    uint32_t request_index;
    SparkStatus status;

    status = SparkRequestApiValidate(api);
    if (status != SPARK_STATUS_OK || dispatch == 0 || dispatch->accepted == 0u)
    {
        return status == SPARK_STATUS_OK ? SPARK_STATUS_INVALID_ARGUMENT : status;
    }
    if (dispatch->kind == SPARK_REQUEST_API_DISPATCH_KIND_PREFILL)
    {
        status = SparkSchedulerCancel(
            api->scheduler,
            &dispatch->prefill_decision);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        for (request_index = 0u;
             request_index < dispatch->request_count;
             ++request_index)
        {
            SparkRequestApiSlot *slot;

            slot = SparkRequestApiFindSlotByHandle(
                api,
                dispatch->request_handles[request_index]);
            if (slot == 0 ||
                (request_index == 0u &&
                 slot->state != SPARK_REQUEST_API_STATE_RUNNING_PREFILL) ||
                (request_index != 0u &&
                 slot->state !=
                    SPARK_REQUEST_API_STATE_WAITING_PREFIX_COHORT))
            {
                return SPARK_STATUS_INVALID_ARGUMENT;
            }
            api->running_request_count -= 1u;
            slot->state = SPARK_REQUEST_API_STATE_CANCELLED;
            api->cancelled_request_count += 1u;
            status = SparkRequestModelReleaseSlotSequence(api, slot);
            if (status != SPARK_STATUS_OK)
            {
                return status;
            }
        }
        return SPARK_STATUS_OK;
    }
    if (dispatch->kind == SPARK_REQUEST_API_DISPATCH_KIND_PREFILL_BATCH)
    {
        status = SparkSchedulerCancelPrefillBatch(
            api->scheduler,
            &dispatch->prefill_batch_decision);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        for (request_index = 0u;
             request_index < dispatch->request_count;
             ++request_index)
        {
            SparkRequestApiSlot *slot;

            slot = SparkRequestApiFindSlotByHandle(
                api,
                dispatch->request_handles[request_index]);
            if (slot == 0 ||
                slot->state != SPARK_REQUEST_API_STATE_RUNNING_PREFILL)
            {
                return SPARK_STATUS_INVALID_ARGUMENT;
            }
            api->running_request_count -= 1u;
            slot->state = SPARK_REQUEST_API_STATE_CANCELLED;
            api->cancelled_request_count += 1u;
            status = SparkRequestModelReleaseSlotSequence(api, slot);
            if (status != SPARK_STATUS_OK)
            {
                return status;
            }
        }
        return SPARK_STATUS_OK;
    }
    if (dispatch->kind == SPARK_REQUEST_API_DISPATCH_KIND_DECODE_BATCH)
    {
        status = SparkSchedulerCancelDecodeBatch(
            api->scheduler,
            &dispatch->decode_batch_decision);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        for (request_index = 0u;
             request_index < dispatch->request_count;
             ++request_index)
        {
            SparkRequestApiSlot *slot;

            slot = SparkRequestApiFindSlotByHandle(
                api,
                dispatch->request_handles[request_index]);
            if (slot == 0 || slot->state !=
                SPARK_REQUEST_API_STATE_RUNNING_DECODE)
            {
                return SPARK_STATUS_INVALID_ARGUMENT;
            }
            api->running_request_count -= 1u;
            slot->state = SPARK_REQUEST_API_STATE_CANCELLED;
            api->cancelled_request_count += 1u;
            status = SparkRequestModelReleaseSlotSequence(api, slot);
            if (status != SPARK_STATUS_OK)
            {
                return status;
            }
        }
        return SPARK_STATUS_OK;
    }
    if (dispatch->kind ==
        SPARK_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH)
    {
        status = SparkSchedulerCancelDecodeBatch(
            api->scheduler,
            &dispatch->decode_batch_decision);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        for (request_index = 0u;
             request_index < dispatch->request_count;
             ++request_index)
        {
            SparkRequestApiSlot *slot;

            slot = SparkRequestApiFindSlotByHandle(
                api,
                dispatch->request_handles[request_index]);
            if (slot == 0 ||
                slot->state !=
                    SPARK_REQUEST_API_STATE_RUNNING_SPECULATIVE_VERIFY)
            {
                return SPARK_STATUS_INVALID_ARGUMENT;
            }
            if (SparkRequestModelDsparkSpeculationIsEnabled(api))
            {
                (void)SparkRequestModelCancelSequence(
                    api,
                    slot->sequence_id);
            }
            api->running_request_count -= 1u;
            slot->state = SPARK_REQUEST_API_STATE_CANCELLED;
            api->cancelled_request_count += 1u;
            status = SparkRequestModelReleaseSlotSequence(api, slot);
            if (status != SPARK_STATUS_OK)
            {
                return status;
            }
        }
        return SPARK_STATUS_OK;
    }
    return SPARK_STATUS_INVALID_ARGUMENT;
}

SparkStatus SparkRequestApiGetRequestCacheState(
    SparkRequestApi *api,
    SparkRequestApiHandle handle,
    SparkRequestApiCacheState *cache_state)
{
    SparkRequestApiSlot *slot;
    uint32_t physical_block_count;
    uint32_t resident_block_count;
    uint32_t nonresident_block_count;
    SparkStatus status;

    status = SparkRequestApiValidate(api);
    if (status != SPARK_STATUS_OK || cache_state == 0)
    {
        return status == SPARK_STATUS_OK ? SPARK_STATUS_INVALID_ARGUMENT : status;
    }
    slot = SparkRequestApiFindSlotByHandle(api, handle);
    if (slot == 0)
    {
        return SPARK_STATUS_NOT_FOUND;
    }

    memset(cache_state, 0, sizeof(*cache_state));
    cache_state->abi_version = SPARK_REQUEST_API_ABI_VERSION;
    cache_state->descriptor_bytes =
        SPARK_REQUEST_API_CACHE_STATE_DESCRIPTOR_BYTES;
    cache_state->state = slot->state;
    cache_state->computed_prompt_token_count =
        slot->computed_prompt_token_count;
    cache_state->last_committed_prefix_token_count =
        slot->last_committed_prefix_token_count;
    cache_state->request_id = slot->request_id;
    cache_state->sequence_id = slot->sequence_id;
    cache_state->last_committed_prefix_hash =
        slot->last_committed_prefix_hash;

    if (slot->computed_prompt_token_count != 0u &&
        api->scheduler != 0 &&
        api->scheduler->prefix_cache != 0)
    {
        status = SparkPrefixCacheProbeSequenceResidency(
            api->scheduler->prefix_cache,
            slot->sequence_id,
            slot->computed_prompt_token_count,
            &physical_block_count,
            &resident_block_count,
            &nonresident_block_count);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        (void)resident_block_count;
        (void)nonresident_block_count;
        cache_state->physical_block_count = physical_block_count;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkRequestApiFinishRequestGeneration(
    SparkRequestApi *api,
    SparkRequestApiHandle handle)
{
    SparkRequestApiSlot *slot;
    SparkStatus status;

    status = SparkRequestApiValidate(api);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    slot = SparkRequestApiFindSlotByHandle(api, handle);
    if (slot == 0)
    {
        return SPARK_STATUS_NOT_FOUND;
    }

    if (slot->state == SPARK_REQUEST_API_STATE_COMPLETED)
    {
        return SPARK_STATUS_OK;
    }
    if (slot->state == SPARK_REQUEST_API_STATE_CANCELLED)
    {
        return SPARK_STATUS_NOT_FOUND;
    }
    if (slot->state == SPARK_REQUEST_API_STATE_RUNNING_PREFILL ||
        slot->state == SPARK_REQUEST_API_STATE_RUNNING_DECODE ||
        slot->state ==
            SPARK_REQUEST_API_STATE_RUNNING_SPECULATIVE_VERIFY ||
        slot->state == SPARK_REQUEST_API_STATE_WAITING_PREFIX_COHORT)
    {
        return SPARK_STATUS_BUSY;
    }

    if (slot->state == SPARK_REQUEST_API_STATE_QUEUED_PREFILL)
    {
        if (api->queued_request_count == 0u)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        api->queued_request_count -= 1u;
    }
    else if (slot->state != SPARK_REQUEST_API_STATE_READY_DECODE &&
             slot->state !=
                SPARK_REQUEST_API_STATE_READY_SPECULATIVE_VERIFY)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if (slot->state == SPARK_REQUEST_API_STATE_READY_SPECULATIVE_VERIFY &&
        SparkRequestModelDsparkSpeculationIsEnabled(api))
    {
        status = SparkRequestModelCancelSequence(
            api,
            slot->sequence_id);
        if (status != SPARK_STATUS_OK && status != SPARK_STATUS_NOT_FOUND)
        {
            return status;
        }
    }

    slot->remaining_thinking_token_budget = 0u;
    slot->remaining_output_token_budget = 0u;
    slot->state = SPARK_REQUEST_API_STATE_COMPLETED;
    api->completed_request_count += 1u;
    return SPARK_STATUS_OK;
}

SparkStatus SparkRequestApiCancelRequest(
    SparkRequestApi *api,
    SparkRequestApiHandle handle)
{
    SparkRequestApiSlot *slot;
    SparkStatus status;

    status = SparkRequestApiValidate(api);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    slot = SparkRequestApiFindSlotByHandle(api, handle);
    if (slot == 0)
    {
        return SPARK_STATUS_NOT_FOUND;
    }
    if (slot->state == SPARK_REQUEST_API_STATE_CANCELLED)
    {
        return SPARK_STATUS_OK;
    }
    if (slot->state == SPARK_REQUEST_API_STATE_RUNNING_PREFILL ||
        slot->state == SPARK_REQUEST_API_STATE_RUNNING_DECODE)
    {
        return SPARK_STATUS_BUSY;
    }
    if (slot->state == SPARK_REQUEST_API_STATE_QUEUED_PREFILL)
    {
        api->queued_request_count -= 1u;
    }
    if (slot->state == SPARK_REQUEST_API_STATE_COMPLETED)
    {
        api->completed_request_count -= 1u;
    }
    slot->state = SPARK_REQUEST_API_STATE_CANCELLED;
    api->cancelled_request_count += 1u;
    return SparkRequestModelReleaseSlotSequence(api, slot);
}

SparkStatus SparkRequestApiReleaseCompletedRequest(
    SparkRequestApi *api,
    SparkRequestApiHandle handle)
{
    SparkRequestApiSlot *slot;
    SparkStatus status;

    status = SparkRequestApiValidate(api);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    slot = SparkRequestApiFindSlotByHandle(api, handle);
    if (slot == 0)
    {
        return SPARK_STATUS_NOT_FOUND;
    }
    if (slot->state != SPARK_REQUEST_API_STATE_COMPLETED &&
        slot->state != SPARK_REQUEST_API_STATE_CANCELLED)
    {
        return SPARK_STATUS_BUSY;
    }
    status = SparkRequestModelReleaseSlotSequence(api, slot);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    SparkRequestApiRemoveSlotHash(api, slot);
    {
        uint32_t released_slot_index;

        released_slot_index = SparkRequestApiSlotIndex(api, slot);
        if (released_slot_index == SPARK_REQUEST_API_NO_SLOT)
        {
            return SPARK_STATUS_INTERNAL_ERROR;
        }
        SparkRequestApiInitializeSlot(slot);
        slot->free_slot_next = api->free_slot_head;
        api->free_slot_head = released_slot_index;
    }
    return SPARK_STATUS_OK;
}

uint32_t SparkRequestApiAssignDraftBudgets(
    SparkRequestApi *api,
    uint32_t firing_row_cap,
    struct SparkRowAllocatorSlotInput *scratch_inputs,
    uint32_t *scratch_budgets)
{
    uint32_t slot_index,eligible_count,total,apply_index;
    if (api == 0 || scratch_inputs == 0 || scratch_budgets == 0 || firing_row_cap == 0u)
        return 0u;
    eligible_count = 0u;
    for (slot_index = 0u; slot_index < api->request_capacity; ++slot_index)
    {
        SparkRequestApiSlot *slot = &api->request_slots[slot_index];
        if ((slot->state != SPARK_REQUEST_API_STATE_READY_DECODE &&
             slot->state != SPARK_REQUEST_API_STATE_READY_SPECULATIVE_VERIFY) ||
            (slot->remaining_thinking_token_budget == 0u &&
             slot->remaining_output_token_budget == 0u))
            continue;
        scratch_inputs[eligible_count].commit_ema_milli = slot->mtp_commit_ema_milli;
        scratch_inputs[eligible_count].maximum_draft_depth =
            (slot->mtp_next_draft_token_budget == 0u && slot->mtp_probe_countdown != 0u) ?
            0u : SPARK_MODEL_MTP_TREE_CANDIDATE_COUNT;
        scratch_inputs[eligible_count].probe = 0u;
        eligible_count += 1u;
    }
    total = SparkRowAllocatorAssign(scratch_inputs, eligible_count, firing_row_cap, 1000u, scratch_budgets);
    apply_index = 0u;
    for (slot_index = 0u; slot_index < api->request_capacity && apply_index < eligible_count; ++slot_index)
    {
        SparkRequestApiSlot *slot = &api->request_slots[slot_index];
        if ((slot->state != SPARK_REQUEST_API_STATE_READY_DECODE &&
             slot->state != SPARK_REQUEST_API_STATE_READY_SPECULATIVE_VERIFY) ||
            (slot->remaining_thinking_token_budget == 0u &&
             slot->remaining_output_token_budget == 0u))
            continue;
        slot->mtp_next_draft_token_budget = scratch_budgets[apply_index];
        apply_index += 1u;
    }
    return total;
}
