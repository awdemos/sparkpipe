#include <assert.h>
#include <stdint.h>
#include <string.h>

#include "sparkpipe/spark_service.h"

#define SPARK_TEST_SERVICE_REQUEST_SLOT_COUNT 4u
#define SPARK_TEST_SERVICE_REQUEST_RECORD_COUNT 4u
#define SPARK_TEST_SERVICE_REQUEST_TOKEN_STRIDE 128u
#define SPARK_TEST_SERVICE_PREFILL_TOKEN_STRIDE 64u
#define SPARK_TEST_SERVICE_PROMPT_TOKEN_COUNT 64u
#define SPARK_TEST_SERVICE_KV_BLOCK_COUNT \
    SPARK_SCHEDULER_KV_BLOCK_TABLE_CAPACITY
#define SPARK_TEST_SERVICE_PREFIX_ENTRY_COUNT \
    SPARK_SCHEDULER_KV_BLOCK_TABLE_CAPACITY
#define SPARK_TEST_SERVICE_PREFIX_BINDING_COUNT \
    (SPARK_SCHEDULER_KV_BLOCK_TABLE_CAPACITY + 8u)
#define SPARK_TEST_SERVICE_EVENT_CAPACITY 16384u
#define SPARK_TEST_SERVICE_CLIENT_CAPACITY 4u
#define SPARK_TEST_SERVICE_REQUEST_MAP_CAPACITY 8u
#define SPARK_TEST_SERVICE_LANE_CAPACITY \
    SPARK_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT

typedef struct SparkTestServiceFixture
{
    SparkKvCacheArena kv_arena;
    SparkKvCacheBlock kv_blocks[SPARK_TEST_SERVICE_KV_BLOCK_COUNT];
    SparkPrefixCache prefix_cache;
    SparkPrefixCacheEntry prefix_entries[
        SPARK_TEST_SERVICE_PREFIX_ENTRY_COUNT];
    SparkPrefixCacheSequenceBinding prefix_bindings[
        SPARK_TEST_SERVICE_PREFIX_BINDING_COUNT];
    SparkScheduler scheduler;
    SparkRequestApiSlot request_slots[SPARK_TEST_SERVICE_REQUEST_SLOT_COUNT];
    SparkRequestApi request_api;
    SparkServingEngine serving_engine;
    SparkServingRequestRecord request_records[
        SPARK_TEST_SERVICE_REQUEST_RECORD_COUNT];
    uint32_t request_token_storage[
        SPARK_TEST_SERVICE_REQUEST_RECORD_COUNT *
        SPARK_TEST_SERVICE_REQUEST_TOKEN_STRIDE];
    SparkServingEvent serving_event_ring[SPARK_TEST_SERVICE_EVENT_CAPACITY];
    uint32_t host_prefill_token_ids[
        SPARK_TEST_SERVICE_LANE_CAPACITY *
        SPARK_TEST_SERVICE_PREFILL_TOKEN_STRIDE];
    uint32_t physical_block_indices[
        SPARK_TEST_SERVICE_LANE_CAPACITY *
        SPARK_SCHEDULER_KV_BLOCK_TABLE_CAPACITY];
    uint32_t lane_physical_block_counts[SPARK_TEST_SERVICE_LANE_CAPACITY];
    SparkServiceRuntime service;
    SparkServiceClientSession client_sessions[SPARK_TEST_SERVICE_CLIENT_CAPACITY];
    SparkServiceRequestMap request_maps[SPARK_TEST_SERVICE_REQUEST_MAP_CAPACITY];
    SparkServiceEvent service_event_ring[SPARK_TEST_SERVICE_EVENT_CAPACITY];
    uint32_t first_prompt_tokens[SPARK_TEST_SERVICE_PROMPT_TOKEN_COUNT];
    uint32_t second_prompt_tokens[SPARK_TEST_SERVICE_PROMPT_TOKEN_COUNT];
} SparkTestServiceFixture;

typedef struct SparkTestServiceCallbackContext
{
    uint32_t prefill_callback_count;
    uint32_t decode_callback_count;
    uint32_t largest_prefill_lane_count;
    uint32_t saw_kv_block_table;
} SparkTestServiceCallbackContext;

static SparkTestServiceFixture Fixture;
static SparkTestServiceCallbackContext CallbackContext;

static void SparkTestServiceFillTokenIds(
    uint32_t *token_ids,
    uint32_t token_count,
    uint32_t first_token_id)
{
    uint32_t token_index;

    for (token_index = 0u; token_index < token_count; ++token_index)
    {
        token_ids[token_index] = first_token_id + token_index;
    }
}

static SparkStatus SparkTestServiceKvPrefetch(
    void *context,
    const SparkKvCachePrefetchPlan *prefetch_plan)
{
    (void)context;
    assert(prefetch_plan != 0);
    return SPARK_STATUS_OK;
}

static SparkStatus SparkTestServicePrefill(
    void *context,
    const SparkPromptPipelinePrefillDispatch *prefill_dispatch)
{
    SparkTestServiceCallbackContext *callback_context;

    callback_context = (SparkTestServiceCallbackContext *)context;
    assert(callback_context != 0);
    assert(prefill_dispatch != 0);
    assert(prefill_dispatch->host_token_ids != 0);
    assert(prefill_dispatch->kv_block_table_view != 0);
    assert(prefill_dispatch->kv_block_table_view->host_physical_block_indices != 0);
    assert(prefill_dispatch->lane_count != 0u);
    assert(prefill_dispatch->prompt_token_count != 0u);
    assert(prefill_dispatch->prompt_token_count <=
        SPARK_TEST_SERVICE_PREFILL_TOKEN_STRIDE);

    if (prefill_dispatch->lane_count > callback_context->largest_prefill_lane_count)
    {
        callback_context->largest_prefill_lane_count = prefill_dispatch->lane_count;
    }
    callback_context->prefill_callback_count += 1u;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkTestServiceDecode(
    void *context,
    const SparkServingDecodeDispatch *decode_dispatch,
    SparkServingDecodeResult *decode_result)
{
    SparkTestServiceCallbackContext *callback_context;
    uint32_t lane_index;

    callback_context = (SparkTestServiceCallbackContext *)context;
    assert(callback_context != 0);
    assert(decode_dispatch != 0);
    assert(decode_dispatch->request_dispatch != 0);
    assert(decode_dispatch->kv_block_table_view != 0);
    assert(decode_dispatch->decode_view != 0);
    assert(decode_dispatch->request_count != 0u);
    assert(decode_result != 0);

    SparkServingInitializeDecodeResult(
        decode_result,
        decode_dispatch->request_count,
        SPARK_SERVING_MAX_DECODE_TOKENS_PER_LANE);
    for (lane_index = 0u;
         lane_index < decode_dispatch->request_count;
         ++lane_index)
    {
        decode_result->token_counts[lane_index] = 1u;
        decode_result->token_ids[lane_index][0u] =
            710000u + callback_context->decode_callback_count + lane_index;
    }
    callback_context->decode_callback_count += 1u;
    callback_context->saw_kv_block_table = 1u;
    return SPARK_STATUS_OK;
}

static void SparkTestServiceInitializeFixture(
    SparkTestServiceFixture *fixture,
    SparkTestServiceCallbackContext *callback_context)
{
    SparkKvCacheConfiguration kv_configuration;
    SparkPrefixCacheConfiguration prefix_configuration;
    SparkSchedulerConfiguration scheduler_configuration;
    SparkRequestApiConfiguration request_api_configuration;
    SparkServingEngineConfiguration serving_configuration;
    SparkServiceConfiguration service_configuration;

    memset(fixture, 0, sizeof(*fixture));
    memset(callback_context, 0, sizeof(*callback_context));
    SparkTestServiceFillTokenIds(
        fixture->first_prompt_tokens,
        SPARK_TEST_SERVICE_PROMPT_TOKEN_COUNT,
        111000u);
    SparkTestServiceFillTokenIds(
        fixture->second_prompt_tokens,
        SPARK_TEST_SERVICE_PROMPT_TOKEN_COUNT,
        222000u);

    memset(&kv_configuration, 0, sizeof(kv_configuration));
    kv_configuration.abi_version = SPARK_KV_CACHE_ABI_VERSION;
    kv_configuration.descriptor_bytes =
        SPARK_KV_CACHE_CONFIGURATION_DESCRIPTOR_BYTES;
    kv_configuration.physical_block_count = SPARK_TEST_SERVICE_KV_BLOCK_COUNT;
    kv_configuration.block_token_count =
        SPARK_SCHEDULER_PREFILL_BLOCK_TOKENS;
    kv_configuration.layer_count = 78u;
    kv_configuration.kv_head_count = 8u;
    kv_configuration.head_dim = 128u;
    kv_configuration.bytes_per_scalar = (uint32_t)sizeof(uint16_t);
    kv_configuration.key_device_base = (void *)(uintptr_t)0x100000000ull;
    kv_configuration.value_device_base = (void *)(uintptr_t)0x200000000ull;
    kv_configuration.blocks = fixture->kv_blocks;
    assert(SparkKvCacheArenaInitialize(
        &fixture->kv_arena,
        &kv_configuration) == SPARK_STATUS_OK);

    memset(&prefix_configuration, 0, sizeof(prefix_configuration));
    prefix_configuration.abi_version = SPARK_PREFIX_CACHE_ABI_VERSION;
    prefix_configuration.descriptor_bytes =
        SPARK_PREFIX_CACHE_CONFIGURATION_DESCRIPTOR_BYTES;
    prefix_configuration.block_token_count =
        SPARK_SCHEDULER_PREFILL_BLOCK_TOKENS;
    prefix_configuration.entry_count = SPARK_TEST_SERVICE_PREFIX_ENTRY_COUNT;
    prefix_configuration.physical_block_count = SPARK_TEST_SERVICE_KV_BLOCK_COUNT;
    prefix_configuration.sequence_binding_count =
        SPARK_TEST_SERVICE_PREFIX_BINDING_COUNT;
    prefix_configuration.entries = fixture->prefix_entries;
    prefix_configuration.sequence_bindings = fixture->prefix_bindings;
    prefix_configuration.kv_cache_arena = &fixture->kv_arena;
    assert(SparkPrefixCacheInitialize(
        &fixture->prefix_cache,
        &prefix_configuration) == SPARK_STATUS_OK);

    memset(&scheduler_configuration, 0, sizeof(scheduler_configuration));
    scheduler_configuration.abi_version = SPARK_SCHEDULER_ABI_VERSION;
    scheduler_configuration.descriptor_bytes =
        SPARK_SCHEDULER_CONFIGURATION_DESCRIPTOR_BYTES;
    scheduler_configuration.spark_count = SPARK_SCHEDULER_MAX_SPARK_COUNT;
    scheduler_configuration.queue_depth_per_spark = 2u;
    scheduler_configuration.measured_profile_id =
        SPARK_STAGE_PLAN_MEASURED_PROFILE_20260701;
    scheduler_configuration.quantization_mode =
        SPARK_STAGE_PLAN_QUANTIZATION_NVFP4_4BIT;
    scheduler_configuration.max_prefill_tokens_per_step =
        SPARK_TEST_SERVICE_PREFILL_TOKEN_STRIDE;
    scheduler_configuration.configuration_flags =
        SPARK_SCHEDULER_CONFIGURATION_DEFAULT_FLAGS;
    scheduler_configuration.prefix_cache_block_tokens =
        SPARK_SCHEDULER_PREFILL_BLOCK_TOKENS;
    scheduler_configuration.prefix_cache = &fixture->prefix_cache;
    scheduler_configuration.stage_geometry.layer_count = SPARK_GLM52_MODEL_LAYER_COUNT;
    scheduler_configuration.stage_geometry.first_routed_layer = SPARK_GLM52_MODEL_FIRST_ROUTED_LAYER;
    assert(SparkSchedulerInitialize(
        &fixture->scheduler,
        &scheduler_configuration) == SPARK_STATUS_OK);

    memset(&request_api_configuration, 0, sizeof(request_api_configuration));
    request_api_configuration.abi_version = SPARK_REQUEST_API_ABI_VERSION;
    request_api_configuration.descriptor_bytes =
        SPARK_REQUEST_API_CONFIGURATION_DESCRIPTOR_BYTES;
    request_api_configuration.request_capacity = SPARK_TEST_SERVICE_REQUEST_SLOT_COUNT;
    request_api_configuration.prefetch_lane_count =
        SPARK_SCHEDULER_MAX_SPARK_COUNT;
    request_api_configuration.decode_batch_target = 2u;
    request_api_configuration.scheduler = &fixture->scheduler;
    request_api_configuration.request_slots = fixture->request_slots;
    request_api_configuration.kv_prefetch_function = SparkTestServiceKvPrefetch;
    assert(SparkRequestApiInitialize(
        &fixture->request_api,
        &request_api_configuration) == SPARK_STATUS_OK);

    memset(&serving_configuration, 0, sizeof(serving_configuration));
    serving_configuration.abi_version = SPARK_SERVING_ENGINE_ABI_VERSION;
    serving_configuration.descriptor_bytes =
        SPARK_SERVING_ENGINE_CONFIGURATION_DESCRIPTOR_BYTES;
    serving_configuration.runtime_contract_flags =
        SPARK_SERVING_RUNTIME_CONTRACT_PRODUCTION_REQUIRED_FLAGS |
        SPARK_SERVING_RUNTIME_CONTRACT_FLAG_JIT_KV_PREFETCH_CONNECTED |
        SPARK_SERVING_RUNTIME_CONTRACT_FLAG_OVERLAPPED_STAGING_READY;
    serving_configuration.default_output_token_budget = 1u;
    serving_configuration.default_max_prefill_tokens_per_step =
        SPARK_TEST_SERVICE_PREFILL_TOKEN_STRIDE;
    serving_configuration.max_context_tokens = 256u;
    serving_configuration.request_api = &fixture->request_api;
    serving_configuration.request_records = fixture->request_records;
    serving_configuration.request_record_capacity =
        SPARK_TEST_SERVICE_REQUEST_RECORD_COUNT;
    serving_configuration.request_token_storage = fixture->request_token_storage;
    serving_configuration.request_token_stride =
        SPARK_TEST_SERVICE_REQUEST_TOKEN_STRIDE;
    serving_configuration.event_ring = fixture->serving_event_ring;
    serving_configuration.event_ring_capacity = SPARK_TEST_SERVICE_EVENT_CAPACITY;
    serving_configuration.host_prefill_token_ids = fixture->host_prefill_token_ids;
    serving_configuration.host_prefill_token_stride =
        SPARK_TEST_SERVICE_PREFILL_TOKEN_STRIDE;
    serving_configuration.host_prefill_lane_capacity =
        SPARK_TEST_SERVICE_LANE_CAPACITY;
    serving_configuration.host_physical_block_indices =
        fixture->physical_block_indices;
    serving_configuration.kv_block_lane_stride =
        SPARK_SCHEDULER_KV_BLOCK_TABLE_CAPACITY;
    serving_configuration.kv_block_lane_capacity =
        SPARK_SCHEDULER_KV_BLOCK_TABLE_CAPACITY;
    serving_configuration.lane_physical_block_counts =
        fixture->lane_physical_block_counts;
    serving_configuration.lane_count_capacity = SPARK_TEST_SERVICE_LANE_CAPACITY;
    serving_configuration.prefill_function = SparkTestServicePrefill;
    serving_configuration.decode_function = SparkTestServiceDecode;
    serving_configuration.callback_context = callback_context;
    assert(SparkServingEngineInitialize(
        &fixture->serving_engine,
        &serving_configuration) == SPARK_STATUS_OK);

    memset(&service_configuration, 0, sizeof(service_configuration));
    service_configuration.abi_version = SPARK_SERVICE_ABI_VERSION;
    service_configuration.descriptor_bytes =
        SPARK_SERVICE_CONFIGURATION_DESCRIPTOR_BYTES;
    service_configuration.serving_engine = &fixture->serving_engine;
    service_configuration.client_sessions = fixture->client_sessions;
    service_configuration.client_session_capacity = SPARK_TEST_SERVICE_CLIENT_CAPACITY;
    service_configuration.request_maps = fixture->request_maps;
    service_configuration.request_map_capacity = SPARK_TEST_SERVICE_REQUEST_MAP_CAPACITY;
    service_configuration.event_ring = fixture->service_event_ring;
    service_configuration.event_ring_capacity = SPARK_TEST_SERVICE_EVENT_CAPACITY;
    assert(SparkServiceInitialize(
        &fixture->service,
        &service_configuration) == SPARK_STATUS_OK);
}

static void SparkTestServiceClientsUseInternalQueueing(void)
{
    SparkServiceClientId first_client_id;
    SparkServiceClientId second_client_id;
    SparkServiceSubmitTokenIdsRequest submit_request;
    SparkServiceSubmitResult submit_result;
    SparkServiceStats stats;
    SparkServiceEvent event;
    uint32_t first_token_event_count;
    uint32_t second_token_event_count;
    uint32_t completion_event_count;
    SparkStatus status;

    SparkTestServiceInitializeFixture(&Fixture, &CallbackContext);
    assert(SparkServiceRegisterClient(
        &Fixture.service,
        100u,
        &first_client_id) == SPARK_STATUS_OK);
    assert(SparkServiceRegisterClient(
        &Fixture.service,
        200u,
        &second_client_id) == SPARK_STATUS_OK);

    SparkServiceInitializeSubmitTokenIdsRequest(&submit_request);
    submit_request.client_id = first_client_id;
    submit_request.client_request_id = 11u;
    submit_request.token_count = SPARK_TEST_SERVICE_PROMPT_TOKEN_COUNT;
    submit_request.token_ids = Fixture.first_prompt_tokens;
    assert(SparkServiceSubmitTokenIds(
        &Fixture.service,
        &submit_request,
        &submit_result) == SPARK_STATUS_OK);
    assert(submit_result.client_id == first_client_id);
    assert(submit_result.client_request_id == 11u);

    assert(SparkServiceSubmitTokenIds(
        &Fixture.service,
        &submit_request,
        &submit_result) == SPARK_STATUS_DUPLICATE);

    submit_request.client_id = second_client_id;
    submit_request.client_request_id = 22u;
    submit_request.token_ids = Fixture.second_prompt_tokens;
    assert(SparkServiceSubmitTokenIds(
        &Fixture.service,
        &submit_request,
        &submit_result) == SPARK_STATUS_OK);

    status = SparkServicePump(&Fixture.service, 16u, &stats);
    assert(status == SPARK_STATUS_OK);
    assert(CallbackContext.prefill_callback_count == 1u);
    assert(CallbackContext.largest_prefill_lane_count == 2u);
    assert(CallbackContext.decode_callback_count == 1u);
    assert(CallbackContext.saw_kv_block_table == 1u);
    assert(stats.serving_stats.prefill_batch_dispatch_count == 1u);
    assert(stats.serving_stats.decode_dispatch_count == 1u);

    first_token_event_count = 0u;
    second_token_event_count = 0u;
    completion_event_count = 0u;
    while (SparkServicePopEvent(&Fixture.service, &event) == SPARK_STATUS_OK)
    {
        if (event.kind == SPARK_SERVICE_EVENT_KIND_TOKEN)
        {
            if (event.client_id == first_client_id)
            {
                first_token_event_count += 1u;
                assert(event.client_request_id == 11u);
            }
            else if (event.client_id == second_client_id)
            {
                second_token_event_count += 1u;
                assert(event.client_request_id == 22u);
            }
        }
        else if (event.kind == SPARK_SERVICE_EVENT_KIND_REQUEST_COMPLETED)
        {
            completion_event_count += 1u;
        }
    }
    assert(first_token_event_count == 1u);
    assert(second_token_event_count == 1u);
    assert(completion_event_count == 2u);
}

static void SparkTestServiceTokenFrameSubmitWorks(void)
{
    SparkServiceClientId client_id;
    SparkServiceFrameHeader frame_header;
    SparkServiceSubmitTokenIdsFrameBody frame_body;
    SparkServiceSubmitResult submit_result;
    unsigned char frame_body_storage[
        SPARK_SERVICE_FRAME_SUBMIT_TOKENS_DESCRIPTOR_BYTES +
        8u * sizeof(uint32_t)];
    uint32_t *token_ids;
    uint32_t token_index;

    SparkTestServiceInitializeFixture(&Fixture, &CallbackContext);
    assert(SparkServiceRegisterClient(
        &Fixture.service,
        300u,
        &client_id) == SPARK_STATUS_OK);

    SparkServiceInitializeFrameHeader(
        &frame_header,
        SPARK_SERVICE_FRAME_KIND_SUBMIT_TOKEN_IDS);
    frame_header.client_id = client_id;
    frame_header.client_request_id = 44u;
    frame_header.body_bytes = sizeof(frame_body_storage);

    memset(&frame_body, 0, sizeof(frame_body));
    frame_body.abi_version = SPARK_SERVICE_ABI_VERSION;
    frame_body.descriptor_bytes =
        SPARK_SERVICE_FRAME_SUBMIT_TOKENS_DESCRIPTOR_BYTES;
    frame_body.output_token_budget = 1u;
    frame_body.token_count = 8u;
    memcpy(frame_body_storage, &frame_body, sizeof(frame_body));
    token_ids = (uint32_t *)(void *)(frame_body_storage + sizeof(frame_body));
    for (token_index = 0u; token_index < frame_body.token_count; ++token_index)
    {
        token_ids[token_index] = 88000u + token_index;
    }

    assert(SparkServiceHandleSubmitTokenIdsFrame(
        &Fixture.service,
        client_id,
        &frame_header,
        frame_body_storage,
        sizeof(frame_body_storage),
        &submit_result) == SPARK_STATUS_OK);
    assert(submit_result.client_id == client_id);
    assert(submit_result.client_request_id == 44u);
    assert(submit_result.prompt_token_count == 8u);
}

int main(void)
{
    SparkTestServiceClientsUseInternalQueueing();
    SparkTestServiceTokenFrameSubmitWorks();
    return 0;
}
