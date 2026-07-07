#include <assert.h>
#include <stdint.h>
#include <string.h>

#include "sparkpipe/spark_glm52_prompt_pipeline.h"

#define SPARK_TEST_PROMPT_PIPELINE_REQUEST_SLOT_COUNT 4u
#define SPARK_TEST_PROMPT_PIPELINE_KV_BLOCK_COUNT \
    SPARK_GLM52_SCHEDULER_KV_BLOCK_TABLE_CAPACITY
#define SPARK_TEST_PROMPT_PIPELINE_PREFIX_ENTRY_COUNT \
    SPARK_GLM52_SCHEDULER_KV_BLOCK_TABLE_CAPACITY
#define SPARK_TEST_PROMPT_PIPELINE_PREFIX_BINDING_COUNT \
    (SPARK_GLM52_SCHEDULER_KV_BLOCK_TABLE_CAPACITY + 4u)
#define SPARK_TEST_PROMPT_PIPELINE_PREFILL_TOKEN_STRIDE 64u
#define SPARK_TEST_PROMPT_PIPELINE_LANE_CAPACITY \
    SPARK_GLM52_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT
#define SPARK_TEST_PROMPT_PIPELINE_CONTEXT_TOKENS 97u

typedef struct SparkTestPromptPipelineFixture
{
    SparkGlm52KvCacheArena kv_arena;
    SparkGlm52KvCacheBlock kv_blocks[
        SPARK_TEST_PROMPT_PIPELINE_KV_BLOCK_COUNT];
    SparkGlm52PrefixCache prefix_cache;
    SparkGlm52PrefixCacheEntry prefix_entries[
        SPARK_TEST_PROMPT_PIPELINE_PREFIX_ENTRY_COUNT];
    SparkGlm52PrefixCacheSequenceBinding prefix_bindings[
        SPARK_TEST_PROMPT_PIPELINE_PREFIX_BINDING_COUNT];
    SparkGlm52Scheduler scheduler;
    SparkGlm52RequestApiSlot request_slots[
        SPARK_TEST_PROMPT_PIPELINE_REQUEST_SLOT_COUNT];
    SparkGlm52RequestApi api;
    uint32_t prompt_tokens[SPARK_TEST_PROMPT_PIPELINE_CONTEXT_TOKENS];
    uint32_t prefill_token_staging[
        SPARK_TEST_PROMPT_PIPELINE_LANE_CAPACITY *
        SPARK_TEST_PROMPT_PIPELINE_PREFILL_TOKEN_STRIDE];
    uint32_t physical_block_indices[
        SPARK_TEST_PROMPT_PIPELINE_LANE_CAPACITY *
        SPARK_GLM52_SCHEDULER_KV_BLOCK_TABLE_CAPACITY];
    uint32_t lane_physical_block_counts[
        SPARK_TEST_PROMPT_PIPELINE_LANE_CAPACITY];
} SparkTestPromptPipelineFixture;

typedef struct SparkTestPromptPipelineCallbackContext
{
    const uint32_t *expected_prompt_tokens;
    uint32_t prefill_callback_count;
    uint32_t decode_callback_count;
} SparkTestPromptPipelineCallbackContext;

static SparkTestPromptPipelineFixture Fixture;

static void SparkTestPromptPipelineFillTokenIds(
    uint32_t *tokens,
    uint32_t token_count,
    uint32_t first_token_id)
{
    uint32_t token_index;

    for (token_index = 0u; token_index < token_count; ++token_index)
    {
        tokens[token_index] = first_token_id + token_index;
    }
}

static SparkStatus SparkTestPromptPipelineKvPrefetch(
    void *context,
    const SparkGlm52KvCachePrefetchPlan *prefetch_plan)
{
    (void)context;
    (void)prefetch_plan;
    return SPARK_STATUS_OK;
}

static void SparkTestPromptPipelineInitialize(
    SparkTestPromptPipelineFixture *fixture)
{
    SparkGlm52KvCacheConfiguration kv_configuration;
    SparkGlm52PrefixCacheConfiguration prefix_configuration;
    SparkGlm52SchedulerConfiguration scheduler_configuration;
    SparkGlm52RequestApiConfiguration api_configuration;

    memset(fixture, 0, sizeof(*fixture));
    SparkTestPromptPipelineFillTokenIds(
        fixture->prompt_tokens,
        SPARK_TEST_PROMPT_PIPELINE_CONTEXT_TOKENS,
        70000u);

    memset(&kv_configuration, 0, sizeof(kv_configuration));
    kv_configuration.abi_version = SPARK_GLM52_KV_CACHE_ABI_VERSION;
    kv_configuration.descriptor_bytes =
        SPARK_GLM52_KV_CACHE_CONFIGURATION_DESCRIPTOR_BYTES;
    kv_configuration.physical_block_count =
        SPARK_TEST_PROMPT_PIPELINE_KV_BLOCK_COUNT;
    kv_configuration.block_token_count =
        SPARK_GLM52_SCHEDULER_PREFILL_BLOCK_TOKENS;
    kv_configuration.layer_count = 78u;
    kv_configuration.kv_head_count = 8u;
    kv_configuration.head_dim = 128u;
    kv_configuration.bytes_per_scalar = 2u;
    kv_configuration.key_device_base = (void *)(uintptr_t)0x100000000ull;
    kv_configuration.value_device_base = (void *)(uintptr_t)0x200000000ull;
    kv_configuration.blocks = fixture->kv_blocks;
    assert(SparkGlm52KvCacheArenaInitialize(
        &fixture->kv_arena,
        &kv_configuration) == SPARK_STATUS_OK);

    memset(&prefix_configuration, 0, sizeof(prefix_configuration));
    prefix_configuration.abi_version = SPARK_GLM52_PREFIX_CACHE_ABI_VERSION;
    prefix_configuration.descriptor_bytes =
        SPARK_GLM52_PREFIX_CACHE_CONFIGURATION_DESCRIPTOR_BYTES;
    prefix_configuration.block_token_count =
        SPARK_GLM52_SCHEDULER_PREFILL_BLOCK_TOKENS;
    prefix_configuration.entry_count =
        SPARK_TEST_PROMPT_PIPELINE_PREFIX_ENTRY_COUNT;
    prefix_configuration.physical_block_count =
        SPARK_TEST_PROMPT_PIPELINE_KV_BLOCK_COUNT;
    prefix_configuration.sequence_binding_count =
        SPARK_TEST_PROMPT_PIPELINE_PREFIX_BINDING_COUNT;
    prefix_configuration.entries = fixture->prefix_entries;
    prefix_configuration.sequence_bindings = fixture->prefix_bindings;
    prefix_configuration.kv_cache_arena = &fixture->kv_arena;
    assert(SparkGlm52PrefixCacheInitialize(
        &fixture->prefix_cache,
        &prefix_configuration) == SPARK_STATUS_OK);

    memset(&scheduler_configuration, 0, sizeof(scheduler_configuration));
    scheduler_configuration.abi_version = SPARK_GLM52_SCHEDULER_ABI_VERSION;
    scheduler_configuration.descriptor_bytes =
        SPARK_GLM52_SCHEDULER_CONFIGURATION_DESCRIPTOR_BYTES;
    scheduler_configuration.spark_count =
        SPARK_GLM52_SCHEDULER_MAX_SPARK_COUNT;
    scheduler_configuration.queue_depth_per_spark = 2u;
    scheduler_configuration.measured_profile_id =
        SPARK_GLM52_STAGE_PLAN_MEASURED_PROFILE_20260701;
    scheduler_configuration.quantization_mode =
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_NVFP4_4BIT;
    scheduler_configuration.max_prefill_tokens_per_step =
        SPARK_TEST_PROMPT_PIPELINE_PREFILL_TOKEN_STRIDE;
    scheduler_configuration.configuration_flags =
        SPARK_GLM52_SCHEDULER_CONFIGURATION_DEFAULT_FLAGS;
    scheduler_configuration.prefix_cache_block_tokens =
        SPARK_GLM52_SCHEDULER_PREFILL_BLOCK_TOKENS;
    scheduler_configuration.prefix_cache = &fixture->prefix_cache;
    assert(SparkGlm52SchedulerInitialize(
        &fixture->scheduler,
        &scheduler_configuration) == SPARK_STATUS_OK);

    memset(&api_configuration, 0, sizeof(api_configuration));
    api_configuration.abi_version = SPARK_GLM52_REQUEST_API_ABI_VERSION;
    api_configuration.descriptor_bytes =
        SPARK_GLM52_REQUEST_API_CONFIGURATION_DESCRIPTOR_BYTES;
    api_configuration.request_capacity =
        SPARK_TEST_PROMPT_PIPELINE_REQUEST_SLOT_COUNT;
    api_configuration.prefetch_lane_count =
        SPARK_GLM52_SCHEDULER_MAX_SPARK_COUNT;
    api_configuration.decode_batch_target = 1u;
    api_configuration.scheduler = &fixture->scheduler;
    api_configuration.request_slots = fixture->request_slots;
    api_configuration.kv_prefetch_function = SparkTestPromptPipelineKvPrefetch;
    assert(SparkGlm52RequestApiInitialize(
        &fixture->api,
        &api_configuration) == SPARK_STATUS_OK);
}

static void SparkTestPromptPipelineSubmit(
    SparkTestPromptPipelineFixture *fixture)
{
    SparkGlm52RequestApiSubmitRequest request;
    SparkGlm52RequestApiHandle handle;

    memset(&request, 0, sizeof(request));
    request.abi_version = SPARK_GLM52_REQUEST_API_ABI_VERSION;
    request.descriptor_bytes =
        SPARK_GLM52_REQUEST_API_SUBMIT_DESCRIPTOR_BYTES;
    request.prompt_token_count = SPARK_TEST_PROMPT_PIPELINE_CONTEXT_TOKENS;
    request.output_token_budget = 1u;
    request.max_prefill_tokens_per_step =
        SPARK_TEST_PROMPT_PIPELINE_PREFILL_TOKEN_STRIDE;
    request.request_id = 701u;
    request.sequence_id = 1701u;
    request.prompt_token_ids = fixture->prompt_tokens;
    assert(SparkGlm52RequestApiSubmit(
        &fixture->api,
        &request,
        &handle) == SPARK_STATUS_OK);
    assert(handle != SPARK_GLM52_REQUEST_API_INVALID_HANDLE);
}

static SparkStatus SparkTestPromptPipelinePrefillCallback(
    void *context,
    const SparkGlm52PromptPipelinePrefillDispatch *prefill_dispatch)
{
    SparkTestPromptPipelineCallbackContext *callback_context;
    uint32_t token_index;
    uint32_t expected_offset;
    uint32_t expected_count;

    callback_context = (SparkTestPromptPipelineCallbackContext *)context;
    assert(callback_context != 0);
    assert(prefill_dispatch != 0);
    assert(prefill_dispatch->abi_version ==
        SPARK_GLM52_PROMPT_PIPELINE_ABI_VERSION);
    assert(prefill_dispatch->descriptor_bytes ==
        SPARK_GLM52_PROMPT_PIPELINE_PREFILL_DISPATCH_DESCRIPTOR_BYTES);
    assert(prefill_dispatch->active_sequence_count == 1u);
    assert(prefill_dispatch->lane_count == 1u);
    assert(prefill_dispatch->host_token_ids != 0);
    assert(prefill_dispatch->kv_block_table_view != 0);
    assert(prefill_dispatch->kv_block_table_view->lane_count == 1u);
    assert(prefill_dispatch->kv_block_table_view->host_physical_block_indices != 0);

    expected_offset = callback_context->prefill_callback_count == 0u ? 0u : 64u;
    expected_count = callback_context->prefill_callback_count == 0u ? 64u : 33u;
    assert(prefill_dispatch->prompt_token_offset == expected_offset);
    assert(prefill_dispatch->prompt_token_count == expected_count);
    assert(prefill_dispatch->prompt_token_stride == expected_count);

    for (token_index = 0u; token_index < expected_count; ++token_index)
    {
        assert(prefill_dispatch->host_token_ids[token_index] ==
            callback_context->expected_prompt_tokens[expected_offset + token_index]);
    }
    for (token_index = expected_count;
         token_index < prefill_dispatch->host_token_stride;
         ++token_index)
    {
        assert(prefill_dispatch->host_token_ids[token_index] == 0u);
    }

    callback_context->prefill_callback_count += 1u;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkTestPromptPipelineDecodeCallback(
    void *context,
    const SparkGlm52RequestApiDispatch *decode_dispatch)
{
    SparkTestPromptPipelineCallbackContext *callback_context;

    callback_context = (SparkTestPromptPipelineCallbackContext *)context;
    assert(callback_context != 0);
    assert(decode_dispatch != 0);
    assert(decode_dispatch->kind ==
        SPARK_GLM52_REQUEST_API_DISPATCH_KIND_DECODE_BATCH);
    assert(decode_dispatch->decode_batch_decision.active_sequence_count == 1u);
    callback_context->decode_callback_count += 1u;
    return SPARK_STATUS_OK;
}

static void SparkTestPromptPipelineRunsPrefillThenDecodeWithoutPython(void)
{
    SparkGlm52PromptPipelineConfiguration configuration;
    SparkGlm52PromptPipelineRunStats stats;
    SparkTestPromptPipelineCallbackContext callback_context;

    SparkTestPromptPipelineInitialize(&Fixture);
    SparkTestPromptPipelineSubmit(&Fixture);

    memset(&callback_context, 0, sizeof(callback_context));
    callback_context.expected_prompt_tokens = Fixture.prompt_tokens;

    memset(&configuration, 0, sizeof(configuration));
    configuration.abi_version = SPARK_GLM52_PROMPT_PIPELINE_ABI_VERSION;
    configuration.descriptor_bytes =
        SPARK_GLM52_PROMPT_PIPELINE_CONFIGURATION_DESCRIPTOR_BYTES;
    configuration.run_flags =
        SPARK_GLM52_PROMPT_PIPELINE_RUN_FLAG_STOP_AFTER_FIRST_DECODE_DISPATCH;
    configuration.request_api = &Fixture.api;
    configuration.host_prefill_token_ids = Fixture.prefill_token_staging;
    configuration.host_prefill_token_stride =
        SPARK_TEST_PROMPT_PIPELINE_PREFILL_TOKEN_STRIDE;
    configuration.host_prefill_lane_capacity =
        SPARK_TEST_PROMPT_PIPELINE_LANE_CAPACITY;
    configuration.host_physical_block_indices = Fixture.physical_block_indices;
    configuration.kv_block_lane_stride =
        SPARK_GLM52_SCHEDULER_KV_BLOCK_TABLE_CAPACITY;
    configuration.kv_block_lane_capacity =
        SPARK_GLM52_SCHEDULER_KV_BLOCK_TABLE_CAPACITY;
    configuration.lane_physical_block_counts =
        Fixture.lane_physical_block_counts;
    configuration.lane_count_capacity =
        SPARK_TEST_PROMPT_PIPELINE_LANE_CAPACITY;
    configuration.prefill_function = SparkTestPromptPipelinePrefillCallback;
    configuration.decode_function = SparkTestPromptPipelineDecodeCallback;
    configuration.callback_context = &callback_context;

    assert(SparkGlm52PromptPipelineRun(
        &configuration,
        &stats) == SPARK_STATUS_OK);
    assert(stats.abi_version == SPARK_GLM52_PROMPT_PIPELINE_ABI_VERSION);
    assert(stats.completed_dispatch_count == 3u);
    assert(stats.prefill_dispatch_count == 2u);
    assert(stats.decode_dispatch_count == 1u);
    assert(stats.prefill_token_count == SPARK_TEST_PROMPT_PIPELINE_CONTEXT_TOKENS);
    assert(stats.maximum_prefill_token_count == 64u);
    assert(stats.maximum_prefill_lane_count == 1u);
    assert(stats.reached_decode_dispatch == 1u);
    assert(callback_context.prefill_callback_count == 2u);
    assert(callback_context.decode_callback_count == 1u);
}

int main(void)
{
    SparkTestPromptPipelineRunsPrefillThenDecodeWithoutPython();
    return 0;
}
