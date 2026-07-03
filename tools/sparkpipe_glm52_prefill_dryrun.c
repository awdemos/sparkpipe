#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "sparkpipe/spark_glm52_request_api.h"

#define SPARK_PREFILL_DRYRUN_REQUEST_SLOT_COUNT 4u
#define SPARK_PREFILL_DRYRUN_KV_BLOCK_COUNT \
    SPARK_GLM52_SCHEDULER_KV_BLOCK_TABLE_CAPACITY
#define SPARK_PREFILL_DRYRUN_PREFIX_ENTRY_COUNT \
    SPARK_GLM52_SCHEDULER_KV_BLOCK_TABLE_CAPACITY
#define SPARK_PREFILL_DRYRUN_PREFIX_BINDING_COUNT \
    (SPARK_GLM52_SCHEDULER_KV_BLOCK_TABLE_CAPACITY + 4u)
#define SPARK_PREFILL_DRYRUN_MAX_STEPS 1024u

typedef struct SparkPrefillDryrun
{
    SparkGlm52KvCacheArena kv_arena;
    SparkGlm52KvCacheBlock kv_blocks[
        SPARK_PREFILL_DRYRUN_KV_BLOCK_COUNT];
    SparkGlm52PrefixCache prefix_cache;
    SparkGlm52PrefixCacheEntry prefix_entries[
        SPARK_PREFILL_DRYRUN_PREFIX_ENTRY_COUNT];
    SparkGlm52PrefixCacheSequenceBinding prefix_bindings[
        SPARK_PREFILL_DRYRUN_PREFIX_BINDING_COUNT];
    SparkGlm52Scheduler scheduler;
    SparkGlm52RequestApiSlot request_slots[
        SPARK_PREFILL_DRYRUN_REQUEST_SLOT_COUNT];
    SparkGlm52RequestApi api;
    uint32_t physical_block_indices[
        SPARK_GLM52_SCHEDULER_KV_BLOCK_TABLE_CAPACITY];
    uint32_t lane_physical_block_counts[1u];
} SparkPrefillDryrun;

static SparkPrefillDryrun Dryrun;
static uint32_t PromptTokens[SPARK_GLM52_SCHEDULER_MAX_CONTEXT_TOKENS];

static int32_t SparkPrefillDryrunParseU32(
    const char *text,
    uint32_t *value_out)
{
    uint64_t value;
    uint32_t index;

    if (text == 0 || text[0] == '\0' || value_out == 0)
        return(-1);
    value = 0u;
    for (index=0u; text[index] != '\0'; ++index)
    {
        if (text[index] < '0' || text[index] > '9')
            return(-2);
        value = ((value * 10u) + (uint32_t)(text[index] - '0'));
        if (value > 0xffffffffull)
            return(-3);
    }
    *value_out = (uint32_t)value;
    return(0);
}

static uint32_t SparkPrefillDryrunCharacterIsSeparator(
    int32_t ch)
{
    return ch == ',' || ch == ' ' || ch == '\n' || ch == '\r' || ch == '\t';
}

static int32_t SparkPrefillDryrunReadTokenFile(
    const char *path,
    uint32_t *tokens,
    uint32_t token_capacity,
    uint32_t *token_count_out)
{
    FILE *fp;
    uint64_t value;
    uint32_t token_count;
    uint32_t have_value;
    int32_t ch;

    if (path == 0 || tokens == 0 || token_capacity == 0u ||
        token_count_out == 0)
        return(-1);
    fp = fopen(path, "rb");
    if (fp == 0)
        return(-2);
    value = 0u;
    token_count = 0u;
    have_value = 0u;
    while ((ch = fgetc(fp)) != EOF)
    {
        if (ch >= '0' && ch <= '9')
        {
            have_value = 1u;
            value = ((value * 10u) + (uint32_t)(ch - '0'));
            if (value > 0xffffffffull)
            {
                fclose(fp);
                return(-3);
            }
            continue;
        }
        if (!SparkPrefillDryrunCharacterIsSeparator(ch))
        {
            fclose(fp);
            return(-4);
        }
        if (have_value != 0u)
        {
            if (token_count >= token_capacity)
            {
                fclose(fp);
                return(-5);
            }
            tokens[token_count++] = (uint32_t)value;
            value = 0u;
            have_value = 0u;
        }
    }
    if (have_value != 0u)
    {
        if (token_count >= token_capacity)
        {
            fclose(fp);
            return(-6);
        }
        tokens[token_count++] = (uint32_t)value;
    }
    fclose(fp);
    if (token_count == 0u)
        return(-7);
    *token_count_out = token_count;
    return(0);
}

static SparkStatus SparkPrefillDryrunKvPrefetch(
    void *context,
    const SparkGlm52KvCachePrefetchPlan *prefetch_plan)
{
    (void)context;
    (void)prefetch_plan;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkPrefillDryrunInitialize(
    SparkPrefillDryrun *dryrun,
    uint32_t max_prefill_tokens_per_step)
{
    SparkGlm52KvCacheConfiguration kv_configuration;
    SparkGlm52PrefixCacheConfiguration prefix_configuration;
    SparkGlm52SchedulerConfiguration scheduler_configuration;
    SparkGlm52RequestApiConfiguration api_configuration;
    SparkStatus status;

    memset(dryrun, 0, sizeof(*dryrun));
    memset(&kv_configuration, 0, sizeof(kv_configuration));
    kv_configuration.abi_version = SPARK_GLM52_KV_CACHE_ABI_VERSION;
    kv_configuration.descriptor_bytes =
        SPARK_GLM52_KV_CACHE_CONFIGURATION_DESCRIPTOR_BYTES;
    kv_configuration.physical_block_count =
        SPARK_PREFILL_DRYRUN_KV_BLOCK_COUNT;
    kv_configuration.block_token_count =
        SPARK_GLM52_SCHEDULER_PREFILL_BLOCK_TOKENS;
    kv_configuration.layer_count = 78u;
    kv_configuration.kv_head_count = 8u;
    kv_configuration.head_dim = 128u;
    kv_configuration.bytes_per_scalar = 2u;
    kv_configuration.key_device_base = (void *)(uintptr_t)0x100000000ull;
    kv_configuration.value_device_base = (void *)(uintptr_t)0x200000000ull;
    kv_configuration.blocks = dryrun->kv_blocks;
    status = SparkGlm52KvCacheArenaInitialize(
        &dryrun->kv_arena,
        &kv_configuration);
    if (status != SPARK_STATUS_OK)
        return status;
    memset(&prefix_configuration, 0, sizeof(prefix_configuration));
    prefix_configuration.abi_version = SPARK_GLM52_PREFIX_CACHE_ABI_VERSION;
    prefix_configuration.descriptor_bytes =
        SPARK_GLM52_PREFIX_CACHE_CONFIGURATION_DESCRIPTOR_BYTES;
    prefix_configuration.block_token_count =
        SPARK_GLM52_SCHEDULER_PREFILL_BLOCK_TOKENS;
    prefix_configuration.entry_count =
        SPARK_PREFILL_DRYRUN_PREFIX_ENTRY_COUNT;
    prefix_configuration.physical_block_count =
        SPARK_PREFILL_DRYRUN_KV_BLOCK_COUNT;
    prefix_configuration.sequence_binding_count =
        SPARK_PREFILL_DRYRUN_PREFIX_BINDING_COUNT;
    prefix_configuration.entries = dryrun->prefix_entries;
    prefix_configuration.sequence_bindings = dryrun->prefix_bindings;
    prefix_configuration.kv_cache_arena = &dryrun->kv_arena;
    status = SparkGlm52PrefixCacheInitialize(
        &dryrun->prefix_cache,
        &prefix_configuration);
    if (status != SPARK_STATUS_OK)
        return status;
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
        max_prefill_tokens_per_step;
    scheduler_configuration.prefix_cache = &dryrun->prefix_cache;
    status = SparkGlm52SchedulerInitialize(
        &dryrun->scheduler,
        &scheduler_configuration);
    if (status != SPARK_STATUS_OK)
        return status;
    memset(&api_configuration, 0, sizeof(api_configuration));
    api_configuration.abi_version = SPARK_GLM52_REQUEST_API_ABI_VERSION;
    api_configuration.descriptor_bytes =
        SPARK_GLM52_REQUEST_API_CONFIGURATION_DESCRIPTOR_BYTES;
    api_configuration.request_capacity =
        SPARK_PREFILL_DRYRUN_REQUEST_SLOT_COUNT;
    api_configuration.prefetch_lane_count =
        SPARK_GLM52_SCHEDULER_MAX_SPARK_COUNT;
    api_configuration.decode_batch_target = 1u;
    api_configuration.scheduler = &dryrun->scheduler;
    api_configuration.request_slots = dryrun->request_slots;
    api_configuration.kv_prefetch_function = SparkPrefillDryrunKvPrefetch;
    status = SparkGlm52RequestApiInitialize(
        &dryrun->api,
        &api_configuration);
    return status;
}

static SparkStatus SparkPrefillDryrunSubmit(
    SparkPrefillDryrun *dryrun,
    const uint32_t *tokens,
    uint32_t token_count,
    uint32_t max_prefill_tokens_per_step,
    SparkGlm52RequestApiHandle *handle_out)
{
    SparkGlm52RequestApiSubmitRequest request;

    memset(&request, 0, sizeof(request));
    request.abi_version = SPARK_GLM52_REQUEST_API_ABI_VERSION;
    request.descriptor_bytes =
        SPARK_GLM52_REQUEST_API_SUBMIT_DESCRIPTOR_BYTES;
    request.prompt_token_count = token_count;
    request.output_token_budget = 1u;
    request.max_prefill_tokens_per_step = max_prefill_tokens_per_step;
    request.request_id = 1u;
    request.sequence_id = 1u;
    request.prompt_token_ids = tokens;
    return SparkGlm52RequestApiSubmit(&dryrun->api, &request, handle_out);
}

static void SparkPrefillDryrunPrintPrefill(
    uint32_t step_index,
    const SparkGlm52RequestApiDispatch *dispatch,
    const SparkGlm52SchedulerDecision *decision,
    uint32_t block_count)
{
    printf("%u\tprefill\t%u\t%u\t%u\t%u\t%u\t%u\t%u\n",
        step_index,
        decision->scheduled_prompt_token_offset,
        decision->scheduled_prompt_token_count,
        decision->remaining_prompt_token_count_after_step,
        decision->cache_commit_token_count_after_step,
        decision->prefill_block_count,
        block_count,
        dispatch->flags);
}

static SparkStatus SparkPrefillDryrunPrintDispatch(
    SparkPrefillDryrun *dryrun,
    uint32_t step_index,
    const SparkGlm52RequestApiDispatch *dispatch)
{
    SparkStatus status;

    if (dispatch->kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL)
    {
        memset(dryrun->physical_block_indices, 0,
            sizeof(dryrun->physical_block_indices));
        memset(dryrun->lane_physical_block_counts, 0,
            sizeof(dryrun->lane_physical_block_counts));
        status = SparkGlm52RequestApiBuildDispatchKvBlockTables(
            &dryrun->api,
            dispatch,
            dryrun->physical_block_indices,
            SPARK_GLM52_SCHEDULER_KV_BLOCK_TABLE_CAPACITY,
            SPARK_GLM52_SCHEDULER_KV_BLOCK_TABLE_CAPACITY,
            dryrun->lane_physical_block_counts,
            1u);
        if (status != SPARK_STATUS_OK)
            return status;
        SparkPrefillDryrunPrintPrefill(
            step_index,
            dispatch,
            &dispatch->prefill_decision,
            dryrun->lane_physical_block_counts[0u]);
        return SPARK_STATUS_OK;
    }
    if (dispatch->kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_DECODE_BATCH)
    {
        printf("%u\tdecode_ready\t0\t0\t0\t0\t0\t0\t%u\n",
            step_index,
            dispatch->flags);
        return SPARK_STATUS_OK;
    }
    return SPARK_STATUS_INVALID_ARGUMENT;
}

static SparkStatus SparkPrefillDryrunRun(
    SparkPrefillDryrun *dryrun)
{
    SparkGlm52RequestApiDispatch dispatch;
    SparkStatus status;
    uint32_t step_index;

    printf("step\tkind\ttoken_offset\ttoken_count\tremaining\tcommit_after\tprefill_blocks\tkv_blocks\tflags\n");
    for (step_index=0u; step_index<SPARK_PREFILL_DRYRUN_MAX_STEPS; ++step_index)
    {
        memset(&dispatch, 0, sizeof(dispatch));
        status = SparkGlm52RequestApiScheduleNext(&dryrun->api, &dispatch);
        if (status != SPARK_STATUS_OK)
            return status;
        if (dispatch.accepted == 0u)
            return SPARK_STATUS_NOT_FOUND;
        status = SparkPrefillDryrunPrintDispatch(
            dryrun,
            step_index,
            &dispatch);
        if (status != SPARK_STATUS_OK)
            return status;
        status = SparkGlm52RequestApiCompleteDispatch(
            &dryrun->api,
            &dispatch);
        if (status != SPARK_STATUS_OK)
            return status;
        if (dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_DECODE_BATCH)
            return SPARK_STATUS_OK;
    }
    return SPARK_STATUS_CAPACITY_EXCEEDED;
}

static int32_t SparkPrefillDryrunUsage(
    const char *program)
{
    fprintf(stderr,
        "usage: %s --tokens path [--max-prefill-tokens n]\n",
        program);
    return(2);
}

int main(
    int argc,
    char **argv)
{
    const char *token_path;
    uint32_t token_count;
    uint32_t max_prefill_tokens_per_step;
    SparkGlm52RequestApiHandle handle;
    SparkStatus status;
    int32_t arg_index;
    int32_t parse_status;

    token_path = 0;
    max_prefill_tokens_per_step =
        SPARK_GLM52_SCHEDULER_DEFAULT_MAX_PREFILL_TOKENS_PER_STEP;
    for (arg_index=1; arg_index<argc; ++arg_index)
    {
        if (strcmp(argv[arg_index], "--tokens") == 0 &&
            arg_index + 1 < argc)
        {
            token_path = argv[++arg_index];
            continue;
        }
        if (strcmp(argv[arg_index], "--max-prefill-tokens") == 0 &&
            arg_index + 1 < argc)
        {
            parse_status = SparkPrefillDryrunParseU32(
                argv[++arg_index],
                &max_prefill_tokens_per_step);
            if (parse_status < 0)
                return SparkPrefillDryrunUsage(argv[0]);
            continue;
        }
        return SparkPrefillDryrunUsage(argv[0]);
    }
    if (token_path == 0)
        return SparkPrefillDryrunUsage(argv[0]);
    parse_status = SparkPrefillDryrunReadTokenFile(
        token_path,
        PromptTokens,
        SPARK_GLM52_SCHEDULER_MAX_CONTEXT_TOKENS,
        &token_count);
    if (parse_status < 0)
    {
        fprintf(stderr, "failed to read token file: %d\n", parse_status);
        return(1);
    }
    status = SparkPrefillDryrunInitialize(
        &Dryrun,
        max_prefill_tokens_per_step);
    if (status != SPARK_STATUS_OK)
    {
        fprintf(stderr, "initialize failed: %s\n", SparkStatusToString(status));
        return(1);
    }
    status = SparkPrefillDryrunSubmit(
        &Dryrun,
        PromptTokens,
        token_count,
        max_prefill_tokens_per_step,
        &handle);
    if (status != SPARK_STATUS_OK || handle == 0u)
    {
        fprintf(stderr, "submit failed: %s\n", SparkStatusToString(status));
        return(1);
    }
    status = SparkPrefillDryrunRun(&Dryrun);
    if (status != SPARK_STATUS_OK)
    {
        fprintf(stderr, "dryrun failed: %s\n", SparkStatusToString(status));
        return(1);
    }
    return(0);
}
