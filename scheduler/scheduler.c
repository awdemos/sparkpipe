#include "sparkpipe/spark_scheduler.h"

#include <string.h>

static uint32_t SparkSchedulerNormalizeQuantizationMode(
    uint32_t quantization_mode)
{
    if (quantization_mode == SPARK_STAGE_PLAN_QUANTIZATION_AUTO)
    {
        return SPARK_STAGE_PLAN_QUANTIZATION_NVFP4_4BIT;
    }
    return quantization_mode;
}

static uint32_t SparkSchedulerQuantizationModeIsSupported(
    uint32_t quantization_mode)
{
    return quantization_mode == SPARK_STAGE_PLAN_QUANTIZATION_AUTO ||
        quantization_mode == SPARK_STAGE_PLAN_QUANTIZATION_NVFP4_4BIT ||
        quantization_mode == SPARK_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT;
}

static uint32_t SparkSchedulerMinimumU32(
    uint32_t left,
    uint32_t right)
{
    return left < right ? left : right;
}

static uint32_t SparkSchedulerMaximumU32(
    uint32_t left,
    uint32_t right)
{
    return left > right ? left : right;
}

static uint32_t SparkSchedulerRequestIsDecode(
    const SparkSchedulerRequest *request);

static uint32_t SparkGlm52SchedulerRoundDownToMultiple(
    uint32_t value,
    uint32_t multiple)
{
    if (multiple == 0u)
    {
        return value;
    }
    return value - (value % multiple);
}

static uint64_t SparkSchedulerStageCostNs(
    const uint64_t layer_cost_ns[SPARK_STAGE_PLAN_MAX_LAYER_COUNT],
    uint64_t final_stage_extra_cost_ns,
    const SparkStagePlanStage *stage)
{
    uint64_t stage_cost_ns;
    uint32_t layer_index;
    uint32_t layer_end;

    stage_cost_ns = 0u;
    layer_end = stage->first_layer_index + stage->layer_count;
    for (layer_index = stage->first_layer_index;
         layer_index < layer_end;
         ++layer_index)
    {
        stage_cost_ns += layer_cost_ns[layer_index];
    }
    if ((stage->flags & SPARK_STAGE_PLAN_STAGE_FLAG_FINAL_TOKEN) != 0u)
    {
        stage_cost_ns += final_stage_extra_cost_ns;
    }
    return stage_cost_ns;
}

static uint32_t SparkSchedulerNormalizeQueueDepthPerSpark(
    uint32_t queue_depth_per_spark)
{
    if (queue_depth_per_spark == 0u)
    {
        return SPARK_SCHEDULER_DEFAULT_QUEUE_DEPTH_PER_SPARK;
    }
    return queue_depth_per_spark;
}

static uint32_t SparkSchedulerNormalizeMeasuredProfileId(
    uint32_t measured_profile_id)
{
    if (measured_profile_id == 0u)
    {
        return SPARK_STAGE_PLAN_MEASURED_PROFILE_20260701;
    }
    return measured_profile_id;
}

static uint32_t SparkSchedulerNormalizeMaxPrefillTokensPerStep(
    uint32_t max_prefill_tokens_per_step,
    uint32_t prefix_cache_block_tokens)
{
    uint32_t normalized_token_count;

    if (max_prefill_tokens_per_step == 0u)
    {
        normalized_token_count =
            SPARK_SCHEDULER_DEFAULT_MAX_PREFILL_TOKENS_PER_STEP;
    }
    else
    {
        normalized_token_count = max_prefill_tokens_per_step;
    }
    if (normalized_token_count < prefix_cache_block_tokens)
    {
        normalized_token_count = prefix_cache_block_tokens;
    }
    return normalized_token_count;
}

static uint32_t SparkSchedulerConfigurationFlagsAreValid(
    uint32_t configuration_flags)
{
    return (configuration_flags &
        ~SPARK_SCHEDULER_CONFIGURATION_KNOWN_FLAGS) == 0u;
}

static uint32_t SparkSchedulerPromptCacheIsEnabled(
    const SparkScheduler *scheduler)
{
    return (scheduler->configuration_flags &
        SPARK_SCHEDULER_CONFIGURATION_FLAG_PREFIX_CACHE) != 0u;
}

static uint32_t SparkSchedulerCrossSequencePrefixReuseIsEnabled(
    const SparkScheduler *scheduler)
{
    return (scheduler->configuration_flags &
        SPARK_SCHEDULER_CONFIGURATION_FLAG_CROSS_SEQUENCE_PREFIX_REUSE) != 0u;
}

static uint32_t SparkSchedulerChunkedPrefillIsEnabled(
    const SparkScheduler *scheduler)
{
    return (scheduler->configuration_flags &
        SPARK_SCHEDULER_CONFIGURATION_FLAG_CHUNKED_PREFILL) != 0u;
}

static uint32_t SparkSchedulerCudaGraphPaddingIsEnabled(
    const SparkScheduler *scheduler)
{
    return (scheduler->configuration_flags &
        SPARK_SCHEDULER_CONFIGURATION_FLAG_CUDAGRAPH_PADDING) != 0u;
}

static uint32_t SparkSchedulerMeasuredDecodeBucketSelectionIsEnabled(
    const SparkScheduler *scheduler)
{
    return (scheduler->configuration_flags &
        SPARK_SCHEDULER_CONFIGURATION_FLAG_MEASURED_DECODE_BUCKET_SELECTION) != 0u;
}

static SparkStatus SparkSchedulerBuildMeasuredPlanAndCosts(
    const SparkScheduler *scheduler,
    uint32_t batch_bucket,
    SparkStagePlan *stage_plan,
    uint64_t layer_cost_ns[SPARK_STAGE_PLAN_MAX_LAYER_COUNT],
    uint64_t *final_stage_extra_cost_ns_out)
{
    SparkStatus status;

    if (scheduler == 0 || stage_plan == 0 || layer_cost_ns == 0 ||
        final_stage_extra_cost_ns_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    status = SparkStagePlanBuildCurrentSparkMeasuredBalancedForQuantization(
        &(const SparkStagePlanGeometry){SPARK_GLM52_MODEL_LAYER_COUNT, SPARK_GLM52_MODEL_FIRST_ROUTED_LAYER},
        scheduler->measured_profile_id,
        batch_bucket,
        scheduler->quantization_mode,
        stage_plan,
        0,
        0u);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    return SparkStagePlanLoadMeasuredCostProfileForQuantization(
        &(const SparkStagePlanGeometry){SPARK_GLM52_MODEL_LAYER_COUNT, SPARK_GLM52_MODEL_FIRST_ROUTED_LAYER},
        scheduler->measured_profile_id,
        batch_bucket,
        scheduler->quantization_mode,
        layer_cost_ns,
        final_stage_extra_cost_ns_out);
}

static uint64_t SparkSchedulerPlanCriticalPathNs(
    const SparkStagePlan *stage_plan,
    const uint64_t layer_cost_ns[SPARK_STAGE_PLAN_MAX_LAYER_COUNT],
    uint64_t final_stage_extra_cost_ns,
    uint32_t prefill_block_count)
{
    uint64_t critical_path_ns;
    uint32_t stage_index;

    if (prefill_block_count == 0u)
    {
        prefill_block_count = 1u;
    }

    critical_path_ns = 0u;
    for (stage_index = 0u;
         stage_index < stage_plan->stage_count;
         ++stage_index)
    {
        uint64_t stage_service_time_ns;

        stage_service_time_ns = SparkSchedulerStageCostNs(
            layer_cost_ns,
            final_stage_extra_cost_ns,
            &stage_plan->stages[stage_index]) * (uint64_t)prefill_block_count;
        if (stage_service_time_ns > critical_path_ns)
        {
            critical_path_ns = stage_service_time_ns;
        }
    }
    return critical_path_ns;
}

static SparkStatus SparkSchedulerSelectDecodeBatchBucket(
    const SparkScheduler *scheduler,
    uint32_t active_sequence_count,
    uint32_t *batch_bucket_out,
    uint32_t *minimal_batch_bucket_out)
{
    static const uint32_t candidate_buckets[] = {
        SPARK_STAGE_PLAN_BATCH_BUCKETS
    };
    uint64_t best_critical_path_ns;
    uint32_t best_bucket;
    uint32_t minimal_bucket;
    uint32_t candidate_index;
    SparkStatus status;

    if (batch_bucket_out == 0 || minimal_batch_bucket_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    status = SparkStagePlanSelectBatchBucket(
        active_sequence_count,
        &minimal_bucket);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    *minimal_batch_bucket_out = minimal_bucket;
    if (!SparkSchedulerMeasuredDecodeBucketSelectionIsEnabled(scheduler) ||
        active_sequence_count <= SPARK_STAGE_PLAN_BUCKET_B16)
    {
        *batch_bucket_out = minimal_bucket;
        return SPARK_STATUS_OK;
    }

    best_bucket = minimal_bucket;
    best_critical_path_ns = UINT64_MAX;
    for (candidate_index = 0u;
         candidate_index < (uint32_t)(sizeof(candidate_buckets) / sizeof(candidate_buckets[0]));
         ++candidate_index)
    {
        SparkStagePlan candidate_stage_plan;
        uint64_t candidate_layer_cost_ns[SPARK_STAGE_PLAN_MAX_LAYER_COUNT];
        uint64_t candidate_final_stage_extra_cost_ns;
        uint64_t candidate_critical_path_ns;
        uint32_t candidate_bucket;

        candidate_bucket = candidate_buckets[candidate_index];
        if (candidate_bucket < active_sequence_count)
        {
            continue;
        }

        status = SparkSchedulerBuildMeasuredPlanAndCosts(
            scheduler,
            candidate_bucket,
            &candidate_stage_plan,
            candidate_layer_cost_ns,
            &candidate_final_stage_extra_cost_ns);
        if (status != SPARK_STATUS_OK)
        {
            continue;
        }

        candidate_critical_path_ns = SparkSchedulerPlanCriticalPathNs(
            &candidate_stage_plan,
            candidate_layer_cost_ns,
            candidate_final_stage_extra_cost_ns,
            1u);
        if (candidate_critical_path_ns < best_critical_path_ns ||
            (candidate_critical_path_ns == best_critical_path_ns &&
             candidate_bucket < best_bucket))
        {
            best_bucket = candidate_bucket;
            best_critical_path_ns = candidate_critical_path_ns;
        }
    }

    *batch_bucket_out = best_bucket;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkSchedulerSelectRequestBatchBucket(
    const SparkScheduler *scheduler,
    const SparkSchedulerRequest *request,
    uint32_t *batch_bucket_out,
    uint32_t *minimal_batch_bucket_out)
{
    if (request == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkSchedulerRequestIsDecode(request))
    {
        return SparkSchedulerSelectDecodeBatchBucket(
            scheduler,
            request->active_sequence_count,
            batch_bucket_out,
            minimal_batch_bucket_out);
    }
    if (minimal_batch_bucket_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SparkStagePlanSelectBatchBucket(
        request->active_sequence_count,
        batch_bucket_out) == SPARK_STATUS_OK
            ? ((*minimal_batch_bucket_out = *batch_bucket_out), SPARK_STATUS_OK)
            : SPARK_STATUS_CAPACITY_EXCEEDED;
}

static uint32_t SparkSchedulerPrefillDecodeInterleaveIsEnabled(
    const SparkScheduler *scheduler)
{
    return (scheduler->configuration_flags &
        SPARK_SCHEDULER_CONFIGURATION_FLAG_PREFILL_DECODE_INTERLEAVE) != 0u;
}

void SparkSchedulerSetPrefillDemand(
    SparkScheduler *scheduler,
    uint32_t prefill_demand)
{
    if (scheduler == 0)
        return;
    scheduler->prefill_demand = prefill_demand != 0u ? 1u : 0u;
}

static uint32_t SparkSchedulerStageHasCapacity(
    const SparkScheduler *scheduler,
    uint32_t spark_index,
    uint32_t request_is_prefill)
{
    uint32_t reserved_slot;

    if (spark_index >= scheduler->spark_count ||
        scheduler->spark_inflight_counts[spark_index] >=
            scheduler->queue_depth_per_spark)
    {
        return 0u;
    }

    reserved_slot = SparkSchedulerPrefillDecodeInterleaveIsEnabled(
        scheduler) &&
        scheduler->queue_depth_per_spark > 1u &&
        (request_is_prefill != 0u ||
         scheduler->prefill_demand != 0u)
        ? 1u
        : 0u;
    if (reserved_slot != 0u &&
        scheduler->spark_inflight_counts[spark_index] >=
            scheduler->queue_depth_per_spark - reserved_slot)
    {
        return 0u;
    }
    return 1u;
}

static uint32_t SparkSchedulerDecodeBypassIsActive(
    const SparkScheduler *scheduler)
{
    uint32_t spark_index;

    if (!SparkSchedulerPrefillDecodeInterleaveIsEnabled(scheduler))
    {
        return 0u;
    }
    for (spark_index = 0u; spark_index < scheduler->spark_count; ++spark_index)
    {
        if (scheduler->spark_inflight_counts[spark_index] != 0u)
        {
            return 1u;
        }
    }
    return 0u;
}

static uint32_t SparkSchedulerRequestIsPrefill(
    const SparkSchedulerRequest *request)
{
    return (request->flags & SPARK_SCHEDULER_REQUEST_FLAG_PREFILL) != 0u;
}

static uint32_t SparkSchedulerRequestIsDecode(
    const SparkSchedulerRequest *request)
{
    return (request->flags & SPARK_SCHEDULER_REQUEST_FLAG_DECODE) != 0u;
}

static SparkStatus SparkSchedulerLookupCachedPrefixTokenCount(
    SparkScheduler *scheduler,
    const SparkSchedulerRequest *request,
    uint32_t *cached_prefix_token_count_out)
{
    SparkPrefixCacheLookup lookup;
    SparkStatus status;

    if (cached_prefix_token_count_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *cached_prefix_token_count_out = 0u;
    if (SparkSchedulerRequestIsPrefill(request) == 0u ||
        SparkSchedulerPromptCacheIsEnabled(scheduler) == 0u ||
        SparkSchedulerCrossSequencePrefixReuseIsEnabled(scheduler) == 0u)
    {
        return SPARK_STATUS_OK;
    }
    status = SparkPrefixCacheProbePrompt(
        scheduler->prefix_cache,
        request->sequence_id,
        request->prompt_token_ids,
        request->prompt_token_count,
        &lookup);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    *cached_prefix_token_count_out = lookup.matched_token_count;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkSchedulerReservePrompt(
    SparkScheduler *scheduler,
    const SparkSchedulerRequest *request,
    uint32_t token_count,
    SparkPrefixCacheReservation *reservation)
{
    if (SparkSchedulerCrossSequencePrefixReuseIsEnabled(scheduler) != 0u)
        return SparkPrefixCacheReservePrompt(
            scheduler->prefix_cache,request->sequence_id,
            request->prompt_token_ids,token_count,reservation);
    return SparkPrefixCacheReserveSequencePrompt(
        scheduler->prefix_cache,request->sequence_id,
        request->prompt_token_ids,token_count,reservation);
}

static uint32_t SparkSchedulerEffectiveComputedPromptTokenCount(
    const SparkSchedulerRequest *request,
    uint32_t cached_prefix_token_count)
{
    return SparkSchedulerMaximumU32(
        request->computed_prompt_token_count,
        cached_prefix_token_count);
}

static uint32_t SparkSchedulerRequestMaxPrefillTokensPerStep(
    const SparkScheduler *scheduler,
    const SparkSchedulerRequest *request)
{
    uint32_t max_prefill_tokens_per_step;

    max_prefill_tokens_per_step = request->max_scheduled_prompt_token_count;
    if (max_prefill_tokens_per_step == 0u ||
        max_prefill_tokens_per_step > scheduler->max_prefill_tokens_per_step)
    {
        max_prefill_tokens_per_step = scheduler->max_prefill_tokens_per_step;
    }
    if (max_prefill_tokens_per_step < scheduler->prefix_cache_block_tokens)
    {
        max_prefill_tokens_per_step = scheduler->prefix_cache_block_tokens;
    }
    return max_prefill_tokens_per_step;
}

static uint32_t SparkSchedulerScheduledPrefillTokenCount(
    const SparkScheduler *scheduler,
    const SparkSchedulerRequest *request,
    uint32_t computed_prompt_token_count)
{
    uint32_t remaining_prompt_token_count;
    uint32_t max_prefill_tokens_per_step;
    uint32_t scheduled_prompt_token_count;

    remaining_prompt_token_count =
        request->prompt_token_count - computed_prompt_token_count;
    if (!SparkSchedulerChunkedPrefillIsEnabled(scheduler))
    {
        return remaining_prompt_token_count;
    }

    max_prefill_tokens_per_step =
        SparkSchedulerRequestMaxPrefillTokensPerStep(scheduler, request);
    if (remaining_prompt_token_count <= max_prefill_tokens_per_step)
    {
        return remaining_prompt_token_count;
    }

    scheduled_prompt_token_count = SparkGlm52SchedulerRoundDownToMultiple(
        max_prefill_tokens_per_step,
        scheduler->prefix_cache_block_tokens);
    if (scheduled_prompt_token_count == 0u)
    {
        scheduled_prompt_token_count = SparkSchedulerMinimumU32(
            remaining_prompt_token_count,
            scheduler->prefix_cache_block_tokens);
    }
    return scheduled_prompt_token_count;
}

static uint32_t SparkSchedulerPrefillBlockCount(
    const SparkScheduler *scheduler,
    uint32_t prompt_token_count)
{
    if (prompt_token_count == 0u)
    {
        return 1u;
    }
    return SparkCeilDivU32(
        prompt_token_count,
        scheduler->prefix_cache_block_tokens);
}

static uint32_t SparkSchedulerBuildDecisionFlags(
    const SparkScheduler *scheduler,
    const SparkSchedulerRequest *request,
    uint32_t batch_bucket,
    uint32_t cached_prefix_token_count,
    uint32_t scheduled_prompt_token_count,
    uint32_t remaining_prompt_token_count_after_step,
    uint32_t decode_bypass_active)
{
    uint32_t decision_flags;

    decision_flags = 0u;
    if (SparkSchedulerRequestIsDecode(request))
    {
        decision_flags |= SPARK_SCHEDULER_DECISION_FLAG_DECODE_STEP;
        if (decode_bypass_active != 0u)
        {
            decision_flags |=
                SPARK_SCHEDULER_DECISION_FLAG_DECODE_BYPASS_PREFILL;
        }
    }
    if (SparkSchedulerRequestIsPrefill(request))
    {
        decision_flags |= SPARK_SCHEDULER_DECISION_FLAG_PREFILL_STEP;
        if (SparkSchedulerPrefillDecodeInterleaveIsEnabled(scheduler) &&
            scheduler->queue_depth_per_spark > 1u)
        {
            decision_flags |=
                SPARK_SCHEDULER_DECISION_FLAG_PREFILL_RESERVED_DECODE_SLOT;
        }
        if (remaining_prompt_token_count_after_step == 0u)
        {
            decision_flags |=
                SPARK_SCHEDULER_DECISION_FLAG_PREFILL_FINAL_CHUNK;
        }
        if (scheduled_prompt_token_count != 0u &&
            remaining_prompt_token_count_after_step != 0u)
        {
            decision_flags |= SPARK_SCHEDULER_DECISION_FLAG_PREFILL_CHUNK;
        }
        if (cached_prefix_token_count != 0u)
        {
            decision_flags |=
                SPARK_SCHEDULER_DECISION_FLAG_PREFIX_CACHE_USED;
        }
    }
    if (SparkSchedulerCudaGraphPaddingIsEnabled(scheduler) &&
        request->active_sequence_count < batch_bucket)
    {
        decision_flags |= SPARK_SCHEDULER_DECISION_FLAG_CUDAGRAPH_PADDING;
    }
    return decision_flags;
}

static uint32_t SparkSchedulerBuildDispatchFlags(
    uint32_t decision_flags)
{
    uint32_t dispatch_flags;

    dispatch_flags = 0u;
    if ((decision_flags & SPARK_SCHEDULER_DECISION_FLAG_DECODE_STEP) != 0u)
    {
        dispatch_flags |= SPARK_SCHEDULER_DISPATCH_STAGE_FLAG_DECODE;
    }
    if ((decision_flags & SPARK_SCHEDULER_DECISION_FLAG_PREFILL_STEP) != 0u)
    {
        dispatch_flags |= SPARK_SCHEDULER_DISPATCH_STAGE_FLAG_PREFILL;
    }
    if ((decision_flags & SPARK_SCHEDULER_DECISION_FLAG_PREFILL_CHUNK) != 0u)
    {
        dispatch_flags |= SPARK_SCHEDULER_DISPATCH_STAGE_FLAG_PREFILL_CHUNK;
    }
    if ((decision_flags &
         SPARK_SCHEDULER_DECISION_FLAG_PREFILL_FINAL_CHUNK) != 0u)
    {
        dispatch_flags |=
            SPARK_SCHEDULER_DISPATCH_STAGE_FLAG_PREFILL_FINAL_CHUNK;
    }
    if ((decision_flags &
         SPARK_SCHEDULER_DECISION_FLAG_CUDAGRAPH_PADDING) != 0u)
    {
        dispatch_flags |=
            SPARK_SCHEDULER_DISPATCH_STAGE_FLAG_CUDAGRAPH_PADDING;
    }
    if ((decision_flags &
         SPARK_SCHEDULER_DECISION_FLAG_PREFILL_RESERVED_DECODE_SLOT) != 0u)
    {
        dispatch_flags |=
            SPARK_SCHEDULER_DISPATCH_STAGE_FLAG_PREFILL_RESERVED_DECODE_SLOT;
    }
    if ((decision_flags &
         SPARK_SCHEDULER_DECISION_FLAG_DECODE_BYPASS_PREFILL) != 0u)
    {
        dispatch_flags |=
            SPARK_SCHEDULER_DISPATCH_STAGE_FLAG_DECODE_BYPASS_PREFILL;
    }
    if ((decision_flags &
         SPARK_SCHEDULER_DECISION_FLAG_MEASURED_DECODE_BUCKET) != 0u)
    {
        dispatch_flags |=
            SPARK_SCHEDULER_DISPATCH_STAGE_FLAG_MEASURED_DECODE_BUCKET;
    }
    return dispatch_flags;
}

static SparkStatus SparkSchedulerReject(
    SparkScheduler *scheduler,
    SparkSchedulerDecision *decision,
    SparkStatus rejected_status)
{
    if (decision != 0)
    {
        decision->accepted = 0u;
        decision->rejected_status = rejected_status;
    }
    if (scheduler != 0)
    {
        scheduler->rejected_count += 1u;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkSchedulerValidateConfiguration(const SparkSchedulerConfiguration *configuration)
{
    if (configuration == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (configuration->abi_version != SPARK_SCHEDULER_ABI_VERSION ||
        configuration->descriptor_bytes != SPARK_SCHEDULER_CONFIGURATION_DESCRIPTOR_BYTES ||
        configuration->spark_count != SPARK_SCHEDULER_MAX_SPARK_COUNT ||
        configuration->reserved != 0u ||
        configuration->configuration_flags == 0u ||
        configuration->prefix_cache_block_tokens == 0u ||
        !SparkSchedulerQuantizationModeIsSupported(configuration->quantization_mode) ||
        !SparkSchedulerConfigurationFlagsAreValid(configuration->configuration_flags))
        return SPARK_STATUS_INVALID_ARGUMENT;
    if ((configuration->configuration_flags & SPARK_SCHEDULER_CONFIGURATION_FLAG_PREFIX_CACHE) != 0u)
    {
        if (configuration->prefix_cache == 0 ||
            configuration->prefix_cache->abi_version != SPARK_PREFIX_CACHE_ABI_VERSION ||
            configuration->prefix_cache->descriptor_bytes != SPARK_PREFIX_CACHE_DESCRIPTOR_BYTES ||
            configuration->prefix_cache->block_token_count != configuration->prefix_cache_block_tokens ||
            configuration->prefix_cache->entries == 0 ||
            configuration->prefix_cache->sequence_bindings == 0)
            return SPARK_STATUS_INVALID_ARGUMENT;
        if ((configuration->configuration_flags & SPARK_SCHEDULER_CONFIGURATION_FLAG_KV_CACHE_REQUIRED) != 0u &&
            configuration->prefix_cache->kv_cache_arena == 0)
            return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkSchedulerValidateRequest(
    const SparkScheduler *scheduler,
    const SparkSchedulerRequest *request)
{
    uint32_t is_decode;
    uint32_t is_prefill;

    if (scheduler == 0 || request == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    is_decode = SparkSchedulerRequestIsDecode(request);
    is_prefill = SparkSchedulerRequestIsPrefill(request);
    if (request->abi_version != SPARK_SCHEDULER_ABI_VERSION ||
        request->descriptor_bytes != SPARK_SCHEDULER_REQUEST_DESCRIPTOR_BYTES ||
        request->reserved != 0u ||
        (request->flags & ~SPARK_SCHEDULER_REQUEST_KNOWN_FLAGS) != 0u ||
        request->active_sequence_count == 0u ||
        is_decode == is_prefill ||
        (is_decode && request->prompt_token_count != 0u) ||
        (is_decode && request->computed_prompt_token_count != 0u) ||
        (is_decode && request->cached_prefix_token_count != 0u) ||
        (is_decode && request->max_scheduled_prompt_token_count != 0u) ||
        (is_decode && request->sequence_id != 0u) ||
        (is_decode && request->prompt_token_ids != 0) ||
        (is_prefill && request->prompt_token_count == 0u) ||
        (is_prefill && request->cached_prefix_token_count != 0u) ||
        (is_prefill &&
         SparkSchedulerPromptCacheIsEnabled(scheduler) &&
         (request->sequence_id == 0u || request->prompt_token_ids == 0)) ||
        (is_prefill &&
         SparkSchedulerPromptCacheIsEnabled(scheduler) &&
         SparkSchedulerCrossSequencePrefixReuseIsEnabled(scheduler) &&
         request->computed_prompt_token_count != 0u) ||
        (is_prefill &&
         request->computed_prompt_token_count >= request->prompt_token_count))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkSchedulerInitialize(
    SparkScheduler *scheduler,
    const SparkSchedulerConfiguration *configuration)
{
    uint32_t queue_depth_per_spark;
    uint32_t measured_profile_id;
    uint32_t quantization_mode;
    uint32_t prefix_cache_block_tokens;
    uint32_t max_prefill_tokens_per_step;
    uint32_t configuration_flags;
    SparkStatus status;

    if (scheduler == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkSchedulerValidateConfiguration(configuration);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    queue_depth_per_spark = SparkSchedulerNormalizeQueueDepthPerSpark(
        configuration->queue_depth_per_spark);
    measured_profile_id = SparkSchedulerNormalizeMeasuredProfileId(
        configuration->measured_profile_id);
    quantization_mode = SparkSchedulerNormalizeQuantizationMode(
        configuration->quantization_mode);
    configuration_flags = configuration->configuration_flags;
    prefix_cache_block_tokens = configuration->prefix_cache_block_tokens;
    max_prefill_tokens_per_step =
        SparkSchedulerNormalizeMaxPrefillTokensPerStep(
            configuration->max_prefill_tokens_per_step,
            prefix_cache_block_tokens);

    memset(scheduler, 0, sizeof(*scheduler));
    scheduler->abi_version = SPARK_SCHEDULER_ABI_VERSION;
    scheduler->descriptor_bytes = SPARK_SCHEDULER_DESCRIPTOR_BYTES;
    scheduler->spark_count = SPARK_SCHEDULER_MAX_SPARK_COUNT;
    scheduler->queue_depth_per_spark = queue_depth_per_spark;
    scheduler->measured_profile_id = measured_profile_id;
    scheduler->quantization_mode = quantization_mode;
    scheduler->max_prefill_tokens_per_step = max_prefill_tokens_per_step;
    scheduler->prefix_cache_block_tokens = prefix_cache_block_tokens;
    scheduler->configuration_flags = configuration_flags;
    scheduler->prefix_cache = configuration->prefix_cache;
    return SPARK_STATUS_OK;
}

uint32_t SparkSchedulerSelectPipelineBatchWidth(
    const SparkScheduler *scheduler,
    uint32_t active_request_count,
    uint32_t batch_capacity)
{
    uint32_t batch_width;

    if (scheduler == 0 || scheduler->spark_count == 0u ||
        active_request_count == 0u || batch_capacity == 0u)
    {
        return 0u;
    }
    batch_width = active_request_count / scheduler->spark_count;
    if (active_request_count % scheduler->spark_count != 0u)
    {
        batch_width += 1u;
    }
    if (batch_width > batch_capacity)
    {
        batch_width = batch_capacity;
    }
    return batch_width;
}

static SparkStatus SparkSchedulerEstimateDecodeChunkNs(
    const SparkScheduler *scheduler,
    uint32_t execution_row_count,
    uint64_t *estimated_work_ns_out)
{
    SparkStagePlan stage_plan;
    uint64_t layer_cost_ns[SPARK_STAGE_PLAN_MAX_LAYER_COUNT];
    uint64_t final_stage_extra_cost_ns;
    uint32_t batch_bucket;
    uint32_t minimal_batch_bucket;
    SparkStatus status;

    status = SparkSchedulerSelectDecodeBatchBucket(
        scheduler,
        execution_row_count,
        &batch_bucket,
        &minimal_batch_bucket);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkSchedulerBuildMeasuredPlanAndCosts(
        scheduler,
        batch_bucket,
        &stage_plan,
        layer_cost_ns,
        &final_stage_extra_cost_ns);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    *estimated_work_ns_out = SparkSchedulerPlanCriticalPathNs(
        &stage_plan,
        layer_cost_ns,
        final_stage_extra_cost_ns,
        1u);
    return *estimated_work_ns_out != 0u
        ? SPARK_STATUS_OK
        : SPARK_STATUS_MODULE_NOT_VALIDATED;
}

static SparkStatus SparkSchedulerSumDecodeChunkWorkNs(
    const SparkScheduler *scheduler,
    uint32_t logical_sequence_count,
    uint32_t rows_per_sequence,
    uint32_t maximum_sequences_per_chunk,
    uint32_t chunk_count,
    uint64_t *estimated_work_ns_out)
{
    uint64_t chunk_work_ns;
    uint64_t total_work_ns;
    uint32_t chunk_index;
    uint32_t chunk_sequence_count;
    uint32_t remaining_sequence_count;
    SparkStatus status;

    total_work_ns = 0u;
    remaining_sequence_count = logical_sequence_count;
    for (chunk_index = 0u; chunk_index < chunk_count; ++chunk_index)
    {
        chunk_sequence_count =
            remaining_sequence_count < maximum_sequences_per_chunk
            ? remaining_sequence_count
            : maximum_sequences_per_chunk;
        status = SparkSchedulerEstimateDecodeChunkNs(
            scheduler,
            chunk_sequence_count * rows_per_sequence,
            &chunk_work_ns);
        if (status != SPARK_STATUS_OK ||
            UINT64_MAX - total_work_ns < chunk_work_ns)
        {
            return status != SPARK_STATUS_OK
                ? status
                : SPARK_STATUS_CAPACITY_EXCEEDED;
        }
        total_work_ns += chunk_work_ns;
        remaining_sequence_count -= chunk_sequence_count;
    }
    if (remaining_sequence_count != 0u)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    *estimated_work_ns_out = total_work_ns;
    return SPARK_STATUS_OK;
}

SparkStatus SparkSchedulerEstimateDecodeWorkNs(
    const SparkScheduler *scheduler,
    uint32_t logical_sequence_count,
    uint32_t rows_per_sequence,
    uint32_t execution_row_capacity,
    uint64_t *estimated_work_ns_out)
{
    uint32_t chunk_count;
    uint32_t maximum_sequences_per_chunk;
    SparkStatus status;

    if (scheduler == 0 || estimated_work_ns_out == 0 ||
        scheduler->abi_version != SPARK_SCHEDULER_ABI_VERSION ||
        scheduler->descriptor_bytes != SPARK_SCHEDULER_DESCRIPTOR_BYTES)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkStagePlanExecutionChunkShape(
        logical_sequence_count,
        rows_per_sequence,
        execution_row_capacity,
        &maximum_sequences_per_chunk,
        &chunk_count);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    return SparkSchedulerSumDecodeChunkWorkNs(
        scheduler,
        logical_sequence_count,
        rows_per_sequence,
        maximum_sequences_per_chunk,
        chunk_count,
        estimated_work_ns_out);
}

SparkStatus SparkSchedulerAdmit(
    SparkScheduler *scheduler,
    const SparkSchedulerRequest *request,
    SparkSchedulerDecision *decision)
{
    uint64_t layer_cost_ns[SPARK_STAGE_PLAN_MAX_LAYER_COUNT];
    uint64_t final_stage_extra_cost_ns;
    uint64_t stage_cost_ns;
    uint64_t stage_service_time_ns;
    uint32_t batch_bucket;
    uint32_t minimal_batch_bucket;
    uint32_t measured_decode_bucket_selected;
    uint32_t stage_index;
    uint32_t prefill_block_count;
    uint32_t cached_prefix_token_count;
    uint32_t computed_prompt_token_count;
    uint32_t scheduled_prompt_token_count;
    uint32_t remaining_prompt_token_count_after_step;
    uint32_t graph_sequence_padding_count;
    uint32_t decision_flags;
    uint32_t decode_bypass_active;
    uint32_t dispatch_flags;
    SparkPrefixCacheReservation prefix_cache_reservation;
    SparkPrefixCachePromptHash prefix_cache_parent_hash;
    SparkStatus status;

    if (scheduler == 0 || request == 0 || decision == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (scheduler->abi_version != SPARK_SCHEDULER_ABI_VERSION ||
        scheduler->descriptor_bytes != SPARK_SCHEDULER_DESCRIPTOR_BYTES)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    status = SparkSchedulerValidateRequest(scheduler, request);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    memset(decision, 0, sizeof(*decision));
    decision->abi_version = SPARK_SCHEDULER_ABI_VERSION;
    decision->descriptor_bytes = SPARK_SCHEDULER_DECISION_DESCRIPTOR_BYTES;
    decision->quantization_mode = scheduler->quantization_mode;
    decision->spark_count = scheduler->spark_count;

    minimal_batch_bucket = 0u;
    status = SparkSchedulerSelectRequestBatchBucket(
        scheduler,
        request,
        &batch_bucket,
        &minimal_batch_bucket);
    if (status != SPARK_STATUS_OK)
    {
        return SparkSchedulerReject(scheduler, decision, status);
    }
    measured_decode_bucket_selected = SparkSchedulerRequestIsDecode(request) &&
        batch_bucket != minimal_batch_bucket;

    status = SparkSchedulerBuildMeasuredPlanAndCosts(
        scheduler,
        batch_bucket,
        &decision->stage_plan,
        layer_cost_ns,
        &final_stage_extra_cost_ns);
    if (status != SPARK_STATUS_OK)
    {
        return SparkSchedulerReject(scheduler, decision, status);
    }

    for (stage_index = 0u;
         stage_index < decision->stage_plan.stage_count;
         ++stage_index)
    {
        if (!SparkSchedulerStageHasCapacity(
                scheduler,
                stage_index,
                SparkSchedulerRequestIsPrefill(request)))
        {
            return SparkSchedulerReject(
                scheduler,
                decision,
                SPARK_STATUS_BUSY);
        }
    }

    cached_prefix_token_count = 0u;
    computed_prompt_token_count = 0u;
    scheduled_prompt_token_count = 0u;
    remaining_prompt_token_count_after_step = 0u;
    prefill_block_count = 1u;
    if (SparkSchedulerRequestIsPrefill(request))
    {
        status = SparkSchedulerLookupCachedPrefixTokenCount(
            scheduler,
            request,
            &cached_prefix_token_count);
        if (status != SPARK_STATUS_OK)
        {
            return SparkSchedulerReject(scheduler, decision, status);
        }
        computed_prompt_token_count =
            SparkSchedulerEffectiveComputedPromptTokenCount(
                request,
                cached_prefix_token_count);
        scheduled_prompt_token_count =
            SparkSchedulerScheduledPrefillTokenCount(
                scheduler,
                request,
                computed_prompt_token_count);
        remaining_prompt_token_count_after_step =
            request->prompt_token_count - computed_prompt_token_count -
            scheduled_prompt_token_count;
        prefill_block_count = SparkSchedulerPrefillBlockCount(
            scheduler,
            scheduled_prompt_token_count);
        if (SparkSchedulerPromptCacheIsEnabled(scheduler))
        {
            memset(&prefix_cache_reservation, 0, sizeof(prefix_cache_reservation));
            prefix_cache_reservation.abi_version =
                SPARK_PREFIX_CACHE_ABI_VERSION;
            prefix_cache_reservation.descriptor_bytes =
                SPARK_PREFIX_CACHE_RESERVATION_DESCRIPTOR_BYTES;
            prefix_cache_reservation.physical_block_indices =
                decision->kv_physical_block_indices;
            prefix_cache_reservation.physical_block_capacity =
                SPARK_SCHEDULER_KV_BLOCK_TABLE_CAPACITY;
            status = SparkSchedulerReservePrompt(
                scheduler,
                request,
                computed_prompt_token_count + scheduled_prompt_token_count,
                &prefix_cache_reservation);
            if (status != SPARK_STATUS_OK)
            {
                return SparkSchedulerReject(scheduler, decision, status);
            }
        }
    }

    graph_sequence_padding_count = batch_bucket - request->active_sequence_count;
    decode_bypass_active = SparkSchedulerRequestIsDecode(request)
        ? SparkSchedulerDecodeBypassIsActive(scheduler)
        : 0u;
    decision_flags = SparkSchedulerBuildDecisionFlags(
        scheduler,
        request,
        batch_bucket,
        cached_prefix_token_count,
        scheduled_prompt_token_count,
        remaining_prompt_token_count_after_step,
        decode_bypass_active);
    if (measured_decode_bucket_selected != 0u)
    {
        decision_flags |= SPARK_SCHEDULER_DECISION_FLAG_MEASURED_DECODE_BUCKET;
    }
    dispatch_flags = SparkSchedulerBuildDispatchFlags(decision_flags);

    decision->accepted = 1u;
    decision->batch_bucket = batch_bucket;
    decision->stage_count = decision->stage_plan.stage_count;
    decision->rejected_status = SPARK_STATUS_OK;
    decision->decision_flags = decision_flags;
    decision->active_sequence_count = request->active_sequence_count;
    decision->graph_sequence_capacity = batch_bucket;
    decision->graph_sequence_padding_count = graph_sequence_padding_count;
    decision->prompt_token_count = request->prompt_token_count;
    decision->computed_prompt_token_count = computed_prompt_token_count;
    decision->cached_prefix_token_count = cached_prefix_token_count;
    decision->prefix_cache_block_count = cached_prefix_token_count /
        scheduler->prefix_cache_block_tokens;
    decision->scheduled_prompt_token_offset = computed_prompt_token_count;
    decision->scheduled_prompt_token_count = scheduled_prompt_token_count;
    decision->remaining_prompt_token_count_after_step =
        remaining_prompt_token_count_after_step;
    decision->prefill_block_count = prefill_block_count;
    decision->cache_commit_token_count_after_step =
        computed_prompt_token_count + scheduled_prompt_token_count;
    if (SparkSchedulerRequestIsPrefill(request) &&
        SparkSchedulerPromptCacheIsEnabled(scheduler))
    {
        decision->kv_block_token_count = scheduler->prefix_cache_block_tokens;
        decision->kv_physical_block_count =
            prefix_cache_reservation.physical_block_count;
        decision->kv_cached_physical_block_count =
            prefix_cache_reservation.cached_physical_block_count;
        decision->kv_pending_physical_block_count =
            prefix_cache_reservation.pending_physical_block_count;
        decision->kv_block_table_token_count =
            prefix_cache_reservation.reserved_token_count;
        decision->prefix_cache_reservation_epoch =
            prefix_cache_reservation.reservation_epoch;
        decision->prefix_cache_result_hash =
            prefix_cache_reservation.last_block_hash;
        if (computed_prompt_token_count == 0u)
        {
            decision->prefix_cache_parent_hash =
                SPARK_PREFIX_CACHE_EMPTY_PARENT_HASH;
        }
        else
        {
            status = SparkPrefixCacheHashPromptTokens(
                scheduler->prefix_cache_block_tokens,
                SPARK_PREFIX_CACHE_EMPTY_PARENT_HASH,
                request->prompt_token_ids,
                computed_prompt_token_count,
                &prefix_cache_parent_hash);
            if (status != SPARK_STATUS_OK)
            {
                SparkPrefixCacheCancelReservation(
                    scheduler->prefix_cache,
                    request->sequence_id,
                    prefix_cache_reservation.reservation_epoch);
                return SparkSchedulerReject(scheduler, decision, status);
            }
            decision->prefix_cache_parent_hash =
                prefix_cache_parent_hash.prompt_hash;
        }
    }
    decision->sequence_id = request->sequence_id;
    decision->prompt_token_ids = request->prompt_token_ids;
    if (SparkSchedulerRequestIsPrefill(request))
    {
        decision->total_scheduled_token_count =
            (uint64_t)request->active_sequence_count *
            (uint64_t)scheduled_prompt_token_count;
    }
    else
    {
        decision->total_scheduled_token_count =
            (uint64_t)request->active_sequence_count;
    }
    decision->estimated_critical_path_ns = 0u;

    for (stage_index = 0u;
         stage_index < decision->stage_plan.stage_count;
         ++stage_index)
    {
        stage_cost_ns = SparkSchedulerStageCostNs(
            layer_cost_ns,
            final_stage_extra_cost_ns,
            &decision->stage_plan.stages[stage_index]);
        stage_service_time_ns = stage_cost_ns * (uint64_t)prefill_block_count;
        decision->dispatch_stages[stage_index].spark_index = stage_index;
        decision->dispatch_stages[stage_index].batch_bucket = batch_bucket;
        decision->dispatch_stages[stage_index].first_layer_index =
            decision->stage_plan.stages[stage_index].first_layer_index;
        decision->dispatch_stages[stage_index].layer_count =
            decision->stage_plan.stages[stage_index].layer_count;
        decision->dispatch_stages[stage_index].stage_flags =
            decision->stage_plan.stages[stage_index].flags;
        decision->dispatch_stages[stage_index].dispatch_flags = dispatch_flags;
        decision->dispatch_stages[stage_index].active_sequence_count =
            request->active_sequence_count;
        decision->dispatch_stages[stage_index].graph_sequence_capacity =
            batch_bucket;
        decision->dispatch_stages[stage_index].graph_sequence_padding_count =
            graph_sequence_padding_count;
        decision->dispatch_stages[stage_index].scheduled_prompt_token_offset =
            computed_prompt_token_count;
        decision->dispatch_stages[stage_index].scheduled_prompt_token_count =
            scheduled_prompt_token_count;
        decision->dispatch_stages[stage_index].cached_prefix_token_count =
            cached_prefix_token_count;
        decision->dispatch_stages[stage_index].estimated_service_time_ns =
            stage_service_time_ns;
        if (stage_service_time_ns > decision->estimated_critical_path_ns)
        {
            decision->estimated_critical_path_ns = stage_service_time_ns;
        }
        scheduler->spark_inflight_counts[stage_index] += 1u;
    }

    scheduler->admitted_count += 1u;
    if (SparkSchedulerRequestIsPrefill(request))
    {
        scheduler->scheduled_prefill_token_count +=
            decision->total_scheduled_token_count;
        scheduler->prefix_cache_hit_token_count +=
            (uint64_t)request->active_sequence_count *
            (uint64_t)cached_prefix_token_count;
        if ((decision_flags & SPARK_SCHEDULER_DECISION_FLAG_PREFILL_CHUNK) != 0u)
        {
            scheduler->chunked_prefill_count += 1u;
        }
        if ((decision_flags &
             SPARK_SCHEDULER_DECISION_FLAG_PREFILL_RESERVED_DECODE_SLOT) != 0u)
        {
            scheduler->interleaved_prefill_admission_count += 1u;
        }
        if (SparkSchedulerPromptCacheIsEnabled(scheduler))
        {
            scheduler->kv_block_reservation_count +=
                decision->kv_pending_physical_block_count;
            scheduler->kv_block_reservation_token_count +=
                decision->kv_block_table_token_count;
        }
    }
    else
    {
        scheduler->scheduled_decode_token_count +=
            decision->total_scheduled_token_count;
        if ((decision_flags &
             SPARK_SCHEDULER_DECISION_FLAG_DECODE_BYPASS_PREFILL) != 0u)
        {
            scheduler->decode_bypass_admission_count += 1u;
        }
        if ((decision_flags &
             SPARK_SCHEDULER_DECISION_FLAG_MEASURED_DECODE_BUCKET) != 0u)
        {
            scheduler->measured_decode_bucket_selection_count += 1u;
            scheduler->measured_decode_bucket_padding_token_count +=
                decision->graph_sequence_padding_count;
        }
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkSchedulerComplete(
    SparkScheduler *scheduler,
    const SparkSchedulerDecision *decision)
{
    uint32_t stage_index;
    uint32_t spark_index;

    if (scheduler == 0 || decision == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (scheduler->abi_version != SPARK_SCHEDULER_ABI_VERSION ||
        scheduler->descriptor_bytes != SPARK_SCHEDULER_DESCRIPTOR_BYTES ||
        decision->abi_version != SPARK_SCHEDULER_ABI_VERSION ||
        decision->descriptor_bytes != SPARK_SCHEDULER_DECISION_DESCRIPTOR_BYTES ||
        decision->accepted == 0u ||
        decision->stage_count == 0u ||
        decision->stage_count > scheduler->spark_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    for (stage_index = 0u; stage_index < decision->stage_count; ++stage_index)
    {
        spark_index = decision->dispatch_stages[stage_index].spark_index;
        if (spark_index >= scheduler->spark_count ||
            scheduler->spark_inflight_counts[spark_index] == 0u)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
    }

    for (stage_index = 0u; stage_index < decision->stage_count; ++stage_index)
    {
        spark_index = decision->dispatch_stages[stage_index].spark_index;
        scheduler->spark_inflight_counts[spark_index] -= 1u;
    }
    if ((decision->decision_flags &
            SPARK_SCHEDULER_DECISION_FLAG_PREFILL_STEP) != 0u &&
        SparkSchedulerPromptCacheIsEnabled(scheduler) &&
        decision->cache_commit_token_count_after_step != 0u)
    {
        SparkStatus status;

        status = SparkPrefixCacheCommitReservation(
            scheduler->prefix_cache,
            decision->sequence_id,
            decision->prefix_cache_reservation_epoch);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }
    scheduler->completed_count += 1u;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkSchedulerValidateAcceptedDecision(
    const SparkScheduler *scheduler,
    const SparkSchedulerDecision *decision)
{
    if (scheduler == 0 || decision == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (scheduler->abi_version != SPARK_SCHEDULER_ABI_VERSION ||
        scheduler->descriptor_bytes != SPARK_SCHEDULER_DESCRIPTOR_BYTES ||
        decision->abi_version != SPARK_SCHEDULER_ABI_VERSION ||
        decision->descriptor_bytes != SPARK_SCHEDULER_DECISION_DESCRIPTOR_BYTES ||
        decision->accepted == 0u ||
        decision->stage_count == 0u ||
        decision->stage_count > scheduler->spark_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkSchedulerReleaseDecisionInflight(
    SparkScheduler *scheduler,
    const SparkSchedulerDecision *decision)
{
    uint32_t stage_index;

    for (stage_index = 0u; stage_index < decision->stage_count; ++stage_index)
    {
        uint32_t spark_index;

        spark_index = decision->dispatch_stages[stage_index].spark_index;
        if (spark_index >= scheduler->spark_count ||
            scheduler->spark_inflight_counts[spark_index] == 0u)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
    }
    for (stage_index = 0u; stage_index < decision->stage_count; ++stage_index)
    {
        uint32_t spark_index;

        spark_index = decision->dispatch_stages[stage_index].spark_index;
        scheduler->spark_inflight_counts[spark_index] -= 1u;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkSchedulerCancel(
    SparkScheduler *scheduler,
    const SparkSchedulerDecision *decision)
{
    SparkStatus status;

    status = SparkSchedulerValidateAcceptedDecision(scheduler, decision);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkSchedulerReleaseDecisionInflight(scheduler, decision);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if ((decision->decision_flags &
            SPARK_SCHEDULER_DECISION_FLAG_PREFILL_STEP) != 0u &&
        SparkSchedulerPromptCacheIsEnabled(scheduler) &&
        decision->prefix_cache_reservation_epoch != 0u)
    {
        status = SparkPrefixCacheCancelReservation(
            scheduler->prefix_cache,
            decision->sequence_id,
            decision->prefix_cache_reservation_epoch);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        scheduler->kv_block_cancel_count += 1u;
    }
    return SPARK_STATUS_OK;
}


static void SparkSchedulerRejectPrefillBatch(
    SparkSchedulerPrefillBatchDecision *batch_decision,
    const SparkSchedulerPrefillBatchRequest *batch_request,
    SparkStatus rejected_status)
{
    if (batch_decision == 0)
    {
        return;
    }
    memset(batch_decision, 0, sizeof(*batch_decision));
    batch_decision->abi_version = SPARK_SCHEDULER_ABI_VERSION;
    batch_decision->descriptor_bytes =
        SPARK_SCHEDULER_PREFILL_BATCH_DECISION_DESCRIPTOR_BYTES;
    batch_decision->accepted = 0u;
    batch_decision->rejected_status = rejected_status;
    if (batch_request != 0)
    {
        batch_decision->source_request_count = batch_request->request_count;
    }
}

static SparkStatus SparkSchedulerValidatePrefillBatchRequest(
    const SparkScheduler *scheduler,
    const SparkSchedulerPrefillBatchRequest *batch_request)
{
    uint32_t request_index;

    if (scheduler == 0 || batch_request == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (batch_request->abi_version != SPARK_SCHEDULER_ABI_VERSION ||
        batch_request->descriptor_bytes !=
            SPARK_SCHEDULER_PREFILL_BATCH_REQUEST_DESCRIPTOR_BYTES ||
        batch_request->reserved != 0u ||
        batch_request->request_count == 0u ||
        batch_request->request_count > SPARK_SCHEDULER_MAX_BATCH_REQUEST_COUNT ||
        batch_request->requests == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    for (request_index = 0u;
         request_index < batch_request->request_count;
         ++request_index)
    {
        const SparkSchedulerRequest *request;
        SparkStatus status;

        request = &batch_request->requests[request_index];
        status = SparkSchedulerValidateRequest(scheduler, request);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        if (!SparkSchedulerRequestIsPrefill(request) ||
            request->active_sequence_count != 1u)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
    }
    return SPARK_STATUS_OK;
}

static void SparkSchedulerInitializePrefillBatchLane(
    SparkSchedulerPrefillBatchLane *lane,
    const SparkSchedulerRequest *request,
    uint32_t request_index,
    uint32_t active_sequence_offset,
    uint32_t cached_prefix_token_count,
    uint32_t computed_prompt_token_count,
    uint32_t scheduled_prompt_token_count,
    uint32_t prefix_cache_block_tokens,
    const SparkPrefixCacheReservation *prefix_cache_reservation,
    uint64_t prefix_cache_parent_hash)
{
    memset(lane, 0, sizeof(*lane));
    lane->abi_version = SPARK_SCHEDULER_ABI_VERSION;
    lane->descriptor_bytes =
        SPARK_SCHEDULER_PREFILL_BATCH_LANE_DESCRIPTOR_BYTES;
    lane->request_index = request_index;
    lane->active_sequence_offset = active_sequence_offset;
    lane->active_sequence_count = request->active_sequence_count;
    lane->prompt_token_count = request->prompt_token_count;
    lane->computed_prompt_token_count = computed_prompt_token_count;
    lane->cached_prefix_token_count = cached_prefix_token_count;
    lane->scheduled_prompt_token_offset = computed_prompt_token_count;
    lane->scheduled_prompt_token_count = scheduled_prompt_token_count;
    lane->remaining_prompt_token_count_after_step =
        request->prompt_token_count - computed_prompt_token_count -
        scheduled_prompt_token_count;
    lane->cache_commit_token_count_after_step =
        computed_prompt_token_count + scheduled_prompt_token_count;
    lane->prefix_cache_block_count = cached_prefix_token_count /
        prefix_cache_block_tokens;
    lane->sequence_id = request->sequence_id;
    lane->prompt_token_ids = request->prompt_token_ids;
    lane->prefix_cache_parent_hash = prefix_cache_parent_hash;
    if (prefix_cache_reservation != 0)
    {
        lane->kv_block_token_count = prefix_cache_block_tokens;
        lane->kv_physical_block_count =
            prefix_cache_reservation->physical_block_count;
        lane->kv_cached_physical_block_count =
            prefix_cache_reservation->cached_physical_block_count;
        lane->kv_pending_physical_block_count =
            prefix_cache_reservation->pending_physical_block_count;
        lane->kv_block_table_token_count =
            prefix_cache_reservation->reserved_token_count;
        lane->prefix_cache_reservation_epoch =
            prefix_cache_reservation->reservation_epoch;
        lane->prefix_cache_result_hash =
            prefix_cache_reservation->last_block_hash;
    }
}

static SparkStatus SparkSchedulerCancelAcceptedPrefillBatchReservations(
    SparkScheduler *scheduler,
    SparkSchedulerPrefillBatchDecision *batch_decision)
{
    uint32_t lane_index;

    if (!SparkSchedulerPromptCacheIsEnabled(scheduler))
    {
        return SPARK_STATUS_OK;
    }
    for (lane_index = 0u;
         lane_index < batch_decision->packed_request_count;
         ++lane_index)
    {
        SparkSchedulerPrefillBatchLane *lane;
        SparkStatus status;

        lane = &batch_decision->lanes[lane_index];
        if (lane->prefix_cache_reservation_epoch == 0u)
        {
            continue;
        }
        status = SparkPrefixCacheCancelReservation(
            scheduler->prefix_cache,
            lane->sequence_id,
            lane->prefix_cache_reservation_epoch);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkSchedulerAdmitPrefillBatch(
    SparkScheduler *scheduler,
    const SparkSchedulerPrefillBatchRequest *batch_request,
    SparkSchedulerPrefillBatchDecision *batch_decision)
{
    uint64_t layer_cost_ns[SPARK_STAGE_PLAN_MAX_LAYER_COUNT];
    uint64_t final_stage_extra_cost_ns;
    uint64_t stage_cost_ns;
    uint64_t stage_service_time_ns;
    uint32_t batch_bucket;
    uint32_t request_index;
    uint32_t stage_index;
    uint32_t active_sequence_count;
    uint32_t packed_request_count;
    uint32_t graph_sequence_padding_count;
    uint32_t maximum_scheduled_prompt_token_count;
    uint32_t maximum_prefill_block_count;
    uint32_t decision_flags;
    uint32_t dispatch_flags;
    uint32_t any_remaining_prompt_tokens;
    uint32_t total_cached_prefix_token_count;
    uint32_t total_pending_physical_block_count;
    uint64_t total_scheduled_token_count;
    SparkStagePlan stage_plan;
    SparkStatus status;

    if (batch_decision == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    memset(batch_decision, 0, sizeof(*batch_decision));
    batch_decision->abi_version = SPARK_SCHEDULER_ABI_VERSION;
    batch_decision->descriptor_bytes =
        SPARK_SCHEDULER_PREFILL_BATCH_DECISION_DESCRIPTOR_BYTES;

    if (scheduler == 0 ||
        scheduler->abi_version != SPARK_SCHEDULER_ABI_VERSION ||
        scheduler->descriptor_bytes != SPARK_SCHEDULER_DESCRIPTOR_BYTES)
    {
        SparkSchedulerRejectPrefillBatch(
            batch_decision,
            batch_request,
            SPARK_STATUS_INVALID_ARGUMENT);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    status = SparkSchedulerValidatePrefillBatchRequest(
        scheduler,
        batch_request);
    if (status != SPARK_STATUS_OK)
    {
        SparkSchedulerRejectPrefillBatch(
            batch_decision,
            batch_request,
            status);
        return status;
    }

    active_sequence_count = 0u;
    packed_request_count = 0u;
    for (request_index = 0u;
         request_index < batch_request->request_count &&
             packed_request_count < SPARK_SCHEDULER_MAX_PACKED_REQUEST_COUNT;
         ++request_index)
    {
        const SparkSchedulerRequest *request;

        request = &batch_request->requests[request_index];
        if (active_sequence_count + request->active_sequence_count >
            SPARK_SCHEDULER_MAX_PACKED_REQUEST_COUNT)
        {
            break;
        }
        active_sequence_count += request->active_sequence_count;
        packed_request_count += 1u;
    }
    if (packed_request_count == 0u || active_sequence_count == 0u)
    {
        SparkSchedulerRejectPrefillBatch(
            batch_decision,
            batch_request,
            SPARK_STATUS_CAPACITY_EXCEEDED);
        return SPARK_STATUS_OK;
    }

    status = SparkStagePlanSelectBatchBucket(
        active_sequence_count,
        &batch_bucket);
    if (status != SPARK_STATUS_OK)
    {
        SparkSchedulerRejectPrefillBatch(
            batch_decision,
            batch_request,
            status);
        return SPARK_STATUS_OK;
    }

    status = SparkStagePlanBuildCurrentSparkMeasuredBalancedForQuantization(
        &(const SparkStagePlanGeometry){SPARK_GLM52_MODEL_LAYER_COUNT, SPARK_GLM52_MODEL_FIRST_ROUTED_LAYER},
        scheduler->measured_profile_id,
        batch_bucket,
        scheduler->quantization_mode,
        &batch_decision->stage_decision.stage_plan,
        0,
        0u);
    if (status != SPARK_STATUS_OK)
    {
        SparkSchedulerRejectPrefillBatch(
            batch_decision,
            batch_request,
            status);
        return SPARK_STATUS_OK;
    }

    status = SparkStagePlanLoadMeasuredCostProfileForQuantization(
        &(const SparkStagePlanGeometry){SPARK_GLM52_MODEL_LAYER_COUNT, SPARK_GLM52_MODEL_FIRST_ROUTED_LAYER},
        scheduler->measured_profile_id,
        batch_bucket,
        scheduler->quantization_mode,
        layer_cost_ns,
        &final_stage_extra_cost_ns);
    if (status != SPARK_STATUS_OK)
    {
        SparkSchedulerRejectPrefillBatch(
            batch_decision,
            batch_request,
            status);
        return SPARK_STATUS_OK;
    }

    for (stage_index = 0u;
         stage_index < batch_decision->stage_decision.stage_plan.stage_count;
         ++stage_index)
    {
        if (!SparkSchedulerStageHasCapacity(
                scheduler,
                stage_index,
                1u))
        {
            SparkSchedulerRejectPrefillBatch(
                batch_decision,
                batch_request,
                SPARK_STATUS_BUSY);
            return SPARK_STATUS_OK;
        }
    }

    batch_decision->source_request_count = batch_request->request_count;
    batch_decision->packed_request_count = 0u;
    batch_decision->active_sequence_count = active_sequence_count;
    maximum_scheduled_prompt_token_count = 0u;
    maximum_prefill_block_count = 1u;
    any_remaining_prompt_tokens = 0u;
    total_cached_prefix_token_count = 0u;
    total_pending_physical_block_count = 0u;
    total_scheduled_token_count = 0u;

    for (request_index = 0u;
         request_index < packed_request_count;
         ++request_index)
    {
        const SparkSchedulerRequest *request;
        SparkPrefixCacheReservation prefix_cache_reservation;
        SparkPrefixCachePromptHash prefix_cache_parent_hash;
        uint32_t cached_prefix_token_count;
        uint32_t computed_prompt_token_count;
        uint32_t scheduled_prompt_token_count;
        uint32_t prefill_block_count;
        uint64_t parent_hash;

        request = &batch_request->requests[request_index];
        status = SparkSchedulerLookupCachedPrefixTokenCount(
            scheduler,
            request,
            &cached_prefix_token_count);
        if (status != SPARK_STATUS_OK)
        {
            SparkSchedulerCancelAcceptedPrefillBatchReservations(
                scheduler,
                batch_decision);
            SparkSchedulerRejectPrefillBatch(
                batch_decision,
                batch_request,
                status);
            return status;
        }
        computed_prompt_token_count =
            SparkSchedulerEffectiveComputedPromptTokenCount(
                request,
                cached_prefix_token_count);
        scheduled_prompt_token_count =
            SparkSchedulerScheduledPrefillTokenCount(
                scheduler,
                request,
                computed_prompt_token_count);
        if (scheduled_prompt_token_count == 0u)
        {
            SparkSchedulerCancelAcceptedPrefillBatchReservations(
                scheduler,
                batch_decision);
            SparkSchedulerRejectPrefillBatch(
                batch_decision,
                batch_request,
                SPARK_STATUS_INVALID_ARGUMENT);
            return SPARK_STATUS_INVALID_ARGUMENT;
        }

        memset(&prefix_cache_reservation, 0, sizeof(prefix_cache_reservation));
        prefix_cache_reservation.abi_version =
            SPARK_PREFIX_CACHE_ABI_VERSION;
        prefix_cache_reservation.descriptor_bytes =
            SPARK_PREFIX_CACHE_RESERVATION_DESCRIPTOR_BYTES;
        status = SparkSchedulerReservePrompt(
            scheduler,
            request,
            computed_prompt_token_count + scheduled_prompt_token_count,
            &prefix_cache_reservation);
        if (status != SPARK_STATUS_OK)
        {
            SparkSchedulerCancelAcceptedPrefillBatchReservations(
                scheduler,
                batch_decision);
            SparkSchedulerRejectPrefillBatch(
                batch_decision,
                batch_request,
                status);
            return status;
        }

        if (computed_prompt_token_count == 0u)
        {
            parent_hash = SPARK_PREFIX_CACHE_EMPTY_PARENT_HASH;
        }
        else
        {
            status = SparkPrefixCacheHashPromptTokens(
                scheduler->prefix_cache_block_tokens,
                SPARK_PREFIX_CACHE_EMPTY_PARENT_HASH,
                request->prompt_token_ids,
                computed_prompt_token_count,
                &prefix_cache_parent_hash);
            if (status != SPARK_STATUS_OK)
            {
                SparkPrefixCacheCancelReservation(
                    scheduler->prefix_cache,
                    request->sequence_id,
                    prefix_cache_reservation.reservation_epoch);
                SparkSchedulerCancelAcceptedPrefillBatchReservations(
                    scheduler,
                    batch_decision);
                SparkSchedulerRejectPrefillBatch(
                    batch_decision,
                    batch_request,
                    status);
                return status;
            }
            parent_hash = prefix_cache_parent_hash.prompt_hash;
        }

        SparkSchedulerInitializePrefillBatchLane(
            &batch_decision->lanes[request_index],
            request,
            request_index,
            request_index,
            cached_prefix_token_count,
            computed_prompt_token_count,
            scheduled_prompt_token_count,
            scheduler->prefix_cache_block_tokens,
            &prefix_cache_reservation,
            parent_hash);
        batch_decision->packed_request_count += 1u;
        total_cached_prefix_token_count += cached_prefix_token_count;
        total_pending_physical_block_count +=
            prefix_cache_reservation.pending_physical_block_count;
        total_scheduled_token_count += scheduled_prompt_token_count;
        if (scheduled_prompt_token_count > maximum_scheduled_prompt_token_count)
        {
            maximum_scheduled_prompt_token_count = scheduled_prompt_token_count;
        }
        prefill_block_count = SparkSchedulerPrefillBlockCount(
            scheduler,
            scheduled_prompt_token_count);
        if (prefill_block_count > maximum_prefill_block_count)
        {
            maximum_prefill_block_count = prefill_block_count;
        }
        if (batch_decision->lanes[request_index].remaining_prompt_token_count_after_step != 0u)
        {
            any_remaining_prompt_tokens = 1u;
        }
    }

    graph_sequence_padding_count = batch_bucket - active_sequence_count;
    decision_flags = SPARK_SCHEDULER_DECISION_FLAG_PREFILL_STEP |
        SPARK_SCHEDULER_DECISION_FLAG_ADAPTIVE_PREFILL_PACK;
    if (any_remaining_prompt_tokens != 0u)
    {
        decision_flags |= SPARK_SCHEDULER_DECISION_FLAG_PREFILL_CHUNK;
    }
    else
    {
        decision_flags |= SPARK_SCHEDULER_DECISION_FLAG_PREFILL_FINAL_CHUNK;
    }
    if (total_cached_prefix_token_count != 0u)
    {
        decision_flags |= SPARK_SCHEDULER_DECISION_FLAG_PREFIX_CACHE_USED;
    }
    if (graph_sequence_padding_count != 0u &&
        SparkSchedulerCudaGraphPaddingIsEnabled(scheduler))
    {
        decision_flags |= SPARK_SCHEDULER_DECISION_FLAG_CUDAGRAPH_PADDING;
    }
    if (SparkSchedulerPrefillDecodeInterleaveIsEnabled(scheduler) &&
        scheduler->queue_depth_per_spark > 1u)
    {
        decision_flags |=
            SPARK_SCHEDULER_DECISION_FLAG_PREFILL_RESERVED_DECODE_SLOT;
    }
    dispatch_flags = SparkSchedulerBuildDispatchFlags(decision_flags) |
        SPARK_SCHEDULER_DISPATCH_STAGE_FLAG_ADAPTIVE_PREFILL_PACK;

    batch_decision->accepted = 1u;
    batch_decision->rejected_status = SPARK_STATUS_OK;
    batch_decision->batch_bucket = batch_bucket;
    batch_decision->graph_sequence_capacity = batch_bucket;
    batch_decision->graph_sequence_padding_count = graph_sequence_padding_count;
    batch_decision->decision_flags = decision_flags;
    batch_decision->maximum_scheduled_prompt_token_count =
        maximum_scheduled_prompt_token_count;
    batch_decision->total_scheduled_token_count = total_scheduled_token_count;

    stage_plan = batch_decision->stage_decision.stage_plan;
    memset(&batch_decision->stage_decision, 0, sizeof(batch_decision->stage_decision));
    batch_decision->stage_decision.stage_plan = stage_plan;
    batch_decision->stage_decision.abi_version = SPARK_SCHEDULER_ABI_VERSION;
    batch_decision->stage_decision.descriptor_bytes =
        SPARK_SCHEDULER_DECISION_DESCRIPTOR_BYTES;
    batch_decision->stage_decision.accepted = 1u;
    batch_decision->stage_decision.batch_bucket = batch_bucket;
    batch_decision->stage_decision.quantization_mode = scheduler->quantization_mode;
    batch_decision->stage_decision.spark_count = scheduler->spark_count;
    batch_decision->stage_decision.stage_count =
        batch_decision->stage_decision.stage_plan.stage_count;
    batch_decision->stage_decision.rejected_status = SPARK_STATUS_OK;
    batch_decision->stage_decision.decision_flags = decision_flags;
    batch_decision->stage_decision.active_sequence_count = active_sequence_count;
    batch_decision->stage_decision.graph_sequence_capacity = batch_bucket;
    batch_decision->stage_decision.graph_sequence_padding_count =
        graph_sequence_padding_count;
    batch_decision->stage_decision.scheduled_prompt_token_count =
        maximum_scheduled_prompt_token_count;
    batch_decision->stage_decision.prefill_block_count =
        maximum_prefill_block_count;
    batch_decision->stage_decision.total_scheduled_token_count =
        total_scheduled_token_count;

    for (stage_index = 0u;
         stage_index < batch_decision->stage_decision.stage_count;
         ++stage_index)
    {
        stage_cost_ns = SparkSchedulerStageCostNs(
            layer_cost_ns,
            final_stage_extra_cost_ns,
            &batch_decision->stage_decision.stage_plan.stages[stage_index]);
        stage_service_time_ns = stage_cost_ns *
            (uint64_t)maximum_prefill_block_count;
        batch_decision->stage_decision.dispatch_stages[stage_index].spark_index =
            stage_index;
        batch_decision->stage_decision.dispatch_stages[stage_index].batch_bucket =
            batch_bucket;
        batch_decision->stage_decision.dispatch_stages[stage_index].first_layer_index =
            batch_decision->stage_decision.stage_plan.stages[stage_index].first_layer_index;
        batch_decision->stage_decision.dispatch_stages[stage_index].layer_count =
            batch_decision->stage_decision.stage_plan.stages[stage_index].layer_count;
        batch_decision->stage_decision.dispatch_stages[stage_index].stage_flags =
            batch_decision->stage_decision.stage_plan.stages[stage_index].flags;
        batch_decision->stage_decision.dispatch_stages[stage_index].dispatch_flags =
            dispatch_flags;
        batch_decision->stage_decision.dispatch_stages[stage_index].active_sequence_count =
            active_sequence_count;
        batch_decision->stage_decision.dispatch_stages[stage_index].graph_sequence_capacity =
            batch_bucket;
        batch_decision->stage_decision.dispatch_stages[stage_index].graph_sequence_padding_count =
            graph_sequence_padding_count;
        batch_decision->stage_decision.dispatch_stages[stage_index].scheduled_prompt_token_count =
            maximum_scheduled_prompt_token_count;
        batch_decision->stage_decision.dispatch_stages[stage_index].estimated_service_time_ns =
            stage_service_time_ns;
        if (stage_service_time_ns > batch_decision->estimated_critical_path_ns)
        {
            batch_decision->estimated_critical_path_ns = stage_service_time_ns;
        }
        scheduler->spark_inflight_counts[stage_index] += 1u;
    }
    batch_decision->stage_decision.estimated_critical_path_ns =
        batch_decision->estimated_critical_path_ns;

    scheduler->admitted_count += 1u;
    scheduler->scheduled_prefill_token_count += total_scheduled_token_count;
    scheduler->prefix_cache_hit_token_count += total_cached_prefix_token_count;
    if ((decision_flags & SPARK_SCHEDULER_DECISION_FLAG_PREFILL_CHUNK) != 0u)
    {
        scheduler->chunked_prefill_count += 1u;
    }
    if ((decision_flags &
         SPARK_SCHEDULER_DECISION_FLAG_PREFILL_RESERVED_DECODE_SLOT) != 0u)
    {
        scheduler->interleaved_prefill_admission_count += 1u;
    }
    scheduler->kv_block_reservation_count += total_pending_physical_block_count;
    scheduler->kv_block_reservation_token_count += total_scheduled_token_count;
    scheduler->adaptive_prefill_pack_admission_count += 1u;
    scheduler->adaptive_prefill_pack_request_count +=
        batch_decision->packed_request_count;
    scheduler->adaptive_prefill_pack_padding_token_count +=
        graph_sequence_padding_count;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkSchedulerValidateAcceptedPrefillBatchDecision(
    const SparkScheduler *scheduler,
    const SparkSchedulerPrefillBatchDecision *batch_decision)
{
    if (scheduler == 0 || batch_decision == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (scheduler->abi_version != SPARK_SCHEDULER_ABI_VERSION ||
        scheduler->descriptor_bytes != SPARK_SCHEDULER_DESCRIPTOR_BYTES ||
        batch_decision->abi_version != SPARK_SCHEDULER_ABI_VERSION ||
        batch_decision->descriptor_bytes !=
            SPARK_SCHEDULER_PREFILL_BATCH_DECISION_DESCRIPTOR_BYTES ||
        batch_decision->accepted == 0u ||
        batch_decision->packed_request_count == 0u ||
        batch_decision->packed_request_count >
            SPARK_SCHEDULER_MAX_PACKED_REQUEST_COUNT ||
        batch_decision->stage_decision.accepted == 0u ||
        (batch_decision->decision_flags &
            SPARK_SCHEDULER_DECISION_FLAG_ADAPTIVE_PREFILL_PACK) == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

// Completing and cancelling a prefill batch are the same retirement: validate
// the decision, release its in-flight reservation, then settle every lane's
// prefix-cache reservation. They differ in which way that reservation settles
// and which counter records it, so those are the parameters and the retirement
// is written once.
typedef SparkStatus (*SparkGlm52SchedulerReservationFunction)(
    SparkPrefixCache *prefix_cache,
    uint64_t sequence_id,
    uint64_t reservation_epoch);

static SparkStatus SparkSchedulerRetirePrefillBatch(
    SparkScheduler *scheduler,
    const SparkSchedulerPrefillBatchDecision *batch_decision,
    SparkGlm52SchedulerReservationFunction settle_reservation,
    uint64_t *retire_counter)
{
    uint32_t lane_index;
    SparkStatus status;

    status = SparkSchedulerValidateAcceptedPrefillBatchDecision(
        scheduler,
        batch_decision);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkSchedulerReleaseDecisionInflight(
        scheduler,
        &batch_decision->stage_decision);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    for (lane_index = 0u;
         lane_index < batch_decision->packed_request_count;
         ++lane_index)
    {
        const SparkSchedulerPrefillBatchLane *lane;

        lane = &batch_decision->lanes[lane_index];
        if (lane->prefix_cache_reservation_epoch == 0u)
        {
            continue;
        }
        status = settle_reservation(
            scheduler->prefix_cache,
            lane->sequence_id,
            lane->prefix_cache_reservation_epoch);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }
    *retire_counter += 1u;
    return SPARK_STATUS_OK;
}

SparkStatus SparkSchedulerCompletePrefillBatch(
    SparkScheduler *scheduler,
    const SparkSchedulerPrefillBatchDecision *batch_decision)
{
    return SparkSchedulerRetirePrefillBatch(
        scheduler,
        batch_decision,
        SparkPrefixCacheCommitReservation,
        &scheduler->completed_count);
}

SparkStatus SparkSchedulerCancelPrefillBatch(
    SparkScheduler *scheduler,
    const SparkSchedulerPrefillBatchDecision *batch_decision)
{
    return SparkSchedulerRetirePrefillBatch(
        scheduler,
        batch_decision,
        SparkPrefixCacheCancelReservation,
        &scheduler->kv_block_cancel_count);
}


SparkStatus SparkSchedulerBuildPrefillBatchKvBlockTables(
    SparkScheduler *scheduler,
    const SparkSchedulerPrefillBatchDecision *batch_decision,
    uint32_t *physical_block_indices,
    uint32_t lane_stride,
    uint32_t lane_capacity,
    uint32_t *lane_physical_block_counts,
    uint32_t lane_count_capacity)
{
    uint32_t lane_index;
    SparkStatus status;

    status = SparkSchedulerValidateAcceptedPrefillBatchDecision(
        scheduler,
        batch_decision);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (!SparkSchedulerPromptCacheIsEnabled(scheduler) ||
        physical_block_indices == 0 ||
        lane_physical_block_counts == 0 ||
        lane_count_capacity < batch_decision->packed_request_count ||
        lane_capacity == 0u ||
        lane_stride < lane_capacity)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    for (lane_index = 0u; lane_index < lane_count_capacity; ++lane_index)
    {
        lane_physical_block_counts[lane_index] = 0u;
    }
    for (lane_index = 0u;
         lane_index < batch_decision->packed_request_count;
         ++lane_index)
    {
        const SparkSchedulerPrefillBatchLane *lane;

        lane = &batch_decision->lanes[lane_index];
        if (lane->sequence_id == 0u || lane->kv_block_table_token_count == 0u)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        status = SparkPrefixCacheBuildPhysicalBlockTable(
            scheduler->prefix_cache,
            lane->sequence_id,
            lane->kv_block_table_token_count,
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

SparkStatus SparkSchedulerBuildKvBlockTable(
    SparkScheduler *scheduler,
    const SparkSchedulerDecision *decision,
    uint32_t *physical_block_indices,
    uint32_t physical_block_capacity,
    uint32_t *physical_block_count_out)
{
    SparkStatus status;

    status = SparkSchedulerValidateAcceptedDecision(scheduler, decision);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (!SparkSchedulerPromptCacheIsEnabled(scheduler) ||
        (decision->decision_flags &
            SPARK_SCHEDULER_DECISION_FLAG_PREFILL_STEP) == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SparkPrefixCacheBuildPhysicalBlockTable(
        scheduler->prefix_cache,
        decision->sequence_id,
        decision->kv_block_table_token_count,
        physical_block_indices,
        physical_block_capacity,
        physical_block_count_out);
}

SparkStatus SparkSchedulerReleaseSequence(
    SparkScheduler *scheduler,
    uint64_t sequence_id)
{
    if (scheduler == 0 || sequence_id == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (scheduler->abi_version != SPARK_SCHEDULER_ABI_VERSION ||
        scheduler->descriptor_bytes != SPARK_SCHEDULER_DESCRIPTOR_BYTES)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (!SparkSchedulerPromptCacheIsEnabled(scheduler))
    {
        return SPARK_STATUS_OK;
    }
    return SparkPrefixCacheReleaseSequence(
        scheduler->prefix_cache,
        sequence_id);
}

static void SparkSchedulerInitializePackedRequest(
    SparkSchedulerPackedRequest *packed_request,
    uint32_t request_index,
    uint32_t active_sequence_offset,
    const SparkSchedulerRequest *request)
{
    memset(packed_request, 0, sizeof(*packed_request));
    packed_request->abi_version = SPARK_SCHEDULER_ABI_VERSION;
    packed_request->descriptor_bytes =
        SPARK_SCHEDULER_PACKED_REQUEST_DESCRIPTOR_BYTES;
    packed_request->request_index = request_index;
    packed_request->active_sequence_offset = active_sequence_offset;
    packed_request->active_sequence_count = request->active_sequence_count;
    packed_request->request_flags = request->flags;
    packed_request->scheduled_token_count = request->active_sequence_count;
    packed_request->total_scheduled_token_count = request->active_sequence_count;
}

static SparkStatus SparkSchedulerValidateDecodeBatchRequest(
    const SparkScheduler *scheduler,
    const SparkSchedulerBatchRequest *batch_request)
{
    uint32_t request_index;

    if (scheduler == 0 || batch_request == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (batch_request->abi_version != SPARK_SCHEDULER_ABI_VERSION ||
        batch_request->descriptor_bytes !=
            SPARK_SCHEDULER_BATCH_REQUEST_DESCRIPTOR_BYTES ||
        batch_request->reserved != 0u ||
        batch_request->request_count == 0u ||
        batch_request->request_count > SPARK_SCHEDULER_MAX_BATCH_REQUEST_COUNT ||
        batch_request->requests == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    for (request_index = 0u;
         request_index < batch_request->request_count;
         ++request_index)
    {
        SparkStatus status;

        status = SparkSchedulerValidateRequest(
            scheduler,
            &batch_request->requests[request_index]);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        if (!SparkSchedulerRequestIsDecode(
                &batch_request->requests[request_index]))
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
    }
    return SPARK_STATUS_OK;
}

static void SparkSchedulerRejectDecodeBatch(
    SparkSchedulerBatchDecision *batch_decision,
    const SparkSchedulerBatchRequest *batch_request,
    SparkStatus rejected_status)
{
    if (batch_decision == 0)
    {
        return;
    }
    memset(batch_decision, 0, sizeof(*batch_decision));
    batch_decision->abi_version = SPARK_SCHEDULER_ABI_VERSION;
    batch_decision->descriptor_bytes =
        SPARK_SCHEDULER_BATCH_DECISION_DESCRIPTOR_BYTES;
    batch_decision->accepted = 0u;
    batch_decision->rejected_status = rejected_status;
    if (batch_request != 0)
    {
        batch_decision->source_request_count = batch_request->request_count;
    }
}

SparkStatus SparkSchedulerAdmitDecodeBatch(
    SparkScheduler *scheduler,
    const SparkSchedulerBatchRequest *batch_request,
    SparkSchedulerBatchDecision *batch_decision)
{
    SparkSchedulerRequest aggregate_request;
    SparkStatus status;
    uint32_t request_index;
    uint32_t active_sequence_offset;
    uint32_t packed_request_count;
    uint32_t active_sequence_count;

    if (batch_decision == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    memset(batch_decision, 0, sizeof(*batch_decision));
    batch_decision->abi_version = SPARK_SCHEDULER_ABI_VERSION;
    batch_decision->descriptor_bytes =
        SPARK_SCHEDULER_BATCH_DECISION_DESCRIPTOR_BYTES;

    if (scheduler == 0 ||
        scheduler->abi_version != SPARK_SCHEDULER_ABI_VERSION ||
        scheduler->descriptor_bytes != SPARK_SCHEDULER_DESCRIPTOR_BYTES)
    {
        SparkSchedulerRejectDecodeBatch(
            batch_decision,
            batch_request,
            SPARK_STATUS_INVALID_ARGUMENT);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    status = SparkSchedulerValidateDecodeBatchRequest(
        scheduler,
        batch_request);
    if (status != SPARK_STATUS_OK)
    {
        SparkSchedulerRejectDecodeBatch(
            batch_decision,
            batch_request,
            status);
        return status;
    }

    active_sequence_offset = 0u;
    packed_request_count = 0u;
    active_sequence_count = 0u;
    for (request_index = 0u;
         request_index < batch_request->request_count &&
             packed_request_count < SPARK_SCHEDULER_MAX_PACKED_REQUEST_COUNT;
         ++request_index)
    {
        const SparkSchedulerRequest *request;

        request = &batch_request->requests[request_index];
        if (request->active_sequence_count >
            SPARK_SCHEDULER_MAX_PACKED_REQUEST_COUNT)
        {
            SparkSchedulerRejectDecodeBatch(
                batch_decision,
                batch_request,
                SPARK_STATUS_CAPACITY_EXCEEDED);
            return SPARK_STATUS_OK;
        }
        if (active_sequence_count + request->active_sequence_count >
            SPARK_SCHEDULER_MAX_PACKED_REQUEST_COUNT)
        {
            break;
        }
        SparkSchedulerInitializePackedRequest(
            &batch_decision->packed_requests[packed_request_count],
            request_index,
            active_sequence_offset,
            request);
        active_sequence_offset += request->active_sequence_count;
        active_sequence_count += request->active_sequence_count;
        packed_request_count += 1u;
    }
    if (packed_request_count == 0u || active_sequence_count == 0u)
    {
        SparkSchedulerRejectDecodeBatch(
            batch_decision,
            batch_request,
            SPARK_STATUS_CAPACITY_EXCEEDED);
        return SPARK_STATUS_OK;
    }

    memset(&aggregate_request, 0, sizeof(aggregate_request));
    aggregate_request.abi_version = SPARK_SCHEDULER_ABI_VERSION;
    aggregate_request.descriptor_bytes =
        SPARK_SCHEDULER_REQUEST_DESCRIPTOR_BYTES;
    aggregate_request.active_sequence_count = active_sequence_count;
    aggregate_request.flags = SPARK_SCHEDULER_REQUEST_FLAG_DECODE;

    status = SparkSchedulerAdmit(
        scheduler,
        &aggregate_request,
        &batch_decision->stage_decision);
    if (status != SPARK_STATUS_OK)
    {
        SparkSchedulerRejectDecodeBatch(
            batch_decision,
            batch_request,
            status);
        return status;
    }
    if (batch_decision->stage_decision.accepted == 0u)
    {
        SparkStatus rejected_status;

        rejected_status = batch_decision->stage_decision.rejected_status;
        SparkSchedulerRejectDecodeBatch(
            batch_decision,
            batch_request,
            rejected_status);
        return SPARK_STATUS_OK;
    }

    batch_decision->accepted = 1u;
    batch_decision->rejected_status = SPARK_STATUS_OK;
    batch_decision->source_request_count = batch_request->request_count;
    batch_decision->packed_request_count = packed_request_count;
    batch_decision->batch_bucket = batch_decision->stage_decision.batch_bucket;
    batch_decision->active_sequence_count = active_sequence_count;
    batch_decision->graph_sequence_capacity =
        batch_decision->stage_decision.graph_sequence_capacity;
    batch_decision->graph_sequence_padding_count =
        batch_decision->stage_decision.graph_sequence_padding_count;
    batch_decision->decision_flags =
        batch_decision->stage_decision.decision_flags |
        SPARK_SCHEDULER_DECISION_FLAG_ADAPTIVE_DECODE_PACK;
    batch_decision->total_scheduled_token_count =
        batch_decision->stage_decision.total_scheduled_token_count;
    batch_decision->estimated_critical_path_ns =
        batch_decision->stage_decision.estimated_critical_path_ns;

    batch_decision->stage_decision.decision_flags |=
        SPARK_SCHEDULER_DECISION_FLAG_ADAPTIVE_DECODE_PACK;
    for (request_index = 0u;
         request_index < batch_decision->stage_decision.stage_count;
         ++request_index)
    {
        batch_decision->stage_decision.dispatch_stages[request_index].dispatch_flags |=
            SPARK_SCHEDULER_DISPATCH_STAGE_FLAG_ADAPTIVE_DECODE_PACK;
    }

    scheduler->adaptive_decode_pack_admission_count += 1u;
    scheduler->adaptive_decode_pack_request_count += packed_request_count;
    scheduler->adaptive_decode_pack_padding_token_count +=
        batch_decision->graph_sequence_padding_count;
    return SPARK_STATUS_OK;
}

SparkStatus SparkSchedulerCompleteDecodeBatch(
    SparkScheduler *scheduler,
    const SparkSchedulerBatchDecision *batch_decision)
{
    if (batch_decision == 0 ||
        batch_decision->abi_version != SPARK_SCHEDULER_ABI_VERSION ||
        batch_decision->descriptor_bytes !=
            SPARK_SCHEDULER_BATCH_DECISION_DESCRIPTOR_BYTES ||
        batch_decision->accepted == 0u ||
        batch_decision->packed_request_count == 0u ||
        (batch_decision->decision_flags &
            SPARK_SCHEDULER_DECISION_FLAG_ADAPTIVE_DECODE_PACK) == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SparkSchedulerComplete(
        scheduler,
        &batch_decision->stage_decision);
}

SparkStatus SparkSchedulerCancelDecodeBatch(
    SparkScheduler *scheduler,
    const SparkSchedulerBatchDecision *batch_decision)
{
    if (batch_decision == 0 ||
        batch_decision->abi_version != SPARK_SCHEDULER_ABI_VERSION ||
        batch_decision->descriptor_bytes !=
            SPARK_SCHEDULER_BATCH_DECISION_DESCRIPTOR_BYTES ||
        batch_decision->accepted == 0u ||
        batch_decision->packed_request_count == 0u ||
        (batch_decision->decision_flags &
            SPARK_SCHEDULER_DECISION_FLAG_ADAPTIVE_DECODE_PACK) == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SparkSchedulerReleaseDecisionInflight(
        scheduler,
        &batch_decision->stage_decision);
}
