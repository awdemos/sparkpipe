#include <assert.h>
#include <stdint.h>
#include <string.h>

#include "sparkpipe/spark_long_context.h"

static void SparkTestLongContextDefaultPolicyIsBounded(void)
{
    SparkLongContextPolicy policy;

    SparkLongContextInitializeDefaultPolicy(&policy);
    assert(SparkLongContextValidatePolicy(&policy) == SPARK_STATUS_OK);
    assert(policy.max_context_tokens == SPARK_LONG_CONTEXT_DEFAULT_MAX_CONTEXT_TOKENS);
    assert(policy.selected_token_capacity == 2048u);
    assert(policy.maximum_decode_scan_token_count == 2048u);
    assert((policy.policy_flags &
        SPARK_LONG_CONTEXT_POLICY_FLAG_REQUIRE_BOUNDED_DECODE) != 0u);
}

static void SparkTestLongContextBuildsBoundedDecodeSelectionFor256k(void)
{
    SparkLongContextPolicy policy;
    SparkLongContextDecodePlan decode_plan;
    uint32_t selected_token_indices[
        SPARK_LONG_CONTEXT_DEFAULT_SELECTED_TOKEN_CAPACITY];
    uint32_t token_index;
    uint32_t saw_last_context_token;

    SparkLongContextInitializeDefaultPolicy(&policy);
    memset(selected_token_indices, 0, sizeof(selected_token_indices));

    assert(SparkLongContextBuildDecodeSelection(
        &policy,
        262144u,
        selected_token_indices,
        SPARK_LONG_CONTEXT_DEFAULT_SELECTED_TOKEN_CAPACITY,
        &decode_plan) == SPARK_STATUS_OK);

    assert((decode_plan.flags &
        SPARK_LONG_CONTEXT_DECODE_PLAN_FLAG_BOUNDED_SELECTION) != 0u);
    assert((decode_plan.flags &
        SPARK_LONG_CONTEXT_DECODE_PLAN_FLAG_LONG_CONTEXT) != 0u);
    assert((decode_plan.flags &
        SPARK_LONG_CONTEXT_DECODE_PLAN_FLAG_CONTEXT_TRUNCATED) != 0u);
    assert(decode_plan.context_token_count == 262144u);
    assert(decode_plan.selected_token_count ==
        SPARK_LONG_CONTEXT_DEFAULT_SELECTED_TOKEN_CAPACITY);
    assert(decode_plan.selected_block_count < decode_plan.kv_block_count_for_context);
    assert(decode_plan.selected_block_count <= 2048u);
    assert(decode_plan.avoided_full_scan_token_reads > 250000u);
    assert(selected_token_indices[0u] == 0u);

    saw_last_context_token = 0u;
    for (token_index = 0u;
         token_index < decode_plan.selected_token_count;
         ++token_index)
    {
        assert(selected_token_indices[token_index] < 262144u);
        if (selected_token_indices[token_index] == 262143u)
        {
            saw_last_context_token = 1u;
        }
    }
    assert(saw_last_context_token != 0u);
}

static void SparkTestLongContextBuildsChunkedPrefillPlanFor256k(void)
{
    SparkLongContextPolicy policy;
    SparkLongContextPrefillPlan prefill_plan;

    SparkLongContextInitializeDefaultPolicy(&policy);
    assert(SparkLongContextBuildPrefillPlan(
        &policy,
        262144u,
        256u,
        &prefill_plan) == SPARK_STATUS_OK);
    assert((prefill_plan.flags &
        SPARK_LONG_CONTEXT_PREFILL_PLAN_FLAG_CHUNKED_PREFILL) != 0u);
    assert((prefill_plan.flags &
        SPARK_LONG_CONTEXT_PREFILL_PLAN_FLAG_LONG_CONTEXT) != 0u);
    assert(prefill_plan.prefill_chunk_count == 1024u);
    assert(prefill_plan.kv_block_count_for_prompt == 16384u);
    assert(prefill_plan.total_prompt_token_visits == 262144u);
}

static void SparkTestLongContextFullScanRequiresExplicitUnsafePolicy(void)
{
    SparkLongContextPolicy policy;
    SparkLongContextDecodePlan decode_plan;
    uint32_t selected_token_indices[
        SPARK_LONG_CONTEXT_DEFAULT_SELECTED_TOKEN_CAPACITY];

    SparkLongContextInitializeDefaultPolicy(&policy);
    policy.policy_mode = SPARK_LONG_CONTEXT_POLICY_MODE_FULL_CONTEXT_SCAN;
    policy.policy_flags &=
        ~SPARK_LONG_CONTEXT_POLICY_FLAG_REQUIRE_BOUNDED_DECODE;
    assert(SparkLongContextBuildDecodeSelection(
        &policy,
        262144u,
        selected_token_indices,
        SPARK_LONG_CONTEXT_DEFAULT_SELECTED_TOKEN_CAPACITY,
        &decode_plan) == SPARK_STATUS_INVALID_ARGUMENT);

    policy.policy_flags |= SPARK_LONG_CONTEXT_POLICY_FLAG_ALLOW_FULL_CONTEXT_SCAN;
    assert(SparkLongContextBuildDecodeSelection(
        &policy,
        262144u,
        selected_token_indices,
        SPARK_LONG_CONTEXT_DEFAULT_SELECTED_TOKEN_CAPACITY,
        &decode_plan) == SPARK_STATUS_CAPACITY_EXCEEDED);
}

int main(void)
{
    SparkTestLongContextDefaultPolicyIsBounded();
    SparkTestLongContextBuildsBoundedDecodeSelectionFor256k();
    SparkTestLongContextBuildsChunkedPrefillPlanFor256k();
    SparkTestLongContextFullScanRequiresExplicitUnsafePolicy();
    return 0;
}
