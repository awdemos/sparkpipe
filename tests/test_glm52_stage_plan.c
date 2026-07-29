#include <assert.h>
#include <string.h>

#include "sparkpipe/spark_stage_plan.h"


static uint64_t SparkTestGlm52StagePlanMaximumStageCostNs(
    const SparkStagePlan *stage_plan,
    const uint64_t layer_cost_ns[SPARK_STAGE_PLAN_LAYER_COUNT],
    uint64_t final_stage_extra_cost_ns)
{
    uint64_t maximum_stage_cost_ns;
    uint64_t stage_cost_ns;
    uint32_t stage_index;
    uint32_t layer_offset;

    maximum_stage_cost_ns = 0u;
    for (stage_index = 0u;
         stage_index < stage_plan->stage_count;
         ++stage_index)
    {
        stage_cost_ns = 0u;
        for (layer_offset = 0u;
             layer_offset < stage_plan->stages[stage_index].layer_count;
             ++layer_offset)
        {
            stage_cost_ns += layer_cost_ns[
                stage_plan->stages[stage_index].first_layer_index +
                    layer_offset];
        }
        if ((stage_plan->stages[stage_index].flags &
                SPARK_STAGE_PLAN_STAGE_FLAG_FINAL_TOKEN) != 0u)
        {
            stage_cost_ns += final_stage_extra_cost_ns;
        }
        if (stage_cost_ns > maximum_stage_cost_ns)
        {
            maximum_stage_cost_ns = stage_cost_ns;
        }
    }
    return maximum_stage_cost_ns;
}

static void SparkTestGlm52StagePlanValidRing(void)
{
    SparkStagePlan stage_plan;
    char error_buffer[256];
    uint32_t stage_index;

    assert(SPARK_STAGE_PLAN_PIPELINE_INFLIGHT_REQUEST_CAPACITY ==
        13u * 1024u);

    memset(&stage_plan, 0, sizeof(stage_plan));
    stage_plan.abi_version = SPARK_STAGE_PLAN_ABI_VERSION;
    stage_plan.descriptor_bytes = SPARK_STAGE_PLAN_DESCRIPTOR_BYTES;
    stage_plan.stage_count = 13u;
    for (stage_index = 0u; stage_index < 13u; ++stage_index)
    {
        stage_plan.stages[stage_index].first_layer_index = stage_index * 6u;
        stage_plan.stages[stage_index].layer_count = 6u;
        stage_plan.stages[stage_index].flags =
            SPARK_STAGE_PLAN_STAGE_FLAG_INPUT_HIDDEN |
            SPARK_STAGE_PLAN_STAGE_FLAG_OUTPUT_HIDDEN;
    }
    stage_plan.stages[0].flags |=
        SPARK_STAGE_PLAN_STAGE_FLAG_DENSE_PREFIX;
    stage_plan.stages[12].flags =
        SPARK_STAGE_PLAN_STAGE_FLAG_INPUT_HIDDEN |
        SPARK_STAGE_PLAN_STAGE_FLAG_FINAL_TOKEN;

    assert(SparkStagePlanValidate(
        &stage_plan,
        error_buffer,
        sizeof(error_buffer)) == SPARK_STATUS_OK);
}

static void SparkTestGlm52StagePlanLayerCountTable(void)
{
    const uint32_t layer_counts[12] =
        {6u, 7u, 7u, 7u, 7u, 7u, 7u, 6u, 6u, 6u, 6u, 6u};
    uint32_t invalid_layer_counts[12];
    SparkStagePlan stage_plan;
    char error_buffer[256];

    assert(SparkStagePlanBuildFromLayerCounts(
        layer_counts,
        12u,
        &stage_plan,
        error_buffer,
        sizeof(error_buffer)) == SPARK_STATUS_OK);
    assert(stage_plan.stage_count == 12u);
    assert(stage_plan.stages[0u].first_layer_index == 0u);
    assert(stage_plan.stages[0u].layer_count == 6u);
    assert(stage_plan.stages[1u].first_layer_index == 6u);
    assert(stage_plan.stages[1u].layer_count == 7u);
    assert(stage_plan.stages[11u].first_layer_index == 72u);
    assert(stage_plan.stages[11u].layer_count == 6u);
    assert((stage_plan.stages[11u].flags &
        SPARK_STAGE_PLAN_STAGE_FLAG_FINAL_TOKEN) != 0u);

    memcpy(invalid_layer_counts,layer_counts,sizeof(layer_counts));
    invalid_layer_counts[11u] = 5u;
    assert(SparkStagePlanBuildFromLayerCounts(
        invalid_layer_counts,
        12u,
        &stage_plan,
        error_buffer,
        sizeof(error_buffer)) == SPARK_STATUS_INVALID_ARGUMENT);
}

static void SparkTestGlm52StagePlanBuilderAndBuckets(void)
{
    SparkStagePlan stage_plan;
    uint64_t layer_cost_ns[SPARK_STAGE_PLAN_LAYER_COUNT];
    char error_buffer[256];
    uint32_t layer_index;
    uint32_t bucket;

    for (layer_index = 0u;
         layer_index < SPARK_STAGE_PLAN_LAYER_COUNT;
         ++layer_index)
    {
        layer_cost_ns[layer_index] = 1000u + (uint64_t)layer_index;
    }
    assert(SparkStagePlanBuildBalanced(
        layer_cost_ns,
        13u,
        &stage_plan,
        error_buffer,
        sizeof(error_buffer)) == SPARK_STATUS_OK);
    assert(stage_plan.stage_count == 13u);
    assert(stage_plan.stages[0].first_layer_index == 0u);
    assert(stage_plan.stages[12].first_layer_index +
        stage_plan.stages[12].layer_count ==
            SPARK_STAGE_PLAN_LAYER_COUNT);
    assert((stage_plan.stages[12].flags &
        SPARK_STAGE_PLAN_STAGE_FLAG_FINAL_TOKEN) != 0u);

    assert(SparkStagePlanSelectBatchBucket(1u, &bucket) ==
        SPARK_STATUS_OK);
    assert(bucket == SPARK_STAGE_PLAN_BUCKET_B16);
    assert(SparkStagePlanSelectBatchBucket(17u, &bucket) ==
        SPARK_STATUS_OK);
    assert(bucket == SPARK_STAGE_PLAN_BUCKET_B32);
    assert(SparkStagePlanSelectBatchBucket(33u, &bucket) ==
        SPARK_STATUS_OK);
    assert(bucket == SPARK_STAGE_PLAN_BUCKET_B64);
    assert(SparkStagePlanSelectBatchBucket(65u, &bucket) ==
        SPARK_STATUS_OK);
    assert(bucket == SPARK_STAGE_PLAN_BUCKET_B128);
    assert(SparkStagePlanSelectBatchBucket(129u, &bucket) ==
        SPARK_STATUS_OK);
    assert(bucket == SPARK_STAGE_PLAN_BUCKET_B256);
    assert(SparkStagePlanSelectBatchBucket(257u, &bucket) ==
        SPARK_STATUS_OK);
    assert(bucket == SPARK_STAGE_PLAN_BUCKET_B512);
    assert(SparkStagePlanSelectBatchBucket(513u, &bucket) ==
        SPARK_STATUS_OK);
    assert(bucket == SPARK_STAGE_PLAN_BUCKET_B1024);
    assert(SparkStagePlanSelectBatchBucket(1025u, &bucket) ==
        SPARK_STATUS_CAPACITY_EXCEEDED);
}


static void SparkTestGlm52StagePlanMeasuredBalanced(void)
{
    SparkStagePlan stage_plan;
    uint64_t layer_cost_ns[SPARK_STAGE_PLAN_LAYER_COUNT];
    uint64_t final_stage_extra_cost_ns;
    char error_buffer[256];

    assert(SparkStagePlanLoadMeasuredCostProfile(
        SPARK_STAGE_PLAN_MEASURED_PROFILE_20260701,
        SPARK_STAGE_PLAN_BUCKET_B64,
        layer_cost_ns,
        &final_stage_extra_cost_ns) == SPARK_STATUS_OK);
    assert(final_stage_extra_cost_ns == 0u);
    assert(SparkStagePlanBuildCurrentSparkMeasuredBalanced(
        SPARK_STAGE_PLAN_MEASURED_PROFILE_20260701,
        SPARK_STAGE_PLAN_BUCKET_B64,
        &stage_plan,
        error_buffer,
        sizeof(error_buffer)) == SPARK_STATUS_OK);
    assert(stage_plan.stage_count == 13u);
    assert(stage_plan.stages[0].first_layer_index == 0u);
    assert(stage_plan.stages[0].layer_count == 6u);
    assert(stage_plan.stages[1].first_layer_index == 6u);
    assert(stage_plan.stages[1].layer_count == 6u);
    assert(stage_plan.stages[2].first_layer_index == 12u);
    assert(stage_plan.stages[2].layer_count == 6u);
    assert(stage_plan.stages[3].first_layer_index == 18u);
    assert(stage_plan.stages[3].layer_count == 6u);
    assert(stage_plan.stages[4].first_layer_index == 24u);
    assert(stage_plan.stages[4].layer_count == 6u);
    assert(stage_plan.stages[12].first_layer_index == 72u);
    assert(stage_plan.stages[12].layer_count == 6u);
    assert((stage_plan.stages[12].flags &
        SPARK_STAGE_PLAN_STAGE_FLAG_FINAL_TOKEN) != 0u);
    assert(SparkTestGlm52StagePlanMaximumStageCostNs(
        &stage_plan,
        layer_cost_ns,
        final_stage_extra_cost_ns) == 50660288u);

    assert(SparkStagePlanLoadMeasuredCostProfile(
        SPARK_STAGE_PLAN_MEASURED_PROFILE_20260701,
        SPARK_STAGE_PLAN_BUCKET_B128,
        layer_cost_ns,
        &final_stage_extra_cost_ns) == SPARK_STATUS_OK);
    assert(final_stage_extra_cost_ns == 0u);
    assert(SparkStagePlanBuildCurrentSparkMeasuredBalanced(
        SPARK_STAGE_PLAN_MEASURED_PROFILE_20260701,
        SPARK_STAGE_PLAN_BUCKET_B128,
        &stage_plan,
        error_buffer,
        sizeof(error_buffer)) == SPARK_STATUS_OK);
    assert(stage_plan.stage_count == 13u);
    assert(stage_plan.stages[0].first_layer_index == 0u);
    assert(stage_plan.stages[0].layer_count == 6u);
    assert(stage_plan.stages[12].first_layer_index == 72u);
    assert(stage_plan.stages[12].layer_count == 6u);
    assert(SparkTestGlm52StagePlanMaximumStageCostNs(
        &stage_plan,
        layer_cost_ns,
        final_stage_extra_cost_ns) == 197361562u);

    assert(SparkStagePlanLoadMeasuredCostProfile(
        SPARK_STAGE_PLAN_MEASURED_PROFILE_20260701,
        SPARK_STAGE_PLAN_BUCKET_B256,
        layer_cost_ns,
        &final_stage_extra_cost_ns) == SPARK_STATUS_OK);
    assert(final_stage_extra_cost_ns == 0u);
    assert(SparkStagePlanBuildCurrentSparkMeasuredBalanced(
        SPARK_STAGE_PLAN_MEASURED_PROFILE_20260701,
        SPARK_STAGE_PLAN_BUCKET_B256,
        &stage_plan,
        error_buffer,
        sizeof(error_buffer)) == SPARK_STATUS_OK);
    assert(stage_plan.stage_count == 13u);
    assert(stage_plan.stages[0].first_layer_index == 0u);
    assert(stage_plan.stages[0].layer_count == 6u);
    assert(stage_plan.stages[12].first_layer_index == 72u);
    assert(stage_plan.stages[12].layer_count == 6u);

    assert(SparkStagePlanLoadMeasuredCostProfile(
        SPARK_STAGE_PLAN_MEASURED_PROFILE_20260701,
        SPARK_STAGE_PLAN_BUCKET_B32,
        layer_cost_ns,
        &final_stage_extra_cost_ns) == SPARK_STATUS_OK);
    assert(final_stage_extra_cost_ns == 39342000u);
    assert(SparkStagePlanBuildCurrentSparkMeasuredBalanced(
        SPARK_STAGE_PLAN_MEASURED_PROFILE_20260701,
        SPARK_STAGE_PLAN_BUCKET_B32,
        &stage_plan,
        error_buffer,
        sizeof(error_buffer)) == SPARK_STATUS_OK);
    assert(stage_plan.stage_count == SPARK_STAGE_PLAN_CURRENT_SPARK_COUNT);
    assert(stage_plan.stages[0].first_layer_index == 0u);
    assert(stage_plan.stages[0].layer_count == 10u);
    assert(stage_plan.stages[12].first_layer_index == 77u);
    assert(stage_plan.stages[12].layer_count == 1u);
    assert((stage_plan.stages[12].flags &
        SPARK_STAGE_PLAN_STAGE_FLAG_FINAL_TOKEN) != 0u);
    assert(SparkTestGlm52StagePlanMaximumStageCostNs(
        &stage_plan,
        layer_cost_ns,
        final_stage_extra_cost_ns) == 63119000u);

    assert(SparkStagePlanBuildMeasuredBalanced(
        SPARK_STAGE_PLAN_MEASURED_PROFILE_20260701,
        SPARK_STAGE_PLAN_BUCKET_B32,
        SPARK_STAGE_PLAN_CURRENT_SPARK_COUNT + 1u,
        &stage_plan,
        error_buffer,
        sizeof(error_buffer)) == SPARK_STATUS_INVALID_ARGUMENT);
}


static void SparkTestGlm52StagePlanMeasuredBalancedQuantizationModes(void)
{
    SparkStagePlan stage_plan_4bit;
    SparkStagePlan stage_plan_8bit;
    uint64_t layer_cost_ns[SPARK_STAGE_PLAN_LAYER_COUNT];
    uint64_t final_stage_extra_cost_ns;
    char error_buffer[256];

    assert(SparkStagePlanLoadMeasuredCostProfileForQuantization(
        SPARK_STAGE_PLAN_MEASURED_PROFILE_20260701,
        SPARK_STAGE_PLAN_BUCKET_B64,
        SPARK_STAGE_PLAN_QUANTIZATION_NVFP4_4BIT,
        layer_cost_ns,
        &final_stage_extra_cost_ns) == SPARK_STATUS_OK);
    assert(final_stage_extra_cost_ns == 0u);
    assert(SparkStagePlanLoadMeasuredCostProfileForQuantization(
        SPARK_STAGE_PLAN_MEASURED_PROFILE_20260701,
        SPARK_STAGE_PLAN_BUCKET_B64,
        SPARK_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT,
        layer_cost_ns,
        &final_stage_extra_cost_ns) == SPARK_STATUS_OK);
    assert(final_stage_extra_cost_ns == 0u);
    assert(SparkStagePlanLoadMeasuredCostProfileForQuantization(
        SPARK_STAGE_PLAN_MEASURED_PROFILE_20260701,
        SPARK_STAGE_PLAN_BUCKET_B128,
        SPARK_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT,
        layer_cost_ns,
        &final_stage_extra_cost_ns) == SPARK_STATUS_OK);
    assert(final_stage_extra_cost_ns == 0u);
    assert(SparkStagePlanLoadMeasuredCostProfileForQuantization(
        SPARK_STAGE_PLAN_MEASURED_PROFILE_20260701,
        SPARK_STAGE_PLAN_BUCKET_B128,
        layer_cost_ns,
        &final_stage_extra_cost_ns) == SPARK_STATUS_OK);
    assert(final_stage_extra_cost_ns == 0u);

    assert(SparkStagePlanBuildCurrentSparkMeasuredBalancedForQuantization(
        SPARK_STAGE_PLAN_MEASURED_PROFILE_20260701,
        SPARK_STAGE_PLAN_BUCKET_B64,
        SPARK_STAGE_PLAN_QUANTIZATION_NVFP4_4BIT,
        &stage_plan_4bit,
        error_buffer,
        sizeof(error_buffer)) == SPARK_STATUS_OK);
    assert(SparkStagePlanBuildCurrentSparkMeasuredBalancedForQuantization(
        SPARK_STAGE_PLAN_MEASURED_PROFILE_20260701,
        SPARK_STAGE_PLAN_BUCKET_B64,
        SPARK_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT,
        &stage_plan_8bit,
        error_buffer,
        sizeof(error_buffer)) == SPARK_STATUS_OK);
    assert(SparkStagePlanBuildCurrentSparkMeasuredBalancedForQuantization(
        SPARK_STAGE_PLAN_MEASURED_PROFILE_20260701,
        SPARK_STAGE_PLAN_BUCKET_B64,
        error_buffer,
        sizeof(error_buffer)) == SPARK_STATUS_OK);
    assert(memcmp(
        &stage_plan_4bit,
        &stage_plan_8bit,
        sizeof(stage_plan_4bit)) == 0);
    assert(memcmp(
        &stage_plan_8bit,
        sizeof(stage_plan_8bit)) == 0);
    assert(SparkStagePlanBuildCurrentSparkMeasuredBalancedForQuantization(
        SPARK_STAGE_PLAN_MEASURED_PROFILE_20260701,
        SPARK_STAGE_PLAN_BUCKET_B64,
        99u,
        &stage_plan_8bit,
        error_buffer,
        sizeof(error_buffer)) == SPARK_STATUS_INVALID_ARGUMENT);
}

static void SparkTestGlm52StagePlanInvalidCuts(void)
{
    SparkStagePlan stage_plan;
    char error_buffer[256];

    assert(SparkStagePlanBuildUniform(
        13u,
        &stage_plan,
        error_buffer,
        sizeof(error_buffer)) == SPARK_STATUS_OK);
    stage_plan.stages[1].layer_count = 9u;
    stage_plan.stages[2].first_layer_index = 15u;
    assert(SparkStagePlanValidate(
        &stage_plan,
        error_buffer,
        sizeof(error_buffer)) == SPARK_STATUS_INVALID_ARGUMENT);

    assert(SparkStagePlanBuildUniform(
        13u,
        &stage_plan,
        error_buffer,
        sizeof(error_buffer)) == SPARK_STATUS_OK);
    stage_plan.stages[0].flags |=
        SPARK_STAGE_PLAN_STAGE_FLAG_FINAL_TOKEN;
    assert(SparkStagePlanValidate(
        &stage_plan,
        error_buffer,
        sizeof(error_buffer)) == SPARK_STATUS_INVALID_ARGUMENT);
}

int main(void)
{
    SparkTestGlm52StagePlanValidRing();
    SparkTestGlm52StagePlanLayerCountTable();
    SparkTestGlm52StagePlanBuilderAndBuckets();
    SparkTestGlm52StagePlanMeasuredBalanced();
    SparkTestGlm52StagePlanMeasuredBalancedQuantizationModes();
    SparkTestGlm52StagePlanInvalidCuts();
    return 0;
}
