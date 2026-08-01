#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "sparkpipe/spark_ring_runtime.h"

#define SPARK_TEST_PACK_ROOT "build/test_glm52_ring_runtime_packs"

static void SparkTestWriteFile(const char *path)
{
    FILE *file;

    file = fopen(path, "wb");
    assert(file != 0);
    assert(fputs("x\n", file) >= 0);
    assert(fclose(file) == 0);
}

static void SparkTestRemoveFile(const char *path)
{
    if (remove(path) != 0)
    {
        assert(access(path, F_OK) != 0);
    }
}

static void SparkTestResetPackRoot(void)
{
    (void)mkdir("build", 0775);
    (void)mkdir(SPARK_TEST_PACK_ROOT, 0775);
    SparkTestRemoveFile(SPARK_TEST_PACK_ROOT "/fp8_moe_pack_manifest.json");
    SparkTestRemoveFile(SPARK_TEST_PACK_ROOT "/resident_moe_pack_manifest.json");
    SparkTestRemoveFile(SPARK_TEST_PACK_ROOT "/glm52_layer_0003_fp8_moe.spfp8");
    SparkTestRemoveFile(SPARK_TEST_PACK_ROOT "/glm52_layer_0004_fp8_moe.spfp8");
    SparkTestRemoveFile(SPARK_TEST_PACK_ROOT "/glm52_layer_0005_fp8_moe.spfp8");
    SparkTestRemoveFile(SPARK_TEST_PACK_ROOT "/glm52_layer_0003_b12x_moe.spb12x");
    SparkTestRemoveFile(SPARK_TEST_PACK_ROOT "/glm52_layer_0004_b12x_moe.spb12x");
    SparkTestRemoveFile(SPARK_TEST_PACK_ROOT "/glm52_layer_0005_b12x_moe.spb12x");
}

static void SparkTestGlm52RingRuntimeRankPlan(void)
{
    SparkRingRuntimeRankPlan rank_plan;
    char error_buffer[256];
    char host_name[16];

    assert(SparkRingRuntimeRankHostName(
        0u, host_name, sizeof(host_name)) == SPARK_STATUS_OK);
    assert(strcmp(host_name, "10.10.100.10") == 0);
    assert(SparkRingRuntimeRankHostName(
        10u, host_name, sizeof(host_name)) == SPARK_STATUS_OK);
    assert(strcmp(host_name, "10.10.100.20") == 0);
    assert(SparkRingRuntimeRankHostName(
        12u, host_name, sizeof(host_name)) == SPARK_STATUS_OK);
    assert(strcmp(host_name, "10.10.100.22") == 0);
    assert(SparkRingRuntimeBuildRankPlan(
        0u,
        1024u,
        52100u,
        SPARK_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT,
        &rank_plan,
        error_buffer,
        sizeof(error_buffer)) == SPARK_STATUS_OK);
    assert(rank_plan.first_layer_index == 0u);
    assert(rank_plan.layer_count == 6u);
    assert(rank_plan.listen_port == 52100u);
    assert(rank_plan.next_port == 52101u);
    assert(rank_plan.logical_lane_capacity == 1024u);
    assert(rank_plan.maximum_speculative_rows_per_lane == 8u);
    assert(rank_plan.execution_row_capacity == 1024u);
    assert(rank_plan.output_endpoint.max_active_sequence_count == 1024u);
    assert((rank_plan.flags & SPARK_RING_RUNTIME_RANK_FLAG_DENSE_PREFIX) != 0u);
    assert((rank_plan.flags & SPARK_RING_RUNTIME_RANK_FLAG_HAS_PREVIOUS) == 0u);
    assert(strcmp(rank_plan.next_host_name, "10.10.100.11") == 0);
    assert(strcmp(
        rank_plan.output_route_name,
        "10.10.100.10_to_10.10.100.11_hidden") == 0);
    assert(rank_plan.quantization_mode ==
        SPARK_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT);
    assert(SparkRingRuntimeExecutionRowCapacity(1u) == 8u);
    assert(SparkRingRuntimeExecutionRowCapacity(64u) == 512u);
    assert(SparkRingRuntimeExecutionRowCapacity(256u) == 1024u);
    assert(SparkRingRuntimeExecutionRowCapacity(1024u) == 1024u);
    assert(SparkRingRuntimeBuildRankPlan(
        12u,
        1024u,
        52100u,
        SPARK_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT,
        &rank_plan,
        error_buffer,
        sizeof(error_buffer)) == SPARK_STATUS_OK);
    assert(rank_plan.first_layer_index == 72u);
    assert(rank_plan.layer_count == 6u);
    assert(rank_plan.listen_port == 52112u);
    assert(rank_plan.next_port == 0u);
    assert((rank_plan.flags & SPARK_RING_RUNTIME_RANK_FLAG_FINAL_STAGE) != 0u);
    assert((rank_plan.flags & SPARK_RING_RUNTIME_RANK_FLAG_HAS_NEXT) == 0u);
    assert(strcmp(rank_plan.previous_host_name, "10.10.100.21") == 0);
    assert(strcmp(
        rank_plan.input_route_name,
        "10.10.100.21_to_10.10.100.22_hidden") == 0);
}

static void SparkTestGlm52RingRuntimeDsaCandidateBucket(void)
{
    assert(SparkRingRuntimeDsaCandidateBucket(0u) == 0u);
    assert(SparkRingRuntimeDsaCandidateBucket(1u) == 2048u);
    assert(SparkRingRuntimeDsaCandidateBucket(2048u) == 2048u);
    assert(SparkRingRuntimeDsaCandidateBucket(2049u) == 4096u);
    assert(SparkRingRuntimeDsaCandidateBucket(65536u) == 65536u);
    assert(SparkRingRuntimeDsaCandidateBucket(1048575u) == 1048576u);
    assert(SparkRingRuntimeDsaCandidateBucket(1048576u) == 1048576u);
    assert(SparkRingRuntimeDsaCandidateBucket(1048577u) == 0u);
}

static void SparkTestWriteStagePacks(
    const SparkRingRuntimeRankPlan *rank_plan,
    uint32_t quantization_mode)
{
    char pack_path[SPARK_RING_RUNTIME_PACK_PATH_BYTES];
    uint32_t layer_index;

    for (layer_index = SPARK_GLM52_MODEL_FIRST_ROUTED_LAYER;
         layer_index < rank_plan->first_layer_index + rank_plan->layer_count;
         ++layer_index)
    {
        assert(SparkRingRuntimeBuildMoePackPath(
            SPARK_TEST_PACK_ROOT,
            quantization_mode,
            layer_index,
            rank_plan->tp_degree,
            rank_plan->tp_rank,
            pack_path,
            sizeof(pack_path)) == SPARK_STATUS_OK);
        SparkTestWriteFile(pack_path);
    }
}

static void SparkTestGlm52RingRuntimeMoePacks(void)
{
    SparkRingRuntimeRankPlan rank_plan;
    char error_buffer[256];
    char pack_path[SPARK_RING_RUNTIME_PACK_PATH_BYTES];
    uint32_t backend_kind;
    uint32_t quantization_mode;

    SparkTestResetPackRoot();
    assert(SparkRingRuntimeBuildRankPlan(
        0u,
        1024u,
        52100u,
        SPARK_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT,
        &rank_plan,
        error_buffer,
        sizeof(error_buffer)) == SPARK_STATUS_OK);
    assert(SparkRingRuntimeValidateStageMoePackFiles(
        &rank_plan,
        SPARK_TEST_PACK_ROOT,
        error_buffer,
        sizeof(error_buffer)) == SPARK_STATUS_NOT_FOUND);
    SparkTestWriteFile(SPARK_TEST_PACK_ROOT "/fp8_moe_pack_manifest.json");
    SparkTestWriteStagePacks(
        &rank_plan,
        SPARK_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT);
    assert(SparkRingRuntimeValidateStageMoePackFiles(
        &rank_plan,
        SPARK_TEST_PACK_ROOT,
        error_buffer,
        sizeof(error_buffer)) == SPARK_STATUS_OK);
    assert(SparkRingRuntimeBuildMoePackPath(
        SPARK_TEST_PACK_ROOT,
        rank_plan.quantization_mode,
        SPARK_GLM52_MODEL_MTP_LAYER_INDEX,
        1u,
        0u,
        pack_path,
        sizeof(pack_path)) == SPARK_STATUS_OK);
    assert(strcmp(
        pack_path,
        SPARK_TEST_PACK_ROOT "/glm52_layer_0078_fp8_moe.spfp8") == 0);
    assert(SparkRingRuntimeBuildMoePackPath(
        SPARK_TEST_PACK_ROOT,
        rank_plan.quantization_mode,
        SPARK_GLM52_MODEL_WEIGHT_LAYER_COUNT,
        1u,
        0u,
        pack_path,
        sizeof(pack_path)) == SPARK_STATUS_INVALID_ARGUMENT);

    SparkTestWriteFile(SPARK_TEST_PACK_ROOT "/resident_moe_pack_manifest.json");
    assert(SparkRingRuntimeValidateStageMoePackFiles(
        &rank_plan,
        SPARK_TEST_PACK_ROOT,
        error_buffer,
        sizeof(error_buffer)) == SPARK_STATUS_MODULE_NOT_VALIDATED);

    SparkTestResetPackRoot();
    SparkTestWriteFile(SPARK_TEST_PACK_ROOT "/resident_moe_pack_manifest.json");
    assert(SparkRingRuntimeBuildRankPlan(
        0u,
        1024u,
        52100u,
        SPARK_STAGE_PLAN_QUANTIZATION_NVFP4_4BIT,
        &rank_plan,
        error_buffer,
        sizeof(error_buffer)) == SPARK_STATUS_OK);
    assert(SparkRingRuntimeValidateStageMoePackFiles(
        &rank_plan,
        SPARK_TEST_PACK_ROOT,
        error_buffer,
        sizeof(error_buffer)) == SPARK_STATUS_NOT_FOUND);
    SparkTestWriteStagePacks(
        &rank_plan,
        SPARK_STAGE_PLAN_QUANTIZATION_NVFP4_4BIT);
    assert(SparkRingRuntimeValidateStageMoePackFiles(
        &rank_plan,
        SPARK_TEST_PACK_ROOT,
        error_buffer,
        sizeof(error_buffer)) == SPARK_STATUS_OK);
    assert(SparkRingRuntimeBuildMoePackPath(
        SPARK_TEST_PACK_ROOT,
        UINT32_MAX,
        3u,
        1u,
        0u,
        pack_path,
        sizeof(pack_path)) == SPARK_STATUS_INVALID_ARGUMENT);

    assert(SparkRingRuntimeParseQuantizationMode(
        "fp8", &quantization_mode) == SPARK_STATUS_OK);
    assert(strcmp(
        SparkRingRuntimeQuantizationModeName(quantization_mode),
        "fp8") == 0);
    assert(SparkRingRuntimeParseQuantizationMode(
        "nvfp4", &quantization_mode) == SPARK_STATUS_OK);
    assert(strcmp(
        SparkRingRuntimeQuantizationModeName(quantization_mode),
        "nvfp4") == 0);
    assert(SparkRingRuntimeParseQuantizationMode(
        "auto", &quantization_mode) == SPARK_STATUS_INVALID_ARGUMENT);
    assert(SparkRingRuntimeQuantizationModeName(UINT32_MAX) == 0);

    assert(SparkRingRuntimeValidateFp8PlanCounts(
        SPARK_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT,
        42u,
        42u) == SPARK_STATUS_OK);
    assert(SparkRingRuntimeValidateFp8PlanCounts(
        SPARK_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT,
        0u,
        0u) == SPARK_STATUS_MODULE_NOT_VALIDATED);
    assert(SparkRingRuntimeValidateFp8PlanCounts(
        SPARK_STAGE_PLAN_QUANTIZATION_NVFP4_4BIT,
        0u,
        0u) == SPARK_STATUS_OK);
    assert(SparkRingRuntimeValidateFp8PlanCounts(
        SPARK_STAGE_PLAN_QUANTIZATION_NVFP4_4BIT,
        1u,
        1u) == SPARK_STATUS_MODULE_NOT_VALIDATED);
    assert(SparkRingRuntimeValidateFp8PlanCounts(
        UINT32_MAX,
        0u,
        0u) == SPARK_STATUS_INVALID_ARGUMENT);

    assert(SparkRingRuntimeExpectedMoeBackendKind(
        SPARK_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT,
        &backend_kind) == SPARK_STATUS_OK);
    assert(backend_kind == SPARK_RING_RUNTIME_MOE_BACKEND_FP8_FLASHINFER_GROUPED);
    assert(SparkRingRuntimeExpectedMoeBackendKind(
        SPARK_STAGE_PLAN_QUANTIZATION_NVFP4_4BIT,
        &backend_kind) == SPARK_STATUS_OK);
    assert(backend_kind == SPARK_RING_RUNTIME_MOE_BACKEND_NVFP4_B12X);
    assert(SparkRingRuntimeExpectedMoeBackendKind(
        UINT32_MAX,
        &backend_kind) == SPARK_STATUS_INVALID_ARGUMENT);
    assert(SparkRingRuntimeExpectedMoeBackendKind(
        SPARK_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT,
        0) == SPARK_STATUS_INVALID_ARGUMENT);

    SparkTestResetPackRoot();
    assert(rmdir(SPARK_TEST_PACK_ROOT) == 0);
}

static void SparkTestGlm52RingRuntimeFinalEventRoute(void)
{
    SparkRingRuntimeFinalEventRoute route;
    char error_buffer[256];

    assert(SparkRingRuntimeBuildFinalEventRoute(
        52100u,
        &route,
        error_buffer,
        sizeof(error_buffer)) == SPARK_STATUS_OK);
    assert(route.source_rank_index == 12u);
    assert(route.sink_rank_index == 0u);
    assert(route.listen_port == 52300u);
    assert(route.connect_port == 52300u);
    assert(strcmp(route.source_host_name, "10.10.100.22") == 0);
    assert(strcmp(route.sink_host_name, "10.10.100.10") == 0);
    assert(strcmp(
        route.route_name,
        "10.10.100.22_to_10.10.100.10_final_events") == 0);
    assert(SparkRingRuntimeValidateFinalEventRoute(
        &route,
        error_buffer,
        sizeof(error_buffer)) == SPARK_STATUS_OK);
    route.sink_rank_index = 1u;
    assert(SparkRingRuntimeValidateFinalEventRoute(
        &route,
        error_buffer,
        sizeof(error_buffer)) == SPARK_STATUS_INVALID_ARGUMENT);
}

static void SparkTestRingRuntimeShapePlans(void)
{
    SparkRingRuntimeRankPlan plan;
    SparkTpShapeDescriptor shape;
    char error_buffer[256];
    char pack_path[512];

    assert(SparkRingRuntimeBuildRankPlan(
        5u,
        16u,
        42000u,
        SPARK_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT,
        &plan,
        error_buffer,
        sizeof(error_buffer)) == SPARK_STATUS_OK);
    assert(plan.tp_degree == 1u && plan.tp_rank == 0u);
    assert(plan.pp_stage_count == 13u && plan.pp_stage_index == 5u);
    assert(plan.shape_configuration_hash != 0u);

    memset(&shape, 0, sizeof(shape));
    shape.abi_version = SPARK_TP_SHARD_ABI_VERSION;
    shape.tp_degree = 4u;
    shape.tp_rank = 2u;
    shape.pp_stage_count = 3u;
    shape.pp_stage_index = 1u;
    assert(SparkRingRuntimeBuildShapeRankPlan(
        &shape,
        16u,
        42000u,
        43000u,
        SPARK_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT,
        &plan,
        error_buffer,
        sizeof(error_buffer)) == SPARK_STATUS_OK);
    assert(plan.rank_index == 6u);
    assert(plan.first_layer_index == 26u && plan.layer_count == 26u);
    assert(plan.previous_rank_index == 2u && plan.next_rank_index == 10u);
    assert(plan.tp_collective_listen_port == 43006u);
    assert(plan.tp_peer_ports[0] == 43007u && plan.tp_peer_ports[1] == 43004u);
    assert(plan.tp_peer_host_names[0][0] != '\0');
    assert((plan.flags & SPARK_RING_RUNTIME_RANK_FLAG_DENSE_PREFIX) == 0u);
    assert((plan.flags & SPARK_RING_RUNTIME_RANK_FLAG_FINAL_STAGE) == 0u);
    assert(SparkRingRuntimeBuildMoePackPath(
        "packs",
        plan.quantization_mode,
        7u,
        plan.tp_degree,
        plan.tp_rank,
        pack_path,
        sizeof(pack_path)) == SPARK_STATUS_OK);
    assert(strstr(pack_path, "_tp4r2.spfp8") != 0);
    plan.shape_configuration_hash += 1u;
    assert(SparkRingRuntimeValidateRankPlan(
        &plan,
        error_buffer,
        sizeof(error_buffer)) != SPARK_STATUS_OK);

    shape.tp_degree = 16u;
    shape.tp_rank = 15u;
    shape.pp_stage_count = 1u;
    shape.pp_stage_index = 0u;
    assert(SparkRingRuntimeBuildShapeRankPlan(
        &shape,
        16u,
        42000u,
        43000u,
        SPARK_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT,
        &plan,
        error_buffer,
        sizeof(error_buffer)) != SPARK_STATUS_OK);
}

int main(void)
{
    SparkTestRingRuntimeShapePlans();
    SparkTestGlm52RingRuntimeRankPlan();
    SparkTestGlm52RingRuntimeDsaCandidateBucket();
    SparkTestGlm52RingRuntimeMoePacks();
    SparkTestGlm52RingRuntimeFinalEventRoute();
    return 0;
}
