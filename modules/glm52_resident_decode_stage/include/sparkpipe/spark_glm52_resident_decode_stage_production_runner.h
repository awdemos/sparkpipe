#pragma once

#include <stdint.h>

#include "sparkpipe/spark_glm52_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_hidden_transport.h"
#include "sparkpipe/spark_model_driver.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_ABI_VERSION 3u
#define SPARK_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_CONFIGURATION_BYTES \
    ((uint32_t)sizeof(SparkResidentDecodeStageProductionRunnerConfiguration))
#define SPARK_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_BYTES \
    ((uint32_t)sizeof(SparkResidentDecodeStageProductionRunner))
#define SPARK_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_DISPATCH_BYTES \
    ((uint32_t)sizeof(SparkResidentDecodeStageProductionRunnerDispatch))
#define SPARK_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_STATS_BYTES \
    ((uint32_t)sizeof(SparkResidentDecodeStageProductionRunnerStats))

#define SPARK_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_FLAG_REQUIRE_ADMISSION \
    0x00000001u
#define SPARK_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_FLAG_REQUIRE_INPUT_TRANSPORT \
    0x00000002u
#define SPARK_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_FLAG_REQUIRE_OUTPUT_TRANSPORT \
    0x00000004u
#define SPARK_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_DEFAULT_FLAGS \
    (SPARK_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_FLAG_REQUIRE_ADMISSION | \
     SPARK_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_FLAG_REQUIRE_INPUT_TRANSPORT | \
     SPARK_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_FLAG_REQUIRE_OUTPUT_TRANSPORT)
#define SPARK_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_KNOWN_FLAGS \
    SPARK_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_DEFAULT_FLAGS

#define SPARK_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_REQUIRED_PROGRAM_FLAGS \
    (SPARK_MODEL_DRIVER_PROGRAM_FLAG_STREAM_ORDERED | \
     SPARK_MODEL_DRIVER_PROGRAM_FLAG_DRIVER_OWNS_RESIDENT_STATE | \
     SPARK_MODEL_DRIVER_PROGRAM_FLAG_DRIVER_OWNS_KV_CACHE | \
     SPARK_MODEL_DRIVER_PROGRAM_FLAG_FIXED_FIRMWARE | \
     SPARK_MODEL_DRIVER_PROGRAM_FLAG_REQUIRES_HIDDEN_TRANSPORT | \
     SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_FILE_TRANSPORT | \
     SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_SHELL_TRANSPORT)

#define SPARK_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_DISPATCH_FLAG_PREFILL \
    SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL
#define SPARK_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_DISPATCH_FLAG_MTP_TREE_VERIFY \
    0x00000004u
#define SPARK_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_DISPATCH_FLAG_HIDDEN_INPUT_PRERECEIVED \
    0x00000008u
#define SPARK_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_DISPATCH_KNOWN_FLAGS \
    (SPARK_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_DISPATCH_FLAG_PREFILL | \
     SPARK_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_DISPATCH_FLAG_MTP_TREE_VERIFY | \
     SPARK_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_DISPATCH_FLAG_HIDDEN_INPUT_PRERECEIVED)

typedef struct SparkResidentDecodeStageProductionRunnerConfiguration
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t flags;
    uint32_t reserved0;
    const SparkModelDriverInterface *driver_interface;
    void *driver_instance;
    const SparkModelDriverProgramDescriptor *program;
    void *execution_stream;
} SparkResidentDecodeStageProductionRunnerConfiguration;

typedef struct SparkResidentDecodeStageProductionRunnerDispatch
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t flags;
    uint32_t priority;
    uint64_t request_id;
    uint64_t sequence_id;
    uint64_t sequence_position;
    uint64_t deadline_time_ns;
    uint32_t active_sequence_count;
    uint32_t new_token_count;
    uint32_t pipeline_slot;
    uint32_t logical_lane_count;
    uint32_t rows_per_lane;
    uint32_t reserved0;
    SparkModelDriverResidencyToken residency;
    const uint32_t *mtp_draft_token_budgets;
    const SparkGlm52DsparkHiddenTapPlan *dspark_hidden_tap_plan;
    void *const *dspark_hidden_tap_outputs_bf16;
    uint64_t dspark_hidden_tap_lane_stride_bytes;
    const SparkKvBlockTableView *kv_block_table;
    const SparkResidentDecodeStagePrefillFrameView *prefill_view;
    SparkHiddenTransportSession *hidden_input_transport_session;
    SparkHiddenTransportSession *hidden_output_transport_session;
    SparkHiddenTransportPacket hidden_input_packet;
    SparkHiddenTransportPacket hidden_output_packet;
    SparkModelDriverCompletionFunction completion_function;
    void *completion_context;
} SparkResidentDecodeStageProductionRunnerDispatch;

typedef struct SparkResidentDecodeStageProductionRunnerStats
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t last_status;
    uint32_t last_admission_rejection;
    uint64_t submitted_count;
    uint64_t admitted_count;
    uint64_t rejected_count;
    uint64_t submit_failed_count;
} SparkResidentDecodeStageProductionRunnerStats;

typedef struct SparkResidentDecodeStageProductionRunner
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t flags;
    uint32_t program_id;
    const SparkModelDriverInterface *driver_interface;
    void *driver_instance;
    const SparkModelDriverProgramDescriptor *program;
    void *execution_stream;
    SparkResidentDecodeStageProductionRunnerStats stats;
} SparkResidentDecodeStageProductionRunner;

SparkStatus SparkResidentDecodeStageProductionRunnerInitialize(
    SparkResidentDecodeStageProductionRunner *runner,
    const SparkResidentDecodeStageProductionRunnerConfiguration *configuration);

SparkStatus SparkResidentDecodeStageProductionRunnerSubmit(
    SparkResidentDecodeStageProductionRunner *runner,
    const SparkResidentDecodeStageProductionRunnerDispatch *dispatch);

SparkStatus SparkResidentDecodeStageProductionRunnerProgress(
    SparkResidentDecodeStageProductionRunner *runner);

SparkStatus SparkResidentDecodeStageProductionRunnerWaitIdle(
    SparkResidentDecodeStageProductionRunner *runner,
    uint32_t max_poll_count);

SparkStatus SparkResidentDecodeStageProductionRunnerGetStats(
    const SparkResidentDecodeStageProductionRunner *runner,
    SparkResidentDecodeStageProductionRunnerStats *stats_out);

#ifdef __cplusplus
}
#endif
