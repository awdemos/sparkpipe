#pragma once

#include <stdint.h>

#include "sparkpipe/spark_glm52_dspark.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_GLM52_DSPARK_DRAFT_BACKEND_ABI_VERSION 2u
#define SPARK_GLM52_DSPARK_DRAFT_BACKEND_CONFIGURATION_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52DsparkDraftBackendConfiguration))
#define SPARK_GLM52_DSPARK_DRAFT_BACKEND_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52DsparkDraftBackend))
#define SPARK_GLM52_DSPARK_DRAFT_BACKEND_MAX_LANE_COUNT 1024u
#define SPARK_GLM52_DSPARK_DRAFT_BACKEND_WEIGHT_COUNT 64u
#define SPARK_GLM52_DSPARK_DRAFT_BACKEND_ATTENTION_DIMENSION \
    (SPARK_GLM52_DSPARK_DRAFT_ATTENTION_HEAD_COUNT * \
     SPARK_GLM52_DSPARK_DRAFT_HEAD_DIMENSION)
#define SPARK_GLM52_DSPARK_DRAFT_BACKEND_FUSED_INPUT_DIMENSION \
    (SPARK_GLM52_DSPARK_AUX_LAYER_COUNT * SPARK_GLM52_DSPARK_HIDDEN_DIMENSION)

typedef struct SparkGlm52DsparkDraftBackendConfiguration
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t maximum_lane_count;
    uint32_t maximum_context_token_count;
    uint32_t restricted_vocabulary_count;
    const uint32_t *restricted_token_ids;
    const char *safetensors_path;
    void *cuda_stream;
} SparkGlm52DsparkDraftBackendConfiguration;

typedef struct SparkGlm52DsparkDraftBackendLaneState
{
    uint64_t sequence_id;
    uint64_t sequence_position;
    uint64_t tap_generation;
    uint32_t last_token_id;
    uint32_t context_token_count;
    uint32_t staged;
    uint32_t reserved;
} SparkGlm52DsparkDraftBackendLaneState;

typedef struct SparkGlm52DsparkDraftBackend
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t maximum_lane_count;
    uint32_t maximum_context_token_count;
    uint32_t restricted_vocabulary_count;
    uint32_t owns_cuda_stream;
    uint32_t weight_count;
    uint32_t reserved0;
    uint32_t reserved1;
    SparkGlm52DsparkModelContract contract;
    void *cuda_stream;
    void *cublas_handle;
    void *device_weights[SPARK_GLM52_DSPARK_DRAFT_BACKEND_WEIGHT_COUNT];
    uint32_t *device_restricted_token_ids;
    uint16_t *device_tap_arena_bf16;
    uint64_t tap_arena_lane_stride_bytes;
    uint16_t *device_target_hidden_bf16;
    uint16_t *device_context_key_bf16;
    uint16_t *device_context_value_bf16;
    uint16_t *device_block_hidden_bf16;
    uint16_t *device_block_normed_bf16;
    uint16_t *device_block_attention_bf16;
    uint16_t *device_block_query_bf16;
    uint16_t *device_block_key_bf16;
    uint16_t *device_block_value_bf16;
    uint16_t *device_block_gate_bf16;
    uint16_t *device_block_up_bf16;
    uint16_t *device_block_mlp_bf16;
    uint16_t *device_block_final_bf16;
    uint16_t *device_block_logits_bf16;
    uint16_t *device_markov_logits_bf16;
    uint32_t *device_argmax_u32;
    float *device_confidence_f32;
    uint32_t *host_argmax_u32;
    float *host_confidence_f32;
    SparkGlm52DsparkDraftBackendLaneState
        lane_states[SPARK_GLM52_DSPARK_DRAFT_BACKEND_MAX_LANE_COUNT];
} SparkGlm52DsparkDraftBackend;

SparkStatus SparkGlm52DsparkDraftBackendInitialize(
    SparkGlm52DsparkDraftBackend *backend,
    const SparkGlm52DsparkDraftBackendConfiguration *configuration);

SparkStatus SparkGlm52DsparkDraftBackendTeardown(
    SparkGlm52DsparkDraftBackend *backend);

SparkStatus SparkGlm52DsparkDraftBackendModelContract(
    const SparkGlm52DsparkDraftBackend *backend,
    SparkGlm52DsparkModelContract *contract_out);

SparkStatus SparkGlm52DsparkDraftBackendTapOutputPointers(
    SparkGlm52DsparkDraftBackend *backend,
    uint32_t lane_index,
    void *tap_output_bf16[SPARK_GLM52_DSPARK_AUX_LAYER_COUNT],
    uint64_t *lane_stride_bytes_out);

SparkStatus SparkGlm52DsparkDraftBackendStageLane(
    SparkGlm52DsparkDraftBackend *backend,
    uint32_t lane_index,
    uint64_t sequence_id,
    uint64_t sequence_position,
    uint32_t last_token_id,
    uint64_t tap_generation);

SparkStatus SparkGlm52DsparkDraftBackendDraft(
    void *context,
    const SparkGlm52DsparkDraftRequest *request,
    SparkGlm52DsparkDraftResult *result);

#ifdef __cplusplus
}

#endif
