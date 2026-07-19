#ifndef SPARKPIPE_SPARK_MIMO25_RESIDENT_DECODE_STAGE_FIRMWARE_H
#define SPARKPIPE_SPARK_MIMO25_RESIDENT_DECODE_STAGE_FIRMWARE_H

#include <stdint.h>

#include "sparkpipe/spark_module_abi.h"
#include "sparkpipe/spark_hidden_transport.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * MiMo-V2.5 resident decode stage. The variant model header arrives via
 * the build's -include (shared guard); every view here is geometry-free.
 * The stage boundary carries plain hidden rows - no stream expansion.
 *
 * Version 1 executes DECODE batches across both attention branches (full
 * with the dense per-lane history, SWA with the 128-slot ring and its
 * sink), the dense layer-zero MLP, and the sigmoid noaux_tc MoE. Prefill
 * frames and MTP execution refuse with distinct statuses; the MTP draft
 * layers load and verify so the pack contract is final. Caches are dense
 * per lane bounded by SPARK_MIMO25_STAGE_MAX_SEQ; the paged migration
 * rides the family PP pass.
 */

#define SPARK_MIMO25_RESIDENT_DECODE_STAGE_NODE_CONTEXT_ABI_VERSION 1u
#define SPARK_MIMO25_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION 1u
#define SPARK_MIMO25_RESIDENT_DECODE_STAGE_DECODE_BATCH_VIEW_ABI_VERSION 1u
#define SPARK_MIMO25_RESIDENT_DECODE_STAGE_LINEAR_VIEW_ABI_VERSION 1u
#define SPARK_MIMO25_RESIDENT_DECODE_STAGE_MAX_STAGE_COUNT 16u
#define SPARK_MIMO25_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT 128u
#define SPARK_MIMO25_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT 4u
#define SPARK_MIMO25_RESIDENT_DECODE_STAGE_MAX_LAYER_COUNT 70u

typedef struct SparkMimo25LinearView
{
	uint32_t abi_version;
	uint32_t weight_format;
	uint32_t rows;
	uint32_t columns;
	const void *payload;
	const void *scale;
} SparkMimo25LinearView;

typedef struct SparkMimo25AttnWeights
{
	const void *attn_norm_bf16;
	const float *sink_f32;
	SparkMimo25LinearView qkv;
	SparkMimo25LinearView o;
} SparkMimo25AttnWeights;

typedef struct SparkMimo25DenseWeights
{
	SparkMimo25LinearView w1;
	SparkMimo25LinearView w2;
	SparkMimo25LinearView w3;
} SparkMimo25DenseWeights;

// Routed experts are STACKED: expert e's block is rows_per_expert
// consecutive rows of one view; the launch offsets payload by
// e * rows_per_expert * columns bytes and the f32 scales by the matching
// [128,128] block count.
typedef struct SparkMimo25MoeWeights
{
	SparkMimo25LinearView gate;
	const float *gate_bias_f32;
	SparkMimo25LinearView experts_w1;
	SparkMimo25LinearView experts_w2;
	SparkMimo25LinearView experts_w3;
} SparkMimo25MoeWeights;

typedef struct SparkMimo25MtpLayerWeights
{
	SparkMimo25AttnWeights attn;
	const void *ffn_norm_bf16;
	SparkMimo25DenseWeights mlp;
	SparkMimo25LinearView eh_proj;
	const void *enorm_bf16;
	const void *hnorm_bf16;
	const void *final_norm_bf16;
} SparkMimo25MtpLayerWeights;

typedef struct SparkMimo25DecodeBatchView
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t row_count;
	uint32_t reserved0;
	const uint32_t *row_lane_indices;
	const uint64_t *row_positions;
	const uint64_t *row_sequence_ids;
} SparkMimo25DecodeBatchView;

#define SPARK_MIMO25_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_DECODE_BATCH_VIEW 0x00000001u
#define SPARK_MIMO25_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_INPUT_TRANSPORT 0x00000002u
#define SPARK_MIMO25_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_OUTPUT_TRANSPORT 0x00000004u
#define SPARK_MIMO25_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_PREFILL_FRAME_VIEW 0x00000008u

typedef SparkStatus (*SparkMimo25HiddenTransportPostReceiveFunction)(SparkHiddenTransportSession *transport_session, SparkHiddenTransportPacket *packet);
typedef SparkStatus (*SparkMimo25HiddenTransportSendFunction)(SparkHiddenTransportSession *transport_session, const SparkHiddenTransportPacket *packet);

typedef struct SparkMimo25ResidentDecodeStageFrameContext
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t flags;
	uint32_t reserved0;
	const SparkMimo25DecodeBatchView *decode_batch;
	SparkHiddenTransportSession *hidden_input_transport_session;
	SparkHiddenTransportSession *hidden_output_transport_session;
	SparkMimo25HiddenTransportPostReceiveFunction hidden_input_post_receive_function;
	SparkMimo25HiddenTransportSendFunction hidden_output_send_function;
	SparkHiddenTransportPacket hidden_input_packet;
	SparkHiddenTransportPacket hidden_output_packet;
} SparkMimo25ResidentDecodeStageFrameContext;

SparkStatus SparkMimo25ResidentDecodeStageInitialize(const SparkFirmwareModuleConfiguration *configuration, const SparkFirmwareModuleHostServices *host_services, void **module_state);
SparkStatus SparkMimo25ResidentDecodeStageExecute(void *module_state, SparkModelDriverFrame *frame);
SparkStatus SparkMimo25ResidentDecodeStageAdmit(void *module_state, const SparkModelDriverAdmissionRequest *request, SparkModelDriverAdmissionDecision *decision);
SparkStatus SparkMimo25ResidentDecodeStageSnapshot(void *module_state, uint32_t program_id, SparkModelDriverRuntimeSnapshot *snapshot);
void SparkMimo25ResidentDecodeStageDestroy(void *module_state);

#ifdef __cplusplus
}
#endif

#endif
