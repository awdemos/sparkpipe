#include "sparkpipe/spark_glm52_pp13_node_context_builder.h"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>

#include "sparkpipe/spark_glm52_rope.h"
#include "sparkpipe/spark_glm52_dspark_draft_backend.h"
#include "sparkpipe/spark_glm52_production_topology.h"
#include "sparkpipe/spark_glm52_resident_decode_stage_fp8_moe_plan.h"
#include "sparkpipe/spark_glm52_resident_decode_stage_linear_plan.h"
#include "sparkpipe/spark_glm52_resident_decode_stage_production_runner.h"
#include "sparkpipe/spark_glm52_resident_decode_stage_required_cuda.h"
#include "sparkpipe/spark_glm52_kv_cache.h"
#include "sparkpipe/spark_glm52_stage_plan.h"
#include "sparkpipe/spark_glm52_stagepack.h"

#define SPARK_GLM52_PP13_BUILDER_LAYER_COUNT 6u
#define SPARK_GLM52_PP13_BUILDER_PIPELINE_SLOT_COUNT 1u
#define SPARK_GLM52_PP13_BUILDER_POSITION_COUNT SPARK_GLM52_KV_CONTEXT_TOKENS
#define SPARK_GLM52_PP13_BUILDER_MAX_BLOCKS_PER_SEQUENCE \
	(SPARK_GLM52_KV_CONTEXT_TOKENS / SPARK_GLM52_RESIDENT_DECODE_STAGE_BLOCK_TOKENS)
#define SPARK_GLM52_PP13_BUILDER_PREFILL_ROWS 1024u
#define SPARK_GLM52_PP13_BUILDER_COPY_CHUNK_BYTES (64ull * 1024ull * 1024ull)
#define SPARK_GLM52_PP13_BUILDER_EXECUTION_LAYER_COUNT \
	(SPARK_GLM52_PP13_BUILDER_LAYER_COUNT + 1u)
#define SPARK_GLM52_PP13_BUILDER_LAYER_BUFFER_ALLOCATION_COUNT 53u
#define SPARK_GLM52_PP13_BUILDER_LAYER_WEIGHT_ALLOCATION_COUNT 29u
#define SPARK_GLM52_PP13_BUILDER_SHARED_BUFFER_ALLOCATION_COUNT 8u
#define SPARK_GLM52_PP13_BUILDER_TABLE_ALLOCATION_COUNT 15u
#define SPARK_GLM52_PP13_BUILDER_INPUT_ALLOCATION_COUNT 12u
#define SPARK_GLM52_PP13_BUILDER_PLAN_ALLOCATION_COUNT 2u
#define SPARK_GLM52_PP13_BUILDER_MTP_SUPPORT_ALLOCATION_COUNT 9u
#define SPARK_GLM52_PP13_BUILDER_FINAL_OUTPUT_ALLOCATION_COUNT 2u
#define SPARK_GLM52_PP13_BUILDER_MAX_ALLOCATIONS \
	((SPARK_GLM52_PP13_BUILDER_EXECUTION_LAYER_COUNT * \
	  (SPARK_GLM52_PP13_BUILDER_LAYER_BUFFER_ALLOCATION_COUNT + \
	   SPARK_GLM52_PP13_BUILDER_LAYER_WEIGHT_ALLOCATION_COUNT)) + \
	 SPARK_GLM52_PP13_BUILDER_SHARED_BUFFER_ALLOCATION_COUNT + \
	 SPARK_GLM52_PP13_BUILDER_TABLE_ALLOCATION_COUNT + \
	 SPARK_GLM52_PP13_BUILDER_INPUT_ALLOCATION_COUNT + \
	 SPARK_GLM52_PP13_BUILDER_PLAN_ALLOCATION_COUNT + \
	 SPARK_GLM52_PP13_BUILDER_MTP_SUPPORT_ALLOCATION_COUNT + \
	 SPARK_GLM52_PP13_BUILDER_FINAL_OUTPUT_ALLOCATION_COUNT)
#define SPARK_GLM52_PP13_BUILDER_THREADS 256u
#define SPARK_GLM52_PP13_BUILDER_PROBE_HASH_SLOT_COUNT 18u
#define SPARK_GLM52_PP13_BUILDER_MAX_PREFILL_TOKENS \
	SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_MAX_PREFILL_TOKENS
#define SPARK_GLM52_PP13_BUILDER_INVALID_SLOT UINT32_MAX
#define SPARK_GLM52_PP13_BUILDER_SPECULATIVE_VERIFY_TARGET_COUNT \
	(SPARK_GLM52_PP13_WORK_CONTROL_MAX_SPECULATIVE_TOKEN_COUNT + 1u)
#define SPARK_GLM52_PP13_BUILDER_MTP_EH_INPUT_DIMENSION \
	(SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION * 2u)
#define SPARK_GLM52_PP13_BUILDER_MTP_NORM_COUNT_PER_LANE 2u
#define SPARK_GLM52_PP13_BUILDER_MTP_EMBEDDING_TENSOR \
	"sparkpipe.mtp.embed_tokens.weight"

typedef struct SparkGlm52Pp13BuilderLayer
{
	SparkGlm52ResidentDecodeStageNodeContext node;
	SparkGlm52ResidentDecodeStagePipelineSlot slot;
	SparkGlm52ResidentDecodeStageCudaPipelineSlotState cuda_slot;
	SparkGlm52ResidentDecodeStageLinearPlanResidentBinding *linear_binding;
	SparkGlm52ResidentDecodeStageFp8MoeResidentBinding fp8_moe_binding;
	uint32_t fp8_moe_ready;
	void *raw_query_a_weight_fp8;
	void *raw_query_a_scale;
	void *raw_query_b_weight_fp8;
	void *raw_query_b_scale;
	void *raw_kv_a_weight_fp8;
	void *raw_kv_a_scale;
	void *raw_kv_b_weight_fp8;
	void *raw_kv_b_scale;
	void *attention_output_weight_fp8;
	void *attention_output_scale;
	void *attention_norm_weight;
	void *raw_query_a_norm_weight;
	void *raw_kv_a_norm_weight;
	void *post_attention_norm_weight;
	void *dense_gate_weight_fp8;
	void *dense_gate_scale;
	void *dense_up_weight_fp8;
	void *dense_up_scale;
	void *dense_down_weight_fp8;
	void *dense_down_scale;
	void *router_weight;
	void *router_bias;
	void *index_query_weight_fp8;
	void *index_query_scale;
	void *index_key_weight_fp8;
	void *index_key_scale;
	void *index_weights_proj_weight;
	void *index_key_norm_weight;
	void *index_key_norm_bias;
	void *final_norm_weight;
	void *restricted_lm_head_weight;
	void *input_hidden;
	void *normalized_hidden;
	void *query_latent;
	void *query_rope_input;
	void *key_rope_input;
	void *current_kv_latent;
	void *raw_query_a;
	void *raw_query_a_norm;
	void *raw_query_b;
	void *raw_kv_a;
	void *raw_kv_a_norm;
	void *raw_kv_b;
	void *query_index_heads;
	void *current_key_index;
	void *index_head_weights;
	void *sparse_token_indices;
	void *rotated_query_rope;
	void *attention_output_latent;
	void *attention_projected_hidden;
	void *post_attention_hidden;
	void *post_attention_normalized_hidden;
	void *moe_topk_expert_ids;
	void *moe_topk_weights;
	void *moe_router_logits;
	void *moe_gate;
	void *moe_up;
	void *moe_intermediate;
	void *moe_route_output;
	void *layer_output_hidden;
	void *mtp_draft_hidden;
	void *restricted_logits;
	void *restricted_selected_token_ids;
	void *restricted_selected_token_scores;
	void *mtp_draft_logits;
	void *mtp_draft_token_ids;
	void *mtp_draft_token_budgets;
	void *mtp_target_token_ids;
	void *mtp_accept_mask;
	void *mtp_committed_token_ids;
	void *mtp_event_counters;
	void *phase_clock_cycles;
	void *positions;
	void *slot_mapping;
	void *block_table;
	void *context_lengths;
	void *first_block_token_offsets;
	void *mla_cache;
	void *key_nope_cache;
	void *value_cache;
	void *key_index_cache;
	void *key_index_block_min;
	void *key_index_block_max;
	void *dsa_summary_dirty_flags;
	SparkGlm52ResidentDecodeStagePagedPrefillPlan serial_prefill_paged_plan;
	SparkGlm52ResidentDecodeStageBulkPrefillPlan serial_prefill_bulk_plan;
} SparkGlm52Pp13BuilderLayer;

typedef struct SparkGlm52Pp13BuilderState
{
	SparkGlm52Pp13NodeContextBuilderConfiguration configuration;
	SparkGlm52Pp13RuntimeRankPlan rank_plan;
	SparkGlm52Pp13BuilderLayer layers[SPARK_GLM52_PP13_BUILDER_LAYER_COUNT];
	SparkGlm52Pp13BuilderLayer mtp_layer;
	SparkGlm52ResidentDecodeStageMtpDraftPlan mtp_draft_plan;
	const SparkGlm52ResidentDecodeStageNodeContext *layer_pointers[
		SPARK_GLM52_PP13_BUILDER_LAYER_COUNT];
	SparkGlm52ResidentDecodeStageSliceNodeContext slice_context;
	SparkGlm52ResidentDecodeStageStageSlicePlan stage_slice_plan;
	SparkGlm52ResidentDecodeStageExactStageSlicePlan exact_plan;
	SparkGlm52Sm121RequiredDecodeStageBuiltinFp8ScaledGemmState fp8_scaled_gemm_state;
	SparkGlm52Sm121RequiredDecodeStageFp8ScaledGemmBackend fp8_scaled_gemm_backend;
	SparkGlm52ResidentDecodeStageProductionRunner runner;
	SparkGlm52ProductionTopology production_topology;
	SparkGlm52DsparkHiddenTapPlan dspark_tap_plan;
	SparkGlm52DsparkDraftBackend dspark_backend;
	SparkGlm52DsparkModelContract dspark_model_contract;
	void *dspark_tap_outputs_bf16[SPARK_GLM52_DSPARK_AUX_LAYER_COUNT];
	uint64_t dspark_tap_lane_stride_bytes;
	SparkGlm52DsparkDraftResult dspark_ready_draft;
	SparkModelDriverCompletion captured_completion;
	uint32_t dspark_backend_ready;
	uint32_t dspark_ready_draft_valid;
	uint32_t captured_completion_valid;
	uint32_t speculative_verify_active;
	uint32_t speculative_verify_draft_count;
	uint32_t speculative_verify_mismatch;
	uint32_t speculative_verify_accepted_count;
	uint64_t speculative_verify_request_id;
	uint64_t speculative_verify_sequence_id;
	uint64_t speculative_verify_base_position;
	uint32_t speculative_verify_draft_token_ids[
		SPARK_GLM52_PP13_WORK_CONTROL_MAX_SPECULATIVE_TOKEN_COUNT];
	uint32_t speculative_verify_target_token_ids[
		SPARK_GLM52_PP13_BUILDER_SPECULATIVE_VERIFY_TARGET_COUNT];
	SparkGlm52Pp13WorkControlKvState kv_state;
	SparkGlm52KvBlockTableView host_kv_view;
	SparkGlm52KvBlockTableView device_kv_view;
	SparkGlm52Pp13NodeContextBuilderResult result;
	void *allocations[SPARK_GLM52_PP13_BUILDER_MAX_ALLOCATIONS];
	uint8_t allocation_is_host_mapped[SPARK_GLM52_PP13_BUILDER_MAX_ALLOCATIONS];
	uint32_t allocation_count;
	uint32_t *host_physical_block_indices;
	uint32_t *host_lane_physical_block_counts;
	uint8_t *host_physical_block_states;
	uint32_t *device_physical_block_indices;
	uint32_t *device_lane_physical_block_counts;
	void *selected_token_indices_by_layer;
	void *selected_block_indices_by_layer;
	void *selected_block_counts_by_layer;
	void *dsa_selection_epoch_by_layer;
	void *restricted_token_ids;
	void *embedding_weight;
	void *mtp_embedding_weight;
	void *mtp_enorm_weight;
	void *mtp_hnorm_weight;
	void *mtp_eh_proj_weight;
	void *mtp_shared_head_norm_weight;
	void *mtp_eh_input;
	void *mtp_previous_target_hidden;
	uint32_t *mtp_base_positions;
	float *mtp_norm_inv;
	void *cos_table;
	void *device_probe_hash_slots;
	void *sin_table;
	float *dsa_score_tiles;
	uint32_t *dsa_prefill_selected;
	uint32_t *dsa_prefill_row_context_lengths;
	uint32_t *dsa_prefill_row_sequences;
	uint32_t *dsa_prefill_row_positions;
	void *dsa_prefill_key_scratch;
	void *dsa_prefill_query_a;
	void *dsa_prefill_query_index_heads;
	void *dsa_prefill_index_weights;
	void *dsa_prefill_normalized_hidden;
	void *dsa_prefill_low_scratch;
	void *input_sideband;
	void *output_sideband;
	void *final_epilogue_workspace;
	void *fp8_scaled_gemm_workspace;
	uint32_t *host_prefill_lane_offsets;
	uint32_t *host_prefill_lane_counts;
	uint32_t *host_decode_positions;
	uint32_t *host_decode_token_ids;
	uint32_t *host_decode_result_token_ids;
	uint32_t *host_mtp_draft_budgets;
	uint32_t *host_mtp_committed_token_ids;
	void *device_prefill_token_ids;
	void *device_prefill_positions;
	void *device_prefill_slot_mapping;
	void *device_prefill_context_lengths;
	void *device_prefill_first_block_token_offsets;
	void *device_prefill_token_counts;
	void *device_prefill_hidden;
	void *device_prefill_output_hidden;
	void *device_decode_positions;
	void *device_decode_token_ids;
	void *device_mtp_draft_token_budgets;
	cudaStream_t stream;
	cudaStream_t query_stream;
	cudaStream_t kv_stream;
	cudaEvent_t branch_ready_event;
	cudaEvent_t query_event;
	cudaEvent_t kv_event;
	cublasHandle_t mtp_cublas_handle;
	uint64_t mtp_previous_request_id;
	uint64_t mtp_previous_sequence_id;
	uint64_t mtp_previous_position;
	uint32_t mtp_previous_valid;
	uint32_t mtp_use_previous_for_draft;
	uint32_t mtp_ready;
	const SparkModelDriverInterface *driver_interface;
	void *driver_instance;
	const SparkModelDriverProgramDescriptor *program;
	SparkHiddenTransportSession *output_transport_session;
	uint32_t built;
	uint32_t runner_ready;
} SparkGlm52Pp13BuilderState;

static uint32_t SparkGlm52Pp13BuilderDsparkEnabled(
	const SparkGlm52Pp13BuilderState *state)
{
	return state != 0 &&
		(state->configuration.flags &
			SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_CONFIGURATION_FLAG_DSPARK) != 0u;
}

static uint32_t SparkGlm52Pp13BuilderMtpEnabled(
	const SparkGlm52Pp13BuilderState *state)
{
	return state != 0 &&
		(state->configuration.flags &
			SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_CONFIGURATION_FLAG_MTP) != 0u;
}

static uint32_t SparkGlm52Pp13BuilderIsFinalRank(
	const SparkGlm52Pp13BuilderState *state)
{
	return state != 0 &&
		(state->rank_plan.flags &
			SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_FINAL_STAGE) != 0u;
}

static uint32_t SparkGlm52Pp13BuilderWorkCapturesDspark(
	const SparkGlm52Pp13WorkControlPacket *work_packet)
{
	return work_packet != 0 &&
		(work_packet->flags &
			SPARK_GLM52_PP13_WORK_CONTROL_FLAG_DSPARK_TAP_CAPTURE) != 0u;
}

static uint32_t SparkGlm52Pp13BuilderWorkIsDsparkVerify(
	const SparkGlm52Pp13WorkControlPacket *work_packet)
{
	return work_packet != 0 &&
		(work_packet->flags &
			SPARK_GLM52_PP13_WORK_CONTROL_FLAG_DSPARK_SPECULATIVE_VERIFY) != 0u;
}

static uint32_t SparkGlm52Pp13BuilderWorkIsMtpVerify(
	const SparkGlm52Pp13WorkControlPacket *work_packet)
{
	return work_packet != 0 &&
		(work_packet->flags &
			SPARK_GLM52_PP13_WORK_CONTROL_FLAG_MTP_SPECULATIVE_VERIFY) != 0u;
}

static uint32_t SparkGlm52Pp13BuilderWorkIsSpeculativeVerify(
	const SparkGlm52Pp13WorkControlPacket *work_packet)
{
	return SparkGlm52Pp13BuilderWorkIsDsparkVerify(work_packet) |
		SparkGlm52Pp13BuilderWorkIsMtpVerify(work_packet);
}

static uint32_t SparkGlm52Pp13BuilderWorkIsPlainDecodeBatch(
	const SparkGlm52Pp13WorkControlPacket *work_packet)
{
	if (work_packet == 0 || work_packet->active_sequence_count <= 1u)
		return 0u;
	return (work_packet->flags &
		(SPARK_GLM52_PP13_WORK_CONTROL_FLAG_PREFILL |
		 SPARK_GLM52_PP13_WORK_CONTROL_FLAG_MTP_DRAFT |
		 SPARK_GLM52_PP13_WORK_CONTROL_FLAG_DSPARK_TAP_CAPTURE |
		 SPARK_GLM52_PP13_WORK_CONTROL_FLAG_DSPARK_SPECULATIVE_VERIFY |
		 SPARK_GLM52_PP13_WORK_CONTROL_FLAG_MTP_SPECULATIVE_VERIFY)) == 0u;
}

static uint32_t SparkGlm52Pp13BuilderWorkNeedsCapturedCompletion(
	const SparkGlm52Pp13WorkControlPacket *work_packet)
{
	return SparkGlm52Pp13BuilderWorkCapturesDspark(work_packet) |
		SparkGlm52Pp13BuilderWorkIsSpeculativeVerify(work_packet) |
		SparkGlm52Pp13BuilderWorkIsPlainDecodeBatch(work_packet);
}

static SparkStatus SparkGlm52Pp13BuilderInitializeDsparkTopology(
	SparkGlm52Pp13BuilderState *state)
{
	SparkGlm52StagePlan stage_plan;
	char error_buffer[256];
	SparkStatus status;

	status = SparkGlm52Pp13RuntimeBuildFixedStagePlan(
		&stage_plan,
		error_buffer,
		sizeof(error_buffer));
	if (status != SPARK_STATUS_OK)
		return status;
	status = SparkGlm52ProductionTopologyBuild(
		&stage_plan,
		state->rank_plan.max_active_sequence_count,
		SPARK_GLM52_RESIDENT_DECODE_STAGE_SELECTED_TOKEN_COUNT,
		SPARK_GLM52_RESIDENT_DECODE_STAGE_BLOCK_TOKENS,
		SPARK_GLM52_RESIDENT_DECODE_STAGE_CACHE_TOKEN_ELEMENTS,
		&state->production_topology,
		error_buffer,
		sizeof(error_buffer));
	if (status != SPARK_STATUS_OK)
		return status;
	return SparkGlm52DsparkBuildDefaultHiddenTapPlan(&state->dspark_tap_plan);
}

static SparkStatus SparkGlm52Pp13BuilderInitializeDsparkBackend(
	SparkGlm52Pp13BuilderState *state)
{
	SparkGlm52DsparkDraftBackendConfiguration configuration;
	SparkStatus status;

	if (!SparkGlm52Pp13BuilderDsparkEnabled(state) ||
		!SparkGlm52Pp13BuilderIsFinalRank(state))
		return SPARK_STATUS_OK;
	if (state->configuration.dspark_safetensors_path == 0 ||
		state->configuration.dspark_safetensors_path[0] == '\0' ||
		state->configuration.dspark_maximum_lane_count == 0u ||
		state->configuration.dspark_maximum_lane_count >
			state->rank_plan.max_active_sequence_count ||
		state->configuration.dspark_maximum_context_token_count == 0u ||
		state->configuration.dspark_maximum_context_token_count >
			SPARK_GLM52_KV_CONTEXT_TOKENS)
		return SPARK_STATUS_INVALID_ARGUMENT;
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version =
		SPARK_GLM52_DSPARK_DRAFT_BACKEND_ABI_VERSION;
	configuration.descriptor_bytes =
		SPARK_GLM52_DSPARK_DRAFT_BACKEND_CONFIGURATION_DESCRIPTOR_BYTES;
	configuration.maximum_lane_count =
		state->configuration.dspark_maximum_lane_count;
	configuration.maximum_context_token_count =
		state->configuration.dspark_maximum_context_token_count;
	configuration.safetensors_path =
		state->configuration.dspark_safetensors_path;
	configuration.cuda_stream = (void *)state->stream;
	status = SparkGlm52DsparkDraftBackendInitialize(
		&state->dspark_backend,
		&configuration);
	if (status != SPARK_STATUS_OK)
		return status;
	status = SparkGlm52DsparkDraftBackendModelContract(
		&state->dspark_backend,
		&state->dspark_model_contract);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52DsparkDraftBackendTapOutputPointers(
			&state->dspark_backend,
			0u,
			state->dspark_tap_outputs_bf16,
			&state->dspark_tap_lane_stride_bytes);
	if (status != SPARK_STATUS_OK)
	{
		SparkGlm52DsparkDraftBackendTeardown(&state->dspark_backend);
		return status;
	}
	state->dspark_backend_ready = 1u;
	return SPARK_STATUS_OK;
}

static uint32_t SparkGlm52Pp13BuilderDsaSourceLayer(uint32_t layer_index)
{
	uint32_t adjusted_layer_index;
	if (layer_index >= SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_COUNT)
		return UINT32_MAX;
	if (layer_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_FIRST_ROUTED_LAYER)
		return layer_index;
	adjusted_layer_index =
		layer_index - (SPARK_GLM52_MODEL_DSA_INDEX_SKIP_TOPK_OFFSET - 1u);
	return (SPARK_GLM52_MODEL_DSA_INDEX_SKIP_TOPK_OFFSET - 1u) +
		(adjusted_layer_index -
		 (adjusted_layer_index %
		  SPARK_GLM52_MODEL_DSA_INDEX_SHARE_GROUP_LAYER_COUNT));
}

static uint32_t SparkGlm52Pp13BuilderDsaGroupEnd(uint32_t source_layer_index)
{
	if (source_layer_index + 1u <
		SPARK_GLM52_RESIDENT_DECODE_STAGE_FIRST_ROUTED_LAYER)
		return source_layer_index + 1u;
	if (source_layer_index +
		SPARK_GLM52_MODEL_DSA_INDEX_SHARE_GROUP_LAYER_COUNT >
		SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_COUNT)
		return SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_COUNT;
	return source_layer_index +
		SPARK_GLM52_MODEL_DSA_INDEX_SHARE_GROUP_LAYER_COUNT;
}

static SparkStatus SparkGlm52Pp13BuilderCudaStatus(cudaError_t status)
{
	if (status == cudaSuccess)
		return SPARK_STATUS_OK;
	fprintf(stderr,"pp13_builder_cuda_error code=%d name=%s\n",(int32_t)status,cudaGetErrorString(status));
	return SPARK_STATUS_IO_ERROR;
}

static SparkStatus SparkGlm52Pp13BuilderCublasStatus(cublasStatus_t status)
{
	if (status == CUBLAS_STATUS_SUCCESS)
		return SPARK_STATUS_OK;
	fprintf(stderr,"pp13_builder_cublas_error code=%d\n",(int32_t)status);
	return SPARK_STATUS_IO_ERROR;
}

__global__ static void SparkGlm52Pp13BuilderBuildPrefillMetadataKernel(
	const uint32_t *__restrict__ lane_offsets,
	const uint32_t *__restrict__ lane_counts,
	const uint32_t *__restrict__ block_table,
	const uint32_t *__restrict__ lane_block_counts,
	uint32_t lane_stride,
	uint32_t block_token_count,
	uint32_t lane_count,
	uint32_t token_stride,
	uint32_t *__restrict__ positions,
	uint32_t *__restrict__ slot_mapping,
	uint32_t *__restrict__ context_lengths,
	uint32_t *__restrict__ first_block_token_offsets,
	uint32_t *__restrict__ token_counts)
{
	uint32_t global_index;
	uint32_t lane_index;
	uint32_t token_index;
	uint32_t position;
	uint32_t block_index;
	uint32_t in_block_index;
	uint32_t physical_block_index;
	global_index = (uint32_t)(blockIdx.x * blockDim.x + threadIdx.x);
	if (global_index >= lane_count * token_stride)
		return;
	lane_index = global_index / token_stride;
	token_index = global_index - (lane_index * token_stride);
	if (token_index == 0u)
	{
		context_lengths[lane_index] =
			lane_offsets[lane_index] + lane_counts[lane_index];
		first_block_token_offsets[lane_index] =
			lane_offsets[lane_index] % block_token_count;
		token_counts[lane_index] = lane_counts[lane_index];
	}
	if (token_index >= lane_counts[lane_index])
		return;
	position = lane_offsets[lane_index] + token_index;
	block_index = position / block_token_count;
	in_block_index = position - (block_index * block_token_count);
	if (block_index >= lane_block_counts[lane_index])
	{
		positions[global_index] = position;
		slot_mapping[global_index] = SPARK_GLM52_PP13_BUILDER_INVALID_SLOT;
		return;
	}
	physical_block_index = block_table[(lane_index * lane_stride) + block_index];
	positions[global_index] = position;
	slot_mapping[global_index] =
		(physical_block_index * block_token_count) + in_block_index;
}

__global__ static void SparkGlm52Pp13BuilderGatherPrefillEmbeddingKernel(
	const uint32_t *__restrict__ token_ids,
	const uint32_t *__restrict__ lane_counts,
	const uint32_t *__restrict__ embedding_bf16_words,
	uint32_t *__restrict__ output_bf16_words,
	uint32_t lane_count,
	uint32_t token_stride,
	uint32_t hidden_words)
{
	uint64_t word_index;
	uint64_t route_index;
	uint32_t lane_index;
	uint32_t token_index;
	uint32_t hidden_word_index;
	uint32_t token_id;
	word_index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (word_index >= (uint64_t)lane_count * token_stride * hidden_words)
		return;
	route_index = word_index / hidden_words;
	hidden_word_index = (uint32_t)(word_index - (route_index * hidden_words));
	lane_index = (uint32_t)(route_index / token_stride);
	token_index = (uint32_t)(route_index - ((uint64_t)lane_index * token_stride));
	if (token_index >= lane_counts[lane_index])
		return;
	token_id = token_ids[(lane_index * token_stride) + token_index];
	output_bf16_words[word_index] =
		embedding_bf16_words[((uint64_t)token_id * hidden_words) + hidden_word_index];
}

__global__ static void SparkGlm52Pp13BuilderBuildDecodeMetadataKernel(
	const uint32_t *__restrict__ decode_positions,
	const uint32_t *__restrict__ block_table,
	const uint32_t *__restrict__ lane_block_counts,
	uint32_t lane_stride,
	uint32_t block_token_count,
	uint32_t lane_count,
	uint32_t *__restrict__ positions,
	uint32_t *__restrict__ slot_mapping,
	uint32_t *__restrict__ context_lengths,
	uint32_t *__restrict__ first_block_token_offsets)
{
	uint32_t lane_index;
	uint32_t position;
	uint32_t block_index;
	uint32_t in_block_index;
	uint32_t physical_block_index;
	lane_index = (uint32_t)(blockIdx.x * blockDim.x + threadIdx.x);
	if (lane_index >= lane_count)
		return;
	position = decode_positions[lane_index];
	block_index = position / block_token_count;
	in_block_index = position - (block_index * block_token_count);
	positions[lane_index] = position;
	context_lengths[lane_index] = position + 1u;
	first_block_token_offsets[lane_index] = 0u;
	if (block_index >= lane_block_counts[lane_index])
	{
		slot_mapping[lane_index] = SPARK_GLM52_PP13_BUILDER_INVALID_SLOT;
		return;
	}
	physical_block_index = block_table[(lane_index * lane_stride) + block_index];
	slot_mapping[lane_index] =
		(physical_block_index * block_token_count) + in_block_index;
}

__global__ static void SparkGlm52Pp13BuilderBuildDecodeSparseTokenIndicesKernel(
	const uint32_t *__restrict__ decode_positions,
	uint32_t lane_count,
	uint32_t *__restrict__ sparse_token_indices)
{
	uint32_t global_index;
	uint32_t lane_index;
	uint32_t sparse_index;
	uint32_t context_length;
	global_index = (uint32_t)(blockIdx.x * blockDim.x + threadIdx.x);
	if (global_index >=
		lane_count * SPARK_GLM52_RESIDENT_DECODE_STAGE_SELECTED_TOKEN_COUNT)
		return;
	lane_index = global_index /
		SPARK_GLM52_RESIDENT_DECODE_STAGE_SELECTED_TOKEN_COUNT;
	sparse_index = global_index -
		(lane_index * SPARK_GLM52_RESIDENT_DECODE_STAGE_SELECTED_TOKEN_COUNT);
	context_length = decode_positions[lane_index] + 1u;
	sparse_token_indices[global_index] = sparse_index < context_length
		? sparse_index
		: SPARK_GLM52_RESIDENT_DECODE_STAGE_INVALID_TOKEN_ID;
}

__global__ static void SparkGlm52Pp13BuilderBuildSerialPrefillMetadataKernel(
	uint32_t absolute_position,
	const uint32_t *__restrict__ block_table,
	const uint32_t *__restrict__ lane_block_counts,
	uint32_t lane_stride,
	uint32_t block_token_count,
	uint32_t lane_count,
	uint32_t *__restrict__ prompt_positions,
	uint32_t *__restrict__ prompt_slot_mapping,
	uint32_t *__restrict__ prompt_context_lengths,
	uint32_t *__restrict__ prompt_first_block_token_offsets,
	uint32_t *__restrict__ prompt_token_counts)
{
	uint32_t lane_index;
	uint32_t block_index;
	uint32_t in_block_index;
	uint32_t physical_block_index;
	lane_index = (uint32_t)(blockIdx.x * blockDim.x + threadIdx.x);
	if (lane_index >= lane_count)
		return;
	block_index = absolute_position / block_token_count;
	in_block_index = absolute_position - (block_index * block_token_count);
	prompt_positions[lane_index] = absolute_position;
	prompt_context_lengths[lane_index] = absolute_position + 1u;
	prompt_first_block_token_offsets[lane_index] = 0u;
	prompt_token_counts[lane_index] = 1u;
	if (block_index >= lane_block_counts[lane_index])
	{
		prompt_slot_mapping[lane_index] = SPARK_GLM52_PP13_BUILDER_INVALID_SLOT;
		return;
	}
	physical_block_index = block_table[(lane_index * lane_stride) + block_index];
	prompt_slot_mapping[lane_index] =
		(physical_block_index * block_token_count) + in_block_index;
}

__global__ static void SparkGlm52Pp13BuilderGatherDecodeEmbeddingKernel(
	const uint32_t *__restrict__ token_ids,
	const uint32_t *__restrict__ embedding_bf16_words,
	uint32_t *__restrict__ output_bf16_words,
	uint32_t lane_count,
	uint32_t hidden_words)
{
	uint64_t word_index;
	uint32_t lane_index;
	uint32_t hidden_word_index;
	uint32_t token_id;
	word_index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (word_index >= (uint64_t)lane_count * hidden_words)
		return;
	lane_index = (uint32_t)(word_index / hidden_words);
	hidden_word_index = (uint32_t)(word_index - ((uint64_t)lane_index * hidden_words));
	token_id = token_ids[lane_index];
	output_bf16_words[word_index] =
		embedding_bf16_words[((uint64_t)token_id * hidden_words) + hidden_word_index];
}

static __device__ __forceinline__ float SparkGlm52Pp13BuilderBf16ToFloat(
	uint16_t value)
{
	return __bfloat162float(__ushort_as_bfloat16(value));
}

static __device__ __forceinline__ uint16_t SparkGlm52Pp13BuilderFloatToBf16(
	float value)
{
	return __bfloat16_as_ushort(__float2bfloat16_rn(value));
}

__global__ static void SparkGlm52Pp13BuilderTargetFinalNormKernel(
	const uint16_t *__restrict__ input_bf16,
	const uint16_t *__restrict__ norm_weight_bf16,
	uint16_t *__restrict__ output_bf16,
	uint32_t active_sequence_count,
	float epsilon)
{
	__shared__ float sum[SPARK_GLM52_PP13_BUILDER_THREADS];
	uint32_t lane_index;
	uint32_t hidden_index;
	uint32_t stride;
	float local_sum;
	float norm_inv;
	float value;
	lane_index = blockIdx.x;
	if (lane_index >= active_sequence_count)
		return;
	local_sum = 0.0f;
	for (hidden_index = threadIdx.x;
		 hidden_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION;
		 hidden_index += blockDim.x)
	{
		value = SparkGlm52Pp13BuilderBf16ToFloat(input_bf16[
			((uint64_t)lane_index *
			 SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION) + hidden_index]);
		local_sum += value * value;
	}
	sum[threadIdx.x] = local_sum;
	__syncthreads();
	for (stride = blockDim.x >> 1u; stride != 0u; stride >>= 1u)
	{
		if (threadIdx.x < stride)
			sum[threadIdx.x] += sum[threadIdx.x + stride];
		__syncthreads();
	}
	norm_inv = rsqrtf(
		(sum[0] / SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION) + epsilon);
	for (hidden_index = threadIdx.x;
		 hidden_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION;
		 hidden_index += blockDim.x)
	{
		value = SparkGlm52Pp13BuilderBf16ToFloat(input_bf16[
			((uint64_t)lane_index *
			 SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION) + hidden_index]);
		value *= SparkGlm52Pp13BuilderBf16ToFloat(norm_weight_bf16[hidden_index]) *
			norm_inv;
		output_bf16[((uint64_t)lane_index *
			SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION) + hidden_index] =
			SparkGlm52Pp13BuilderFloatToBf16(value);
	}
}

__global__ static void SparkGlm52Pp13BuilderMtpNormInvKernel(
	const uint32_t *__restrict__ token_ids,
	const uint32_t *__restrict__ positions,
	const uint16_t *__restrict__ embedding_bf16,
	const uint16_t *__restrict__ hidden_bf16,
	float *__restrict__ norm_inv,
	uint32_t active_sequence_count,
	float epsilon)
{
	__shared__ float embedding_sum[SPARK_GLM52_PP13_BUILDER_THREADS];
	__shared__ float hidden_sum[SPARK_GLM52_PP13_BUILDER_THREADS];
	uint32_t lane_index;
	uint32_t hidden_index;
	uint32_t stride;
	uint32_t token_id;
	float embedding_value;
	float hidden_value;
	float local_embedding_sum;
	float local_hidden_sum;
	lane_index = blockIdx.x;
	if (lane_index >= active_sequence_count)
		return;
	token_id = token_ids[lane_index];
	local_embedding_sum = 0.0f;
	local_hidden_sum = 0.0f;
	for (hidden_index = threadIdx.x;
		 hidden_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION;
		 hidden_index += blockDim.x)
	{
		embedding_value = positions[lane_index] == 0u ? 0.0f :
			SparkGlm52Pp13BuilderBf16ToFloat(
				embedding_bf16[((uint64_t)token_id *
					SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION) + hidden_index]);
		hidden_value = SparkGlm52Pp13BuilderBf16ToFloat(
			hidden_bf16[((uint64_t)lane_index *
				SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION) + hidden_index]);
		local_embedding_sum += embedding_value * embedding_value;
		local_hidden_sum += hidden_value * hidden_value;
	}
	embedding_sum[threadIdx.x] = local_embedding_sum;
	hidden_sum[threadIdx.x] = local_hidden_sum;
	__syncthreads();
	for (stride = blockDim.x >> 1u; stride != 0u; stride >>= 1u)
	{
		if (threadIdx.x < stride)
		{
			embedding_sum[threadIdx.x] += embedding_sum[threadIdx.x + stride];
			hidden_sum[threadIdx.x] += hidden_sum[threadIdx.x + stride];
		}
		__syncthreads();
	}
	if (threadIdx.x == 0u)
	{
		norm_inv[(lane_index * 2u)] = rsqrtf(
			(embedding_sum[0] /
			 SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION) + epsilon);
		norm_inv[(lane_index * 2u) + 1u] = rsqrtf(
			(hidden_sum[0] /
			 SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION) + epsilon);
	}
}

__global__ static void SparkGlm52Pp13BuilderMtpFusionKernel(
	const uint32_t *__restrict__ token_ids,
	const uint32_t *__restrict__ positions,
	const uint16_t *__restrict__ embedding_bf16,
	const uint16_t *__restrict__ hidden_bf16,
	const uint16_t *__restrict__ embedding_norm_weight_bf16,
	const uint16_t *__restrict__ hidden_norm_weight_bf16,
	const float *__restrict__ norm_inv,
	uint16_t *__restrict__ output_bf16,
	uint32_t active_sequence_count)
{
	uint64_t output_index;
	uint32_t lane_index;
	uint32_t input_index;
	uint32_t hidden_index;
	uint32_t token_id;
	float value;
	output_index = ((uint64_t)blockIdx.x * blockDim.x) + threadIdx.x;
	if (output_index >= (uint64_t)active_sequence_count *
		SPARK_GLM52_PP13_BUILDER_MTP_EH_INPUT_DIMENSION)
		return;
	lane_index = (uint32_t)(output_index /
		SPARK_GLM52_PP13_BUILDER_MTP_EH_INPUT_DIMENSION);
	input_index = (uint32_t)(output_index - ((uint64_t)lane_index *
		SPARK_GLM52_PP13_BUILDER_MTP_EH_INPUT_DIMENSION));
	hidden_index = input_index % SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION;
	if (input_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION)
	{
		token_id = token_ids[lane_index];
		value = positions[lane_index] == 0u ? 0.0f :
			SparkGlm52Pp13BuilderBf16ToFloat(embedding_bf16[
				((uint64_t)token_id * SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION) +
				hidden_index]);
		value *= SparkGlm52Pp13BuilderBf16ToFloat(
			embedding_norm_weight_bf16[hidden_index]) * norm_inv[lane_index * 2u];
	}
	else
	{
		value = SparkGlm52Pp13BuilderBf16ToFloat(hidden_bf16[
			((uint64_t)lane_index * SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION) +
			hidden_index]);
		value *= SparkGlm52Pp13BuilderBf16ToFloat(
			hidden_norm_weight_bf16[hidden_index]) * norm_inv[(lane_index * 2u) + 1u];
	}
	output_bf16[output_index] = SparkGlm52Pp13BuilderFloatToBf16(value);
}

__global__ static void SparkGlm52Pp13BuilderMtpMetadataKernel(
	const uint32_t *__restrict__ base_positions,
	const uint32_t *__restrict__ physical_block_indices,
	uint32_t *__restrict__ positions,
	uint32_t *__restrict__ slot_mapping,
	uint32_t *__restrict__ context_lengths,
	uint32_t *__restrict__ first_block_token_offsets,
	uint32_t block_table_stride,
	uint32_t block_token_count,
	uint32_t draft_index,
	uint32_t active_sequence_count)
{
	uint32_t block_index;
	uint32_t block_token_index;
	uint32_t lane_index;
	uint32_t physical_block_index;
	uint32_t position;
	lane_index = (uint32_t)(blockIdx.x * blockDim.x + threadIdx.x);
	if (lane_index >= active_sequence_count)
		return;
	position = base_positions[lane_index] + draft_index;
	block_index = position / block_token_count;
	block_token_index = position - (block_index * block_token_count);
	physical_block_index = physical_block_indices[
		((uint64_t)lane_index * block_table_stride) + block_index];
	positions[lane_index] = position;
	slot_mapping[lane_index] =
		(physical_block_index * block_token_count) + block_token_index;
	context_lengths[lane_index] = position + 1u;
	first_block_token_offsets[lane_index] = 0u;
}

__global__ static void SparkGlm52Pp13BuilderMtpStoreKernel(
	const uint16_t *__restrict__ hidden_bf16,
	const uint32_t *__restrict__ token_ids,
	uint16_t *__restrict__ draft_hidden_bf16,
	uint32_t *__restrict__ draft_token_ids,
	uint32_t draft_index,
	uint32_t active_sequence_count)
{
	uint64_t element_index;
	uint32_t lane_index;
	uint32_t hidden_index;
	element_index = ((uint64_t)blockIdx.x * blockDim.x) + threadIdx.x;
	if (element_index >= (uint64_t)active_sequence_count *
		SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION)
		return;
	lane_index = (uint32_t)(element_index /
		SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION);
	hidden_index = (uint32_t)(element_index - ((uint64_t)lane_index *
		SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION));
	draft_hidden_bf16[(((uint64_t)lane_index *
		SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT) + draft_index) *
		SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION + hidden_index] =
		hidden_bf16[element_index];
	if (hidden_index == 0u)
		draft_token_ids[(lane_index *
			SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT) + draft_index] =
			token_ids[lane_index];
}

static SparkStatus SparkGlm52Pp13BuilderReportStatus(
	const char *step,
	uint32_t layer_index,
	SparkStatus status)
{
	if (status != SPARK_STATUS_OK)
		fprintf(stderr,"pp13_builder_error layer=%u step=%s status=%d\n",
			layer_index,step == 0 ? "unknown" : step,(int)status);
	return status;
}

static SparkStatus SparkGlm52Pp13BuilderRememberAllocationWithKind(
	SparkGlm52Pp13BuilderState *state,
	void *pointer,
	uint32_t is_host_mapped)
{
	if (state == 0 || pointer == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (state->allocation_count >= SPARK_GLM52_PP13_BUILDER_MAX_ALLOCATIONS)
	{
		fprintf(stderr,
			"pp13_builder_allocation_registry_full count=%u capacity=%u\n",
			state->allocation_count,
			SPARK_GLM52_PP13_BUILDER_MAX_ALLOCATIONS);
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	}
	state->allocations[state->allocation_count] = pointer;
	state->allocation_is_host_mapped[state->allocation_count] =
		is_host_mapped != 0u ? 1u : 0u;
	state->allocation_count += 1u;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderRememberAllocation(
	SparkGlm52Pp13BuilderState *state,
	void *pointer)
{
	return SparkGlm52Pp13BuilderRememberAllocationWithKind(state,pointer,0u);
}

static SparkStatus SparkGlm52Pp13BuilderCudaAlloc(SparkGlm52Pp13BuilderState *state, void **pointer_out, uint64_t bytes)
{
	void *pointer;
	size_t free_bytes, total_bytes;
	SparkStatus status;
	if (state == 0 || pointer_out == 0 || bytes == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	pointer = 0;
	status = SparkGlm52Pp13BuilderCudaStatus(cudaMalloc(&pointer,(size_t)bytes));
	if (status != SPARK_STATUS_OK)
	{
		free_bytes = 0u;
		total_bytes = 0u;
		if (cudaMemGetInfo(&free_bytes,&total_bytes) != cudaSuccess)
			cudaGetLastError();
		fprintf(stderr,"pp13_builder_cuda_alloc_failed requested_bytes=%llu free_bytes=%llu total_bytes=%llu\n",(unsigned long long)bytes,(unsigned long long)free_bytes,(unsigned long long)total_bytes);
		return status;
	}
	status = SparkGlm52Pp13BuilderRememberAllocation(state,pointer);
	if (status != SPARK_STATUS_OK)
	{
		cudaFree(pointer);
		return status;
	}
	*pointer_out = pointer;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderCudaHostMappedAlloc(
	SparkGlm52Pp13BuilderState *state,
	void **device_pointer_out,
	uint64_t bytes)
{
	void *host_pointer;
	void *device_pointer;
	SparkStatus status;
	if (state == 0 || device_pointer_out == 0 || bytes == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	host_pointer = 0;
	device_pointer = 0;
	status = SparkGlm52Pp13BuilderCudaStatus(cudaHostAlloc(
		&host_pointer,
		(size_t)bytes,
		cudaHostAllocMapped | cudaHostAllocPortable));
	if (status != SPARK_STATUS_OK)
		return status;
	status = SparkGlm52Pp13BuilderCudaStatus(cudaHostGetDevicePointer(
		&device_pointer,
		host_pointer,
		0));
	if (status != SPARK_STATUS_OK)
	{
		cudaFreeHost(host_pointer);
		return status;
	}
	status = SparkGlm52Pp13BuilderRememberAllocationWithKind(
		state,
		host_pointer,
		1u);
	if (status != SPARK_STATUS_OK)
	{
		cudaFreeHost(host_pointer);
		return status;
	}
	*device_pointer_out = device_pointer;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderCudaZero(
	void *pointer,
	uint64_t bytes)
{
	if (pointer == 0 || bytes == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	return SparkGlm52Pp13BuilderCudaStatus(cudaMemset(pointer,0,(size_t)bytes));
}

static SparkStatus SparkGlm52Pp13BuilderReadToDevice(
	const char *path,
	uint64_t offset,
	uint64_t bytes,
	void *device_pointer)
{
	FILE *file;
	uint8_t *buffer;
	uint8_t *device_bytes;
	uint64_t copied;
	uint64_t chunk;
	size_t got;
	SparkStatus status;
	if (path == 0 || device_pointer == 0 || bytes == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	file = fopen(path,"rb");
	if (file == 0)
	{
		fprintf(stderr,"pp13_builder_read_missing path=%s offset=%llu bytes=%llu\n",
			path,
			(unsigned long long)offset,
			(unsigned long long)bytes);
		return SPARK_STATUS_NOT_FOUND;
	}
	buffer = (uint8_t *)malloc((size_t)SPARK_GLM52_PP13_BUILDER_COPY_CHUNK_BYTES);
	if (buffer == 0)
	{
		fclose(file);
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	}
	if (fseeko(file,(off_t)offset,SEEK_SET) != 0)
	{
		free(buffer);
		fclose(file);
		return SPARK_STATUS_IO_ERROR;
	}
	device_bytes = (uint8_t *)device_pointer;
	copied = 0u;
	status = SPARK_STATUS_OK;
	while (copied < bytes)
	{
		chunk = bytes - copied;
		if (chunk > SPARK_GLM52_PP13_BUILDER_COPY_CHUNK_BYTES)
			chunk = SPARK_GLM52_PP13_BUILDER_COPY_CHUNK_BYTES;
		got = fread(buffer,1u,(size_t)chunk,file);
		if (got != (size_t)chunk)
		{
			status = SPARK_STATUS_IO_ERROR;
			break;
		}
		status = SparkGlm52Pp13BuilderCudaStatus(cudaMemcpy(
			device_bytes + copied,
			buffer,
			(size_t)chunk,
			cudaMemcpyHostToDevice));
		if (status != SPARK_STATUS_OK)
			break;
		copied += chunk;
	}
	free(buffer);
	fclose(file);
	return status;
}

static SparkStatus SparkGlm52Pp13BuilderTensorSpec(
	SparkGlm52StagePackTensorSpec *spec,
	const char *name,
	const char *dtype,
	uint64_t bytes_per_element,
	uint32_t rank,
	uint64_t d0,
	uint64_t d1)
{
	if (spec == 0 || name == 0 || dtype == 0 || rank == 0u || rank > 2u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	memset(spec,0,sizeof(*spec));
	spec->abi_version = SPARK_GLM52_STAGEPACK_ABI_VERSION;
	spec->rank = rank;
	spec->bytes_per_element = bytes_per_element;
	spec->shape[0] = d0;
	spec->shape[1] = rank > 1u ? d1 : 1u;
	spec->tensor_name = name;
	spec->dtype = dtype;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderLoadTensor(
	SparkGlm52Pp13BuilderState *state,
	const char *name,
	const char *dtype,
	uint64_t bytes_per_element,
	uint32_t rank,
	uint64_t d0,
	uint64_t d1,
	void **device_out)
{
	SparkGlm52StagePackTensorSpec spec;
	SparkGlm52StagePackTensorRegion region;
	SparkStatus status;
	if (device_out == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	*device_out = 0;
	status = SparkGlm52Pp13BuilderTensorSpec(
		&spec,name,dtype,bytes_per_element,rank,d0,d1);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52StagePackResolveTensor(
			state->configuration.stagepack_root,
			&spec,
			&region);
	if (status != SPARK_STATUS_OK)
		return SparkGlm52Pp13BuilderReportStatus(name,UINT32_MAX,status);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaAlloc(
			state,
			device_out,
			region.tensor_bytes);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderReadToDevice(
			region.file_path,
			region.file_offset,
			region.tensor_bytes,
			*device_out);
	return status;
}

static SparkStatus SparkGlm52Pp13BuilderTensorName(
	char *name,
	uint32_t name_bytes,
	uint32_t layer_index,
	const char *suffix)
{
	int written;
	if (name == 0 || name_bytes == 0u || suffix == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	written = snprintf(name,name_bytes,"model.layers.%u.%s",layer_index,suffix);
	if (written < 0 || (uint32_t)written >= name_bytes)
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderLoadLayerTensor(
	SparkGlm52Pp13BuilderState *state,
	uint32_t layer_index,
	const char *suffix,
	const char *dtype,
	uint64_t bytes_per_element,
	uint32_t rank,
	uint64_t d0,
	uint64_t d1,
	void **device_out)
{
	char name[256];
	SparkStatus status;
	status = SparkGlm52Pp13BuilderTensorName(
		name,(uint32_t)sizeof(name),layer_index,suffix);
	if (status != SPARK_STATUS_OK)
		return status;
	return SparkGlm52Pp13BuilderLoadTensor(
		state,name,dtype,bytes_per_element,rank,d0,d1,device_out);
}

static SparkStatus SparkGlm52Pp13BuilderLoadLmHeadRestricted(
	SparkGlm52Pp13BuilderState *state,
	void **device_out)
{
	SparkGlm52StagePackTensorSpec spec;
	SparkGlm52StagePackTensorRegion region;
	uint64_t bytes;
	SparkStatus status;
	status = SparkGlm52Pp13BuilderTensorSpec(
		&spec,
		"lm_head.weight",
		"BF16",
		(uint32_t)sizeof(uint16_t),
		2u,
		SPARK_GLM52_RESIDENT_DECODE_STAGE_OUTPUT_VOCAB_COUNT,
		SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52StagePackResolveTensor(
			state->configuration.stagepack_root,
			&spec,
			&region);
	bytes =
		(uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_OUTPUT_VOCAB_COUNT *
		(uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION *
		sizeof(uint16_t);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaAlloc(state,device_out,bytes);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderReadToDevice(
			region.file_path,
			region.file_offset,
			bytes,
			*device_out);
	return status;
}

static SparkStatus SparkGlm52Pp13BuilderLoadEmbedding(
	SparkGlm52Pp13BuilderState *state)
{
	if ((state->rank_plan.flags & SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_HAS_PREVIOUS) != 0u)
		return SPARK_STATUS_OK;
	return SparkGlm52Pp13BuilderLoadTensor(
		state,
		"model.embed_tokens.weight",
		"BF16",
		(uint32_t)sizeof(uint16_t),
		2u,
		SPARK_GLM52_RESIDENT_DECODE_STAGE_OUTPUT_VOCAB_COUNT,
		SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
		&state->embedding_weight);
}

static SparkStatus SparkGlm52Pp13BuilderCopyU32ToDevice(
	SparkGlm52Pp13BuilderState *state,
	void **device_out,
	const uint32_t *values,
	uint32_t count)
{
	uint64_t bytes;
	SparkStatus status;
	if (values == 0 || count == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	bytes = (uint64_t)count * sizeof(uint32_t);
	status = SparkGlm52Pp13BuilderCudaAlloc(state,device_out,bytes);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaStatus(cudaMemcpy(
			*device_out,values,(size_t)bytes,cudaMemcpyHostToDevice));
	return status;
}

static SparkStatus SparkGlm52Pp13BuilderInitializeTables(
	SparkGlm52Pp13BuilderState *state)
{
	uint32_t *tokens;
	uint64_t rope_pairs;
	uint64_t table_count;
	uint64_t table_bytes;
	uint32_t index;
	SparkStatus status;
	float *cos_host;
	float *sin_host;
	rope_pairs = SPARK_GLM52_RESIDENT_DECODE_STAGE_ROPE_DIMENSION / 2u;
	table_count = (uint64_t)SPARK_GLM52_PP13_BUILDER_POSITION_COUNT * rope_pairs;
	table_bytes = table_count * sizeof(float);
	status = SparkGlm52Pp13BuilderCudaAlloc(state,(void **)&state->dsa_score_tiles,(uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_SCORE_TILE_ROWS * SPARK_GLM52_KV_CONTEXT_TOKENS * sizeof(float));
	if (status != SPARK_STATUS_OK)
		return status;
	status = SparkGlm52Pp13BuilderCudaAlloc(state,(void **)&state->dsa_prefill_selected,(uint64_t)SPARK_GLM52_PP13_BUILDER_PREFILL_ROWS * SPARK_GLM52_RESIDENT_DECODE_STAGE_SELECTED_TOKEN_COUNT * sizeof(uint32_t));
	if (status != SPARK_STATUS_OK)
		return status;
	status = SparkGlm52Pp13BuilderCudaAlloc(state,(void **)&state->dsa_prefill_row_context_lengths,(uint64_t)SPARK_GLM52_PP13_BUILDER_PREFILL_ROWS * sizeof(uint32_t));
	if (status != SPARK_STATUS_OK)
		return status;
	status = SparkGlm52Pp13BuilderCudaAlloc(state,(void **)&state->dsa_prefill_row_sequences,(uint64_t)SPARK_GLM52_PP13_BUILDER_PREFILL_ROWS * sizeof(uint32_t));
	if (status != SPARK_STATUS_OK)
		return status;
	status = SparkGlm52Pp13BuilderCudaAlloc(state,(void **)&state->dsa_prefill_row_positions,(uint64_t)SPARK_GLM52_PP13_BUILDER_PREFILL_ROWS * sizeof(uint32_t));
	if (status != SPARK_STATUS_OK)
		return status;
	status = SparkGlm52Pp13BuilderCudaAlloc(state,&state->dsa_prefill_key_scratch,(uint64_t)SPARK_GLM52_PP13_BUILDER_PREFILL_ROWS * SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_INDEX_KEY_DIMENSION * sizeof(uint16_t));
	if (status != SPARK_STATUS_OK)
		return status;
	status = SparkGlm52Pp13BuilderCudaAlloc(state,&state->dsa_prefill_query_a,(uint64_t)SPARK_GLM52_PP13_BUILDER_PREFILL_ROWS * SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION * sizeof(uint16_t));
	if (status != SPARK_STATUS_OK)
		return status;
	status = SparkGlm52Pp13BuilderCudaAlloc(state,&state->dsa_prefill_query_index_heads,(uint64_t)SPARK_GLM52_PP13_BUILDER_PREFILL_ROWS * SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_INDEX_QUERY_DIMENSION * sizeof(uint16_t));
	if (status != SPARK_STATUS_OK)
		return status;
	status = SparkGlm52Pp13BuilderCudaAlloc(state,&state->dsa_prefill_index_weights,(uint64_t)SPARK_GLM52_PP13_BUILDER_PREFILL_ROWS * SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_INDEX_WEIGHT_DIMENSION * sizeof(uint16_t));
	if (status != SPARK_STATUS_OK)
		return status;
	status = SparkGlm52Pp13BuilderCudaAlloc(state,&state->dsa_prefill_normalized_hidden,(uint64_t)SPARK_GLM52_PP13_BUILDER_PREFILL_ROWS * SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION * sizeof(uint16_t));
	if (status != SPARK_STATUS_OK)
		return status;
	status = SparkGlm52Pp13BuilderCudaAlloc(state,&state->dsa_prefill_low_scratch,(uint64_t)SPARK_GLM52_PP13_BUILDER_PREFILL_ROWS * SPARK_GLM52_RESIDENT_DECODE_STAGE_KV_B_DIMENSION * sizeof(uint16_t));
	if (status != SPARK_STATUS_OK)
		return status;
	status = SparkGlm52Pp13BuilderCudaAlloc(state,&state->device_probe_hash_slots,SPARK_GLM52_PP13_BUILDER_PROBE_HASH_SLOT_COUNT * sizeof(uint64_t));
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaAlloc(state,&state->cos_table,table_bytes);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaAlloc(state,&state->sin_table,table_bytes);
	if (status == SPARK_STATUS_OK)
	{
		cos_host = (float *)malloc((size_t)table_bytes);
		sin_host = (float *)malloc((size_t)table_bytes);
		if (cos_host == 0 || sin_host == 0)
		{
			free(cos_host);
			free(sin_host);
			return SPARK_STATUS_CAPACITY_EXCEEDED;
		}
		SparkGlm52BuildRopeTables(
			cos_host,
			sin_host,
			SPARK_GLM52_PP13_BUILDER_POSITION_COUNT);
		status = SparkGlm52Pp13BuilderCudaStatus(cudaMemcpy(
			state->cos_table,cos_host,(size_t)table_bytes,cudaMemcpyHostToDevice));
		if (status == SPARK_STATUS_OK)
			status = SparkGlm52Pp13BuilderCudaStatus(cudaMemcpy(
				state->sin_table,sin_host,(size_t)table_bytes,cudaMemcpyHostToDevice));
		free(cos_host);
		free(sin_host);
	}
	tokens = (uint32_t *)malloc(
		SPARK_GLM52_RESIDENT_DECODE_STAGE_OUTPUT_VOCAB_COUNT *
		sizeof(uint32_t));
	if (status != SPARK_STATUS_OK || tokens == 0)
	{
		free(tokens);
		return status == SPARK_STATUS_OK ? SPARK_STATUS_CAPACITY_EXCEEDED : status;
	}
	for (index = 0u;
		 index < SPARK_GLM52_RESIDENT_DECODE_STAGE_OUTPUT_VOCAB_COUNT;
		 ++index)
		tokens[index] = index;
	status = SparkGlm52Pp13BuilderCopyU32ToDevice(
		state,
		&state->restricted_token_ids,
		tokens,
		SPARK_GLM52_RESIDENT_DECODE_STAGE_OUTPUT_VOCAB_COUNT);
	free(tokens);
	return status;
}

static SparkStatus SparkGlm52Pp13BuilderAllocateLayerBuffers(
	SparkGlm52Pp13BuilderState *state,
	SparkGlm52Pp13BuilderLayer *layer,
	uint32_t layer_offset,
	uint64_t cache_token_capacity,
	uint32_t max_blocks_per_sequence)
{
	uint64_t b;
	uint64_t kv_block_count;
	uint64_t route_count;
	uint32_t input_crosses_rank_boundary;
	uint32_t output_crosses_rank_boundary;
	SparkStatus status;
	b = state->rank_plan.max_active_sequence_count;
	kv_block_count = cache_token_capacity /
		SPARK_GLM52_RESIDENT_DECODE_STAGE_BLOCK_TOKENS;
	route_count = b * SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_TOP_K;
	input_crosses_rank_boundary =
		layer_offset == 0u &&
		(state->rank_plan.flags & SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_HAS_PREVIOUS) != 0u;
	output_crosses_rank_boundary =
		layer_offset + 1u == state->rank_plan.layer_count &&
		(state->rank_plan.flags & SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_HAS_NEXT) != 0u;
#define ALLOC_FIELD(field, count, type) \
	do { status = SparkGlm52Pp13BuilderCudaAlloc(state,&layer->field,(uint64_t)(count) * sizeof(type)); if (status != SPARK_STATUS_OK) return status; } while (0)
#define ALLOC_FIELD_MAPPED(field, count, type) \
	do { status = SparkGlm52Pp13BuilderCudaHostMappedAlloc(state,&layer->field,(uint64_t)(count) * sizeof(type)); if (status != SPARK_STATUS_OK) return status; } while (0)
#define ZERO_FIELD(field, count, type) \
	do { status = SparkGlm52Pp13BuilderCudaZero(layer->field,(uint64_t)(count) * sizeof(type)); if (status != SPARK_STATUS_OK) return status; } while (0)
	if (input_crosses_rank_boundary != 0u)
		ALLOC_FIELD_MAPPED(input_hidden,b * SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,uint16_t);
	else
		ALLOC_FIELD(input_hidden,b * SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,uint16_t);
	ALLOC_FIELD(normalized_hidden,b * SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,uint16_t);
	ALLOC_FIELD(query_latent,b * SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_LATENT_PROJECTION_DIMENSION,uint16_t);
	ALLOC_FIELD(query_rope_input,b * SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_ROPE_PROJECTION_DIMENSION,uint16_t);
	ALLOC_FIELD(key_rope_input,b * SPARK_GLM52_RESIDENT_DECODE_STAGE_ROPE_DIMENSION,uint16_t);
	ALLOC_FIELD(current_kv_latent,b * SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION,uint16_t);
	ALLOC_FIELD(raw_query_a,b * SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION,uint16_t);
	ALLOC_FIELD(raw_query_a_norm,b * SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION,uint16_t);
	ALLOC_FIELD(raw_query_b,b * SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_B_DIMENSION,uint16_t);
	ALLOC_FIELD(raw_kv_a,b * SPARK_GLM52_RESIDENT_DECODE_STAGE_KV_A_DIMENSION,uint16_t);
	ALLOC_FIELD(raw_kv_a_norm,b * SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION,uint16_t);
	ALLOC_FIELD(raw_kv_b,b * SPARK_GLM52_RESIDENT_DECODE_STAGE_KV_B_DIMENSION,uint16_t);
	ALLOC_FIELD(query_index_heads,b * SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_INDEX_QUERY_DIMENSION,uint16_t);
	ALLOC_FIELD(current_key_index,b * SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_INDEX_KEY_DIMENSION,uint16_t);
	ALLOC_FIELD(index_head_weights,b * SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_INDEX_WEIGHT_DIMENSION,uint16_t);
	ALLOC_FIELD(sparse_token_indices,b * SPARK_GLM52_RESIDENT_DECODE_STAGE_SELECTED_TOKEN_COUNT,uint32_t);
	ALLOC_FIELD(rotated_query_rope,b * SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_ROPE_PROJECTION_DIMENSION,uint16_t);
	ALLOC_FIELD(attention_output_latent,b * SPARK_GLM52_RESIDENT_DECODE_STAGE_ATTENTION_PROJECTION_DIMENSION,uint16_t);
	ALLOC_FIELD(attention_projected_hidden,b * SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,uint16_t);
	ALLOC_FIELD(post_attention_hidden,b * SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,uint16_t);
	ALLOC_FIELD(post_attention_normalized_hidden,b * SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,uint16_t);
	ALLOC_FIELD(moe_topk_expert_ids,route_count,uint32_t);
	ALLOC_FIELD(moe_topk_weights,route_count,float);
	ALLOC_FIELD(moe_router_logits,b * SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT,float);
	ALLOC_FIELD(moe_gate,route_count * SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION,uint16_t);
	ALLOC_FIELD(moe_up,route_count * SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION,uint16_t);
	ALLOC_FIELD(moe_intermediate,route_count * SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION,uint16_t);
	ALLOC_FIELD(moe_route_output,route_count * SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,uint16_t);
	if (output_crosses_rank_boundary != 0u)
		ALLOC_FIELD_MAPPED(layer_output_hidden,b * SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,uint16_t);
	else
		ALLOC_FIELD(layer_output_hidden,b * SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,uint16_t);
	ALLOC_FIELD(mtp_draft_hidden,b * SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT * SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,uint16_t);
	ALLOC_FIELD(restricted_logits,b * SPARK_GLM52_RESIDENT_DECODE_STAGE_RESTRICTED_VOCAB_COUNT,float);
	ALLOC_FIELD(restricted_selected_token_ids,b,uint32_t);
	ALLOC_FIELD(restricted_selected_token_scores,b,float);
	ALLOC_FIELD(mtp_draft_logits,b * SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT * SPARK_GLM52_RESIDENT_DECODE_STAGE_RESTRICTED_VOCAB_COUNT,float);
	ALLOC_FIELD(mtp_draft_token_ids,b * SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT,uint32_t);
	ALLOC_FIELD(mtp_draft_token_budgets,b,uint32_t);
	ALLOC_FIELD(mtp_target_token_ids,b * SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT,uint32_t);
	ALLOC_FIELD(mtp_accept_mask,b * SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT,uint32_t);
	ALLOC_FIELD(mtp_committed_token_ids,b * SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT,uint32_t);
	ALLOC_FIELD(mtp_event_counters,SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_EVENT_COUNTER_COUNT,uint32_t);
	ALLOC_FIELD(phase_clock_cycles,SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_CLOCK_COUNT,uint64_t);
	ALLOC_FIELD(positions,b,uint32_t);
	ALLOC_FIELD(slot_mapping,b,uint32_t);
	ALLOC_FIELD(block_table,b * max_blocks_per_sequence,uint32_t);
	ALLOC_FIELD(context_lengths,b,uint32_t);
	ALLOC_FIELD(first_block_token_offsets,b,uint32_t);
	ALLOC_FIELD(mla_cache,cache_token_capacity * SPARK_GLM52_RESIDENT_DECODE_STAGE_CACHE_TOKEN_ELEMENTS,uint16_t);
	ALLOC_FIELD(key_index_cache,cache_token_capacity * SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_INDEX_KEY_DIMENSION,uint16_t);
	ALLOC_FIELD(key_index_block_min,kv_block_count * SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_INDEX_KEY_DIMENSION,uint16_t);
	ALLOC_FIELD(key_index_block_max,kv_block_count * SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_INDEX_KEY_DIMENSION,uint16_t);
	ALLOC_FIELD(dsa_summary_dirty_flags,kv_block_count,uint8_t);
	ZERO_FIELD(input_hidden,b * SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,uint16_t);
	ZERO_FIELD(mla_cache,cache_token_capacity * SPARK_GLM52_RESIDENT_DECODE_STAGE_CACHE_TOKEN_ELEMENTS,uint16_t);
	ZERO_FIELD(key_index_cache,cache_token_capacity * SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_INDEX_KEY_DIMENSION,uint16_t);
	ZERO_FIELD(key_index_block_min,kv_block_count * SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_INDEX_KEY_DIMENSION,uint16_t);
	ZERO_FIELD(key_index_block_max,kv_block_count * SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_INDEX_KEY_DIMENSION,uint16_t);
	ZERO_FIELD(dsa_summary_dirty_flags,kv_block_count,uint8_t);
#undef ALLOC_FIELD
#undef ALLOC_FIELD_MAPPED
#undef ZERO_FIELD
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderLoadLayerWeights(
	SparkGlm52Pp13BuilderState *state,
	SparkGlm52Pp13BuilderLayer *layer,
	uint32_t layer_index)
{
	SparkStatus status;
#define LOAD(suffix, dtype, bpe, rank, d0, d1, field) \
	do { status = SparkGlm52Pp13BuilderLoadLayerTensor(state,layer_index,suffix,dtype,bpe,rank,d0,d1,&layer->field); if (status != SPARK_STATUS_OK) return status; } while (0)
	LOAD("input_layernorm.weight","BF16",sizeof(uint16_t),1u,SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,1u,attention_norm_weight);
	LOAD("post_attention_layernorm.weight","BF16",sizeof(uint16_t),1u,SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,1u,post_attention_norm_weight);
	LOAD("self_attn.q_a_layernorm.weight","BF16",sizeof(uint16_t),1u,SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION,1u,raw_query_a_norm_weight);
	LOAD("self_attn.kv_a_layernorm.weight","BF16",sizeof(uint16_t),1u,SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION,1u,raw_kv_a_norm_weight);
	LOAD("self_attn.q_a_proj.weight","F8_E4M3",sizeof(uint8_t),2u,SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION,SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,raw_query_a_weight_fp8);
	LOAD("self_attn.q_a_proj.weight_scale_inv","F32",sizeof(float),2u,SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_EXTENT(SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION),SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_EXTENT(SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION),raw_query_a_scale);
	LOAD("self_attn.q_b_proj.weight","F8_E4M3",sizeof(uint8_t),2u,SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_B_DIMENSION,SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION,raw_query_b_weight_fp8);
	LOAD("self_attn.q_b_proj.weight_scale_inv","F32",sizeof(float),2u,SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_EXTENT(SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_B_DIMENSION),SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_EXTENT(SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION),raw_query_b_scale);
	LOAD("self_attn.kv_a_proj_with_mqa.weight","F8_E4M3",sizeof(uint8_t),2u,SPARK_GLM52_RESIDENT_DECODE_STAGE_KV_A_DIMENSION,SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,raw_kv_a_weight_fp8);
	LOAD("self_attn.kv_a_proj_with_mqa.weight_scale_inv","F32",sizeof(float),2u,SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_EXTENT(SPARK_GLM52_RESIDENT_DECODE_STAGE_KV_A_DIMENSION),SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_EXTENT(SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION),raw_kv_a_scale);
	LOAD("self_attn.kv_b_proj.weight","F8_E4M3",sizeof(uint8_t),2u,SPARK_GLM52_RESIDENT_DECODE_STAGE_KV_B_DIMENSION,SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION,raw_kv_b_weight_fp8);
	LOAD("self_attn.kv_b_proj.weight_scale_inv","F32",sizeof(float),2u,SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_EXTENT(SPARK_GLM52_RESIDENT_DECODE_STAGE_KV_B_DIMENSION),SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_EXTENT(SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION),raw_kv_b_scale);
	LOAD("self_attn.o_proj.weight","F8_E4M3",sizeof(uint8_t),2u,SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,SPARK_GLM52_RESIDENT_DECODE_STAGE_ATTENTION_PROJECTION_DIMENSION,attention_output_weight_fp8);
	LOAD("self_attn.o_proj.weight_scale_inv","F32",sizeof(float),2u,SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_EXTENT(SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION),SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_EXTENT(SPARK_GLM52_RESIDENT_DECODE_STAGE_ATTENTION_PROJECTION_DIMENSION),attention_output_scale);
	if (SparkGlm52Pp13BuilderDsaSourceLayer(layer_index) == layer_index)
	{
		LOAD("self_attn.indexer.wq_b.weight","F8_E4M3",sizeof(uint8_t),2u,SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_INDEX_QUERY_DIMENSION,SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION,index_query_weight_fp8);
		LOAD("self_attn.indexer.wq_b.weight_scale_inv","F32",sizeof(float),2u,SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_EXTENT(SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_INDEX_QUERY_DIMENSION),SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_EXTENT(SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION),index_query_scale);
		LOAD("self_attn.indexer.wk.weight","F8_E4M3",sizeof(uint8_t),2u,SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_INDEX_KEY_DIMENSION,SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,index_key_weight_fp8);
		LOAD("self_attn.indexer.wk.weight_scale_inv","F32",sizeof(float),2u,SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_EXTENT(SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_INDEX_KEY_DIMENSION),SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_EXTENT(SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION),index_key_scale);
		LOAD("self_attn.indexer.weights_proj.weight","BF16",sizeof(uint16_t),2u,SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_INDEX_WEIGHT_DIMENSION,SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,index_weights_proj_weight);
		LOAD("self_attn.indexer.k_norm.weight","BF16",sizeof(uint16_t),1u,SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_INDEX_KEY_DIMENSION,1u,index_key_norm_weight);
		LOAD("self_attn.indexer.k_norm.bias","BF16",sizeof(uint16_t),1u,SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_INDEX_KEY_DIMENSION,1u,index_key_norm_bias);
	}
	if (layer_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_FIRST_ROUTED_LAYER)
	{
		LOAD("mlp.gate_proj.weight","F8_E4M3",sizeof(uint8_t),2u,SPARK_GLM52_RESIDENT_DECODE_STAGE_DENSE_INTERMEDIATE_DIMENSION,SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,dense_gate_weight_fp8);
		LOAD("mlp.gate_proj.weight_scale_inv","F32",sizeof(float),2u,SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_EXTENT(SPARK_GLM52_RESIDENT_DECODE_STAGE_DENSE_INTERMEDIATE_DIMENSION),SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_EXTENT(SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION),dense_gate_scale);
		LOAD("mlp.up_proj.weight","F8_E4M3",sizeof(uint8_t),2u,SPARK_GLM52_RESIDENT_DECODE_STAGE_DENSE_INTERMEDIATE_DIMENSION,SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,dense_up_weight_fp8);
		LOAD("mlp.up_proj.weight_scale_inv","F32",sizeof(float),2u,SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_EXTENT(SPARK_GLM52_RESIDENT_DECODE_STAGE_DENSE_INTERMEDIATE_DIMENSION),SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_EXTENT(SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION),dense_up_scale);
		LOAD("mlp.down_proj.weight","F8_E4M3",sizeof(uint8_t),2u,SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,SPARK_GLM52_RESIDENT_DECODE_STAGE_DENSE_INTERMEDIATE_DIMENSION,dense_down_weight_fp8);
		LOAD("mlp.down_proj.weight_scale_inv","F32",sizeof(float),2u,SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_EXTENT(SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION),SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_EXTENT(SPARK_GLM52_RESIDENT_DECODE_STAGE_DENSE_INTERMEDIATE_DIMENSION),dense_down_scale);
	}
	else
	{
		LOAD("mlp.gate.weight","BF16",sizeof(uint16_t),2u,SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT,SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,router_weight);
		LOAD("mlp.gate.e_score_correction_bias","F32",sizeof(float),1u,SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT,1u,router_bias);
		LOAD("mlp.shared_experts.gate_proj.weight","F8_E4M3",sizeof(uint8_t),2u,SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION,SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,dense_gate_weight_fp8);
		LOAD("mlp.shared_experts.gate_proj.weight_scale_inv","F32",sizeof(float),2u,SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_EXTENT(SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION),SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_EXTENT(SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION),dense_gate_scale);
		LOAD("mlp.shared_experts.up_proj.weight","F8_E4M3",sizeof(uint8_t),2u,SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION,SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,dense_up_weight_fp8);
		LOAD("mlp.shared_experts.up_proj.weight_scale_inv","F32",sizeof(float),2u,SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_EXTENT(SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION),SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_EXTENT(SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION),dense_up_scale);
		LOAD("mlp.shared_experts.down_proj.weight","F8_E4M3",sizeof(uint8_t),2u,SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION,dense_down_weight_fp8);
		LOAD("mlp.shared_experts.down_proj.weight_scale_inv","F32",sizeof(float),2u,SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_EXTENT(SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION),SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_EXTENT(SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION),dense_down_scale);
	}
#undef LOAD
	return SPARK_STATUS_OK;
}

static void SparkGlm52Pp13BuilderWireLayerSerialPrefillPlan(
	SparkGlm52Pp13BuilderState *state,
	SparkGlm52Pp13BuilderLayer *layer)
{
	memset(&layer->serial_prefill_paged_plan,0,sizeof(layer->serial_prefill_paged_plan));
	layer->serial_prefill_paged_plan.abi_version =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_PAGED_PREFILL_PLAN_ABI_VERSION;
	layer->serial_prefill_paged_plan.descriptor_bytes =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_PAGED_PREFILL_PLAN_DESCRIPTOR_BYTES;
	layer->serial_prefill_paged_plan.block_token_count =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_BLOCK_TOKENS;
	layer->serial_prefill_paged_plan.maximum_prompt_token_count = 1u;
	layer->serial_prefill_paged_plan.maximum_active_sequence_count =
		state->rank_plan.max_active_sequence_count;
	layer->serial_prefill_paged_plan.hidden_dimension =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION;
	layer->serial_prefill_paged_plan.cache_token_elements =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_CACHE_TOKEN_ELEMENTS;
	layer->serial_prefill_paged_plan.prompt_positions =
		(const uint32_t *)state->device_prefill_positions;
	layer->serial_prefill_paged_plan.prompt_slot_mapping =
		(const uint32_t *)state->device_prefill_slot_mapping;
	layer->serial_prefill_paged_plan.prompt_context_lengths =
		(const uint32_t *)state->device_prefill_context_lengths;
	layer->serial_prefill_paged_plan.prompt_first_block_token_offsets =
		(const uint32_t *)state->device_prefill_first_block_token_offsets;
	layer->serial_prefill_paged_plan.prompt_block_table =
		(const uint32_t *)state->device_physical_block_indices;
	layer->serial_prefill_paged_plan.prompt_hidden_bf16 =
		layer->slot.input_hidden_bf16;
	layer->serial_prefill_paged_plan.prompt_token_counts =
		(const uint32_t *)state->device_prefill_token_counts;
	layer->serial_prefill_paged_plan.prompt_token_stride = 1u;
	layer->serial_prefill_paged_plan.query_tile_token_count =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_PAGED_PREFILL_QUERY_TILE_TOKENS;
	layer->serial_prefill_paged_plan.key_tile_token_count =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_PAGED_PREFILL_KEY_TILE_TOKENS;
	layer->serial_prefill_paged_plan.prompt_query_latent_bf16 =
		layer->query_latent;
	layer->serial_prefill_paged_plan.prompt_rotated_query_rope_bf16 =
		layer->rotated_query_rope;
	layer->serial_prefill_paged_plan.prompt_attention_output_latent_bf16 =
		layer->attention_output_latent;
	layer->serial_prefill_paged_plan.prompt_output_hidden_bf16 =
		layer->layer_output_hidden;
	layer->serial_prefill_paged_plan.validated_maximum_latency_ns =
		0u;

	memset(&layer->serial_prefill_bulk_plan,0,sizeof(layer->serial_prefill_bulk_plan));
	layer->serial_prefill_bulk_plan.abi_version =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_BULK_PREFILL_PLAN_ABI_VERSION;
	layer->serial_prefill_bulk_plan.maximum_active_sequence_count =
		state->rank_plan.max_active_sequence_count;
	layer->serial_prefill_bulk_plan.maximum_prompt_token_count = 1u;
	layer->serial_prefill_bulk_plan.capability_flags =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_BULK_PREFILL_PAGED_ATTENTION_CAPABILITIES;
	layer->serial_prefill_bulk_plan.opaque_state =
		&layer->serial_prefill_paged_plan;
	layer->serial_prefill_bulk_plan.validated_maximum_latency_ns =
		0u;
	layer->node.bulk_prefill_plan = &layer->serial_prefill_bulk_plan;
}

static void SparkGlm52Pp13BuilderWireLayer(
	SparkGlm52Pp13BuilderState *state,
	SparkGlm52Pp13BuilderLayer *layer,
	uint32_t layer_index)
{
	SparkGlm52ResidentDecodeStagePipelineSlot *slot;
	SparkGlm52ResidentDecodeStageNodeContext *node;
	uint32_t source_layer;
	uint32_t group_end;
	slot = &layer->slot;
	node = &layer->node;
	memset(slot,0,sizeof(*slot));
	slot->cuda_stream = (void *)state->stream;
	slot->input_hidden_bf16 = layer->input_hidden;
	slot->normalized_hidden_bf16 = layer->normalized_hidden;
	slot->query_latent_bf16 = layer->query_latent;
	slot->query_rope_input_bf16 = layer->query_rope_input;
	slot->key_rope_input_bf16 = layer->key_rope_input;
	slot->current_kv_latent_bf16 = layer->current_kv_latent;
	slot->raw_query_a_bf16 = layer->raw_query_a;
	slot->raw_query_a_normalized_bf16 = layer->raw_query_a_norm;
	slot->raw_query_b_bf16 = layer->raw_query_b;
	slot->raw_kv_a_bf16 = layer->raw_kv_a;
	slot->raw_kv_a_normalized_bf16 = layer->raw_kv_a_norm;
	slot->raw_kv_b_bf16 = layer->raw_kv_b;
	slot->positions = (const uint32_t *)layer->positions;
	slot->slot_mapping = (const uint32_t *)layer->slot_mapping;
	slot->block_table = (const uint32_t *)layer->block_table;
	slot->context_lengths = (const uint32_t *)layer->context_lengths;
	slot->first_block_token_offsets = (const uint32_t *)layer->first_block_token_offsets;
	slot->query_index_heads_bf16 = layer->query_index_heads;
	slot->current_key_index_bf16 = layer->current_key_index;
	slot->index_head_weights_bf16 = layer->index_head_weights;
	slot->dsa_candidate_count = SPARK_GLM52_RESIDENT_DECODE_STAGE_SELECTED_TOKEN_COUNT;
	slot->sparse_token_indices = (uint32_t *)layer->sparse_token_indices;
	slot->rotated_query_rope_bf16 = layer->rotated_query_rope;
	slot->attention_output_latent_bf16 = layer->attention_output_latent;
	slot->attention_projected_hidden_bf16 = layer->attention_projected_hidden;
	slot->post_attention_hidden_bf16 = layer->post_attention_hidden;
	slot->post_attention_normalized_hidden_bf16 = layer->post_attention_normalized_hidden;
	slot->moe_router_logits = (float *)layer->moe_router_logits;
	slot->moe_topk_expert_ids = (uint32_t *)layer->moe_topk_expert_ids;
	slot->moe_topk_weights = (float *)layer->moe_topk_weights;
	slot->moe_gate_bf16 = layer->moe_gate;
	slot->moe_up_bf16 = layer->moe_up;
	slot->moe_intermediate_bf16 = layer->moe_intermediate;
	slot->moe_route_output_bf16 = layer->moe_route_output;
	slot->layer_output_hidden_bf16 = layer->layer_output_hidden;
	slot->mtp_draft_hidden_bf16 = layer->mtp_draft_hidden;
	slot->restricted_logits = (float *)layer->restricted_logits;
	slot->restricted_selected_token_ids = (uint32_t *)layer->restricted_selected_token_ids;
	slot->restricted_selected_token_scores = (float *)layer->restricted_selected_token_scores;
	slot->mtp_draft_logits = (float *)layer->mtp_draft_logits;
	slot->mtp_draft_token_ids = (uint32_t *)layer->mtp_draft_token_ids;
	slot->mtp_draft_token_budgets = (const uint32_t *)layer->mtp_draft_token_budgets;
	slot->mtp_target_token_ids = (const uint32_t *)layer->mtp_target_token_ids;
	slot->mtp_accept_mask = (uint32_t *)layer->mtp_accept_mask;
	slot->mtp_committed_token_ids = (uint32_t *)layer->mtp_committed_token_ids;
	slot->mtp_event_counters = (uint32_t *)layer->mtp_event_counters;
	slot->phase_clock_cycles = (uint64_t *)layer->phase_clock_cycles;
	memset(&layer->cuda_slot,0,sizeof(layer->cuda_slot));
	layer->cuda_slot.abi_version =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_SLOT_STATE_ABI_VERSION;
	memset(node,0,sizeof(*node));
	node->abi_version = SPARK_GLM52_RESIDENT_DECODE_STAGE_NODE_CONTEXT_ABI_VERSION;
	node->pipeline_slot_count = SPARK_GLM52_PP13_BUILDER_PIPELINE_SLOT_COUNT;
	node->max_active_sequence_count = state->rank_plan.max_active_sequence_count;
	node->cache_token_capacity = state->configuration.kv_pool_token_capacity;
	node->kv_block_count = state->configuration.kv_pool_token_capacity /
		SPARK_GLM52_RESIDENT_DECODE_STAGE_BLOCK_TOKENS;
	node->max_blocks_per_sequence = SPARK_GLM52_PP13_BUILDER_MAX_BLOCKS_PER_SEQUENCE;
	node->position_count = SPARK_GLM52_PP13_BUILDER_POSITION_COUNT;
	node->dsa_candidate_capacity = SPARK_GLM52_KV_CONTEXT_TOKENS;
	node->qk_scale = SPARK_GLM52_MODEL_QK_SCALE;
	node->rms_norm_epsilon = SPARK_GLM52_MODEL_RMS_NORM_EPSILON;
	node->cos_table = (const float *)state->cos_table;
	node->sin_table = (const float *)state->sin_table;
	node->mla_cache_bf16 = layer->mla_cache;
	node->key_nope_cache_bf16 = layer->key_nope_cache;
	node->value_cache_bf16 = layer->value_cache;
	node->attention_norm_weight_bf16 = layer->attention_norm_weight;
	node->raw_query_a_norm_weight_bf16 = layer->raw_query_a_norm_weight;
	node->raw_kv_a_norm_weight_bf16 = layer->raw_kv_a_norm_weight;
	node->raw_query_a_weight_fp8_e4m3 = (const uint8_t *)layer->raw_query_a_weight_fp8;
	node->raw_query_a_weight_scale_inv_f32 = (const float *)layer->raw_query_a_scale;
	node->raw_query_b_weight_fp8_e4m3 = (const uint8_t *)layer->raw_query_b_weight_fp8;
	node->raw_query_b_weight_scale_inv_f32 = (const float *)layer->raw_query_b_scale;
	node->raw_kv_a_weight_fp8_e4m3 = (const uint8_t *)layer->raw_kv_a_weight_fp8;
	node->raw_kv_a_weight_scale_inv_f32 = (const float *)layer->raw_kv_a_scale;
	node->raw_kv_b_weight_fp8_e4m3 = (const uint8_t *)layer->raw_kv_b_weight_fp8;
	node->raw_kv_b_weight_scale_inv_f32 = (const float *)layer->raw_kv_b_scale;
	node->attention_output_weight_fp8_e4m3 = (const uint8_t *)layer->attention_output_weight_fp8;
	node->attention_output_weight_scale_inv_f32 = (const float *)layer->attention_output_scale;
	node->post_attention_norm_weight_bf16 = layer->post_attention_norm_weight;
	node->dense_gate_weight_fp8_e4m3 = (const uint8_t *)layer->dense_gate_weight_fp8;
	node->dense_gate_weight_scale_inv_f32 = (const float *)layer->dense_gate_scale;
	node->dense_up_weight_fp8_e4m3 = (const uint8_t *)layer->dense_up_weight_fp8;
	node->dense_up_weight_scale_inv_f32 = (const float *)layer->dense_up_scale;
	node->dense_down_weight_fp8_e4m3 = (const uint8_t *)layer->dense_down_weight_fp8;
	node->dense_down_weight_scale_inv_f32 = (const float *)layer->dense_down_scale;
	node->final_norm_weight_bf16 = layer->final_norm_weight;
	node->restricted_lm_head_weight_bf16 = layer->restricted_lm_head_weight;
	node->restricted_token_ids = (const uint32_t *)state->restricted_token_ids;
	node->pipeline_slots = slot;
	node->cuda_pipeline_slot_states = &layer->cuda_slot;
	node->projection_mode = SPARK_GLM52_RESIDENT_DECODE_STAGE_PROJECTION_RAW_GLM_FP8_E4M3;
	node->projection_backend_mode =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_PROJECTION_BACKEND_PREBOUND_TENSOR_CORE;
	node->attention_execution_mode =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_ATTENTION_EXECUTION_ABSORBED_LATENT;
	node->dsa_score_tiles_f32 = state->dsa_score_tiles;
	node->dsa_prefill_selected_u32 = state->dsa_prefill_selected;
	node->dsa_prefill_row_context_lengths_u32 = state->dsa_prefill_row_context_lengths;
	node->dsa_prefill_row_sequences_u32 = state->dsa_prefill_row_sequences;
	node->dsa_prefill_row_positions_u32 = state->dsa_prefill_row_positions;
	node->dsa_prefill_key_scratch_bf16 = state->dsa_prefill_key_scratch;
	node->dsa_prefill_query_a_bf16 = state->dsa_prefill_query_a;
	node->dsa_prefill_query_index_heads_bf16 = state->dsa_prefill_query_index_heads;
	node->dsa_prefill_index_weights_bf16 = state->dsa_prefill_index_weights;
	node->dsa_prefill_normalized_hidden_bf16 = state->dsa_prefill_normalized_hidden;
	node->dsa_prefill_low_scratch_bf16 = state->dsa_prefill_low_scratch;
	node->dsa_prefill_row_capacity = SPARK_GLM52_PP13_BUILDER_PREFILL_ROWS;
	node->dsa_score_row_capacity =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_SCORE_TILE_ROWS;
	node->model_quantization_mode =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_MODEL_QUANTIZATION_FP8_E4M3_8BIT;
	node->layer_index = layer_index;
	node->device_probe_hash_slots = state->device_probe_hash_slots;
	node->kv_block_token_count = SPARK_GLM52_RESIDENT_DECODE_STAGE_BLOCK_TOKENS;
	node->launch_check_mode = SPARK_GLM52_RESIDENT_DECODE_STAGE_LAUNCH_CHECK_NONE;
	node->phase_clock_mode = SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_CLOCK_DISABLED;
	node->enable_cuda_graph_replay = 1u;
	node->reserved_execution_flags =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_PREBOUND_PROJECTIONS |
		SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_GRAPH_REPLAY |
		SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_FAST_MLP |
		SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_FORBID_DEBUG_SYNCHRONIZATION |
		SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_STAGE_SLICE_PLAN |
		SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_MODEL_QUANTIZATION;
	if ((state->rank_plan.flags & SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_HAS_PREVIOUS) != 0u)
		node->reserved_execution_flags |=
			SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_HIDDEN_TRANSPORT_INPUT;
	if ((state->rank_plan.flags & SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_HAS_NEXT) != 0u)
		node->reserved_execution_flags |=
			SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_HIDDEN_TRANSPORT_OUTPUT;
	if ((state->rank_plan.flags & SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_FINAL_STAGE) == 0u ||
		layer_index + 1u != state->rank_plan.first_layer_index + state->rank_plan.layer_count)
		node->reserved_execution_flags |=
			SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_OUTPUT_HIDDEN_ONLY;
	node->validated_stage_latency_ns = 0u;
	node->estimated_service_time_ns = 0u;
	node->index_softmax_scale = SPARK_GLM52_MODEL_DSA_INDEX_SOFTMAX_SCALE;
	node->dsa_index_head_count = SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_INDEX_HEAD_COUNT;
	node->dsa_index_head_dimension = SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_INDEX_HEAD_DIMENSION;
	node->index_query_weight_fp8_e4m3 = (const uint8_t *)layer->index_query_weight_fp8;
	node->index_query_weight_scale_inv_f32 = (const float *)layer->index_query_scale;
	node->index_key_weight_fp8_e4m3 = (const uint8_t *)layer->index_key_weight_fp8;
	node->index_key_weight_scale_inv_f32 = (const float *)layer->index_key_scale;
	node->index_weights_proj_weight_bf16 = layer->index_weights_proj_weight;
	node->index_key_norm_weight_bf16 = layer->index_key_norm_weight;
	node->index_key_norm_bias_bf16 = layer->index_key_norm_bias;
	node->key_index_cache_bf16 = layer->key_index_cache;
	node->key_index_block_min_bf16 = layer->key_index_block_min;
	node->key_index_block_max_bf16 = layer->key_index_block_max;
	node->dsa_summary_dirty_flags_u8 = (uint8_t *)layer->dsa_summary_dirty_flags;
	node->selected_token_indices_by_layer = (uint32_t *)state->selected_token_indices_by_layer;
	node->selected_block_indices_by_layer = (uint32_t *)state->selected_block_indices_by_layer;
	node->selected_block_counts_by_layer = (uint32_t *)state->selected_block_counts_by_layer;
	node->dsa_selection_epoch_by_layer = (uint32_t *)state->dsa_selection_epoch_by_layer;
	node->dsa_selected_block_stride = SPARK_GLM52_RESIDENT_DECODE_STAGE_SELECTED_TOKEN_COUNT;
	node->dsa_selected_block_capacity = SPARK_GLM52_RESIDENT_DECODE_STAGE_SELECTED_TOKEN_COUNT;
	node->dsa_selected_block_layer_count =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_COUNT;
	source_layer = SparkGlm52Pp13BuilderDsaSourceLayer(layer_index);
	if (source_layer == UINT32_MAX)
	{
		node->sparse_index_mode =
			SPARK_GLM52_RESIDENT_DECODE_STAGE_SPARSE_INDEX_COPY_CONTEXT_PREFIX;
	}
	else
	{
		group_end = SparkGlm52Pp13BuilderDsaGroupEnd(source_layer);
		node->sparse_index_mode = source_layer == layer_index
			? SPARK_GLM52_RESIDENT_DECODE_STAGE_SPARSE_INDEX_DSA_INDEXSHARE_FULL
			: SPARK_GLM52_RESIDENT_DECODE_STAGE_SPARSE_INDEX_DSA_INDEXSHARE_SHARED;
		node->dsa_indexshare_source_layer_index = source_layer;
		node->dsa_indexshare_group_end_layer_exclusive = group_end;
		node->dsa_indexshare_selected_token_count =
			SPARK_GLM52_RESIDENT_DECODE_STAGE_SELECTED_TOKEN_COUNT;
		node->dsa_indexshare_layer_count =
			SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_COUNT;
	}
	if (layer_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_FIRST_ROUTED_LAYER)
	{
		node->layer_progression_mode =
			SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_DENSE_BF16_MLP;
		node->mlp_execution_mode =
			SPARK_GLM52_RESIDENT_DECODE_STAGE_MLP_EXECUTION_PREBOUND_QUANTIZED_TENSOR_CORE;
		node->dense_intermediate_dimension =
			SPARK_GLM52_RESIDENT_DECODE_STAGE_DENSE_INTERMEDIATE_DIMENSION;
	}
	else
	{
		node->layer_progression_mode =
			SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_ROUTED_FP8_TOPK;
		node->mlp_execution_mode =
			SPARK_GLM52_RESIDENT_DECODE_STAGE_MLP_EXECUTION_FP8_EXPERT_TENSOR_CORE;
		node->moe_router_weight_bf16 = layer->router_weight;
		node->moe_router_score_bias_f32 = (const float *)layer->router_bias;
		node->moe_routed_scaling_factor =
			SPARK_GLM52_MODEL_MOE_ROUTED_SCALING_FACTOR;
		node->moe_norm_topk_prob = 1u;
		node->moe_expert_count = SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT;
		node->moe_bound_expert_count = SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT;
		node->moe_top_k = SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_TOP_K;
		node->moe_intermediate_dimension =
			SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION;
		node->dense_intermediate_dimension =
			SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION;
		node->reserved_execution_flags |=
			SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_FAST_MOE_ROUTER;
	}
}

static SparkStatus SparkGlm52Pp13BuilderBindLayerPlans(
	SparkGlm52Pp13BuilderState *state,
	SparkGlm52Pp13BuilderLayer *layer,
	uint32_t layer_index)
{
	SparkGlm52ResidentDecodeStageLinearPlanResidentBindingCreateInfo create_info;
	SparkGlm52ResidentDecodeStageLinearPlan *plans;
	uint32_t plan_count;
	uint32_t mask;
	SparkStatus status;
	memset(&create_info,0,sizeof(create_info));
	create_info.abi_version =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BINDING_ABI_VERSION;
	create_info.maximum_active_sequence_count = state->rank_plan.max_active_sequence_count;
	create_info.dense_intermediate_dimension =
		layer_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_FIRST_ROUTED_LAYER
		? SPARK_GLM52_RESIDENT_DECODE_STAGE_DENSE_INTERMEDIATE_DIMENSION
		: SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION;
	create_info.expert_count = SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT;
	mask = SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_RAW_ATTENTION_PROJECTIONS;
	if (SparkGlm52Pp13BuilderDsaSourceLayer(layer_index) == layer_index)
		mask |= SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_DSA_INDEXER;
	if (layer_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_FIRST_ROUTED_LAYER)
		mask |=
			SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_DENSE_GATE |
			SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_DENSE_UP |
			SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_DENSE_DOWN;
	else
		mask |=
			SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_DENSE_GATE |
			SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_DENSE_UP |
			SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_DENSE_DOWN |
			SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_ROUTER_LOGITS;
	if ((state->rank_plan.flags & SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_FINAL_STAGE) != 0u &&
		layer_index + 1u == state->rank_plan.first_layer_index + state->rank_plan.layer_count)
		mask |= SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_RESTRICTED_LOGITS;
	create_info.required_plan_mask = mask;
	create_info.workspace_limit_bytes =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BINDING_DEFAULT_WORKSPACE_BYTES;
	create_info.autotune_warmup_iterations = 1u;
	create_info.autotune_measurement_iterations = 1u;
	create_info.cuda_stream = (void *)state->stream;
	create_info.dense_input_bf16 = layer->post_attention_normalized_hidden;
	create_info.dense_gate_weight_fp8_e4m3 = layer->dense_gate_weight_fp8;
	create_info.dense_gate_weight_scale_inv_f32 = layer->dense_gate_scale;
	create_info.dense_up_weight_fp8_e4m3 = layer->dense_up_weight_fp8;
	create_info.dense_up_weight_scale_inv_f32 = layer->dense_up_scale;
	create_info.dense_down_weight_fp8_e4m3 = layer->dense_down_weight_fp8;
	create_info.dense_down_weight_scale_inv_f32 = layer->dense_down_scale;
	create_info.dense_gate_output_bf16 = layer->moe_gate;
	create_info.dense_up_output_bf16 = layer->moe_up;
	create_info.dense_intermediate_bf16 = layer->moe_intermediate;
	create_info.dense_down_output_bf16 = layer->layer_output_hidden;
	create_info.router_input_bf16 = layer->post_attention_normalized_hidden;
	create_info.router_weight_bf16 = layer->router_weight;
	create_info.router_logits_f32 = layer->moe_router_logits;
	create_info.raw_projection_input_bf16 = layer->normalized_hidden;
	create_info.raw_query_a_weight_fp8_e4m3 = layer->raw_query_a_weight_fp8;
	create_info.raw_query_a_weight_scale_inv_f32 = layer->raw_query_a_scale;
	create_info.raw_query_a_output_bf16 = layer->raw_query_a;
	create_info.raw_query_b_input_bf16 = layer->raw_query_a_norm;
	create_info.raw_query_b_weight_fp8_e4m3 = layer->raw_query_b_weight_fp8;
	create_info.raw_query_b_weight_scale_inv_f32 = layer->raw_query_b_scale;
	create_info.raw_query_b_output_bf16 = layer->raw_query_b;
	create_info.raw_kv_a_weight_fp8_e4m3 = layer->raw_kv_a_weight_fp8;
	create_info.raw_kv_a_weight_scale_inv_f32 = layer->raw_kv_a_scale;
	create_info.raw_kv_a_output_bf16 = layer->raw_kv_a;
	create_info.raw_kv_b_input_bf16 = layer->raw_kv_a_norm;
	create_info.raw_kv_b_weight_fp8_e4m3 = layer->raw_kv_b_weight_fp8;
	create_info.raw_kv_b_weight_scale_inv_f32 = layer->raw_kv_b_scale;
	create_info.raw_kv_b_output_bf16 = layer->raw_kv_b;
	create_info.attention_output_input_bf16 = layer->attention_output_latent;
	create_info.attention_output_weight_fp8_e4m3 = layer->attention_output_weight_fp8;
	create_info.attention_output_weight_scale_inv_f32 = layer->attention_output_scale;
	create_info.attention_output_bf16 = layer->attention_projected_hidden;
	create_info.restricted_logits_input_bf16 = layer->layer_output_hidden;
	create_info.restricted_lm_head_weight_bf16 = layer->restricted_lm_head_weight;
	create_info.restricted_logits_f32 = layer->restricted_logits;
	create_info.dsa_query_input_bf16 = layer->raw_query_a_norm;
	create_info.dsa_query_weight_fp8_e4m3 = layer->index_query_weight_fp8;
	create_info.dsa_query_weight_scale_inv_f32 = layer->index_query_scale;
	create_info.dsa_query_output_bf16 = layer->query_index_heads;
	create_info.dsa_key_input_bf16 = layer->normalized_hidden;
	create_info.dsa_key_weight_fp8_e4m3 = layer->index_key_weight_fp8;
	create_info.dsa_key_weight_scale_inv_f32 = layer->index_key_scale;
	create_info.dsa_key_output_bf16 = layer->current_key_index;
	create_info.dsa_weights_input_bf16 = layer->normalized_hidden;
	create_info.dsa_weights_proj_weight_bf16 = layer->index_weights_proj_weight;
	create_info.dsa_weights_output_bf16 = layer->index_head_weights;
	status = SparkGlm52ResidentDecodeStageLinearPlanResidentBindingCreate(
		&layer->linear_binding,
		&create_info);
	if (status != SPARK_STATUS_OK)
		return status;
	plans = SparkGlm52ResidentDecodeStageLinearPlanResidentBindingMutablePlans(
		layer->linear_binding,
		&plan_count);
	if (plans == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	status = SparkGlm52Sm121RequiredDecodeStageBindBlackwellQuantizedRegularLinearPlans(
		plans,
		plan_count);
	if (status != SPARK_STATUS_OK)
		return status;
	status = SparkGlm52Sm121RequiredDecodeStageBindFp8E4m3LinearPlansScaledGemmBackend(
		plans,
		plan_count,
		&state->fp8_scaled_gemm_backend);
	if (status != SPARK_STATUS_OK)
		return status;
	layer->node.linear_plans = plans;
	layer->node.linear_plan_count = plan_count;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderBindFp8Moe(
	SparkGlm52Pp13BuilderState *state,
	SparkGlm52Pp13BuilderLayer *layer,
	uint32_t layer_index)
{
	SparkGlm52ResidentDecodeStageFp8MoeResidentBindingCreateInfo create_info;
	char path[SPARK_GLM52_PP13_RUNTIME_PACK_PATH_BYTES];
	SparkStatus status;
	if (layer_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_FIRST_ROUTED_LAYER)
		return SPARK_STATUS_OK;
	status = SparkGlm52Pp13RuntimeBuildFp8PackPath(
		state->configuration.fp8_pack_root,
		layer_index,
		path,
		(uint32_t)sizeof(path));
	if (status != SPARK_STATUS_OK)
		return status;
	memset(&create_info,0,sizeof(create_info));
	create_info.abi_version =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_PACK_ABI_VERSION;
	create_info.layer_index = layer_index;
	create_info.maximum_active_sequence_count = state->rank_plan.max_active_sequence_count;
	create_info.pack_path = path;
	status = SparkGlm52ResidentDecodeStageFp8MoeResidentBindingCreateFromPackFile(
		&layer->fp8_moe_binding,
		&create_info);
	if (status != SPARK_STATUS_OK)
		return status;
	layer->node.fp8_moe_plan = &layer->fp8_moe_binding.plan;
	layer->fp8_moe_ready = 1u;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderLaunchMtpFusion(
	SparkGlm52Pp13BuilderState *state,
	const uint32_t *token_ids,
	const uint32_t *positions,
	const void *hidden_bf16,
	uint32_t active_sequence_count,
	cudaStream_t stream)
{
	uint64_t fusion_elements;
	uint32_t fusion_blocks;
	SparkGlm52Pp13BuilderMtpNormInvKernel<<<
		active_sequence_count,
		SPARK_GLM52_PP13_BUILDER_THREADS,
		0u,
		stream>>>(
		token_ids,
		positions,
		(const uint16_t *)state->mtp_embedding_weight,
		(const uint16_t *)hidden_bf16,
		state->mtp_norm_inv,
		active_sequence_count,
		SPARK_GLM52_MODEL_RMS_NORM_EPSILON);
	if (cudaGetLastError() != cudaSuccess)
		return SPARK_STATUS_IO_ERROR;
	fusion_elements = (uint64_t)active_sequence_count *
		SPARK_GLM52_PP13_BUILDER_MTP_EH_INPUT_DIMENSION;
	fusion_blocks = (uint32_t)((fusion_elements +
		SPARK_GLM52_PP13_BUILDER_THREADS - 1u) /
		SPARK_GLM52_PP13_BUILDER_THREADS);
	SparkGlm52Pp13BuilderMtpFusionKernel<<<
		fusion_blocks,
		SPARK_GLM52_PP13_BUILDER_THREADS,
		0u,
		stream>>>(
		token_ids,
		positions,
		(const uint16_t *)state->mtp_embedding_weight,
		(const uint16_t *)hidden_bf16,
		(const uint16_t *)state->mtp_enorm_weight,
		(const uint16_t *)state->mtp_hnorm_weight,
		state->mtp_norm_inv,
		(uint16_t *)state->mtp_eh_input,
		active_sequence_count);
	return SparkGlm52Pp13BuilderCudaStatus(cudaGetLastError());
}

static SparkStatus SparkGlm52Pp13BuilderProjectMtpEh(
	SparkGlm52Pp13BuilderState *state,
	uint32_t active_sequence_count)
{
	float alpha;
	float beta;
	cublasStatus_t status;
	alpha = 1.0f;
	beta = 0.0f;
	status = cublasGemmEx(
		state->mtp_cublas_handle,
		CUBLAS_OP_T,
		CUBLAS_OP_N,
		SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
		active_sequence_count,
		SPARK_GLM52_PP13_BUILDER_MTP_EH_INPUT_DIMENSION,
		&alpha,
		state->mtp_eh_proj_weight,
		CUDA_R_16BF,
		SPARK_GLM52_PP13_BUILDER_MTP_EH_INPUT_DIMENSION,
		state->mtp_eh_input,
		CUDA_R_16BF,
		SPARK_GLM52_PP13_BUILDER_MTP_EH_INPUT_DIMENSION,
		&beta,
		state->mtp_layer.input_hidden,
		CUDA_R_16BF,
		SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
		CUBLAS_COMPUTE_32F,
		CUBLAS_GEMM_DEFAULT_TENSOR_OP);
	return SparkGlm52Pp13BuilderCublasStatus(status);
}

static SparkStatus SparkGlm52Pp13BuilderLaunchMtpMetadata(
	SparkGlm52Pp13BuilderState *state,
	const uint32_t *base_positions,
	uint32_t draft_index,
	uint32_t active_sequence_count,
	cudaStream_t stream)
{
	uint32_t blocks;
	blocks = (active_sequence_count + SPARK_GLM52_PP13_BUILDER_THREADS - 1u) /
		SPARK_GLM52_PP13_BUILDER_THREADS;
	SparkGlm52Pp13BuilderMtpMetadataKernel<<<
		blocks,
		SPARK_GLM52_PP13_BUILDER_THREADS,
		0u,
		stream>>>(
		base_positions,
		state->device_physical_block_indices,
		(uint32_t *)state->mtp_layer.positions,
		(uint32_t *)state->mtp_layer.slot_mapping,
		(uint32_t *)state->mtp_layer.context_lengths,
		(uint32_t *)state->mtp_layer.first_block_token_offsets,
		SPARK_GLM52_PP13_BUILDER_MAX_BLOCKS_PER_SEQUENCE,
		SPARK_GLM52_RESIDENT_DECODE_STAGE_BLOCK_TOKENS,
		draft_index,
		active_sequence_count);
	return SparkGlm52Pp13BuilderCudaStatus(cudaGetLastError());
}

static SparkStatus SparkGlm52Pp13BuilderLaunchMtpLayer(
	SparkGlm52Pp13BuilderState *state,
	const uint32_t *token_ids,
	const uint32_t *base_positions,
	const void *hidden_bf16,
	uint32_t draft_index,
	uint32_t active_sequence_count,
	cudaStream_t stream)
{
	SparkStatus status;
	status = SparkGlm52Pp13BuilderLaunchMtpMetadata(
		state,base_positions,draft_index,active_sequence_count,stream);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderLaunchMtpFusion(
			state,token_ids,(const uint32_t *)state->mtp_layer.positions,
			hidden_bf16,active_sequence_count,stream);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderProjectMtpEh(state,active_sequence_count);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Sm121RequiredDecodeStageLaunch(
			&state->mtp_layer.node,&state->mtp_layer.slot,0u,
			active_sequence_count,0,stream);
	return status;
}

static SparkStatus SparkGlm52Pp13BuilderStoreMtpDraft(
	SparkGlm52Pp13BuilderState *state,
	const SparkGlm52ResidentDecodeStagePipelineSlot *base_slot,
	uint32_t draft_index,
	uint32_t active_sequence_count,
	cudaStream_t stream)
{
	uint64_t elements;
	uint32_t blocks;
	elements = (uint64_t)active_sequence_count *
		SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION;
	blocks = (uint32_t)((elements + SPARK_GLM52_PP13_BUILDER_THREADS - 1u) /
		SPARK_GLM52_PP13_BUILDER_THREADS);
	SparkGlm52Pp13BuilderMtpStoreKernel<<<
		blocks,
		SPARK_GLM52_PP13_BUILDER_THREADS,
		0u,
		stream>>>(
		(const uint16_t *)state->mtp_layer.layer_output_hidden,
		(const uint32_t *)state->mtp_layer.restricted_selected_token_ids,
		(uint16_t *)base_slot->mtp_draft_hidden_bf16,
		base_slot->mtp_draft_token_ids,
		draft_index,
		active_sequence_count);
	return SparkGlm52Pp13BuilderCudaStatus(cudaGetLastError());
}

static SparkStatus SparkGlm52Pp13BuilderLaunchMtpDraftPlan(
	const SparkGlm52ResidentDecodeStageMtpDraftPlan *plan,
	const SparkGlm52ResidentDecodeStageNodeContext *node_context,
	const SparkGlm52ResidentDecodeStagePipelineSlot *base_slot,
	uint32_t active_sequence_count,
	void *cuda_stream)
{
	SparkGlm52Pp13BuilderState *state;
	const uint32_t *token_ids;
	const uint32_t *base_positions;
	const void *hidden_bf16;
	uint32_t draft_index;
	uint32_t use_previous;
	SparkStatus status;
	state = plan != 0 ? (SparkGlm52Pp13BuilderState *)plan->opaque_state : 0;
	if (state == 0 || state->mtp_ready == 0u || node_context == 0 ||
		base_slot == 0 || cuda_stream != (void *)state->stream ||
		active_sequence_count == 0u ||
		active_sequence_count > state->rank_plan.max_active_sequence_count ||
		base_slot->positions == 0 || base_slot->layer_output_hidden_bf16 == 0 ||
		base_slot->restricted_selected_token_ids == 0 ||
		base_slot->mtp_draft_hidden_bf16 == 0 ||
		base_slot->mtp_draft_token_ids == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	use_previous = state->mtp_use_previous_for_draft;
	base_positions = use_previous != 0u
		? state->mtp_base_positions : base_slot->positions;
	for (draft_index = 0u;
		 draft_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT;
		 ++draft_index)
	{
		token_ids = draft_index == 0u
			? base_slot->restricted_selected_token_ids
			: (const uint32_t *)state->mtp_layer.restricted_selected_token_ids;
		hidden_bf16 = draft_index == 0u
			? (use_previous != 0u
				? state->mtp_previous_target_hidden
				: base_slot->layer_output_hidden_bf16)
			: state->mtp_layer.layer_output_hidden;
		status = SparkGlm52Pp13BuilderLaunchMtpLayer(
			state,token_ids,base_positions,hidden_bf16,draft_index,
			active_sequence_count,(cudaStream_t)cuda_stream);
		if (status == SPARK_STATUS_OK)
			status = SparkGlm52Sm121RequiredDecodeStageLaunchFullVocabGreedy(
				state->mtp_layer.layer_output_hidden,state->mtp_shared_head_norm_weight,
				state->mtp_layer.normalized_hidden,
				node_context->restricted_lm_head_weight_bf16,
				node_context->restricted_token_ids,0,
				(uint32_t *)state->mtp_layer.restricted_selected_token_ids,
				(float *)state->mtp_layer.restricted_selected_token_scores,
				state->final_epilogue_workspace,state->exact_plan.workspace_bytes,
				active_sequence_count,state->rank_plan.max_active_sequence_count,
				SPARK_GLM52_MODEL_RMS_NORM_EPSILON,cuda_stream);
		if (status == SPARK_STATUS_OK)
			status = SparkGlm52Pp13BuilderStoreMtpDraft(
				state,base_slot,draft_index,active_sequence_count,
				(cudaStream_t)cuda_stream);
		if (status != SPARK_STATUS_OK)
			return status;
	}
	if (use_previous != 0u)
		state->mtp_previous_valid = 0u;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderLoadMtpWeights(
	SparkGlm52Pp13BuilderState *state)
{
	SparkStatus status;
	status = SparkGlm52Pp13BuilderLoadTensor(
		state,SPARK_GLM52_PP13_BUILDER_MTP_EMBEDDING_TENSOR,"BF16",
		sizeof(uint16_t),2u,SPARK_GLM52_RESIDENT_DECODE_STAGE_OUTPUT_VOCAB_COUNT,
		SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
		&state->mtp_embedding_weight);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderLoadLayerTensor(
			state,SPARK_GLM52_MODEL_MTP_LAYER_INDEX,"enorm.weight","BF16",
			sizeof(uint16_t),1u,SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
			1u,&state->mtp_enorm_weight);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderLoadLayerTensor(
			state,SPARK_GLM52_MODEL_MTP_LAYER_INDEX,"hnorm.weight","BF16",
			sizeof(uint16_t),1u,SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
			1u,&state->mtp_hnorm_weight);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderLoadLayerTensor(
			state,SPARK_GLM52_MODEL_MTP_LAYER_INDEX,"eh_proj.weight","BF16",
			sizeof(uint16_t),2u,SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
			SPARK_GLM52_PP13_BUILDER_MTP_EH_INPUT_DIMENSION,
			&state->mtp_eh_proj_weight);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderLoadLayerTensor(
			state,SPARK_GLM52_MODEL_MTP_LAYER_INDEX,"shared_head.norm.weight",
			"BF16",sizeof(uint16_t),1u,
			SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,1u,
			&state->mtp_shared_head_norm_weight);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaAlloc(
			state,&state->mtp_eh_input,
			(uint64_t)state->rank_plan.max_active_sequence_count *
			SPARK_GLM52_PP13_BUILDER_MTP_EH_INPUT_DIMENSION * sizeof(uint16_t));
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaAlloc(
			state,(void **)&state->mtp_norm_inv,
			(uint64_t)state->rank_plan.max_active_sequence_count *
			SPARK_GLM52_PP13_BUILDER_MTP_NORM_COUNT_PER_LANE * sizeof(float));
	return status;
}

static void SparkGlm52Pp13BuilderConfigureMtpLayer(
	SparkGlm52Pp13BuilderState *state)
{
	SparkGlm52ResidentDecodeStageNodeContext *node;
	uint32_t clear_flags;
	node = &state->mtp_layer.node;
	node->cache_token_capacity = state->configuration.kv_pool_token_capacity;
	node->kv_block_count = state->configuration.kv_pool_token_capacity /
		SPARK_GLM52_RESIDENT_DECODE_STAGE_BLOCK_TOKENS;
	node->max_blocks_per_sequence =
		SPARK_GLM52_PP13_BUILDER_MAX_BLOCKS_PER_SEQUENCE;
	node->dsa_candidate_capacity =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_SELECTED_TOKEN_COUNT;
	node->sparse_index_mode =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_SPARSE_INDEX_COPY_CONTEXT_PREFIX;
	node->dsa_indexshare_source_layer_index = 0u;
	node->dsa_indexshare_group_end_layer_exclusive = 0u;
	node->dsa_indexshare_selected_token_count = 0u;
	node->dsa_indexshare_layer_count = 0u;
	node->enable_cuda_graph_replay = 0u;
	node->bulk_prefill_plan = 0;
	state->mtp_layer.slot.block_table = state->device_physical_block_indices;
	clear_flags =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_GRAPH_REPLAY |
		SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_STAGE_SLICE_PLAN |
		SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_HIDDEN_TRANSPORT_INPUT |
		SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_HIDDEN_TRANSPORT_OUTPUT;
	node->reserved_execution_flags &= ~clear_flags;
	node->reserved_execution_flags |=
		SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_OUTPUT_HIDDEN_ONLY;
}

static SparkStatus SparkGlm52Pp13BuilderInitializeMtp(
	SparkGlm52Pp13BuilderState *state)
{
	SparkStatus status;
	if (!SparkGlm52Pp13BuilderMtpEnabled(state) ||
		!SparkGlm52Pp13BuilderIsFinalRank(state))
		return SPARK_STATUS_OK;
	status = SparkGlm52Pp13BuilderAllocateLayerBuffers(
		state,&state->mtp_layer,UINT32_MAX,
		state->configuration.kv_pool_token_capacity,
		SPARK_GLM52_PP13_BUILDER_MAX_BLOCKS_PER_SEQUENCE);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderLoadLayerWeights(
			state,&state->mtp_layer,SPARK_GLM52_MODEL_MTP_LAYER_INDEX);
	if (status == SPARK_STATUS_OK)
	{
		SparkGlm52Pp13BuilderWireLayer(
			state,&state->mtp_layer,SPARK_GLM52_MODEL_MTP_LAYER_INDEX);
		SparkGlm52Pp13BuilderConfigureMtpLayer(state);
	}
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderLoadMtpWeights(state);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaAlloc(
			state,&state->mtp_previous_target_hidden,
			(uint64_t)state->rank_plan.max_active_sequence_count *
			SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_BF16_BYTES);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaAlloc(
			state,(void **)&state->mtp_base_positions,
			(uint64_t)state->rank_plan.max_active_sequence_count * sizeof(uint32_t));
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderBindFp8Moe(
			state,&state->mtp_layer,SPARK_GLM52_MODEL_MTP_LAYER_INDEX);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderBindLayerPlans(
			state,&state->mtp_layer,SPARK_GLM52_MODEL_MTP_LAYER_INDEX);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Sm121RequiredDecodeStageInitialize(&state->mtp_layer.node);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCublasStatus(
			cublasCreate(&state->mtp_cublas_handle));
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCublasStatus(
			cublasSetStream(state->mtp_cublas_handle,state->stream));
	if (status != SPARK_STATUS_OK)
		return SparkGlm52Pp13BuilderReportStatus(
			"initialize_mtp",SPARK_GLM52_MODEL_MTP_LAYER_INDEX,status);
	memset(&state->mtp_draft_plan,0,sizeof(state->mtp_draft_plan));
	state->mtp_draft_plan.abi_version =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_PLAN_ABI_VERSION;
	state->mtp_draft_plan.restricted_vocab_count =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_RESTRICTED_VOCAB_COUNT;
	state->mtp_draft_plan.hidden_dimension =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION;
	state->mtp_draft_plan.draft_token_count =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT;
	state->mtp_draft_plan.weight_format =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_BF16;
	state->mtp_draft_plan.launch_function =
		(void *)SparkGlm52Pp13BuilderLaunchMtpDraftPlan;
	state->mtp_draft_plan.opaque_state = state;
	state->mtp_draft_plan.validated_maximum_latency_ns = 0u;
	state->mtp_ready = 1u;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderBuildLayer(
	SparkGlm52Pp13BuilderState *state,
	uint32_t layer_offset)
{
	SparkGlm52Pp13BuilderLayer *layer;
	uint32_t layer_index;
	SparkStatus status;
	layer = &state->layers[layer_offset];
	layer_index = state->rank_plan.first_layer_index + layer_offset;
	status = SparkGlm52Pp13BuilderAllocateLayerBuffers(
		state,
		layer,
		layer_offset,
		state->configuration.kv_pool_token_capacity,
		SPARK_GLM52_PP13_BUILDER_MAX_BLOCKS_PER_SEQUENCE);
	if (status != SPARK_STATUS_OK)
		return SparkGlm52Pp13BuilderReportStatus("allocate_layer_buffers",layer_index,status);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderLoadLayerWeights(state,layer,layer_index);
	if (status != SPARK_STATUS_OK)
		return SparkGlm52Pp13BuilderReportStatus("load_layer_weights",layer_index,status);
	SparkGlm52Pp13BuilderWireLayer(state,layer,layer_index);
	if (layer_offset > 0u)
		layer->slot.input_hidden_bf16 =
			state->layers[layer_offset - 1u].layer_output_hidden;
	SparkGlm52Pp13BuilderWireLayerSerialPrefillPlan(state,layer);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderBindFp8Moe(state,layer,layer_index);
	if (status != SPARK_STATUS_OK)
		return SparkGlm52Pp13BuilderReportStatus("bind_fp8_moe",layer_index,status);
	if (status == SPARK_STATUS_OK &&
		(state->rank_plan.flags & SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_FINAL_STAGE) != 0u &&
		layer_offset + 1u == state->rank_plan.layer_count)
	{
		status = SparkGlm52Pp13BuilderLoadTensor(
			state,
			"model.norm.weight",
			"BF16",
			sizeof(uint16_t),
			1u,
			SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
			1u,
			&layer->final_norm_weight);
		if (status == SPARK_STATUS_OK)
			status = SparkGlm52Pp13BuilderLoadLmHeadRestricted(
				state,
				&layer->restricted_lm_head_weight);
		layer->node.final_norm_weight_bf16 = layer->final_norm_weight;
		layer->node.restricted_lm_head_weight_bf16 =
			layer->restricted_lm_head_weight;
	}
	if (status != SPARK_STATUS_OK)
		return SparkGlm52Pp13BuilderReportStatus("load_final_outputs",layer_index,status);
	if (SparkGlm52Pp13BuilderMtpEnabled(state) &&
		(state->rank_plan.flags & SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_FINAL_STAGE) != 0u &&
		layer_offset + 1u == state->rank_plan.layer_count)
	{
		if (state->mtp_ready == 0u)
			return SparkGlm52Pp13BuilderReportStatus(
				"mtp_not_ready",layer_index,SPARK_STATUS_MODULE_NOT_VALIDATED);
		layer->node.mtp_draft_plan = &state->mtp_draft_plan;
	}
	status = SparkGlm52Pp13BuilderBindLayerPlans(state,layer,layer_index);
	if (status != SPARK_STATUS_OK)
		return SparkGlm52Pp13BuilderReportStatus("bind_layer_plans",layer_index,status);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Sm121RequiredDecodeStageInitialize(&layer->node);
	layer->node.reserved_execution_flags &=
		~SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_DSA_KV_FRAGMENT_TRANSPORT;
	layer->node.dsa_kv_fragment_prefetch_plan = 0;
	layer->node.dsa_kv_fragment_save_plan = 0;
	if (status != SPARK_STATUS_OK)
		return SparkGlm52Pp13BuilderReportStatus("required_stage_initialize",layer_index,status);
	if (status == SPARK_STATUS_OK)
	{
		state->layer_pointers[layer_offset] = &layer->node;
	}
	return status;
}

static SparkStatus SparkGlm52Pp13BuilderInitializeExactPlan(
	SparkGlm52Pp13BuilderState *state)
{
	uint64_t candidates;
	uint64_t workspace_bytes;
	uint32_t batch_bucket;
	uint32_t capability_flags;
	SparkStatus status;
	batch_bucket =
		SparkGlm52StagePlanSelectBatchBucketValue(
			state->rank_plan.max_active_sequence_count);
	candidates =
		(uint64_t)batch_bucket *
		(uint64_t)(SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT + 1u) *
		(uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_FINAL_EPILOGUE_CANDIDATE_GROUP_COUNT;
	workspace_bytes =
		(candidates * sizeof(float)) + (candidates * sizeof(uint32_t)) + 15u;
	status = SparkGlm52Pp13BuilderCudaAlloc(
		state,
		&state->final_epilogue_workspace,
		workspace_bytes);
	if (status != SPARK_STATUS_OK)
		return status;
	if (cudaStreamCreate(&state->query_stream) != cudaSuccess ||
		cudaStreamCreate(&state->kv_stream) != cudaSuccess ||
		cudaEventCreate(&state->branch_ready_event) != cudaSuccess ||
		cudaEventCreate(&state->query_event) != cudaSuccess ||
		cudaEventCreate(&state->kv_event) != cudaSuccess)
		return SPARK_STATUS_IO_ERROR;
	capability_flags =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_SLICE_PRODUCTION_PP13_CAPABILITIES &
		~SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_SLICE_CAPABILITY_QKV_BRANCH_OVERLAP;
	memset(&state->exact_plan,0,sizeof(state->exact_plan));
	state->exact_plan.abi_version =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_EXACT_STAGE_SLICE_PLAN_ABI_VERSION;
	state->exact_plan.descriptor_bytes =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_EXACT_STAGE_SLICE_PLAN_DESCRIPTOR_BYTES;
	state->exact_plan.stage_index = state->rank_plan.rank_index;
	state->exact_plan.first_layer_index = state->rank_plan.first_layer_index;
	state->exact_plan.layer_count = state->rank_plan.layer_count;
	state->exact_plan.batch_bucket = batch_bucket;
	state->exact_plan.maximum_active_sequence_count =
		state->rank_plan.max_active_sequence_count;
	state->exact_plan.capability_flags = capability_flags;
	state->exact_plan.query_branch_stream = (void *)state->query_stream;
	state->exact_plan.kv_branch_stream = (void *)state->kv_stream;
	state->exact_plan.branch_ready_event = (void *)state->branch_ready_event;
	state->exact_plan.query_branch_event = (void *)state->query_event;
	state->exact_plan.kv_branch_event = (void *)state->kv_event;
	state->exact_plan.workspace = state->final_epilogue_workspace;
	state->exact_plan.workspace_bytes = workspace_bytes;
	state->exact_plan.validated_maximum_latency_ns = 0u;
	memset(&state->stage_slice_plan,0,sizeof(state->stage_slice_plan));
	state->stage_slice_plan.abi_version =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_SLICE_PLAN_ABI_VERSION;
	state->stage_slice_plan.maximum_active_sequence_count =
		state->rank_plan.max_active_sequence_count;
	state->stage_slice_plan.maximum_layer_count = state->rank_plan.layer_count;
	state->stage_slice_plan.capability_flags = capability_flags;
	state->stage_slice_plan.opaque_state = &state->exact_plan;
	state->stage_slice_plan.workspace = state->final_epilogue_workspace;
	state->stage_slice_plan.workspace_bytes = workspace_bytes;
	state->stage_slice_plan.validated_maximum_latency_ns = 0u;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderInitializeFp8ScaledGemm(
	SparkGlm52Pp13BuilderState *state)
{
	uint64_t workspace_bytes;
	SparkStatus status;
	workspace_bytes =
		SparkGlm52Sm121RequiredDecodeStageCalculateBuiltinFp8ScaledGemmWorkspaceBytes();
	if (workspace_bytes == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	status = SparkGlm52Pp13BuilderCudaAlloc(
		state,
		&state->fp8_scaled_gemm_workspace,
		workspace_bytes);
	if (status != SPARK_STATUS_OK)
		return status;
	return SparkGlm52Sm121RequiredDecodeStageInitializeBuiltinFp8ScaledGemmBackend(
		&state->fp8_scaled_gemm_state,
		state->fp8_scaled_gemm_workspace,
		workspace_bytes,
		&state->fp8_scaled_gemm_backend);
}

static SparkStatus SparkGlm52Pp13BuilderInitializeRank0InputBuffers(
	SparkGlm52Pp13BuilderState *state)
{
	uint64_t max_active;
	uint64_t prefill_tokens;
	uint64_t prefill_hidden_words;
	uint64_t prefill_hidden_bytes;
	uint32_t rank_has_previous;
	SparkStatus status;
	max_active = state->rank_plan.max_active_sequence_count;
	prefill_tokens = max_active * SPARK_GLM52_PP13_BUILDER_MAX_PREFILL_TOKENS;
	prefill_hidden_words =
		prefill_tokens *
		(SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION / 2u);
	prefill_hidden_bytes = prefill_hidden_words * sizeof(uint32_t);
	rank_has_previous =
		(state->rank_plan.flags & SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_HAS_PREVIOUS) != 0u;
	status = SPARK_STATUS_OK;
	if (rank_has_previous == 0u)
	{
		status = SparkGlm52Pp13BuilderLoadEmbedding(state);
		if (status == SPARK_STATUS_OK)
			status = SparkGlm52Pp13BuilderCudaAlloc(
				state,&state->device_prefill_token_ids,
				prefill_tokens * sizeof(uint32_t));
		if (status == SPARK_STATUS_OK)
			status = SparkGlm52Pp13BuilderCudaAlloc(
				state,&state->device_prefill_hidden,
				prefill_hidden_bytes);
		if (status == SPARK_STATUS_OK)
			status = SparkGlm52Pp13BuilderCudaAlloc(
				state,&state->device_prefill_output_hidden,
				prefill_hidden_bytes);
	}
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaAlloc(
			state,&state->device_prefill_positions,
			prefill_tokens * sizeof(uint32_t));
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaAlloc(
			state,&state->device_prefill_slot_mapping,
			prefill_tokens * sizeof(uint32_t));
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaAlloc(
			state,&state->device_prefill_context_lengths,
			max_active * sizeof(uint32_t));
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaAlloc(
			state,&state->device_prefill_first_block_token_offsets,
			max_active * sizeof(uint32_t));
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaAlloc(
			state,&state->device_prefill_token_counts,
			max_active * sizeof(uint32_t));
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaAlloc(
			state,&state->device_decode_positions,
			max_active * sizeof(uint32_t));
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaAlloc(
			state,&state->device_decode_token_ids,
			max_active * sizeof(uint32_t));
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaAlloc(
			state,&state->device_mtp_draft_token_budgets,
			max_active * sizeof(uint32_t));
	if (status != SPARK_STATUS_OK)
		return status;
	state->host_prefill_lane_offsets =
		(uint32_t *)malloc((size_t)(max_active * sizeof(uint32_t)));
	state->host_prefill_lane_counts =
		(uint32_t *)malloc((size_t)(max_active * sizeof(uint32_t)));
	state->host_decode_positions =
		(uint32_t *)malloc((size_t)(max_active * sizeof(uint32_t)));
	state->host_decode_token_ids =
		(uint32_t *)malloc((size_t)(max_active * sizeof(uint32_t)));
	state->host_decode_result_token_ids =
		(uint32_t *)malloc((size_t)(max_active * sizeof(uint32_t)));
	state->host_mtp_committed_token_ids =
		(uint32_t *)malloc(
			(size_t)(max_active *
				SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT *
				sizeof(uint32_t)));
	if (state->host_prefill_lane_offsets == 0 ||
		state->host_prefill_lane_counts == 0 ||
		state->host_decode_positions == 0 ||
		state->host_decode_token_ids == 0 ||
		state->host_decode_result_token_ids == 0 ||
		state->host_mtp_committed_token_ids == 0)
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderInitializeSharedBuffers(
	SparkGlm52Pp13BuilderState *state)
{
	uint64_t max_active;
	uint64_t sideband_bytes;
	uint64_t selected_indices_bytes;
	uint64_t selected_block_bytes;
	uint64_t kv_entries;
	uint32_t physical_block_count;
	SparkStatus status;
	max_active = state->rank_plan.max_active_sequence_count;
	physical_block_count = state->configuration.kv_pool_token_capacity /
		SPARK_GLM52_RESIDENT_DECODE_STAGE_BLOCK_TOKENS;
	selected_indices_bytes =
		(uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_COUNT *
		max_active *
		(uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_SELECTED_TOKEN_COUNT *
		sizeof(uint32_t);
	selected_block_bytes =
		(uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_COUNT *
		max_active *
		(uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_SELECTED_TOKEN_COUNT *
		sizeof(uint32_t);
	sideband_bytes =
		max_active *
		(uint64_t)SPARK_GLM52_PP13_RUNTIME_MAX_SIDEBAND_BYTES_PER_SEQUENCE;
	kv_entries = max_active * SPARK_GLM52_PP13_BUILDER_MAX_BLOCKS_PER_SEQUENCE;
	status = SparkGlm52Pp13BuilderCudaAlloc(
		state,&state->selected_token_indices_by_layer,selected_indices_bytes);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaAlloc(
			state,&state->selected_block_indices_by_layer,selected_block_bytes);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaAlloc(
			state,
			&state->selected_block_counts_by_layer,
			(uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_COUNT *
			max_active *
			sizeof(uint32_t));
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaAlloc(
			state,
			&state->dsa_selection_epoch_by_layer,
			(uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_COUNT *
			max_active *
			sizeof(uint32_t));
	if (status == SPARK_STATUS_OK)
	{
		if ((state->rank_plan.flags & SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_HAS_PREVIOUS) != 0u)
			status = SparkGlm52Pp13BuilderCudaHostMappedAlloc(state,&state->input_sideband,sideband_bytes);
		else
			status = SparkGlm52Pp13BuilderCudaAlloc(state,&state->input_sideband,sideband_bytes);
	}
	if (status == SPARK_STATUS_OK)
	{
		if ((state->rank_plan.flags & SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_HAS_NEXT) != 0u)
			status = SparkGlm52Pp13BuilderCudaHostMappedAlloc(state,&state->output_sideband,sideband_bytes);
		else
			status = SparkGlm52Pp13BuilderCudaAlloc(state,&state->output_sideband,sideband_bytes);
	}
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaAlloc(
			state,
			(void **)&state->device_physical_block_indices,
			kv_entries * sizeof(uint32_t));
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaAlloc(
			state,
			(void **)&state->device_lane_physical_block_counts,
			max_active * sizeof(uint32_t));
	if (status != SPARK_STATUS_OK)
		return status;
	state->host_physical_block_indices =
		(uint32_t *)malloc((size_t)(kv_entries * sizeof(uint32_t)));
	state->host_lane_physical_block_counts =
		(uint32_t *)malloc((size_t)(max_active * sizeof(uint32_t)));
	state->host_physical_block_states =
		(uint8_t *)malloc((size_t)(physical_block_count * sizeof(uint8_t)));
	state->host_mtp_draft_budgets =
		(uint32_t *)malloc((size_t)(max_active * sizeof(uint32_t)));
	if (state->host_physical_block_indices == 0 ||
		state->host_lane_physical_block_counts == 0 ||
		state->host_physical_block_states == 0 ||
		state->host_mtp_draft_budgets == 0)
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	status = SparkGlm52Pp13WorkControlInitializeKvState(
		&state->kv_state,
		state->rank_plan.max_active_sequence_count,
		SPARK_GLM52_PP13_BUILDER_MAX_BLOCKS_PER_SEQUENCE,
		SPARK_GLM52_RESIDENT_DECODE_STAGE_BLOCK_TOKENS,
		physical_block_count,
		state->host_physical_block_indices,
		state->host_lane_physical_block_counts,
		state->host_physical_block_states);
	if (status != SPARK_STATUS_OK)
		return status;
	status = SparkGlm52Pp13BuilderInitializeTables(state);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderInitializeRank0InputBuffers(state);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaZero(
			state->selected_token_indices_by_layer,
			selected_indices_bytes);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaZero(
			state->selected_block_indices_by_layer,
			selected_block_bytes);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaZero(
			state->selected_block_counts_by_layer,
			(uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_COUNT *
			max_active *
			sizeof(uint32_t));
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaZero(
			state->dsa_selection_epoch_by_layer,
			(uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_COUNT *
			max_active *
			sizeof(uint32_t));
	return status;
}

static SparkStatus SparkGlm52Pp13BuilderValidateConfiguration(
	const SparkGlm52Pp13NodeContextBuilderConfiguration *configuration)
{
	if (configuration == 0 ||
		configuration->abi_version !=
			SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_ABI_VERSION ||
		configuration->descriptor_bytes !=
			SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_CONFIGURATION_BYTES ||
		(configuration->flags &
			~SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_CONFIGURATION_KNOWN_FLAGS) != 0u ||
		configuration->rank_plan == 0 ||
		configuration->fp8_pack_root == 0 ||
		configuration->stagepack_root == 0 ||
		configuration->max_active_sequence_count == 0u ||
		configuration->max_active_sequence_count >
			SPARK_GLM52_STAGE_PLAN_MAX_BATCH_BUCKET ||
		configuration->kv_pool_token_capacity == 0u ||
		configuration->kv_pool_token_capacity > SPARK_GLM52_KV_POOL_TOKENS ||
		(configuration->kv_pool_token_capacity %
			SPARK_GLM52_RESIDENT_DECODE_STAGE_BLOCK_TOKENS) != 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if ((configuration->flags &
			(SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_CONFIGURATION_FLAG_DSPARK |
			 SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_CONFIGURATION_FLAG_MTP)) ==
			(SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_CONFIGURATION_FLAG_DSPARK |
			 SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_CONFIGURATION_FLAG_MTP))
		return SPARK_STATUS_INVALID_ARGUMENT;
	if ((configuration->flags &
			SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_CONFIGURATION_FLAG_DSPARK) != 0u &&
		(configuration->dspark_maximum_lane_count == 0u ||
		 configuration->dspark_maximum_lane_count >
			configuration->max_active_sequence_count ||
		 configuration->dspark_maximum_context_token_count == 0u ||
		 configuration->dspark_maximum_context_token_count >
			SPARK_GLM52_KV_CONTEXT_TOKENS))
		return SPARK_STATUS_INVALID_ARGUMENT;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderInitialize(
	const SparkGlm52Pp13NodeContextBuilderConfiguration *configuration,
	void **builder_state)
{
	SparkGlm52Pp13BuilderState *state;
	SparkStatus status;
	if (builder_state == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	*builder_state = 0;
	status = SparkGlm52Pp13BuilderValidateConfiguration(configuration);
	if (status != SPARK_STATUS_OK)
		return status;
	state = (SparkGlm52Pp13BuilderState *)calloc(1u,sizeof(*state));
	if (state == 0)
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	state->configuration = *configuration;
	state->rank_plan = *configuration->rank_plan;
	if (cudaStreamCreate(&state->stream) != cudaSuccess)
	{
		free(state);
		return SPARK_STATUS_IO_ERROR;
	}
	*builder_state = state;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderBuild(
	void *builder_state,
	SparkGlm52Pp13NodeContextBuilderResult *result)
{
	SparkGlm52Pp13BuilderState *state;
	uint32_t layer_offset;
	SparkStatus status;
	state = (SparkGlm52Pp13BuilderState *)builder_state;
	if (state == 0 || result == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (state->built != 0u)
	{
		*result = state->result;
		return SPARK_STATUS_OK;
	}
	if (state->rank_plan.layer_count != SPARK_GLM52_PP13_BUILDER_LAYER_COUNT)
		return SPARK_STATUS_INVALID_ARGUMENT;
	status = SparkGlm52Pp13BuilderInitializeDsparkTopology(state);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderInitializeSharedBuffers(state);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderInitializeFp8ScaledGemm(state);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderInitializeExactPlan(state);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderInitializeMtp(state);
	for (layer_offset = 0u;
		 status == SPARK_STATUS_OK && layer_offset < state->rank_plan.layer_count;
		 ++layer_offset)
		status = SparkGlm52Pp13BuilderBuildLayer(state,layer_offset);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderInitializeDsparkBackend(state);
	if (status != SPARK_STATUS_OK)
		return status;
	memset(&state->slice_context,0,sizeof(state->slice_context));
	state->slice_context.abi_version =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_SLICE_NODE_CONTEXT_ABI_VERSION;
	state->slice_context.descriptor_bytes =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_SLICE_NODE_CONTEXT_DESCRIPTOR_BYTES;
	state->slice_context.first_layer_index = state->rank_plan.first_layer_index;
	state->slice_context.layer_count = state->rank_plan.layer_count;
	state->slice_context.final_token_stage =
		(state->rank_plan.flags & SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_FINAL_STAGE) != 0u ? 1u : 0u;
	state->slice_context.layer_node_contexts = state->layer_pointers;
	state->slice_context.stage_slice_plan = &state->stage_slice_plan;
	memset(&state->result,0,sizeof(state->result));
	state->result.abi_version = SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_ABI_VERSION;
	state->result.descriptor_bytes =
		SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_RESULT_BYTES;
	state->result.rank_index = state->rank_plan.rank_index;
	state->result.first_layer_index = state->rank_plan.first_layer_index;
	state->result.layer_count = state->rank_plan.layer_count;
	state->result.hidden_dimension = state->rank_plan.hidden_dimension;
	state->result.vocabulary_size =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_OUTPUT_VOCAB_COUNT;
	state->result.node_context = &state->slice_context;
	state->result.embedding_weight_bf16 = state->embedding_weight;
	state->result.private_state = state;
	state->built = 1u;
	*result = state->result;
	return SPARK_STATUS_OK;
}

static void SparkGlm52Pp13BuilderDestroyResult(
	void *builder_state,
	SparkGlm52Pp13NodeContextBuilderResult *result)
{
	(void)builder_state;
	if (result != 0)
		memset(result,0,sizeof(*result));
}

static void SparkGlm52Pp13BuilderDestroy(void *builder_state)
{
	SparkGlm52Pp13BuilderState *state;
	uint32_t index;
	state = (SparkGlm52Pp13BuilderState *)builder_state;
	if (state == 0)
		return;
	for (index = 0u; index < SPARK_GLM52_PP13_BUILDER_LAYER_COUNT; ++index)
	{
		if (state->layers[index].linear_binding != 0)
			SparkGlm52ResidentDecodeStageLinearPlanResidentBindingDestroy(
				state->layers[index].linear_binding);
		if (state->layers[index].fp8_moe_ready != 0u)
			SparkGlm52ResidentDecodeStageFp8MoeResidentBindingDestroy(
				&state->layers[index].fp8_moe_binding);
	}
	if (state->mtp_layer.linear_binding != 0)
		SparkGlm52ResidentDecodeStageLinearPlanResidentBindingDestroy(
			state->mtp_layer.linear_binding);
	if (state->mtp_layer.fp8_moe_ready != 0u)
		SparkGlm52ResidentDecodeStageFp8MoeResidentBindingDestroy(
			&state->mtp_layer.fp8_moe_binding);
	for (index = 0u; index < state->allocation_count; ++index)
	{
		if (state->allocation_is_host_mapped[index] != 0u)
			cudaFreeHost(state->allocations[index]);
		else
			cudaFree(state->allocations[index]);
	}
	if (state->dspark_backend_ready != 0u)
		SparkGlm52DsparkDraftBackendTeardown(&state->dspark_backend);
	if (state->mtp_cublas_handle != 0)
		cublasDestroy(state->mtp_cublas_handle);
	if (state->stream != 0)
		cudaStreamDestroy(state->stream);
	if (state->query_stream != 0)
		cudaStreamDestroy(state->query_stream);
	if (state->kv_stream != 0)
		cudaStreamDestroy(state->kv_stream);
	if (state->branch_ready_event != 0)
		cudaEventDestroy(state->branch_ready_event);
	if (state->query_event != 0)
		cudaEventDestroy(state->query_event);
	if (state->kv_event != 0)
		cudaEventDestroy(state->kv_event);
	free(state->host_physical_block_indices);
	free(state->host_lane_physical_block_counts);
	free(state->host_physical_block_states);
	free(state->host_prefill_lane_offsets);
	free(state->host_prefill_lane_counts);
	free(state->host_decode_positions);
	free(state->host_decode_token_ids);
	free(state->host_decode_result_token_ids);
	free(state->host_mtp_draft_budgets);
	free(state->host_mtp_committed_token_ids);
	free(state);
}

static SparkStatus SparkGlm52Pp13BuilderAttachDriver(
	void *builder_state,
	const SparkModelDriverInterface *driver_interface,
	void *driver_instance,
	const SparkModelDriverProgramDescriptor *program,
	SparkHiddenTransportSession *output_transport_session)
{
	SparkGlm52Pp13BuilderState *state;
	SparkGlm52ResidentDecodeStageProductionRunnerConfiguration configuration;
	SparkStatus status;
	state = (SparkGlm52Pp13BuilderState *)builder_state;
	if (state == 0 || driver_interface == 0 || driver_instance == 0 ||
		program == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	state->driver_interface = driver_interface;
	state->driver_instance = driver_instance;
	state->program = program;
	state->output_transport_session = output_transport_session;
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_ABI_VERSION;
	configuration.descriptor_bytes =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_CONFIGURATION_BYTES;
	configuration.flags =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_FLAG_REQUIRE_ADMISSION;
	if ((state->rank_plan.flags & SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_HAS_PREVIOUS) != 0u)
		configuration.flags |=
			SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_FLAG_REQUIRE_INPUT_TRANSPORT;
	if ((state->rank_plan.flags & SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_HAS_NEXT) != 0u)
		configuration.flags |=
			SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_FLAG_REQUIRE_OUTPUT_TRANSPORT;
	configuration.driver_interface = driver_interface;
	configuration.driver_instance = driver_instance;
	configuration.program = program;
	configuration.execution_stream = (void *)state->stream;
	status = SparkGlm52ResidentDecodeStageProductionRunnerInitialize(
		&state->runner,
		&configuration);
	if (status == SPARK_STATUS_OK)
		state->runner_ready = 1u;
	return status;
}

static SparkStatus SparkGlm52Pp13BuilderPrepareDeviceKvView(
	SparkGlm52Pp13BuilderState *state,
	const SparkGlm52KvBlockTableView *source_view)
{
	const uint32_t *source_blocks;
	const uint32_t *source_counts;
	uint32_t lane_index;
	uint32_t block_index;
	uint32_t copy_count;
	uint64_t destination_base;
	uint64_t source_base;
	SparkStatus status;
	if (state == 0 || source_view == 0 ||
		source_view->abi_version != SPARK_GLM52_KV_CACHE_ABI_VERSION ||
		source_view->descriptor_bytes !=
			SPARK_GLM52_KV_BLOCK_TABLE_VIEW_DESCRIPTOR_BYTES ||
		source_view->lane_count == 0u ||
		source_view->lane_count > state->rank_plan.max_active_sequence_count ||
		source_view->lane_stride == 0u ||
		source_view->block_token_count == 0u ||
		source_view->physical_block_indices == 0 ||
		source_view->lane_physical_block_counts == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	source_blocks = source_view->host_physical_block_indices != 0 ?
		source_view->host_physical_block_indices :
		source_view->physical_block_indices;
	source_counts = source_view->lane_physical_block_counts;
	memset(
		state->host_physical_block_indices,
		0xff,
		(size_t)(source_view->lane_count *
			SPARK_GLM52_PP13_BUILDER_MAX_BLOCKS_PER_SEQUENCE *
			sizeof(uint32_t)));
	memset(
		state->host_lane_physical_block_counts,
		0,
		(size_t)(source_view->lane_count * sizeof(uint32_t)));
	for (lane_index = 0u; lane_index < source_view->lane_count; ++lane_index)
	{
		copy_count = source_counts[lane_index];
		if (copy_count > SPARK_GLM52_PP13_BUILDER_MAX_BLOCKS_PER_SEQUENCE)
			return SPARK_STATUS_CAPACITY_EXCEEDED;
		state->host_lane_physical_block_counts[lane_index] = copy_count;
		destination_base =
			(uint64_t)lane_index *
			(uint64_t)SPARK_GLM52_PP13_BUILDER_MAX_BLOCKS_PER_SEQUENCE;
		source_base =
			(uint64_t)lane_index * (uint64_t)source_view->lane_stride;
		for (block_index = 0u; block_index < copy_count; ++block_index)
			state->host_physical_block_indices[destination_base + block_index] =
				source_blocks[source_base + block_index];
	}
	status = SparkGlm52Pp13BuilderCudaStatus(cudaMemcpyAsync(
		state->device_physical_block_indices,
		state->host_physical_block_indices,
		(size_t)(source_view->lane_count *
			SPARK_GLM52_PP13_BUILDER_MAX_BLOCKS_PER_SEQUENCE *
			sizeof(uint32_t)),
		cudaMemcpyHostToDevice,
		state->stream));
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaStatus(cudaMemcpyAsync(
			state->device_lane_physical_block_counts,
			state->host_lane_physical_block_counts,
			(size_t)(source_view->lane_count * sizeof(uint32_t)),
			cudaMemcpyHostToDevice,
			state->stream));
	if (status != SPARK_STATUS_OK)
		return status;
	state->device_kv_view = *source_view;
	state->device_kv_view.lane_stride =
		SPARK_GLM52_PP13_BUILDER_MAX_BLOCKS_PER_SEQUENCE;
	state->device_kv_view.lane_capacity =
		SPARK_GLM52_PP13_BUILDER_MAX_BLOCKS_PER_SEQUENCE;
	state->device_kv_view.physical_block_indices =
		state->device_physical_block_indices;
	state->device_kv_view.lane_physical_block_counts =
		state->device_lane_physical_block_counts;
	state->device_kv_view.host_physical_block_indices =
		state->host_physical_block_indices;
	state->device_kv_view.host_lane_physical_block_counts =
		state->host_lane_physical_block_counts;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderUploadMtpBudget(
	SparkGlm52Pp13BuilderState *state,
	uint32_t active_sequence_count,
	uint32_t mtp_draft_token_count)
{
	uint32_t lane_index;
	if (state == 0 ||
		active_sequence_count == 0u ||
		active_sequence_count > state->rank_plan.max_active_sequence_count ||
		mtp_draft_token_count >
			SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT ||
		state->host_mtp_draft_budgets == 0 ||
		state->layers[0].mtp_draft_token_budgets == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	for (lane_index = 0u; lane_index < active_sequence_count; ++lane_index)
		state->host_mtp_draft_budgets[lane_index] = mtp_draft_token_count;
	return SparkGlm52Pp13BuilderCudaStatus(cudaMemcpyAsync(
		state->layers[0].mtp_draft_token_budgets,
		state->host_mtp_draft_budgets,
		(size_t)(active_sequence_count * sizeof(uint32_t)),
		cudaMemcpyHostToDevice,
		state->stream));
}

static SparkStatus SparkGlm52Pp13BuilderLaunchDecodeMetadataForAllLayers(
	SparkGlm52Pp13BuilderState *state,
	uint32_t active_sequence_count)
{
	uint32_t block_count;
	uint32_t sparse_block_count;
	uint32_t layer_offset;
	uint64_t block_table_bytes;
	SparkStatus status;

	if (state == 0 || active_sequence_count == 0u ||
		active_sequence_count > state->rank_plan.max_active_sequence_count)
		return SPARK_STATUS_INVALID_ARGUMENT;
	block_count =
		(active_sequence_count + SPARK_GLM52_PP13_BUILDER_THREADS - 1u) /
		SPARK_GLM52_PP13_BUILDER_THREADS;
	sparse_block_count = (
		(active_sequence_count *
		 SPARK_GLM52_RESIDENT_DECODE_STAGE_SELECTED_TOKEN_COUNT) +
		SPARK_GLM52_PP13_BUILDER_THREADS - 1u) /
		SPARK_GLM52_PP13_BUILDER_THREADS;
	block_table_bytes =
		(uint64_t)active_sequence_count *
		(uint64_t)SPARK_GLM52_PP13_BUILDER_MAX_BLOCKS_PER_SEQUENCE *
		(uint64_t)sizeof(uint32_t);
	for (layer_offset = 0u;
		 layer_offset < state->rank_plan.layer_count;
		 ++layer_offset)
	{
		status = SparkGlm52Pp13BuilderCudaStatus(cudaMemcpyAsync(
			state->layers[layer_offset].block_table,
			state->device_physical_block_indices,
			(size_t)block_table_bytes,
			cudaMemcpyDeviceToDevice,
			state->stream));
		if (status != SPARK_STATUS_OK)
			return status;
		SparkGlm52Pp13BuilderBuildDecodeMetadataKernel<<<
			block_count,
			SPARK_GLM52_PP13_BUILDER_THREADS,
			0,
			state->stream>>>(
				(const uint32_t *)state->device_decode_positions,
				state->device_kv_view.physical_block_indices,
				state->device_kv_view.lane_physical_block_counts,
				state->device_kv_view.lane_stride,
				state->device_kv_view.block_token_count,
				active_sequence_count,
				(uint32_t *)state->layers[layer_offset].positions,
				(uint32_t *)state->layers[layer_offset].slot_mapping,
				(uint32_t *)state->layers[layer_offset].context_lengths,
				(uint32_t *)state->layers[layer_offset].first_block_token_offsets);
		status = SparkGlm52Pp13BuilderCudaStatus(cudaGetLastError());
		if (status != SPARK_STATUS_OK)
			return status;
		SparkGlm52Pp13BuilderBuildDecodeSparseTokenIndicesKernel<<<
			sparse_block_count,
			SPARK_GLM52_PP13_BUILDER_THREADS,
			0,
			state->stream>>>(
				(const uint32_t *)state->device_decode_positions,
				active_sequence_count,
				(uint32_t *)state->layers[layer_offset].sparse_token_indices);
		status = SparkGlm52Pp13BuilderCudaStatus(cudaGetLastError());
		if (status != SPARK_STATUS_OK)
			return status;
	}
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderUploadWorkDecodePositions(
	SparkGlm52Pp13BuilderState *state,
	const SparkGlm52Pp13WorkControlPacket *work_packet)
{
	uint32_t lane_index;

	if (state == 0 || work_packet == 0 ||
		work_packet->active_sequence_count == 0u ||
		work_packet->active_sequence_count >
			state->rank_plan.max_active_sequence_count ||
		state->host_decode_positions == 0 ||
		state->device_decode_positions == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	for (lane_index = 0u;
		 lane_index < work_packet->active_sequence_count;
		 ++lane_index)
	{
		state->host_decode_positions[lane_index] =
			(uint32_t)work_packet->lanes[lane_index].sequence_position;
	}
	return SparkGlm52Pp13BuilderCudaStatus(cudaMemcpyAsync(
		state->device_decode_positions,
		state->host_decode_positions,
		(size_t)(work_packet->active_sequence_count * sizeof(uint32_t)),
		cudaMemcpyHostToDevice,
		state->stream));
}

static SparkStatus SparkGlm52Pp13BuilderLaunchSerialPrefillMetadata(
	SparkGlm52Pp13BuilderState *state,
	uint32_t active_sequence_count,
	uint32_t absolute_position)
{
	uint32_t block_count;

	if (state == 0 || active_sequence_count == 0u ||
		active_sequence_count > state->rank_plan.max_active_sequence_count ||
		state->device_prefill_positions == 0 ||
		state->device_prefill_slot_mapping == 0 ||
		state->device_prefill_context_lengths == 0 ||
		state->device_prefill_first_block_token_offsets == 0 ||
		state->device_prefill_token_counts == 0 ||
		state->device_kv_view.physical_block_indices == 0 ||
		state->device_kv_view.lane_physical_block_counts == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	block_count =
		(active_sequence_count + SPARK_GLM52_PP13_BUILDER_THREADS - 1u) /
		SPARK_GLM52_PP13_BUILDER_THREADS;
	SparkGlm52Pp13BuilderBuildSerialPrefillMetadataKernel<<<
		block_count,
		SPARK_GLM52_PP13_BUILDER_THREADS,
		0,
		state->stream>>>(
		absolute_position,
		state->device_kv_view.physical_block_indices,
		state->device_kv_view.lane_physical_block_counts,
		state->device_kv_view.lane_stride,
		state->device_kv_view.block_token_count,
		active_sequence_count,
		(uint32_t *)state->device_prefill_positions,
		(uint32_t *)state->device_prefill_slot_mapping,
		(uint32_t *)state->device_prefill_context_lengths,
		(uint32_t *)state->device_prefill_first_block_token_offsets,
		(uint32_t *)state->device_prefill_token_counts);
	return SparkGlm52Pp13BuilderCudaStatus(cudaGetLastError());
}

static void SparkGlm52Pp13BuilderBuildPacket(
	const SparkGlm52Pp13BuilderState *state,
	const SparkGlm52Pp13WorkControlPacket *work_packet,
	const void *hidden,
	void *sideband,
	uint32_t needs_sideband,
	SparkHiddenTransportPacket *packet)
{
	memset(packet,0,sizeof(*packet));
	packet->abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
	packet->descriptor_bytes = SPARK_HIDDEN_TRANSPORT_PACKET_BYTES;
	packet->flags =
		SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_BF16 |
		SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_DEVICE_POINTER;
	packet->active_sequence_count = work_packet->active_sequence_count;
	packet->hidden_dimension = SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION;
	packet->bytes_per_sequence =
		SPARK_GLM52_PP13_RUNTIME_BF16_HIDDEN_BYTES_PER_SEQUENCE;
	packet->sequence_id = work_packet->sequence_id;
	packet->token_index = work_packet->sequence_position;
	packet->hidden_bf16 = hidden;
	packet->cuda_stream = (void *)state->stream;
	if (needs_sideband != 0u)
	{
		packet->flags |= SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_SIDEBAND_PAYLOAD;
		packet->sideband_payload = sideband;
		packet->sideband_kind =
			SPARK_GLM52_RESIDENT_DECODE_STAGE_TRANSPORT_SIDEBAND_INDEXSHARE_SELECTED_TOKENS;
		packet->sideband_bytes_per_sequence =
			SPARK_GLM52_RESIDENT_DECODE_STAGE_SELECTED_INDEX_BYTES;
	}
}

static SparkStatus SparkGlm52Pp13BuilderArmDsparkSideband(
	const SparkGlm52Pp13BuilderState *state,
	const SparkGlm52Pp13WorkControlPacket *work_packet,
	uint32_t export_stage_index,
	void *sideband_payload,
	SparkHiddenTransportPacket *packet)
{
	if (!SparkGlm52Pp13BuilderWorkCapturesDspark(work_packet))
		return SPARK_STATUS_OK;
	return SparkGlm52ProductionTopologyArmHopSidebandPacket(
		&state->production_topology,
		export_stage_index,
		sideband_payload,
		packet);
}

static void SparkGlm52Pp13BuilderApplyDsparkDispatch(
	const SparkGlm52Pp13BuilderState *state,
	const SparkGlm52Pp13WorkControlPacket *work_packet,
	SparkGlm52ResidentDecodeStageProductionRunnerDispatch *dispatch)
{
	if (!SparkGlm52Pp13BuilderWorkCapturesDspark(work_packet))
		return;
	dispatch->dspark_hidden_tap_plan = &state->dspark_tap_plan;
	dispatch->dspark_hidden_tap_lane_stride_bytes =
		SPARK_GLM52_PP13_RUNTIME_BF16_HIDDEN_BYTES_PER_SEQUENCE;
	if (state->dspark_backend_ready != 0u)
	{
		dispatch->dspark_hidden_tap_outputs_bf16 =
			(void *const *)state->dspark_tap_outputs_bf16;
		dispatch->dspark_hidden_tap_lane_stride_bytes =
			state->dspark_tap_lane_stride_bytes;
	}
}

static uint32_t SparkGlm52Pp13BuilderNeedsInputSideband(
	const SparkGlm52Pp13BuilderState *state)
{
	uint32_t first_layer;
	uint32_t layer_offset;
	first_layer = state->rank_plan.first_layer_index;
	for (layer_offset = 0u; layer_offset < state->rank_plan.layer_count; ++layer_offset)
	{
		if (state->layers[layer_offset].node.sparse_index_mode ==
				SPARK_GLM52_RESIDENT_DECODE_STAGE_SPARSE_INDEX_DSA_INDEXSHARE_SHARED &&
			state->layers[layer_offset].node.dsa_indexshare_source_layer_index <
				first_layer)
			return 1u;
	}
	return 0u;
}

static uint32_t SparkGlm52Pp13BuilderNeedsOutputSideband(
	const SparkGlm52Pp13BuilderState *state)
{
	uint32_t stage_end;
	uint32_t layer_offset;
	stage_end = state->rank_plan.first_layer_index + state->rank_plan.layer_count;
	for (layer_offset = 0u; layer_offset < state->rank_plan.layer_count; ++layer_offset)
	{
		if (state->layers[layer_offset].node.sparse_index_mode ==
				SPARK_GLM52_RESIDENT_DECODE_STAGE_SPARSE_INDEX_DSA_INDEXSHARE_FULL &&
			state->layers[layer_offset].node.dsa_indexshare_group_end_layer_exclusive >
				stage_end)
			return 1u;
	}
	return 0u;
}

static SparkStatus SparkGlm52Pp13BuilderSetDsaCandidateCount(
	SparkGlm52Pp13BuilderState *state,
	uint32_t context_token_count)
{
	uint32_t candidate_count;
	uint32_t layer_offset;
	if (state == 0 || context_token_count == 0u ||
		context_token_count > SPARK_GLM52_KV_CONTEXT_TOKENS)
		return SPARK_STATUS_INVALID_ARGUMENT;
	candidate_count = SparkGlm52Pp13RuntimeDsaCandidateBucket(
		context_token_count);
	if (candidate_count == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	for (layer_offset = 0u; layer_offset < state->rank_plan.layer_count;
		 ++layer_offset)
	{
		if (candidate_count >
			state->layers[layer_offset].node.dsa_candidate_capacity)
			return SPARK_STATUS_CAPACITY_EXCEEDED;
		state->layers[layer_offset].slot.dsa_candidate_count = candidate_count;
	}
	return SPARK_STATUS_OK;
}

static void SparkGlm52Pp13BuilderCaptureCompletion(
	void *completion_context,
	const SparkModelDriverCompletion *completion)
{
	SparkGlm52Pp13BuilderState *state;

	state = (SparkGlm52Pp13BuilderState *)completion_context;
	if (state == 0 || completion == 0)
		return;
	state->captured_completion = *completion;
	state->captured_completion_valid = 1u;
}

static SparkStatus SparkGlm52Pp13BuilderFinalizePlainDecodeBatch(
	SparkGlm52Pp13BuilderState *state,
	const SparkGlm52Pp13WorkControlPacket *work_packet,
	SparkModelDriverCompletionFunction completion_function,
	void *completion_context)
{
	SparkGlm52Pp13BuilderLayer *final_layer;
	SparkModelDriverCompletion completion;
	uint32_t lane_index;
	SparkStatus status;

	if (state == 0 || work_packet == 0 || completion_function == 0 ||
		state->captured_completion_valid == 0u ||
		state->host_decode_result_token_ids == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	final_layer = &state->layers[state->rank_plan.layer_count - 1u];
	status = SparkGlm52Pp13BuilderCudaStatus(cudaMemcpy(
		state->host_decode_result_token_ids,
		final_layer->restricted_selected_token_ids,
		(size_t)(work_packet->active_sequence_count * sizeof(uint32_t)),
		cudaMemcpyDeviceToHost));
	if (status != SPARK_STATUS_OK)
		return status;
	for (lane_index = 0u;
		 lane_index < work_packet->active_sequence_count;
		 ++lane_index)
	{
		completion = state->captured_completion;
		completion.request_id = work_packet->lanes[lane_index].request_id;
		completion.sequence_id = work_packet->lanes[lane_index].sequence_id;
		completion.sequence_position =
			work_packet->lanes[lane_index].sequence_position;
		completion.accepted_token_count = 0u;
		completion.token_count = 1u;
		memset(completion.token_ids,0,sizeof(completion.token_ids));
		completion.token_ids[0u] = state->host_decode_result_token_ids[lane_index];
		completion_function(completion_context,&completion);
	}
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderStageDsparkLane(
	SparkGlm52Pp13BuilderState *state,
	const SparkGlm52Pp13WorkControlPacket *work_packet,
	uint32_t last_token_id)
{
	uint64_t sequence_position;

	if (work_packet->sequence_position == UINT64_MAX)
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	sequence_position = work_packet->sequence_position + 1u;
	return SparkGlm52DsparkDraftBackendStageLane(
		&state->dspark_backend,
		0u,
		work_packet->sequence_id,
		sequence_position,
		last_token_id,
		sequence_position);
}

static void SparkGlm52Pp13BuilderTraceDsparkDraft(
	const SparkGlm52DsparkDraftRequest *request,
	const SparkGlm52DsparkDraftResult *result)
{
	uint32_t token_index;

	if (getenv("SPARKPIPE_DSPARK_TRACE") == 0 || request == 0 || result == 0)
		return;
	fprintf(stderr,"dspark_trace draft request=%llu sequence=%llu position=%llu count=%u",
		(unsigned long long)request->request_id,
		(unsigned long long)request->sequence_id,
		(unsigned long long)request->sequence_position,
		result->token_count);
	for (token_index = 0u; token_index < result->token_count; ++token_index)
		fprintf(stderr," token%u=%u confidence%u=%u",token_index,
			result->token_ids[token_index],token_index,
			result->confidence_milli[token_index]);
	fputc('\n',stderr);
}

static void SparkGlm52Pp13BuilderTraceDsparkVerifyStep(
	const SparkGlm52Pp13BuilderState *state,
	const SparkGlm52Pp13WorkControlPacket *work_packet,
	uint32_t target_token_id)
{
	uint32_t token_index;
	uint32_t draft_token_id;

	if (getenv("SPARKPIPE_DSPARK_TRACE") == 0 || state == 0 || work_packet == 0)
		return;
	token_index = work_packet->speculative_token_index;
	draft_token_id = token_index < state->speculative_verify_draft_count ?
		state->speculative_verify_draft_token_ids[token_index] : UINT32_MAX;
	fprintf(stderr,"dspark_trace verify request=%llu sequence=%llu base=%llu index=%u draft=%u target=%u mismatch=%u accepted=%u\n",
		(unsigned long long)work_packet->request_id,
		(unsigned long long)work_packet->sequence_id,
		(unsigned long long)state->speculative_verify_base_position,
		token_index,draft_token_id,target_token_id,
		state->speculative_verify_mismatch,
		state->speculative_verify_accepted_count);
}

static void SparkGlm52Pp13BuilderTraceDsparkVerifierVector(
	const SparkGlm52Pp13BuilderState *state)
{
	uint32_t token_index;

	if (getenv("SPARKPIPE_DSPARK_TRACE") == 0 || state == 0)
		return;
	fprintf(stderr,"dspark_trace verifier_vector request=%llu sequence=%llu base=%llu accepted=%u count=%u",
		(unsigned long long)state->speculative_verify_request_id,
		(unsigned long long)state->speculative_verify_sequence_id,
		(unsigned long long)state->speculative_verify_base_position,
		state->speculative_verify_accepted_count,
		state->captured_completion.token_count);
	for (token_index = 0u;
		 token_index < state->captured_completion.token_count;
		 ++token_index)
		fprintf(stderr," token%u=%u",token_index,
			state->captured_completion.token_ids[token_index]);
	fputc('\n',stderr);
}

static SparkStatus SparkGlm52Pp13BuilderGenerateDsparkDraft(
	SparkGlm52Pp13BuilderState *state,
	const SparkGlm52Pp13WorkControlPacket *work_packet)
{
	SparkGlm52DsparkDraftRequest request;
	const SparkGlm52DsparkDraftBackendLaneState *lane_state;
	SparkStatus status;

	lane_state = &state->dspark_backend.lane_states[0u];
	memset(&request,0,sizeof(request));
	request.abi_version = SPARK_GLM52_DSPARK_ABI_VERSION;
	request.descriptor_bytes =
		SPARK_GLM52_DSPARK_DRAFT_REQUEST_DESCRIPTOR_BYTES;
	request.requested_token_count =
		SPARK_GLM52_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT;
	request.priority = work_packet->priority;
	request.request_id = work_packet->request_id;
	request.sequence_id = work_packet->sequence_id;
	request.sequence_position = lane_state->sequence_position;
	request.tap_generation = lane_state->tap_generation;
	status = SparkGlm52DsparkDraftBackendDraft(
		&state->dspark_backend,
		&request,
		&state->dspark_ready_draft);
	if (status == SPARK_STATUS_OK)
	{
		state->dspark_ready_draft_valid = 1u;
		SparkGlm52Pp13BuilderTraceDsparkDraft(
			&request,
			&state->dspark_ready_draft);
	}
	return status;
}

static SparkStatus SparkGlm52Pp13BuilderBeginSpeculativeVerify(
	SparkGlm52Pp13BuilderState *state,
	const SparkGlm52Pp13WorkControlPacket *work_packet)
{
	if (work_packet->speculative_token_index != 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	state->speculative_verify_active = 1u;
	state->speculative_verify_draft_count =
		work_packet->speculative_token_count;
	state->speculative_verify_mismatch = 0u;
	state->speculative_verify_accepted_count = 0u;
	state->speculative_verify_request_id = work_packet->request_id;
	state->speculative_verify_sequence_id = work_packet->sequence_id;
	state->speculative_verify_base_position = work_packet->sequence_position;
	memcpy(
		state->speculative_verify_draft_token_ids,
		work_packet->speculative_draft_token_ids,
		state->speculative_verify_draft_count *
			sizeof(state->speculative_verify_draft_token_ids[0u]));
	memset(
		state->speculative_verify_target_token_ids,
		0,
		sizeof(state->speculative_verify_target_token_ids));
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderValidateSpeculativeVerifyStep(
	const SparkGlm52Pp13BuilderState *state,
	const SparkGlm52Pp13WorkControlPacket *work_packet)
{
	uint64_t expected_position;

	if (state->speculative_verify_active == 0u ||
		state->speculative_verify_request_id != work_packet->request_id ||
		state->speculative_verify_sequence_id != work_packet->sequence_id ||
		state->speculative_verify_draft_count !=
			work_packet->speculative_token_count ||
		memcmp(state->speculative_verify_draft_token_ids,
			work_packet->speculative_draft_token_ids,
			state->speculative_verify_draft_count *
				sizeof(state->speculative_verify_draft_token_ids[0u])) != 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	expected_position = state->speculative_verify_base_position +
		work_packet->speculative_token_index;
	if (expected_position < state->speculative_verify_base_position ||
		work_packet->sequence_position != expected_position)
		return SPARK_STATUS_INVALID_ARGUMENT;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderFinalizeSpeculativeVerify(
	SparkGlm52Pp13BuilderState *state,
	const SparkGlm52Pp13WorkControlPacket *work_packet)
{
	uint32_t token_index;
	uint32_t target_token_id;
	SparkStatus status;

	if (work_packet->speculative_token_index == 0u)
	{
		status = SparkGlm52Pp13BuilderBeginSpeculativeVerify(state,work_packet);
		if (status != SPARK_STATUS_OK)
			return status;
	}
	status = SparkGlm52Pp13BuilderValidateSpeculativeVerifyStep(state,work_packet);
	if (status != SPARK_STATUS_OK ||
		state->captured_completion.token_count != 1u ||
		(state->captured_completion.completion_flags &
			SPARK_MODEL_DRIVER_COMPLETION_FLAG_TOKEN_IDS) == 0u)
		return status != SPARK_STATUS_OK ? status : SPARK_STATUS_VALIDATION_FAILED;
	target_token_id = state->captured_completion.token_ids[0u];
	state->speculative_verify_target_token_ids[
		work_packet->speculative_token_index] = target_token_id;
	if (state->speculative_verify_mismatch == 0u)
	{
		if (SparkGlm52Pp13BuilderWorkIsDsparkVerify(work_packet) != 0u)
		{
			status = SparkGlm52Pp13BuilderStageDsparkLane(
				state,
				work_packet,
				target_token_id);
			if (status != SPARK_STATUS_OK)
				return status;
		}
		if (work_packet->speculative_token_index <
				state->speculative_verify_draft_count &&
			target_token_id != state->speculative_verify_draft_token_ids[
				work_packet->speculative_token_index])
		{
			state->speculative_verify_mismatch = 1u;
			state->speculative_verify_accepted_count =
				work_packet->speculative_token_index;
		}
		else if (work_packet->speculative_token_index ==
			state->speculative_verify_draft_count)
		{
			state->speculative_verify_accepted_count =
				state->speculative_verify_draft_count;
		}
	}
	if (SparkGlm52Pp13BuilderWorkIsDsparkVerify(work_packet) != 0u)
		SparkGlm52Pp13BuilderTraceDsparkVerifyStep(
			state,
			work_packet,
			target_token_id);
	if (work_packet->speculative_token_index <
		state->speculative_verify_draft_count)
	{
		state->captured_completion.completion_flags &=
			~SPARK_MODEL_DRIVER_COMPLETION_FLAG_TOKEN_IDS;
		state->captured_completion.token_count = 0u;
		memset(state->captured_completion.token_ids,0,
			sizeof(state->captured_completion.token_ids));
		return SPARK_STATUS_OK;
	}
	state->captured_completion.sequence_position =
		state->speculative_verify_base_position;
	state->captured_completion.token_count =
		state->speculative_verify_draft_count + 1u;
	for (token_index = 0u;
		 token_index < state->captured_completion.token_count;
		 ++token_index)
	{
		state->captured_completion.token_ids[token_index] =
			state->speculative_verify_target_token_ids[token_index];
	}
	status = SPARK_STATUS_OK;
	if (SparkGlm52Pp13BuilderWorkIsDsparkVerify(work_packet) != 0u)
	{
		SparkGlm52Pp13BuilderTraceDsparkVerifierVector(state);
		status = SparkGlm52Pp13BuilderGenerateDsparkDraft(state,work_packet);
	}
	state->speculative_verify_active = 0u;
	return status;
}

static SparkStatus SparkGlm52Pp13BuilderFinalizeCapturedCompletion(
	SparkGlm52Pp13BuilderState *state,
	const SparkGlm52Pp13WorkControlPacket *work_packet)
{
	SparkStatus status;

	if (!SparkGlm52Pp13BuilderWorkNeedsCapturedCompletion(work_packet) ||
		!SparkGlm52Pp13BuilderIsFinalRank(state))
		return SPARK_STATUS_OK;
	if (state->captured_completion_valid == 0u ||
		(SparkGlm52Pp13BuilderWorkCapturesDspark(work_packet) != 0u &&
		 state->dspark_backend_ready == 0u))
		return SPARK_STATUS_MODULE_NOT_VALIDATED;
	if (SparkGlm52Pp13BuilderWorkIsSpeculativeVerify(work_packet) != 0u)
		return SparkGlm52Pp13BuilderFinalizeSpeculativeVerify(
			state,
			work_packet);
	if ((work_packet->flags &
			SPARK_GLM52_PP13_WORK_CONTROL_FLAG_PREFILL) != 0u)
	{
		return SparkGlm52Pp13BuilderStageDsparkLane(
			state,
			work_packet,
			work_packet->input_token_id);
	}
	if (state->captured_completion.token_count != 1u ||
		(state->captured_completion.completion_flags &
			SPARK_MODEL_DRIVER_COMPLETION_FLAG_TOKEN_IDS) == 0u)
		return SPARK_STATUS_VALIDATION_FAILED;
	status = SparkGlm52Pp13BuilderStageDsparkLane(
		state,
		work_packet,
		state->captured_completion.token_ids[0u]);
	if (status != SPARK_STATUS_OK)
		return status;
	return SparkGlm52Pp13BuilderGenerateDsparkDraft(state,work_packet);
}

static SparkStatus SparkGlm52Pp13BuilderStoreMtpPreviousTarget(
	SparkGlm52Pp13BuilderState *state,
	const SparkGlm52Pp13WorkControlPacket *work_packet)
{
	SparkGlm52Pp13BuilderLayer *final_layer;
	const void *hidden;
	SparkStatus status;
	if (state == 0 || work_packet == 0 || state->mtp_ready == 0u ||
		!SparkGlm52Pp13BuilderIsFinalRank(state) ||
		state->mtp_previous_target_hidden == 0)
		return SPARK_STATUS_OK;
	final_layer = &state->layers[state->rank_plan.layer_count - 1u];
	if (final_layer->final_norm_weight == 0)
		return SPARK_STATUS_MODULE_NOT_VALIDATED;
	hidden = final_layer->layer_output_hidden;
	state->host_decode_positions[0u] = (uint32_t)work_packet->sequence_position;
	status = SparkGlm52Pp13BuilderCudaStatus(cudaMemcpyAsync(
		state->mtp_base_positions,state->host_decode_positions,
		sizeof(uint32_t),cudaMemcpyHostToDevice,state->stream));
	if (status != SPARK_STATUS_OK)
		return status;
	SparkGlm52Pp13BuilderTargetFinalNormKernel<<<
		work_packet->active_sequence_count,
		SPARK_GLM52_PP13_BUILDER_THREADS,
		0u,
		state->stream>>>(
		(const uint16_t *)hidden,
		(const uint16_t *)final_layer->final_norm_weight,
		(uint16_t *)state->mtp_previous_target_hidden,
		work_packet->active_sequence_count,
		SPARK_GLM52_MODEL_RMS_NORM_EPSILON);
	status = SparkGlm52Pp13BuilderCudaStatus(cudaGetLastError());
	if (status != SPARK_STATUS_OK)
		return status;
	state->mtp_previous_request_id = work_packet->request_id;
	state->mtp_previous_sequence_id = work_packet->sequence_id;
	state->mtp_previous_position = work_packet->sequence_position;
	state->mtp_previous_valid = 1u;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderPrefillMtpPreviousTarget(
	SparkGlm52Pp13BuilderState *state,
	const SparkGlm52Pp13WorkControlPacket *work_packet)
{
	SparkStatus status;
	if (state == 0 || work_packet == 0 || state->mtp_ready == 0u ||
		!SparkGlm52Pp13BuilderIsFinalRank(state) ||
		(work_packet->flags & SPARK_GLM52_PP13_WORK_CONTROL_FLAG_PREFILL) == 0u)
		return SPARK_STATUS_OK;
	if (state->mtp_previous_valid != 0u &&
		state->mtp_previous_request_id == work_packet->request_id &&
		state->mtp_previous_sequence_id == work_packet->sequence_id &&
		state->mtp_previous_position + 1u == work_packet->sequence_position)
	{
		state->host_decode_token_ids[0u] = work_packet->input_token_id;
		state->host_decode_positions[0u] =
			(uint32_t)state->mtp_previous_position;
		status = SparkGlm52Pp13BuilderCudaStatus(cudaMemcpyAsync(
			state->device_decode_token_ids,state->host_decode_token_ids,
			sizeof(uint32_t),cudaMemcpyHostToDevice,state->stream));
		if (status == SPARK_STATUS_OK)
			status = SparkGlm52Pp13BuilderCudaStatus(cudaMemcpyAsync(
				state->mtp_base_positions,state->host_decode_positions,
				sizeof(uint32_t),cudaMemcpyHostToDevice,state->stream));
		if (status == SPARK_STATUS_OK)
			status = SparkGlm52Pp13BuilderLaunchMtpLayer(
				state,(const uint32_t *)state->device_decode_token_ids,
				state->mtp_base_positions,state->mtp_previous_target_hidden,
				0u,work_packet->active_sequence_count,state->stream);
		if (status != SPARK_STATUS_OK)
			return status;
	}
	else
	{
		state->mtp_previous_valid = 0u;
	}
	return SparkGlm52Pp13BuilderStoreMtpPreviousTarget(state,work_packet);
}

static SparkStatus SparkGlm52Pp13BuilderSubmitWork(
	void *builder_state,
	const SparkGlm52Pp13WorkControlPacket *work_packet,
	SparkHiddenTransportSession *input_transport_session,
	SparkHiddenTransportSession *output_transport_session,
	SparkModelDriverCompletionFunction completion_function,
	void *completion_context)
{
	SparkGlm52Pp13BuilderState *state;
	SparkGlm52ResidentDecodeStageProductionRunnerDispatch dispatch;
	SparkStatus status;
	uint64_t kv_entries;
	state = (SparkGlm52Pp13BuilderState *)builder_state;
	if (state == 0 || work_packet == 0 || state->runner_ready == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	status = SparkGlm52Pp13WorkControlValidatePacket(
		work_packet,
		state->rank_plan.max_active_sequence_count,
		SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT);
	if (status != SPARK_STATUS_OK)
		return status;
	status = SparkGlm52Pp13WorkControlBuildHostKvBlockTable(
		work_packet,
		&state->kv_state,
		&state->host_kv_view);
	if (status != SPARK_STATUS_OK)
		return status;
	kv_entries =
		(uint64_t)work_packet->active_sequence_count *
		SPARK_GLM52_PP13_BUILDER_MAX_BLOCKS_PER_SEQUENCE;
	status = SparkGlm52Pp13BuilderCudaStatus(cudaMemcpy(
		state->device_physical_block_indices,
		state->host_physical_block_indices,
		(size_t)(kv_entries * sizeof(uint32_t)),
		cudaMemcpyHostToDevice));
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaStatus(cudaMemcpy(
		state->device_lane_physical_block_counts,
		state->host_lane_physical_block_counts,
		(size_t)(work_packet->active_sequence_count * sizeof(uint32_t)),
			cudaMemcpyHostToDevice));
	if (status != SPARK_STATUS_OK)
		return status;
	status = SparkGlm52Pp13BuilderSetDsaCandidateCount(
		state,
		work_packet->kv_block_table_token_count);
	if (status != SPARK_STATUS_OK)
		return status;
	status = SparkGlm52Pp13BuilderUploadMtpBudget(
		state,
		work_packet->active_sequence_count,
		work_packet->mtp_draft_token_count);
	if (status != SPARK_STATUS_OK)
		return status;
	state->device_kv_view = state->host_kv_view;
	state->device_kv_view.physical_block_indices =
		state->device_physical_block_indices;
	state->device_kv_view.lane_physical_block_counts =
		state->device_lane_physical_block_counts;
	state->device_kv_view.host_physical_block_indices =
		state->host_physical_block_indices;
	state->device_kv_view.host_lane_physical_block_counts =
		state->host_lane_physical_block_counts;
	status = SparkGlm52Pp13BuilderUploadWorkDecodePositions(
		state,
		work_packet);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderLaunchDecodeMetadataForAllLayers(
			state,
			work_packet->active_sequence_count);
	if (status == SPARK_STATUS_OK &&
		(work_packet->flags & SPARK_GLM52_PP13_WORK_CONTROL_FLAG_PREFILL) != 0u)
		status = SparkGlm52Pp13BuilderLaunchSerialPrefillMetadata(
			state,
			work_packet->active_sequence_count,
			(uint32_t)work_packet->sequence_position);
	if (status != SPARK_STATUS_OK)
		return status;
	memset(&dispatch,0,sizeof(dispatch));
	dispatch.abi_version =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_ABI_VERSION;
	dispatch.descriptor_bytes =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_DISPATCH_BYTES;
	dispatch.flags = (work_packet->flags & SPARK_GLM52_PP13_WORK_CONTROL_FLAG_PREFILL) != 0u
		? SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_DISPATCH_FLAG_PREFILL
		: 0u;
	dispatch.priority = work_packet->priority;
	dispatch.request_id = work_packet->request_id;
	dispatch.sequence_id = work_packet->sequence_id;
	dispatch.sequence_position = work_packet->sequence_position;
	dispatch.deadline_time_ns = work_packet->deadline_time_ns;
	dispatch.active_sequence_count = work_packet->active_sequence_count;
	dispatch.new_token_count = work_packet->new_token_count;
	dispatch.pipeline_slot = work_packet->pipeline_slot;
	dispatch.kv_block_table = &state->device_kv_view;
	if ((work_packet->flags & SPARK_GLM52_PP13_WORK_CONTROL_FLAG_MTP_DRAFT) != 0u &&
		work_packet->mtp_draft_token_count != 0u)
		dispatch.mtp_draft_token_budgets =
			(const uint32_t *)state->layers[0].mtp_draft_token_budgets;
	dispatch.hidden_input_transport_session = input_transport_session;
	dispatch.hidden_output_transport_session = output_transport_session;
	SparkGlm52Pp13BuilderApplyDsparkDispatch(state,work_packet,&dispatch);
	SparkGlm52Pp13BuilderBuildPacket(
		state,
		work_packet,
		state->layers[0].input_hidden,
		state->input_sideband,
		SparkGlm52Pp13BuilderNeedsInputSideband(state),
		&dispatch.hidden_input_packet);
	SparkGlm52Pp13BuilderBuildPacket(
		state,
		work_packet,
		state->layers[state->rank_plan.layer_count - 1u].layer_output_hidden,
		state->output_sideband,
		SparkGlm52Pp13BuilderNeedsOutputSideband(state),
		&dispatch.hidden_output_packet);
	if (input_transport_session != 0)
	{
		status = SparkGlm52Pp13BuilderArmDsparkSideband(
			state,
			work_packet,
			state->rank_plan.rank_index - 1u,
			state->input_sideband,
			&dispatch.hidden_input_packet);
		if (status != SPARK_STATUS_OK)
			return status;
	}
	if (output_transport_session != 0)
	{
		status = SparkGlm52Pp13BuilderArmDsparkSideband(
			state,
			work_packet,
			state->rank_plan.rank_index,
			state->output_sideband,
			&dispatch.hidden_output_packet);
		if (status != SPARK_STATUS_OK)
			return status;
	}
	state->captured_completion_valid = 0u;
	state->dspark_ready_draft_valid = 0u;
	state->mtp_use_previous_for_draft =
		SparkGlm52Pp13BuilderIsFinalRank(state) &&
		(work_packet->flags & SPARK_GLM52_PP13_WORK_CONTROL_FLAG_MTP_DRAFT) != 0u &&
		state->mtp_previous_valid != 0u &&
		state->mtp_previous_request_id == work_packet->request_id &&
		state->mtp_previous_sequence_id == work_packet->sequence_id &&
		state->mtp_previous_position + 1u == work_packet->sequence_position;
	if (SparkGlm52Pp13BuilderWorkNeedsCapturedCompletion(work_packet) &&
		SparkGlm52Pp13BuilderIsFinalRank(state))
	{
		dispatch.completion_function = SparkGlm52Pp13BuilderCaptureCompletion;
		dispatch.completion_context = state;
	}
	else
	{
		dispatch.completion_function = completion_function;
		dispatch.completion_context = completion_context;
	}
	status = SparkGlm52ResidentDecodeStageProductionRunnerSubmit(
		&state->runner,
		&dispatch);
	if (status == SPARK_STATUS_OK)
	{
		status = SparkGlm52ResidentDecodeStageProductionRunnerWaitIdle(&state->runner,25000u);
		if (status != SPARK_STATUS_OK)
			fprintf(stderr,"pp13_builder_submit_drain_failed status=%u position=%llu\n",status,(unsigned long long)work_packet->sequence_position);
	}
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderPrefillMtpPreviousTarget(state,work_packet);
	if (status == SPARK_STATUS_OK &&
		SparkGlm52Pp13BuilderWorkIsPlainDecodeBatch(work_packet) != 0u &&
		SparkGlm52Pp13BuilderIsFinalRank(state))
	{
		status = SparkGlm52Pp13BuilderFinalizePlainDecodeBatch(
			state,
			work_packet,
			completion_function,
			completion_context);
	}
	if (status == SPARK_STATUS_OK &&
		SparkGlm52Pp13BuilderWorkIsPlainDecodeBatch(work_packet) == 0u &&
		SparkGlm52Pp13BuilderWorkNeedsCapturedCompletion(work_packet) &&
		SparkGlm52Pp13BuilderIsFinalRank(state))
	{
		status = SparkGlm52Pp13BuilderFinalizeCapturedCompletion(
			state,
			work_packet);
		if (status == SPARK_STATUS_OK && completion_function != 0)
			completion_function(completion_context,&state->captured_completion);
	}
	if (status == SPARK_STATUS_OK)
		(void)SparkGlm52Pp13WorkControlCommitHostKvBlockTable(
			work_packet,
			&state->kv_state);
	else
		(void)SparkGlm52Pp13WorkControlCancelHostKvBlockTable(
			work_packet,
			&state->kv_state);
	return status;
}

static uint64_t SparkGlm52Pp13BuilderProbeFnv64(const uint8_t *data,uint64_t bytes)
{
	uint64_t hash;
	uint64_t offset;
	hash = 0xcbf29ce484222325ull;
	for (offset = 0u; offset < bytes; ++offset)
		hash = (hash ^ (uint64_t)data[offset]) * 0x100000001b3ull;
	return hash;
}

static void SparkGlm52Pp13BuilderMaybeProbeMlaSlots(SparkGlm52Pp13BuilderState *state)
{
	static uint8_t slot_host[
		SPARK_GLM52_RESIDENT_DECODE_STAGE_CACHE_TOKEN_ELEMENTS *
		sizeof(uint16_t)];
	uint32_t slot_index;
	uint64_t slot_bytes;
	const uint8_t *cache_base;
	if (getenv("SPARKPIPE_MLA_SLOT_PROBE") == 0 || state == 0 || state->layers[0].mla_cache == 0)
		return;
	slot_bytes =
		(uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_CACHE_TOKEN_ELEMENTS *
		sizeof(uint16_t);
	cache_base = (const uint8_t *)state->layers[0].mla_cache;
	for (slot_index = 0u; slot_index < 2u; ++slot_index)
	{
		if (cudaMemcpy(slot_host,cache_base + ((uint64_t)slot_index * slot_bytes),(size_t)slot_bytes,cudaMemcpyDeviceToHost) != cudaSuccess)
			return;
		fprintf(stderr,"mla_slot%u=%016llx mla_slot%u_rope_pair0=%016llx\n",slot_index,(unsigned long long)SparkGlm52Pp13BuilderProbeFnv64(slot_host,slot_bytes),slot_index,(unsigned long long)SparkGlm52Pp13BuilderProbeFnv64(slot_host + ((uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION * sizeof(uint16_t)),2u * sizeof(uint16_t)));
	}
}

static uint32_t SparkGlm52Pp13BuilderProbeHostSlot(
	const SparkGlm52Pp13BuilderState *state,
	uint32_t position,
	uint32_t *physical_block_index_out)
{
	uint32_t block_index;
	uint32_t block_token_index;
	uint32_t block_token_count;
	if (physical_block_index_out != 0)
		*physical_block_index_out = SPARK_GLM52_PP13_BUILDER_INVALID_SLOT;
	if (state == 0 || state->host_lane_physical_block_counts == 0 ||
		state->host_physical_block_indices == 0 ||
		state->device_kv_view.block_token_count == 0u)
		return SPARK_GLM52_PP13_BUILDER_INVALID_SLOT;
	block_token_count = state->device_kv_view.block_token_count;
	block_index = position / block_token_count;
	block_token_index = position - (block_index * block_token_count);
	if (block_index >= state->host_lane_physical_block_counts[0u])
		return SPARK_GLM52_PP13_BUILDER_INVALID_SLOT;
	if (physical_block_index_out != 0)
		*physical_block_index_out = state->host_physical_block_indices[block_index];
	return (state->host_physical_block_indices[block_index] * block_token_count) +
		block_token_index;
}

static void SparkGlm52Pp13BuilderMaybeProbePrefillKvState(
	SparkGlm52Pp13BuilderState *state,
	const SparkGlm52PromptPipelinePrefillDispatch *prefill_dispatch)
{
	uint32_t slot0,slot1,block0,block1,lane_blocks;
	uint64_t request_id;
	if (getenv("SPARKPIPE_MLA_SLOT_PROBE") == 0 || state == 0 ||
		prefill_dispatch == 0 || prefill_dispatch->request_dispatch == 0)
		return;
	request_id = prefill_dispatch->request_dispatch->request_ids[0u];
	lane_blocks = state->host_lane_physical_block_counts != 0 ?
		state->host_lane_physical_block_counts[0u] : 0u;
	slot0 = SparkGlm52Pp13BuilderProbeHostSlot(state,0u,&block0);
	slot1 = SparkGlm52Pp13BuilderProbeHostSlot(state,1u,&block1);
	fprintf(stderr,"pp13_kv_probe_entry request=%llu offset=%u count=%u kv_missing=%u kv_inflight=%u kv_resident=%u lane0_blocks=%u pos0_block=%u pos0_slot=%u pos1_block=%u pos1_slot=%u\n",
		(unsigned long long)request_id,
		prefill_dispatch->prompt_token_offset,
		prefill_dispatch->prompt_token_count,
		state->kv_state.missing_block_count,
		state->kv_state.in_flight_block_count,
		state->kv_state.resident_block_count,
		lane_blocks,
		block0,
		slot0,
		block1,
		slot1);
}

static void SparkGlm52Pp13BuilderMaybeProbePrefillTokenSlot(
	SparkGlm52Pp13BuilderState *state,
	uint64_t request_id,
	uint32_t token_offset,
	uint32_t position)
{
	uint32_t slot_mapping;
	if (getenv("SPARKPIPE_MLA_SLOT_PROBE") == 0 || state == 0 ||
		state->device_prefill_slot_mapping == 0)
		return;
	slot_mapping = SPARK_GLM52_PP13_BUILDER_INVALID_SLOT;
	if (cudaMemcpy(&slot_mapping,state->device_prefill_slot_mapping,sizeof(slot_mapping),cudaMemcpyDeviceToHost) != cudaSuccess)
		return;
	fprintf(stderr,"pp13_kv_probe_token request=%llu token_offset=%u position=%u device_slot_mapping=%u\n",
		(unsigned long long)request_id,
		token_offset,
		position,
		slot_mapping);
}

static void SparkGlm52Pp13BuilderMaybeProbeLayer1Sublayers(
	SparkGlm52Pp13BuilderState *state,
	uint64_t request_id,
	uint32_t token_offset,
	uint32_t position)
{
	uint64_t probe_slots[SPARK_GLM52_PP13_BUILDER_PROBE_HASH_SLOT_COUNT];
	if (getenv("SPARKPIPE_FP8_AMAX_PROBE") == 0 || state == 0 ||
		state->device_probe_hash_slots == 0)
		return;
	if (cudaMemcpy(probe_slots,state->device_probe_hash_slots,
			sizeof(probe_slots),cudaMemcpyDeviceToHost) != cudaSuccess)
		return;
	fprintf(stderr,
		"fp8_layer1_attention_probe request=%llu token_offset=%u position=%u weight=%016llx scale=%016llx input=%016llx attention_norm=%016llx kv_latent=%016llx raw_kv_b=%016llx fp8_value_cache=%016llx attention_value=%016llx attention_projection=%016llx post_attention=%016llx\n",
		(unsigned long long)request_id,
		token_offset,
		position,
		(unsigned long long)probe_slots[16],
		(unsigned long long)probe_slots[17],
		(unsigned long long)probe_slots[8],
		(unsigned long long)probe_slots[9],
		(unsigned long long)probe_slots[10],
		(unsigned long long)probe_slots[11],
		(unsigned long long)probe_slots[12],
		(unsigned long long)probe_slots[13],
		(unsigned long long)probe_slots[14],
		(unsigned long long)probe_slots[15]);
}

static SparkStatus SparkGlm52Pp13BuilderPrefill(
	void *builder_state,
	const SparkGlm52PromptPipelinePrefillDispatch *prefill_dispatch,
	SparkGlm52Pp13NodeContextBuilderIdlePumpFunction idle_pump_function,
	void *idle_pump_context)
{
	SparkGlm52Pp13BuilderState *state;
	SparkGlm52ResidentDecodeStageProductionRunnerDispatch dispatch;
	SparkGlm52Pp13WorkControlPacket work_packet;
	uint32_t token_offset;
	uint32_t submit_retry;
	uint32_t position;
	uint32_t token_id;
	uint32_t block_count;
	uint64_t word_count;
	const char *debug_enabled;
	SparkStatus status;
	state = (SparkGlm52Pp13BuilderState *)builder_state;
	if (state == 0 || prefill_dispatch == 0 ||
		state->runner_ready == 0u ||
		state->embedding_weight == 0 ||
		(state->rank_plan.flags & SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_HAS_PREVIOUS) != 0u ||
		prefill_dispatch->request_dispatch == 0 ||
		prefill_dispatch->prefill_view == 0 ||
		prefill_dispatch->kv_block_table_view == 0 ||
		prefill_dispatch->lane_count != 1u ||
		prefill_dispatch->active_sequence_count != 1u ||
		prefill_dispatch->prefill_view->lane_count != 1u ||
		prefill_dispatch->host_token_ids == 0 ||
		prefill_dispatch->prompt_token_count == 0u ||
		prefill_dispatch->prompt_token_count >
			SPARK_GLM52_PP13_BUILDER_MAX_PREFILL_TOKENS)
		return SPARK_STATUS_INVALID_ARGUMENT;
	debug_enabled = getenv("SPARKPIPE_STAGE_COMPLETION_DEBUG");
	if (debug_enabled != 0)
		fprintf(stderr,"pp13_builder_prefill_begin offset=%u count=%u\n",prefill_dispatch->prompt_token_offset,prefill_dispatch->prompt_token_count);
	status = SparkGlm52Pp13BuilderPrepareDeviceKvView(
		state,
		prefill_dispatch->kv_block_table_view);
	if (status != SPARK_STATUS_OK)
		return status;
	SparkGlm52Pp13BuilderMaybeProbePrefillKvState(state,prefill_dispatch);
	status = SparkGlm52Pp13BuilderUploadMtpBudget(state,1u,0u);
	if (status != SPARK_STATUS_OK)
		return status;
	for (token_offset = 0u;
		 token_offset < prefill_dispatch->prompt_token_count;
		 ++token_offset)
	{
		position = prefill_dispatch->prompt_token_offset + token_offset;
		token_id = prefill_dispatch->host_token_ids[token_offset];
		status = SparkGlm52Pp13BuilderSetDsaCandidateCount(
			state,
			position + 1u);
		if (status != SPARK_STATUS_OK)
			return status;
		state->host_decode_positions[0u] = position;
		state->host_decode_token_ids[0u] = token_id;
		status = SparkGlm52Pp13BuilderCudaStatus(cudaMemcpyAsync(
			state->device_decode_positions,
			state->host_decode_positions,
			sizeof(uint32_t),
			cudaMemcpyHostToDevice,
			state->stream));
		if (status == SPARK_STATUS_OK)
			status = SparkGlm52Pp13BuilderCudaStatus(cudaMemcpyAsync(
				state->device_decode_token_ids,
				state->host_decode_token_ids,
				sizeof(uint32_t),
				cudaMemcpyHostToDevice,
				state->stream));
		if (status != SPARK_STATUS_OK)
			return status;
		status = SparkGlm52Pp13BuilderLaunchDecodeMetadataForAllLayers(
			state,
			1u);
		if (status == SPARK_STATUS_OK)
			status = SparkGlm52Pp13BuilderLaunchSerialPrefillMetadata(
				state,
				1u,
				position);
		if (status != SPARK_STATUS_OK)
			return status;
		SparkGlm52Pp13BuilderMaybeProbePrefillTokenSlot(
			state,
			prefill_dispatch->request_dispatch->request_ids[0u],
			token_offset,
			position);
		word_count =
			SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION / 2u;
		block_count = (uint32_t)(
			(word_count + SPARK_GLM52_PP13_BUILDER_THREADS - 1u) /
			SPARK_GLM52_PP13_BUILDER_THREADS);
		SparkGlm52Pp13BuilderGatherDecodeEmbeddingKernel<<<
			block_count,
			SPARK_GLM52_PP13_BUILDER_THREADS,
			0,
			state->stream>>>(
				(const uint32_t *)state->device_decode_token_ids,
				(const uint32_t *)state->embedding_weight,
				(uint32_t *)state->layers[0].input_hidden,
				1u,
				SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION / 2u);
		status = SparkGlm52Pp13BuilderCudaStatus(cudaGetLastError());
		if (status != SPARK_STATUS_OK)
			return status;
		memset(&dispatch,0,sizeof(dispatch));
		dispatch.abi_version =
			SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_ABI_VERSION;
		dispatch.descriptor_bytes =
			SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_DISPATCH_BYTES;
		dispatch.flags =
			SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_DISPATCH_FLAG_PREFILL;
		dispatch.priority = prefill_dispatch->request_dispatch->highest_priority;
		dispatch.request_id = prefill_dispatch->request_dispatch->request_ids[0u];
		dispatch.sequence_id = prefill_dispatch->request_dispatch->sequence_ids[0u];
		dispatch.sequence_position = position;
		dispatch.active_sequence_count = 1u;
		dispatch.new_token_count = 1u;
		dispatch.pipeline_slot = 0u;
		dispatch.kv_block_table = &state->device_kv_view;
		dispatch.mtp_draft_token_budgets =
			(const uint32_t *)state->layers[0].mtp_draft_token_budgets;
		dispatch.hidden_output_transport_session = state->output_transport_session;
		status = SparkGlm52Pp13WorkControlBuildPrefillPacket(
			prefill_dispatch,token_offset,&work_packet);
		if (status != SPARK_STATUS_OK)
			return status;
		SparkGlm52Pp13BuilderApplyDsparkDispatch(state,&work_packet,&dispatch);
		SparkGlm52Pp13BuilderBuildPacket(
			state,
			&work_packet,
			state->layers[state->rank_plan.layer_count - 1u].layer_output_hidden,
			state->output_sideband,
			SparkGlm52Pp13BuilderNeedsOutputSideband(state),
			&dispatch.hidden_output_packet);
		status = SparkGlm52Pp13BuilderArmDsparkSideband(
			state,
			&work_packet,
			state->rank_plan.rank_index,
			state->output_sideband,
			&dispatch.hidden_output_packet);
		if (status != SPARK_STATUS_OK)
			return status;
		for (submit_retry = 0u; submit_retry < 25000u; ++submit_retry)
		{
			status = SparkGlm52ResidentDecodeStageProductionRunnerSubmit(
				&state->runner,
				&dispatch);
			if (status != SPARK_STATUS_BUSY)
				break;
			(void)SparkGlm52ResidentDecodeStageProductionRunnerProgress(
				&state->runner);
			if (idle_pump_function != 0)
				(void)idle_pump_function(idle_pump_context);
			usleep(200u);
		}
		if (status != SPARK_STATUS_OK)
		{
			fprintf(
				stderr,
				"pp13_builder_prefill_runner status=%u token_offset=%u position=%u token_id=%u\n",
				status,
				token_offset,
				position,
				token_id);
			return status;
		}
		status = SparkGlm52ResidentDecodeStageProductionRunnerWaitIdle(&state->runner,25000u);
		if (status != SPARK_STATUS_OK)
		{
			fprintf(stderr,"pp13_builder_prefill_drain status=%u position=%u\n",status,position);
			return status;
		}
		status = SparkGlm52Pp13BuilderCudaStatus(
			cudaStreamSynchronize(state->stream));
		if (status != SPARK_STATUS_OK)
		{
			fprintf(
				stderr,
				"pp13_builder_prefill_sync status=%u token_offset=%u position=%u token_id=%u\n",
				status,
				token_offset,
				position,
				token_id);
			return status;
		}
		status = SparkGlm52ResidentDecodeStageProductionRunnerProgress(
			&state->runner);
		if (status != SPARK_STATUS_OK)
		{
			fprintf(
				stderr,
				"pp13_builder_prefill_progress status=%u token_offset=%u position=%u token_id=%u\n",
				status,
				token_offset,
				position,
				token_id);
			return status;
		}
		SparkGlm52Pp13BuilderMaybeProbeLayer1Sublayers(
			state,
			prefill_dispatch->request_dispatch->request_ids[0u],
			token_offset,
			position);
		if (idle_pump_function != 0)
			(void)idle_pump_function(idle_pump_context);
		if (debug_enabled != 0)
			fprintf(stderr,"pp13_builder_prefill_submitted position=%u token_offset=%u busy_retries=%u\n",position,token_offset,submit_retry);
	}
	if (debug_enabled != 0)
		fprintf(stderr,"pp13_builder_prefill_done offset=%u count=%u\n",prefill_dispatch->prompt_token_offset,prefill_dispatch->prompt_token_count);
	SparkGlm52Pp13BuilderMaybeProbeMlaSlots(state);
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderDecode(
	void *builder_state,
	const SparkGlm52ServingDecodeDispatch *decode_dispatch,
	SparkGlm52ServingDecodeResult *decode_result)
{
	SparkGlm52Pp13BuilderState *state;
	SparkGlm52ResidentDecodeStageProductionRunnerDispatch dispatch;
	SparkGlm52Pp13WorkControlPacket work_packet;
	uint32_t block_count;
	uint32_t dspark_verify;
	uint32_t mtp_verify;
	uint32_t speculative_verify;
	uint32_t mtp_budget;
	uint32_t lane_index;
	uint64_t word_count;
	SparkStatus status;
	state = (SparkGlm52Pp13BuilderState *)builder_state;
	if (state == 0 || decode_dispatch == 0 || decode_result == 0 ||
		state->runner_ready == 0u ||
		state->embedding_weight == 0 ||
		decode_dispatch->abi_version != SPARK_GLM52_SERVING_ENGINE_ABI_VERSION ||
		decode_dispatch->descriptor_bytes !=
			SPARK_GLM52_SERVING_DECODE_DISPATCH_DESCRIPTOR_BYTES ||
		decode_dispatch->request_dispatch == 0 ||
		decode_dispatch->decode_view == 0 ||
		decode_dispatch->kv_block_table_view == 0 ||
		decode_dispatch->request_count == 0u ||
		decode_dispatch->request_count !=
			decode_dispatch->active_sequence_count ||
		decode_dispatch->active_sequence_count >
			state->rank_plan.max_active_sequence_count ||
		decode_dispatch->decode_view->lane_count !=
			decode_dispatch->active_sequence_count ||
		(decode_dispatch->dispatch_kind !=
				SPARK_GLM52_REQUEST_API_DISPATCH_KIND_DECODE_BATCH &&
		 decode_dispatch->dispatch_kind !=
				SPARK_GLM52_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH))
		return SPARK_STATUS_INVALID_ARGUMENT;
	dspark_verify = (decode_dispatch->request_dispatch->flags &
		SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_DSPARK_SPECULATIVE_VERIFY) != 0u;
	mtp_verify = (decode_dispatch->request_dispatch->flags &
		SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_MTP_SPECULATIVE_VERIFY) != 0u;
	speculative_verify = dspark_verify | mtp_verify;
	if (dspark_verify != 0u && mtp_verify != 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (decode_dispatch->dispatch_kind ==
			SPARK_GLM52_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH &&
		(speculative_verify == 0u ||
		 decode_dispatch->speculative_token_count == 0u ||
		 decode_dispatch->speculative_token_count >
			SPARK_GLM52_PP13_WORK_CONTROL_MAX_SPECULATIVE_TOKEN_COUNT ||
		 decode_dispatch->speculative_token_index >
			decode_dispatch->speculative_token_count))
		return SPARK_STATUS_INVALID_ARGUMENT;
	status = SparkGlm52Pp13WorkControlBuildDecodePacket(
		decode_dispatch,
		decode_dispatch->speculative_token_index,
		&work_packet);
	if (status != SPARK_STATUS_OK)
		return status;
	status = SparkGlm52Pp13BuilderPrepareDeviceKvView(
		state,
		decode_dispatch->kv_block_table_view);
	if (status != SPARK_STATUS_OK)
		return status;
	status = SparkGlm52Pp13BuilderSetDsaCandidateCount(
		state,
		work_packet.kv_block_table_token_count);
	if (status != SPARK_STATUS_OK)
		return status;
	for (lane_index = 0u;
		 lane_index < decode_dispatch->active_sequence_count;
		 ++lane_index)
	{
		state->host_decode_positions[lane_index] =
			(uint32_t)work_packet.lanes[lane_index].sequence_position;
		state->host_decode_token_ids[lane_index] =
			work_packet.lanes[lane_index].input_token_id;
	}
	status = SparkGlm52Pp13BuilderCudaStatus(cudaMemcpyAsync(
		state->device_decode_positions,
		state->host_decode_positions,
		(size_t)(decode_dispatch->active_sequence_count * sizeof(uint32_t)),
		cudaMemcpyHostToDevice,
		state->stream));
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaStatus(cudaMemcpyAsync(
			state->device_decode_token_ids,
			state->host_decode_token_ids,
			(size_t)(decode_dispatch->active_sequence_count * sizeof(uint32_t)),
			cudaMemcpyHostToDevice,
			state->stream));
	if (status != SPARK_STATUS_OK)
		return status;
	status = SparkGlm52Pp13BuilderLaunchDecodeMetadataForAllLayers(
		state,
		decode_dispatch->active_sequence_count);
	if (status != SPARK_STATUS_OK)
		return status;
	word_count =
		(uint64_t)decode_dispatch->active_sequence_count *
		(SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION / 2u);
	block_count = (uint32_t)(
		(word_count + SPARK_GLM52_PP13_BUILDER_THREADS - 1u) /
		SPARK_GLM52_PP13_BUILDER_THREADS);
	SparkGlm52Pp13BuilderGatherDecodeEmbeddingKernel<<<
		block_count,
		SPARK_GLM52_PP13_BUILDER_THREADS,
		0,
		state->stream>>>(
			(const uint32_t *)state->device_decode_token_ids,
			(const uint32_t *)state->embedding_weight,
			(uint32_t *)state->layers[0].input_hidden,
			decode_dispatch->active_sequence_count,
			SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION / 2u);
	status = SparkGlm52Pp13BuilderCudaStatus(cudaGetLastError());
	if (status != SPARK_STATUS_OK)
		return status;
	mtp_budget = 0u;
	if (speculative_verify == 0u &&
		(decode_dispatch->request_dispatch->flags &
			SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_MTP_COMMIT) != 0u)
		mtp_budget = decode_dispatch->request_dispatch->mtp_draft_token_budget;
	status = SparkGlm52Pp13BuilderUploadMtpBudget(
		state,decode_dispatch->active_sequence_count,mtp_budget);
	if (status != SPARK_STATUS_OK)
		return status;
	memset(&dispatch,0,sizeof(dispatch));
	dispatch.abi_version =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_ABI_VERSION;
	dispatch.descriptor_bytes =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_DISPATCH_BYTES;
	dispatch.priority = work_packet.priority;
	dispatch.request_id = work_packet.request_id;
	dispatch.sequence_id = work_packet.sequence_id;
	dispatch.sequence_position = work_packet.sequence_position;
	dispatch.active_sequence_count = work_packet.active_sequence_count;
	dispatch.new_token_count = work_packet.new_token_count;
	dispatch.pipeline_slot = 0u;
	dispatch.kv_block_table = &state->device_kv_view;
	if (mtp_budget != 0u)
		dispatch.mtp_draft_token_budgets =
			(const uint32_t *)state->layers[0].mtp_draft_token_budgets;
	dispatch.hidden_output_transport_session = state->output_transport_session;
	SparkGlm52Pp13BuilderApplyDsparkDispatch(state,&work_packet,&dispatch);
	SparkGlm52Pp13BuilderBuildPacket(
		state,
		&work_packet,
		state->layers[state->rank_plan.layer_count - 1u].layer_output_hidden,
		state->output_sideband,
		SparkGlm52Pp13BuilderNeedsOutputSideband(state),
		&dispatch.hidden_output_packet);
	status = SparkGlm52Pp13BuilderArmDsparkSideband(
		state,
		&work_packet,
		state->rank_plan.rank_index,
		state->output_sideband,
		&dispatch.hidden_output_packet);
	if (status != SPARK_STATUS_OK)
		return status;
	status = SparkGlm52ResidentDecodeStageProductionRunnerSubmit(
		&state->runner,
		&dispatch);
	if (status != SPARK_STATUS_OK)
		return status;
	memset(decode_result,0,sizeof(*decode_result));
	decode_result->abi_version = SPARK_GLM52_SERVING_ENGINE_ABI_VERSION;
	decode_result->descriptor_bytes =
		SPARK_GLM52_SERVING_DECODE_RESULT_DESCRIPTOR_BYTES;
	decode_result->lane_count = decode_dispatch->active_sequence_count;
	decode_result->token_stride = SPARK_GLM52_SERVING_MAX_DECODE_TOKENS_PER_LANE;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderTakeDsparkDraft(
	void *builder_state,
	SparkGlm52DsparkDraftResult *draft_result)
{
	SparkGlm52Pp13BuilderState *state;

	state = (SparkGlm52Pp13BuilderState *)builder_state;
	if (state == 0 || draft_result == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (state->dspark_ready_draft_valid == 0u)
		return SPARK_STATUS_NOT_FOUND;
	*draft_result = state->dspark_ready_draft;
	memset(&state->dspark_ready_draft,0,sizeof(state->dspark_ready_draft));
	state->dspark_ready_draft_valid = 0u;
	return SPARK_STATUS_OK;
}

static const SparkGlm52Pp13NodeContextBuilderInterface SparkGlm52Pp13BuilderInterface =
{
	SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_ABI_VERSION,
	SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_INTERFACE_BYTES,
	SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_REQUIRED_PRODUCTION_CAPS,
	0u,
	SparkGlm52Pp13BuilderInitialize,
	SparkGlm52Pp13BuilderDestroy,
	SparkGlm52Pp13BuilderBuild,
	SparkGlm52Pp13BuilderDestroyResult,
	SparkGlm52Pp13BuilderAttachDriver,
	SparkGlm52Pp13BuilderPrefill,
	SparkGlm52Pp13BuilderDecode,
	SparkGlm52Pp13BuilderSubmitWork,
	SparkGlm52Pp13BuilderTakeDsparkDraft
};

extern "C" const SparkGlm52Pp13NodeContextBuilderInterface *
SparkGlm52Pp13NodeContextBuilderGetInterface(void)
{
	return &SparkGlm52Pp13BuilderInterface;
}
