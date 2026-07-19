/* Large stage packs exceed 2 GB: 64-bit file offsets are required. */
#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>

#include "sparkpipe/spark_qwen36_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_stage_kv_client.h"
#include "sparkpipe/spark_stage_module_common.h"
#include "spark_qwen36_stagepack_format.h"

/*
 * Qwen 3.6 27B resident decode stage host module, PP-Nx native.
 *
 * One process is one STAGE: configuration names the stage count, the stage
 * index and the layer slice; the pack must declare exactly that slice and
 * exactly the computed tensor inventory. Embedding and head ownership are
 * derived from slice position, and the frame transport flags must agree with
 * the position in both directions - a mid-pipeline stage without both
 * transports, or an edge stage with the wrong one, is a refused frame.
 *
 * v1 executes DECODE frames only: one next token per row for up to
 * max_active_sequence_count distinct lanes. Prefill is the runtime feeding
 * prompt tokens through the same path one position at a time, which the
 * carry oracle proves bitwise equal to any batched prefill; the chunked
 * prefill kernel is a later throughput commit. Execute is synchronous.
 */

#define SPARK_QWEN36_MODULE_TAG "qwen36_stage"

typedef struct SparkQwen36ModuleSlot
{
	void *cuda_stream;
	uint32_t *input_token_ids;
	uint32_t *output_token_ids;
	uint32_t *row_lane_indices;
	uint32_t *slot_mapping;
	uint32_t *context_lengths;
	uint32_t *row_cold;
	uint64_t *row_positions;
	void *hidden_bf16;
	void *normalized_bf16;
	void *delta_bf16;
	void *qkv_bf16;
	void *conv_out_bf16;
	void *z_bf16;
	void *beta_pre_bf16;
	void *decay_pre_bf16;
	float *log_decay_f32;
	float *beta_f32;
	void *core_bf16;
	void *gated_bf16;
	void *q_fused_bf16;
	void *k_bf16;
	void *v_bf16;
	void *head_out_bf16;
	void *ffn_gate_bf16;
	void *ffn_up_bf16;
} SparkQwen36ModuleSlot;

typedef struct SparkQwen36ModuleState
{
	SparkStageModuleLedger ledger;
	uint32_t stage_count;
	uint32_t stage_index;
	uint32_t first_layer_index;
	uint32_t layer_count;
	uint32_t owns_embedding;
	uint32_t owns_final_head;
	uint32_t max_active_sequence_count;
	uint32_t pipeline_slot_count;
	uint32_t kv_block_count;
	uint32_t gdn_layer_count;
	uint32_t attn_layer_count;
	uint32_t gdn_ordinal_by_layer[SPARK_QWEN36_RESIDENT_DECODE_STAGE_LAYER_COUNT];
	uint32_t attn_ordinal_by_layer[SPARK_QWEN36_RESIDENT_DECODE_STAGE_LAYER_COUNT];
	uint32_t layer_seen_bits[SPARK_QWEN36_RESIDENT_DECODE_STAGE_LAYER_COUNT];
	uint32_t global_seen_bits;
	uint32_t mtp_seen_bits;
	SparkQwen36MtpWeights mtp;
	const void *token_embedding_bf16;
	const void *final_norm_weight_bf16;
	const void *lm_head_weight_bf16;
	const void *attention_norm_by_layer[SPARK_QWEN36_RESIDENT_DECODE_STAGE_LAYER_COUNT];
	const void *mlp_norm_by_layer[SPARK_QWEN36_RESIDENT_DECODE_STAGE_LAYER_COUNT];
	SparkQwen36GdnLayerWeights gdn_by_layer[SPARK_QWEN36_RESIDENT_DECODE_STAGE_LAYER_COUNT];
	SparkQwen36AttnLayerWeights attn_by_layer[SPARK_QWEN36_RESIDENT_DECODE_STAGE_LAYER_COUNT];
	SparkQwen36FfnLayerWeights ffn_by_layer[SPARK_QWEN36_RESIDENT_DECODE_STAGE_LAYER_COUNT];
	SparkQwen36GdnStatePool gdn_pool;
	void *kv_cache_bf16;
	uint64_t cache_layer_stride;
	uint64_t cache_block_stride;
	uint8_t lane_warm[SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t host_row_cold[SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t host_slot_mapping[SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t host_context_lengths[SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	SparkQwen36ModuleSlot slots[SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
	atomic_uint slot_states[SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
	SparkStageKvClient kv_client;
	atomic_ullong frames_executed;
	atomic_ullong tokens_emitted;
} SparkQwen36ModuleState;

extern cudaError_t SparkQwen36LaunchRmsNorm(cudaStream_t stream, const void *input_bf16, const void *gain_bf16, void *output_bf16, uint32_t row_count, uint32_t dimension, float epsilon);
extern cudaError_t SparkQwen36LaunchLinear(cudaStream_t stream, const SparkQwen36LinearView *view, const void *input_bf16, void *output_bf16, uint32_t row_count);
extern cudaError_t SparkQwen36LaunchEmbeddingGather(cudaStream_t stream, const uint32_t *token_ids, const void *embedding_bf16, void *hidden_bf16, uint32_t row_count);
extern cudaError_t SparkQwen36LaunchConvUpdate(cudaStream_t stream, const void *qkv_bf16, const SparkQwen36GdnLayerWeights *weights, void *conv_out_bf16, const SparkQwen36GdnStatePool *pool, const uint32_t *row_lane_indices, uint32_t row_count, uint32_t gdn_layer_ordinal);
extern cudaError_t SparkQwen36LaunchDecayBeta(cudaStream_t stream, const void *decay_pre_bf16, const void *beta_pre_bf16, const SparkQwen36GdnLayerWeights *weights, float *log_decay_f32, float *beta_f32, uint32_t row_count);
extern cudaError_t SparkQwen36LaunchGdnStep(cudaStream_t stream, const void *conv_out_bf16, const float *log_decay_f32, const float *beta_f32, const SparkQwen36GdnStatePool *pool, void *core_out_bf16, const uint32_t *row_lane_indices, uint32_t row_count, uint32_t gdn_layer_ordinal);
extern cudaError_t SparkQwen36LaunchGatedNorm(cudaStream_t stream, const void *core_bf16, const void *z_bf16, const SparkQwen36GdnLayerWeights *weights, void *output_bf16, uint32_t row_count, float epsilon);
extern cudaError_t SparkQwen36LaunchAttnPrepare(cudaStream_t stream, void *q_fused_bf16, const void *k_bf16, const void *v_bf16, const SparkQwen36AttnLayerWeights *weights, void *kv_cache_bf16, const uint32_t *slot_mapping, const uint64_t *row_positions, uint32_t row_count, uint32_t attn_layer_ordinal, uint64_t cache_layer_stride, uint64_t cache_block_stride, float epsilon);
extern cudaError_t SparkQwen36LaunchAttnDecode(cudaStream_t stream, const void *q_fused_bf16, const void *kv_cache_bf16, const SparkQwen36KvBlockTableView *table, const uint32_t *row_lane_indices, const uint32_t *context_lengths, void *head_out_bf16, uint32_t row_count, uint32_t attn_layer_ordinal, uint64_t cache_layer_stride, uint64_t cache_block_stride);
extern cudaError_t SparkQwen36LaunchResidualAdd(cudaStream_t stream, void *hidden_bf16, const void *delta_bf16, uint32_t row_count, uint32_t dimension);
extern cudaError_t SparkQwen36LaunchSwiGlu(cudaStream_t stream, const void *gate_bf16, void *up_bf16, uint32_t row_count, uint32_t dimension);
extern cudaError_t SparkQwen36LaunchHeadArgmax(cudaStream_t stream, const void *hidden_bf16, const void *head_weight_bf16, const uint32_t *token_ids, uint32_t *output_token_ids, uint32_t row_count, uint32_t candidate_count);

static SparkStatus SparkQwen36ModuleConfigure(SparkQwen36ModuleState *state)
{
	SparkStatus status;
	status = SparkStageModuleEnvironmentUnsigned(SPARK_QWEN36_MODULE_TAG,"SPARK_QWEN36_STAGE_COUNT",1u,SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_STAGE_COUNT,&state->stage_count);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_QWEN36_MODULE_TAG,"SPARK_QWEN36_STAGE_INDEX",0u,SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_STAGE_COUNT - 1u,&state->stage_index);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_QWEN36_MODULE_TAG,"SPARK_QWEN36_STAGE_FIRST_LAYER",0u,SPARK_QWEN36_RESIDENT_DECODE_STAGE_LAYER_COUNT - 1u,&state->first_layer_index);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_QWEN36_MODULE_TAG,"SPARK_QWEN36_STAGE_LAYER_COUNT",1u,SPARK_QWEN36_RESIDENT_DECODE_STAGE_LAYER_COUNT,&state->layer_count);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_QWEN36_MODULE_TAG,"SPARK_QWEN36_STAGE_MAX_ACTIVE_SEQUENCES",1u,SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,&state->max_active_sequence_count);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_QWEN36_MODULE_TAG,"SPARK_QWEN36_STAGE_PIPELINE_SLOTS",1u,SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT,&state->pipeline_slot_count);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_QWEN36_MODULE_TAG,"SPARK_QWEN36_STAGE_KV_BLOCKS",1u,1u << 20u,&state->kv_block_count);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( state->stage_index >= state->stage_count || state->first_layer_index + state->layer_count > SPARK_QWEN36_RESIDENT_DECODE_STAGE_LAYER_COUNT )
	{
		fprintf(stderr,"%s config_slice_invalid stage=%u/%u slice=%u+%u\n",SPARK_QWEN36_MODULE_TAG,state->stage_index,state->stage_count,state->first_layer_index,state->layer_count);
		return(SPARK_STATUS_INVALID_ARGUMENT);
	}
	state->owns_embedding = state->first_layer_index == 0u ? 1u : 0u;
	state->owns_final_head = state->first_layer_index + state->layer_count == SPARK_QWEN36_RESIDENT_DECODE_STAGE_LAYER_COUNT ? 1u : 0u;
	if ( (state->stage_index == 0u) != (state->owns_embedding != 0u) || (state->stage_index + 1u == state->stage_count) != (state->owns_final_head != 0u) )
	{
		fprintf(stderr,"%s config_position_mismatch stage=%u/%u slice=%u+%u\n",SPARK_QWEN36_MODULE_TAG,state->stage_index,state->stage_count,state->first_layer_index,state->layer_count);
		return(SPARK_STATUS_INVALID_ARGUMENT);
	}
	return(SPARK_STATUS_OK);
}

static void SparkQwen36ModuleBuildOrdinals(SparkQwen36ModuleState *state)
{
	uint32_t layer;
	for (layer = 0; layer < SPARK_QWEN36_RESIDENT_DECODE_STAGE_LAYER_COUNT; layer++)
	{
		state->gdn_ordinal_by_layer[layer] = UINT32_MAX;
		state->attn_ordinal_by_layer[layer] = UINT32_MAX;
	}
	for (layer = state->first_layer_index; layer < state->first_layer_index + state->layer_count; layer++)
	{
		if ( SPARK_QWEN36_MODEL_LAYER_IS_GDN(layer) != 0u )
			state->gdn_ordinal_by_layer[layer] = state->gdn_layer_count++;
		else
			state->attn_ordinal_by_layer[layer] = state->attn_layer_count++;
	}
}

static void SparkQwen36ModuleFillLinearView(SparkQwen36LinearView *view, const SparkQwen36StagePackEntry *entry, void *payload, void *scale)
{
	view->abi_version = SPARK_QWEN36_RESIDENT_DECODE_STAGE_LINEAR_VIEW_ABI_VERSION;
	view->weight_format = entry->weight_format;
	view->input_dimension = entry->columns;
	view->output_dimension = entry->rows;
	view->weight_payload = payload;
	view->weight_scale_e8m0 = (const uint8_t *)scale;
	view->weight_payload_bytes = entry->payload_bytes;
	view->weight_scale_bytes = entry->scale_bytes;
}

static SparkStatus SparkQwen36ModuleValidateEntry(SparkQwen36ModuleState *state, const SparkQwen36StagePackEntry *entry, uint64_t file_bytes, uint32_t *is_global)
{
	SparkQwen36StagePackTensorShape shape;
	uint32_t global = entry->layer_index == SPARK_QWEN36_STAGEPACK_GLOBAL_LAYER ? 1u : 0u;
	if ( SparkQwen36StagePackResolvedShape(entry->tensor_kind,global != 0u ? 0u : entry->layer_index,global,&shape) != 0 || entry->rows != shape.rows || entry->columns != shape.columns )
		return(SPARK_STATUS_VALIDATION_FAILED);
	if ( shape.quantizable != 0u )
	{
		if ( entry->weight_format != SPARK_QWEN36_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16 && entry->weight_format != SPARK_QWEN36_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_MXFP4_E2M1 )
			return(SPARK_STATUS_VALIDATION_FAILED);
	}
	else if ( entry->weight_format != shape.natural_format )
		return(SPARK_STATUS_VALIDATION_FAILED);
	if ( entry->weight_format == SPARK_QWEN36_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_MXFP4_E2M1 ? entry->scale_group_size != 32u : entry->scale_group_size != 0u )
		return(SPARK_STATUS_VALIDATION_FAILED);
	if ( entry->payload_bytes != SparkQwen36StagePackPayloadBytes(entry->weight_format,entry->rows,entry->columns) || entry->scale_bytes != SparkQwen36StagePackScaleBytes(entry->weight_format,entry->rows,entry->columns) )
		return(SPARK_STATUS_VALIDATION_FAILED);
	if ( entry->payload_offset > file_bytes || entry->payload_bytes > file_bytes - entry->payload_offset )
		return(SPARK_STATUS_VALIDATION_FAILED);
	if ( entry->scale_bytes != 0u && (entry->scale_offset > file_bytes || entry->scale_bytes > file_bytes - entry->scale_offset) )
		return(SPARK_STATUS_VALIDATION_FAILED);
	if ( entry->layer_index == SPARK_QWEN36_STAGEPACK_MTP_LAYER || (global != 0u && (entry->tensor_kind >= SPARK_QWEN36_STAGEPACK_TENSOR_MTP_FC && entry->tensor_kind <= SPARK_QWEN36_STAGEPACK_TENSOR_MTP_FINAL_NORM)) )
	{
		if ( state->owns_final_head == 0u )
			return(SPARK_STATUS_VALIDATION_FAILED);
	}
	else if ( global == 0u && (entry->layer_index < state->first_layer_index || entry->layer_index >= state->first_layer_index + state->layer_count) )
		return(SPARK_STATUS_VALIDATION_FAILED);
	*is_global = global;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkQwen36ModuleBindMtp(SparkQwen36ModuleState *state, const SparkQwen36StagePackEntry *entry, void *payload, void *scale)
{
	switch ( entry->tensor_kind )
	{
	case SPARK_QWEN36_STAGEPACK_TENSOR_MTP_FC: SparkQwen36ModuleFillLinearView(&state->mtp.fc,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN36_STAGEPACK_TENSOR_ATTENTION_NORM: state->mtp.attention_norm_weight_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN36_STAGEPACK_TENSOR_MLP_NORM: state->mtp.mlp_norm_weight_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN36_STAGEPACK_TENSOR_FFN_GATE: SparkQwen36ModuleFillLinearView(&state->mtp.ffn.gate,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN36_STAGEPACK_TENSOR_FFN_UP: SparkQwen36ModuleFillLinearView(&state->mtp.ffn.up,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN36_STAGEPACK_TENSOR_FFN_DOWN: SparkQwen36ModuleFillLinearView(&state->mtp.ffn.down,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN36_STAGEPACK_TENSOR_ATTN_QUERY: SparkQwen36ModuleFillLinearView(&state->mtp.attention.query,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN36_STAGEPACK_TENSOR_ATTN_KEY: SparkQwen36ModuleFillLinearView(&state->mtp.attention.key,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN36_STAGEPACK_TENSOR_ATTN_VALUE: SparkQwen36ModuleFillLinearView(&state->mtp.attention.value,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN36_STAGEPACK_TENSOR_ATTN_OUTPUT: SparkQwen36ModuleFillLinearView(&state->mtp.attention.output,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN36_STAGEPACK_TENSOR_ATTN_QUERY_NORM: state->mtp.attention.query_norm_weight_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN36_STAGEPACK_TENSOR_ATTN_KEY_NORM: state->mtp.attention.key_norm_weight_bf16 = payload; return(SPARK_STATUS_OK);
	default:
		return(SPARK_STATUS_VALIDATION_FAILED);
	}
}

static SparkStatus SparkQwen36ModuleBindGlobal(SparkQwen36ModuleState *state, const SparkQwen36StagePackEntry *entry, void *payload)
{
	switch ( entry->tensor_kind )
	{
	case SPARK_QWEN36_STAGEPACK_TENSOR_EMBEDDING:
		if ( state->owns_embedding == 0u )
			return(SPARK_STATUS_VALIDATION_FAILED);
		state->token_embedding_bf16 = payload;
		return(SPARK_STATUS_OK);
	case SPARK_QWEN36_STAGEPACK_TENSOR_FINAL_NORM:
		if ( state->owns_final_head == 0u )
			return(SPARK_STATUS_VALIDATION_FAILED);
		state->final_norm_weight_bf16 = payload;
		return(SPARK_STATUS_OK);
	case SPARK_QWEN36_STAGEPACK_TENSOR_LM_HEAD:
		if ( state->owns_final_head == 0u )
			return(SPARK_STATUS_VALIDATION_FAILED);
		state->lm_head_weight_bf16 = payload;
		return(SPARK_STATUS_OK);
	case SPARK_QWEN36_STAGEPACK_TENSOR_MTP_EMBED_NORM: state->mtp.embed_norm_weight_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN36_STAGEPACK_TENSOR_MTP_HIDDEN_NORM: state->mtp.hidden_norm_weight_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN36_STAGEPACK_TENSOR_MTP_FINAL_NORM: state->mtp.final_norm_weight_bf16 = payload; return(SPARK_STATUS_OK);
	default:
		return(SPARK_STATUS_VALIDATION_FAILED);
	}
}

static SparkStatus SparkQwen36ModuleBindLayer(SparkQwen36ModuleState *state, const SparkQwen36StagePackEntry *entry, void *payload, void *scale)
{
	uint32_t layer = entry->layer_index;
	switch ( entry->tensor_kind )
	{
	case SPARK_QWEN36_STAGEPACK_TENSOR_ATTENTION_NORM: state->attention_norm_by_layer[layer] = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN36_STAGEPACK_TENSOR_MLP_NORM: state->mlp_norm_by_layer[layer] = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN36_STAGEPACK_TENSOR_FFN_GATE: SparkQwen36ModuleFillLinearView(&state->ffn_by_layer[layer].gate,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN36_STAGEPACK_TENSOR_FFN_UP: SparkQwen36ModuleFillLinearView(&state->ffn_by_layer[layer].up,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN36_STAGEPACK_TENSOR_FFN_DOWN: SparkQwen36ModuleFillLinearView(&state->ffn_by_layer[layer].down,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN36_STAGEPACK_TENSOR_GDN_QKV: SparkQwen36ModuleFillLinearView(&state->gdn_by_layer[layer].qkv,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN36_STAGEPACK_TENSOR_GDN_GATE: SparkQwen36ModuleFillLinearView(&state->gdn_by_layer[layer].gate,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN36_STAGEPACK_TENSOR_GDN_BETA: SparkQwen36ModuleFillLinearView(&state->gdn_by_layer[layer].beta,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN36_STAGEPACK_TENSOR_GDN_DECAY: SparkQwen36ModuleFillLinearView(&state->gdn_by_layer[layer].decay,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN36_STAGEPACK_TENSOR_GDN_OUTPUT: SparkQwen36ModuleFillLinearView(&state->gdn_by_layer[layer].output,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN36_STAGEPACK_TENSOR_GDN_CONV_WEIGHT: state->gdn_by_layer[layer].conv_weight_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN36_STAGEPACK_TENSOR_GDN_A_LOG: state->gdn_by_layer[layer].a_log_f32 = (const float *)payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN36_STAGEPACK_TENSOR_GDN_DT_BIAS: state->gdn_by_layer[layer].dt_bias_f32 = (const float *)payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN36_STAGEPACK_TENSOR_GDN_NORM: state->gdn_by_layer[layer].gdn_norm_weight_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN36_STAGEPACK_TENSOR_ATTN_QUERY: SparkQwen36ModuleFillLinearView(&state->attn_by_layer[layer].query,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN36_STAGEPACK_TENSOR_ATTN_KEY: SparkQwen36ModuleFillLinearView(&state->attn_by_layer[layer].key,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN36_STAGEPACK_TENSOR_ATTN_VALUE: SparkQwen36ModuleFillLinearView(&state->attn_by_layer[layer].value,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN36_STAGEPACK_TENSOR_ATTN_OUTPUT: SparkQwen36ModuleFillLinearView(&state->attn_by_layer[layer].output,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN36_STAGEPACK_TENSOR_ATTN_QUERY_NORM: state->attn_by_layer[layer].query_norm_weight_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN36_STAGEPACK_TENSOR_ATTN_KEY_NORM: state->attn_by_layer[layer].key_norm_weight_bf16 = payload; return(SPARK_STATUS_OK);
	default:
		return(SPARK_STATUS_VALIDATION_FAILED);
	}
}

static SparkStatus SparkQwen36ModuleLoadEntry(SparkQwen36ModuleState *state, FILE *file, const SparkQwen36StagePackEntry *entry, uint64_t file_bytes)
{
	SparkStatus status;
	uint32_t is_global = 0u,bit = 1u << entry->tensor_kind;
	uint32_t *seen;
	void *payload = 0,*scale = 0;
	status = SparkQwen36ModuleValidateEntry(state,entry,file_bytes,&is_global);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"%s pack_entry_invalid kind=%u layer=%u\n",SPARK_QWEN36_MODULE_TAG,entry->tensor_kind,entry->layer_index);
		return(status);
	}
	if ( entry->layer_index == SPARK_QWEN36_STAGEPACK_MTP_LAYER || (is_global != 0u && entry->tensor_kind >= SPARK_QWEN36_STAGEPACK_TENSOR_MTP_FC && entry->tensor_kind <= SPARK_QWEN36_STAGEPACK_TENSOR_MTP_FINAL_NORM) )
		seen = &state->mtp_seen_bits;
	else
		seen = is_global != 0u ? &state->global_seen_bits : &state->layer_seen_bits[entry->layer_index];
	if ( (*seen & bit) != 0u )
	{
		fprintf(stderr,"%s pack_entry_duplicate kind=%u layer=%u\n",SPARK_QWEN36_MODULE_TAG,entry->tensor_kind,entry->layer_index);
		return(SPARK_STATUS_VALIDATION_FAILED);
	}
	*seen |= bit;
	status = SparkStageModuleLoadDeviceRegion(&state->ledger,file,entry->payload_offset,entry->payload_bytes,&payload);
	if ( status == SPARK_STATUS_OK && entry->scale_bytes != 0u )
		status = SparkStageModuleLoadDeviceRegion(&state->ledger,file,entry->scale_offset,entry->scale_bytes,&scale);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( entry->layer_index == SPARK_QWEN36_STAGEPACK_MTP_LAYER || entry->tensor_kind == SPARK_QWEN36_STAGEPACK_TENSOR_MTP_FC )
		return(SparkQwen36ModuleBindMtp(state,entry,payload,scale));
	return(is_global != 0u ? SparkQwen36ModuleBindGlobal(state,entry,payload) : SparkQwen36ModuleBindLayer(state,entry,payload,scale));
}

static SparkStatus SparkQwen36ModuleVerifyCoverage(SparkQwen36ModuleState *state)
{
	uint32_t layer,expected_global = 0u,expected_layer;
	if ( state->owns_embedding != 0u )
		expected_global |= 1u << SPARK_QWEN36_STAGEPACK_TENSOR_EMBEDDING;
	if ( state->owns_final_head != 0u )
		expected_global |= (1u << SPARK_QWEN36_STAGEPACK_TENSOR_FINAL_NORM) | (1u << SPARK_QWEN36_STAGEPACK_TENSOR_LM_HEAD);
	if ( state->owns_final_head != 0u )
	{
		uint32_t expected_mtp = (1u << SPARK_QWEN36_STAGEPACK_TENSOR_MTP_FC) | (1u << SPARK_QWEN36_STAGEPACK_TENSOR_MTP_EMBED_NORM) | (1u << SPARK_QWEN36_STAGEPACK_TENSOR_MTP_HIDDEN_NORM) | (1u << SPARK_QWEN36_STAGEPACK_TENSOR_MTP_FINAL_NORM) | (1u << SPARK_QWEN36_STAGEPACK_TENSOR_ATTENTION_NORM) | (1u << SPARK_QWEN36_STAGEPACK_TENSOR_MLP_NORM) | (1u << SPARK_QWEN36_STAGEPACK_TENSOR_FFN_GATE) | (1u << SPARK_QWEN36_STAGEPACK_TENSOR_FFN_UP) | (1u << SPARK_QWEN36_STAGEPACK_TENSOR_FFN_DOWN) | (1u << SPARK_QWEN36_STAGEPACK_TENSOR_ATTN_QUERY) | (1u << SPARK_QWEN36_STAGEPACK_TENSOR_ATTN_KEY) | (1u << SPARK_QWEN36_STAGEPACK_TENSOR_ATTN_VALUE) | (1u << SPARK_QWEN36_STAGEPACK_TENSOR_ATTN_OUTPUT) | (1u << SPARK_QWEN36_STAGEPACK_TENSOR_ATTN_QUERY_NORM) | (1u << SPARK_QWEN36_STAGEPACK_TENSOR_ATTN_KEY_NORM);
		if ( state->mtp_seen_bits != expected_mtp )
		{
			fprintf(stderr,"%s pack_mtp_incomplete seen=%08x expected=%08x\n",SPARK_QWEN36_MODULE_TAG,state->mtp_seen_bits,expected_mtp);
			return(SPARK_STATUS_VALIDATION_FAILED);
		}
	}
	if ( state->global_seen_bits != expected_global )
	{
		fprintf(stderr,"%s pack_globals_incomplete seen=%08x expected=%08x\n",SPARK_QWEN36_MODULE_TAG,state->global_seen_bits,expected_global);
		return(SPARK_STATUS_VALIDATION_FAILED);
	}
	for (layer = state->first_layer_index; layer < state->first_layer_index + state->layer_count; layer++)
	{
		expected_layer = (1u << SPARK_QWEN36_STAGEPACK_TENSOR_ATTENTION_NORM) | (1u << SPARK_QWEN36_STAGEPACK_TENSOR_MLP_NORM) | (1u << SPARK_QWEN36_STAGEPACK_TENSOR_FFN_GATE) | (1u << SPARK_QWEN36_STAGEPACK_TENSOR_FFN_UP) | (1u << SPARK_QWEN36_STAGEPACK_TENSOR_FFN_DOWN);
		if ( SPARK_QWEN36_MODEL_LAYER_IS_GDN(layer) != 0u )
			expected_layer |= (1u << SPARK_QWEN36_STAGEPACK_TENSOR_GDN_QKV) | (1u << SPARK_QWEN36_STAGEPACK_TENSOR_GDN_GATE) | (1u << SPARK_QWEN36_STAGEPACK_TENSOR_GDN_BETA) | (1u << SPARK_QWEN36_STAGEPACK_TENSOR_GDN_DECAY) | (1u << SPARK_QWEN36_STAGEPACK_TENSOR_GDN_OUTPUT) | (1u << SPARK_QWEN36_STAGEPACK_TENSOR_GDN_CONV_WEIGHT) | (1u << SPARK_QWEN36_STAGEPACK_TENSOR_GDN_A_LOG) | (1u << SPARK_QWEN36_STAGEPACK_TENSOR_GDN_DT_BIAS) | (1u << SPARK_QWEN36_STAGEPACK_TENSOR_GDN_NORM);
		else
			expected_layer |= (1u << SPARK_QWEN36_STAGEPACK_TENSOR_ATTN_QUERY) | (1u << SPARK_QWEN36_STAGEPACK_TENSOR_ATTN_KEY) | (1u << SPARK_QWEN36_STAGEPACK_TENSOR_ATTN_VALUE) | (1u << SPARK_QWEN36_STAGEPACK_TENSOR_ATTN_OUTPUT) | (1u << SPARK_QWEN36_STAGEPACK_TENSOR_ATTN_QUERY_NORM) | (1u << SPARK_QWEN36_STAGEPACK_TENSOR_ATTN_KEY_NORM);
		if ( state->layer_seen_bits[layer] != expected_layer )
		{
			fprintf(stderr,"%s pack_layer_incomplete layer=%u seen=%08x expected=%08x\n",SPARK_QWEN36_MODULE_TAG,layer,state->layer_seen_bits[layer],expected_layer);
			return(SPARK_STATUS_VALIDATION_FAILED);
		}
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkQwen36ModuleLoadPack(SparkQwen36ModuleState *state, const char *path)
{
	SparkQwen36StagePackHeader header,expected;
	SparkQwen36StagePackEntry *directory;
	FILE *file;
	SparkStatus status;
	int32_t compare;
	uint32_t index;
	file = fopen(path,"rb");
	if ( file == 0 )
	{
		fprintf(stderr,"%s pack_open_failed path=%s\n",SPARK_QWEN36_MODULE_TAG,path);
		return(SPARK_STATUS_IO_ERROR);
	}
	status = SparkStageModulePackRead(SPARK_QWEN36_MODULE_TAG,file,0u,&header,sizeof(header));
	if ( status == SPARK_STATUS_OK )
	{
		SparkQwen36StagePackExpectedGeometry(&expected,state->first_layer_index,state->layer_count);
		compare = SparkQwen36StagePackCompareGeometry(&header,&expected);
		if ( compare != 0 || header.directory_offset != SPARK_QWEN36_STAGEPACK_HEADER_BYTES )
		{
			fprintf(stderr,"%s pack_geometry_mismatch field=%s\n",SPARK_QWEN36_MODULE_TAG,compare != 0 ? SparkQwen36StagePackGeometryFieldName(compare) : "directory_offset");
			status = SPARK_STATUS_VALIDATION_FAILED;
		}
	}
	directory = status == SPARK_STATUS_OK ? (SparkQwen36StagePackEntry *)malloc((size_t)header.tensor_count * sizeof(SparkQwen36StagePackEntry)) : 0;
	if ( status == SPARK_STATUS_OK && directory == 0 )
		status = SPARK_STATUS_CAPACITY_EXCEEDED;
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModulePackRead(SPARK_QWEN36_MODULE_TAG,file,header.directory_offset,directory,(uint64_t)header.tensor_count * sizeof(SparkQwen36StagePackEntry));
	for (index = 0; status == SPARK_STATUS_OK && index < header.tensor_count; index++)
		status = SparkQwen36ModuleLoadEntry(state,file,&directory[index],header.file_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkQwen36ModuleVerifyCoverage(state);
	free(directory);
	fclose(file);
	return(status);
}

static SparkStatus SparkQwen36ModuleAllocatePools(SparkQwen36ModuleState *state)
{
	SparkStatus status = SPARK_STATUS_OK;
	uint64_t state_elements,tail_elements,cache_elements;
	state->gdn_pool.abi_version = SPARK_QWEN36_RESIDENT_DECODE_STAGE_GDN_STATE_POOL_ABI_VERSION;
	state->gdn_pool.lane_capacity = state->max_active_sequence_count;
	state->gdn_pool.gdn_layer_count = state->gdn_layer_count;
	state->gdn_pool.state_layer_stride_elements = (uint64_t)SPARK_QWEN36_MODEL_GDN_VALUE_HEAD_COUNT * SPARK_QWEN36_MODEL_GDN_HEAD_KEY_DIMENSION * SPARK_QWEN36_MODEL_GDN_HEAD_VALUE_DIMENSION;
	state->gdn_pool.state_lane_stride_elements = state->gdn_pool.state_layer_stride_elements * state->gdn_layer_count;
	state->gdn_pool.conv_tail_layer_stride_elements = (uint64_t)SPARK_QWEN36_MODEL_GDN_CONV_CHANNELS * (SPARK_QWEN36_MODEL_GDN_CONV_KERNEL - 1u);
	state->gdn_pool.conv_tail_lane_stride_elements = state->gdn_pool.conv_tail_layer_stride_elements * state->gdn_layer_count;
	if ( state->gdn_layer_count != 0u )
	{
		state_elements = state->gdn_pool.state_lane_stride_elements * state->max_active_sequence_count;
		tail_elements = state->gdn_pool.conv_tail_lane_stride_elements * state->max_active_sequence_count;
		status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,state_elements * sizeof(float),(void **)&state->gdn_pool.state_f32);
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,tail_elements * SPARK_QWEN36_MODEL_BF16_ELEMENT_BYTES,&state->gdn_pool.conv_tail_bf16);
	}
	if ( status == SPARK_STATUS_OK && state->attn_layer_count != 0u )
	{
		state->cache_layer_stride = (uint64_t)SPARK_QWEN36_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS * SPARK_QWEN36_MODEL_ATTN_CACHE_TOKEN_ELEMENTS;
		state->cache_block_stride = state->cache_layer_stride * state->attn_layer_count;
		cache_elements = state->cache_block_stride * state->kv_block_count;
		status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,cache_elements * SPARK_QWEN36_MODEL_BF16_ELEMENT_BYTES,&state->kv_cache_bf16);
	}
	return(status);
}

static SparkStatus SparkQwen36ModuleAllocateSlotControl(SparkQwen36ModuleState *state, SparkQwen36ModuleSlot *slot)
{
	uint64_t rows = state->max_active_sequence_count;
	SparkStatus status;
	cudaStream_t stream = 0;
	status = SparkStageModuleCudaStatus(SPARK_QWEN36_MODULE_TAG,cudaStreamCreate(&stream),"cudaStreamCreate");
	if ( status != SPARK_STATUS_OK )
		return(status);
	slot->cuda_stream = stream;
	status = SparkStageModuleDeviceAllocate(&state->ledger,rows * sizeof(uint32_t),(void **)&slot->input_token_ids);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * sizeof(uint32_t),(void **)&slot->output_token_ids);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * sizeof(uint32_t),(void **)&slot->row_lane_indices);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * sizeof(uint32_t),(void **)&slot->slot_mapping);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * sizeof(uint32_t),(void **)&slot->context_lengths);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * sizeof(uint32_t),(void **)&slot->row_cold);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * sizeof(uint64_t),(void **)&slot->row_positions);
	return(status);
}

static SparkStatus SparkQwen36ModuleAllocateSlot(SparkQwen36ModuleState *state, SparkQwen36ModuleSlot *slot)
{
	uint64_t rows = state->max_active_sequence_count;
	SparkStatus status = SparkQwen36ModuleAllocateSlotControl(state,slot);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_QWEN36_MODEL_HIDDEN_DIMENSION * SPARK_QWEN36_MODEL_BF16_ELEMENT_BYTES,&slot->hidden_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_QWEN36_MODEL_HIDDEN_DIMENSION * SPARK_QWEN36_MODEL_BF16_ELEMENT_BYTES,&slot->normalized_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_QWEN36_MODEL_HIDDEN_DIMENSION * SPARK_QWEN36_MODEL_BF16_ELEMENT_BYTES,&slot->delta_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_QWEN36_MODEL_GDN_CONV_CHANNELS * SPARK_QWEN36_MODEL_BF16_ELEMENT_BYTES,&slot->qkv_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_QWEN36_MODEL_GDN_CONV_CHANNELS * SPARK_QWEN36_MODEL_BF16_ELEMENT_BYTES,&slot->conv_out_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_QWEN36_MODEL_GDN_VALUE_DIMENSION * SPARK_QWEN36_MODEL_BF16_ELEMENT_BYTES,&slot->z_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_QWEN36_MODEL_GDN_VALUE_HEAD_COUNT * SPARK_QWEN36_MODEL_BF16_ELEMENT_BYTES,&slot->beta_pre_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_QWEN36_MODEL_GDN_VALUE_HEAD_COUNT * SPARK_QWEN36_MODEL_BF16_ELEMENT_BYTES,&slot->decay_pre_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_QWEN36_MODEL_GDN_VALUE_HEAD_COUNT * sizeof(float),(void **)&slot->log_decay_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_QWEN36_MODEL_GDN_VALUE_HEAD_COUNT * sizeof(float),(void **)&slot->beta_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_QWEN36_MODEL_GDN_VALUE_DIMENSION * SPARK_QWEN36_MODEL_BF16_ELEMENT_BYTES,&slot->core_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_QWEN36_MODEL_GDN_VALUE_DIMENSION * SPARK_QWEN36_MODEL_BF16_ELEMENT_BYTES,&slot->gated_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * 2u * SPARK_QWEN36_MODEL_ATTN_QUERY_DIMENSION * SPARK_QWEN36_MODEL_BF16_ELEMENT_BYTES,&slot->q_fused_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_QWEN36_MODEL_ATTN_KV_DIMENSION * SPARK_QWEN36_MODEL_BF16_ELEMENT_BYTES,&slot->k_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_QWEN36_MODEL_ATTN_KV_DIMENSION * SPARK_QWEN36_MODEL_BF16_ELEMENT_BYTES,&slot->v_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_QWEN36_MODEL_ATTN_QUERY_DIMENSION * SPARK_QWEN36_MODEL_BF16_ELEMENT_BYTES,&slot->head_out_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_QWEN36_MODEL_FFN_INTERMEDIATE_DIMENSION * SPARK_QWEN36_MODEL_BF16_ELEMENT_BYTES,&slot->ffn_gate_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_QWEN36_MODEL_FFN_INTERMEDIATE_DIMENSION * SPARK_QWEN36_MODEL_BF16_ELEMENT_BYTES,&slot->ffn_up_bf16);
	return(status);
}

static SparkStatus SparkQwen36ModuleRunGdnLayer(SparkQwen36ModuleState *state, SparkQwen36ModuleSlot *slot, uint32_t layer, uint32_t rows)
{
	const SparkQwen36GdnLayerWeights *weights = &state->gdn_by_layer[layer];
	uint32_t ordinal = state->gdn_ordinal_by_layer[layer];
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	cudaError_t error;
	error = SparkQwen36LaunchLinear(stream,&weights->qkv,slot->normalized_bf16,slot->qkv_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkQwen36LaunchLinear(stream,&weights->gate,slot->normalized_bf16,slot->z_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkQwen36LaunchLinear(stream,&weights->beta,slot->normalized_bf16,slot->beta_pre_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkQwen36LaunchLinear(stream,&weights->decay,slot->normalized_bf16,slot->decay_pre_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkQwen36LaunchConvUpdate(stream,slot->qkv_bf16,weights,slot->conv_out_bf16,&state->gdn_pool,slot->row_lane_indices,rows,ordinal);
	if ( error == cudaSuccess )
		error = SparkQwen36LaunchDecayBeta(stream,slot->decay_pre_bf16,slot->beta_pre_bf16,weights,slot->log_decay_f32,slot->beta_f32,rows);
	if ( error == cudaSuccess )
		error = SparkQwen36LaunchGdnStep(stream,slot->conv_out_bf16,slot->log_decay_f32,slot->beta_f32,&state->gdn_pool,slot->core_bf16,slot->row_lane_indices,rows,ordinal);
	if ( error == cudaSuccess )
		error = SparkQwen36LaunchGatedNorm(stream,slot->core_bf16,slot->z_bf16,weights,slot->gated_bf16,rows,SPARK_QWEN36_MODEL_RMS_NORM_EPSILON);
	if ( error == cudaSuccess )
		error = SparkQwen36LaunchLinear(stream,&weights->output,slot->gated_bf16,slot->delta_bf16,rows);
	return(SparkStageModuleCudaStatus(SPARK_QWEN36_MODULE_TAG,error,"gdn_layer"));
}

static SparkStatus SparkQwen36ModuleRunAttnLayer(SparkQwen36ModuleState *state, SparkQwen36ModuleSlot *slot, const SparkQwen36KvBlockTableView *table, uint32_t layer, uint32_t rows)
{
	const SparkQwen36AttnLayerWeights *weights = &state->attn_by_layer[layer];
	uint32_t ordinal = state->attn_ordinal_by_layer[layer];
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	cudaError_t error;
	error = SparkQwen36LaunchLinear(stream,&weights->query,slot->normalized_bf16,slot->q_fused_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkQwen36LaunchLinear(stream,&weights->key,slot->normalized_bf16,slot->k_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkQwen36LaunchLinear(stream,&weights->value,slot->normalized_bf16,slot->v_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkQwen36LaunchAttnPrepare(stream,slot->q_fused_bf16,slot->k_bf16,slot->v_bf16,weights,state->kv_cache_bf16,slot->slot_mapping,slot->row_positions,rows,ordinal,state->cache_layer_stride,state->cache_block_stride,SPARK_QWEN36_MODEL_RMS_NORM_EPSILON);
	if ( error == cudaSuccess )
		error = SparkQwen36LaunchAttnDecode(stream,slot->q_fused_bf16,state->kv_cache_bf16,table,slot->row_lane_indices,slot->context_lengths,slot->head_out_bf16,rows,ordinal,state->cache_layer_stride,state->cache_block_stride);
	if ( error == cudaSuccess )
		error = SparkQwen36LaunchLinear(stream,&weights->output,slot->head_out_bf16,slot->delta_bf16,rows);
	return(SparkStageModuleCudaStatus(SPARK_QWEN36_MODULE_TAG,error,"attn_layer"));
}

static SparkStatus SparkQwen36ModuleRunFfn(SparkQwen36ModuleState *state, SparkQwen36ModuleSlot *slot, uint32_t layer, uint32_t rows)
{
	const SparkQwen36FfnLayerWeights *weights = &state->ffn_by_layer[layer];
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	cudaError_t error;
	error = SparkQwen36LaunchRmsNorm(stream,slot->hidden_bf16,state->mlp_norm_by_layer[layer],slot->normalized_bf16,rows,SPARK_QWEN36_MODEL_HIDDEN_DIMENSION,SPARK_QWEN36_MODEL_RMS_NORM_EPSILON);
	if ( error == cudaSuccess )
		error = SparkQwen36LaunchLinear(stream,&weights->gate,slot->normalized_bf16,slot->ffn_gate_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkQwen36LaunchLinear(stream,&weights->up,slot->normalized_bf16,slot->ffn_up_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkQwen36LaunchSwiGlu(stream,slot->ffn_gate_bf16,slot->ffn_up_bf16,rows,SPARK_QWEN36_MODEL_FFN_INTERMEDIATE_DIMENSION);
	if ( error == cudaSuccess )
		error = SparkQwen36LaunchLinear(stream,&weights->down,slot->ffn_up_bf16,slot->delta_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkQwen36LaunchResidualAdd(stream,slot->hidden_bf16,slot->delta_bf16,rows,SPARK_QWEN36_MODEL_HIDDEN_DIMENSION);
	return(SparkStageModuleCudaStatus(SPARK_QWEN36_MODULE_TAG,error,"ffn"));
}

static SparkStatus SparkQwen36ModuleRunLayer(SparkQwen36ModuleState *state, SparkQwen36ModuleSlot *slot, const SparkQwen36KvBlockTableView *table, uint32_t layer, uint32_t rows)
{
	SparkStatus status;
	cudaError_t error = SparkQwen36LaunchRmsNorm((cudaStream_t)slot->cuda_stream,slot->hidden_bf16,state->attention_norm_by_layer[layer],slot->normalized_bf16,rows,SPARK_QWEN36_MODEL_HIDDEN_DIMENSION,SPARK_QWEN36_MODEL_RMS_NORM_EPSILON);
	status = SparkStageModuleCudaStatus(SPARK_QWEN36_MODULE_TAG,error,"attention_norm");
	if ( status == SPARK_STATUS_OK )
		status = SPARK_QWEN36_MODEL_LAYER_IS_GDN(layer) != 0u ? SparkQwen36ModuleRunGdnLayer(state,slot,layer,rows) : SparkQwen36ModuleRunAttnLayer(state,slot,table,layer,rows);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleCudaStatus(SPARK_QWEN36_MODULE_TAG,SparkQwen36LaunchResidualAdd((cudaStream_t)slot->cuda_stream,slot->hidden_bf16,slot->delta_bf16,rows,SPARK_QWEN36_MODEL_HIDDEN_DIMENSION),"residual");
	if ( status == SPARK_STATUS_OK )
		status = SparkQwen36ModuleRunFfn(state,slot,layer,rows);
	return(status);
}

static SparkStatus SparkQwen36ModuleValidateFrame(SparkQwen36ModuleState *state, const SparkModelDriverFrame *frame, const SparkQwen36ResidentDecodeStageFrameContext **context_out)
{
	const SparkQwen36ResidentDecodeStageFrameContext *context;
	uint32_t needs_input = state->stage_index > 0u ? 1u : 0u,needs_output = state->stage_index + 1u < state->stage_count ? 1u : 0u;
	uint32_t expected_buffers = state->owns_embedding + state->owns_final_head;
	if ( frame == 0 || frame->user_context == 0 || frame->buffer_count != expected_buffers )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	context = (const SparkQwen36ResidentDecodeStageFrameContext *)frame->user_context;
	if ( context->abi_version != SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION || context->descriptor_bytes != (uint32_t)sizeof(*context) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( (context->flags & SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_DECODE_BATCH_VIEW) == 0u || context->decode_batch == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( ((context->flags & SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_INPUT_TRANSPORT) != 0u) != (needs_input != 0u) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( ((context->flags & SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_OUTPUT_TRANSPORT) != 0u) != (needs_output != 0u) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( state->attn_layer_count != 0u && ((context->flags & SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_KV_BLOCK_TABLE) == 0u || context->kv_block_table == 0 || context->kv_block_table->host_physical_block_indices == 0 || context->kv_block_table->host_lane_physical_block_counts == 0) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( context->decode_batch->row_count == 0u || context->decode_batch->row_count > state->max_active_sequence_count || context->decode_batch->row_count != frame->new_token_count )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	*context_out = context;
	return(SPARK_STATUS_OK);
}

/*
 * Host staging for one decode microbatch: distinct-lane check, cold flags
 * from the lane warm map, and for attention stages the physical slot and
 * context length per row proven against the host block-table mirrors. Any
 * uncovered position is a refused frame, never a stray cache write.
 */
static SparkStatus SparkQwen36ModuleStageRows(SparkQwen36ModuleState *state, const SparkQwen36ResidentDecodeStageFrameContext *context, uint8_t *lane_used)
{
	const SparkQwen36DecodeBatchView *batch = context->decode_batch;
	const SparkQwen36KvBlockTableView *table = context->kv_block_table;
	uint32_t row,lane,block_ordinal,block;
	uint64_t position;
	for (row = 0; row < batch->row_count; row++)
	{
		lane = batch->row_lane_indices[row];
		position = batch->row_positions[row];
		if ( lane >= state->max_active_sequence_count || lane_used[lane] != 0u )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		lane_used[lane] = 1u;
		state->host_row_cold[row] = state->lane_warm[lane] != 0u ? 0u : 1u;
		if ( state->attn_layer_count == 0u )
			continue;
		if ( position + 1u > (uint64_t)table->lane_stride * SPARK_QWEN36_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		block_ordinal = (uint32_t)(position / SPARK_QWEN36_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS);
		if ( lane >= table->lane_count || block_ordinal >= table->host_lane_physical_block_counts[lane] )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		block = table->host_physical_block_indices[((uint64_t)lane * table->lane_stride) + block_ordinal];
		if ( block == SPARK_QWEN36_RESIDENT_DECODE_STAGE_NO_BLOCK || block >= state->kv_block_count )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		state->host_slot_mapping[row] = (block * SPARK_QWEN36_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS) + (uint32_t)(position % SPARK_QWEN36_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS);
		state->host_context_lengths[row] = (uint32_t)(position + 1u);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkQwen36ModuleUploadRows(SparkQwen36ModuleState *state, SparkQwen36ModuleSlot *slot, const SparkQwen36ResidentDecodeStageFrameContext *context, const SparkModelDriverFrame *frame, uint32_t rows)
{
	const SparkQwen36DecodeBatchView *batch = context->decode_batch;
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	cudaError_t error;
	error = cudaMemcpyAsync(slot->row_lane_indices,batch->row_lane_indices,rows * sizeof(uint32_t),cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess )
		error = cudaMemcpyAsync(slot->row_positions,batch->row_positions,rows * sizeof(uint64_t),cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess )
		error = cudaMemcpyAsync(slot->row_cold,state->host_row_cold,rows * sizeof(uint32_t),cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess && state->attn_layer_count != 0u )
		error = cudaMemcpyAsync(slot->slot_mapping,state->host_slot_mapping,rows * sizeof(uint32_t),cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess && state->attn_layer_count != 0u )
		error = cudaMemcpyAsync(slot->context_lengths,state->host_context_lengths,rows * sizeof(uint32_t),cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess && state->owns_embedding != 0u )
		error = cudaMemcpyAsync(slot->input_token_ids,frame->buffers[0].address,rows * sizeof(uint32_t),cudaMemcpyHostToDevice,stream);
	return(SparkStageModuleCudaStatus(SPARK_QWEN36_MODULE_TAG,error,"stage_upload"));
}

static SparkStatus SparkQwen36ModuleBeginHidden(SparkQwen36ModuleState *state, SparkQwen36ModuleSlot *slot, SparkQwen36ResidentDecodeStageFrameContext *context, uint32_t rows)
{
	SparkStatus status;
	cudaError_t error;
	if ( state->owns_embedding != 0u )
	{
		error = SparkQwen36LaunchEmbeddingGather((cudaStream_t)slot->cuda_stream,slot->input_token_ids,state->token_embedding_bf16,slot->hidden_bf16,rows);
		return(SparkStageModuleCudaStatus(SPARK_QWEN36_MODULE_TAG,error,"embedding"));
	}
	if ( context->hidden_input_post_receive_function == 0 || context->hidden_input_transport_session == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	status = context->hidden_input_post_receive_function(context->hidden_input_transport_session,&context->hidden_input_packet);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( context->hidden_input_packet.active_sequence_count != rows || context->hidden_input_packet.hidden_dimension != SPARK_QWEN36_MODEL_HIDDEN_DIMENSION || context->hidden_input_packet.hidden_bf16 == 0 )
		return(SPARK_STATUS_VALIDATION_FAILED);
	error = cudaMemcpyAsync(slot->hidden_bf16,context->hidden_input_packet.hidden_bf16,(uint64_t)rows * SPARK_QWEN36_MODEL_HIDDEN_DIMENSION * SPARK_QWEN36_MODEL_BF16_ELEMENT_BYTES,cudaMemcpyDeviceToDevice,(cudaStream_t)slot->cuda_stream);
	return(SparkStageModuleCudaStatus(SPARK_QWEN36_MODULE_TAG,error,"hidden_receive"));
}

static SparkStatus SparkQwen36ModuleFinish(SparkQwen36ModuleState *state, SparkQwen36ModuleSlot *slot, SparkQwen36ResidentDecodeStageFrameContext *context, SparkModelDriverFrame *frame, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	cudaError_t error = cudaSuccess;
	SparkStatus status = SPARK_STATUS_OK;
	uint32_t out_index = state->owns_embedding != 0u ? 1u : 0u;
	if ( state->owns_final_head != 0u )
	{
		error = SparkQwen36LaunchRmsNorm(stream,slot->hidden_bf16,state->final_norm_weight_bf16,slot->normalized_bf16,rows,SPARK_QWEN36_MODEL_HIDDEN_DIMENSION,SPARK_QWEN36_MODEL_RMS_NORM_EPSILON);
		if ( error == cudaSuccess )
			error = SparkQwen36LaunchHeadArgmax(stream,slot->normalized_bf16,state->lm_head_weight_bf16,0,slot->output_token_ids,rows,SPARK_QWEN36_MODEL_OUTPUT_VOCAB_COUNT);
		if ( error == cudaSuccess )
			error = cudaMemcpyAsync(frame->buffers[out_index].address,slot->output_token_ids,rows * sizeof(uint32_t),cudaMemcpyDeviceToHost,stream);
		status = SparkStageModuleCudaStatus(SPARK_QWEN36_MODULE_TAG,error,"head");
	}
	else
	{
		if ( context->hidden_output_send_function == 0 || context->hidden_output_transport_session == 0 )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		context->hidden_output_packet.active_sequence_count = rows;
		context->hidden_output_packet.hidden_dimension = SPARK_QWEN36_MODEL_HIDDEN_DIMENSION;
		context->hidden_output_packet.bytes_per_sequence = SPARK_QWEN36_MODEL_HIDDEN_DIMENSION * SPARK_QWEN36_MODEL_BF16_ELEMENT_BYTES;
		context->hidden_output_packet.hidden_bf16 = slot->hidden_bf16;
		context->hidden_output_packet.cuda_stream = stream;
		context->hidden_output_packet.sideband_payload = 0;
		context->hidden_output_packet.sideband_kind = 0u;
		context->hidden_output_packet.sideband_bytes_per_sequence = 0u;
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleCudaStatus(SPARK_QWEN36_MODULE_TAG,cudaStreamSynchronize(stream),"sync");
	if ( status == SPARK_STATUS_OK && state->owns_final_head == 0u )
		status = context->hidden_output_send_function(context->hidden_output_transport_session,&context->hidden_output_packet);
	return(status);
}

static uint64_t SparkQwen36ModuleFingerprint(const void *bytes, uint64_t count, uint64_t basis)
{
	const uint8_t *data = (const uint8_t *)bytes;
	uint64_t hash = basis,index;
	for (index = 0; index < count; index++)
		hash = (hash ^ data[index]) * 1099511628211ull;
	return(hash);
}

static SparkStatus SparkQwen36ModuleOpenKvTier(SparkQwen36ModuleState *state)
{
	SparkQwen36StagePackHeader geometry;
	const char *provider = 0,*service = 0,*socket_path = 0;
	uint64_t pool_bytes = 0u,model_fp,layout_fp,layout_bits[3];
	uint32_t workers = 0u;
	SparkStatus status = SparkStageModuleEnvironmentText(SPARK_QWEN36_MODULE_TAG,"SPARK_QWEN36_STAGE_KV_STORE",&provider);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( strcmp(provider,"none") == 0 )
		return(SparkStageKvClientOpen(&state->kv_client,SPARK_QWEN36_MODULE_TAG,provider,0u,0u,0u,0u,0u,0,0,0u,0u));
	status = SparkStageModuleEnvironmentText(SPARK_QWEN36_MODULE_TAG,"SPARK_QWEN36_STAGE_KV_SERVICE",&service);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentText(SPARK_QWEN36_MODULE_TAG,"SPARK_QWEN36_STAGE_KV_SOCKET",&socket_path);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned64(SPARK_QWEN36_MODULE_TAG,"SPARK_QWEN36_STAGE_KV_POOL_BYTES",1u,1ull << 40u,&pool_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_QWEN36_MODULE_TAG,"SPARK_QWEN36_STAGE_KV_WORKERS",1u,64u,&workers);
	if ( status != SPARK_STATUS_OK )
		return(status);
	SparkQwen36StagePackExpectedGeometry(&geometry,state->first_layer_index,state->layer_count);
	model_fp = SparkQwen36ModuleFingerprint(&geometry,sizeof(geometry),14695981039346656037ull);
	layout_bits[0] = state->cache_layer_stride;
	layout_bits[1] = state->cache_block_stride;
	layout_bits[2] = SPARK_QWEN36_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS;
	layout_fp = SparkQwen36ModuleFingerprint(layout_bits,sizeof(layout_bits),model_fp);
	return(SparkStageKvClientOpen(&state->kv_client,SPARK_QWEN36_MODULE_TAG,provider,state->stage_index,state->first_layer_index,state->layer_count,model_fp,layout_fp,service,socket_path,pool_bytes,workers));
}

SparkStatus SparkQwen36ResidentDecodeStageExecute(void *module_state, SparkModelDriverFrame *frame)
{
	SparkQwen36ModuleState *state = (SparkQwen36ModuleState *)module_state;
	SparkQwen36ResidentDecodeStageFrameContext *context = 0;
	uint8_t lane_used[SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	SparkQwen36ModuleSlot *slot;
	uint32_t slot_index = 0u,rows,layer,row;
	SparkStatus status;
	if ( state == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkQwen36ModuleValidateFrame(state,frame,(const SparkQwen36ResidentDecodeStageFrameContext **)&context);
	if ( status != SPARK_STATUS_OK )
		return(status);
	rows = context->decode_batch->row_count;
	memset(lane_used,0,sizeof(lane_used));
	status = SparkQwen36ModuleStageRows(state,context,lane_used);
	if ( status != SPARK_STATUS_OK )
		return(status);
	status = SparkStageModuleSlotClaim(state->slot_states,state->pipeline_slot_count,&slot_index);
	if ( status != SPARK_STATUS_OK )
		return(status);
	slot = &state->slots[slot_index];
	status = SparkQwen36ModuleUploadRows(state,slot,context,frame,rows);
	if ( status == SPARK_STATUS_OK )
		status = SparkQwen36ModuleBeginHidden(state,slot,context,rows);
	for (layer = state->first_layer_index; status == SPARK_STATUS_OK && layer < state->first_layer_index + state->layer_count; layer++)
		status = SparkQwen36ModuleRunLayer(state,slot,context->kv_block_table,layer,rows);
	if ( status == SPARK_STATUS_OK )
		status = SparkQwen36ModuleFinish(state,slot,context,frame,rows);
	SparkStageModuleSlotRelease(state->slot_states,slot_index);
	if ( status != SPARK_STATUS_OK )
		return(status);
	for (row = 0; row < rows; row++)
		state->lane_warm[context->decode_batch->row_lane_indices[row]] = 1u;
	atomic_fetch_add(&state->frames_executed,1u);
	atomic_fetch_add(&state->tokens_emitted,rows);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkQwen36ResidentDecodeStageAdmit(void *module_state, const SparkModelDriverAdmissionRequest *request, SparkModelDriverAdmissionDecision *decision)
{
	SparkQwen36ModuleState *state = (SparkQwen36ModuleState *)module_state;
	if ( state == 0 || request == 0 || decision == 0 || request->descriptor_bytes != (uint32_t)sizeof(*request) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(decision,0,sizeof(*decision));
	decision->descriptor_bytes = (uint32_t)sizeof(*decision);
	decision->accepted = 1u;
	decision->available_dispatch_slot_count = state->pipeline_slot_count;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkQwen36ResidentDecodeStageSnapshot(void *module_state, uint32_t program_id, SparkModelDriverRuntimeSnapshot *snapshot)
{
	SparkQwen36ModuleState *state = (SparkQwen36ModuleState *)module_state;
	if ( state == 0 || snapshot == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(snapshot,0,sizeof(*snapshot));
	snapshot->descriptor_bytes = (uint32_t)sizeof(*snapshot);
	snapshot->program_id = program_id;
	snapshot->submitted_count = atomic_load(&state->frames_executed);
	snapshot->completed_count = atomic_load(&state->frames_executed);
	snapshot->resident_token_count = atomic_load(&state->tokens_emitted);
	snapshot->kv_token_capacity = (uint64_t)state->kv_block_count * SPARK_QWEN36_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS;
	snapshot->available_dispatch_slot_count = state->pipeline_slot_count;
	return(SPARK_STATUS_OK);
}

void SparkQwen36ResidentDecodeStageDestroy(void *module_state)
{
	SparkQwen36ModuleState *state = (SparkQwen36ModuleState *)module_state;
	uint32_t slot_index;
	if ( state == 0 )
		return;
	for (slot_index = 0; slot_index < state->pipeline_slot_count; slot_index++)
		if ( state->slots[slot_index].cuda_stream != 0 )
			cudaStreamDestroy((cudaStream_t)state->slots[slot_index].cuda_stream);
	SparkStageKvClientClose(&state->kv_client);
	SparkStageModuleLedgerRelease(&state->ledger);
	free(state);
}

SparkStatus SparkQwen36ResidentDecodeStageInitialize(const SparkFirmwareModuleConfiguration *configuration, const SparkFirmwareModuleHostServices *host_services, void **module_state)
{
	SparkQwen36ModuleState *state;
	const char *pack_path = 0;
	SparkStatus status;
	uint32_t slot_index;
	(void)host_services;
	if ( configuration == 0 || module_state == 0 || configuration->abi_version != SPARK_FIRMWARE_MODULE_ABI_VERSION )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	state = (SparkQwen36ModuleState *)calloc(1u,sizeof(*state));
	if ( state == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	state->ledger.module_tag = SPARK_QWEN36_MODULE_TAG;
	status = SparkQwen36ModuleConfigure(state);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentText(SPARK_QWEN36_MODULE_TAG,"SPARK_QWEN36_STAGE_PACK_PATH",&pack_path);
	if ( status == SPARK_STATUS_OK )
	{
		SparkQwen36ModuleBuildOrdinals(state);
		status = SparkQwen36ModuleLoadPack(state,pack_path);
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkQwen36ModuleAllocatePools(state);
	if ( status == SPARK_STATUS_OK )
		status = SparkQwen36ModuleOpenKvTier(state);
	for (slot_index = 0; status == SPARK_STATUS_OK && slot_index < state->pipeline_slot_count; slot_index++)
		status = SparkQwen36ModuleAllocateSlot(state,&state->slots[slot_index]);
	if ( status != SPARK_STATUS_OK )
	{
		SparkQwen36ResidentDecodeStageDestroy(state);
		return(status);
	}
	fprintf(stderr,"%s ready stage=%u/%u slice=%u+%u gdn=%u attn=%u lanes=%u kv_blocks=%u device_gib=%.1f\n",SPARK_QWEN36_MODULE_TAG,state->stage_index,state->stage_count,state->first_layer_index,state->layer_count,state->gdn_layer_count,state->attn_layer_count,state->max_active_sequence_count,state->kv_block_count,(double)state->ledger.device_bytes_resident / (1024.0 * 1024.0 * 1024.0));
	*module_state = state;
	return(SPARK_STATUS_OK);
}
