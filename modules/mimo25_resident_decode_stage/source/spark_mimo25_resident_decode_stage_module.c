/* Large stage packs exceed 2 GB: 64-bit file offsets are required. */
#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64

#include <math.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>

#include "sparkpipe/spark_mimo25_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_stage_module_common.h"
#include "spark_mimo25_stagepack_format.h"

/*
 * MiMo-V2.5 resident decode stage host module, PP-Nx native, one variant
 * per build through the -include'd model header. The family substrate
 * discipline holds: environment-only required configuration, ledgered
 * device allocation, staged pack loading, CAS pipeline slots.
 *
 * Execute serves DECODE frames: one token per lane, both attention
 * branches on their own k/v pools (full layers keep the dense per-lane
 * history bounded by SPARK_MIMO25_STAGE_MAX_SEQ; SWA layers keep the
 * 128-slot ring and carry the sink), the fused qkv sliced in place -
 * value scale folded before the cache write, rope on the q and k slices
 * with the branch's theta - and the FFN dispatched dense or MoE by the
 * layer table. Prefill refuses with prefill_pending; MTP arms to zero
 * only, its three draft layers loading and verifying so the pack
 * contract is final while execution rides the family MTP pass. Routed
 * experts run per (row, rank) through stacked-view slices after a host
 * readback - the correctness-first form, expert batching scheduled with
 * the wmma pass.
 */

#define SPARK_MIMO25_MODULE_TAG "mimo25_stage"

typedef struct SparkMimo25LayerWeights
{
	SparkMimo25AttnWeights attn;
	const void *ffn_norm_bf16;
	SparkMimo25DenseWeights dense;
	SparkMimo25MoeWeights moe;
} SparkMimo25LayerWeights;

typedef struct SparkMimo25ModuleSlot
{
	void *cuda_stream;
	uint32_t *input_token_ids;
	uint32_t *output_token_ids;
	uint32_t *row_lane_indices;
	uint64_t *row_positions;
	void *hidden_bf16;
	void *normalized_bf16;
	void *qkv_bf16;
	void *attn_out_bf16;
	void *delta_bf16;
	float *moe_scores_f32;
	uint32_t *moe_indices_u32;
	float *moe_weights_f32;
	void *ffn_gate_bf16;
	void *ffn_up_bf16;
	void *ffn_delta_bf16;
	void *ffn_accum_bf16;
	uint64_t *mtp_positions_u64;
	uint32_t *mtp_step_ids_u32;
	uint32_t *mtp_draft_ids_u32;
	void *mtp_concat_bf16;
	void *mtp_hidden_bf16;
	void *mtp_scratch_bf16;
	uint64_t *host_mtp_positions;
	uint32_t *grouped_rows_u32;
	uint32_t *grouped_weight_slots_u32;
	void *moe_slot_gate_bf16;
	void *moe_slot_up_bf16;
	void *moe_slot_out_bf16;
	void *head_logits_bf16;
	uint32_t *host_moe_indices;
	uint32_t *host_grouped_rows;
	uint32_t *host_grouped_weight_slots;
	uint32_t host_expert_offsets[SPARK_MIMO25_MODEL_ROUTED_EXPERT_COUNT + 1u];
} SparkMimo25ModuleSlot;

typedef struct SparkMimo25ModuleState
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
	uint32_t max_sequence_positions;
	uint32_t mtp_armed;
	uint32_t full_layer_count;
	uint32_t swa_layer_count;
	uint32_t full_ordinal_by_layer[SPARK_MIMO25_RESIDENT_DECODE_STAGE_MAX_LAYER_COUNT];
	uint32_t swa_ordinal_by_layer[SPARK_MIMO25_RESIDENT_DECODE_STAGE_MAX_LAYER_COUNT];
	uint32_t layer_seen_bits[SPARK_MIMO25_RESIDENT_DECODE_STAGE_MAX_LAYER_COUNT];
	uint32_t mtp_seen_bits[SPARK_MIMO25_MODEL_MTP_LAYER_COUNT];
	uint32_t global_seen_bits;
	SparkMimo25LayerWeights layers[SPARK_MIMO25_RESIDENT_DECODE_STAGE_MAX_LAYER_COUNT];
	SparkMimo25MtpLayerWeights mtp_layers[SPARK_MIMO25_MODEL_MTP_LAYER_COUNT];
	const void *token_embedding_bf16;
	const void *final_norm_weight_bf16;
	const void *lm_head_weight_bf16;
	float *full_freqs_f32;
	float *swa_freqs_f32;
	void *k_full_cache_bf16;
	void *v_full_cache_bf16;
	void *k_swa_cache_bf16;
	void *v_swa_cache_bf16;
	void *k_mtp_cache_bf16;
	void *v_mtp_cache_bf16;
	uint64_t k_full_lane_stride;
	uint64_t v_full_lane_stride;
	uint64_t k_swa_lane_stride;
	uint64_t v_swa_lane_stride;
	SparkMimo25ModuleSlot slots[SPARK_MIMO25_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
	atomic_uint slot_states[SPARK_MIMO25_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
	atomic_ullong frames_executed;
	atomic_ullong tokens_emitted;
} SparkMimo25ModuleState;

extern cudaError_t SparkMimo25LaunchRmsNorm(cudaStream_t stream, const void *input_bf16, const void *gain_bf16, void *output_bf16, uint32_t row_count, uint32_t dimension, float epsilon);
extern cudaError_t SparkMimo25LaunchLinear(cudaStream_t stream, const SparkMimo25LinearView *view, const void *payload, const void *scale, const void *input_bf16, void *output_bf16, uint32_t row_count);
extern cudaError_t SparkMimo25LaunchEmbeddingGather(cudaStream_t stream, const uint32_t *token_ids, const void *embedding_bf16, void *hidden_bf16, uint32_t row_count, uint32_t hidden_dimension);
extern cudaError_t SparkMimo25LaunchHeadArgmax(cudaStream_t stream, const void *hidden_bf16, const void *head_weight_bf16, uint32_t *output_token_ids, uint32_t row_count, uint32_t candidate_count, uint32_t hidden_dimension);
extern cudaError_t SparkMimo25LaunchRope(cudaStream_t stream, void *data_bf16, uint64_t row_stride, uint32_t row_offset, const float *freqs_f32, const uint64_t *row_positions, uint32_t row_count, uint32_t head_count, uint32_t head_dim, uint32_t rope_dim, uint32_t inverse);
extern cudaError_t SparkMimo25LaunchScaleSlice(cudaStream_t stream, void *data_bf16, uint64_t row_stride, uint32_t column_offset, uint32_t width, float scale, uint32_t row_count);
extern cudaError_t SparkMimo25LaunchAttnDecode(cudaStream_t stream, const void *q_bf16, uint64_t q_row_stride, const void *k_cache_bf16, const void *v_cache_bf16, uint64_t k_lane_stride, uint64_t v_lane_stride, uint64_t k_slot_stride, uint64_t v_slot_stride, const uint32_t *row_lane_indices, const uint64_t *row_positions, const float *sink_f32, float scale, void *out_bf16, uint32_t row_count, uint32_t head_count, uint32_t group_size, uint32_t head_dim, uint32_t value_dim, uint32_t window_slots);
extern cudaError_t SparkMimo25LaunchGateScores(cudaStream_t stream, const SparkMimo25LinearView *gate, const void *input_bf16, float *scores_f32, uint32_t row_count);
extern cudaError_t SparkMimo25LaunchGateSelect(cudaStream_t stream, const float *scores_f32, const float *bias_f32, uint32_t row_count, uint32_t expert_count, uint32_t topk, float epsilon, float route_scale, uint32_t *indices_u32, float *weights_f32);
extern cudaError_t SparkMimo25LaunchSiluMul(cudaStream_t stream, const void *gate_bf16, void *up_bf16, uint32_t row_count, uint32_t width);
extern cudaError_t SparkMimo25LaunchAccumScaledAdd(cudaStream_t stream, void *destination_bf16, const void *source_bf16, const float *weight_f32, uint32_t row_count, uint32_t width);
extern cudaError_t SparkMimo25LaunchGatherLinear(cudaStream_t stream, const SparkMimo25LinearView *view, const void *payload, const void *scale, const void *input_bf16, const uint32_t *input_row_map, void *output_bf16, uint32_t slot_count);
extern cudaError_t SparkMimo25LaunchExpertTile(cudaStream_t stream, const SparkMimo25LinearView *view, const void *payload, const void *scale, const void *input_bf16, const uint32_t *input_row_map, void *output_bf16, uint32_t slot_count);
extern cudaError_t SparkMimo25LaunchScatterScaledAdd(cudaStream_t stream, void *destination_bf16, const void *source_bf16, const uint32_t *row_map, const float *weights_f32, const uint32_t *weight_map, uint32_t slot_count, uint32_t width);
extern cudaError_t SparkMimo25LaunchCacheScatter(cudaStream_t stream, const void *qkv_bf16, uint64_t qkv_row_stride, uint32_t k_offset, uint32_t k_width, uint32_t v_offset, uint32_t v_width, void *k_cache_bf16, void *v_cache_bf16, uint64_t k_lane_stride, uint64_t v_lane_stride, const uint32_t *row_lane_indices, const uint64_t *row_positions, uint32_t window_slots, uint32_t row_count);
extern cudaError_t SparkMimo25LaunchHeadTiledArgmax(cudaStream_t stream, const void *hidden_bf16, const void *head_weight_bf16, void *logits_bf16, uint32_t *output_token_ids, uint32_t row_count, uint32_t candidate_count, uint32_t hidden_dimension);
extern cudaError_t SparkMimo25LaunchResidualAdd(cudaStream_t stream, void *hidden_bf16, const void *delta_bf16, uint32_t row_count, uint32_t width);

static SparkStatus SparkMimo25ModuleConfigure(SparkMimo25ModuleState *state)
{
	SparkStatus status;
	status = SparkStageModuleEnvironmentUnsigned(SPARK_MIMO25_MODULE_TAG,"SPARK_MIMO25_STAGE_COUNT",1u,SPARK_MIMO25_RESIDENT_DECODE_STAGE_MAX_STAGE_COUNT,&state->stage_count);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_MIMO25_MODULE_TAG,"SPARK_MIMO25_STAGE_INDEX",0u,SPARK_MIMO25_RESIDENT_DECODE_STAGE_MAX_STAGE_COUNT - 1u,&state->stage_index);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_MIMO25_MODULE_TAG,"SPARK_MIMO25_STAGE_FIRST_LAYER",0u,SPARK_MIMO25_MODEL_LAYER_COUNT - 1u,&state->first_layer_index);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_MIMO25_MODULE_TAG,"SPARK_MIMO25_STAGE_LAYER_COUNT",1u,SPARK_MIMO25_MODEL_LAYER_COUNT,&state->layer_count);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_MIMO25_MODULE_TAG,"SPARK_MIMO25_STAGE_MAX_ACTIVE_SEQUENCES",1u,SPARK_MIMO25_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,&state->max_active_sequence_count);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_MIMO25_MODULE_TAG,"SPARK_MIMO25_STAGE_PIPELINE_SLOTS",1u,SPARK_MIMO25_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT,&state->pipeline_slot_count);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_MIMO25_MODULE_TAG,"SPARK_MIMO25_STAGE_MAX_SEQ",SPARK_MIMO25_MODEL_SLIDING_WINDOW_TOKENS,SPARK_MIMO25_MODEL_MAX_POSITIONS,&state->max_sequence_positions);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_MIMO25_MODULE_TAG,"SPARK_MIMO25_STAGE_MTP",0u,SPARK_MIMO25_MODEL_MTP_LAYER_COUNT,&state->mtp_armed);
	return(status);
}

static SparkStatus SparkMimo25ModuleValidateSlice(SparkMimo25ModuleState *state)
{
	if ( state->stage_index >= state->stage_count || state->first_layer_index + state->layer_count > SPARK_MIMO25_MODEL_LAYER_COUNT )
	{
		fprintf(stderr,"%s config_slice_invalid stage=%u/%u slice=%u+%u\n",SPARK_MIMO25_MODULE_TAG,state->stage_index,state->stage_count,state->first_layer_index,state->layer_count);
		return(SPARK_STATUS_INVALID_ARGUMENT);
	}
	state->owns_embedding = state->first_layer_index == 0u ? 1u : 0u;
	state->owns_final_head = state->first_layer_index + state->layer_count == SPARK_MIMO25_MODEL_LAYER_COUNT ? 1u : 0u;
	if ( state->mtp_armed != 0u && state->owns_final_head == 0u )
	{
		fprintf(stderr,"%s config_mtp_requires_head stage=%u/%u\n",SPARK_MIMO25_MODULE_TAG,state->stage_index,state->stage_count);
		return(SPARK_STATUS_INVALID_ARGUMENT);
	}
	if ( (state->stage_index == 0u) != (state->owns_embedding != 0u) || (state->stage_index + 1u == state->stage_count) != (state->owns_final_head != 0u) )
	{
		fprintf(stderr,"%s config_position_mismatch stage=%u/%u slice=%u+%u\n",SPARK_MIMO25_MODULE_TAG,state->stage_index,state->stage_count,state->first_layer_index,state->layer_count);
		return(SPARK_STATUS_INVALID_ARGUMENT);
	}
	return(SPARK_STATUS_OK);
}

static void SparkMimo25ModuleBuildOrdinals(SparkMimo25ModuleState *state)
{
	uint32_t layer;
	for (layer = 0; layer < SPARK_MIMO25_RESIDENT_DECODE_STAGE_MAX_LAYER_COUNT; layer++)
	{
		state->full_ordinal_by_layer[layer] = UINT32_MAX;
		state->swa_ordinal_by_layer[layer] = UINT32_MAX;
	}
	for (layer = state->first_layer_index; layer < state->first_layer_index + state->layer_count; layer++)
	{
		if ( SparkMimo25ModelLayerKind(layer) == SPARK_MIMO25_MODEL_LAYER_KIND_FULL )
			state->full_ordinal_by_layer[layer] = state->full_layer_count++;
		else
			state->swa_ordinal_by_layer[layer] = state->swa_layer_count++;
	}
}

static void SparkMimo25ModuleFillLinearView(SparkMimo25LinearView *view, const SparkMimo25StagePackEntry *entry, void *payload, void *scale)
{
	view->abi_version = SPARK_MIMO25_RESIDENT_DECODE_STAGE_LINEAR_VIEW_ABI_VERSION;
	view->weight_format = entry->weight_format == SPARK_MIMO25_STAGEPACK_FORMAT_FP8_E4M3_B128 ? 5u : entry->weight_format == SPARK_MIMO25_STAGEPACK_FORMAT_F32 ? 1u : 0u;
	view->rows = entry->rows;
	view->columns = entry->columns;
	view->payload = payload;
	view->scale = scale;
}

static SparkStatus SparkMimo25ModuleValidateEntry(SparkMimo25ModuleState *state, const SparkMimo25StagePackEntry *entry, uint64_t file_bytes, uint32_t *is_global)
{
	SparkMimo25StagePackTensorShape shape;
	uint64_t payload_bytes,scale_bytes;
	uint32_t global = entry->layer_index == SPARK_MIMO25_STAGEPACK_GLOBAL_LAYER ? 1u : 0u;
	uint32_t in_slice = SparkMimo25StagePackLayerIsMtp(entry->layer_index) != 0u || (entry->layer_index >= state->first_layer_index && entry->layer_index < state->first_layer_index + state->layer_count) ? 1u : 0u;
	if ( entry->tensor_kind >= SPARK_MIMO25_STAGEPACK_TENSOR_KIND_COUNT || (global == 0u && in_slice == 0u) )
		return(SPARK_STATUS_VALIDATION_FAILED);
	if ( SparkMimo25StagePackResolvedShape(entry->tensor_kind,global != 0u ? 0u : entry->layer_index,global,&shape) < 0 )
		return(SPARK_STATUS_VALIDATION_FAILED);
	if ( shape.rows != entry->rows || shape.columns != entry->columns || shape.weight_format != entry->weight_format )
		return(SPARK_STATUS_VALIDATION_FAILED);
	payload_bytes = SparkMimo25StagePackPayloadBytes(entry->weight_format,entry->rows,entry->columns);
	scale_bytes = SparkMimo25StagePackScaleBytes(entry->weight_format,entry->rows,entry->columns);
	if ( entry->payload_offset + payload_bytes > file_bytes || (scale_bytes != 0u && (entry->scale_offset != entry->payload_offset + payload_bytes || entry->scale_offset + scale_bytes > file_bytes)) )
		return(SPARK_STATUS_VALIDATION_FAILED);
	*is_global = global;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkMimo25ModuleBindGlobal(SparkMimo25ModuleState *state, const SparkMimo25StagePackEntry *entry, void *payload)
{
	switch ( entry->tensor_kind )
	{
	case SPARK_MIMO25_STAGEPACK_TENSOR_EMBEDDING: state->token_embedding_bf16 = payload; break;
	case SPARK_MIMO25_STAGEPACK_TENSOR_FINAL_NORM: state->final_norm_weight_bf16 = payload; break;
	case SPARK_MIMO25_STAGEPACK_TENSOR_LM_HEAD: state->lm_head_weight_bf16 = payload; break;
	default:
		return(SPARK_STATUS_VALIDATION_FAILED);
	}
	state->global_seen_bits |= 1u << entry->tensor_kind % 32u;
	return(SPARK_STATUS_OK);
}

// One bind table serves real and MTP layers: the attn/dense/norm kinds
// land in whichever weight struct the layer index selects, MoE kinds only
// on real MoE layers, MTP-only kinds only past the base.
static SparkStatus SparkMimo25ModuleBindLayerCore(SparkMimo25AttnWeights *attn, const void **ffn_norm, SparkMimo25DenseWeights *dense, const SparkMimo25StagePackEntry *entry, void *payload, void *scale)
{
	switch ( entry->tensor_kind )
	{
	case SPARK_MIMO25_STAGEPACK_TENSOR_ATTN_QKV: SparkMimo25ModuleFillLinearView(&attn->qkv,entry,payload,scale); break;
	case SPARK_MIMO25_STAGEPACK_TENSOR_ATTN_O: SparkMimo25ModuleFillLinearView(&attn->o,entry,payload,scale); break;
	case SPARK_MIMO25_STAGEPACK_TENSOR_ATTN_SINK: attn->sink_f32 = (const float *)payload; break;
	case SPARK_MIMO25_STAGEPACK_TENSOR_ATTN_NORM: attn->attn_norm_bf16 = payload; break;
	case SPARK_MIMO25_STAGEPACK_TENSOR_FFN_NORM: *ffn_norm = payload; break;
	case SPARK_MIMO25_STAGEPACK_TENSOR_DENSE_W1: SparkMimo25ModuleFillLinearView(&dense->w1,entry,payload,scale); break;
	case SPARK_MIMO25_STAGEPACK_TENSOR_DENSE_W2: SparkMimo25ModuleFillLinearView(&dense->w2,entry,payload,scale); break;
	case SPARK_MIMO25_STAGEPACK_TENSOR_DENSE_W3: SparkMimo25ModuleFillLinearView(&dense->w3,entry,payload,scale); break;
	default:
		return(SPARK_STATUS_VALIDATION_FAILED);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkMimo25ModuleBindLayerMoe(SparkMimo25MoeWeights *moe, const SparkMimo25StagePackEntry *entry, void *payload, void *scale)
{
	switch ( entry->tensor_kind )
	{
	case SPARK_MIMO25_STAGEPACK_TENSOR_GATE_WEIGHT: SparkMimo25ModuleFillLinearView(&moe->gate,entry,payload,scale); break;
	case SPARK_MIMO25_STAGEPACK_TENSOR_GATE_BIAS: moe->gate_bias_f32 = (const float *)payload; break;
	case SPARK_MIMO25_STAGEPACK_TENSOR_EXPERTS_W1: SparkMimo25ModuleFillLinearView(&moe->experts_w1,entry,payload,scale); break;
	case SPARK_MIMO25_STAGEPACK_TENSOR_EXPERTS_W2: SparkMimo25ModuleFillLinearView(&moe->experts_w2,entry,payload,scale); break;
	case SPARK_MIMO25_STAGEPACK_TENSOR_EXPERTS_W3: SparkMimo25ModuleFillLinearView(&moe->experts_w3,entry,payload,scale); break;
	default:
		return(SPARK_STATUS_VALIDATION_FAILED);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkMimo25ModuleBindMtpOnly(SparkMimo25MtpLayerWeights *mtp, const SparkMimo25StagePackEntry *entry, void *payload, void *scale)
{
	switch ( entry->tensor_kind )
	{
	case SPARK_MIMO25_STAGEPACK_TENSOR_MTP_EH_PROJ: SparkMimo25ModuleFillLinearView(&mtp->eh_proj,entry,payload,scale); break;
	case SPARK_MIMO25_STAGEPACK_TENSOR_MTP_ENORM: mtp->enorm_bf16 = payload; break;
	case SPARK_MIMO25_STAGEPACK_TENSOR_MTP_HNORM: mtp->hnorm_bf16 = payload; break;
	case SPARK_MIMO25_STAGEPACK_TENSOR_MTP_FINAL_NORM: mtp->final_norm_bf16 = payload; break;
	default:
		return(SPARK_STATUS_VALIDATION_FAILED);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkMimo25ModuleBindLayer(SparkMimo25ModuleState *state, const SparkMimo25StagePackEntry *entry, void *payload, void *scale)
{
	uint32_t mtp = SparkMimo25StagePackLayerIsMtp(entry->layer_index);
	uint32_t mtp_index = mtp != 0u ? entry->layer_index - SPARK_MIMO25_STAGEPACK_MTP_LAYER_BASE : 0u;
	SparkMimo25MtpLayerWeights *mtp_layer = &state->mtp_layers[mtp_index];
	SparkMimo25LayerWeights *layer = &state->layers[mtp != 0u ? 0u : entry->layer_index];
	uint32_t *seen = mtp != 0u ? &state->mtp_seen_bits[mtp_index] : &state->layer_seen_bits[entry->layer_index];
	SparkStatus status;
	if ( entry->tensor_kind >= SPARK_MIMO25_STAGEPACK_TENSOR_MTP_EH_PROJ && entry->tensor_kind <= SPARK_MIMO25_STAGEPACK_TENSOR_MTP_FINAL_NORM )
		status = mtp != 0u ? SparkMimo25ModuleBindMtpOnly(mtp_layer,entry,payload,scale) : SPARK_STATUS_VALIDATION_FAILED;
	else if ( entry->tensor_kind >= SPARK_MIMO25_STAGEPACK_TENSOR_GATE_WEIGHT && entry->tensor_kind <= SPARK_MIMO25_STAGEPACK_TENSOR_EXPERTS_W3 )
		status = mtp != 0u ? SPARK_STATUS_VALIDATION_FAILED : SparkMimo25ModuleBindLayerMoe(&layer->moe,entry,payload,scale);
	else if ( mtp != 0u )
		status = SparkMimo25ModuleBindLayerCore(&mtp_layer->attn,&mtp_layer->ffn_norm_bf16,&mtp_layer->mlp,entry,payload,scale);
	else
		status = SparkMimo25ModuleBindLayerCore(&layer->attn,&layer->ffn_norm_bf16,&layer->dense,entry,payload,scale);
	if ( status != SPARK_STATUS_OK )
		return(status);
	*seen |= 1u << entry->tensor_kind;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkMimo25ModuleLoadEntry(SparkMimo25ModuleState *state, FILE *file, const SparkMimo25StagePackEntry *entry, uint64_t file_bytes)
{
	uint64_t payload_bytes = SparkMimo25StagePackPayloadBytes(entry->weight_format,entry->rows,entry->columns);
	uint64_t scale_bytes = SparkMimo25StagePackScaleBytes(entry->weight_format,entry->rows,entry->columns);
	void *payload = 0,*scale = 0;
	uint32_t is_global = 0u;
	SparkStatus status = SparkMimo25ModuleValidateEntry(state,entry,file_bytes,&is_global);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"%s pack_entry_invalid kind=%u layer=%u\n",SPARK_MIMO25_MODULE_TAG,entry->tensor_kind,entry->layer_index);
		return(status);
	}
	status = SparkStageModuleLoadDeviceRegion(&state->ledger,file,entry->payload_offset,payload_bytes,&payload);
	if ( status == SPARK_STATUS_OK && scale_bytes != 0u )
		status = SparkStageModuleLoadDeviceRegion(&state->ledger,file,entry->scale_offset,scale_bytes,&scale);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( is_global != 0u )
		return(SparkMimo25ModuleBindGlobal(state,entry,payload));
	return(SparkMimo25ModuleBindLayer(state,entry,payload,scale));
}

static uint32_t SparkMimo25ModuleExpectedLayerBits(uint32_t layer_index)
{
	uint32_t bits = (1u << SPARK_MIMO25_STAGEPACK_TENSOR_ATTN_QKV) | (1u << SPARK_MIMO25_STAGEPACK_TENSOR_ATTN_O) | (1u << SPARK_MIMO25_STAGEPACK_TENSOR_ATTN_NORM) | (1u << SPARK_MIMO25_STAGEPACK_TENSOR_FFN_NORM);
	if ( SparkMimo25StagePackLayerKind(layer_index) == SPARK_MIMO25_MODEL_LAYER_KIND_SWA )
		bits |= 1u << SPARK_MIMO25_STAGEPACK_TENSOR_ATTN_SINK;
	if ( SparkMimo25StagePackLayerHasMoe(layer_index) != 0u )
		bits |= (1u << SPARK_MIMO25_STAGEPACK_TENSOR_GATE_WEIGHT) | (1u << SPARK_MIMO25_STAGEPACK_TENSOR_GATE_BIAS) | (1u << SPARK_MIMO25_STAGEPACK_TENSOR_EXPERTS_W1) | (1u << SPARK_MIMO25_STAGEPACK_TENSOR_EXPERTS_W2) | (1u << SPARK_MIMO25_STAGEPACK_TENSOR_EXPERTS_W3);
	else
		bits |= (1u << SPARK_MIMO25_STAGEPACK_TENSOR_DENSE_W1) | (1u << SPARK_MIMO25_STAGEPACK_TENSOR_DENSE_W2) | (1u << SPARK_MIMO25_STAGEPACK_TENSOR_DENSE_W3);
	if ( SparkMimo25StagePackLayerIsMtp(layer_index) != 0u )
		bits |= (1u << SPARK_MIMO25_STAGEPACK_TENSOR_MTP_EH_PROJ) | (1u << SPARK_MIMO25_STAGEPACK_TENSOR_MTP_ENORM) | (1u << SPARK_MIMO25_STAGEPACK_TENSOR_MTP_HNORM) | (1u << SPARK_MIMO25_STAGEPACK_TENSOR_MTP_FINAL_NORM);
	return(bits);
}

static SparkStatus SparkMimo25ModuleVerifyCoverage(SparkMimo25ModuleState *state)
{
	uint32_t expected_globals = 0u,layer;
	for (layer = state->first_layer_index; layer < state->first_layer_index + state->layer_count; layer++)
		if ( state->layer_seen_bits[layer] != SparkMimo25ModuleExpectedLayerBits(layer) )
		{
			fprintf(stderr,"%s pack_layer_coverage layer=%u seen=%x\n",SPARK_MIMO25_MODULE_TAG,layer,state->layer_seen_bits[layer]);
			return(SPARK_STATUS_VALIDATION_FAILED);
		}
	if ( state->owns_embedding != 0u || state->owns_final_head != 0u )
		expected_globals |= 1u << SPARK_MIMO25_STAGEPACK_TENSOR_EMBEDDING % 32u;
	if ( state->owns_final_head != 0u )
	{
		expected_globals |= (1u << SPARK_MIMO25_STAGEPACK_TENSOR_FINAL_NORM % 32u) | (1u << SPARK_MIMO25_STAGEPACK_TENSOR_LM_HEAD % 32u);
		for (layer = 0; layer < SPARK_MIMO25_MODEL_MTP_LAYER_COUNT; layer++)
			if ( state->mtp_seen_bits[layer] != SparkMimo25ModuleExpectedLayerBits(SPARK_MIMO25_STAGEPACK_MTP_LAYER_BASE + layer) )
			{
				fprintf(stderr,"%s pack_mtp_coverage mtp=%u seen=%x\n",SPARK_MIMO25_MODULE_TAG,layer,state->mtp_seen_bits[layer]);
				return(SPARK_STATUS_VALIDATION_FAILED);
			}
	}
	if ( state->global_seen_bits != expected_globals )
	{
		fprintf(stderr,"%s pack_global_coverage seen=%x expected=%x\n",SPARK_MIMO25_MODULE_TAG,state->global_seen_bits,expected_globals);
		return(SPARK_STATUS_VALIDATION_FAILED);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkMimo25ModuleLoadPack(SparkMimo25ModuleState *state, const char *path)
{
	SparkMimo25StagePackHeader header,expected;
	SparkMimo25StagePackEntry *directory;
	FILE *file;
	SparkStatus status;
	int32_t compare;
	uint32_t index;
	file = fopen(path,"rb");
	if ( file == 0 )
	{
		fprintf(stderr,"%s pack_open_failed path=%s\n",SPARK_MIMO25_MODULE_TAG,path);
		return(SPARK_STATUS_IO_ERROR);
	}
	status = SparkStageModulePackRead(SPARK_MIMO25_MODULE_TAG,file,0u,&header,sizeof(header));
	if ( status == SPARK_STATUS_OK )
	{
		SparkMimo25StagePackExpectedGeometry(&expected,state->first_layer_index,state->layer_count);
		compare = SparkMimo25StagePackCompareGeometry(&header,&expected);
		if ( compare != 0 )
		{
			fprintf(stderr,"%s pack_geometry_mismatch field=%s\n",SPARK_MIMO25_MODULE_TAG,SparkMimo25StagePackGeometryFieldName(compare));
			status = SPARK_STATUS_VALIDATION_FAILED;
		}
	}
	directory = status == SPARK_STATUS_OK ? (SparkMimo25StagePackEntry *)malloc((size_t)header.tensor_count * sizeof(SparkMimo25StagePackEntry)) : 0;
	if ( status == SPARK_STATUS_OK && directory == 0 )
		status = SPARK_STATUS_CAPACITY_EXCEEDED;
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModulePackRead(SPARK_MIMO25_MODULE_TAG,file,header.directory_offset,directory,(uint64_t)header.tensor_count * sizeof(SparkMimo25StagePackEntry));
	for (index = 0; status == SPARK_STATUS_OK && index < header.tensor_count; index++)
		status = SparkMimo25ModuleLoadEntry(state,file,&directory[index],header.file_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkMimo25ModuleVerifyCoverage(state);
	free(directory);
	fclose(file);
	return(status);
}

static SparkStatus SparkMimo25ModuleUploadFreqs(SparkMimo25ModuleState *state)
{
	float host_freqs[SPARK_MIMO25_MODEL_ATTN_ROPE_DIMENSION / 2u];
	uint32_t pair;
	SparkStatus status;
	cudaError_t error;
	status = SparkStageModuleDeviceAllocate(&state->ledger,sizeof(host_freqs),(void **)&state->full_freqs_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,sizeof(host_freqs),(void **)&state->swa_freqs_f32);
	if ( status != SPARK_STATUS_OK )
		return(status);
	for (pair = 0; pair < SPARK_MIMO25_MODEL_ATTN_ROPE_DIMENSION / 2u; pair++)
		host_freqs[pair] = 1.0f / powf(SPARK_MIMO25_MODEL_FULL_ROPE_THETA,(float)(2u * pair) / (float)SPARK_MIMO25_MODEL_ATTN_ROPE_DIMENSION);
	error = cudaMemcpy(state->full_freqs_f32,host_freqs,sizeof(host_freqs),cudaMemcpyHostToDevice);
	if ( error == cudaSuccess )
	{
		for (pair = 0; pair < SPARK_MIMO25_MODEL_ATTN_ROPE_DIMENSION / 2u; pair++)
			host_freqs[pair] = 1.0f / powf(SPARK_MIMO25_MODEL_SWA_ROPE_THETA,(float)(2u * pair) / (float)SPARK_MIMO25_MODEL_ATTN_ROPE_DIMENSION);
		error = cudaMemcpy(state->swa_freqs_f32,host_freqs,sizeof(host_freqs),cudaMemcpyHostToDevice);
	}
	return(SparkStageModuleCudaStatus(SPARK_MIMO25_MODULE_TAG,error,"freq_upload"));
}

// Two cache pools per side: full layers keep max_seq history, SWA layers
// the ring; a slot holds every kv head of one position contiguously, k
// at 192 wide and v at 128.
static SparkStatus SparkMimo25ModuleAllocatePools(SparkMimo25ModuleState *state)
{
	uint64_t lanes = state->max_active_sequence_count,bf16 = SPARK_MIMO25_MODEL_BF16_ELEMENT_BYTES;
	uint64_t k_full_slot = (uint64_t)SPARK_MIMO25_MODEL_FULL_KV_HEAD_COUNT * SPARK_MIMO25_MODEL_ATTN_HEAD_DIMENSION;
	uint64_t v_full_slot = (uint64_t)SPARK_MIMO25_MODEL_FULL_KV_HEAD_COUNT * SPARK_MIMO25_MODEL_ATTN_VALUE_DIMENSION;
	uint64_t k_swa_slot = (uint64_t)SPARK_MIMO25_MODEL_SWA_KV_HEAD_COUNT * SPARK_MIMO25_MODEL_ATTN_HEAD_DIMENSION;
	uint64_t v_swa_slot = (uint64_t)SPARK_MIMO25_MODEL_SWA_KV_HEAD_COUNT * SPARK_MIMO25_MODEL_ATTN_VALUE_DIMENSION;
	SparkStatus status = SPARK_STATUS_OK;
	state->k_full_lane_stride = (uint64_t)state->max_sequence_positions * k_full_slot;
	state->v_full_lane_stride = (uint64_t)state->max_sequence_positions * v_full_slot;
	state->k_swa_lane_stride = (uint64_t)SPARK_MIMO25_MODEL_SLIDING_WINDOW_TOKENS * k_swa_slot;
	state->v_swa_lane_stride = (uint64_t)SPARK_MIMO25_MODEL_SLIDING_WINDOW_TOKENS * v_swa_slot;
	if ( state->full_layer_count != 0u )
		status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,(uint64_t)state->full_layer_count * lanes * state->k_full_lane_stride * bf16,&state->k_full_cache_bf16);
	if ( status == SPARK_STATUS_OK && state->full_layer_count != 0u )
		status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,(uint64_t)state->full_layer_count * lanes * state->v_full_lane_stride * bf16,&state->v_full_cache_bf16);
	if ( status == SPARK_STATUS_OK && state->swa_layer_count != 0u )
		status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,(uint64_t)state->swa_layer_count * lanes * state->k_swa_lane_stride * bf16,&state->k_swa_cache_bf16);
	if ( status == SPARK_STATUS_OK && state->swa_layer_count != 0u )
		status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,(uint64_t)state->swa_layer_count * lanes * state->v_swa_lane_stride * bf16,&state->v_swa_cache_bf16);
	if ( status == SPARK_STATUS_OK && state->mtp_armed != 0u )
		status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,(uint64_t)state->mtp_armed * lanes * state->k_swa_lane_stride * bf16,&state->k_mtp_cache_bf16);
	if ( status == SPARK_STATUS_OK && state->mtp_armed != 0u )
		status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,(uint64_t)state->mtp_armed * lanes * state->v_swa_lane_stride * bf16,&state->v_mtp_cache_bf16);
	return(status);
}

static SparkStatus SparkMimo25ModuleAllocateSlotBatched(SparkMimo25ModuleState *state, SparkMimo25ModuleSlot *slot)
{
	uint64_t rows = state->max_active_sequence_count,dim = SPARK_MIMO25_MODEL_HIDDEN_DIMENSION,bf16 = SPARK_MIMO25_MODEL_BF16_ELEMENT_BYTES;
	SparkStatus status;
	status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_MIMO25_MODEL_EXPERTS_PER_TOKEN * sizeof(uint32_t),(void **)&slot->grouped_rows_u32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_MIMO25_MODEL_EXPERTS_PER_TOKEN * sizeof(uint32_t),(void **)&slot->grouped_weight_slots_u32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleCudaStatus(SPARK_MIMO25_MODULE_TAG,cudaHostAlloc((void **)&slot->host_moe_indices,rows * SPARK_MIMO25_MODEL_EXPERTS_PER_TOKEN * sizeof(uint32_t),cudaHostAllocDefault),"pin_moe_indices");
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleCudaStatus(SPARK_MIMO25_MODULE_TAG,cudaHostAlloc((void **)&slot->host_grouped_rows,rows * SPARK_MIMO25_MODEL_EXPERTS_PER_TOKEN * sizeof(uint32_t),cudaHostAllocDefault),"pin_grouped_rows");
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleCudaStatus(SPARK_MIMO25_MODULE_TAG,cudaHostAlloc((void **)&slot->host_grouped_weight_slots,rows * SPARK_MIMO25_MODEL_EXPERTS_PER_TOKEN * sizeof(uint32_t),cudaHostAllocDefault),"pin_grouped_slots");
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_MIMO25_MODEL_EXPERTS_PER_TOKEN * SPARK_MIMO25_MODEL_EXPERT_INTERMEDIATE_DIMENSION * bf16,&slot->moe_slot_gate_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_MIMO25_MODEL_EXPERTS_PER_TOKEN * SPARK_MIMO25_MODEL_EXPERT_INTERMEDIATE_DIMENSION * bf16,&slot->moe_slot_up_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_MIMO25_MODEL_EXPERTS_PER_TOKEN * dim * bf16,&slot->moe_slot_out_bf16);
	return(status);
}

static SparkStatus SparkMimo25ModuleAllocateSlotMtp(SparkMimo25ModuleState *state, SparkMimo25ModuleSlot *slot)
{
	uint64_t rows = state->max_active_sequence_count,dim = SPARK_MIMO25_MODEL_HIDDEN_DIMENSION,bf16 = SPARK_MIMO25_MODEL_BF16_ELEMENT_BYTES;
	SparkStatus status = SPARK_STATUS_OK;
	if ( state->mtp_armed == 0u )
		return(SPARK_STATUS_OK);
	status = SparkStageModuleDeviceAllocate(&state->ledger,rows * sizeof(uint64_t),(void **)&slot->mtp_positions_u64);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleCudaStatus(SPARK_MIMO25_MODULE_TAG,cudaHostAlloc((void **)&slot->host_mtp_positions,rows * sizeof(uint64_t),cudaHostAllocDefault),"pin_mtp_positions");
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * sizeof(uint32_t),(void **)&slot->mtp_step_ids_u32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * state->mtp_armed * sizeof(uint32_t),(void **)&slot->mtp_draft_ids_u32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * 2u * dim * bf16,&slot->mtp_concat_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * dim * bf16,&slot->mtp_hidden_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * dim * bf16,&slot->mtp_scratch_bf16);
	return(status);
}

static SparkStatus SparkMimo25ModuleAllocateSlot(SparkMimo25ModuleState *state, SparkMimo25ModuleSlot *slot)
{
	uint64_t rows = state->max_active_sequence_count,dim = SPARK_MIMO25_MODEL_HIDDEN_DIMENSION,bf16 = SPARK_MIMO25_MODEL_BF16_ELEMENT_BYTES;
	uint32_t qkv_max = SPARK_MIMO25_MODEL_SWA_QKV_DIMENSION > SPARK_MIMO25_MODEL_FULL_QKV_DIMENSION ? SPARK_MIMO25_MODEL_SWA_QKV_DIMENSION : SPARK_MIMO25_MODEL_FULL_QKV_DIMENSION;
	SparkStatus status;
	status = SparkStageModuleDeviceAllocate(&state->ledger,rows * sizeof(uint32_t),(void **)&slot->input_token_ids);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * sizeof(uint32_t),(void **)&slot->output_token_ids);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * sizeof(uint32_t),(void **)&slot->row_lane_indices);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * sizeof(uint64_t),(void **)&slot->row_positions);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * dim * bf16,&slot->hidden_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * dim * bf16,&slot->normalized_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * qkv_max * bf16,&slot->qkv_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_MIMO25_MODEL_O_INPUT_DIMENSION * bf16,&slot->attn_out_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * dim * bf16,&slot->delta_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_MIMO25_MODEL_ROUTED_EXPERT_COUNT * sizeof(float),(void **)&slot->moe_scores_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_MIMO25_MODEL_EXPERTS_PER_TOKEN * sizeof(uint32_t),(void **)&slot->moe_indices_u32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_MIMO25_MODEL_EXPERTS_PER_TOKEN * sizeof(float),(void **)&slot->moe_weights_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_MIMO25_MODEL_DENSE_INTERMEDIATE_DIMENSION * bf16,&slot->ffn_gate_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_MIMO25_MODEL_DENSE_INTERMEDIATE_DIMENSION * bf16,&slot->ffn_up_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * dim * bf16,&slot->ffn_delta_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * dim * bf16,&slot->ffn_accum_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkMimo25ModuleAllocateSlotBatched(state,slot);
	if ( status == SPARK_STATUS_OK && state->owns_final_head != 0u )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_MIMO25_MODEL_VOCAB_COUNT * bf16,&slot->head_logits_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkMimo25ModuleAllocateSlotMtp(state,slot);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleCudaStatus(SPARK_MIMO25_MODULE_TAG,cudaStreamCreateWithFlags((cudaStream_t *)&slot->cuda_stream,cudaStreamNonBlocking),"stream_create");
	return(status);
}

static SparkStatus SparkMimo25ModuleValidateFrame(SparkMimo25ModuleState *state, const SparkModelDriverFrame *frame, const SparkMimo25ResidentDecodeStageFrameContext **context_out)
{
	const SparkMimo25ResidentDecodeStageFrameContext *context;
	const SparkMimo25DecodeBatchView *batch;
	uint32_t needs_input = state->stage_index > 0u ? 1u : 0u,needs_output = state->stage_index + 1u < state->stage_count ? 1u : 0u;
	uint32_t expected_buffers = state->owns_embedding + state->owns_final_head;
	if ( frame == 0 || frame->user_context == 0 || frame->buffer_count != expected_buffers )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	context = (const SparkMimo25ResidentDecodeStageFrameContext *)frame->user_context;
	if ( context->abi_version != SPARK_MIMO25_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION || context->descriptor_bytes != (uint32_t)sizeof(*context) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( (context->flags & SPARK_MIMO25_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_PREFILL_FRAME_VIEW) != 0u )
	{
		fprintf(stderr,"%s prefill_pending\n",SPARK_MIMO25_MODULE_TAG);
		return(SPARK_STATUS_MODULE_NOT_VALIDATED);
	}
	if ( (context->flags & SPARK_MIMO25_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_DECODE_BATCH_VIEW) == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( ((context->flags & SPARK_MIMO25_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_INPUT_TRANSPORT) != 0u) != (needs_input != 0u) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( ((context->flags & SPARK_MIMO25_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_OUTPUT_TRANSPORT) != 0u) != (needs_output != 0u) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	batch = context->decode_batch;
	if ( batch == 0 || batch->abi_version != SPARK_MIMO25_RESIDENT_DECODE_STAGE_DECODE_BATCH_VIEW_ABI_VERSION || batch->descriptor_bytes != (uint32_t)sizeof(*batch) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( batch->row_count == 0u || batch->row_count > state->max_active_sequence_count || batch->row_count != frame->new_token_count || batch->row_lane_indices == 0 || batch->row_positions == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	*context_out = context;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkMimo25ModuleStageRows(SparkMimo25ModuleState *state, const SparkMimo25DecodeBatchView *batch, SparkMimo25ModuleSlot *slot)
{
	uint8_t lane_used[SPARK_MIMO25_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	uint32_t row,lane;
	cudaError_t error;
	memset(lane_used,0,sizeof(lane_used));
	for (row = 0; row < batch->row_count; row++)
	{
		lane = batch->row_lane_indices[row];
		if ( lane >= state->max_active_sequence_count || lane_used[lane] != 0u || batch->row_positions[row] >= state->max_sequence_positions )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		lane_used[lane] = 1u;
	}
	error = cudaMemcpyAsync(slot->row_lane_indices,batch->row_lane_indices,(uint64_t)batch->row_count * sizeof(uint32_t),cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess )
		error = cudaMemcpyAsync(slot->row_positions,batch->row_positions,(uint64_t)batch->row_count * sizeof(uint64_t),cudaMemcpyHostToDevice,stream);
	return(SparkStageModuleCudaStatus(SPARK_MIMO25_MODULE_TAG,error,"stage_rows"));
}

static SparkStatus SparkMimo25ModuleBeginHidden(SparkMimo25ModuleState *state, SparkMimo25ModuleSlot *slot, SparkMimo25ResidentDecodeStageFrameContext *context, const SparkModelDriverFrame *frame, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	SparkStatus status;
	cudaError_t error;
	if ( state->owns_embedding != 0u )
	{
		error = cudaMemcpyAsync(slot->input_token_ids,frame->buffers[0].address,(uint64_t)rows * sizeof(uint32_t),cudaMemcpyHostToDevice,stream);
		if ( error == cudaSuccess )
			error = SparkMimo25LaunchEmbeddingGather(stream,slot->input_token_ids,state->token_embedding_bf16,slot->hidden_bf16,rows,SPARK_MIMO25_MODEL_HIDDEN_DIMENSION);
		return(SparkStageModuleCudaStatus(SPARK_MIMO25_MODULE_TAG,error,"embedding"));
	}
	if ( context->hidden_input_post_receive_function == 0 || context->hidden_input_transport_session == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	status = context->hidden_input_post_receive_function(context->hidden_input_transport_session,&context->hidden_input_packet);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( context->hidden_input_packet.active_sequence_count != rows || context->hidden_input_packet.hidden_dimension != SPARK_MIMO25_MODEL_HIDDEN_DIMENSION || context->hidden_input_packet.hidden_bf16 == 0 )
		return(SPARK_STATUS_VALIDATION_FAILED);
	error = cudaMemcpyAsync(slot->hidden_bf16,context->hidden_input_packet.hidden_bf16,(uint64_t)rows * SPARK_MIMO25_MODEL_HIDDEN_DIMENSION * SPARK_MIMO25_MODEL_BF16_ELEMENT_BYTES,cudaMemcpyDeviceToDevice,stream);
	return(SparkStageModuleCudaStatus(SPARK_MIMO25_MODULE_TAG,error,"hidden_receive"));
}

// The fused qkv row for the layer's branch: q at zero, k after q, v after
// k; the v slice scales before the per-row cache writes land k and v at
// the position's slot (ring for SWA, dense for full).
static SparkStatus SparkMimo25ModuleWriteCaches(SparkMimo25ModuleState *state, SparkMimo25ModuleSlot *slot, const SparkMimo25DecodeBatchView *batch, uint32_t swa, uint32_t ordinal, uint32_t qkv_dim, uint32_t k_offset, uint32_t v_offset, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	uint64_t lanes = state->max_active_sequence_count,bf16 = SPARK_MIMO25_MODEL_BF16_ELEMENT_BYTES;
	uint64_t k_lane = swa != 0u ? state->k_swa_lane_stride : state->k_full_lane_stride;
	uint64_t v_lane = swa != 0u ? state->v_swa_lane_stride : state->v_full_lane_stride;
	uint64_t k_slot = (uint64_t)(swa != 0u ? SPARK_MIMO25_MODEL_SWA_KV_HEAD_COUNT : SPARK_MIMO25_MODEL_FULL_KV_HEAD_COUNT) * SPARK_MIMO25_MODEL_ATTN_HEAD_DIMENSION;
	uint64_t v_slot = (uint64_t)(swa != 0u ? SPARK_MIMO25_MODEL_SWA_KV_HEAD_COUNT : SPARK_MIMO25_MODEL_FULL_KV_HEAD_COUNT) * SPARK_MIMO25_MODEL_ATTN_VALUE_DIMENSION;
	uint8_t *k_cache = (uint8_t *)(swa != 0u ? state->k_swa_cache_bf16 : state->k_full_cache_bf16) + (uint64_t)ordinal * lanes * k_lane * bf16;
	uint8_t *v_cache = (uint8_t *)(swa != 0u ? state->v_swa_cache_bf16 : state->v_full_cache_bf16) + (uint64_t)ordinal * lanes * v_lane * bf16;
	cudaError_t error;
	(void)batch;
	error = SparkMimo25LaunchCacheScatter(stream,slot->qkv_bf16,qkv_dim,k_offset,(uint32_t)k_slot,v_offset,(uint32_t)v_slot,k_cache,v_cache,k_lane,v_lane,slot->row_lane_indices,slot->row_positions,swa != 0u ? SPARK_MIMO25_MODEL_SLIDING_WINDOW_TOKENS : 0u,rows);
	return(SparkStageModuleCudaStatus(SPARK_MIMO25_MODULE_TAG,error,"cache_write"));
}

static SparkStatus SparkMimo25ModuleRunAttention(SparkMimo25ModuleState *state, SparkMimo25ModuleSlot *slot, const SparkMimo25LayerWeights *layer, const SparkMimo25DecodeBatchView *batch, uint32_t layer_index, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	uint32_t swa = SparkMimo25ModelLayerKind(layer_index) == SPARK_MIMO25_MODEL_LAYER_KIND_SWA ? 1u : 0u;
	uint32_t kv_heads = swa != 0u ? SPARK_MIMO25_MODEL_SWA_KV_HEAD_COUNT : SPARK_MIMO25_MODEL_FULL_KV_HEAD_COUNT;
	uint32_t qkv_dim = swa != 0u ? SPARK_MIMO25_MODEL_SWA_QKV_DIMENSION : SPARK_MIMO25_MODEL_FULL_QKV_DIMENSION;
	uint32_t k_offset = SPARK_MIMO25_MODEL_Q_DIMENSION,v_offset = k_offset + kv_heads * SPARK_MIMO25_MODEL_ATTN_HEAD_DIMENSION;
	uint32_t ordinal = swa != 0u ? state->swa_ordinal_by_layer[layer_index] : state->full_ordinal_by_layer[layer_index];
	uint64_t lanes = state->max_active_sequence_count,bf16 = SPARK_MIMO25_MODEL_BF16_ELEMENT_BYTES;
	const float *freqs = swa != 0u ? state->swa_freqs_f32 : state->full_freqs_f32;
	uint64_t k_lane = swa != 0u ? state->k_swa_lane_stride : state->k_full_lane_stride;
	uint64_t v_lane = swa != 0u ? state->v_swa_lane_stride : state->v_full_lane_stride;
	const void *k_cache = (const uint8_t *)(swa != 0u ? state->k_swa_cache_bf16 : state->k_full_cache_bf16) + (uint64_t)ordinal * lanes * k_lane * bf16;
	const void *v_cache = (const uint8_t *)(swa != 0u ? state->v_swa_cache_bf16 : state->v_full_cache_bf16) + (uint64_t)ordinal * lanes * v_lane * bf16;
	cudaError_t error;
	SparkStatus status;
	error = SparkMimo25LaunchLinear(stream,&layer->attn.qkv,0,0,slot->normalized_bf16,slot->qkv_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkMimo25LaunchScaleSlice(stream,slot->qkv_bf16,qkv_dim,v_offset,kv_heads * SPARK_MIMO25_MODEL_ATTN_VALUE_DIMENSION,SPARK_MIMO25_MODEL_ATTN_VALUE_SCALE,rows);
	if ( error == cudaSuccess )
		error = SparkMimo25LaunchRope(stream,slot->qkv_bf16,qkv_dim,0u,freqs,slot->row_positions,rows,SPARK_MIMO25_MODEL_ATTN_HEAD_COUNT,SPARK_MIMO25_MODEL_ATTN_HEAD_DIMENSION,SPARK_MIMO25_MODEL_ATTN_ROPE_DIMENSION,0u);
	if ( error == cudaSuccess )
		error = SparkMimo25LaunchRope(stream,slot->qkv_bf16,qkv_dim,k_offset,freqs,slot->row_positions,rows,kv_heads,SPARK_MIMO25_MODEL_ATTN_HEAD_DIMENSION,SPARK_MIMO25_MODEL_ATTN_ROPE_DIMENSION,0u);
	if ( error != cudaSuccess )
		return(SparkStageModuleCudaStatus(SPARK_MIMO25_MODULE_TAG,error,"qkv_prepare"));
	status = SparkMimo25ModuleWriteCaches(state,slot,batch,swa,ordinal,qkv_dim,k_offset,v_offset,rows);
	if ( status != SPARK_STATUS_OK )
		return(status);
	error = SparkMimo25LaunchAttnDecode(stream,slot->qkv_bf16,qkv_dim,k_cache,v_cache,k_lane,v_lane,(uint64_t)kv_heads * SPARK_MIMO25_MODEL_ATTN_HEAD_DIMENSION,(uint64_t)kv_heads * SPARK_MIMO25_MODEL_ATTN_VALUE_DIMENSION,slot->row_lane_indices,slot->row_positions,swa != 0u ? layer->attn.sink_f32 : 0,1.0f / sqrtf((float)SPARK_MIMO25_MODEL_ATTN_HEAD_DIMENSION),slot->attn_out_bf16,rows,SPARK_MIMO25_MODEL_ATTN_HEAD_COUNT,SPARK_MIMO25_MODEL_ATTN_HEAD_COUNT / kv_heads,SPARK_MIMO25_MODEL_ATTN_HEAD_DIMENSION,SPARK_MIMO25_MODEL_ATTN_VALUE_DIMENSION,swa != 0u ? SPARK_MIMO25_MODEL_SLIDING_WINDOW_TOKENS : 0u);
	if ( error == cudaSuccess )
		error = SparkMimo25LaunchLinear(stream,&layer->attn.o,0,0,slot->attn_out_bf16,slot->delta_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkMimo25LaunchResidualAdd(stream,slot->hidden_bf16,slot->delta_bf16,rows,SPARK_MIMO25_MODEL_HIDDEN_DIMENSION);
	return(SparkStageModuleCudaStatus(SPARK_MIMO25_MODULE_TAG,error,"attention"));
}

// A stacked-expert slice on the f32-block fp8 layout: payload advances a
// whole expert block of bytes, scales the matching [128,128] tile count.
static void SparkMimo25ModuleExpertView(SparkMimo25LinearView *view, const SparkMimo25LinearView *stacked, uint32_t expert, uint64_t rows_per_expert, uint64_t columns, const void **payload, const void **scale)
{
	*view = *stacked;
	view->rows = (uint32_t)rows_per_expert;
	view->columns = (uint32_t)columns;
	*payload = (const uint8_t *)stacked->payload + expert * rows_per_expert * columns;
	*scale = (const float *)stacked->scale + expert * (rows_per_expert / SPARK_MIMO25_MODEL_FP8_SCALE_BLOCK) * (columns / SPARK_MIMO25_MODEL_FP8_SCALE_BLOCK);
}

// Counting sort of the batch's (row, rank) pairs by expert: dense slot
// lists per expert, weight slots naming the device-resident routing
// weight each slot applies at the OUTPUT scatter.
static SparkStatus SparkMimo25ModuleGroupByExpert(SparkMimo25ModuleSlot *slot, uint32_t rows, uint32_t *active_out)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	uint32_t counts[SPARK_MIMO25_MODEL_ROUTED_EXPERT_COUNT];
	uint32_t pair_count = rows * SPARK_MIMO25_MODEL_EXPERTS_PER_TOKEN,pair,expert,cursor = 0u,active = 0u;
	cudaError_t error;
	memset(counts,0,sizeof(counts));
	for (pair = 0; pair < pair_count; pair++)
	{
		expert = slot->host_moe_indices[pair];
		if ( expert >= SPARK_MIMO25_MODEL_ROUTED_EXPERT_COUNT )
			return(SPARK_STATUS_VALIDATION_FAILED);
		counts[expert]++;
	}
	for (expert = 0; expert < SPARK_MIMO25_MODEL_ROUTED_EXPERT_COUNT; expert++)
	{
		slot->host_expert_offsets[expert] = cursor;
		cursor += counts[expert];
		active += counts[expert] != 0u ? 1u : 0u;
		counts[expert] = slot->host_expert_offsets[expert];
	}
	slot->host_expert_offsets[SPARK_MIMO25_MODEL_ROUTED_EXPERT_COUNT] = cursor;
	for (pair = 0; pair < pair_count; pair++)
	{
		expert = slot->host_moe_indices[pair];
		slot->host_grouped_rows[counts[expert]] = pair / SPARK_MIMO25_MODEL_EXPERTS_PER_TOKEN;
		slot->host_grouped_weight_slots[counts[expert]] = pair;
		counts[expert]++;
	}
	error = cudaMemcpyAsync(slot->grouped_rows_u32,slot->host_grouped_rows,(uint64_t)pair_count * sizeof(uint32_t),cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess )
		error = cudaMemcpyAsync(slot->grouped_weight_slots_u32,slot->host_grouped_weight_slots,(uint64_t)pair_count * sizeof(uint32_t),cudaMemcpyHostToDevice,stream);
	*active_out = active;
	return(SparkStageModuleCudaStatus(SPARK_MIMO25_MODULE_TAG,error,"moe_group"));
}

// One expert's group: gathered w1/w3 into dense slots (the tensor-core
// tile once the group is a full tile wide), silu-mul across the group,
// down projection, then the weighted OUTPUT scatter into the true rows.
static cudaError_t SparkMimo25ModuleRunExpertGroup(SparkMimo25ModuleSlot *slot, const SparkMimo25MoeWeights *moe, uint32_t expert, uint32_t offset, uint32_t count)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	SparkMimo25LinearView expert_view;
	const void *payload,*scale;
	uint64_t inter = SPARK_MIMO25_MODEL_EXPERT_INTERMEDIATE_DIMENSION,dim = SPARK_MIMO25_MODEL_HIDDEN_DIMENSION,bf16 = SPARK_MIMO25_MODEL_BF16_ELEMENT_BYTES;
	const uint32_t *row_map = slot->grouped_rows_u32 + offset;
	uint8_t *slot_gate = (uint8_t *)slot->moe_slot_gate_bf16 + (uint64_t)offset * inter * bf16;
	uint8_t *slot_up = (uint8_t *)slot->moe_slot_up_bf16 + (uint64_t)offset * inter * bf16;
	uint8_t *slot_out = (uint8_t *)slot->moe_slot_out_bf16 + (uint64_t)offset * dim * bf16;
	cudaError_t error;
	SparkMimo25ModuleExpertView(&expert_view,&moe->experts_w1,expert,inter,dim,&payload,&scale);
	error = count >= 16u ? SparkMimo25LaunchExpertTile(stream,&expert_view,payload,scale,slot->normalized_bf16,row_map,slot_gate,count) : SparkMimo25LaunchGatherLinear(stream,&expert_view,payload,scale,slot->normalized_bf16,row_map,slot_gate,count);
	if ( error == cudaSuccess )
	{
		SparkMimo25ModuleExpertView(&expert_view,&moe->experts_w3,expert,inter,dim,&payload,&scale);
		error = count >= 16u ? SparkMimo25LaunchExpertTile(stream,&expert_view,payload,scale,slot->normalized_bf16,row_map,slot_up,count) : SparkMimo25LaunchGatherLinear(stream,&expert_view,payload,scale,slot->normalized_bf16,row_map,slot_up,count);
	}
	if ( error == cudaSuccess )
		error = SparkMimo25LaunchSiluMul(stream,slot_gate,slot_up,count,(uint32_t)inter);
	if ( error == cudaSuccess )
	{
		SparkMimo25ModuleExpertView(&expert_view,&moe->experts_w2,expert,dim,inter,&payload,&scale);
		error = count >= 16u ? SparkMimo25LaunchExpertTile(stream,&expert_view,payload,scale,slot_up,0,slot_out,count) : SparkMimo25LaunchGatherLinear(stream,&expert_view,payload,scale,slot_up,0,slot_out,count);
	}
	if ( error == cudaSuccess )
		error = SparkMimo25LaunchScatterScaledAdd(stream,slot->ffn_accum_bf16,slot_out,row_map,slot->moe_weights_f32,slot->grouped_weight_slots_u32 + offset,count,(uint32_t)dim);
	return(error);
}

static SparkStatus SparkMimo25ModuleRunMoeRouted(SparkMimo25ModuleSlot *slot, const SparkMimo25MoeWeights *moe, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	uint32_t expert,offset,count,active = 0u;
	SparkStatus status;
	cudaError_t error;
	error = cudaMemcpyAsync(slot->host_moe_indices,slot->moe_indices_u32,(uint64_t)rows * SPARK_MIMO25_MODEL_EXPERTS_PER_TOKEN * sizeof(uint32_t),cudaMemcpyDeviceToHost,stream);
	if ( error == cudaSuccess )
		error = cudaStreamSynchronize(stream);
	if ( error != cudaSuccess )
		return(SparkStageModuleCudaStatus(SPARK_MIMO25_MODULE_TAG,error,"moe_readback"));
	status = SparkMimo25ModuleGroupByExpert(slot,rows,&active);
	if ( status != SPARK_STATUS_OK )
		return(status);
	for (expert = 0; error == cudaSuccess && expert < SPARK_MIMO25_MODEL_ROUTED_EXPERT_COUNT; expert++)
	{
		offset = slot->host_expert_offsets[expert];
		count = slot->host_expert_offsets[expert + 1u] - offset;
		if ( count == 0u )
			continue;
		error = SparkMimo25ModuleRunExpertGroup(slot,moe,expert,offset,count);
	}
	return(SparkStageModuleCudaStatus(SPARK_MIMO25_MODULE_TAG,error,"moe_routed"));
}

static SparkStatus SparkMimo25ModuleRunFfn(SparkMimo25ModuleSlot *slot, const SparkMimo25LayerWeights *layer, uint32_t layer_index, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	cudaError_t error;
	SparkStatus status;
	error = SparkMimo25LaunchRmsNorm(stream,slot->hidden_bf16,layer->ffn_norm_bf16,slot->normalized_bf16,rows,SPARK_MIMO25_MODEL_HIDDEN_DIMENSION,SPARK_MIMO25_MODEL_RMS_NORM_EPSILON);
	if ( error != cudaSuccess )
		return(SparkStageModuleCudaStatus(SPARK_MIMO25_MODULE_TAG,error,"ffn_norm"));
	if ( SparkMimo25ModelLayerIsMoe(layer_index) == 0u )
	{
		error = SparkMimo25LaunchLinear(stream,&layer->dense.w1,0,0,slot->normalized_bf16,slot->ffn_gate_bf16,rows);
		if ( error == cudaSuccess )
			error = SparkMimo25LaunchLinear(stream,&layer->dense.w3,0,0,slot->normalized_bf16,slot->ffn_up_bf16,rows);
		if ( error == cudaSuccess )
			error = SparkMimo25LaunchSiluMul(stream,slot->ffn_gate_bf16,slot->ffn_up_bf16,rows,SPARK_MIMO25_MODEL_DENSE_INTERMEDIATE_DIMENSION);
		if ( error == cudaSuccess )
			error = SparkMimo25LaunchLinear(stream,&layer->dense.w2,0,0,slot->ffn_up_bf16,slot->ffn_accum_bf16,rows);
		if ( error != cudaSuccess )
			return(SparkStageModuleCudaStatus(SPARK_MIMO25_MODULE_TAG,error,"dense_mlp"));
	}
	else
	{
		error = SparkMimo25LaunchGateScores(stream,&layer->moe.gate,slot->normalized_bf16,slot->moe_scores_f32,rows);
		if ( error == cudaSuccess )
			error = SparkMimo25LaunchGateSelect(stream,slot->moe_scores_f32,layer->moe.gate_bias_f32,rows,SPARK_MIMO25_MODEL_ROUTED_EXPERT_COUNT,SPARK_MIMO25_MODEL_EXPERTS_PER_TOKEN,SPARK_MIMO25_MODEL_ROUTER_NORM_EPSILON,SPARK_MIMO25_MODEL_ROUTED_SCALING_FACTOR,slot->moe_indices_u32,slot->moe_weights_f32);
		if ( error == cudaSuccess )
			error = cudaMemsetAsync(slot->ffn_accum_bf16,0,(uint64_t)rows * SPARK_MIMO25_MODEL_HIDDEN_DIMENSION * SPARK_MIMO25_MODEL_BF16_ELEMENT_BYTES,stream);
		if ( error != cudaSuccess )
			return(SparkStageModuleCudaStatus(SPARK_MIMO25_MODULE_TAG,error,"moe_gate"));
		status = SparkMimo25ModuleRunMoeRouted(slot,&layer->moe,rows);
		if ( status != SPARK_STATUS_OK )
			return(status);
	}
	error = SparkMimo25LaunchResidualAdd(stream,slot->hidden_bf16,slot->ffn_accum_bf16,rows,SPARK_MIMO25_MODEL_HIDDEN_DIMENSION);
	return(SparkStageModuleCudaStatus(SPARK_MIMO25_MODULE_TAG,error,"ffn_residual"));
}

/*
 * dspark aux-layer tap: after a listed layer completes, the rows' hidden
 * is strided into the caller's fused arena - one pitched device copy per
 * tap, arena row = [aux0|aux1|...]. The list must sit inside the slice
 * so a misconfigured serving plane fails before the first frame runs.
 */
static SparkStatus SparkMimo25ModuleValidateTap(const SparkMimo25ModuleState *state, const SparkMimo25ResidentDecodeStageFrameContext *context)
{
	uint32_t tap;
	if ( (context->flags & SPARK_MIMO25_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_TAP) == 0u )
		return(SPARK_STATUS_OK);
	if ( context->tap_layer_count == 0u || context->tap_layer_count > SPARK_MIMO25_RESIDENT_DECODE_STAGE_MAX_TAP_LAYERS || context->tap_layer_indices == 0 || context->tap_arena_bf16 == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	for (tap = 0; tap < context->tap_layer_count; tap++)
		if ( context->tap_layer_indices[tap] < state->first_layer_index || context->tap_layer_indices[tap] >= state->first_layer_index + state->layer_count )
		{
			fprintf(stderr,"%s tap_layer_out_of_slice layer=%u slice=%u+%u\n",SPARK_MIMO25_MODULE_TAG,context->tap_layer_indices[tap],state->first_layer_index,state->layer_count);
			return(SPARK_STATUS_INVALID_ARGUMENT);
		}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkMimo25ModuleTapLayer(SparkMimo25ModuleSlot *slot, const SparkMimo25ResidentDecodeStageFrameContext *context, uint32_t layer_index, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	uint64_t dim = SPARK_MIMO25_MODEL_HIDDEN_DIMENSION,bf16 = SPARK_MIMO25_MODEL_BF16_ELEMENT_BYTES;
	uint64_t arena_pitch;
	uint32_t tap;
	cudaError_t error = cudaSuccess;
	if ( context == 0 || (context->flags & SPARK_MIMO25_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_TAP) == 0u )
		return(SPARK_STATUS_OK);
	arena_pitch = (uint64_t)context->tap_layer_count * dim * bf16;
	for (tap = 0; error == cudaSuccess && tap < context->tap_layer_count; tap++)
	{
		if ( context->tap_layer_indices[tap] != layer_index )
			continue;
		error = cudaMemcpy2DAsync((uint8_t *)context->tap_arena_bf16 + (uint64_t)tap * dim * bf16,arena_pitch,slot->hidden_bf16,dim * bf16,dim * bf16,rows,cudaMemcpyDeviceToDevice,stream);
	}
	return(SparkStageModuleCudaStatus(SPARK_MIMO25_MODULE_TAG,error,"hidden_tap"));
}

static SparkStatus SparkMimo25ModuleRunLayer(SparkMimo25ModuleState *state, SparkMimo25ModuleSlot *slot, const SparkMimo25DecodeBatchView *batch, uint32_t layer_index, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	const SparkMimo25LayerWeights *layer = &state->layers[layer_index];
	cudaError_t error;
	SparkStatus status;
	error = SparkMimo25LaunchRmsNorm(stream,slot->hidden_bf16,layer->attn.attn_norm_bf16,slot->normalized_bf16,rows,SPARK_MIMO25_MODEL_HIDDEN_DIMENSION,SPARK_MIMO25_MODEL_RMS_NORM_EPSILON);
	if ( error != cudaSuccess )
		return(SparkStageModuleCudaStatus(SPARK_MIMO25_MODULE_TAG,error,"attn_norm"));
	status = SparkMimo25ModuleRunAttention(state,slot,layer,batch,layer_index,rows);
	if ( status != SPARK_STATUS_OK )
		return(status);
	return(SparkMimo25ModuleRunFfn(slot,layer,layer_index,rows));
}

/*
 * MTP draft chain, head stage only. Step d runs draft layer d at
 * positions p+1+d on the layer's own SWA ring: the previous step's token
 * (step zero: the main argmax) embeds through enorm, the chained hidden
 * (step zero: the trunk output, pre final norm) through hnorm, eh_proj
 * folds the concat, then the SWA attention block with sink and the dense
 * MLP, the per-layer final norm feeding the shared head for the step's
 * draft token. The hidden chains PRE-final-norm, the DeepSeek-V3
 * convention the checkpoint shapes follow.
 */
static SparkStatus SparkMimo25ModuleMtpStageStep(SparkMimo25ModuleState *state, SparkMimo25ModuleSlot *slot, const SparkMimo25DecodeBatchView *batch, uint32_t step, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	const SparkMimo25MtpLayerWeights *mtp = &state->mtp_layers[step];
	uint64_t dim = SPARK_MIMO25_MODEL_HIDDEN_DIMENSION,bf16 = SPARK_MIMO25_MODEL_BF16_ELEMENT_BYTES;
	uint32_t row;
	cudaError_t error;
	for (row = 0; row < rows; row++)
		slot->host_mtp_positions[row] = batch->row_positions[row] + 1u + step;
	error = cudaMemcpyAsync(slot->mtp_positions_u64,slot->host_mtp_positions,(uint64_t)rows * sizeof(uint64_t),cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess )
		error = SparkMimo25LaunchEmbeddingGather(stream,step == 0u ? slot->output_token_ids : slot->mtp_step_ids_u32,state->token_embedding_bf16,slot->mtp_scratch_bf16,rows,(uint32_t)dim);
	if ( error == cudaSuccess )
		error = SparkMimo25LaunchRmsNorm(stream,slot->mtp_scratch_bf16,mtp->enorm_bf16,slot->mtp_scratch_bf16,rows,(uint32_t)dim,SPARK_MIMO25_MODEL_RMS_NORM_EPSILON);
	if ( error == cudaSuccess )
		error = cudaMemcpy2DAsync(slot->mtp_concat_bf16,2u * dim * bf16,slot->mtp_scratch_bf16,dim * bf16,dim * bf16,rows,cudaMemcpyDeviceToDevice,stream);
	if ( error == cudaSuccess )
		error = SparkMimo25LaunchRmsNorm(stream,step == 0u ? slot->hidden_bf16 : slot->mtp_hidden_bf16,mtp->hnorm_bf16,slot->mtp_scratch_bf16,rows,(uint32_t)dim,SPARK_MIMO25_MODEL_RMS_NORM_EPSILON);
	if ( error == cudaSuccess )
		error = cudaMemcpy2DAsync((uint8_t *)slot->mtp_concat_bf16 + dim * bf16,2u * dim * bf16,slot->mtp_scratch_bf16,dim * bf16,dim * bf16,rows,cudaMemcpyDeviceToDevice,stream);
	if ( error == cudaSuccess )
		error = SparkMimo25LaunchLinear(stream,&mtp->eh_proj,0,0,slot->mtp_concat_bf16,slot->mtp_hidden_bf16,rows);
	return(SparkStageModuleCudaStatus(SPARK_MIMO25_MODULE_TAG,error,"mtp_stage"));
}

static SparkStatus SparkMimo25ModuleMtpAttention(SparkMimo25ModuleState *state, SparkMimo25ModuleSlot *slot, uint32_t step, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	const SparkMimo25AttnWeights *attn = &state->mtp_layers[step].attn;
	uint64_t lanes = state->max_active_sequence_count,bf16 = SPARK_MIMO25_MODEL_BF16_ELEMENT_BYTES;
	uint64_t k_slot = (uint64_t)SPARK_MIMO25_MODEL_SWA_KV_HEAD_COUNT * SPARK_MIMO25_MODEL_ATTN_HEAD_DIMENSION;
	uint64_t v_slot = (uint64_t)SPARK_MIMO25_MODEL_SWA_KV_HEAD_COUNT * SPARK_MIMO25_MODEL_ATTN_VALUE_DIMENSION;
	uint8_t *k_cache = (uint8_t *)state->k_mtp_cache_bf16 + (uint64_t)step * lanes * state->k_swa_lane_stride * bf16;
	uint8_t *v_cache = (uint8_t *)state->v_mtp_cache_bf16 + (uint64_t)step * lanes * state->v_swa_lane_stride * bf16;
	uint32_t k_offset = SPARK_MIMO25_MODEL_Q_DIMENSION,v_offset = k_offset + SPARK_MIMO25_MODEL_SWA_KV_HEAD_COUNT * SPARK_MIMO25_MODEL_ATTN_HEAD_DIMENSION;
	cudaError_t error;
	error = SparkMimo25LaunchLinear(stream,&attn->qkv,0,0,slot->normalized_bf16,slot->qkv_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkMimo25LaunchScaleSlice(stream,slot->qkv_bf16,SPARK_MIMO25_MODEL_SWA_QKV_DIMENSION,v_offset,SPARK_MIMO25_MODEL_SWA_KV_HEAD_COUNT * SPARK_MIMO25_MODEL_ATTN_VALUE_DIMENSION,SPARK_MIMO25_MODEL_ATTN_VALUE_SCALE,rows);
	if ( error == cudaSuccess )
		error = SparkMimo25LaunchRope(stream,slot->qkv_bf16,SPARK_MIMO25_MODEL_SWA_QKV_DIMENSION,0u,state->swa_freqs_f32,slot->mtp_positions_u64,rows,SPARK_MIMO25_MODEL_ATTN_HEAD_COUNT,SPARK_MIMO25_MODEL_ATTN_HEAD_DIMENSION,SPARK_MIMO25_MODEL_ATTN_ROPE_DIMENSION,0u);
	if ( error == cudaSuccess )
		error = SparkMimo25LaunchRope(stream,slot->qkv_bf16,SPARK_MIMO25_MODEL_SWA_QKV_DIMENSION,k_offset,state->swa_freqs_f32,slot->mtp_positions_u64,rows,SPARK_MIMO25_MODEL_SWA_KV_HEAD_COUNT,SPARK_MIMO25_MODEL_ATTN_HEAD_DIMENSION,SPARK_MIMO25_MODEL_ATTN_ROPE_DIMENSION,0u);
	error = SparkMimo25LaunchCacheScatter(stream,slot->qkv_bf16,SPARK_MIMO25_MODEL_SWA_QKV_DIMENSION,k_offset,(uint32_t)(k_slot),v_offset,(uint32_t)(v_slot),k_cache,v_cache,state->k_swa_lane_stride,state->v_swa_lane_stride,slot->row_lane_indices,slot->mtp_positions_u64,SPARK_MIMO25_MODEL_SLIDING_WINDOW_TOKENS,rows);
	if ( error == cudaSuccess )
		error = SparkMimo25LaunchAttnDecode(stream,slot->qkv_bf16,SPARK_MIMO25_MODEL_SWA_QKV_DIMENSION,k_cache,v_cache,state->k_swa_lane_stride,state->v_swa_lane_stride,k_slot,v_slot,slot->row_lane_indices,slot->mtp_positions_u64,attn->sink_f32,1.0f / sqrtf((float)SPARK_MIMO25_MODEL_ATTN_HEAD_DIMENSION),slot->attn_out_bf16,rows,SPARK_MIMO25_MODEL_ATTN_HEAD_COUNT,SPARK_MIMO25_MODEL_ATTN_HEAD_COUNT / SPARK_MIMO25_MODEL_SWA_KV_HEAD_COUNT,SPARK_MIMO25_MODEL_ATTN_HEAD_DIMENSION,SPARK_MIMO25_MODEL_ATTN_VALUE_DIMENSION,SPARK_MIMO25_MODEL_SLIDING_WINDOW_TOKENS);
	if ( error == cudaSuccess )
		error = SparkMimo25LaunchLinear(stream,&attn->o,0,0,slot->attn_out_bf16,slot->delta_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkMimo25LaunchResidualAdd(stream,slot->mtp_hidden_bf16,slot->delta_bf16,rows,SPARK_MIMO25_MODEL_HIDDEN_DIMENSION);
	return(SparkStageModuleCudaStatus(SPARK_MIMO25_MODULE_TAG,error,"mtp_attention"));
}

static SparkStatus SparkMimo25ModuleMtpFinishStep(SparkMimo25ModuleState *state, SparkMimo25ModuleSlot *slot, uint32_t step, uint32_t depth, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	const SparkMimo25MtpLayerWeights *mtp = &state->mtp_layers[step];
	cudaError_t error;
	error = SparkMimo25LaunchRmsNorm(stream,slot->mtp_hidden_bf16,mtp->ffn_norm_bf16,slot->normalized_bf16,rows,SPARK_MIMO25_MODEL_HIDDEN_DIMENSION,SPARK_MIMO25_MODEL_RMS_NORM_EPSILON);
	if ( error == cudaSuccess )
		error = SparkMimo25LaunchLinear(stream,&mtp->mlp.w1,0,0,slot->normalized_bf16,slot->ffn_gate_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkMimo25LaunchLinear(stream,&mtp->mlp.w3,0,0,slot->normalized_bf16,slot->ffn_up_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkMimo25LaunchSiluMul(stream,slot->ffn_gate_bf16,slot->ffn_up_bf16,rows,SPARK_MIMO25_MODEL_DENSE_INTERMEDIATE_DIMENSION);
	if ( error == cudaSuccess )
		error = SparkMimo25LaunchLinear(stream,&mtp->mlp.w2,0,0,slot->ffn_up_bf16,slot->ffn_accum_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkMimo25LaunchResidualAdd(stream,slot->mtp_hidden_bf16,slot->ffn_accum_bf16,rows,SPARK_MIMO25_MODEL_HIDDEN_DIMENSION);
	if ( error == cudaSuccess )
		error = SparkMimo25LaunchRmsNorm(stream,slot->mtp_hidden_bf16,mtp->final_norm_bf16,slot->normalized_bf16,rows,SPARK_MIMO25_MODEL_HIDDEN_DIMENSION,SPARK_MIMO25_MODEL_RMS_NORM_EPSILON);
	if ( error == cudaSuccess )
		error = SparkMimo25LaunchHeadTiledArgmax(stream,slot->normalized_bf16,state->lm_head_weight_bf16,slot->head_logits_bf16,slot->mtp_step_ids_u32,rows,SPARK_MIMO25_MODEL_VOCAB_COUNT,SPARK_MIMO25_MODEL_HIDDEN_DIMENSION);
	if ( error == cudaSuccess )
		error = cudaMemcpy2DAsync(slot->mtp_draft_ids_u32 + step,(uint64_t)depth * sizeof(uint32_t),slot->mtp_step_ids_u32,sizeof(uint32_t),sizeof(uint32_t),rows,cudaMemcpyDeviceToDevice,stream);
	return(SparkStageModuleCudaStatus(SPARK_MIMO25_MODULE_TAG,error,"mtp_finish"));
}

static SparkStatus SparkMimo25ModuleRunMtp(SparkMimo25ModuleState *state, SparkMimo25ModuleSlot *slot, SparkMimo25ResidentDecodeStageFrameContext *context, const SparkMimo25DecodeBatchView *batch, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	uint32_t step,depth = state->mtp_armed;
	SparkStatus status = SPARK_STATUS_OK;
	cudaError_t error;
	if ( depth == 0u )
		return(SPARK_STATUS_OK);
	if ( context->mtp_draft_depth != depth || context->mtp_draft_tokens == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	for (step = 0; status == SPARK_STATUS_OK && step < depth; step++)
	{
		status = SparkMimo25ModuleMtpStageStep(state,slot,batch,step,rows);
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleCudaStatus(SPARK_MIMO25_MODULE_TAG,SparkMimo25LaunchRmsNorm(stream,slot->mtp_hidden_bf16,state->mtp_layers[step].attn.attn_norm_bf16,slot->normalized_bf16,rows,SPARK_MIMO25_MODEL_HIDDEN_DIMENSION,SPARK_MIMO25_MODEL_RMS_NORM_EPSILON),"mtp_norm");
		if ( status == SPARK_STATUS_OK )
			status = SparkMimo25ModuleMtpAttention(state,slot,step,rows);
		if ( status == SPARK_STATUS_OK )
			status = SparkMimo25ModuleMtpFinishStep(state,slot,step,depth,rows);
	}
	if ( status != SPARK_STATUS_OK )
		return(status);
	error = cudaMemcpyAsync(context->mtp_draft_tokens,slot->mtp_draft_ids_u32,(uint64_t)rows * depth * sizeof(uint32_t),cudaMemcpyDeviceToHost,stream);
	return(SparkStageModuleCudaStatus(SPARK_MIMO25_MODULE_TAG,error,"mtp_draft_out"));
}

static SparkStatus SparkMimo25ModuleFinish(SparkMimo25ModuleState *state, SparkMimo25ModuleSlot *slot, SparkMimo25ResidentDecodeStageFrameContext *context, SparkModelDriverFrame *frame, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	uint32_t out_index = state->owns_embedding != 0u ? 1u : 0u;
	SparkStatus status = SPARK_STATUS_OK;
	cudaError_t error;
	if ( state->owns_final_head != 0u )
	{
		error = SparkMimo25LaunchRmsNorm(stream,slot->hidden_bf16,state->final_norm_weight_bf16,slot->normalized_bf16,rows,SPARK_MIMO25_MODEL_HIDDEN_DIMENSION,SPARK_MIMO25_MODEL_RMS_NORM_EPSILON);
		if ( error == cudaSuccess )
			error = SparkMimo25LaunchHeadTiledArgmax(stream,slot->normalized_bf16,state->lm_head_weight_bf16,slot->head_logits_bf16,slot->output_token_ids,rows,SPARK_MIMO25_MODEL_VOCAB_COUNT,SPARK_MIMO25_MODEL_HIDDEN_DIMENSION);
		if ( error == cudaSuccess )
			error = cudaMemcpyAsync(frame->buffers[out_index].address,slot->output_token_ids,(uint64_t)rows * sizeof(uint32_t),cudaMemcpyDeviceToHost,stream);
		status = SparkStageModuleCudaStatus(SPARK_MIMO25_MODULE_TAG,error,"head");
	}
	else
	{
		if ( context->hidden_output_send_function == 0 || context->hidden_output_transport_session == 0 )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		context->hidden_output_packet.active_sequence_count = rows;
		context->hidden_output_packet.hidden_dimension = SPARK_MIMO25_MODEL_HIDDEN_DIMENSION;
		context->hidden_output_packet.bytes_per_sequence = SPARK_MIMO25_MODEL_HIDDEN_DIMENSION * SPARK_MIMO25_MODEL_BF16_ELEMENT_BYTES;
		context->hidden_output_packet.hidden_bf16 = slot->hidden_bf16;
		context->hidden_output_packet.cuda_stream = stream;
		context->hidden_output_packet.sideband_payload = 0;
		context->hidden_output_packet.sideband_kind = 0u;
		context->hidden_output_packet.sideband_bytes_per_sequence = 0u;
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleCudaStatus(SPARK_MIMO25_MODULE_TAG,cudaStreamSynchronize(stream),"sync");
	if ( status == SPARK_STATUS_OK && state->owns_final_head == 0u )
		status = context->hidden_output_send_function(context->hidden_output_transport_session,&context->hidden_output_packet);
	return(status);
}

SparkStatus SparkMimo25ResidentDecodeStageExecute(void *module_state, SparkModelDriverFrame *frame)
{
	SparkMimo25ModuleState *state = (SparkMimo25ModuleState *)module_state;
	SparkMimo25ResidentDecodeStageFrameContext *context = 0;
	const SparkMimo25DecodeBatchView *batch;
	SparkMimo25ModuleSlot *slot;
	uint32_t slot_index = 0u,rows,layer;
	SparkStatus status;
	if ( state == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkMimo25ModuleValidateFrame(state,frame,(const SparkMimo25ResidentDecodeStageFrameContext **)&context);
	if ( status != SPARK_STATUS_OK )
		return(status);
	batch = context->decode_batch;
	rows = batch->row_count;
	status = SparkMimo25ModuleValidateTap(state,context);
	if ( status != SPARK_STATUS_OK )
		return(status);
	status = SparkStageModuleSlotClaim(state->slot_states,state->pipeline_slot_count,&slot_index);
	if ( status != SPARK_STATUS_OK )
		return(status);
	slot = &state->slots[slot_index];
	status = SparkMimo25ModuleStageRows(state,batch,slot);
	if ( status == SPARK_STATUS_OK )
		status = SparkMimo25ModuleBeginHidden(state,slot,context,frame,rows);
	for (layer = state->first_layer_index; status == SPARK_STATUS_OK && layer < state->first_layer_index + state->layer_count; layer++)
	{
		status = SparkMimo25ModuleRunLayer(state,slot,batch,layer,rows);
		if ( status == SPARK_STATUS_OK )
			status = SparkMimo25ModuleTapLayer(slot,context,layer,rows);
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkMimo25ModuleFinish(state,slot,context,frame,rows);
	if ( status == SPARK_STATUS_OK && state->owns_final_head != 0u && state->mtp_armed != 0u )
	{
		status = SparkMimo25ModuleRunMtp(state,slot,context,batch,rows);
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleCudaStatus(SPARK_MIMO25_MODULE_TAG,cudaStreamSynchronize((cudaStream_t)slot->cuda_stream),"mtp_sync");
	}
	SparkStageModuleSlotRelease(state->slot_states,slot_index);
	if ( status != SPARK_STATUS_OK )
		return(status);
	atomic_fetch_add(&state->frames_executed,1u);
	atomic_fetch_add(&state->tokens_emitted,rows);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkMimo25ResidentDecodeStageAdmit(void *module_state, const SparkModelDriverAdmissionRequest *request, SparkModelDriverAdmissionDecision *decision)
{
	SparkMimo25ModuleState *state = (SparkMimo25ModuleState *)module_state;
	if ( state == 0 || request == 0 || decision == 0 || request->descriptor_bytes != (uint32_t)sizeof(*request) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(decision,0,sizeof(*decision));
	decision->descriptor_bytes = (uint32_t)sizeof(*decision);
	decision->accepted = 1u;
	decision->available_dispatch_slot_count = state->pipeline_slot_count;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkMimo25ResidentDecodeStageSnapshot(void *module_state, uint32_t program_id, SparkModelDriverRuntimeSnapshot *snapshot)
{
	SparkMimo25ModuleState *state = (SparkMimo25ModuleState *)module_state;
	if ( state == 0 || snapshot == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(snapshot,0,sizeof(*snapshot));
	snapshot->descriptor_bytes = (uint32_t)sizeof(*snapshot);
	snapshot->program_id = program_id;
	snapshot->submitted_count = atomic_load(&state->frames_executed);
	snapshot->completed_count = atomic_load(&state->frames_executed);
	snapshot->resident_token_count = atomic_load(&state->tokens_emitted);
	snapshot->kv_token_capacity = (uint64_t)state->max_active_sequence_count * state->max_sequence_positions;
	snapshot->available_dispatch_slot_count = state->pipeline_slot_count;
	return(SPARK_STATUS_OK);
}

void SparkMimo25ResidentDecodeStageDestroy(void *module_state)
{
	SparkMimo25ModuleState *state = (SparkMimo25ModuleState *)module_state;
	uint32_t slot_index;
	if ( state == 0 )
		return;
	for (slot_index = 0; slot_index < state->pipeline_slot_count; slot_index++)
		if ( state->slots[slot_index].cuda_stream != 0 )
			cudaStreamDestroy((cudaStream_t)state->slots[slot_index].cuda_stream);
	SparkStageModuleLedgerRelease(&state->ledger);
	free(state);
}

SparkStatus SparkMimo25ResidentDecodeStageInitialize(const SparkFirmwareModuleConfiguration *configuration, const SparkFirmwareModuleHostServices *host_services, void **module_state)
{
	SparkMimo25ModuleState *state;
	const char *pack_path = 0;
	SparkStatus status;
	uint32_t slot_index;
	(void)host_services;
	if ( configuration == 0 || module_state == 0 || configuration->abi_version != SPARK_FIRMWARE_MODULE_ABI_VERSION )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	state = (SparkMimo25ModuleState *)calloc(1u,sizeof(*state));
	if ( state == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	state->ledger.module_tag = SPARK_MIMO25_MODULE_TAG;
	status = SparkMimo25ModuleConfigure(state);
	if ( status == SPARK_STATUS_OK )
		status = SparkMimo25ModuleValidateSlice(state);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentText(SPARK_MIMO25_MODULE_TAG,"SPARK_MIMO25_STAGE_PACK_PATH",&pack_path);
	if ( status == SPARK_STATUS_OK )
	{
		SparkMimo25ModuleBuildOrdinals(state);
		status = SparkMimo25ModuleLoadPack(state,pack_path);
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkMimo25ModuleUploadFreqs(state);
	if ( status == SPARK_STATUS_OK )
		status = SparkMimo25ModuleAllocatePools(state);
	for (slot_index = 0; status == SPARK_STATUS_OK && slot_index < state->pipeline_slot_count; slot_index++)
		status = SparkMimo25ModuleAllocateSlot(state,&state->slots[slot_index]);
	if ( status != SPARK_STATUS_OK )
	{
		SparkMimo25ResidentDecodeStageDestroy(state);
		return(status);
	}
	fprintf(stderr,"%s ready stage=%u/%u slice=%u+%u full=%u swa=%u lanes=%u max_seq=%u device_gib=%.1f\n",SPARK_MIMO25_MODULE_TAG,state->stage_index,state->stage_count,state->first_layer_index,state->layer_count,state->full_layer_count,state->swa_layer_count,state->max_active_sequence_count,state->max_sequence_positions,(double)state->ledger.device_bytes_resident / (1024.0 * 1024.0 * 1024.0));
	*module_state = state;
	return(SPARK_STATUS_OK);
}
