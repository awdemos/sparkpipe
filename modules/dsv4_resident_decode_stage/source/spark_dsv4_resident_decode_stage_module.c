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

#include "sparkpipe/spark_dsv4_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_stage_module_common.h"
#include "spark_dsv4_stagepack_format.h"

/*
 * DeepSeek V4 resident decode stage host module, PP-Nx native, one variant
 * per build through the -include'd model header.
 *
 * One process is one STAGE over a layer slice; the pack must declare that
 * slice and the computed tensor inventory exactly. The stage boundary
 * carries the FOUR hyper-connection streams (hc_mult x hidden per row);
 * stage zero expands the embedding, the head stage's sigmoid reduction is
 * the only collapse. Execute serves DECODE frames: one token per lane
 * across up to max_active lanes, every attention kind, both router paths,
 * the full mHC chain. Prefill and MTP execution refuse with their own
 * statuses until those passes land; the MTP tensors load and verify so
 * the pack contract is already final.
 *
 * The hash router pins to token ids, which exist only where the embedding
 * lives: a slice that starts inside the hash range without owning the
 * embedding is refused at configuration, not discovered at runtime.
 *
 * Caches are dense per lane: window ring 128 slots always; compress
 * layers append max_seq/ratio compressed slots behind the window in the
 * same lane block (the reference's win offset); CSA layers keep an
 * indexer stream of max_seq/4 rotated 128-wide entries and both
 * compressor f32 states. The paged migration rides the family PP pass.
 */

#define SPARK_DSV4_MODULE_TAG "dsv4_stage"

typedef struct SparkDsv4LayerWeights
{
	const void *attn_norm_bf16;
	const void *ffn_norm_bf16;
	SparkDsv4AttnWeights attn;
	SparkDsv4CompressorWeights compressor;
	SparkDsv4IndexerWeights indexer;
	SparkDsv4MoeWeights moe;
	SparkDsv4HcWeights hc;
} SparkDsv4LayerWeights;

typedef struct SparkDsv4ModuleSlot
{
	void *cuda_stream;
	uint32_t *input_token_ids;
	uint32_t *output_token_ids;
	uint32_t *row_lane_indices;
	uint64_t *row_positions;
	uint64_t *row_emit_positions;
	uint64_t *row_emit_positions_hca;
	uint32_t *slot_counts;
	int32_t *topk_idxs;
	void *streams_bf16;
	void *residual_bf16;
	void *reduced_bf16;
	void *normalized_bf16;
	void *qr_bf16;
	void *q_bf16;
	void *kv_bf16;
	void *attn_out_bf16;
	void *o_ranks_bf16;
	void *delta_bf16;
	void *compress_kv_bf16;
	void *compress_score_bf16;
	float *compress_kv_f32;
	float *compress_score_f32;
	void *emit_bf16;
	uint32_t *emitted_u32;
	void *index_q_bf16;
	void *index_weights_bf16;
	float *index_weights_f32;
	float *index_scores_f32;
	float *mixes_f32;
	float *pre_f32;
	float *post_f32;
	float *comb_f32;
	float *moe_scores_f32;
	uint32_t *moe_indices_u32;
	float *moe_weights_f32;
	void *ffn_gate_bf16;
	void *ffn_up_bf16;
	void *ffn_delta_bf16;
	void *ffn_accum_bf16;
	uint32_t *grouped_rows_u32;
	uint32_t *grouped_weight_slots_u32;
	void *moe_slot_gate_bf16;
	void *moe_slot_up_bf16;
	void *moe_slot_out_bf16;
	void *head_logits_bf16;
	uint32_t *head_candidate_ids_u32;
	uint32_t *head_candidate_counts_u32;
	uint32_t *host_moe_indices;
	uint32_t *host_grouped_rows;
	uint32_t *host_grouped_weight_slots;
	uint32_t host_expert_offsets[SPARK_DSV4_MODEL_ROUTED_EXPERT_COUNT + 1u];
} SparkDsv4ModuleSlot;

typedef struct SparkDsv4ModuleState
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
	uint32_t compress_layer_count;
	uint32_t csa_layer_count;
	uint32_t topk_column_count;
	uint32_t index_slot_capacity;
	float hc_head_scale_value;
	uint32_t layer_local_by_layer[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_LAYER_COUNT];
	uint32_t compress_ordinal_by_layer[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_LAYER_COUNT];
	uint32_t csa_ordinal_by_layer[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_LAYER_COUNT];
	uint64_t layer_seen_bits[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_LAYER_COUNT];
	uint64_t mtp_seen_bits;
	uint64_t global_seen_bits;
	SparkDsv4LayerWeights layers[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_LAYER_COUNT];
	SparkDsv4LayerWeights mtp_layer;
	SparkDsv4MtpWeights mtp;
	const void *token_embedding_bf16;
	const void *final_norm_weight_bf16;
	const void *lm_head_weight_bf16;
	uint8_t *head_shadow_payload;
	uint8_t *head_shadow_scale;
	float *head_error_norm_f32;
	const float *hc_head_fn_f32;
	const float *hc_head_base_f32;
	const float *hc_head_scale_f32;
	float *base_freqs_f32;
	float *compress_freqs_f32;
	void *kv_cache_bf16;
	uint64_t cache_layer_lane_stride;
	uint64_t cache_lane_block_elements;
	void *index_cache_bf16;
	uint64_t index_lane_stride;
	float *compress_kv_state_f32;
	float *compress_score_state_f32;
	uint64_t compress_state_lane_stride;
	float *index_kv_state_f32;
	float *index_score_state_f32;
	uint64_t index_state_lane_stride;
	int32_t *host_topk_idxs;
	uint32_t host_slot_counts[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	SparkDsv4ModuleSlot slots[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
	atomic_uint slot_states[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
	atomic_ullong frames_executed;
	atomic_ullong tokens_emitted;
} SparkDsv4ModuleState;

extern cudaError_t SparkDsv4LaunchRmsNorm(cudaStream_t stream, const void *input_bf16, const void *gain_bf16, void *output_bf16, uint32_t row_count, uint32_t dimension, float epsilon);
extern cudaError_t SparkDsv4LaunchLinear(cudaStream_t stream, const SparkDsv4LinearView *view, const void *input_bf16, void *output_bf16, uint32_t row_count);
extern cudaError_t SparkDsv4LaunchStridedLinear(cudaStream_t stream, const SparkDsv4LinearView *view, const void *payload, const uint8_t *scale, const void *input_bf16, uint64_t input_row_stride, uint32_t input_offset, void *output_bf16, uint64_t output_row_stride, uint32_t output_offset, uint32_t row_count);
extern cudaError_t SparkDsv4LaunchEmbeddingGather(cudaStream_t stream, const uint32_t *token_ids, const void *embedding_bf16, void *hidden_bf16, uint32_t row_count, uint32_t hidden_dimension);
extern cudaError_t SparkDsv4LaunchHeadArgmax(cudaStream_t stream, const void *hidden_bf16, const void *head_weight_bf16, const uint32_t *token_ids, uint32_t *output_token_ids, uint32_t row_count, uint32_t candidate_count, uint32_t hidden_dimension);
extern cudaError_t SparkDsv4LaunchHeadShadowQuantize(cudaStream_t stream, const void *head_bf16, uint8_t *shadow_payload, uint8_t *shadow_scale, float *error_norm, uint32_t candidate_count, uint32_t hidden_dimension);
extern cudaError_t SparkDsv4LaunchHeadScreenedArgmax(cudaStream_t stream, const void *hidden_bf16, const void *head_weight_bf16, const uint8_t *shadow_payload, const uint8_t *shadow_scale, const float *error_norm, void *logits_bf16, uint32_t *candidate_ids, uint32_t *candidate_counts, uint32_t *output_token_ids, uint32_t row_count, uint32_t candidate_count, uint32_t hidden_dimension);
extern cudaError_t SparkDsv4LaunchQuantSim(cudaStream_t stream, void *data_bf16, uint32_t row_count, uint32_t row_stride, uint32_t width, uint32_t block, uint32_t fp4);
extern cudaError_t SparkDsv4LaunchRope(cudaStream_t stream, void *data_bf16, const float *freqs_f32, const uint64_t *row_positions, uint32_t row_count, uint32_t head_count, uint32_t head_dim, uint32_t rope_dim, uint32_t inverse);
extern cudaError_t SparkDsv4LaunchQueryHeadRms(cudaStream_t stream, void *data_bf16, uint32_t row_count, uint32_t head_count, uint32_t head_dim, float epsilon);
extern cudaError_t SparkDsv4LaunchHadamard(cudaStream_t stream, void *data_bf16, uint32_t vector_count, uint32_t width);
extern cudaError_t SparkDsv4LaunchSparseAttn(cudaStream_t stream, const void *q_bf16, const void *kv_cache_bf16, uint64_t lane_stride_elements, const uint32_t *row_lane_indices, const int32_t *topk_idxs, uint32_t topk, const float *sink_f32, float scale, void *out_bf16, uint32_t row_count, uint32_t head_count, uint32_t head_dim);
extern cudaError_t SparkDsv4LaunchWiden(cudaStream_t stream, const void *input_bf16, float *output_f32, uint32_t row_count, uint32_t width, float scale);
extern cudaError_t SparkDsv4LaunchApeAdd(cudaStream_t stream, float *score_f32, const float *ape_f32, const uint64_t *row_positions, uint32_t row_count, uint32_t ratio, uint32_t channels);
extern cudaError_t SparkDsv4LaunchCompressStep(cudaStream_t stream, const float *kv_f32, const float *score_f32, float *kv_state_f32, float *score_state_f32, uint64_t state_lane_stride, const uint32_t *row_lane_indices, const uint64_t *row_positions, uint32_t row_count, uint32_t ratio, uint32_t overlap, uint32_t width, void *emit_bf16, uint32_t *emitted);
extern cudaError_t SparkDsv4LaunchGateScores(cudaStream_t stream, const SparkDsv4LinearView *gate, const void *input_bf16, float *scores_f32, uint32_t row_count);
extern cudaError_t SparkDsv4LaunchGateSelect(cudaStream_t stream, const float *scores_f32, const float *bias_f32, const uint32_t *tid2eid_u32, const uint32_t *token_ids, uint32_t row_count, uint32_t expert_count, uint32_t topk, float route_scale, uint32_t *indices_u32, float *weights_f32);
extern cudaError_t SparkDsv4LaunchSwigluClamp(cudaStream_t stream, const void *gate_bf16, void *up_bf16, uint32_t row_count, uint32_t width, float limit, const float *row_weights_f32, const uint32_t *weight_map);
extern cudaError_t SparkDsv4LaunchGatherLinear(cudaStream_t stream, const SparkDsv4LinearView *view, const void *input_bf16, const uint32_t *input_row_map, void *output_bf16, uint32_t slot_count);
extern cudaError_t SparkDsv4LaunchExpertTile(cudaStream_t stream, const SparkDsv4LinearView *view, const void *input_bf16, const uint32_t *input_row_map, void *output_bf16, uint32_t slot_count);
extern cudaError_t SparkDsv4LaunchScatterAdd(cudaStream_t stream, void *destination_bf16, const void *source_bf16, const uint32_t *row_map, uint32_t slot_count, uint32_t width);
extern cudaError_t SparkDsv4LaunchAccumAdd(cudaStream_t stream, void *destination_bf16, const void *source_bf16, uint32_t row_count, uint32_t width);
extern cudaError_t SparkDsv4LaunchIndexerScore(cudaStream_t stream, const void *q_bf16, const void *kv_cache_bf16, uint64_t lane_stride_elements, const uint32_t *row_lane_indices, const uint32_t *slot_counts, const float *head_weights_f32, float *scores_f32, uint32_t row_count, uint32_t max_slots, uint32_t head_count, uint32_t head_dim);
extern cudaError_t SparkDsv4LaunchTopK(cudaStream_t stream, float *scores_f32, const uint32_t *slot_counts, uint32_t max_slots, uint32_t topk, int32_t offset, int32_t *indices_out, uint64_t out_row_stride, uint32_t row_count);
extern cudaError_t SparkDsv4LaunchHcMix(cudaStream_t stream, const void *streams_bf16, const float *fn_f32, float *mixes_f32, uint32_t row_count, uint32_t flat_dimension, uint32_t mix_rows, float epsilon);
extern cudaError_t SparkDsv4LaunchHcSplitSinkhorn(cudaStream_t stream, const float *mixes_f32, const float *scale3_f32, const float *base_f32, uint32_t row_count, uint32_t hc, uint32_t iterations, float epsilon, float *pre_f32, float *post_f32, float *comb_f32);
extern cudaError_t SparkDsv4LaunchHcPreReduce(cudaStream_t stream, const void *streams_bf16, const float *pre_f32, void *reduced_bf16, uint32_t row_count, uint32_t hc, uint32_t dimension);
extern cudaError_t SparkDsv4LaunchHcPost(cudaStream_t stream, const void *out_bf16, const void *residual_bf16, const float *post_f32, const float *comb_f32, void *streams_bf16, uint32_t row_count, uint32_t hc, uint32_t dimension);
extern cudaError_t SparkDsv4LaunchHcHeadReduce(cudaStream_t stream, const void *streams_bf16, const float *mixes_f32, float scale, const float *base_f32, float epsilon, void *reduced_bf16, uint32_t row_count, uint32_t hc, uint32_t dimension);

static SparkStatus SparkDsv4ModuleConfigure(SparkDsv4ModuleState *state)
{
	SparkStatus status;
	status = SparkStageModuleEnvironmentUnsigned(SPARK_DSV4_MODULE_TAG,"SPARK_DSV4_STAGE_COUNT",1u,SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_STAGE_COUNT,&state->stage_count);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_DSV4_MODULE_TAG,"SPARK_DSV4_STAGE_INDEX",0u,SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_STAGE_COUNT - 1u,&state->stage_index);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_DSV4_MODULE_TAG,"SPARK_DSV4_STAGE_FIRST_LAYER",0u,SPARK_DSV4_MODEL_LAYER_COUNT - 1u,&state->first_layer_index);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_DSV4_MODULE_TAG,"SPARK_DSV4_STAGE_LAYER_COUNT",1u,SPARK_DSV4_MODEL_LAYER_COUNT,&state->layer_count);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_DSV4_MODULE_TAG,"SPARK_DSV4_STAGE_MAX_ACTIVE_SEQUENCES",1u,SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,&state->max_active_sequence_count);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_DSV4_MODULE_TAG,"SPARK_DSV4_STAGE_PIPELINE_SLOTS",1u,SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT,&state->pipeline_slot_count);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_DSV4_MODULE_TAG,"SPARK_DSV4_STAGE_MAX_SEQ",SPARK_DSV4_MODEL_HCA_COMPRESS_RATIO,SPARK_DSV4_MODEL_MAX_POSITIONS,&state->max_sequence_positions);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_DSV4_MODULE_TAG,"SPARK_DSV4_STAGE_MTP",0u,0u,&state->mtp_armed);
	return(status);
}

// The slice sanity beyond ranges: position agreement, and the hash-router
// pin - token ids exist only beside the embedding, so a slice that starts
// inside the hash range without layer zero cannot route and is refused.
static SparkStatus SparkDsv4ModuleValidateSlice(SparkDsv4ModuleState *state)
{
	if ( state->stage_index >= state->stage_count || state->first_layer_index + state->layer_count > SPARK_DSV4_MODEL_LAYER_COUNT )
	{
		fprintf(stderr,"%s config_slice_invalid stage=%u/%u slice=%u+%u\n",SPARK_DSV4_MODULE_TAG,state->stage_index,state->stage_count,state->first_layer_index,state->layer_count);
		return(SPARK_STATUS_INVALID_ARGUMENT);
	}
	state->owns_embedding = state->first_layer_index == 0u ? 1u : 0u;
	state->owns_final_head = state->first_layer_index + state->layer_count == SPARK_DSV4_MODEL_LAYER_COUNT ? 1u : 0u;
	if ( (state->stage_index == 0u) != (state->owns_embedding != 0u) || (state->stage_index + 1u == state->stage_count) != (state->owns_final_head != 0u) )
	{
		fprintf(stderr,"%s config_position_mismatch stage=%u/%u slice=%u+%u\n",SPARK_DSV4_MODULE_TAG,state->stage_index,state->stage_count,state->first_layer_index,state->layer_count);
		return(SPARK_STATUS_INVALID_ARGUMENT);
	}
	if ( state->owns_embedding == 0u && state->first_layer_index < SPARK_DSV4_MODEL_HASH_ROUTED_LAYER_COUNT )
	{
		fprintf(stderr,"%s config_hash_layer_without_tokens slice=%u+%u\n",SPARK_DSV4_MODULE_TAG,state->first_layer_index,state->layer_count);
		return(SPARK_STATUS_INVALID_ARGUMENT);
	}
	return(SPARK_STATUS_OK);
}

// Per-stage ordinals: dense compress and CSA numbering inside the slice,
// and the topk column budget - the window plus the larger of the indexer
// top-k and the full compressed slot count an HCA layer attends.
static void SparkDsv4ModuleBuildOrdinals(SparkDsv4ModuleState *state)
{
	uint32_t layer,kind,hca_columns = state->max_sequence_positions / SPARK_DSV4_MODEL_HCA_COMPRESS_RATIO;
	for (layer = 0; layer < SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_LAYER_COUNT; layer++)
	{
		state->layer_local_by_layer[layer] = UINT32_MAX;
		state->compress_ordinal_by_layer[layer] = UINT32_MAX;
		state->csa_ordinal_by_layer[layer] = UINT32_MAX;
	}
	for (layer = state->first_layer_index; layer < state->first_layer_index + state->layer_count; layer++)
	{
		kind = SparkDsv4ModelLayerKind(layer);
		state->layer_local_by_layer[layer] = layer - state->first_layer_index;
		if ( kind != SPARK_DSV4_MODEL_LAYER_KIND_SWA )
			state->compress_ordinal_by_layer[layer] = state->compress_layer_count++;
		if ( kind == SPARK_DSV4_MODEL_LAYER_KIND_CSA )
			state->csa_ordinal_by_layer[layer] = state->csa_layer_count++;
	}
	state->index_slot_capacity = state->max_sequence_positions / SPARK_DSV4_MODEL_CSA_COMPRESS_RATIO;
	state->topk_column_count = SPARK_DSV4_MODEL_SLIDING_WINDOW_TOKENS + (SPARK_DSV4_MODEL_INDEX_TOP_K > hca_columns ? SPARK_DSV4_MODEL_INDEX_TOP_K : hca_columns);
}

static void SparkDsv4ModuleFillLinearView(SparkDsv4LinearView *view, const SparkDsv4StagePackEntry *entry, void *payload, void *scale)
{
	view->abi_version = SPARK_DSV4_RESIDENT_DECODE_STAGE_LINEAR_VIEW_ABI_VERSION;
	view->weight_format = entry->weight_format;
	view->rows = entry->rows;
	view->columns = entry->columns;
	view->payload = payload;
	view->scale_e8m0 = (const uint8_t *)scale;
}

static SparkStatus SparkDsv4ModuleValidateEntry(SparkDsv4ModuleState *state, const SparkDsv4StagePackEntry *entry, uint64_t file_bytes, uint32_t *is_global)
{
	SparkDsv4StagePackTensorShape shape;
	uint64_t payload_bytes,scale_bytes;
	uint32_t global = entry->layer_index == SPARK_DSV4_STAGEPACK_GLOBAL_LAYER ? 1u : 0u;
	uint32_t in_slice = entry->layer_index == SPARK_DSV4_STAGEPACK_MTP_LAYER || (entry->layer_index >= state->first_layer_index && entry->layer_index < state->first_layer_index + state->layer_count) ? 1u : 0u;
	if ( entry->tensor_kind >= SPARK_DSV4_STAGEPACK_TENSOR_KIND_COUNT || (global == 0u && in_slice == 0u) )
		return(SPARK_STATUS_VALIDATION_FAILED);
	if ( SparkDsv4StagePackResolvedShape(entry->tensor_kind,global != 0u ? 0u : entry->layer_index,global,&shape) < 0 )
		return(SPARK_STATUS_VALIDATION_FAILED);
	if ( shape.rows != entry->rows || shape.columns != entry->columns || shape.weight_format != entry->weight_format )
		return(SPARK_STATUS_VALIDATION_FAILED);
	payload_bytes = SparkDsv4StagePackPayloadBytes(entry->weight_format,entry->rows,entry->columns);
	scale_bytes = SparkDsv4StagePackScaleBytes(entry->weight_format,entry->rows,entry->columns);
	if ( entry->payload_offset + payload_bytes > file_bytes || (scale_bytes != 0u && (entry->scale_offset != entry->payload_offset + payload_bytes || entry->scale_offset + scale_bytes > file_bytes)) )
		return(SPARK_STATUS_VALIDATION_FAILED);
	*is_global = global;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkDsv4ModuleBindGlobal(SparkDsv4ModuleState *state, const SparkDsv4StagePackEntry *entry, void *payload, void *scale)
{
	switch ( entry->tensor_kind )
	{
	case SPARK_DSV4_STAGEPACK_TENSOR_EMBEDDING: state->token_embedding_bf16 = payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_FINAL_NORM: state->final_norm_weight_bf16 = payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_LM_HEAD: state->lm_head_weight_bf16 = payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_HC_HEAD_FN: state->hc_head_fn_f32 = (const float *)payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_HC_HEAD_BASE: state->hc_head_base_f32 = (const float *)payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_HC_HEAD_SCALE: state->hc_head_scale_f32 = (const float *)payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_MTP_E_PROJ: SparkDsv4ModuleFillLinearView(&state->mtp.e_proj,entry,payload,scale); break;
	case SPARK_DSV4_STAGEPACK_TENSOR_MTP_H_PROJ: SparkDsv4ModuleFillLinearView(&state->mtp.h_proj,entry,payload,scale); break;
	case SPARK_DSV4_STAGEPACK_TENSOR_MTP_ENORM: state->mtp.enorm_weight_bf16 = payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_MTP_HNORM: state->mtp.hnorm_weight_bf16 = payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_MTP_FINAL_NORM: state->mtp.final_norm_weight_bf16 = payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_MTP_HC_HEAD_FN: state->mtp.hc_head_fn_f32 = (const float *)payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_MTP_HC_HEAD_BASE: state->mtp.hc_head_base_f32 = (const float *)payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_MTP_HC_HEAD_SCALE: state->mtp.hc_head_scale_f32 = (const float *)payload; break;
	default:
		return(SPARK_STATUS_VALIDATION_FAILED);
	}
	state->global_seen_bits |= 1ull << entry->tensor_kind;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkDsv4ModuleBindLayerAttn(SparkDsv4LayerWeights *layer, const SparkDsv4StagePackEntry *entry, void *payload, void *scale)
{
	switch ( entry->tensor_kind )
	{
	case SPARK_DSV4_STAGEPACK_TENSOR_ATTN_SINK: layer->attn.sink_f32 = (const float *)payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_WQ_A: SparkDsv4ModuleFillLinearView(&layer->attn.wq_a,entry,payload,scale); break;
	case SPARK_DSV4_STAGEPACK_TENSOR_Q_NORM: layer->attn.q_norm_weight_bf16 = payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_WQ_B: SparkDsv4ModuleFillLinearView(&layer->attn.wq_b,entry,payload,scale); break;
	case SPARK_DSV4_STAGEPACK_TENSOR_WKV: SparkDsv4ModuleFillLinearView(&layer->attn.wkv,entry,payload,scale); break;
	case SPARK_DSV4_STAGEPACK_TENSOR_KV_NORM: layer->attn.kv_norm_weight_bf16 = payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_WO_A: SparkDsv4ModuleFillLinearView(&layer->attn.wo_a,entry,payload,scale); break;
	case SPARK_DSV4_STAGEPACK_TENSOR_WO_B: SparkDsv4ModuleFillLinearView(&layer->attn.wo_b,entry,payload,scale); break;
	case SPARK_DSV4_STAGEPACK_TENSOR_ATTN_NORM: layer->attn_norm_bf16 = payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_FFN_NORM: layer->ffn_norm_bf16 = payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_COMPRESS_APE: layer->compressor.ape_f32 = (const float *)payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_COMPRESS_WKV: SparkDsv4ModuleFillLinearView(&layer->compressor.wkv,entry,payload,scale); break;
	case SPARK_DSV4_STAGEPACK_TENSOR_COMPRESS_WGATE: SparkDsv4ModuleFillLinearView(&layer->compressor.wgate,entry,payload,scale); break;
	case SPARK_DSV4_STAGEPACK_TENSOR_COMPRESS_NORM: layer->compressor.norm_weight_bf16 = payload; break;
	default:
		return(SPARK_STATUS_VALIDATION_FAILED);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkDsv4ModuleBindLayerIndexer(SparkDsv4LayerWeights *layer, const SparkDsv4StagePackEntry *entry, void *payload, void *scale)
{
	switch ( entry->tensor_kind )
	{
	case SPARK_DSV4_STAGEPACK_TENSOR_INDEX_WQ_B: SparkDsv4ModuleFillLinearView(&layer->indexer.wq_b,entry,payload,scale); break;
	case SPARK_DSV4_STAGEPACK_TENSOR_INDEX_WEIGHTS: SparkDsv4ModuleFillLinearView(&layer->indexer.weights_proj,entry,payload,scale); break;
	case SPARK_DSV4_STAGEPACK_TENSOR_INDEX_APE: layer->indexer.compressor.ape_f32 = (const float *)payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_INDEX_WKV: SparkDsv4ModuleFillLinearView(&layer->indexer.compressor.wkv,entry,payload,scale); break;
	case SPARK_DSV4_STAGEPACK_TENSOR_INDEX_WGATE: SparkDsv4ModuleFillLinearView(&layer->indexer.compressor.wgate,entry,payload,scale); break;
	case SPARK_DSV4_STAGEPACK_TENSOR_INDEX_NORM: layer->indexer.compressor.norm_weight_bf16 = payload; break;
	default:
		return(SPARK_STATUS_VALIDATION_FAILED);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkDsv4ModuleBindLayerRest(SparkDsv4LayerWeights *layer, const SparkDsv4StagePackEntry *entry, void *payload, void *scale)
{
	switch ( entry->tensor_kind )
	{
	case SPARK_DSV4_STAGEPACK_TENSOR_HC_ATTN_FN: layer->hc.attn_fn_f32 = (const float *)payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_HC_FFN_FN: layer->hc.ffn_fn_f32 = (const float *)payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_HC_ATTN_BASE: layer->hc.attn_base_f32 = (const float *)payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_HC_FFN_BASE: layer->hc.ffn_base_f32 = (const float *)payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_HC_ATTN_SCALE: layer->hc.attn_scale_f32 = (const float *)payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_HC_FFN_SCALE: layer->hc.ffn_scale_f32 = (const float *)payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_GATE_WEIGHT: SparkDsv4ModuleFillLinearView(&layer->moe.gate,entry,payload,scale); break;
	case SPARK_DSV4_STAGEPACK_TENSOR_GATE_BIAS: layer->moe.gate_bias_f32 = (const float *)payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_GATE_TID2EID: layer->moe.gate_tid2eid_u32 = (const uint32_t *)payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_EXPERTS_W1: SparkDsv4ModuleFillLinearView(&layer->moe.experts_w1,entry,payload,scale); break;
	case SPARK_DSV4_STAGEPACK_TENSOR_EXPERTS_W2: SparkDsv4ModuleFillLinearView(&layer->moe.experts_w2,entry,payload,scale); break;
	case SPARK_DSV4_STAGEPACK_TENSOR_EXPERTS_W3: SparkDsv4ModuleFillLinearView(&layer->moe.experts_w3,entry,payload,scale); break;
	case SPARK_DSV4_STAGEPACK_TENSOR_SHARED_W1: SparkDsv4ModuleFillLinearView(&layer->moe.shared_w1,entry,payload,scale); break;
	case SPARK_DSV4_STAGEPACK_TENSOR_SHARED_W2: SparkDsv4ModuleFillLinearView(&layer->moe.shared_w2,entry,payload,scale); break;
	case SPARK_DSV4_STAGEPACK_TENSOR_SHARED_W3: SparkDsv4ModuleFillLinearView(&layer->moe.shared_w3,entry,payload,scale); break;
	default:
		return(SPARK_STATUS_VALIDATION_FAILED);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkDsv4ModuleBindLayer(SparkDsv4ModuleState *state, const SparkDsv4StagePackEntry *entry, void *payload, void *scale)
{
	SparkDsv4LayerWeights *layer = entry->layer_index == SPARK_DSV4_STAGEPACK_MTP_LAYER ? &state->mtp_layer : &state->layers[entry->layer_index];
	uint64_t *seen = entry->layer_index == SPARK_DSV4_STAGEPACK_MTP_LAYER ? &state->mtp_seen_bits : &state->layer_seen_bits[entry->layer_index];
	SparkStatus status;
	if ( entry->tensor_kind <= SPARK_DSV4_STAGEPACK_TENSOR_COMPRESS_NORM && entry->tensor_kind >= SPARK_DSV4_STAGEPACK_TENSOR_COMPRESS_APE )
		status = SparkDsv4ModuleBindLayerAttn(layer,entry,payload,scale);
	else if ( entry->tensor_kind <= SPARK_DSV4_STAGEPACK_TENSOR_FFN_NORM )
		status = SparkDsv4ModuleBindLayerAttn(layer,entry,payload,scale);
	else if ( entry->tensor_kind >= SPARK_DSV4_STAGEPACK_TENSOR_INDEX_WQ_B && entry->tensor_kind <= SPARK_DSV4_STAGEPACK_TENSOR_INDEX_NORM )
		status = SparkDsv4ModuleBindLayerIndexer(layer,entry,payload,scale);
	else
		status = SparkDsv4ModuleBindLayerRest(layer,entry,payload,scale);
	if ( status != SPARK_STATUS_OK )
		return(status);
	*seen |= 1ull << entry->tensor_kind;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkDsv4ModuleLoadEntry(SparkDsv4ModuleState *state, FILE *file, const SparkDsv4StagePackEntry *entry, uint64_t file_bytes)
{
	uint64_t payload_bytes = SparkDsv4StagePackPayloadBytes(entry->weight_format,entry->rows,entry->columns);
	uint64_t scale_bytes = SparkDsv4StagePackScaleBytes(entry->weight_format,entry->rows,entry->columns);
	void *payload = 0,*scale = 0;
	uint32_t is_global = 0u;
	SparkStatus status = SparkDsv4ModuleValidateEntry(state,entry,file_bytes,&is_global);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"%s pack_entry_invalid kind=%u layer=%u\n",SPARK_DSV4_MODULE_TAG,entry->tensor_kind,entry->layer_index);
		return(status);
	}
	status = SparkStageModuleLoadDeviceRegion(&state->ledger,file,entry->payload_offset,payload_bytes,&payload);
	if ( status == SPARK_STATUS_OK && scale_bytes != 0u )
		status = SparkStageModuleLoadDeviceRegion(&state->ledger,file,entry->scale_offset,scale_bytes,&scale);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( is_global != 0u )
		return(SparkDsv4ModuleBindGlobal(state,entry,payload,scale));
	return(SparkDsv4ModuleBindLayer(state,entry,payload,scale));
}

// Coverage: every layer in the slice must have seen the exact kind set its
// attention class demands, the MTP layer its SWA score-routed set, and the
// globals the position-derived set - a missing tensor is a refused pack.
static uint64_t SparkDsv4ModuleExpectedLayerBits(uint32_t layer_index)
{
	uint32_t kind = SparkDsv4StagePackLayerKind(layer_index),tensor;
	uint64_t bits = 0u;
	for (tensor = SPARK_DSV4_STAGEPACK_TENSOR_ATTN_SINK; tensor <= SPARK_DSV4_STAGEPACK_TENSOR_SHARED_W3; tensor++)
		bits |= 1ull << tensor;
	bits &= ~(1ull << (SparkDsv4StagePackLayerIsHashRouted(layer_index) != 0u ? SPARK_DSV4_STAGEPACK_TENSOR_GATE_BIAS : SPARK_DSV4_STAGEPACK_TENSOR_GATE_TID2EID));
	if ( kind != SPARK_DSV4_MODEL_LAYER_KIND_SWA )
		bits |= (1ull << SPARK_DSV4_STAGEPACK_TENSOR_COMPRESS_APE) | (1ull << SPARK_DSV4_STAGEPACK_TENSOR_COMPRESS_WKV) | (1ull << SPARK_DSV4_STAGEPACK_TENSOR_COMPRESS_WGATE) | (1ull << SPARK_DSV4_STAGEPACK_TENSOR_COMPRESS_NORM);
	if ( kind == SPARK_DSV4_MODEL_LAYER_KIND_CSA )
	{
		bits |= (1ull << SPARK_DSV4_STAGEPACK_TENSOR_INDEX_WQ_B) | (1ull << SPARK_DSV4_STAGEPACK_TENSOR_INDEX_WEIGHTS) | (1ull << SPARK_DSV4_STAGEPACK_TENSOR_INDEX_APE);
		bits |= (1ull << SPARK_DSV4_STAGEPACK_TENSOR_INDEX_WKV) | (1ull << SPARK_DSV4_STAGEPACK_TENSOR_INDEX_WGATE) | (1ull << SPARK_DSV4_STAGEPACK_TENSOR_INDEX_NORM);
	}
	return(bits);
}

static SparkStatus SparkDsv4ModuleVerifyCoverage(SparkDsv4ModuleState *state)
{
	uint64_t expected_globals = 0u;
	uint32_t layer,tensor;
	for (layer = state->first_layer_index; layer < state->first_layer_index + state->layer_count; layer++)
		if ( state->layer_seen_bits[layer] != SparkDsv4ModuleExpectedLayerBits(layer) )
		{
			fprintf(stderr,"%s pack_layer_coverage layer=%u seen=%llx\n",SPARK_DSV4_MODULE_TAG,layer,(unsigned long long)state->layer_seen_bits[layer]);
			return(SPARK_STATUS_VALIDATION_FAILED);
		}
	if ( state->owns_embedding != 0u || state->owns_final_head != 0u )
		expected_globals |= 1ull << SPARK_DSV4_STAGEPACK_TENSOR_EMBEDDING;
	if ( state->owns_final_head != 0u )
	{
		for (tensor = SPARK_DSV4_STAGEPACK_TENSOR_FINAL_NORM; tensor <= SPARK_DSV4_STAGEPACK_TENSOR_MTP_HC_HEAD_SCALE; tensor++)
			expected_globals |= 1ull << tensor;
		if ( state->mtp_seen_bits != SparkDsv4ModuleExpectedLayerBits(SPARK_DSV4_STAGEPACK_MTP_LAYER) )
		{
			fprintf(stderr,"%s pack_mtp_coverage seen=%llx\n",SPARK_DSV4_MODULE_TAG,(unsigned long long)state->mtp_seen_bits);
			return(SPARK_STATUS_VALIDATION_FAILED);
		}
	}
	if ( state->global_seen_bits != expected_globals )
	{
		fprintf(stderr,"%s pack_global_coverage seen=%llx expected=%llx\n",SPARK_DSV4_MODULE_TAG,(unsigned long long)state->global_seen_bits,(unsigned long long)expected_globals);
		return(SPARK_STATUS_VALIDATION_FAILED);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkDsv4ModuleLoadPack(SparkDsv4ModuleState *state, const char *path)
{
	SparkDsv4StagePackHeader header,expected;
	SparkDsv4StagePackEntry *directory;
	FILE *file;
	SparkStatus status;
	int32_t compare;
	uint32_t index;
	file = fopen(path,"rb");
	if ( file == 0 )
	{
		fprintf(stderr,"%s pack_open_failed path=%s\n",SPARK_DSV4_MODULE_TAG,path);
		return(SPARK_STATUS_IO_ERROR);
	}
	status = SparkStageModulePackRead(SPARK_DSV4_MODULE_TAG,file,0u,&header,sizeof(header));
	if ( status == SPARK_STATUS_OK )
	{
		SparkDsv4StagePackExpectedGeometry(&expected,state->first_layer_index,state->layer_count);
		compare = SparkDsv4StagePackCompareGeometry(&header,&expected);
		if ( compare != 0 )
		{
			fprintf(stderr,"%s pack_geometry_mismatch field=%s\n",SPARK_DSV4_MODULE_TAG,SparkDsv4StagePackGeometryFieldName(compare));
			status = SPARK_STATUS_VALIDATION_FAILED;
		}
	}
	directory = status == SPARK_STATUS_OK ? (SparkDsv4StagePackEntry *)malloc((size_t)header.tensor_count * sizeof(SparkDsv4StagePackEntry)) : 0;
	if ( status == SPARK_STATUS_OK && directory == 0 )
		status = SPARK_STATUS_CAPACITY_EXCEEDED;
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModulePackRead(SPARK_DSV4_MODULE_TAG,file,header.directory_offset,directory,(uint64_t)header.tensor_count * sizeof(SparkDsv4StagePackEntry));
	for (index = 0; status == SPARK_STATUS_OK && index < header.tensor_count; index++)
		status = SparkDsv4ModuleLoadEntry(state,file,&directory[index],header.file_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkDsv4ModuleVerifyCoverage(state);
	free(directory);
	fclose(file);
	return(status);
}

// Host YaRN frequency table, the reference precompute arithmetic; the
// interpolation ramp engages only when original positions are declared.
static void SparkDsv4ModuleComputeFreqs(float *freqs, float base, uint32_t original, float factor)
{
	uint32_t rope_dim = SPARK_DSV4_MODEL_ATTN_ROPE_DIMENSION,pair;
	float low,high,ramp,smooth,frequency;
	low = floorf((float)rope_dim * logf((float)original / ((float)SPARK_DSV4_MODEL_ROPE_BETA_FAST * 2.0f * 3.14159265f)) / (2.0f * logf(base)));
	high = ceilf((float)rope_dim * logf((float)original / ((float)SPARK_DSV4_MODEL_ROPE_BETA_SLOW * 2.0f * 3.14159265f)) / (2.0f * logf(base)));
	if ( low < 0.0f )
		low = 0.0f;
	if ( high > (float)(rope_dim - 1u) )
		high = (float)(rope_dim - 1u);
	if ( low == high )
		high += 0.001f;
	for (pair = 0; pair < rope_dim / 2u; pair++)
	{
		frequency = 1.0f / powf(base,(float)(2u * pair) / (float)rope_dim);
		if ( original != 0u )
		{
			ramp = ((float)pair - low) / (high - low);
			if ( ramp < 0.0f )
				ramp = 0.0f;
			if ( ramp > 1.0f )
				ramp = 1.0f;
			smooth = 1.0f - ramp;
			frequency = frequency / factor * (1.0f - smooth) + frequency * smooth;
		}
		freqs[pair] = frequency;
	}
}

static SparkStatus SparkDsv4ModuleUploadFreqs(SparkDsv4ModuleState *state)
{
	float host_freqs[SPARK_DSV4_MODEL_ATTN_ROPE_DIMENSION / 2u];
	SparkStatus status;
	cudaError_t error;
	status = SparkStageModuleDeviceAllocate(&state->ledger,sizeof(host_freqs),(void **)&state->base_freqs_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,sizeof(host_freqs),(void **)&state->compress_freqs_f32);
	if ( status != SPARK_STATUS_OK )
		return(status);
	SparkDsv4ModuleComputeFreqs(host_freqs,SPARK_DSV4_MODEL_ATTN_ROPE_THETA,0u,(float)SPARK_DSV4_MODEL_ATTN_YARN_FACTOR);
	error = cudaMemcpy(state->base_freqs_f32,host_freqs,sizeof(host_freqs),cudaMemcpyHostToDevice);
	if ( error == cudaSuccess )
	{
		SparkDsv4ModuleComputeFreqs(host_freqs,SPARK_DSV4_MODEL_COMPRESS_ROPE_THETA,SPARK_DSV4_MODEL_ATTN_YARN_ORIGINAL_POSITIONS,(float)SPARK_DSV4_MODEL_ATTN_YARN_FACTOR);
		error = cudaMemcpy(state->compress_freqs_f32,host_freqs,sizeof(host_freqs),cudaMemcpyHostToDevice);
	}
	return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,"freq_upload"));
}

/*
 * Cache pools. Every layer holds a lane block of window + compressed slots
 * of head_dim bf16 in one contiguous run - the reference's [win | stream]
 * layout, so the attention indices address one base. Compressor states are
 * sized for the WORST class on the stage (HCA's 128x512 doubled) and
 * strided uniformly per (layer ordinal, lane); the indexer keeps its own
 * rotated cache and small overlap state per CSA ordinal.
 */
// One-time MXFP4 shadow of the lm_head plus per-neuron certified error
// norms, the mimo25 screened-head pattern; head stage only, built
// synchronously at initialize.
static SparkStatus SparkDsv4ModuleBuildHeadShadow(SparkDsv4ModuleState *state)
{
	uint64_t vocab = SPARK_DSV4_MODEL_VOCAB_COUNT,dim = SPARK_DSV4_MODEL_HIDDEN_DIMENSION;
	SparkStatus status;
	if ( state->owns_final_head == 0u )
		return(SPARK_STATUS_OK);
	status = SparkStageModuleDeviceAllocate(&state->ledger,(vocab * dim) / 2u,(void **)&state->head_shadow_payload);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(vocab * dim) / 32u,(void **)&state->head_shadow_scale);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,vocab * sizeof(float),(void **)&state->head_error_norm_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,SparkDsv4LaunchHeadShadowQuantize(0,state->lm_head_weight_bf16,state->head_shadow_payload,state->head_shadow_scale,state->head_error_norm_f32,(uint32_t)vocab,(uint32_t)dim),"head_shadow_quantize");
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,cudaDeviceSynchronize(),"head_shadow_sync");
	return(status);
}

static SparkStatus SparkDsv4ModuleAllocatePools(SparkDsv4ModuleState *state)
{
	uint64_t compressed_slots = state->max_sequence_positions / SPARK_DSV4_MODEL_CSA_COMPRESS_RATIO;
	uint64_t lane_block = ((uint64_t)SPARK_DSV4_MODEL_SLIDING_WINDOW_TOKENS + compressed_slots) * SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION;
	uint64_t state_elements = (uint64_t)SPARK_DSV4_MODEL_CSA_OVERLAP_FACTOR * SPARK_DSV4_MODEL_HCA_COMPRESS_RATIO * SPARK_DSV4_MODEL_CSA_OVERLAP_FACTOR * SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION;
	uint64_t index_state_elements = (uint64_t)SPARK_DSV4_MODEL_CSA_OVERLAP_FACTOR * SPARK_DSV4_MODEL_CSA_COMPRESS_RATIO * SPARK_DSV4_MODEL_CSA_OVERLAP_FACTOR * SPARK_DSV4_MODEL_INDEX_HEAD_DIMENSION;
	uint64_t lanes = state->max_active_sequence_count;
	SparkStatus status;
	state->cache_lane_block_elements = lane_block;
	state->cache_layer_lane_stride = lane_block;
	state->index_lane_stride = (uint64_t)state->index_slot_capacity * SPARK_DSV4_MODEL_INDEX_HEAD_DIMENSION;
	state->compress_state_lane_stride = state_elements;
	state->index_state_lane_stride = index_state_elements;
	status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,(uint64_t)state->layer_count * lanes * lane_block * SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES,&state->kv_cache_bf16);
	if ( status == SPARK_STATUS_OK && state->csa_layer_count != 0u )
		status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,(uint64_t)state->csa_layer_count * lanes * state->index_lane_stride * SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES,&state->index_cache_bf16);
	if ( status == SPARK_STATUS_OK && state->compress_layer_count != 0u )
		status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,(uint64_t)state->compress_layer_count * lanes * state_elements * sizeof(float),(void **)&state->compress_kv_state_f32);
	if ( status == SPARK_STATUS_OK && state->compress_layer_count != 0u )
		status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,(uint64_t)state->compress_layer_count * lanes * state_elements * sizeof(float),(void **)&state->compress_score_state_f32);
	if ( status == SPARK_STATUS_OK && state->csa_layer_count != 0u )
		status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,(uint64_t)state->csa_layer_count * lanes * index_state_elements * sizeof(float),(void **)&state->index_kv_state_f32);
	if ( status == SPARK_STATUS_OK && state->csa_layer_count != 0u )
		status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,(uint64_t)state->csa_layer_count * lanes * index_state_elements * sizeof(float),(void **)&state->index_score_state_f32);
	return(status);
}

// GB10 unified memory zero-copy control plane, mirrored from mimo25:
// one mapped pinned allocation per array, host pointer for the CPU
// grouping, device alias for the kernels; every former copy was a read
// plus a write through the same bus.
static SparkStatus SparkDsv4ModuleMappedAllocate(uint64_t bytes, void **host_out, void **device_out, const char *label)
{
	SparkStatus status;
	status = SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,cudaHostAlloc(host_out,bytes,cudaHostAllocMapped),label);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,cudaHostGetDevicePointer(device_out,*host_out,0),label);
	return(status);
}

static SparkStatus SparkDsv4ModuleAllocateSlotSmall(SparkDsv4ModuleState *state, SparkDsv4ModuleSlot *slot)
{
	uint32_t rows = state->max_active_sequence_count;
	SparkStatus status;
	status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)rows * sizeof(uint32_t),(void **)&slot->input_token_ids);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)rows * sizeof(uint32_t),(void **)&slot->output_token_ids);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)rows * sizeof(uint32_t),(void **)&slot->row_lane_indices);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)rows * sizeof(uint64_t),(void **)&slot->row_positions);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)rows * sizeof(uint64_t),(void **)&slot->row_emit_positions);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)rows * sizeof(uint64_t),(void **)&slot->row_emit_positions_hca);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)rows * sizeof(uint32_t),(void **)&slot->slot_counts);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)rows * state->topk_column_count * sizeof(int32_t),(void **)&slot->topk_idxs);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)rows * sizeof(uint32_t),(void **)&slot->emitted_u32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)rows * SPARK_DSV4_MODEL_HC_MIX_ROWS * sizeof(float),(void **)&slot->mixes_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)rows * SPARK_DSV4_MODEL_HC_STREAM_COUNT * sizeof(float),(void **)&slot->pre_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)rows * SPARK_DSV4_MODEL_HC_STREAM_COUNT * sizeof(float),(void **)&slot->post_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)rows * SPARK_DSV4_MODEL_HC_STREAM_COUNT * SPARK_DSV4_MODEL_HC_STREAM_COUNT * sizeof(float),(void **)&slot->comb_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)rows * SPARK_DSV4_MODEL_ROUTED_EXPERT_COUNT * sizeof(float),(void **)&slot->moe_scores_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)rows * SPARK_DSV4_MODEL_EXPERTS_PER_TOKEN * sizeof(float),(void **)&slot->moe_weights_f32);
	return(status);
}

static SparkStatus SparkDsv4ModuleAllocateSlotWide(SparkDsv4ModuleState *state, SparkDsv4ModuleSlot *slot)
{
	uint64_t rows = state->max_active_sequence_count,dim = SPARK_DSV4_MODEL_HIDDEN_DIMENSION,bf16 = SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES;
	uint64_t stream_bytes = rows * SPARK_DSV4_MODEL_HC_STREAM_COUNT * dim * bf16;
	uint64_t compress_channels = (uint64_t)SPARK_DSV4_MODEL_CSA_OVERLAP_FACTOR * SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION;
	SparkStatus status;
	status = SparkStageModuleDeviceAllocate(&state->ledger,stream_bytes,&slot->streams_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,stream_bytes,&slot->residual_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * dim * bf16,&slot->reduced_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * dim * bf16,&slot->normalized_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_DSV4_MODEL_QUERY_LORA_RANK * bf16,&slot->qr_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_DSV4_MODEL_ATTN_QUERY_DIMENSION * bf16,&slot->q_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION * bf16,&slot->kv_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_DSV4_MODEL_ATTN_QUERY_DIMENSION * bf16,&slot->attn_out_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_DSV4_MODEL_OUTPUT_GROUP_COUNT * SPARK_DSV4_MODEL_OUTPUT_LORA_RANK * bf16,&slot->o_ranks_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * dim * bf16,&slot->delta_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * compress_channels * bf16,&slot->compress_kv_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * compress_channels * bf16,&slot->compress_score_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * compress_channels * sizeof(float),(void **)&slot->compress_kv_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * compress_channels * sizeof(float),(void **)&slot->compress_score_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION * bf16,&slot->emit_bf16);
	return(status);
}

static SparkStatus SparkDsv4ModuleAllocateSlotTail(SparkDsv4ModuleState *state, SparkDsv4ModuleSlot *slot)
{
	uint64_t rows = state->max_active_sequence_count,dim = SPARK_DSV4_MODEL_HIDDEN_DIMENSION,bf16 = SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES;
	SparkStatus status;
	status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_DSV4_MODEL_INDEX_DIMENSION * bf16,&slot->index_q_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_DSV4_MODEL_INDEX_HEAD_COUNT * bf16,&slot->index_weights_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_DSV4_MODEL_INDEX_HEAD_COUNT * sizeof(float),(void **)&slot->index_weights_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * (uint64_t)state->index_slot_capacity * sizeof(float),(void **)&slot->index_scores_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_DSV4_MODEL_EXPERT_INTERMEDIATE_DIMENSION * bf16,&slot->ffn_gate_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_DSV4_MODEL_EXPERT_INTERMEDIATE_DIMENSION * bf16,&slot->ffn_up_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * dim * bf16,&slot->ffn_delta_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * dim * bf16,&slot->ffn_accum_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkDsv4ModuleMappedAllocate(rows * SPARK_DSV4_MODEL_EXPERTS_PER_TOKEN * sizeof(uint32_t),(void **)&slot->host_moe_indices,(void **)&slot->moe_indices_u32,"map_moe_indices");
	if ( status == SPARK_STATUS_OK )
		status = SparkDsv4ModuleMappedAllocate(rows * SPARK_DSV4_MODEL_EXPERTS_PER_TOKEN * sizeof(uint32_t),(void **)&slot->host_grouped_rows,(void **)&slot->grouped_rows_u32,"map_grouped_rows");
	if ( status == SPARK_STATUS_OK )
		status = SparkDsv4ModuleMappedAllocate(rows * SPARK_DSV4_MODEL_EXPERTS_PER_TOKEN * sizeof(uint32_t),(void **)&slot->host_grouped_weight_slots,(void **)&slot->grouped_weight_slots_u32,"map_grouped_slots");
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_DSV4_MODEL_EXPERTS_PER_TOKEN * SPARK_DSV4_MODEL_EXPERT_INTERMEDIATE_DIMENSION * bf16,&slot->moe_slot_gate_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_DSV4_MODEL_EXPERTS_PER_TOKEN * SPARK_DSV4_MODEL_EXPERT_INTERMEDIATE_DIMENSION * bf16,&slot->moe_slot_up_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_DSV4_MODEL_EXPERTS_PER_TOKEN * dim * bf16,&slot->moe_slot_out_bf16);
	if ( status == SPARK_STATUS_OK && state->owns_final_head != 0u )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_DSV4_MODEL_VOCAB_COUNT * bf16,&slot->head_logits_bf16);
	if ( status == SPARK_STATUS_OK && state->owns_final_head != 0u )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_DSV4_RESIDENT_DECODE_STAGE_HEAD_SCREEN_CAP * sizeof(uint32_t),(void **)&slot->head_candidate_ids_u32);
	if ( status == SPARK_STATUS_OK && state->owns_final_head != 0u )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * sizeof(uint32_t),(void **)&slot->head_candidate_counts_u32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,cudaStreamCreateWithFlags((cudaStream_t *)&slot->cuda_stream,cudaStreamNonBlocking),"stream_create");
	return(status);
}

static SparkStatus SparkDsv4ModuleAllocateSlot(SparkDsv4ModuleState *state, SparkDsv4ModuleSlot *slot)
{
	SparkStatus status = SparkDsv4ModuleAllocateSlotSmall(state,slot);
	if ( status == SPARK_STATUS_OK )
		status = SparkDsv4ModuleAllocateSlotWide(state,slot);
	if ( status == SPARK_STATUS_OK )
		status = SparkDsv4ModuleAllocateSlotTail(state,slot);
	return(status);
}

static SparkStatus SparkDsv4ModuleValidateFrame(SparkDsv4ModuleState *state, const SparkModelDriverFrame *frame, const SparkDsv4ResidentDecodeStageFrameContext **context_out)
{
	const SparkDsv4ResidentDecodeStageFrameContext *context;
	const SparkDsv4DecodeBatchView *batch;
	uint32_t needs_input = state->stage_index > 0u ? 1u : 0u,needs_output = state->stage_index + 1u < state->stage_count ? 1u : 0u;
	uint32_t expected_buffers = state->owns_embedding + state->owns_final_head;
	if ( frame == 0 || frame->user_context == 0 || frame->buffer_count != expected_buffers )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	context = (const SparkDsv4ResidentDecodeStageFrameContext *)frame->user_context;
	if ( context->abi_version != SPARK_DSV4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION || context->descriptor_bytes != (uint32_t)sizeof(*context) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( (context->flags & SPARK_DSV4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_PREFILL_FRAME_VIEW) != 0u )
	{
		fprintf(stderr,"%s prefill_pending\n",SPARK_DSV4_MODULE_TAG);
		return(SPARK_STATUS_MODULE_NOT_VALIDATED);
	}
	if ( (context->flags & SPARK_DSV4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_DECODE_BATCH_VIEW) == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( ((context->flags & SPARK_DSV4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_INPUT_TRANSPORT) != 0u) != (needs_input != 0u) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( ((context->flags & SPARK_DSV4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_OUTPUT_TRANSPORT) != 0u) != (needs_output != 0u) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	batch = context->decode_batch;
	if ( batch == 0 || batch->abi_version != SPARK_DSV4_RESIDENT_DECODE_STAGE_DECODE_BATCH_VIEW_ABI_VERSION || batch->descriptor_bytes != (uint32_t)sizeof(*batch) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( batch->row_count == 0u || batch->row_count > state->max_active_sequence_count || batch->row_count != frame->new_token_count || batch->row_lane_indices == 0 || batch->row_positions == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	*context_out = context;
	return(SPARK_STATUS_OK);
}

// Decode staging: distinct lanes, in-range positions, and every position
// below the dense cache bound - past it the frame refuses rather than
// truncating a stream the attention would then silently miss. Two emit
// position tables ride along, one per compress ratio: an emitted entry is
// roped at its GROUP START, position+1-ratio, and 4 and 128 differ.
static SparkStatus SparkDsv4ModuleStageRows(SparkDsv4ModuleState *state, const SparkDsv4DecodeBatchView *batch, SparkDsv4ModuleSlot *slot)
{
	uint8_t lane_used[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	uint64_t emit_positions[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint64_t emit_positions_hca[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t row,lane;
	cudaError_t error;
	memset(lane_used,0,sizeof(lane_used));
	for (row = 0; row < batch->row_count; row++)
	{
		lane = batch->row_lane_indices[row];
		if ( lane >= state->max_active_sequence_count || lane_used[lane] != 0u || batch->row_positions[row] >= state->max_sequence_positions )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		lane_used[lane] = 1u;
		emit_positions[row] = batch->row_positions[row] + 1u >= SPARK_DSV4_MODEL_CSA_COMPRESS_RATIO ? batch->row_positions[row] + 1u - SPARK_DSV4_MODEL_CSA_COMPRESS_RATIO : 0u;
		emit_positions_hca[row] = batch->row_positions[row] + 1u >= SPARK_DSV4_MODEL_HCA_COMPRESS_RATIO ? batch->row_positions[row] + 1u - SPARK_DSV4_MODEL_HCA_COMPRESS_RATIO : 0u;
	}
	error = cudaMemcpyAsync(slot->row_lane_indices,batch->row_lane_indices,(uint64_t)batch->row_count * sizeof(uint32_t),cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess )
		error = cudaMemcpyAsync(slot->row_positions,batch->row_positions,(uint64_t)batch->row_count * sizeof(uint64_t),cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess )
		error = cudaMemcpyAsync(slot->row_emit_positions,emit_positions,(uint64_t)batch->row_count * sizeof(uint64_t),cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess )
		error = cudaMemcpyAsync(slot->row_emit_positions_hca,emit_positions_hca,(uint64_t)batch->row_count * sizeof(uint64_t),cudaMemcpyHostToDevice,stream);
	return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,"stage_rows"));
}

static SparkStatus SparkDsv4ModuleBeginStreams(SparkDsv4ModuleState *state, SparkDsv4ModuleSlot *slot, SparkDsv4ResidentDecodeStageFrameContext *context, const SparkModelDriverFrame *frame, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	uint64_t stream_bytes = (uint64_t)rows * SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS * SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES;
	SparkStatus status;
	cudaError_t error;
	uint32_t copy;
	if ( state->owns_embedding != 0u )
	{
		error = cudaMemcpyAsync(slot->input_token_ids,frame->buffers[0].address,(uint64_t)rows * sizeof(uint32_t),cudaMemcpyHostToDevice,stream);
		if ( error == cudaSuccess )
			error = SparkDsv4LaunchEmbeddingGather(stream,slot->input_token_ids,state->token_embedding_bf16,slot->reduced_bf16,rows,SPARK_DSV4_MODEL_HIDDEN_DIMENSION);
		for (copy = 0; error == cudaSuccess && copy < SPARK_DSV4_MODEL_HC_STREAM_COUNT; copy++)
			error = cudaMemcpy2DAsync((uint8_t *)slot->streams_bf16 + (uint64_t)copy * SPARK_DSV4_MODEL_HIDDEN_DIMENSION * SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES,(uint64_t)SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS * SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES,slot->reduced_bf16,(uint64_t)SPARK_DSV4_MODEL_HIDDEN_DIMENSION * SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES,(uint64_t)SPARK_DSV4_MODEL_HIDDEN_DIMENSION * SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES,rows,cudaMemcpyDeviceToDevice,stream);
		return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,"embedding_streams"));
	}
	if ( context->hidden_input_post_receive_function == 0 || context->hidden_input_transport_session == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	status = context->hidden_input_post_receive_function(context->hidden_input_transport_session,&context->hidden_input_packet);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( context->hidden_input_packet.active_sequence_count != rows || context->hidden_input_packet.hidden_dimension != SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS || context->hidden_input_packet.hidden_bf16 == 0 )
		return(SPARK_STATUS_VALIDATION_FAILED);
	error = cudaMemcpyAsync(slot->streams_bf16,context->hidden_input_packet.hidden_bf16,stream_bytes,cudaMemcpyDeviceToDevice,stream);
	return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,"streams_receive"));
}

// One mHC boundary: mix, split with Sinkhorn, reduce - the residual copy
// is the caller's, since attention and ffn share this exactly.
static cudaError_t SparkDsv4ModuleHcEnter(SparkDsv4ModuleSlot *slot, const float *fn, const float *scale3, const float *base, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	cudaError_t error;
	error = cudaMemcpyAsync(slot->residual_bf16,slot->streams_bf16,(uint64_t)rows * SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS * SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES,cudaMemcpyDeviceToDevice,stream);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchHcMix(stream,slot->streams_bf16,fn,slot->mixes_f32,rows,SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS,SPARK_DSV4_MODEL_HC_MIX_ROWS,SPARK_DSV4_MODEL_RMS_NORM_EPSILON);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchHcSplitSinkhorn(stream,slot->mixes_f32,scale3,base,rows,SPARK_DSV4_MODEL_HC_STREAM_COUNT,SPARK_DSV4_MODEL_HC_SINKHORN_ITERATIONS,SPARK_DSV4_MODEL_HC_EPSILON,slot->pre_f32,slot->post_f32,slot->comb_f32);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchHcPreReduce(stream,slot->streams_bf16,slot->pre_f32,slot->reduced_bf16,rows,SPARK_DSV4_MODEL_HC_STREAM_COUNT,SPARK_DSV4_MODEL_HIDDEN_DIMENSION);
	return(error);
}

/*
 * The compressor for one decode token, attention side: wkv/wgate on the
 * normalized x, widen to fp32, ape by in-group position, the state step,
 * and for boundary rows the emitted slot gets norm, rope at the group
 * start position, the fp8 cache sim, and lands at position/ratio behind
 * the window - the host knows the boundary from the position arithmetic.
 */
static cudaError_t SparkDsv4ModuleRunCompressor(SparkDsv4ModuleState *state, SparkDsv4ModuleSlot *slot, const SparkDsv4CompressorWeights *weights, const SparkDsv4DecodeBatchView *batch, float *kv_state, float *score_state, uint64_t state_stride, void *cache_base, uint64_t cache_lane_stride, uint64_t cache_slot_offset, uint32_t cache_width, uint32_t rotate, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	uint32_t channels = (weights->overlap != 0u ? 2u : 1u) * cache_width,row;
	uint64_t ratio = weights->ratio,position,slot_index;
	cudaError_t error;
	error = SparkDsv4LaunchLinear(stream,&weights->wkv,slot->normalized_bf16,slot->compress_kv_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchLinear(stream,&weights->wgate,slot->normalized_bf16,slot->compress_score_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchWiden(stream,slot->compress_kv_bf16,slot->compress_kv_f32,rows,channels,1.0f);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchWiden(stream,slot->compress_score_bf16,slot->compress_score_f32,rows,channels,1.0f);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchApeAdd(stream,slot->compress_score_f32,weights->ape_f32,slot->row_positions,rows,(uint32_t)ratio,channels);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchCompressStep(stream,slot->compress_kv_f32,slot->compress_score_f32,kv_state,score_state,state_stride,slot->row_lane_indices,slot->row_positions,rows,(uint32_t)ratio,weights->overlap,cache_width,slot->emit_bf16,slot->emitted_u32);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchRmsNorm(stream,slot->emit_bf16,weights->norm_weight_bf16,slot->emit_bf16,rows,cache_width,SPARK_DSV4_MODEL_RMS_NORM_EPSILON);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchRope(stream,slot->emit_bf16,state->compress_freqs_f32,weights->ratio == SPARK_DSV4_MODEL_HCA_COMPRESS_RATIO ? slot->row_emit_positions_hca : slot->row_emit_positions,rows,1u,cache_width,SPARK_DSV4_MODEL_ATTN_ROPE_DIMENSION,0u);
	if ( error == cudaSuccess && rotate != 0u )
	{
		error = SparkDsv4LaunchHadamard(stream,slot->emit_bf16,rows,cache_width);
		if ( error == cudaSuccess )
			error = SparkDsv4LaunchQuantSim(stream,slot->emit_bf16,rows,cache_width,cache_width,SPARK_DSV4_MODEL_FP4_QUANT_BLOCK,1u);
	}
	if ( error == cudaSuccess && rotate == 0u )
		error = SparkDsv4LaunchQuantSim(stream,slot->emit_bf16,rows,cache_width,cache_width - SPARK_DSV4_MODEL_ATTN_ROPE_DIMENSION,SPARK_DSV4_MODEL_KV_QUANT_BLOCK,0u);
	for (row = 0; error == cudaSuccess && row < rows; row++)
	{
		position = batch->row_positions[row];
		if ( (position + 1u) % ratio != 0u )
			continue;
		slot_index = cache_slot_offset + position / ratio;
		error = cudaMemcpyAsync((uint8_t *)cache_base + ((uint64_t)batch->row_lane_indices[row] * cache_lane_stride + slot_index * cache_width) * SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES,(const uint8_t *)slot->emit_bf16 + (uint64_t)row * cache_width * SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES,(uint64_t)cache_width * SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES,cudaMemcpyDeviceToDevice,stream);
	}
	return(error);
}

// Host-side attention index assembly: the window slots every kind attends
// (ring slots [0, min(pos+1, 128))), the full compressed run for HCA, and
// the CSA tail left for the device top-k. -1 pads the rest.
static void SparkDsv4ModuleHostTopkFill(SparkDsv4ModuleState *state, const SparkDsv4DecodeBatchView *batch, int32_t *host_idxs, uint32_t *host_counts, uint32_t layer_kind, uint32_t rows)
{
	uint32_t row,column,window,compressed;
	uint64_t position;
	for (row = 0; row < rows; row++)
	{
		position = batch->row_positions[row];
		window = position + 1u < SPARK_DSV4_MODEL_SLIDING_WINDOW_TOKENS ? (uint32_t)(position + 1u) : SPARK_DSV4_MODEL_SLIDING_WINDOW_TOKENS;
		compressed = layer_kind == SPARK_DSV4_MODEL_LAYER_KIND_HCA ? (uint32_t)((position + 1u) / SPARK_DSV4_MODEL_HCA_COMPRESS_RATIO) : 0u;
		host_counts[row] = layer_kind == SPARK_DSV4_MODEL_LAYER_KIND_CSA ? (uint32_t)((position + 1u) / SPARK_DSV4_MODEL_CSA_COMPRESS_RATIO) : 0u;
		for (column = 0; column < state->topk_column_count; column++)
		{
			if ( column < window )
				host_idxs[(uint64_t)row * state->topk_column_count + column] = (int32_t)column;
			else if ( layer_kind == SPARK_DSV4_MODEL_LAYER_KIND_HCA && column < window + compressed )
				host_idxs[(uint64_t)row * state->topk_column_count + column] = (int32_t)(SPARK_DSV4_MODEL_SLIDING_WINDOW_TOKENS + column - window);
			else
				host_idxs[(uint64_t)row * state->topk_column_count + column] = -1;
		}
	}
}

static cudaError_t SparkDsv4ModuleRunIndexer(SparkDsv4ModuleState *state, SparkDsv4ModuleSlot *slot, const SparkDsv4LayerWeights *layer, const SparkDsv4DecodeBatchView *batch, uint32_t csa_ordinal, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	uint64_t lanes = state->max_active_sequence_count;
	void *index_cache = (uint8_t *)state->index_cache_bf16 + (uint64_t)csa_ordinal * lanes * state->index_lane_stride * SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES;
	float *kv_state = state->index_kv_state_f32 + (uint64_t)csa_ordinal * lanes * state->index_state_lane_stride;
	float *score_state = state->index_score_state_f32 + (uint64_t)csa_ordinal * lanes * state->index_state_lane_stride;
	float weight_scale = 1.0f / sqrtf((float)SPARK_DSV4_MODEL_INDEX_HEAD_DIMENSION) / sqrtf((float)SPARK_DSV4_MODEL_INDEX_HEAD_COUNT);
	cudaError_t error;
	error = SparkDsv4LaunchLinear(stream,&layer->indexer.wq_b,slot->qr_bf16,slot->index_q_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchRope(stream,slot->index_q_bf16,state->compress_freqs_f32,slot->row_positions,rows,SPARK_DSV4_MODEL_INDEX_HEAD_COUNT,SPARK_DSV4_MODEL_INDEX_HEAD_DIMENSION,SPARK_DSV4_MODEL_ATTN_ROPE_DIMENSION,0u);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchHadamard(stream,slot->index_q_bf16,rows * SPARK_DSV4_MODEL_INDEX_HEAD_COUNT,SPARK_DSV4_MODEL_INDEX_HEAD_DIMENSION);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchQuantSim(stream,slot->index_q_bf16,rows,SPARK_DSV4_MODEL_INDEX_DIMENSION,SPARK_DSV4_MODEL_INDEX_DIMENSION,SPARK_DSV4_MODEL_FP4_QUANT_BLOCK,1u);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchLinear(stream,&layer->indexer.weights_proj,slot->normalized_bf16,slot->index_weights_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchWiden(stream,slot->index_weights_bf16,slot->index_weights_f32,rows,SPARK_DSV4_MODEL_INDEX_HEAD_COUNT,weight_scale);
	if ( error == cudaSuccess )
		error = SparkDsv4ModuleRunCompressor(state,slot,&layer->indexer.compressor,batch,kv_state,score_state,state->index_state_lane_stride,index_cache,state->index_lane_stride,0u,SPARK_DSV4_MODEL_INDEX_HEAD_DIMENSION,1u,rows);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchIndexerScore(stream,slot->index_q_bf16,index_cache,state->index_lane_stride,slot->row_lane_indices,slot->slot_counts,slot->index_weights_f32,slot->index_scores_f32,rows,state->index_slot_capacity,SPARK_DSV4_MODEL_INDEX_HEAD_COUNT,SPARK_DSV4_MODEL_INDEX_HEAD_DIMENSION);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchTopK(stream,slot->index_scores_f32,slot->slot_counts,state->index_slot_capacity,SPARK_DSV4_MODEL_INDEX_TOP_K,(int32_t)SPARK_DSV4_MODEL_SLIDING_WINDOW_TOKENS,slot->topk_idxs + SPARK_DSV4_MODEL_SLIDING_WINDOW_TOKENS,state->topk_column_count,rows);
	return(error);
}

// Attention index staging: the host fills the window and HCA parts and
// the CSA valid-slot counts into its own buffer, then one upload each -
// the CSA tail stays device-written by the top-k kernel.
static cudaError_t SparkDsv4ModuleStageTopk(SparkDsv4ModuleState *state, SparkDsv4ModuleSlot *slot, const SparkDsv4DecodeBatchView *batch, uint32_t layer_kind, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	cudaError_t error;
	SparkDsv4ModuleHostTopkFill(state,batch,state->host_topk_idxs,state->host_slot_counts,layer_kind,rows);
	error = cudaMemcpyAsync(slot->topk_idxs,state->host_topk_idxs,(uint64_t)rows * state->topk_column_count * sizeof(int32_t),cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess )
		error = cudaMemcpyAsync(slot->slot_counts,state->host_slot_counts,(uint64_t)rows * sizeof(uint32_t),cudaMemcpyHostToDevice,stream);
	return(error);
}

// One o-composition group: wo_a's block against the group's slice of the
// attention output, ranks landing at the group's offset - block-diagonal
// through the strided kernel.
static cudaError_t SparkDsv4ModuleRunOutputGroup(SparkDsv4ModuleSlot *slot, const SparkDsv4LayerWeights *layer, uint32_t group, uint32_t rows)
{
	SparkDsv4LinearView view = layer->attn.wo_a;
	uint64_t block_bytes = (uint64_t)SPARK_DSV4_MODEL_OUTPUT_LORA_RANK * SPARK_DSV4_MODEL_OUTPUT_GROUP_DIMENSION * SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES;
	view.rows = SPARK_DSV4_MODEL_OUTPUT_LORA_RANK;
	view.columns = SPARK_DSV4_MODEL_OUTPUT_GROUP_DIMENSION;
	return(SparkDsv4LaunchStridedLinear((cudaStream_t)slot->cuda_stream,&view,(const uint8_t *)layer->attn.wo_a.payload + (uint64_t)group * block_bytes,0,slot->attn_out_bf16,SPARK_DSV4_MODEL_ATTN_QUERY_DIMENSION,group * SPARK_DSV4_MODEL_OUTPUT_GROUP_DIMENSION,slot->o_ranks_bf16,(uint64_t)SPARK_DSV4_MODEL_OUTPUT_GROUP_COUNT * SPARK_DSV4_MODEL_OUTPUT_LORA_RANK,group * SPARK_DSV4_MODEL_OUTPUT_LORA_RANK,rows));
}

static cudaError_t SparkDsv4ModuleRunAttention(SparkDsv4ModuleState *state, SparkDsv4ModuleSlot *slot, const SparkDsv4LayerWeights *layer, const SparkDsv4DecodeBatchView *batch, uint32_t layer_index, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	uint32_t kind = SparkDsv4ModelLayerKind(layer_index),local = state->layer_local_by_layer[layer_index],group,row;
	const float *freqs = kind == SPARK_DSV4_MODEL_LAYER_KIND_SWA ? state->base_freqs_f32 : state->compress_freqs_f32;
	void *cache = (uint8_t *)state->kv_cache_bf16 + (uint64_t)local * state->max_active_sequence_count * state->cache_lane_block_elements * SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES;
	uint64_t lane_stride = state->cache_lane_block_elements,position;
	cudaError_t error;
	error = SparkDsv4LaunchLinear(stream,&layer->attn.wq_a,slot->normalized_bf16,slot->delta_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchRmsNorm(stream,slot->delta_bf16,layer->attn.q_norm_weight_bf16,slot->qr_bf16,rows,SPARK_DSV4_MODEL_QUERY_LORA_RANK,SPARK_DSV4_MODEL_RMS_NORM_EPSILON);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchLinear(stream,&layer->attn.wq_b,slot->qr_bf16,slot->q_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchQueryHeadRms(stream,slot->q_bf16,rows,SPARK_DSV4_MODEL_ATTN_QUERY_HEAD_COUNT,SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION,SPARK_DSV4_MODEL_RMS_NORM_EPSILON);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchRope(stream,slot->q_bf16,freqs,slot->row_positions,rows,SPARK_DSV4_MODEL_ATTN_QUERY_HEAD_COUNT,SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION,SPARK_DSV4_MODEL_ATTN_ROPE_DIMENSION,0u);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchLinear(stream,&layer->attn.wkv,slot->normalized_bf16,slot->kv_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchRmsNorm(stream,slot->kv_bf16,layer->attn.kv_norm_weight_bf16,slot->kv_bf16,rows,SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION,SPARK_DSV4_MODEL_RMS_NORM_EPSILON);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchRope(stream,slot->kv_bf16,freqs,slot->row_positions,rows,1u,SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION,SPARK_DSV4_MODEL_ATTN_ROPE_DIMENSION,0u);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchQuantSim(stream,slot->kv_bf16,rows,SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION,SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION - SPARK_DSV4_MODEL_ATTN_ROPE_DIMENSION,SPARK_DSV4_MODEL_KV_QUANT_BLOCK,0u);
	for (row = 0; error == cudaSuccess && row < rows; row++)
	{
		position = batch->row_positions[row];
		error = cudaMemcpyAsync((uint8_t *)cache + ((uint64_t)batch->row_lane_indices[row] * lane_stride + (position % SPARK_DSV4_MODEL_SLIDING_WINDOW_TOKENS) * SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION) * SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES,(const uint8_t *)slot->kv_bf16 + (uint64_t)row * SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION * SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES,SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION * SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES,cudaMemcpyDeviceToDevice,stream);
	}
	if ( error == cudaSuccess && kind != SPARK_DSV4_MODEL_LAYER_KIND_SWA )
		error = SparkDsv4ModuleRunCompressor(state,slot,&layer->compressor,batch,state->compress_kv_state_f32 + (uint64_t)state->compress_ordinal_by_layer[layer_index] * state->max_active_sequence_count * state->compress_state_lane_stride,state->compress_score_state_f32 + (uint64_t)state->compress_ordinal_by_layer[layer_index] * state->max_active_sequence_count * state->compress_state_lane_stride,state->compress_state_lane_stride,cache,lane_stride,SPARK_DSV4_MODEL_SLIDING_WINDOW_TOKENS,SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION,0u,rows);
	if ( error == cudaSuccess )
		error = SparkDsv4ModuleStageTopk(state,slot,batch,kind,rows);
	if ( error == cudaSuccess && kind == SPARK_DSV4_MODEL_LAYER_KIND_CSA )
		error = SparkDsv4ModuleRunIndexer(state,slot,layer,batch,state->csa_ordinal_by_layer[layer_index],rows);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchSparseAttn(stream,slot->q_bf16,cache,lane_stride,slot->row_lane_indices,slot->topk_idxs,state->topk_column_count,layer->attn.sink_f32,1.0f / sqrtf((float)SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION),slot->attn_out_bf16,rows,SPARK_DSV4_MODEL_ATTN_QUERY_HEAD_COUNT,SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchRope(stream,slot->attn_out_bf16,freqs,slot->row_positions,rows,SPARK_DSV4_MODEL_ATTN_QUERY_HEAD_COUNT,SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION,SPARK_DSV4_MODEL_ATTN_ROPE_DIMENSION,1u);
	for (group = 0; error == cudaSuccess && group < SPARK_DSV4_MODEL_OUTPUT_GROUP_COUNT; group++)
		error = SparkDsv4ModuleRunOutputGroup(slot,layer,group,rows);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchLinear(stream,&layer->attn.wo_b,slot->o_ranks_bf16,slot->delta_bf16,rows);
	return(error);
}

// A stacked-expert slice: expert e's block of the shared payload and
// scale, exposed as an ordinary view - fp4 nibbles pack two per byte and
// e8m0 scales one byte per 32 columns per row.
static void SparkDsv4ModuleExpertView(SparkDsv4LinearView *view, const SparkDsv4LinearView *stacked, uint32_t expert, uint64_t rows_per_expert, uint64_t columns)
{
	*view = *stacked;
	view->rows = (uint32_t)rows_per_expert;
	view->columns = (uint32_t)columns;
	view->payload = (const uint8_t *)stacked->payload + expert * rows_per_expert * columns / 2u;
	view->scale_e8m0 = stacked->scale_e8m0 + expert * rows_per_expert * (columns / SPARK_DSV4_STAGEPACK_FP4_SCALE_BLOCK);
}

// Counting sort of the batch's (row, rank) pairs by expert, dense slot
// lists per expert; the routing weight applies at the swiglu intermediate
// through the weight-slot indirection, per the dsv4 reference.
static SparkStatus SparkDsv4ModuleGroupByExpert(SparkDsv4ModuleSlot *slot, uint32_t rows, uint32_t *active_out)
{
	uint32_t counts[SPARK_DSV4_MODEL_ROUTED_EXPERT_COUNT];
	uint32_t pair_count = rows * SPARK_DSV4_MODEL_EXPERTS_PER_TOKEN,pair,expert,cursor = 0u,active = 0u;
	memset(counts,0,sizeof(counts));
	for (pair = 0; pair < pair_count; pair++)
	{
		expert = slot->host_moe_indices[pair];
		if ( expert >= SPARK_DSV4_MODEL_ROUTED_EXPERT_COUNT )
			return(SPARK_STATUS_VALIDATION_FAILED);
		counts[expert]++;
	}
	for (expert = 0; expert < SPARK_DSV4_MODEL_ROUTED_EXPERT_COUNT; expert++)
	{
		slot->host_expert_offsets[expert] = cursor;
		cursor += counts[expert];
		active += counts[expert] != 0u ? 1u : 0u;
		counts[expert] = slot->host_expert_offsets[expert];
	}
	slot->host_expert_offsets[SPARK_DSV4_MODEL_ROUTED_EXPERT_COUNT] = cursor;
	for (pair = 0; pair < pair_count; pair++)
	{
		expert = slot->host_moe_indices[pair];
		slot->host_grouped_rows[counts[expert]] = pair / SPARK_DSV4_MODEL_EXPERTS_PER_TOKEN;
		slot->host_grouped_weight_slots[counts[expert]] = pair;
		counts[expert]++;
	}
	*active_out = active;
	return(SPARK_STATUS_OK);
}

static cudaError_t SparkDsv4ModuleRunExpertGroup(SparkDsv4ModuleSlot *slot, const SparkDsv4MoeWeights *moe, uint32_t expert, uint32_t offset, uint32_t count)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	SparkDsv4LinearView expert_view;
	uint64_t inter = SPARK_DSV4_MODEL_EXPERT_INTERMEDIATE_DIMENSION,dim = SPARK_DSV4_MODEL_HIDDEN_DIMENSION,bf16 = SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES;
	const uint32_t *row_map = slot->grouped_rows_u32 + offset;
	uint8_t *slot_gate = (uint8_t *)slot->moe_slot_gate_bf16 + (uint64_t)offset * inter * bf16;
	uint8_t *slot_up = (uint8_t *)slot->moe_slot_up_bf16 + (uint64_t)offset * inter * bf16;
	uint8_t *slot_out = (uint8_t *)slot->moe_slot_out_bf16 + (uint64_t)offset * dim * bf16;
	cudaError_t error;
	SparkDsv4ModuleExpertView(&expert_view,&moe->experts_w1,expert,inter,dim);
	error = count >= 2u ? SparkDsv4LaunchExpertTile(stream,&expert_view,slot->normalized_bf16,row_map,slot_gate,count) : SparkDsv4LaunchGatherLinear(stream,&expert_view,slot->normalized_bf16,row_map,slot_gate,count);
	if ( error == cudaSuccess )
	{
		SparkDsv4ModuleExpertView(&expert_view,&moe->experts_w3,expert,inter,dim);
		error = count >= 2u ? SparkDsv4LaunchExpertTile(stream,&expert_view,slot->normalized_bf16,row_map,slot_up,count) : SparkDsv4LaunchGatherLinear(stream,&expert_view,slot->normalized_bf16,row_map,slot_up,count);
	}
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchSwigluClamp(stream,slot_gate,slot_up,count,SPARK_DSV4_MODEL_EXPERT_INTERMEDIATE_DIMENSION,SPARK_DSV4_MODEL_SWIGLU_LIMIT,slot->moe_weights_f32,slot->grouped_weight_slots_u32 + offset);
	if ( error == cudaSuccess )
	{
		SparkDsv4ModuleExpertView(&expert_view,&moe->experts_w2,expert,dim,inter);
		error = count >= 2u ? SparkDsv4LaunchExpertTile(stream,&expert_view,slot_up,0,slot_out,count) : SparkDsv4LaunchGatherLinear(stream,&expert_view,slot_up,0,slot_out,count);
	}
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchScatterAdd(stream,slot->ffn_accum_bf16,slot_out,row_map,count,(uint32_t)dim);
	return(error);
}

static SparkStatus SparkDsv4ModuleRunMoeRouted(SparkDsv4ModuleSlot *slot, const SparkDsv4MoeWeights *moe, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	uint32_t expert,offset,count,active = 0u;
	SparkStatus status;
	cudaError_t error;
	error = cudaStreamSynchronize(stream);
	if ( error != cudaSuccess )
		return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,"moe_readback"));
	status = SparkDsv4ModuleGroupByExpert(slot,rows,&active);
	if ( status != SPARK_STATUS_OK )
		return(status);
	for (expert = 0; error == cudaSuccess && expert < SPARK_DSV4_MODEL_ROUTED_EXPERT_COUNT; expert++)
	{
		offset = slot->host_expert_offsets[expert];
		count = slot->host_expert_offsets[expert + 1u] - offset;
		if ( count == 0u )
			continue;
		error = SparkDsv4ModuleRunExpertGroup(slot,moe,expert,offset,count);
	}
	return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,"moe_routed"));
}

static SparkStatus SparkDsv4ModuleRunMoe(SparkDsv4ModuleSlot *slot, const SparkDsv4LayerWeights *layer, uint32_t layer_index, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	const SparkDsv4MoeWeights *moe = &layer->moe;
	uint32_t hash = SparkDsv4StagePackLayerIsHashRouted(layer_index);
	cudaError_t error;
	error = SparkDsv4LaunchGateScores(stream,&moe->gate,slot->normalized_bf16,slot->moe_scores_f32,rows);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchGateSelect(stream,slot->moe_scores_f32,hash != 0u ? 0 : moe->gate_bias_f32,hash != 0u ? moe->gate_tid2eid_u32 : 0,slot->input_token_ids,rows,SPARK_DSV4_MODEL_ROUTED_EXPERT_COUNT,SPARK_DSV4_MODEL_EXPERTS_PER_TOKEN,SPARK_DSV4_MODEL_ROUTED_SCALING_FACTOR,slot->moe_indices_u32,slot->moe_weights_f32);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchLinear(stream,&moe->shared_w1,slot->normalized_bf16,slot->ffn_gate_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchLinear(stream,&moe->shared_w3,slot->normalized_bf16,slot->ffn_up_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchSwigluClamp(stream,slot->ffn_gate_bf16,slot->ffn_up_bf16,rows,SPARK_DSV4_MODEL_EXPERT_INTERMEDIATE_DIMENSION,SPARK_DSV4_MODEL_SWIGLU_LIMIT,0,0);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchLinear(stream,&moe->shared_w2,slot->ffn_up_bf16,slot->ffn_accum_bf16,rows);
	if ( error != cudaSuccess )
		return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,"moe_shared"));
	return(SparkDsv4ModuleRunMoeRouted(slot,moe,rows));
}

static SparkStatus SparkDsv4ModuleRunLayer(SparkDsv4ModuleState *state, SparkDsv4ModuleSlot *slot, const SparkDsv4DecodeBatchView *batch, uint32_t layer_index, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	const SparkDsv4LayerWeights *layer = &state->layers[layer_index];
	cudaError_t error;
	SparkStatus status;
	error = SparkDsv4ModuleHcEnter(slot,layer->hc.attn_fn_f32,layer->hc.attn_scale_f32,layer->hc.attn_base_f32,rows);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchRmsNorm(stream,slot->reduced_bf16,layer->attn_norm_bf16,slot->normalized_bf16,rows,SPARK_DSV4_MODEL_HIDDEN_DIMENSION,SPARK_DSV4_MODEL_RMS_NORM_EPSILON);
	if ( error == cudaSuccess )
		error = SparkDsv4ModuleRunAttention(state,slot,layer,batch,layer_index,rows);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchHcPost(stream,slot->delta_bf16,slot->residual_bf16,slot->post_f32,slot->comb_f32,slot->streams_bf16,rows,SPARK_DSV4_MODEL_HC_STREAM_COUNT,SPARK_DSV4_MODEL_HIDDEN_DIMENSION);
	if ( error != cudaSuccess )
		return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,"attn_side"));
	error = SparkDsv4ModuleHcEnter(slot,layer->hc.ffn_fn_f32,layer->hc.ffn_scale_f32,layer->hc.ffn_base_f32,rows);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchRmsNorm(stream,slot->reduced_bf16,layer->ffn_norm_bf16,slot->normalized_bf16,rows,SPARK_DSV4_MODEL_HIDDEN_DIMENSION,SPARK_DSV4_MODEL_RMS_NORM_EPSILON);
	if ( error != cudaSuccess )
		return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,"ffn_enter"));
	status = SparkDsv4ModuleRunMoe(slot,layer,layer_index,rows);
	if ( status != SPARK_STATUS_OK )
		return(status);
	error = SparkDsv4LaunchHcPost(stream,slot->ffn_accum_bf16,slot->residual_bf16,slot->post_f32,slot->comb_f32,slot->streams_bf16,rows,SPARK_DSV4_MODEL_HC_STREAM_COUNT,SPARK_DSV4_MODEL_HIDDEN_DIMENSION);
	return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,"ffn_side"));
}

static SparkStatus SparkDsv4ModuleFinish(SparkDsv4ModuleState *state, SparkDsv4ModuleSlot *slot, SparkDsv4ResidentDecodeStageFrameContext *context, SparkModelDriverFrame *frame, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	uint32_t out_index = state->owns_embedding != 0u ? 1u : 0u;
	SparkStatus status = SPARK_STATUS_OK;
	cudaError_t error = cudaSuccess;
	if ( state->owns_final_head != 0u )
	{
		error = SparkDsv4LaunchHcMix(stream,slot->streams_bf16,state->hc_head_fn_f32,slot->mixes_f32,rows,SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS,SPARK_DSV4_MODEL_HC_STREAM_COUNT,SPARK_DSV4_MODEL_RMS_NORM_EPSILON);
		if ( error == cudaSuccess )
			error = SparkDsv4LaunchHcHeadReduce(stream,slot->streams_bf16,slot->mixes_f32,state->hc_head_scale_value,state->hc_head_base_f32,SPARK_DSV4_MODEL_HC_EPSILON,slot->reduced_bf16,rows,SPARK_DSV4_MODEL_HC_STREAM_COUNT,SPARK_DSV4_MODEL_HIDDEN_DIMENSION);
		if ( error == cudaSuccess )
			error = SparkDsv4LaunchRmsNorm(stream,slot->reduced_bf16,state->final_norm_weight_bf16,slot->normalized_bf16,rows,SPARK_DSV4_MODEL_HIDDEN_DIMENSION,SPARK_DSV4_MODEL_RMS_NORM_EPSILON);
		if ( error == cudaSuccess )
			error = SparkDsv4LaunchHeadScreenedArgmax(stream,slot->normalized_bf16,state->lm_head_weight_bf16,state->head_shadow_payload,state->head_shadow_scale,state->head_error_norm_f32,slot->head_logits_bf16,slot->head_candidate_ids_u32,slot->head_candidate_counts_u32,slot->output_token_ids,rows,SPARK_DSV4_MODEL_VOCAB_COUNT,SPARK_DSV4_MODEL_HIDDEN_DIMENSION);
		if ( error == cudaSuccess )
			error = cudaMemcpyAsync(frame->buffers[out_index].address,slot->output_token_ids,(uint64_t)rows * sizeof(uint32_t),cudaMemcpyDeviceToHost,stream);
		status = SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,"head");
	}
	else
	{
		if ( context->hidden_output_send_function == 0 || context->hidden_output_transport_session == 0 )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		context->hidden_output_packet.active_sequence_count = rows;
		context->hidden_output_packet.hidden_dimension = SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS;
		context->hidden_output_packet.bytes_per_sequence = SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS * SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES;
		context->hidden_output_packet.hidden_bf16 = slot->streams_bf16;
		context->hidden_output_packet.cuda_stream = stream;
		context->hidden_output_packet.sideband_payload = 0;
		context->hidden_output_packet.sideband_kind = 0u;
		context->hidden_output_packet.sideband_bytes_per_sequence = 0u;
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,cudaStreamSynchronize(stream),"sync");
	if ( status == SPARK_STATUS_OK && state->owns_final_head == 0u )
		status = context->hidden_output_send_function(context->hidden_output_transport_session,&context->hidden_output_packet);
	return(status);
}

SparkStatus SparkDsv4ResidentDecodeStageExecute(void *module_state, SparkModelDriverFrame *frame)
{
	SparkDsv4ModuleState *state = (SparkDsv4ModuleState *)module_state;
	SparkDsv4ResidentDecodeStageFrameContext *context = 0;
	const SparkDsv4DecodeBatchView *batch;
	SparkDsv4ModuleSlot *slot;
	uint32_t slot_index = 0u,rows,layer;
	SparkStatus status;
	if ( state == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkDsv4ModuleValidateFrame(state,frame,(const SparkDsv4ResidentDecodeStageFrameContext **)&context);
	if ( status != SPARK_STATUS_OK )
		return(status);
	batch = context->decode_batch;
	rows = batch->row_count;
	status = SparkStageModuleSlotClaim(state->slot_states,state->pipeline_slot_count,&slot_index);
	if ( status != SPARK_STATUS_OK )
		return(status);
	slot = &state->slots[slot_index];
	status = SparkDsv4ModuleStageRows(state,batch,slot);
	if ( status == SPARK_STATUS_OK )
		status = SparkDsv4ModuleBeginStreams(state,slot,context,frame,rows);
	for (layer = state->first_layer_index; status == SPARK_STATUS_OK && layer < state->first_layer_index + state->layer_count; layer++)
		status = SparkDsv4ModuleRunLayer(state,slot,batch,layer,rows);
	if ( status == SPARK_STATUS_OK )
		status = SparkDsv4ModuleFinish(state,slot,context,frame,rows);
	SparkStageModuleSlotRelease(state->slot_states,slot_index);
	if ( status != SPARK_STATUS_OK )
		return(status);
	atomic_fetch_add(&state->frames_executed,1u);
	atomic_fetch_add(&state->tokens_emitted,rows);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkDsv4ResidentDecodeStageAdmit(void *module_state, const SparkModelDriverAdmissionRequest *request, SparkModelDriverAdmissionDecision *decision)
{
	SparkDsv4ModuleState *state = (SparkDsv4ModuleState *)module_state;
	if ( state == 0 || request == 0 || decision == 0 || request->descriptor_bytes != (uint32_t)sizeof(*request) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(decision,0,sizeof(*decision));
	decision->descriptor_bytes = (uint32_t)sizeof(*decision);
	decision->accepted = 1u;
	decision->available_dispatch_slot_count = state->pipeline_slot_count;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkDsv4ResidentDecodeStageSnapshot(void *module_state, uint32_t program_id, SparkModelDriverRuntimeSnapshot *snapshot)
{
	SparkDsv4ModuleState *state = (SparkDsv4ModuleState *)module_state;
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

void SparkDsv4ResidentDecodeStageDestroy(void *module_state)
{
	SparkDsv4ModuleState *state = (SparkDsv4ModuleState *)module_state;
	uint32_t slot_index;
	if ( state == 0 )
		return;
	for (slot_index = 0; slot_index < state->pipeline_slot_count; slot_index++)
	{
		if ( state->slots[slot_index].cuda_stream != 0 )
			cudaStreamDestroy((cudaStream_t)state->slots[slot_index].cuda_stream);
		if ( state->slots[slot_index].host_moe_indices != 0 )
			cudaFreeHost(state->slots[slot_index].host_moe_indices);
		if ( state->slots[slot_index].host_grouped_rows != 0 )
			cudaFreeHost(state->slots[slot_index].host_grouped_rows);
		if ( state->slots[slot_index].host_grouped_weight_slots != 0 )
			cudaFreeHost(state->slots[slot_index].host_grouped_weight_slots);
	}
	SparkStageModuleLedgerRelease(&state->ledger);
	free(state->host_topk_idxs);
	free(state);
}

SparkStatus SparkDsv4ResidentDecodeStageInitialize(const SparkFirmwareModuleConfiguration *configuration, const SparkFirmwareModuleHostServices *host_services, void **module_state)
{
	SparkDsv4ModuleState *state;
	const char *pack_path = 0;
	SparkStatus status;
	cudaError_t error;
	uint32_t slot_index;
	(void)host_services;
	if ( configuration == 0 || module_state == 0 || configuration->abi_version != SPARK_FIRMWARE_MODULE_ABI_VERSION )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	state = (SparkDsv4ModuleState *)calloc(1u,sizeof(*state));
	if ( state == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	state->ledger.module_tag = SPARK_DSV4_MODULE_TAG;
	status = SparkDsv4ModuleConfigure(state);
	if ( status == SPARK_STATUS_OK )
		status = SparkDsv4ModuleValidateSlice(state);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentText(SPARK_DSV4_MODULE_TAG,"SPARK_DSV4_STAGE_PACK_PATH",&pack_path);
	if ( status == SPARK_STATUS_OK )
	{
		SparkDsv4ModuleBuildOrdinals(state);
		status = SparkDsv4ModuleLoadPack(state,pack_path);
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkDsv4ModuleUploadFreqs(state);
	if ( status == SPARK_STATUS_OK )
		status = SparkDsv4ModuleAllocatePools(state);
	if ( status == SPARK_STATUS_OK )
		status = SparkDsv4ModuleBuildHeadShadow(state);
	if ( status == SPARK_STATUS_OK )
	{
		state->host_topk_idxs = (int32_t *)malloc((uint64_t)state->max_active_sequence_count * state->topk_column_count * sizeof(int32_t));
		if ( state->host_topk_idxs == 0 )
			status = SPARK_STATUS_CAPACITY_EXCEEDED;
	}
	if ( status == SPARK_STATUS_OK && state->owns_final_head != 0u )
	{
		error = cudaMemcpy(&state->hc_head_scale_value,state->hc_head_scale_f32,sizeof(float),cudaMemcpyDeviceToHost);
		status = SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,"head_scale_read");
	}
	for (slot_index = 0; status == SPARK_STATUS_OK && slot_index < state->pipeline_slot_count; slot_index++)
		status = SparkDsv4ModuleAllocateSlot(state,&state->slots[slot_index]);
	if ( status != SPARK_STATUS_OK )
	{
		SparkDsv4ResidentDecodeStageDestroy(state);
		return(status);
	}
	fprintf(stderr,"%s ready stage=%u/%u slice=%u+%u compress=%u csa=%u lanes=%u max_seq=%u topk_cols=%u device_gib=%.1f\n",SPARK_DSV4_MODULE_TAG,state->stage_index,state->stage_count,state->first_layer_index,state->layer_count,state->compress_layer_count,state->csa_layer_count,state->max_active_sequence_count,state->max_sequence_positions,state->topk_column_count,(double)state->ledger.device_bytes_resident / (1024.0 * 1024.0 * 1024.0));
	*module_state = state;
	return(SPARK_STATUS_OK);
}
