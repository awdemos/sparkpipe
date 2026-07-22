#pragma once

#include <stdint.h>

#include "sparkpipe/spark_k3_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_status.h"

/*
 * K3 stage pack: a single file holding every tensor this module makes
 * resident, plus the geometry the tensors were produced for.
 *
 * The header restates the model geometry so a pack can never be paired with a
 * driver compiled for different dimensions. Every field is compared against
 * the compiled constant and any mismatch is a hard load failure: the driver
 * never adopts a pack's value and never falls back to a default, because a
 * silently accepted dimension is a wrong answer that looks like a working
 * model. When the K3 report lands and a GUESS constant changes, packs built
 * against the old geometry are rejected by construction.
 *
 * Layout: [header][directory: tensor_count entries][payload bytes].
 * All offsets are absolute file offsets. Payload of a tensor is contiguous.
 */

#define SPARK_K3_STAGEPACK_MAGIC 0x5053334bu /* 'K3SP' little endian */
#define SPARK_K3_STAGEPACK_FORMAT_VERSION 1u
#define SPARK_K3_STAGEPACK_GLOBAL_LAYER UINT32_MAX
#define SPARK_K3_STAGEPACK_PAYLOAD_ALIGNMENT 256u

typedef enum SparkK3StagePackTensorKind
{
	SPARK_K3_STAGEPACK_TENSOR_EMBEDDING = 0,
	SPARK_K3_STAGEPACK_TENSOR_ATTNRES_ATTENTION_QUERY = 1,
	SPARK_K3_STAGEPACK_TENSOR_ATTNRES_ATTENTION_NORM = 2,
	SPARK_K3_STAGEPACK_TENSOR_ATTNRES_MLP_QUERY = 3,
	SPARK_K3_STAGEPACK_TENSOR_ATTNRES_MLP_NORM = 4,
	SPARK_K3_STAGEPACK_TENSOR_ATTNRES_FINAL_QUERY = 5,
	SPARK_K3_STAGEPACK_TENSOR_ATTNRES_FINAL_NORM = 6,
	SPARK_K3_STAGEPACK_TENSOR_ATTENTION_NORM = 7,
	SPARK_K3_STAGEPACK_TENSOR_MLP_NORM = 8,
	SPARK_K3_STAGEPACK_TENSOR_KDA_QUERY = 9,
	SPARK_K3_STAGEPACK_TENSOR_KDA_KEY = 10,
	SPARK_K3_STAGEPACK_TENSOR_KDA_VALUE = 11,
	SPARK_K3_STAGEPACK_TENSOR_KDA_DECAY_LOW = 12,
	SPARK_K3_STAGEPACK_TENSOR_KDA_DECAY_HIGH = 13,
	SPARK_K3_STAGEPACK_TENSOR_KDA_BETA = 14,
	SPARK_K3_STAGEPACK_TENSOR_KDA_GATE_LOW = 15,
	SPARK_K3_STAGEPACK_TENSOR_KDA_GATE_HIGH = 16,
	SPARK_K3_STAGEPACK_TENSOR_KDA_OUTPUT = 17,
	SPARK_K3_STAGEPACK_TENSOR_KDA_HEAD_NORM = 18,
	SPARK_K3_STAGEPACK_TENSOR_MLA_QUERY_A = 19,
	SPARK_K3_STAGEPACK_TENSOR_MLA_QUERY_A_NORM = 20,
	SPARK_K3_STAGEPACK_TENSOR_MLA_QUERY_B = 21,
	SPARK_K3_STAGEPACK_TENSOR_MLA_KV_A = 22,
	SPARK_K3_STAGEPACK_TENSOR_MLA_KV_A_NORM = 23,
	SPARK_K3_STAGEPACK_TENSOR_MLA_KV_B = 24,
	SPARK_K3_STAGEPACK_TENSOR_MLA_HEAD_GATE = 25,
	SPARK_K3_STAGEPACK_TENSOR_MLA_OUTPUT = 26,
	SPARK_K3_STAGEPACK_TENSOR_MOE_ROUTER = 27,
	SPARK_K3_STAGEPACK_TENSOR_MOE_ROUTER_BIAS = 28,
	SPARK_K3_STAGEPACK_TENSOR_MOE_EXPERT_GATE = 29,
	SPARK_K3_STAGEPACK_TENSOR_MOE_EXPERT_UP = 30,
	SPARK_K3_STAGEPACK_TENSOR_MOE_EXPERT_DOWN = 31,
	SPARK_K3_STAGEPACK_TENSOR_MOE_SHARED_GATE = 32,
	SPARK_K3_STAGEPACK_TENSOR_MOE_SHARED_UP = 33,
	SPARK_K3_STAGEPACK_TENSOR_MOE_SHARED_DOWN = 34,
	SPARK_K3_STAGEPACK_TENSOR_FINAL_NORM = 35,
	SPARK_K3_STAGEPACK_TENSOR_LM_HEAD_RESTRICTED = 36,
	SPARK_K3_STAGEPACK_TENSOR_RESTRICTED_TOKEN_IDS = 37,
	SPARK_K3_STAGEPACK_TENSOR_KIND_COUNT = 38
} SparkK3StagePackTensorKind;

typedef struct SparkK3StagePackHeader
{
	uint32_t magic;
	uint32_t format_version;
	uint32_t header_bytes;
	uint32_t directory_entry_bytes;
	uint32_t tensor_count;
	uint32_t hidden_dimension;
	uint32_t layer_count;
	uint32_t first_layer_index;
	uint32_t first_routed_layer;
	uint32_t attention_period;
	uint32_t global_attention_phase;
	uint32_t attnres_block_layers;
	uint32_t kda_head_count;
	uint32_t kda_head_key_dimension;
	uint32_t kda_head_value_dimension;
	uint32_t kda_low_rank_dimension;
	uint32_t mla_head_count;
	uint32_t mla_query_a_dimension;
	uint32_t mla_latent_dimension;
	uint32_t mla_qk_nope_head_dimension;
	uint32_t mla_rope_dimension;
	uint32_t mla_value_head_dimension;
	uint32_t moe_expert_count;
	uint32_t moe_top_k;
	uint32_t moe_shared_expert_count;
	uint32_t moe_intermediate_dimension;
	uint32_t dense_intermediate_dimension;
	uint32_t mxfp4_group_size;
	uint32_t output_vocab_count;
	uint32_t restricted_vocab_count;
	uint64_t directory_offset;
	uint64_t payload_offset;
	uint64_t file_bytes;
} SparkK3StagePackHeader;

typedef struct SparkK3StagePackEntry
{
	uint32_t tensor_kind;
	uint32_t layer_index;
	uint32_t weight_format;
	uint32_t rows;
	uint32_t columns;
	uint32_t scale_group_size;
	uint64_t payload_offset;
	uint64_t payload_bytes;
	uint64_t scale_offset;
	uint64_t scale_bytes;
} SparkK3StagePackEntry;

/*
 * Fixed wire sizes. The structs are ordered so natural alignment produces no
 * padding on the LP64 targets this module builds for; the asserts make that a
 * compile error rather than a silent format drift.
 */
#define SPARK_K3_STAGEPACK_HEADER_BYTES 144u
#define SPARK_K3_STAGEPACK_ENTRY_BYTES 56u
_Static_assert(sizeof(SparkK3StagePackHeader) == SPARK_K3_STAGEPACK_HEADER_BYTES,"k3 stage pack header must be 144 wire bytes");
_Static_assert(sizeof(SparkK3StagePackEntry) == SPARK_K3_STAGEPACK_ENTRY_BYTES,"k3 stage pack directory entry must be 56 wire bytes");

/*
 * The expected geometry, materialized from the compiled constants. A loader
 * fills a header from the file and compares field by field.
 */
static inline void SparkK3StagePackExpectedGeometry(SparkK3StagePackHeader *header,uint32_t first_layer_index,uint32_t layer_count,uint32_t tensor_count)
{
	header->magic = SPARK_K3_STAGEPACK_MAGIC;
	header->format_version = SPARK_K3_STAGEPACK_FORMAT_VERSION;
	header->header_bytes = SPARK_K3_STAGEPACK_HEADER_BYTES;
	header->directory_entry_bytes = SPARK_K3_STAGEPACK_ENTRY_BYTES;
	header->tensor_count = tensor_count;
	header->hidden_dimension = SPARK_K3_MODEL_HIDDEN_DIMENSION;
	header->layer_count = layer_count;
	header->first_layer_index = first_layer_index;
	header->first_routed_layer = SPARK_K3_MODEL_FIRST_ROUTED_LAYER;
	header->attention_period = SPARK_K3_MODEL_ATTENTION_PERIOD;
	header->global_attention_phase = SPARK_K3_MODEL_GLOBAL_ATTENTION_PHASE;
	header->attnres_block_layers = SPARK_K3_MODEL_ATTNRES_BLOCK_LAYERS;
	header->kda_head_count = SPARK_K3_MODEL_KDA_HEAD_COUNT;
	header->kda_head_key_dimension = SPARK_K3_MODEL_KDA_HEAD_KEY_DIMENSION;
	header->kda_head_value_dimension = SPARK_K3_MODEL_KDA_HEAD_VALUE_DIMENSION;
	header->kda_low_rank_dimension = SPARK_K3_MODEL_KDA_LOW_RANK_DIMENSION;
	header->mla_head_count = SPARK_K3_MODEL_MLA_HEAD_COUNT;
	header->mla_query_a_dimension = SPARK_K3_MODEL_MLA_QUERY_A_DIMENSION;
	header->mla_latent_dimension = SPARK_K3_MODEL_MLA_LATENT_DIMENSION;
	header->mla_qk_nope_head_dimension = SPARK_K3_MODEL_MLA_QK_NOPE_HEAD_DIMENSION;
	header->mla_rope_dimension = SPARK_K3_MODEL_MLA_ROPE_DIMENSION;
	header->mla_value_head_dimension = SPARK_K3_MODEL_MLA_VALUE_HEAD_DIMENSION;
	header->moe_expert_count = SPARK_K3_MODEL_MOE_EXPERT_COUNT;
	header->moe_top_k = SPARK_K3_MODEL_MOE_TOP_K;
	header->moe_shared_expert_count = SPARK_K3_MODEL_MOE_SHARED_EXPERT_COUNT;
	header->moe_intermediate_dimension = SPARK_K3_MODEL_MOE_INTERMEDIATE_DIMENSION;
	header->dense_intermediate_dimension = SPARK_K3_MODEL_DENSE_INTERMEDIATE_DIMENSION;
	header->mxfp4_group_size = SPARK_K3_MODEL_MXFP4_GROUP_SIZE;
	header->output_vocab_count = SPARK_K3_MODEL_OUTPUT_VOCAB_COUNT;
	header->restricted_vocab_count = SPARK_K3_MODEL_RESTRICTED_VOCAB_COUNT;
	header->directory_offset = SPARK_K3_STAGEPACK_HEADER_BYTES;
	header->payload_offset = 0u;
	header->file_bytes = 0u;
}

/*
 * Compare every geometry field. Returns 0 on match, or the negative index of
 * the first mismatching field so the caller can name it without a string
 * table on the hot path.
 */
static inline int32_t SparkK3StagePackCompareGeometry(const SparkK3StagePackHeader *file_header,const SparkK3StagePackHeader *expected)
{
	if ( file_header->magic != expected->magic )
		return(-1);
	if ( file_header->format_version != expected->format_version )
		return(-2);
	if ( file_header->header_bytes != expected->header_bytes )
		return(-3);
	if ( file_header->directory_entry_bytes != expected->directory_entry_bytes )
		return(-4);
	if ( file_header->hidden_dimension != expected->hidden_dimension )
		return(-5);
	if ( file_header->layer_count != expected->layer_count )
		return(-6);
	if ( file_header->first_layer_index != expected->first_layer_index )
		return(-7);
	if ( file_header->first_routed_layer != expected->first_routed_layer )
		return(-8);
	if ( file_header->attention_period != expected->attention_period )
		return(-9);
	if ( file_header->global_attention_phase != expected->global_attention_phase )
		return(-10);
	if ( file_header->attnres_block_layers != expected->attnres_block_layers )
		return(-11);
	if ( file_header->kda_head_count != expected->kda_head_count )
		return(-12);
	if ( file_header->kda_head_key_dimension != expected->kda_head_key_dimension )
		return(-13);
	if ( file_header->kda_head_value_dimension != expected->kda_head_value_dimension )
		return(-14);
	if ( file_header->kda_low_rank_dimension != expected->kda_low_rank_dimension )
		return(-15);
	if ( file_header->mla_head_count != expected->mla_head_count )
		return(-16);
	if ( file_header->mla_query_a_dimension != expected->mla_query_a_dimension )
		return(-17);
	if ( file_header->mla_latent_dimension != expected->mla_latent_dimension )
		return(-18);
	if ( file_header->mla_qk_nope_head_dimension != expected->mla_qk_nope_head_dimension )
		return(-19);
	if ( file_header->mla_rope_dimension != expected->mla_rope_dimension )
		return(-20);
	if ( file_header->mla_value_head_dimension != expected->mla_value_head_dimension )
		return(-21);
	if ( file_header->moe_expert_count != expected->moe_expert_count )
		return(-22);
	if ( file_header->moe_top_k != expected->moe_top_k )
		return(-23);
	if ( file_header->moe_shared_expert_count != expected->moe_shared_expert_count )
		return(-24);
	if ( file_header->moe_intermediate_dimension != expected->moe_intermediate_dimension )
		return(-25);
	if ( file_header->dense_intermediate_dimension != expected->dense_intermediate_dimension )
		return(-26);
	if ( file_header->mxfp4_group_size != expected->mxfp4_group_size )
		return(-27);
	if ( file_header->output_vocab_count != expected->output_vocab_count )
		return(-28);
	if ( file_header->restricted_vocab_count != expected->restricted_vocab_count )
		return(-29);
	if ( file_header->tensor_count == 0u )
		return(-30);
	if ( file_header->payload_offset < (uint64_t)file_header->header_bytes + ((uint64_t)file_header->tensor_count * file_header->directory_entry_bytes) )
		return(-31);
	return(0);
}

static inline const char *SparkK3StagePackGeometryFieldName(int32_t compare_result)
{
	static const char *names[] = {
		"ok","magic","format_version","header_bytes","directory_entry_bytes","hidden_dimension",
		"layer_count","first_layer_index","first_routed_layer","attention_period",
		"global_attention_phase","attnres_block_layers","kda_head_count","kda_head_key_dimension",
		"kda_head_value_dimension","kda_low_rank_dimension","mla_head_count","mla_query_a_dimension",
		"mla_latent_dimension","mla_qk_nope_head_dimension","mla_rope_dimension",
		"mla_value_head_dimension","moe_expert_count","moe_top_k","moe_shared_expert_count",
		"moe_intermediate_dimension","dense_intermediate_dimension","mxfp4_group_size",
		"output_vocab_count","restricted_vocab_count","tensor_count","payload_offset" };
	int32_t index = -compare_result;
	if ( index < 0 || index >= (int32_t)(sizeof(names) / sizeof(names[0])) )
		return("unknown");
	return(names[index]);
}

/* Payload bytes a tensor of this shape and format must carry. */
static inline uint64_t SparkK3StagePackPayloadBytes(uint32_t weight_format,uint32_t rows,uint32_t columns)
{
	uint64_t elements = (uint64_t)rows * (uint64_t)columns;
	if ( weight_format == SPARK_K3_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_MXFP4_E2M1 )
		return(elements / 2u);
	if ( weight_format == SPARK_K3_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_F32 || weight_format == SPARK_K3_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_U32 )
		return(elements * 4u);
	return(elements * (uint64_t)SPARK_K3_MODEL_BF16_ELEMENT_BYTES);
}

static inline uint64_t SparkK3StagePackScaleBytes(uint32_t weight_format,uint32_t rows,uint32_t columns)
{
	if ( weight_format == SPARK_K3_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_MXFP4_E2M1 )
		return(((uint64_t)rows * (uint64_t)columns) / (uint64_t)SPARK_K3_MODEL_MXFP4_GROUP_SIZE);
	return(0u);
}

/*
 * Tensor shape table: the single place that says what every tensor kind must
 * look like. The loader validates against it, the pack synthesizer emits
 * against it and the cpu reference reads against it, so a shape can never
 * disagree between producer and consumer.
 *
 * rows is the output extent (the slow axis in memory), columns the input
 * extent (the fast, quantized axis). quantizable says whether the tensor may
 * be carried as MXFP4; everything else must arrive in its natural format:
 * norms, pseudo-queries, the router and the embedding stay bf16, the router
 * bias is f32 and the restricted token id list is u32.
 */
typedef struct SparkK3StagePackTensorShape
{
	uint32_t rows;
	uint32_t columns;
	uint32_t natural_format;
	uint32_t quantizable;
	uint32_t per_layer;
} SparkK3StagePackTensorShape;

static inline int32_t SparkK3StagePackTensorShapeOf(uint32_t tensor_kind,SparkK3StagePackTensorShape *shape)
{
	shape->natural_format = SPARK_K3_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16;
	shape->quantizable = 0u;
	shape->per_layer = 1u;
	switch ( tensor_kind )
	{
	case SPARK_K3_STAGEPACK_TENSOR_EMBEDDING:
		shape->rows = SPARK_K3_MODEL_OUTPUT_VOCAB_COUNT;
		shape->columns = SPARK_K3_MODEL_HIDDEN_DIMENSION;
		shape->per_layer = 0u;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_ATTNRES_FINAL_QUERY:
	case SPARK_K3_STAGEPACK_TENSOR_ATTNRES_FINAL_NORM:
	case SPARK_K3_STAGEPACK_TENSOR_FINAL_NORM:
		shape->rows = 1u;
		shape->columns = SPARK_K3_MODEL_HIDDEN_DIMENSION;
		shape->per_layer = 0u;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_LM_HEAD_RESTRICTED:
		shape->rows = SPARK_K3_MODEL_RESTRICTED_VOCAB_COUNT;
		shape->columns = SPARK_K3_MODEL_HIDDEN_DIMENSION;
		shape->per_layer = 0u;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_RESTRICTED_TOKEN_IDS:
		shape->rows = 1u;
		shape->columns = SPARK_K3_MODEL_RESTRICTED_VOCAB_COUNT;
		shape->natural_format = SPARK_K3_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_U32;
		shape->per_layer = 0u;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_ATTNRES_ATTENTION_QUERY:
	case SPARK_K3_STAGEPACK_TENSOR_ATTNRES_ATTENTION_NORM:
	case SPARK_K3_STAGEPACK_TENSOR_ATTNRES_MLP_QUERY:
	case SPARK_K3_STAGEPACK_TENSOR_ATTNRES_MLP_NORM:
	case SPARK_K3_STAGEPACK_TENSOR_ATTENTION_NORM:
	case SPARK_K3_STAGEPACK_TENSOR_MLP_NORM:
		shape->rows = 1u;
		shape->columns = SPARK_K3_MODEL_HIDDEN_DIMENSION;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_KDA_QUERY:
	case SPARK_K3_STAGEPACK_TENSOR_KDA_KEY:
		shape->rows = SPARK_K3_MODEL_KDA_QK_DIMENSION;
		shape->columns = SPARK_K3_MODEL_HIDDEN_DIMENSION;
		shape->quantizable = 1u;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_KDA_VALUE:
		shape->rows = SPARK_K3_MODEL_KDA_VALUE_DIMENSION;
		shape->columns = SPARK_K3_MODEL_HIDDEN_DIMENSION;
		shape->quantizable = 1u;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_KDA_DECAY_LOW:
	case SPARK_K3_STAGEPACK_TENSOR_KDA_GATE_LOW:
		shape->rows = SPARK_K3_MODEL_KDA_LOW_RANK_DIMENSION;
		shape->columns = SPARK_K3_MODEL_HIDDEN_DIMENSION;
		shape->quantizable = 1u;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_KDA_DECAY_HIGH:
		shape->rows = SPARK_K3_MODEL_KDA_QK_DIMENSION;
		shape->columns = SPARK_K3_MODEL_KDA_LOW_RANK_DIMENSION;
		shape->quantizable = 1u;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_KDA_GATE_HIGH:
		shape->rows = SPARK_K3_MODEL_KDA_VALUE_DIMENSION;
		shape->columns = SPARK_K3_MODEL_KDA_LOW_RANK_DIMENSION;
		shape->quantizable = 1u;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_KDA_BETA:
		shape->rows = SPARK_K3_MODEL_KDA_HEAD_COUNT;
		shape->columns = SPARK_K3_MODEL_HIDDEN_DIMENSION;
		shape->quantizable = 1u;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_KDA_OUTPUT:
		shape->rows = SPARK_K3_MODEL_HIDDEN_DIMENSION;
		shape->columns = SPARK_K3_MODEL_KDA_VALUE_DIMENSION;
		shape->quantizable = 1u;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_KDA_HEAD_NORM:
		shape->rows = 1u;
		shape->columns = SPARK_K3_MODEL_KDA_HEAD_VALUE_DIMENSION;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_MLA_QUERY_A:
		shape->rows = SPARK_K3_MODEL_MLA_QUERY_A_DIMENSION;
		shape->columns = SPARK_K3_MODEL_HIDDEN_DIMENSION;
		shape->quantizable = 1u;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_MLA_QUERY_A_NORM:
		shape->rows = 1u;
		shape->columns = SPARK_K3_MODEL_MLA_QUERY_A_DIMENSION;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_MLA_QUERY_B:
		shape->rows = SPARK_K3_MODEL_MLA_QUERY_B_DIMENSION;
		shape->columns = SPARK_K3_MODEL_MLA_QUERY_A_DIMENSION;
		shape->quantizable = 1u;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_MLA_KV_A:
		shape->rows = SPARK_K3_MODEL_MLA_KV_A_DIMENSION;
		shape->columns = SPARK_K3_MODEL_HIDDEN_DIMENSION;
		shape->quantizable = 1u;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_MLA_KV_A_NORM:
		shape->rows = 1u;
		shape->columns = SPARK_K3_MODEL_MLA_LATENT_DIMENSION;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_MLA_KV_B:
		shape->rows = SPARK_K3_MODEL_MLA_KV_B_DIMENSION;
		shape->columns = SPARK_K3_MODEL_MLA_LATENT_DIMENSION;
		shape->quantizable = 1u;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_MLA_HEAD_GATE:
		shape->rows = SPARK_K3_MODEL_MLA_HEAD_COUNT;
		shape->columns = SPARK_K3_MODEL_HIDDEN_DIMENSION;
		shape->quantizable = 1u;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_MLA_OUTPUT:
		shape->rows = SPARK_K3_MODEL_HIDDEN_DIMENSION;
		shape->columns = SPARK_K3_MODEL_MLA_ATTENTION_PROJECTION_DIMENSION;
		shape->quantizable = 1u;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_MOE_ROUTER:
		shape->rows = SPARK_K3_MODEL_MOE_EXPERT_COUNT;
		shape->columns = SPARK_K3_MODEL_HIDDEN_DIMENSION;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_MOE_ROUTER_BIAS:
		shape->rows = 1u;
		shape->columns = SPARK_K3_MODEL_MOE_EXPERT_COUNT;
		shape->natural_format = SPARK_K3_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_F32;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_MOE_EXPERT_GATE:
	case SPARK_K3_STAGEPACK_TENSOR_MOE_EXPERT_UP:
		shape->rows = SPARK_K3_MODEL_MOE_EXPERT_COUNT * SPARK_K3_MODEL_MOE_INTERMEDIATE_DIMENSION;
		shape->columns = SPARK_K3_MODEL_HIDDEN_DIMENSION;
		shape->quantizable = 1u;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_MOE_EXPERT_DOWN:
		shape->rows = SPARK_K3_MODEL_MOE_EXPERT_COUNT * SPARK_K3_MODEL_HIDDEN_DIMENSION;
		shape->columns = SPARK_K3_MODEL_MOE_INTERMEDIATE_DIMENSION;
		shape->quantizable = 1u;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_MOE_SHARED_GATE:
	case SPARK_K3_STAGEPACK_TENSOR_MOE_SHARED_UP:
		shape->rows = 0u;
		shape->columns = SPARK_K3_MODEL_HIDDEN_DIMENSION;
		shape->quantizable = 1u;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_MOE_SHARED_DOWN:
		shape->rows = SPARK_K3_MODEL_HIDDEN_DIMENSION;
		shape->columns = 0u;
		shape->quantizable = 1u;
		return(0);
	default:
		return(-1);
	}
}

/*
 * Shared-expert and dense-mlp widths depend on the layer: a dense layer runs
 * the dense intermediate width, a routed layer runs one shared expert of moe
 * intermediate width. Kinds with a zero extent above are resolved here.
 */
static inline uint32_t SparkK3StagePackSharedIntermediate(uint32_t layer_index)
{
	if ( layer_index < SPARK_K3_MODEL_FIRST_ROUTED_LAYER )
		return(SPARK_K3_MODEL_DENSE_INTERMEDIATE_DIMENSION);
	return(SPARK_K3_MODEL_MOE_INTERMEDIATE_DIMENSION * SPARK_K3_MODEL_MOE_SHARED_EXPERT_COUNT);
}

static inline int32_t SparkK3StagePackResolvedShape(uint32_t tensor_kind,uint32_t layer_index,SparkK3StagePackTensorShape *shape)
{
	if ( SparkK3StagePackTensorShapeOf(tensor_kind,shape) < 0 )
		return(-1);
	if ( tensor_kind == SPARK_K3_STAGEPACK_TENSOR_MOE_SHARED_GATE || tensor_kind == SPARK_K3_STAGEPACK_TENSOR_MOE_SHARED_UP )
		shape->rows = SparkK3StagePackSharedIntermediate(layer_index);
	if ( tensor_kind == SPARK_K3_STAGEPACK_TENSOR_MOE_SHARED_DOWN )
		shape->columns = SparkK3StagePackSharedIntermediate(layer_index);
	if ( shape->rows == 0u || shape->columns == 0u )
		return(-2);
	return(0);
}
