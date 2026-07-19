#ifndef SPARKPIPE_SPARK_MIMO25_STAGEPACK_FORMAT_H
#define SPARKPIPE_SPARK_MIMO25_STAGEPACK_FORMAT_H

#include <stdint.h>
#include <string.h>

/*
 * MiMo-V2.5 stage pack: one file per pipeline stage carrying the layer
 * slice [first_layer, first_layer + layer_count) plus the position-derived
 * globals. Both variants share this header; the model header included
 * ahead of it (shared guard) sets every dimension. The layout mirrors the
 * dsv4 pack byte for byte at the container level - same 64-byte header,
 * same 40-byte entries, payload immediately followed by its scales - so
 * the family tooling carries over.
 *
 * Formats: BF16 and F32 payloads have no scales. FP8_E4M3_B128 is one
 * e4m3 byte per element with an F32 scale_inv per [128,128] 2-D block,
 * exactly the checkpoint layout (dequant multiplies); every fp8 dimension
 * in both released checkpoints divides 128, asserted by the shape table.
 *
 * The sink bias exists on SWA layers only and is stored F32 (the
 * converter widens the checkpoint's bf16). MTP draft layers address as
 * MTP_LAYER_BASE + i for i in [0, MTP_LAYER_COUNT) and carry the SWA
 * branch shapes, a dense MLP at the full intermediate size, and the four
 * MTP-only tensors; they ride only in head-stage packs.
 */

#define SPARK_MIMO25_STAGEPACK_MAGIC 0x35324F4Du
#define SPARK_MIMO25_STAGEPACK_FORMAT_VERSION 1u
#define SPARK_MIMO25_STAGEPACK_HEADER_BYTES 64u
#define SPARK_MIMO25_STAGEPACK_ENTRY_BYTES 40u
#define SPARK_MIMO25_STAGEPACK_GLOBAL_LAYER UINT32_MAX
#define SPARK_MIMO25_STAGEPACK_MTP_LAYER_BASE 0xFFFFFF00u

#define SPARK_MIMO25_STAGEPACK_FORMAT_BF16 0u
#define SPARK_MIMO25_STAGEPACK_FORMAT_F32 1u
#define SPARK_MIMO25_STAGEPACK_FORMAT_FP8_E4M3_B128 2u

#define SPARK_MIMO25_STAGEPACK_TENSOR_ATTN_QKV 0u
#define SPARK_MIMO25_STAGEPACK_TENSOR_ATTN_O 1u
#define SPARK_MIMO25_STAGEPACK_TENSOR_ATTN_SINK 2u
#define SPARK_MIMO25_STAGEPACK_TENSOR_ATTN_NORM 3u
#define SPARK_MIMO25_STAGEPACK_TENSOR_FFN_NORM 4u
#define SPARK_MIMO25_STAGEPACK_TENSOR_DENSE_W1 5u
#define SPARK_MIMO25_STAGEPACK_TENSOR_DENSE_W2 6u
#define SPARK_MIMO25_STAGEPACK_TENSOR_DENSE_W3 7u
#define SPARK_MIMO25_STAGEPACK_TENSOR_GATE_WEIGHT 8u
#define SPARK_MIMO25_STAGEPACK_TENSOR_GATE_BIAS 9u
#define SPARK_MIMO25_STAGEPACK_TENSOR_EXPERTS_W1 10u
#define SPARK_MIMO25_STAGEPACK_TENSOR_EXPERTS_W2 11u
#define SPARK_MIMO25_STAGEPACK_TENSOR_EXPERTS_W3 12u
#define SPARK_MIMO25_STAGEPACK_TENSOR_MTP_EH_PROJ 13u
#define SPARK_MIMO25_STAGEPACK_TENSOR_MTP_ENORM 14u
#define SPARK_MIMO25_STAGEPACK_TENSOR_MTP_HNORM 15u
#define SPARK_MIMO25_STAGEPACK_TENSOR_MTP_FINAL_NORM 16u
#define SPARK_MIMO25_STAGEPACK_TENSOR_EMBEDDING 17u
#define SPARK_MIMO25_STAGEPACK_TENSOR_FINAL_NORM 18u
#define SPARK_MIMO25_STAGEPACK_TENSOR_LM_HEAD 19u
#define SPARK_MIMO25_STAGEPACK_TENSOR_KIND_COUNT 20u

typedef struct SparkMimo25StagePackHeader
{
	uint32_t magic;
	uint32_t format_version;
	uint32_t header_bytes;
	uint32_t directory_entry_bytes;
	uint32_t tensor_count;
	uint32_t first_layer_index;
	uint32_t layer_count;
	uint32_t total_layer_count;
	uint32_t hidden_dimension;
	uint32_t vocab_count;
	uint32_t routed_expert_count;
	uint32_t mtp_layer_count;
	uint64_t directory_offset;
	uint64_t file_bytes;
} SparkMimo25StagePackHeader;

typedef struct SparkMimo25StagePackEntry
{
	uint32_t tensor_kind;
	uint32_t layer_index;
	uint32_t weight_format;
	uint32_t rows;
	uint32_t columns;
	uint32_t reserved0;
	uint64_t payload_offset;
	uint64_t scale_offset;
} SparkMimo25StagePackEntry;

typedef struct SparkMimo25StagePackTensorShape
{
	uint32_t weight_format;
	uint32_t rows;
	uint32_t columns;
} SparkMimo25StagePackTensorShape;

static inline uint32_t SparkMimo25StagePackLayerIsMtp(uint32_t layer_index)
{
	return(layer_index >= SPARK_MIMO25_STAGEPACK_MTP_LAYER_BASE && layer_index < SPARK_MIMO25_STAGEPACK_MTP_LAYER_BASE + SPARK_MIMO25_MODEL_MTP_LAYER_COUNT ? 1u : 0u);
}

// MTP layers take the SWA branch shapes and a dense MLP; real layers read
// the model tables.
static inline uint32_t SparkMimo25StagePackLayerKind(uint32_t layer_index)
{
	return(SparkMimo25StagePackLayerIsMtp(layer_index) != 0u ? SPARK_MIMO25_MODEL_LAYER_KIND_SWA : SparkMimo25ModelLayerKind(layer_index));
}

static inline uint32_t SparkMimo25StagePackLayerHasMoe(uint32_t layer_index)
{
	return(SparkMimo25StagePackLayerIsMtp(layer_index) != 0u ? 0u : SparkMimo25ModelLayerIsMoe(layer_index));
}

static inline uint64_t SparkMimo25StagePackPayloadBytes(uint32_t weight_format, uint32_t rows, uint32_t columns)
{
	uint64_t elements = (uint64_t)rows * columns;
	if ( weight_format == SPARK_MIMO25_STAGEPACK_FORMAT_BF16 )
		return(elements * 2u);
	if ( weight_format == SPARK_MIMO25_STAGEPACK_FORMAT_F32 )
		return(elements * 4u);
	return(elements);
}

static inline uint64_t SparkMimo25StagePackScaleBytes(uint32_t weight_format, uint32_t rows, uint32_t columns)
{
	if ( weight_format != SPARK_MIMO25_STAGEPACK_FORMAT_FP8_E4M3_B128 )
		return(0u);
	return((uint64_t)(rows / SPARK_MIMO25_MODEL_FP8_SCALE_BLOCK) * (columns / SPARK_MIMO25_MODEL_FP8_SCALE_BLOCK) * 4u);
}

static inline void SparkMimo25StagePackShapeSet(SparkMimo25StagePackTensorShape *shape, uint32_t weight_format, uint32_t rows, uint32_t columns)
{
	shape->weight_format = weight_format;
	shape->rows = rows;
	shape->columns = columns;
}

/*
 * The single shape authority: -1 means the kind does not exist at that
 * layer (or as a global), and the loader treats presence there as a
 * refused pack. The qkv row count follows the layer's branch; MTP layers
 * are SWA-shaped with the dense MLP.
 */
static inline int32_t SparkMimo25StagePackGlobalShape(uint32_t tensor_kind, SparkMimo25StagePackTensorShape *shape)
{
	if ( tensor_kind == SPARK_MIMO25_STAGEPACK_TENSOR_EMBEDDING || tensor_kind == SPARK_MIMO25_STAGEPACK_TENSOR_LM_HEAD )
		SparkMimo25StagePackShapeSet(shape,SPARK_MIMO25_STAGEPACK_FORMAT_BF16,SPARK_MIMO25_MODEL_VOCAB_COUNT,SPARK_MIMO25_MODEL_HIDDEN_DIMENSION);
	else if ( tensor_kind == SPARK_MIMO25_STAGEPACK_TENSOR_FINAL_NORM )
		SparkMimo25StagePackShapeSet(shape,SPARK_MIMO25_STAGEPACK_FORMAT_BF16,1u,SPARK_MIMO25_MODEL_HIDDEN_DIMENSION);
	else
		return(-1);
	return(0);
}

static inline int32_t SparkMimo25StagePackMoeShape(uint32_t tensor_kind, uint32_t moe, SparkMimo25StagePackTensorShape *shape)
{
	uint32_t hidden = SPARK_MIMO25_MODEL_HIDDEN_DIMENSION;
	if ( moe == 0u )
		return(-1);
	switch ( tensor_kind )
	{
	case SPARK_MIMO25_STAGEPACK_TENSOR_GATE_WEIGHT:
		SparkMimo25StagePackShapeSet(shape,SPARK_MIMO25_STAGEPACK_FORMAT_F32,SPARK_MIMO25_MODEL_ROUTED_EXPERT_COUNT,hidden);
		return(0);
	case SPARK_MIMO25_STAGEPACK_TENSOR_GATE_BIAS:
		SparkMimo25StagePackShapeSet(shape,SPARK_MIMO25_STAGEPACK_FORMAT_F32,1u,SPARK_MIMO25_MODEL_ROUTED_EXPERT_COUNT);
		return(0);
	case SPARK_MIMO25_STAGEPACK_TENSOR_EXPERTS_W1:
	case SPARK_MIMO25_STAGEPACK_TENSOR_EXPERTS_W3:
		SparkMimo25StagePackShapeSet(shape,SPARK_MIMO25_STAGEPACK_FORMAT_FP8_E4M3_B128,SPARK_MIMO25_MODEL_ROUTED_EXPERT_COUNT * SPARK_MIMO25_MODEL_EXPERT_INTERMEDIATE_DIMENSION,hidden);
		return(0);
	case SPARK_MIMO25_STAGEPACK_TENSOR_EXPERTS_W2:
		SparkMimo25StagePackShapeSet(shape,SPARK_MIMO25_STAGEPACK_FORMAT_FP8_E4M3_B128,SPARK_MIMO25_MODEL_ROUTED_EXPERT_COUNT * SPARK_MIMO25_MODEL_HIDDEN_DIMENSION,SPARK_MIMO25_MODEL_EXPERT_INTERMEDIATE_DIMENSION);
		return(0);
	default:
		return(-1);
	}
}

static inline int32_t SparkMimo25StagePackResolvedShape(uint32_t tensor_kind, uint32_t layer_index, uint32_t is_global, SparkMimo25StagePackTensorShape *shape)
{
	uint32_t hidden = SPARK_MIMO25_MODEL_HIDDEN_DIMENSION,mtp = SparkMimo25StagePackLayerIsMtp(layer_index);
	uint32_t kind = SparkMimo25StagePackLayerKind(layer_index),moe = SparkMimo25StagePackLayerHasMoe(layer_index);
	if ( is_global != 0u )
		return(SparkMimo25StagePackGlobalShape(tensor_kind,shape));
	if ( layer_index >= SPARK_MIMO25_MODEL_LAYER_COUNT && mtp == 0u )
		return(-1);
	if ( tensor_kind >= SPARK_MIMO25_STAGEPACK_TENSOR_GATE_WEIGHT && tensor_kind <= SPARK_MIMO25_STAGEPACK_TENSOR_EXPERTS_W3 )
		return(SparkMimo25StagePackMoeShape(tensor_kind,moe,shape));
	switch ( tensor_kind )
	{
	case SPARK_MIMO25_STAGEPACK_TENSOR_ATTN_QKV:
		SparkMimo25StagePackShapeSet(shape,SPARK_MIMO25_STAGEPACK_FORMAT_FP8_E4M3_B128,kind == SPARK_MIMO25_MODEL_LAYER_KIND_SWA ? SPARK_MIMO25_MODEL_SWA_QKV_DIMENSION : SPARK_MIMO25_MODEL_FULL_QKV_DIMENSION,hidden);
		return(0);
	case SPARK_MIMO25_STAGEPACK_TENSOR_ATTN_O:
		SparkMimo25StagePackShapeSet(shape,SPARK_MIMO25_STAGEPACK_FORMAT_BF16,hidden,SPARK_MIMO25_MODEL_O_INPUT_DIMENSION);
		return(0);
	case SPARK_MIMO25_STAGEPACK_TENSOR_ATTN_SINK:
		if ( kind != SPARK_MIMO25_MODEL_LAYER_KIND_SWA )
			return(-1);
		SparkMimo25StagePackShapeSet(shape,SPARK_MIMO25_STAGEPACK_FORMAT_F32,1u,SPARK_MIMO25_MODEL_ATTN_HEAD_COUNT);
		return(0);
	case SPARK_MIMO25_STAGEPACK_TENSOR_ATTN_NORM:
	case SPARK_MIMO25_STAGEPACK_TENSOR_FFN_NORM:
		SparkMimo25StagePackShapeSet(shape,SPARK_MIMO25_STAGEPACK_FORMAT_BF16,1u,hidden);
		return(0);
	case SPARK_MIMO25_STAGEPACK_TENSOR_DENSE_W1:
	case SPARK_MIMO25_STAGEPACK_TENSOR_DENSE_W3:
	case SPARK_MIMO25_STAGEPACK_TENSOR_DENSE_W2:
		if ( moe != 0u )
			return(-1);
		if ( tensor_kind == SPARK_MIMO25_STAGEPACK_TENSOR_DENSE_W2 )
			SparkMimo25StagePackShapeSet(shape,SPARK_MIMO25_STAGEPACK_FORMAT_FP8_E4M3_B128,hidden,SPARK_MIMO25_MODEL_DENSE_INTERMEDIATE_DIMENSION);
		else
			SparkMimo25StagePackShapeSet(shape,SPARK_MIMO25_STAGEPACK_FORMAT_FP8_E4M3_B128,SPARK_MIMO25_MODEL_DENSE_INTERMEDIATE_DIMENSION,hidden);
		return(0);
	case SPARK_MIMO25_STAGEPACK_TENSOR_MTP_EH_PROJ:
		if ( mtp == 0u )
			return(-1);
		SparkMimo25StagePackShapeSet(shape,SPARK_MIMO25_STAGEPACK_FORMAT_BF16,hidden,2u * hidden);
		return(0);
	case SPARK_MIMO25_STAGEPACK_TENSOR_MTP_ENORM:
	case SPARK_MIMO25_STAGEPACK_TENSOR_MTP_HNORM:
	case SPARK_MIMO25_STAGEPACK_TENSOR_MTP_FINAL_NORM:
		if ( mtp == 0u )
			return(-1);
		SparkMimo25StagePackShapeSet(shape,SPARK_MIMO25_STAGEPACK_FORMAT_BF16,1u,hidden);
		return(0);
	default:
		return(-1);
	}
}

static inline uint32_t SparkMimo25StagePackLayerTensorCount(uint32_t layer_index)
{
	uint32_t count = 4u;
	if ( SparkMimo25StagePackLayerKind(layer_index) == SPARK_MIMO25_MODEL_LAYER_KIND_SWA )
		count += 1u;
	if ( SparkMimo25StagePackLayerHasMoe(layer_index) != 0u )
		count += 5u;
	else
		count += 3u;
	if ( SparkMimo25StagePackLayerIsMtp(layer_index) != 0u )
		count += 4u;
	return(count);
}

static inline uint32_t SparkMimo25StagePackExpectedTensorCount(uint32_t first_layer_index, uint32_t layer_count)
{
	uint32_t layer,tensors = 0u;
	for (layer = first_layer_index; layer < first_layer_index + layer_count; layer++)
		tensors += SparkMimo25StagePackLayerTensorCount(layer);
	if ( first_layer_index == 0u )
		tensors += 1u;
	if ( first_layer_index + layer_count == SPARK_MIMO25_MODEL_LAYER_COUNT )
	{
		tensors += 2u + (first_layer_index != 0u ? 1u : 0u);
		for (layer = 0; layer < SPARK_MIMO25_MODEL_MTP_LAYER_COUNT; layer++)
			tensors += SparkMimo25StagePackLayerTensorCount(SPARK_MIMO25_STAGEPACK_MTP_LAYER_BASE + layer);
	}
	return(tensors);
}

static inline void SparkMimo25StagePackExpectedGeometry(SparkMimo25StagePackHeader *header, uint32_t first_layer_index, uint32_t layer_count)
{
	memset(header,0,sizeof(*header));
	header->magic = SPARK_MIMO25_STAGEPACK_MAGIC;
	header->format_version = SPARK_MIMO25_STAGEPACK_FORMAT_VERSION;
	header->header_bytes = SPARK_MIMO25_STAGEPACK_HEADER_BYTES;
	header->directory_entry_bytes = SPARK_MIMO25_STAGEPACK_ENTRY_BYTES;
	header->tensor_count = SparkMimo25StagePackExpectedTensorCount(first_layer_index,layer_count);
	header->first_layer_index = first_layer_index;
	header->layer_count = layer_count;
	header->total_layer_count = SPARK_MIMO25_MODEL_LAYER_COUNT;
	header->hidden_dimension = SPARK_MIMO25_MODEL_HIDDEN_DIMENSION;
	header->vocab_count = SPARK_MIMO25_MODEL_VOCAB_COUNT;
	header->routed_expert_count = SPARK_MIMO25_MODEL_ROUTED_EXPERT_COUNT;
	header->mtp_layer_count = SPARK_MIMO25_MODEL_MTP_LAYER_COUNT;
	header->directory_offset = SPARK_MIMO25_STAGEPACK_HEADER_BYTES;
}

static inline int32_t SparkMimo25StagePackCompareGeometry(const SparkMimo25StagePackHeader *header, const SparkMimo25StagePackHeader *expected)
{
	if ( header->magic != expected->magic )
		return(1);
	if ( header->format_version != expected->format_version )
		return(2);
	if ( header->header_bytes != expected->header_bytes || header->directory_entry_bytes != expected->directory_entry_bytes )
		return(3);
	if ( header->tensor_count != expected->tensor_count )
		return(4);
	if ( header->first_layer_index != expected->first_layer_index || header->layer_count != expected->layer_count )
		return(5);
	if ( header->total_layer_count != expected->total_layer_count )
		return(6);
	if ( header->hidden_dimension != expected->hidden_dimension || header->vocab_count != expected->vocab_count )
		return(7);
	if ( header->routed_expert_count != expected->routed_expert_count || header->mtp_layer_count != expected->mtp_layer_count )
		return(8);
	if ( header->directory_offset != expected->directory_offset )
		return(9);
	return(0);
}

static inline const char *SparkMimo25StagePackGeometryFieldName(int32_t compare)
{
	switch ( compare )
	{
	case 1: return("magic");
	case 2: return("format_version");
	case 3: return("layout_bytes");
	case 4: return("tensor_count");
	case 5: return("slice");
	case 6: return("total_layer_count");
	case 7: return("model_dimensions");
	case 8: return("expert_or_mtp_count");
	case 9: return("directory_offset");
	default: return("ok");
	}
}

#endif
