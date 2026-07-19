/* Large stage packs exceed 2 GB: 64-bit file offsets are required. */
#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "spark_k3_stagepack_format.h"

/*
 * Synthetic K3 stage pack writer.
 *
 * Emits every tensor the module will demand, at the geometry the module was
 * compiled for, with reproducible pseudo-random contents. It exercises the
 * loader, the shape table and the layer walk end to end. It says nothing
 * about output quality: these are not K3's weights, they are correctly shaped
 * noise, and the module cannot tell the difference by design.
 *
 * The tensor list is derived from the same shape table the loader validates
 * against, so a shape can never disagree between the two.
 *
 * Dry run reports the pack size without writing it, which at the real
 * geometry is the number worth knowing before allocating a disk.
 */

#define SPARK_K3_SYNTHESIZE_MAX_TENSORS 4096u
#define SPARK_K3_SYNTHESIZE_CHUNK_BYTES (8u * 1024u * 1024u)

typedef struct SparkK3SynthesizeContext
{
	SparkK3StagePackEntry entries[SPARK_K3_SYNTHESIZE_MAX_TENSORS];
	uint32_t entry_count;
	uint64_t payload_cursor;
	uint64_t seed;
} SparkK3SynthesizeContext;

// xorshift64*: a lane of reproducible noise per tensor, seeded from the
// tensor's identity so any tensor regenerates independently of the others.
static uint64_t SparkK3SynthesizeNext(uint64_t *state)
{
	uint64_t value = *state;
	value ^= value >> 12;
	value ^= value << 25;
	value ^= value >> 27;
	*state = value;
	return(value * 2685821657736338717ull);
}

static uint64_t SparkK3SynthesizeTensorSeed(uint64_t seed, uint32_t tensor_kind, uint32_t layer_index)
{
	uint64_t state = seed ^ (0x9e3779b97f4a7c15ull * (uint64_t)(tensor_kind + 1u)) ^ (0xbf58476d1ce4e5b9ull * (uint64_t)(layer_index + 1u));
	if ( state == 0u )
		state = 0x2545f4914f6cdd1dull;
	return(state);
}

static uint64_t SparkK3SynthesizeAlign(uint64_t offset)
{
	uint64_t remainder = offset % SPARK_K3_STAGEPACK_PAYLOAD_ALIGNMENT;
	if ( remainder == 0u )
		return(offset);
	return(offset + (SPARK_K3_STAGEPACK_PAYLOAD_ALIGNMENT - remainder));
}

static int32_t SparkK3SynthesizeAppend(SparkK3SynthesizeContext *context, uint32_t tensor_kind, uint32_t layer_index, uint32_t quantize)
{
	SparkK3StagePackTensorShape shape;
	SparkK3StagePackEntry *entry;
	uint32_t resolve_layer = (layer_index == SPARK_K3_STAGEPACK_GLOBAL_LAYER) ? 0u : layer_index;
	if ( context->entry_count >= SPARK_K3_SYNTHESIZE_MAX_TENSORS )
		return(-1);
	if ( SparkK3StagePackResolvedShape(tensor_kind,resolve_layer,&shape) < 0 )
		return(-2);
	entry = &context->entries[context->entry_count];
	entry->tensor_kind = tensor_kind;
	entry->layer_index = layer_index;
	entry->weight_format = (shape.quantizable != 0u && quantize != 0u) ? SPARK_K3_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_MXFP4_E2M1 : shape.natural_format;
	entry->rows = shape.rows;
	entry->columns = shape.columns;
	entry->scale_group_size = (entry->weight_format == SPARK_K3_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_MXFP4_E2M1) ? SPARK_K3_MODEL_MXFP4_GROUP_SIZE : 0u;
	entry->payload_bytes = SparkK3StagePackPayloadBytes(entry->weight_format,entry->rows,entry->columns);
	entry->scale_bytes = SparkK3StagePackScaleBytes(entry->weight_format,entry->rows,entry->columns);
	entry->payload_offset = SparkK3SynthesizeAlign(context->payload_cursor);
	context->payload_cursor = entry->payload_offset + entry->payload_bytes;
	if ( entry->scale_bytes != 0u )
	{
		entry->scale_offset = SparkK3SynthesizeAlign(context->payload_cursor);
		context->payload_cursor = entry->scale_offset + entry->scale_bytes;
	}
	else
		entry->scale_offset = 0u;
	context->entry_count++;
	return(0);
}

static int32_t SparkK3SynthesizeAppendKdaLayer(SparkK3SynthesizeContext *context, uint32_t layer_index, uint32_t quantize)
{
	static const uint32_t kinds[] = {
		SPARK_K3_STAGEPACK_TENSOR_KDA_QUERY,SPARK_K3_STAGEPACK_TENSOR_KDA_KEY,SPARK_K3_STAGEPACK_TENSOR_KDA_VALUE,
		SPARK_K3_STAGEPACK_TENSOR_KDA_DECAY_LOW,SPARK_K3_STAGEPACK_TENSOR_KDA_DECAY_HIGH,SPARK_K3_STAGEPACK_TENSOR_KDA_BETA,
		SPARK_K3_STAGEPACK_TENSOR_KDA_GATE_LOW,SPARK_K3_STAGEPACK_TENSOR_KDA_GATE_HIGH,SPARK_K3_STAGEPACK_TENSOR_KDA_OUTPUT,
		SPARK_K3_STAGEPACK_TENSOR_KDA_HEAD_NORM };
	uint32_t index;
	int32_t result;
	for (index = 0; index < (sizeof(kinds) / sizeof(kinds[0])); index++)
	{
		result = SparkK3SynthesizeAppend(context,kinds[index],layer_index,quantize);
		if ( result < 0 )
			return(result);
	}
	return(0);
}

static int32_t SparkK3SynthesizeAppendMlaLayer(SparkK3SynthesizeContext *context, uint32_t layer_index, uint32_t quantize)
{
	static const uint32_t kinds[] = {
		SPARK_K3_STAGEPACK_TENSOR_MLA_QUERY_A,SPARK_K3_STAGEPACK_TENSOR_MLA_QUERY_A_NORM,SPARK_K3_STAGEPACK_TENSOR_MLA_QUERY_B,
		SPARK_K3_STAGEPACK_TENSOR_MLA_KV_A,SPARK_K3_STAGEPACK_TENSOR_MLA_KV_A_NORM,SPARK_K3_STAGEPACK_TENSOR_MLA_KV_B,
		SPARK_K3_STAGEPACK_TENSOR_MLA_HEAD_GATE,SPARK_K3_STAGEPACK_TENSOR_MLA_OUTPUT };
	uint32_t index;
	int32_t result;
	for (index = 0; index < (sizeof(kinds) / sizeof(kinds[0])); index++)
	{
		result = SparkK3SynthesizeAppend(context,kinds[index],layer_index,quantize);
		if ( result < 0 )
			return(result);
	}
	return(0);
}

static int32_t SparkK3SynthesizeAppendMlpLayer(SparkK3SynthesizeContext *context, uint32_t layer_index, uint32_t quantize)
{
	static const uint32_t routed_kinds[] = {
		SPARK_K3_STAGEPACK_TENSOR_MOE_ROUTER,SPARK_K3_STAGEPACK_TENSOR_MOE_ROUTER_BIAS,
		SPARK_K3_STAGEPACK_TENSOR_MOE_EXPERT_GATE,SPARK_K3_STAGEPACK_TENSOR_MOE_EXPERT_UP,SPARK_K3_STAGEPACK_TENSOR_MOE_EXPERT_DOWN };
	static const uint32_t shared_kinds[] = {
		SPARK_K3_STAGEPACK_TENSOR_MOE_SHARED_GATE,SPARK_K3_STAGEPACK_TENSOR_MOE_SHARED_UP,SPARK_K3_STAGEPACK_TENSOR_MOE_SHARED_DOWN };
	uint32_t index;
	int32_t result;
	if ( layer_index >= SPARK_K3_MODEL_FIRST_ROUTED_LAYER )
		for (index = 0; index < (sizeof(routed_kinds) / sizeof(routed_kinds[0])); index++)
		{
			result = SparkK3SynthesizeAppend(context,routed_kinds[index],layer_index,quantize);
			if ( result < 0 )
				return(result);
		}
	for (index = 0; index < (sizeof(shared_kinds) / sizeof(shared_kinds[0])); index++)
	{
		result = SparkK3SynthesizeAppend(context,shared_kinds[index],layer_index,quantize);
		if ( result < 0 )
			return(result);
	}
	return(0);
}

static int32_t SparkK3SynthesizeAppendLayer(SparkK3SynthesizeContext *context, uint32_t layer_index, uint32_t quantize)
{
	static const uint32_t site_kinds[] = {
		SPARK_K3_STAGEPACK_TENSOR_ATTNRES_ATTENTION_QUERY,SPARK_K3_STAGEPACK_TENSOR_ATTNRES_ATTENTION_NORM,
		SPARK_K3_STAGEPACK_TENSOR_ATTNRES_MLP_QUERY,SPARK_K3_STAGEPACK_TENSOR_ATTNRES_MLP_NORM,
		SPARK_K3_STAGEPACK_TENSOR_ATTENTION_NORM,SPARK_K3_STAGEPACK_TENSOR_MLP_NORM };
	uint32_t index;
	int32_t result;
	for (index = 0; index < (sizeof(site_kinds) / sizeof(site_kinds[0])); index++)
	{
		result = SparkK3SynthesizeAppend(context,site_kinds[index],layer_index,quantize);
		if ( result < 0 )
			return(result);
	}
	if ( SPARK_K3_MODEL_LAYER_IS_KDA(layer_index) != 0u )
		result = SparkK3SynthesizeAppendKdaLayer(context,layer_index,quantize);
	else
		result = SparkK3SynthesizeAppendMlaLayer(context,layer_index,quantize);
	if ( result < 0 )
		return(result);
	return(SparkK3SynthesizeAppendMlpLayer(context,layer_index,quantize));
}

static int32_t SparkK3SynthesizeBuildDirectory(SparkK3SynthesizeContext *context, uint32_t quantize)
{
	static const uint32_t global_kinds[] = {
		SPARK_K3_STAGEPACK_TENSOR_EMBEDDING,SPARK_K3_STAGEPACK_TENSOR_ATTNRES_FINAL_QUERY,SPARK_K3_STAGEPACK_TENSOR_ATTNRES_FINAL_NORM,
		SPARK_K3_STAGEPACK_TENSOR_FINAL_NORM,SPARK_K3_STAGEPACK_TENSOR_LM_HEAD_RESTRICTED,SPARK_K3_STAGEPACK_TENSOR_RESTRICTED_TOKEN_IDS };
	uint32_t index;
	int32_t result;
	for (index = 0; index < (sizeof(global_kinds) / sizeof(global_kinds[0])); index++)
	{
		result = SparkK3SynthesizeAppend(context,global_kinds[index],SPARK_K3_STAGEPACK_GLOBAL_LAYER,quantize);
		if ( result < 0 )
			return(result);
	}
	for (index = 0; index < SPARK_K3_MODEL_LAYER_COUNT; index++)
	{
		result = SparkK3SynthesizeAppendLayer(context,index,quantize);
		if ( result < 0 )
			return(result);
	}
	return(0);
}

/*
 * Fill one buffer with tensor-appropriate noise. bf16 and f32 weights get
 * small signed values, mxfp4 payload gets random nibble pairs, e8m0 scales sit
 * around unity, and the restricted token id list gets a spread of real vocab
 * ids so the head's argmax maps to something a tokenizer would accept.
 */
static void SparkK3SynthesizeFillPayload(const SparkK3StagePackEntry *entry, uint8_t *buffer, uint64_t bytes, uint64_t element_base, uint64_t *random_state)
{
	uint64_t index;
	uint32_t sample;
	union { float real; uint32_t bits; } value;
	if ( entry->tensor_kind == SPARK_K3_STAGEPACK_TENSOR_RESTRICTED_TOKEN_IDS )
	{
		for (index = 0; index < (bytes / 4u); index++)
			((uint32_t *)buffer)[index] = (uint32_t)(((element_base + index) * (SPARK_K3_MODEL_OUTPUT_VOCAB_COUNT / SPARK_K3_MODEL_RESTRICTED_VOCAB_COUNT)) % SPARK_K3_MODEL_OUTPUT_VOCAB_COUNT);
		return;
	}
	if ( entry->weight_format == SPARK_K3_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_MXFP4_E2M1 )
	{
		for (index = 0; index < bytes; index++)
			buffer[index] = (uint8_t)(SparkK3SynthesizeNext(random_state) & 0x77u);
		return;
	}
	if ( entry->weight_format == SPARK_K3_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_F32 )
	{
		for (index = 0; index < (bytes / 4u); index++)
		{
			sample = (uint32_t)(SparkK3SynthesizeNext(random_state) >> 40);
			((float *)buffer)[index] = ((float)sample / 16777216.0f) - 0.0005f;
		}
		return;
	}
	for (index = 0; index < (bytes / 2u); index++)
	{
		sample = (uint32_t)(SparkK3SynthesizeNext(random_state) >> 40);
		value.real = (((float)sample / 8388608.0f) - 1.0f) * 0.02f;
		((uint16_t *)buffer)[index] = (uint16_t)(value.bits >> 16);
	}
}

// e8m0 scales: exponent 127 is 1.0, so a narrow band around it keeps
// synthetic mxfp4 tensors in the same magnitude range as the bf16 ones.
static void SparkK3SynthesizeFillScale(uint8_t *buffer, uint64_t bytes, uint64_t *random_state)
{
	uint64_t index;
	for (index = 0; index < bytes; index++)
		buffer[index] = (uint8_t)(124u + (SparkK3SynthesizeNext(random_state) % 7u));
}

static int32_t SparkK3SynthesizeWriteRegion(FILE *file, const SparkK3StagePackEntry *entry, uint64_t offset, uint64_t bytes, uint32_t is_scale, uint64_t *random_state, uint8_t *chunk)
{
	uint64_t written,span;
	if ( fseeko(file,(off_t)offset,SEEK_SET) != 0 )
		return(-1);
	for (written = 0; written < bytes; written += span)
	{
		span = bytes - written;
		if ( span > SPARK_K3_SYNTHESIZE_CHUNK_BYTES )
			span = SPARK_K3_SYNTHESIZE_CHUNK_BYTES;
		if ( is_scale != 0u )
			SparkK3SynthesizeFillScale(chunk,span,random_state);
		else
			SparkK3SynthesizeFillPayload(entry,chunk,span,written / 4u,random_state);
		if ( fwrite(chunk,1,(size_t)span,file) != (size_t)span )
			return(-2);
	}
	return(0);
}

static int32_t SparkK3SynthesizeWrite(SparkK3SynthesizeContext *context, const char *path, const SparkK3StagePackHeader *header)
{
	SparkK3StagePackEntry entry;
	uint64_t random_state;
	uint8_t *chunk;
	FILE *file;
	uint32_t index;
	int32_t result = 0;
	file = fopen(path,"wb");
	if ( file == 0 )
	{
		fprintf(stderr,"k3_pack_synthesize open_failed path=%s\n",path);
		return(-1);
	}
	chunk = (uint8_t *)malloc(SPARK_K3_SYNTHESIZE_CHUNK_BYTES);
	if ( chunk == 0 )
	{
		fclose(file);
		return(-2);
	}
	if ( fwrite(header,1,sizeof(*header),file) != sizeof(*header) )
		result = -3;
	if ( result == 0 && fwrite(context->entries,sizeof(SparkK3StagePackEntry),context->entry_count,file) != context->entry_count )
		result = -4;
	for (index = 0; index < context->entry_count && result == 0; index++)
	{
		entry = context->entries[index];
		random_state = SparkK3SynthesizeTensorSeed(context->seed,entry.tensor_kind,entry.layer_index);
		result = SparkK3SynthesizeWriteRegion(file,&entry,entry.payload_offset,entry.payload_bytes,0u,&random_state,chunk);
		if ( result == 0 && entry.scale_bytes != 0u )
			result = SparkK3SynthesizeWriteRegion(file,&entry,entry.scale_offset,entry.scale_bytes,1u,&random_state,chunk);
		if ( result == 0 && (index % 64u) == 0u )
			fprintf(stderr,"k3_pack_synthesize progress tensor=%u/%u bytes=%llu\n",index,context->entry_count,(unsigned long long)entry.payload_offset);
	}
	free(chunk);
	if ( result == 0 && fclose(file) != 0 )
		result = -5;
	else if ( result != 0 )
		fclose(file);
	return(result);
}

/*
 * The directory is built with payload offsets relative to zero because the
 * payload base is not known until the tensor count is. Rebasing once here
 * keeps the append path free of a fixup pass per tensor.
 */
static void SparkK3SynthesizeShiftPayload(SparkK3SynthesizeContext *context, uint64_t payload_offset)
{
	uint32_t index;
	for (index = 0; index < context->entry_count; index++)
	{
		context->entries[index].payload_offset += payload_offset;
		if ( context->entries[index].scale_bytes != 0u )
			context->entries[index].scale_offset += payload_offset;
	}
	context->payload_cursor += payload_offset;
}

static void SparkK3SynthesizeReport(const SparkK3SynthesizeContext *context, const SparkK3StagePackHeader *header)
{
	uint64_t quantized_bytes = 0,dense_bytes = 0;
	uint32_t index;
	for (index = 0; index < context->entry_count; index++)
	{
		if ( context->entries[index].weight_format == SPARK_K3_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_MXFP4_E2M1 )
			quantized_bytes += context->entries[index].payload_bytes + context->entries[index].scale_bytes;
		else
			dense_bytes += context->entries[index].payload_bytes;
	}
	fprintf(stderr,"k3_pack_synthesize tensors=%u mxfp4_bytes=%llu dense_bytes=%llu file_bytes=%llu file_gib=%.1f\n",context->entry_count,(unsigned long long)quantized_bytes,(unsigned long long)dense_bytes,(unsigned long long)header->file_bytes,(double)header->file_bytes / (1024.0 * 1024.0 * 1024.0));
}

int main(int argc, char **argv)
{
	static SparkK3SynthesizeContext context;
	SparkK3StagePackHeader header;
	const char *path = 0;
	uint32_t quantize = 1u,dry_run = 0u;
	int32_t result,argument;
	context.seed = 20260727u;
	for (argument = 1; argument < argc; argument++)
	{
		if ( strcmp(argv[argument],"--output") == 0 && (argument + 1) < argc )
			path = argv[++argument];
		else if ( strcmp(argv[argument],"--seed") == 0 && (argument + 1) < argc )
			context.seed = strtoull(argv[++argument],0,10);
		else if ( strcmp(argv[argument],"--bf16") == 0 )
			quantize = 0u;
		else if ( strcmp(argv[argument],"--dry-run") == 0 )
			dry_run = 1u;
		else
		{
			fprintf(stderr,"usage: k3_pack_synthesize --output PATH [--seed N] [--bf16] [--dry-run]\n");
			return(2);
		}
	}
	if ( path == 0 && dry_run == 0u )
	{
		fprintf(stderr,"usage: k3_pack_synthesize --output PATH [--seed N] [--bf16] [--dry-run]\n");
		return(2);
	}
	context.entry_count = 0u;
	context.payload_cursor = 0u;
	result = SparkK3SynthesizeBuildDirectory(&context,quantize);
	if ( result < 0 )
	{
		fprintf(stderr,"k3_pack_synthesize directory_failed result=%d\n",result);
		return(1);
	}
	SparkK3StagePackExpectedGeometry(&header,0u,SPARK_K3_MODEL_LAYER_COUNT,context.entry_count);
	header.payload_offset = SparkK3SynthesizeAlign((uint64_t)header.header_bytes + ((uint64_t)context.entry_count * header.directory_entry_bytes));
	SparkK3SynthesizeShiftPayload(&context,header.payload_offset);
	header.file_bytes = context.payload_cursor;
	SparkK3SynthesizeReport(&context,&header);
	if ( dry_run != 0u )
		return(0);
	result = SparkK3SynthesizeWrite(&context,path,&header);
	if ( result < 0 )
	{
		fprintf(stderr,"k3_pack_synthesize write_failed result=%d path=%s\n",result,path);
		return(1);
	}
	fprintf(stderr,"k3_pack_synthesize wrote path=%s tensors=%u bytes=%llu\n",path,context.entry_count,(unsigned long long)header.file_bytes);
	return(0);
}
