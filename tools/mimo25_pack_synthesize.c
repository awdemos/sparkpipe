/* Large stage packs exceed 2 GB: 64-bit file offsets are required. */
#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../modules/mimo25_resident_decode_stage/source/spark_mimo25_stagepack_format.h"

/*
 * Synthetic MiMo-V2.5 stage pack builder for loader and extent testing.
 * usage: mimo25_pack_synthesize <output> <first_layer> <layer_count> [sparse]
 *
 * The variant comes from the -include'd model header, the geometry from
 * the format's expected-count arithmetic. Dense mode writes a
 * deterministic byte pattern per tensor; sparse mode seeks the payload
 * extents as filesystem holes and writes one real byte at the end, so a
 * whole-stack Pro pack costs metadata, not disk.
 */

static void SparkMimo25SynthesizeFillEntry(SparkMimo25StagePackEntry *entry, uint32_t tensor_kind, uint32_t layer_index, const SparkMimo25StagePackTensorShape *shape, uint64_t *cursor)
{
	uint64_t payload_bytes = SparkMimo25StagePackPayloadBytes(shape->weight_format,shape->rows,shape->columns);
	uint64_t scale_bytes = SparkMimo25StagePackScaleBytes(shape->weight_format,shape->rows,shape->columns);
	entry->tensor_kind = tensor_kind;
	entry->layer_index = layer_index;
	entry->weight_format = shape->weight_format;
	entry->rows = shape->rows;
	entry->columns = shape->columns;
	entry->reserved0 = 0u;
	entry->payload_offset = *cursor;
	entry->scale_offset = scale_bytes != 0u ? *cursor + payload_bytes : 0u;
	*cursor += payload_bytes + scale_bytes;
}

static uint32_t SparkMimo25SynthesizeLayerEntries(SparkMimo25StagePackEntry *entries, uint32_t layer_index, uint64_t *cursor)
{
	SparkMimo25StagePackTensorShape shape;
	uint32_t kind,count = 0u;
	for (kind = 0; kind < SPARK_MIMO25_STAGEPACK_TENSOR_KIND_COUNT; kind++)
	{
		if ( kind >= SPARK_MIMO25_STAGEPACK_TENSOR_EMBEDDING )
			continue;
		if ( SparkMimo25StagePackResolvedShape(kind,layer_index,0u,&shape) < 0 )
			continue;
		SparkMimo25SynthesizeFillEntry(&entries[count++],kind,layer_index,&shape,cursor);
	}
	return(count);
}

static uint32_t SparkMimo25SynthesizeGlobalEntries(SparkMimo25StagePackEntry *entries, uint32_t first_layer_index, uint32_t layer_count, uint64_t *cursor)
{
	SparkMimo25StagePackTensorShape shape;
	uint32_t count = 0u,owns_head = first_layer_index + layer_count == SPARK_MIMO25_MODEL_LAYER_COUNT ? 1u : 0u,mtp;
	if ( first_layer_index == 0u || owns_head != 0u )
	{
		SparkMimo25StagePackResolvedShape(SPARK_MIMO25_STAGEPACK_TENSOR_EMBEDDING,0u,1u,&shape);
		SparkMimo25SynthesizeFillEntry(&entries[count++],SPARK_MIMO25_STAGEPACK_TENSOR_EMBEDDING,SPARK_MIMO25_STAGEPACK_GLOBAL_LAYER,&shape,cursor);
	}
	if ( owns_head != 0u )
	{
		SparkMimo25StagePackResolvedShape(SPARK_MIMO25_STAGEPACK_TENSOR_FINAL_NORM,0u,1u,&shape);
		SparkMimo25SynthesizeFillEntry(&entries[count++],SPARK_MIMO25_STAGEPACK_TENSOR_FINAL_NORM,SPARK_MIMO25_STAGEPACK_GLOBAL_LAYER,&shape,cursor);
		SparkMimo25StagePackResolvedShape(SPARK_MIMO25_STAGEPACK_TENSOR_LM_HEAD,0u,1u,&shape);
		SparkMimo25SynthesizeFillEntry(&entries[count++],SPARK_MIMO25_STAGEPACK_TENSOR_LM_HEAD,SPARK_MIMO25_STAGEPACK_GLOBAL_LAYER,&shape,cursor);
		for (mtp = 0; mtp < SPARK_MIMO25_MODEL_MTP_LAYER_COUNT; mtp++)
			count += SparkMimo25SynthesizeLayerEntries(&entries[count],SPARK_MIMO25_STAGEPACK_MTP_LAYER_BASE + mtp,cursor);
	}
	return(count);
}

static int32_t SparkMimo25SynthesizeWritePayloads(FILE *file, const SparkMimo25StagePackEntry *entries, uint32_t tensor_count, uint64_t file_bytes, uint32_t sparse)
{
	uint8_t chunk[65536];
	uint64_t remaining,step;
	uint32_t index,byte;
	if ( sparse != 0u )
	{
		if ( fseeko(file,(off_t)(file_bytes - 1u),SEEK_SET) != 0 )
			return(-1);
		chunk[0] = 0x5au;
		return(fwrite(chunk,1u,1u,file) == 1u ? 0 : -1);
	}
	for (index = 0; index < tensor_count; index++)
	{
		for (byte = 0; byte < sizeof(chunk); byte++)
			chunk[byte] = (uint8_t)(entries[index].tensor_kind * 37u + entries[index].layer_index * 11u + byte);
		if ( fseeko(file,(off_t)entries[index].payload_offset,SEEK_SET) != 0 )
			return(-1);
		remaining = SparkMimo25StagePackPayloadBytes(entries[index].weight_format,entries[index].rows,entries[index].columns) + SparkMimo25StagePackScaleBytes(entries[index].weight_format,entries[index].rows,entries[index].columns);
		while ( remaining != 0u )
		{
			step = remaining < sizeof(chunk) ? remaining : sizeof(chunk);
			if ( fwrite(chunk,1u,(size_t)step,file) != (size_t)step )
				return(-1);
			remaining -= step;
		}
	}
	return(0);
}

static int32_t SparkMimo25SynthesizeParseArguments(int argc, char **argv, uint32_t *first_layer, uint32_t *layer_count, uint32_t *sparse)
{
	if ( argc < 4 )
	{
		fprintf(stderr,"usage: %s <output> <first_layer> <layer_count> [sparse]\n",argv[0]);
		return(-1);
	}
	*first_layer = (uint32_t)strtoul(argv[2],0,10);
	*layer_count = (uint32_t)strtoul(argv[3],0,10);
	*sparse = argc > 4 && strcmp(argv[4],"sparse") == 0 ? 1u : 0u;
	if ( *first_layer + *layer_count > SPARK_MIMO25_MODEL_LAYER_COUNT || *layer_count == 0u )
	{
		fprintf(stderr,"invalid slice %u+%u\n",*first_layer,*layer_count);
		return(-1);
	}
	return(0);
}

int main(int argc, char **argv)
{
	SparkMimo25StagePackHeader header;
	SparkMimo25StagePackEntry *entries;
	FILE *file;
	uint64_t cursor;
	uint32_t first_layer,layer_count,written,layer,sparse;
	if ( SparkMimo25SynthesizeParseArguments(argc,argv,&first_layer,&layer_count,&sparse) != 0 )
		return(1);
	SparkMimo25StagePackExpectedGeometry(&header,first_layer,layer_count);
	entries = (SparkMimo25StagePackEntry *)calloc(header.tensor_count,sizeof(*entries));
	if ( entries == 0 )
		return(1);
	cursor = SPARK_MIMO25_STAGEPACK_HEADER_BYTES + (uint64_t)header.tensor_count * SPARK_MIMO25_STAGEPACK_ENTRY_BYTES;
	written = 0u;
	for (layer = first_layer; layer < first_layer + layer_count; layer++)
		written += SparkMimo25SynthesizeLayerEntries(&entries[written],layer,&cursor);
	written += SparkMimo25SynthesizeGlobalEntries(&entries[written],first_layer,layer_count,&cursor);
	if ( written != header.tensor_count )
	{
		fprintf(stderr,"entry_count_mismatch written=%u expected=%u\n",written,header.tensor_count);
		free(entries);
		return(1);
	}
	header.file_bytes = cursor;
	file = fopen(argv[1],"wb");
	if ( file == 0 )
	{
		free(entries);
		return(1);
	}
	if ( fwrite(&header,sizeof(header),1u,file) != 1u || fwrite(entries,sizeof(*entries),header.tensor_count,file) != header.tensor_count || SparkMimo25SynthesizeWritePayloads(file,entries,header.tensor_count,header.file_bytes,sparse) != 0 )
	{
		fprintf(stderr,"write_failed\n");
		fclose(file);
		free(entries);
		return(1);
	}
	fclose(file);
	printf("pack %s slice=%u+%u tensors=%u bytes=%llu%s\n",argv[1],first_layer,layer_count,header.tensor_count,(unsigned long long)header.file_bytes,sparse != 0u ? " sparse" : "");
	free(entries);
	return(0);
}
