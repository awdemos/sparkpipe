#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64

#include "sparkpipe/spark_tp_shard.h"

#include <fcntl.h>
#include <unistd.h>

#include <string.h>

uint32_t SparkTpShardNameEndsWith(const char *tensor_name,const char *suffix)
{
	uint32_t name_bytes,suffix_bytes;
	if (tensor_name == 0 || suffix == 0)
		return 0u;
	name_bytes = (uint32_t)strlen(tensor_name);
	suffix_bytes = (uint32_t)strlen(suffix);
	if (suffix_bytes > name_bytes)
		return 0u;
	return memcmp(tensor_name + (name_bytes - suffix_bytes),suffix,suffix_bytes) == 0 ? 1u : 0u;
}

static SparkStatus SparkTpShardValidate(const SparkGlm52StagePackTensorSpec *spec,const SparkTpShapeDescriptor *shape,const SparkTpModelGeometry *geometry)
{
	if (spec == 0 || shape == 0 || geometry == 0 ||
		shape->abi_version != SPARK_TP_SHARD_ABI_VERSION ||
		geometry->abi_version != SPARK_TP_SHARD_ABI_VERSION)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (shape->tp_degree != 1u && shape->tp_degree != 2u &&
		shape->tp_degree != 4u && shape->tp_degree != 8u &&
		shape->tp_degree != 16u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (shape->tp_rank >= shape->tp_degree)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (geometry->head_count == 0u ||
		geometry->head_count % shape->tp_degree != 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkTpShardSplitDimension(const SparkGlm52StagePackTensorSpec *spec,const SparkTpShapeDescriptor *shape,uint32_t split_dimension,uint64_t block_elements,SparkTpShardView *view_out)
{
	uint64_t dimension_elements,block_count,blocks_per_rank,other_bytes;
	uint32_t dimension_index;
	dimension_elements = spec->shape[split_dimension];
	if (block_elements == 0u || dimension_elements % block_elements != 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	block_count = dimension_elements / block_elements;
	if (block_count % shape->tp_degree != 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	blocks_per_rank = block_count / shape->tp_degree;
	view_out->split_dimension = split_dimension;
	view_out->element_offset = (uint64_t)shape->tp_rank * blocks_per_rank * block_elements;
	view_out->element_extent = blocks_per_rank * block_elements;
	other_bytes = spec->bytes_per_element;
	for (dimension_index = 0u; dimension_index < spec->rank; ++dimension_index)
	{
		if (dimension_index == split_dimension)
			continue;
		other_bytes *= spec->shape[dimension_index];
	}
	view_out->shard_bytes = other_bytes * view_out->element_extent;
	return SPARK_STATUS_OK;
}

SparkStatus SparkTpShardComputeView(
	SparkTpShardClassifier classifier,
	SparkTpShardHeadBlock head_block,
	const SparkGlm52StagePackTensorSpec *spec,
	const SparkTpShapeDescriptor *shape,
	const SparkTpModelGeometry *geometry,
	SparkTpShardView *view_out)
{
	SparkTpShardClass shard_class;
	SparkStatus status;
	uint64_t full_bytes;
	uint32_t dimension_index;
	if (view_out == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	status = SparkTpShardValidate(spec,shape,geometry);
	if (status != SPARK_STATUS_OK)
		return status;
	shard_class = classifier(spec->tensor_name);
	if (shard_class == SPARK_TP_SHARD_CLASS_CONCAT_OUTPUT)
		return SPARK_STATUS_UNSUPPORTED;
	memset(view_out,0,sizeof(*view_out));
	view_out->abi_version = SPARK_TP_SHARD_ABI_VERSION;
	view_out->shard_class = (uint32_t)shard_class;
	full_bytes = spec->bytes_per_element;
	for (dimension_index = 0u; dimension_index < spec->rank; ++dimension_index)
		full_bytes *= spec->shape[dimension_index];
	// Degree one is a whole-tensor view for every class, including unknown, so
	// existing single-shape packs keep loading without touching this module's
	// classification. At any higher degree an unknown tensor fails closed.
	if (shape->tp_degree == 1u || shard_class == SPARK_TP_SHARD_CLASS_REPLICATED)
	{
		if (shape->tp_degree != 1u && shard_class == SPARK_TP_SHARD_CLASS_UNKNOWN)
			return SPARK_STATUS_VALIDATION_FAILED;
		view_out->split_dimension = 0u;
		view_out->element_offset = 0u;
		view_out->element_extent = spec->shape[0];
		view_out->shard_bytes = full_bytes;
		return SPARK_STATUS_OK;
	}
	if (shard_class == SPARK_TP_SHARD_CLASS_UNKNOWN)
		return SPARK_STATUS_VALIDATION_FAILED;
	if (spec->rank < 2u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (shard_class == SPARK_TP_SHARD_CLASS_OUTPUT_DIM_HEADS)
	{
		uint64_t block = head_block(spec->tensor_name,geometry);
		return SparkTpShardSplitDimension(spec,shape,0u,block,view_out);
	}
	if (shard_class == SPARK_TP_SHARD_CLASS_INPUT_DIM_HEADS)
		return SparkTpShardSplitDimension(spec,shape,1u,geometry->o_proj_head_block,view_out);
	if (shard_class == SPARK_TP_SHARD_CLASS_OUTPUT_DIM)
		return SparkTpShardSplitDimension(spec,shape,0u,1u,view_out);
	return SparkTpShardSplitDimension(spec,shape,1u,1u,view_out);
}


uint64_t SparkTpShardGeometryHash(
	const SparkGlm52StagePackTensorSpec *spec,
	const SparkTpShapeDescriptor *shape,
	const SparkTpShardView *view)
{
	uint64_t hash = 1469598103934665603u;
	uint32_t dimension_index;
	if (spec == 0 || shape == 0 || view == 0)
		return 0u;
	hash = SparkHashBytes(hash,spec->tensor_name,(uint32_t)strlen(spec->tensor_name));
	hash = SparkHashBytes(hash,&spec->rank,sizeof(spec->rank));
	for (dimension_index = 0u; dimension_index < spec->rank; ++dimension_index)
		hash = SparkHashBytes(hash,&spec->shape[dimension_index],sizeof(spec->shape[0]));
	hash = SparkHashBytes(hash,&shape->tp_degree,sizeof(shape->tp_degree));
	hash = SparkHashBytes(hash,&shape->tp_rank,sizeof(shape->tp_rank));
	hash = SparkHashBytes(hash,&shape->pp_stage_count,sizeof(shape->pp_stage_count));
	hash = SparkHashBytes(hash,&shape->pp_stage_index,sizeof(shape->pp_stage_index));
	hash = SparkHashBytes(hash,&view->shard_class,sizeof(view->shard_class));
	hash = SparkHashBytes(hash,&view->split_dimension,sizeof(view->split_dimension));
	hash = SparkHashBytes(hash,&view->element_offset,sizeof(view->element_offset));
	hash = SparkHashBytes(hash,&view->element_extent,sizeof(view->element_extent));
	return hash;
}

// Split read geometry for a view: outer rows before the split dimension, the
// full-tensor pitch of one outer row, the shard chunk within that row, and the
// chunk's byte offset. A leading-dimension split degenerates to one outer row
// covering the contiguous shard, so the same loop serves every class.
static void SparkTpShardReadGeometry(const SparkGlm52StagePackTensorSpec *spec,const SparkTpShardView *view,uint64_t *outer_rows,uint64_t *row_pitch_bytes,uint64_t *chunk_bytes,uint64_t *chunk_offset_bytes)
{
	uint64_t inner_bytes = spec->bytes_per_element;
	uint64_t outer = 1u;
	uint32_t dimension_index;
	for (dimension_index = view->split_dimension + 1u; dimension_index < spec->rank; ++dimension_index)
		inner_bytes *= spec->shape[dimension_index];
	for (dimension_index = 0u; dimension_index < view->split_dimension; ++dimension_index)
		outer *= spec->shape[dimension_index];
	*outer_rows = outer;
	*row_pitch_bytes = spec->shape[view->split_dimension] * inner_bytes;
	*chunk_bytes = view->element_extent * inner_bytes;
	*chunk_offset_bytes = view->element_offset * inner_bytes;
}

static SparkStatus SparkTpShardReadRegion(const SparkGlm52StagePackTensorRegion *region,uint64_t outer_rows,uint64_t row_pitch_bytes,uint64_t chunk_bytes,uint64_t chunk_offset_bytes,uint8_t *destination)
{
	int file_descriptor;
	uint64_t row_index;
	file_descriptor = open(region->file_path,O_RDONLY);
	if (file_descriptor < 0)
		return SPARK_STATUS_IO_ERROR;
	for (row_index = 0u; row_index < outer_rows; ++row_index)
	{
		off_t read_offset = (off_t)(region->file_offset +
			row_index * row_pitch_bytes + chunk_offset_bytes);
		uint64_t read_bytes = 0u;
		while (read_bytes < chunk_bytes)
		{
			ssize_t got = pread(file_descriptor,
				destination + row_index * chunk_bytes + read_bytes,
				(size_t)(chunk_bytes - read_bytes),
				read_offset + (off_t)read_bytes);
			if (got <= 0)
			{
				close(file_descriptor);
				return SPARK_STATUS_IO_ERROR;
			}
			read_bytes += (uint64_t)got;
		}
	}
	close(file_descriptor);
	return SPARK_STATUS_OK;
}

SparkStatus SparkTpShardReadTensor(
	SparkTpShardClassifier classifier,
	SparkTpShardHeadBlock head_block,
	const char *stagepack_root,
	const SparkGlm52StagePackTensorSpec *spec,
	const SparkTpShapeDescriptor *shape,
	const SparkTpModelGeometry *geometry,
	void *destination,
	uint64_t destination_bytes,
	SparkTpShardView *view_out)
{
	SparkTpShardView view;
	SparkGlm52StagePackTensorRegion region;
	uint64_t outer_rows,row_pitch_bytes,chunk_bytes,chunk_offset_bytes;
	SparkStatus status;
	if (destination == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	status = SparkTpShardComputeView(classifier,head_block,spec,shape,geometry,&view);
	if (status != SPARK_STATUS_OK)
		return status;
	if (destination_bytes != view.shard_bytes)
		return SPARK_STATUS_INVALID_ARGUMENT;
	status = SparkGlm52StagePackResolveTensor(stagepack_root,spec,&region);
	if (status != SPARK_STATUS_OK)
		return status;
	SparkTpShardReadGeometry(spec,&view,&outer_rows,&row_pitch_bytes,&chunk_bytes,&chunk_offset_bytes);
	status = SparkTpShardReadRegion(&region,outer_rows,row_pitch_bytes,chunk_bytes,chunk_offset_bytes,(uint8_t *)destination);
	if (status != SPARK_STATUS_OK)
		return status;
	if (view_out != 0)
		*view_out = view;
	return SPARK_STATUS_OK;
}
