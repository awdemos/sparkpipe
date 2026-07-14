#include "sparkpipe/spark_glm52_resident_decode_stage_w8lut_moe_plan.h"
#include "sparkpipe/spark_glm52_resident_decode_stage_required_cuda.h"
#include "spark_glm52_resident_decode_stage_pack_io.h"

#include <cuda_runtime_api.h>

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static uint64_t SparkGlm52W8lutMoePlanAlignUp(uint64_t value,uint64_t alignment)
{
	uint64_t remainder;
	if ( alignment == 0u )
		return(0u);
	remainder = (value % alignment);
	if ( remainder == 0u )
		return(value);
	if ( value > (UINT64_MAX - (alignment - remainder)) )
		return(0u);
	return(value + (alignment - remainder));
}

static uint64_t SparkGlm52W8lutMoePlanExpectedRegionBytes(uint32_t region_index)
{
	if ( region_index == SPARK_GLM52_RESIDENT_DECODE_STAGE_W8LUT_MOE_PACK_REGION_W1_CODES )
		return((uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT * (uint64_t)(SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_W1_COMPONENT_COUNT * SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION) * (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION);
	if ( region_index == SPARK_GLM52_RESIDENT_DECODE_STAGE_W8LUT_MOE_PACK_REGION_W1_E0 )
		return((uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT * (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_W1_COMPONENT_COUNT * sizeof(uint16_t));
	if ( region_index == SPARK_GLM52_RESIDENT_DECODE_STAGE_W8LUT_MOE_PACK_REGION_W2_CODES )
		return((uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT * (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION * (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION);
	if ( region_index == SPARK_GLM52_RESIDENT_DECODE_STAGE_W8LUT_MOE_PACK_REGION_W2_E0 )
		return((uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT * sizeof(uint16_t));
	return(0u);
}

static SparkStatus SparkGlm52W8lutMoePlanValidateRegions(const SparkGlm52ResidentDecodeStageW8lutMoePackHeader *header,uint64_t file_size)
{
	const SparkGlm52ResidentDecodeStageW8lutMoePackRegion *region;
	uint64_t end_offset,expected_offset,expected_bytes;
	uint32_t region_index;
	SparkStatus status;
	if ( header == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	expected_offset = SPARK_GLM52_RESIDENT_DECODE_STAGE_W8LUT_MOE_PACK_HEADER_BYTES;
	for (region_index=0u; region_index<SPARK_GLM52_RESIDENT_DECODE_STAGE_W8LUT_MOE_PACK_REGION_COUNT; region_index++)
	{
		region = &header->regions[region_index];
		expected_offset = SparkGlm52W8lutMoePlanAlignUp(expected_offset,SPARK_GLM52_RESIDENT_DECODE_STAGE_W8LUT_MOE_PACK_REGION_ALIGNMENT);
		expected_bytes = SparkGlm52W8lutMoePlanExpectedRegionBytes(region_index);
		status = SparkGlm52ResidentPackIoCheckedAdd(region->offset,region->bytes,&end_offset);
		if ( expected_offset == 0u || expected_bytes == 0u || status != SPARK_STATUS_OK ||
			region->offset != expected_offset || region->bytes != expected_bytes || end_offset > file_size )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		expected_offset = end_offset;
	}
	if ( expected_offset != file_size )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm52W8lutMoePlanValidateHeader(const SparkGlm52ResidentDecodeStageW8lutMoePackHeader *header,const SparkGlm52ResidentDecodeStageW8lutMoeResidentBindingCreateInfo *create_info,uint64_t file_size)
{
	uint32_t index;
	SparkStatus status;
	if ( header == 0 || create_info == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( memcmp(header->magic,SPARK_GLM52_RESIDENT_DECODE_STAGE_W8LUT_MOE_PACK_MAGIC,SPARK_GLM52_RESIDENT_DECODE_STAGE_W8LUT_MOE_PACK_MAGIC_BYTES) != 0 ||
		header->abi_version != SPARK_GLM52_RESIDENT_DECODE_STAGE_W8LUT_MOE_PACK_ABI_VERSION ||
		header->header_bytes != SPARK_GLM52_RESIDENT_DECODE_STAGE_W8LUT_MOE_PACK_HEADER_BYTES ||
		header->layer_index != create_info->layer_index ||
		header->maximum_token_count < create_info->maximum_active_sequence_count ||
		header->hidden_dimension != SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION ||
		header->intermediate_dimension != SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION ||
		header->expert_count != SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT ||
		header->top_k != SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_TOP_K ||
		header->gate_up_order != SPARK_GLM52_RESIDENT_DECODE_STAGE_W8LUT_MOE_GATE_UP_ORDER_UP_GATE ||
		header->weight_layout != SPARK_GLM52_RESIDENT_DECODE_STAGE_W8LUT_MOE_WEIGHT_LAYOUT_EXPERT_MAJOR_ROW_MAJOR ||
		header->scale_layout != SPARK_GLM52_RESIDENT_DECODE_STAGE_W8LUT_MOE_SCALE_LAYOUT_EXPERT_COMPONENT_E0 ||
		header->quant_mode != SPARK_GLM52_RESIDENT_DECODE_STAGE_W8LUT_MOE_QUANT_MODE ||
		header->output_dtype != SPARK_GLM52_RESIDENT_DECODE_STAGE_W8LUT_MOE_OUTPUT_DTYPE_BF16 ||
		header->cuda_architecture != 121u || header->reserved0 != 0u || header->reserved1 != 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	for (index=0u; index<(uint32_t)sizeof(header->reserved_bytes); index++)
		if ( header->reserved_bytes[index] != 0u )
			return(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkGlm52W8lutMoePlanValidateRegions(header,file_size);
	return(status);
}

static void SparkGlm52W8lutMoePlanPopulateBinding(SparkGlm52ResidentDecodeStageW8lutMoeResidentBinding *binding,const SparkGlm52ResidentDecodeStageW8lutMoePackHeader *header,uint32_t maximum_active_sequence_count)
{
	memset(&binding->plan,0,sizeof(binding->plan));
	binding->abi_version = SPARK_GLM52_RESIDENT_DECODE_STAGE_W8LUT_MOE_BINDING_ABI_VERSION;
	binding->layer_index = header->layer_index;
	binding->plan.abi_version = SPARK_GLM52_RESIDENT_DECODE_STAGE_W8LUT_MOE_PLAN_ABI_VERSION;
	binding->plan.capability_flags = SPARK_GLM52_RESIDENT_DECODE_STAGE_W8LUT_MOE_REQUIRED_CAPABILITIES;
	binding->plan.maximum_active_sequence_count = maximum_active_sequence_count;
	binding->plan.maximum_token_count = header->maximum_token_count;
	binding->plan.expert_count = header->expert_count;
	binding->plan.top_k = header->top_k;
	binding->plan.hidden_dimension = header->hidden_dimension;
	binding->plan.intermediate_dimension = header->intermediate_dimension;
	binding->plan.output_dtype = header->output_dtype;
	binding->plan.cuda_architecture = header->cuda_architecture;
	binding->plan.gate_up_order = header->gate_up_order;
	binding->plan.weight_layout = header->weight_layout;
	binding->plan.scale_layout = header->scale_layout;
	binding->plan.quant_mode = header->quant_mode;
	binding->plan.w1_weight_codes = binding->w1_weight_codes;
	binding->plan.w1_exponent_base = binding->w1_exponent_base;
	binding->plan.w2_weight_codes = binding->w2_weight_codes;
	binding->plan.w2_exponent_base = binding->w2_exponent_base;
}

SparkStatus SparkGlm52ResidentDecodeStageW8lutMoeResidentBindingCreateFromPackFile(SparkGlm52ResidentDecodeStageW8lutMoeResidentBinding *binding,const SparkGlm52ResidentDecodeStageW8lutMoeResidentBindingCreateInfo *create_info)
{
	uint8_t header_bytes[SPARK_GLM52_RESIDENT_DECODE_STAGE_W8LUT_MOE_PACK_HEADER_BYTES];
	SparkGlm52ResidentDecodeStageW8lutMoePackHeader header;
	FILE *file;
	uint64_t file_size,workspace_bytes;
	SparkStatus status;
	cudaError_t cuda_status;
	if ( binding == 0 || create_info == 0 || create_info->pack_path == 0 ||
		create_info->abi_version != SPARK_GLM52_RESIDENT_DECODE_STAGE_W8LUT_MOE_BINDING_CREATE_ABI_VERSION ||
		(create_info->flags & ~SPARK_GLM52_RESIDENT_DECODE_STAGE_W8LUT_MOE_BINDING_CREATE_KNOWN_FLAGS) != 0u ||
		create_info->maximum_active_sequence_count == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( ((create_info->flags & SPARK_GLM52_RESIDENT_DECODE_STAGE_W8LUT_MOE_BINDING_CREATE_FLAG_EXTERNAL_WORKSPACE) != 0u && (create_info->external_workspace == 0 || create_info->external_workspace_bytes == 0u)) ||
		((create_info->flags & SPARK_GLM52_RESIDENT_DECODE_STAGE_W8LUT_MOE_BINDING_CREATE_FLAG_EXTERNAL_WORKSPACE) == 0u && (create_info->external_workspace != 0 || create_info->external_workspace_bytes != 0u)) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(binding,0,sizeof(*binding));
	file = fopen(create_info->pack_path,"rb");
	if ( file == 0 )
		return(errno == ENOENT ? SPARK_STATUS_NOT_FOUND : SPARK_STATUS_IO_ERROR);
	status = SparkGlm52ResidentPackIoRead(file,header_bytes,sizeof(header_bytes));
	if ( status == SPARK_STATUS_OK )
		memcpy(&header,header_bytes,sizeof(header));
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm52ResidentPackIoFileSize(file,&file_size);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm52W8lutMoePlanValidateHeader(&header,create_info,file_size);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm52ResidentPackIoLoadDeviceRegion(file,header.regions[SPARK_GLM52_RESIDENT_DECODE_STAGE_W8LUT_MOE_PACK_REGION_W1_CODES].offset,header.regions[SPARK_GLM52_RESIDENT_DECODE_STAGE_W8LUT_MOE_PACK_REGION_W1_CODES].bytes,(void **)&binding->w1_weight_codes);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm52ResidentPackIoLoadDeviceRegion(file,header.regions[SPARK_GLM52_RESIDENT_DECODE_STAGE_W8LUT_MOE_PACK_REGION_W1_E0].offset,header.regions[SPARK_GLM52_RESIDENT_DECODE_STAGE_W8LUT_MOE_PACK_REGION_W1_E0].bytes,(void **)&binding->w1_exponent_base);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm52ResidentPackIoLoadDeviceRegion(file,header.regions[SPARK_GLM52_RESIDENT_DECODE_STAGE_W8LUT_MOE_PACK_REGION_W2_CODES].offset,header.regions[SPARK_GLM52_RESIDENT_DECODE_STAGE_W8LUT_MOE_PACK_REGION_W2_CODES].bytes,(void **)&binding->w2_weight_codes);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm52ResidentPackIoLoadDeviceRegion(file,header.regions[SPARK_GLM52_RESIDENT_DECODE_STAGE_W8LUT_MOE_PACK_REGION_W2_E0].offset,header.regions[SPARK_GLM52_RESIDENT_DECODE_STAGE_W8LUT_MOE_PACK_REGION_W2_E0].bytes,(void **)&binding->w2_exponent_base);
	fclose(file);
	if ( status == SPARK_STATUS_OK )
	{
		SparkGlm52W8lutMoePlanPopulateBinding(binding,&header,create_info->maximum_active_sequence_count);
		workspace_bytes = SparkGlm52Sm121RequiredDecodeStageCalculateW8lutMoeWorkspaceBytes(&binding->plan);
		if ( workspace_bytes == 0u || workspace_bytes > (uint64_t)((size_t)-1) )
			status = SPARK_STATUS_CAPACITY_EXCEEDED;
	}
	if ( status == SPARK_STATUS_OK && (create_info->flags & SPARK_GLM52_RESIDENT_DECODE_STAGE_W8LUT_MOE_BINDING_CREATE_FLAG_EXTERNAL_WORKSPACE) != 0u )
	{
		if ( create_info->external_workspace_bytes < workspace_bytes )
			status = SPARK_STATUS_CAPACITY_EXCEEDED;
		else
		{
			binding->workspace = create_info->external_workspace;
			binding->workspace_owned = 0u;
		}
	}
	else if ( status == SPARK_STATUS_OK )
	{
		cuda_status = cudaMalloc(&binding->workspace,(size_t)workspace_bytes);
		status = SparkGlm52ResidentPackIoCudaStatus(cuda_status);
		if ( status == SPARK_STATUS_OK )
			binding->workspace_owned = 1u;
	}
	if ( status == SPARK_STATUS_OK )
	{
		binding->plan.workspace = binding->workspace;
		binding->plan.workspace_bytes = workspace_bytes;
		status = SparkGlm52Sm121RequiredDecodeStageBindW8lutMoePlan(&binding->plan);
		if ( status == SPARK_STATUS_OK )
			binding->backend_kind = SPARK_GLM52_RESIDENT_DECODE_STAGE_W8LUT_MOE_BACKEND_BUILTIN_BF16_WMMA;
	}
	if ( status != SPARK_STATUS_OK )
	{
		SparkGlm52ResidentDecodeStageW8lutMoeResidentBindingDestroy(binding);
		return(status);
	}
	return(SPARK_STATUS_OK);
}

void SparkGlm52ResidentDecodeStageW8lutMoeResidentBindingDestroy(SparkGlm52ResidentDecodeStageW8lutMoeResidentBinding *binding)
{
	if ( binding == 0 )
		return;
	SparkGlm52ResidentPackIoFreeDevice((void **)&binding->w1_weight_codes);
	SparkGlm52ResidentPackIoFreeDevice((void **)&binding->w1_exponent_base);
	SparkGlm52ResidentPackIoFreeDevice((void **)&binding->w2_weight_codes);
	SparkGlm52ResidentPackIoFreeDevice((void **)&binding->w2_exponent_base);
	if ( binding->workspace_owned != 0u )
		SparkGlm52ResidentPackIoFreeDevice(&binding->workspace);
	else
		binding->workspace = 0;
	memset(binding,0,sizeof(*binding));
}
