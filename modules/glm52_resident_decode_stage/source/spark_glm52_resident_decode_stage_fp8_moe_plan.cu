#include "sparkpipe/spark_glm52_resident_decode_stage_fp8_moe_plan.h"
#include "sparkpipe/spark_glm52_resident_decode_stage_required_cuda.h"
#include "sparkpipe/spark_glm52_sm121_flashinfer_b12x_moe.h"
#include "spark_glm52_resident_decode_stage_pack_io.h"

#include <cuda_runtime_api.h>

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static uint64_t SparkGlm52Fp8MoePlanExpectedRegionBytes(uint32_t region_index)
{
	if (region_index == SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_PACK_REGION_W1_WEIGHT)
		return ((uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT * (uint64_t)(SPARK_RESIDENT_DECODE_STAGE_MOE_W1_COMPONENT_COUNT * SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION) * (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION);
	if (region_index == SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_PACK_REGION_W1_SCALE_INV)
		return ((uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT * SPARK_RESIDENT_DECODE_STAGE_FP8_SCALE_COUNT(SPARK_RESIDENT_DECODE_STAGE_MOE_W1_COMPONENT_COUNT * SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION,SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION) * sizeof(float));
	if (region_index == SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_PACK_REGION_W2_WEIGHT)
		return ((uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT * (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION * (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION);
	if (region_index == SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_PACK_REGION_W2_SCALE_INV)
		return ((uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT * SPARK_RESIDENT_DECODE_STAGE_FP8_SCALE_COUNT(SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION) * sizeof(float));
	return 0u;
}

static SparkStatus SparkGlm52Fp8MoePlanValidateRegion(const SparkGlm52ResidentDecodeStageFp8MoePackHeader *header,uint32_t region_index,uint64_t file_size)
{
	const SparkGlm52ResidentDecodeStageFp8MoePackRegion *region;
	uint64_t end_offset,expected_bytes;
	SparkStatus status;

	if (header == 0 || region_index >= SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_PACK_REGION_COUNT)
		return SPARK_STATUS_INVALID_ARGUMENT;
	region = &header->regions[region_index];
	expected_bytes = SparkGlm52Fp8MoePlanExpectedRegionBytes(region_index);
	status = SparkGlm52ResidentPackIoCheckedAdd(region->offset,region->bytes,&end_offset);
	if (status != SPARK_STATUS_OK)
		return status;
	if (region->offset < SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_PACK_HEADER_BYTES ||
		(region->offset % SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_PACK_REGION_ALIGNMENT) != 0u ||
		region->bytes != expected_bytes ||
		end_offset > file_size)
		return SPARK_STATUS_INVALID_ARGUMENT;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Fp8MoePlanParseHeader(const uint8_t *header_bytes,SparkGlm52ResidentDecodeStageFp8MoePackHeader *header)
{
	if (header_bytes == 0 || header == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	memcpy(header,header_bytes,sizeof(*header));
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Fp8MoePlanValidatePackHeader(const SparkGlm52ResidentDecodeStageFp8MoePackHeader *header,const SparkResidentDecodeStageFp8MoeResidentBindingCreateInfo *create_info,uint64_t file_size)
{
	uint32_t region_index;
	SparkStatus status;

	if (header == 0 || create_info == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (memcmp(header->magic,SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_PACK_MAGIC,SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_PACK_MAGIC_BYTES) != 0 ||
		header->abi_version != SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_PACK_ABI_VERSION ||
		header->header_bytes != SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_PACK_HEADER_BYTES ||
		header->layer_index != create_info->layer_index ||
		header->hidden_dimension != SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION ||
		header->intermediate_dimension != SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION ||
		header->expert_count != SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT ||
		header->top_k != SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_TOP_K ||
		header->gate_up_order != SPARK_RESIDENT_DECODE_STAGE_FP8_MOE_GATE_UP_ORDER_UP_GATE ||
		header->weight_layout != SPARK_RESIDENT_DECODE_STAGE_FP8_MOE_WEIGHT_LAYOUT_EXPERT_MAJOR_ROW_MAJOR ||
		header->scale_layout != SPARK_RESIDENT_DECODE_STAGE_FP8_MOE_SCALE_LAYOUT_EXPERT_MAJOR_ROW_BLOCK_MAJOR ||
		header->quant_mode != SPARK_RESIDENT_DECODE_STAGE_FP8_MOE_QUANT_MODE_E4M3 ||
		header->output_dtype != SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_OUTPUT_DTYPE_BF16 ||
		header->cuda_architecture != 121u ||
		header->reserved0 != 0u ||
		header->reserved1 != 0u ||
		file_size < SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_PACK_HEADER_BYTES)
		return SPARK_STATUS_INVALID_ARGUMENT;
	for (region_index=0; region_index<SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_PACK_REGION_COUNT; region_index++)
	{
		status = SparkGlm52Fp8MoePlanValidateRegion(header,region_index,file_size);
		if (status != SPARK_STATUS_OK)
			return status;
	}
	return SPARK_STATUS_OK;
}


static void SparkGlm52Fp8MoePlanPopulateBinding(SparkResidentDecodeStageFp8MoeResidentBinding *binding,const SparkGlm52ResidentDecodeStageFp8MoePackHeader *header,uint32_t maximum_active_sequence_count)
{
	memset(&binding->plan,0,sizeof(binding->plan));
	binding->abi_version = SPARK_RESIDENT_DECODE_STAGE_FP8_MOE_PLAN_ABI_VERSION;
	binding->layer_index = header->layer_index;
	binding->plan.abi_version = SPARK_RESIDENT_DECODE_STAGE_FP8_MOE_PLAN_ABI_VERSION;
	binding->plan.capability_flags = SPARK_RESIDENT_DECODE_STAGE_FP8_MOE_REQUIRED_CAPABILITIES;
	binding->plan.maximum_active_sequence_count = maximum_active_sequence_count;
	binding->plan.maximum_token_count =
		header->maximum_token_count > maximum_active_sequence_count
		? header->maximum_token_count
		: maximum_active_sequence_count;
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
	binding->plan.scale_block_size = SPARK_RESIDENT_DECODE_STAGE_FP8_MOE_SCALE_BLOCK_SIZE;
	binding->plan.w1_weight_fp8_e4m3 = binding->w1_weight_fp8_e4m3;
	binding->plan.w1_scale_inv_f32 = binding->w1_scale_inv_f32;
	binding->plan.w2_weight_fp8_e4m3 = binding->w2_weight_fp8_e4m3;
	binding->plan.w2_scale_inv_f32 = binding->w2_scale_inv_f32;
	binding->plan.workspace = binding->workspace;
	binding->plan.validated_maximum_latency_ns = 0u;
}

SparkStatus SparkResidentDecodeStageFp8MoeResidentBindingCreateFromPackFile(SparkResidentDecodeStageFp8MoeResidentBinding *binding,const SparkResidentDecodeStageFp8MoeResidentBindingCreateInfo *create_info)
{
	uint8_t header_bytes[SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_PACK_HEADER_BYTES];
	SparkGlm52ResidentDecodeStageFp8MoePackHeader header;
	FILE *file;
	uint64_t file_size,workspace_bytes;
	SparkStatus status;
	cudaError_t cuda_status;

	if (binding == 0 || create_info == 0 || create_info->pack_path == 0 ||
		create_info->abi_version !=
			SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_BINDING_CREATE_ABI_VERSION ||
		(create_info->flags &
			~SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_BINDING_CREATE_KNOWN_FLAGS) != 0u ||
		create_info->maximum_active_sequence_count == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (((create_info->flags &
			SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_BINDING_CREATE_FLAG_EXTERNAL_WORKSPACE) != 0u &&
		 (create_info->external_workspace == 0 ||
		  create_info->external_workspace_bytes == 0u)) ||
		((create_info->flags &
			SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_BINDING_CREATE_FLAG_EXTERNAL_WORKSPACE) == 0u &&
		 (create_info->external_workspace != 0 ||
		  create_info->external_workspace_bytes != 0u)))
		return SPARK_STATUS_INVALID_ARGUMENT;
	memset(binding,0,sizeof(*binding));
	file = fopen(create_info->pack_path,"rb");
	if (file == 0)
		return errno == ENOENT ? SPARK_STATUS_NOT_FOUND : SPARK_STATUS_IO_ERROR;
	status = SparkGlm52ResidentPackIoRead(file,header_bytes,sizeof(header_bytes));
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Fp8MoePlanParseHeader(header_bytes,&header);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52ResidentPackIoFileSize(file,&file_size);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Fp8MoePlanValidatePackHeader(&header,create_info,file_size);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52ResidentPackIoLoadDeviceRegion(file,header.regions[SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_PACK_REGION_W1_WEIGHT].offset,header.regions[SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_PACK_REGION_W1_WEIGHT].bytes,(void **)&binding->w1_weight_fp8_e4m3);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52ResidentPackIoLoadDeviceRegion(file,header.regions[SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_PACK_REGION_W1_SCALE_INV].offset,header.regions[SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_PACK_REGION_W1_SCALE_INV].bytes,(void **)&binding->w1_scale_inv_f32);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52ResidentPackIoLoadDeviceRegion(file,header.regions[SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_PACK_REGION_W2_WEIGHT].offset,header.regions[SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_PACK_REGION_W2_WEIGHT].bytes,(void **)&binding->w2_weight_fp8_e4m3);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52ResidentPackIoLoadDeviceRegion(file,header.regions[SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_PACK_REGION_W2_SCALE_INV].offset,header.regions[SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_PACK_REGION_W2_SCALE_INV].bytes,(void **)&binding->w2_scale_inv_f32);
	fclose(file);
	file = 0;
	if (status == SPARK_STATUS_OK)
	{
		SparkGlm52Fp8MoePlanPopulateBinding(binding,&header,create_info->maximum_active_sequence_count);
		workspace_bytes = SparkGlm52Sm121RequiredDecodeStageCalculateFp8MoeGroupedReferenceWorkspaceBytes(&binding->plan);
		if (workspace_bytes == 0u || workspace_bytes > (uint64_t)((size_t)-1))
			status = SPARK_STATUS_CAPACITY_EXCEEDED;
	}
	if (status == SPARK_STATUS_OK)
	{
		if ((create_info->flags &
				SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_BINDING_CREATE_FLAG_EXTERNAL_WORKSPACE) != 0u)
		{
			if (create_info->external_workspace_bytes < workspace_bytes)
				status = SPARK_STATUS_CAPACITY_EXCEEDED;
			else
			{
				binding->workspace = create_info->external_workspace;
				binding->workspace_owned = 0u;
			}
		}
		else
		{
			cuda_status = cudaMalloc(&binding->workspace,(size_t)workspace_bytes);
			status = SparkGlm52ResidentPackIoCudaStatus(cuda_status);
			if (status == SPARK_STATUS_OK)
				binding->workspace_owned = 1u;
		}
	}
	if (status == SPARK_STATUS_OK)
	{
		binding->plan.workspace = binding->workspace;
		binding->plan.workspace_bytes = workspace_bytes;
		status = SparkGlm52Sm121RequiredDecodeStageBindFp8MoeGroupedReferencePlan(&binding->plan);
		if (status == SPARK_STATUS_OK)
			binding->backend_kind =
				SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_BACKEND_BUILTIN_FLASHINFER_GROUPED;
	}
	if (status != SPARK_STATUS_OK)
	{
		SparkResidentDecodeStageFp8MoeResidentBindingDestroy(binding);
		return status;
	}
	return SPARK_STATUS_OK;
}

void SparkResidentDecodeStageFp8MoeResidentBindingDestroy(SparkResidentDecodeStageFp8MoeResidentBinding *binding)
{
	if (binding == 0)
		return;
	SparkGlm52ResidentPackIoFreeDevice((void **)&binding->w1_weight_fp8_e4m3);
	SparkGlm52ResidentPackIoFreeDevice((void **)&binding->w1_scale_inv_f32);
	SparkGlm52ResidentPackIoFreeDevice((void **)&binding->w2_weight_fp8_e4m3);
	SparkGlm52ResidentPackIoFreeDevice((void **)&binding->w2_scale_inv_f32);
	if (binding->workspace_owned != 0u)
		SparkGlm52ResidentPackIoFreeDevice(&binding->workspace);
	else
		binding->workspace = 0;
	memset(binding,0,sizeof(*binding));
}
