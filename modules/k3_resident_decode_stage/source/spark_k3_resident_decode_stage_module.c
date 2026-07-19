/* Large stage packs exceed 2 GB: 64-bit file offsets and fseeko are required. */
#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime_api.h>

#include "sparkpipe/spark_k3_resident_decode_stage_firmware.h"
#include "spark_k3_stagepack_format.h"

/*
 * K3 resident decode stage, host side.
 *
 * Owns everything that is resident: the stage pack's weights, the KDA
 * recurrent state pool, the paged MLA latent cache and the per-slot
 * activation buffers. Execute walks the layer stack for one dispatch and the
 * CUDA translation unit performs the work; this file never computes on
 * tensors, it only validates, places and sequences.
 *
 * Version 1 is a single-node bring-up driver: the pack must contain the whole
 * stack, so the node owns the embedding and the head, and pipeline transport
 * is rejected rather than silently ignored. See DIFFERENCES.md.
 */

#define SPARK_K3_MODULE_MAX_DEVICE_ALLOCATIONS 4096u
#define SPARK_K3_MODULE_STAGING_CHUNK_BYTES (64u * 1024u * 1024u)
#define SPARK_K3_MODULE_MAX_MOE_INTERMEDIATE_ROW_ELEMENTS \
	((SPARK_K3_MODEL_MOE_TOP_K * SPARK_K3_MODEL_MOE_INTERMEDIATE_DIMENSION) > SPARK_K3_MODEL_DENSE_INTERMEDIATE_DIMENSION \
		? (SPARK_K3_MODEL_MOE_TOP_K * SPARK_K3_MODEL_MOE_INTERMEDIATE_DIMENSION) \
		: SPARK_K3_MODEL_DENSE_INTERMEDIATE_DIMENSION)

/*
 * Host staging owned by one pipeline slot. Concurrent Execute calls claim
 * distinct slots, so nothing here is shared; the device token count is a
 * per-slot single int32 the chunk kernel reads for partial-chunk masking.
 * Every buffer that sources an async H2D copy lives here for the life of the
 * module, so the copies stay correct even if these pages are later pinned.
 */
typedef struct SparkK3ModuleSlotStaging
{
	uint32_t *row_token_ids;
	uint32_t *row_slot_mapping;
	uint32_t *row_lane_indices;
	uint32_t *row_context_lengths;
	uint32_t *row_cold_flags;
	uint32_t *row_output_token_ids;
	int32_t sequence_token_count;
	int32_t *device_sequence_token_count;
	atomic_uint busy;
} SparkK3ModuleSlotStaging;

typedef struct SparkK3ModuleState
{
	SparkK3ResidentDecodeStageNodeContext node_context;
	SparkFirmwareModuleHostServices host_services;
	SparkK3PipelineSlot pipeline_slots[SPARK_K3_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
	SparkK3ModuleSlotStaging slot_staging[SPARK_K3_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
	SparkK3AttnResSiteWeights attnres_sites[SPARK_K3_MODEL_LAYER_COUNT * SPARK_K3_RESIDENT_DECODE_STAGE_ATTNRES_SITES_PER_LAYER];
	SparkK3AttnResSiteWeights attnres_final_site;
	SparkK3KdaLayerWeights kda_weights[SPARK_K3_MODEL_LAYER_COUNT];
	SparkK3MlaLayerWeights mla_weights[SPARK_K3_MODEL_LAYER_COUNT];
	SparkK3MoeLayerWeights moe_weights[SPARK_K3_MODEL_LAYER_COUNT];
	const void *attention_norm_by_layer[SPARK_K3_MODEL_LAYER_COUNT];
	const void *mlp_norm_by_layer[SPARK_K3_MODEL_LAYER_COUNT];
	uint32_t kda_ordinal_by_layer[SPARK_K3_MODEL_LAYER_COUNT];
	uint32_t mla_ordinal_by_layer[SPARK_K3_MODEL_LAYER_COUNT];
	uint32_t kda_layer_count;
	uint32_t mla_layer_count;
	SparkK3MlaBlockTableView owned_block_table;
	uint32_t *host_block_indices;
	uint32_t *host_lane_block_counts;
	uint32_t lane_capacity;
	uint32_t row_capacity;
	uint32_t max_context_tokens;
	uint32_t blocks_per_lane;
	uint32_t scratch_block_index;
	atomic_uint next_pipeline_slot;
	atomic_uint_fast64_t submitted_count;
	atomic_uint_fast64_t completed_count;
	atomic_uint_fast64_t rejected_count;
	atomic_uint_fast64_t failed_count;
	uint64_t device_bytes_resident;
	void *device_allocations[SPARK_K3_MODULE_MAX_DEVICE_ALLOCATIONS];
	uint32_t device_allocation_count;
	uint32_t program_id;
} SparkK3ModuleState;

static SparkStatus SparkK3ModuleCudaStatus(cudaError_t error, const char *site)
{
	if ( error == cudaSuccess )
		return(SPARK_STATUS_OK);
	fprintf(stderr,"k3_stage cuda_error site=%s error=%s\n",site,cudaGetErrorString(error));
	if ( error == cudaErrorMemoryAllocation )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	return(SPARK_STATUS_INTERNAL_ERROR);
}

/*
 * Configuration is read from the environment and every value is required.
 * A missing or unparsable variable is a hard failure: the driver never
 * invents a dimension, a capacity or a path, because a silently defaulted
 * configuration produces a model that runs and is wrong.
 */
static SparkStatus SparkK3ModuleEnvironmentText(const char *name, const char **value)
{
	const char *text = getenv(name);
	if ( text == 0 || text[0] == '\0' )
	{
		fprintf(stderr,"k3_stage config_missing name=%s\n",name);
		return(SPARK_STATUS_INVALID_ARGUMENT);
	}
	*value = text;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkK3ModuleEnvironmentUnsigned(const char *name, uint32_t minimum, uint32_t maximum, uint32_t *value)
{
	const char *text;
	char *end;
	unsigned long parsed;
	SparkStatus status = SparkK3ModuleEnvironmentText(name,&text);
	if ( status != SPARK_STATUS_OK )
		return(status);
	end = 0;
	parsed = strtoul(text,&end,10);
	if ( end == text || (end != 0 && *end != '\0') || parsed < (unsigned long)minimum || parsed > (unsigned long)maximum )
	{
		fprintf(stderr,"k3_stage config_invalid name=%s value=%s allowed=[%u,%u]\n",name,text,minimum,maximum);
		return(SPARK_STATUS_INVALID_ARGUMENT);
	}
	*value = (uint32_t)parsed;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkK3ModuleRecordAllocation(SparkK3ModuleState *state, void *pointer)
{
	if ( state->device_allocation_count >= SPARK_K3_MODULE_MAX_DEVICE_ALLOCATIONS )
	{
		fprintf(stderr,"k3_stage allocation_ledger_full count=%u\n",state->device_allocation_count);
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	state->device_allocations[state->device_allocation_count] = pointer;
	state->device_allocation_count++;
	return(SPARK_STATUS_OK);
}

// Device allocations happen only here, only during initialization, and every
// pointer is recorded so Destroy can release the node without a leak.
static SparkStatus SparkK3ModuleDeviceAllocate(SparkK3ModuleState *state, uint64_t bytes, void **pointer)
{
	SparkStatus status;
	void *allocation = 0;
	if ( bytes == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkK3ModuleCudaStatus(cudaMalloc(&allocation,(size_t)bytes),"cudaMalloc");
	if ( status != SPARK_STATUS_OK )
		return(status);
	status = SparkK3ModuleRecordAllocation(state,allocation);
	if ( status != SPARK_STATUS_OK )
	{
		cudaFree(allocation);
		return(status);
	}
	state->device_bytes_resident += bytes;
	*pointer = allocation;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkK3ModuleDeviceAllocateZeroed(SparkK3ModuleState *state, uint64_t bytes, void **pointer)
{
	SparkStatus status = SparkK3ModuleDeviceAllocate(state,bytes,pointer);
	if ( status != SPARK_STATUS_OK )
		return(status);
	return(SparkK3ModuleCudaStatus(cudaMemset(*pointer,0,(size_t)bytes),"cudaMemset"));
}

static SparkStatus SparkK3ModulePackRead(FILE *file, uint64_t offset, void *destination, uint64_t bytes)
{
	if ( fseeko(file,(off_t)offset,SEEK_SET) != 0 )
	{
		fprintf(stderr,"k3_stage pack_seek_failed offset=%llu\n",(unsigned long long)offset);
		return(SPARK_STATUS_IO_ERROR);
	}
	if ( fread(destination,1,(size_t)bytes,file) != (size_t)bytes )
	{
		fprintf(stderr,"k3_stage pack_read_failed offset=%llu bytes=%llu\n",(unsigned long long)offset,(unsigned long long)bytes);
		return(SPARK_STATUS_IO_ERROR);
	}
	return(SPARK_STATUS_OK);
}

/*
 * Stage a file region into a fresh device allocation through a bounded host
 * buffer, so a multi-gigabyte expert tensor never needs a host copy of its
 * own size.
 */
static SparkStatus SparkK3ModuleLoadDeviceRegion(SparkK3ModuleState *state, FILE *file, uint64_t offset, uint64_t bytes, void **pointer)
{
	SparkStatus status;
	void *device = 0;
	void *staging;
	uint64_t moved,chunk;
	status = SparkK3ModuleDeviceAllocate(state,bytes,&device);
	if ( status != SPARK_STATUS_OK )
		return(status);
	staging = malloc(bytes < SPARK_K3_MODULE_STAGING_CHUNK_BYTES ? (size_t)bytes : (size_t)SPARK_K3_MODULE_STAGING_CHUNK_BYTES);
	if ( staging == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	for (moved = 0; moved < bytes; moved += chunk)
	{
		chunk = bytes - moved;
		if ( chunk > SPARK_K3_MODULE_STAGING_CHUNK_BYTES )
			chunk = SPARK_K3_MODULE_STAGING_CHUNK_BYTES;
		status = SparkK3ModulePackRead(file,offset + moved,staging,chunk);
		if ( status == SPARK_STATUS_OK )
			status = SparkK3ModuleCudaStatus(cudaMemcpy((uint8_t *)device + moved,staging,(size_t)chunk,cudaMemcpyHostToDevice),"cudaMemcpy_h2d");
		if ( status != SPARK_STATUS_OK )
		{
			free(staging);
			return(status);
		}
	}
	free(staging);
	*pointer = device;
	return(SPARK_STATUS_OK);
}

/*
 * A pack tensor lands in exactly one of two shapes: a linear view (payload
 * plus optional scale plus its dimensions) or a bare pointer cell (norm
 * gains, pseudo-queries, the router, the router bias, the embedding, the
 * restricted token ids). The moe expert tensors are neither: they are
 * expert-major concatenations bound with strides, handled separately.
 */
typedef struct SparkK3ModuleTensorBinding
{
	SparkK3Mxfp4LinearView *view;
	const void **pointer_cell;
} SparkK3ModuleTensorBinding;

static int32_t SparkK3ModuleBindingGlobal(SparkK3ModuleState *state, uint32_t tensor_kind, SparkK3ModuleTensorBinding *binding)
{
	switch ( tensor_kind )
	{
	case SPARK_K3_STAGEPACK_TENSOR_EMBEDDING:
		binding->pointer_cell = &state->node_context.token_embedding_bf16;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_ATTNRES_FINAL_QUERY:
		binding->pointer_cell = &state->attnres_final_site.pseudo_query_bf16;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_ATTNRES_FINAL_NORM:
		binding->pointer_cell = &state->attnres_final_site.key_norm_weight_bf16;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_FINAL_NORM:
		binding->pointer_cell = &state->node_context.final_norm_weight_bf16;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_LM_HEAD_RESTRICTED:
		binding->pointer_cell = &state->node_context.restricted_lm_head_weight_bf16;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_RESTRICTED_TOKEN_IDS:
		binding->pointer_cell = (const void **)&state->node_context.restricted_token_ids;
		return(0);
	default:
		return(-1);
	}
}

static int32_t SparkK3ModuleBindingSite(SparkK3ModuleState *state, uint32_t tensor_kind, uint32_t layer_index, SparkK3ModuleTensorBinding *binding)
{
	SparkK3AttnResSiteWeights *attention_site = &state->attnres_sites[(layer_index * SPARK_K3_RESIDENT_DECODE_STAGE_ATTNRES_SITES_PER_LAYER) + SPARK_K3_RESIDENT_DECODE_STAGE_ATTNRES_ATTENTION_SITE];
	SparkK3AttnResSiteWeights *mlp_site = &state->attnres_sites[(layer_index * SPARK_K3_RESIDENT_DECODE_STAGE_ATTNRES_SITES_PER_LAYER) + SPARK_K3_RESIDENT_DECODE_STAGE_ATTNRES_MLP_SITE];
	switch ( tensor_kind )
	{
	case SPARK_K3_STAGEPACK_TENSOR_ATTNRES_ATTENTION_QUERY:
		binding->pointer_cell = &attention_site->pseudo_query_bf16;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_ATTNRES_ATTENTION_NORM:
		binding->pointer_cell = &attention_site->key_norm_weight_bf16;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_ATTNRES_MLP_QUERY:
		binding->pointer_cell = &mlp_site->pseudo_query_bf16;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_ATTNRES_MLP_NORM:
		binding->pointer_cell = &mlp_site->key_norm_weight_bf16;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_ATTENTION_NORM:
		binding->pointer_cell = &state->attention_norm_by_layer[layer_index];
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_MLP_NORM:
		binding->pointer_cell = &state->mlp_norm_by_layer[layer_index];
		return(0);
	default:
		return(-1);
	}
}

static int32_t SparkK3ModuleBindingKda(SparkK3ModuleState *state, uint32_t tensor_kind, uint32_t layer_index, SparkK3ModuleTensorBinding *binding)
{
	SparkK3KdaLayerWeights *weights = &state->kda_weights[layer_index];
	switch ( tensor_kind )
	{
	case SPARK_K3_STAGEPACK_TENSOR_KDA_QUERY:
		binding->view = &weights->query;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_KDA_KEY:
		binding->view = &weights->key;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_KDA_VALUE:
		binding->view = &weights->value;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_KDA_DECAY_LOW:
		binding->view = &weights->decay_low;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_KDA_DECAY_HIGH:
		binding->view = &weights->decay_high;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_KDA_BETA:
		binding->view = &weights->beta;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_KDA_GATE_LOW:
		binding->view = &weights->output_gate_low;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_KDA_GATE_HIGH:
		binding->view = &weights->output_gate_high;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_KDA_OUTPUT:
		binding->view = &weights->output;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_KDA_HEAD_NORM:
		binding->pointer_cell = &weights->head_norm_weight_bf16;
		return(0);
	default:
		return(-1);
	}
}

static int32_t SparkK3ModuleBindingMla(SparkK3ModuleState *state, uint32_t tensor_kind, uint32_t layer_index, SparkK3ModuleTensorBinding *binding)
{
	SparkK3MlaLayerWeights *weights = &state->mla_weights[layer_index];
	switch ( tensor_kind )
	{
	case SPARK_K3_STAGEPACK_TENSOR_MLA_QUERY_A:
		binding->view = &weights->query_a;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_MLA_QUERY_A_NORM:
		binding->pointer_cell = &weights->query_a_norm_weight_bf16;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_MLA_QUERY_B:
		binding->view = &weights->query_b;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_MLA_KV_A:
		binding->view = &weights->kv_a;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_MLA_KV_A_NORM:
		binding->pointer_cell = &weights->kv_a_norm_weight_bf16;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_MLA_KV_B:
		binding->view = &weights->kv_b;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_MLA_HEAD_GATE:
		binding->view = &weights->head_gate;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_MLA_OUTPUT:
		binding->view = &weights->output;
		return(0);
	default:
		return(-1);
	}
}

static int32_t SparkK3ModuleBindingMoe(SparkK3ModuleState *state, uint32_t tensor_kind, uint32_t layer_index, SparkK3ModuleTensorBinding *binding)
{
	SparkK3MoeLayerWeights *weights = &state->moe_weights[layer_index];
	switch ( tensor_kind )
	{
	case SPARK_K3_STAGEPACK_TENSOR_MOE_ROUTER:
		binding->pointer_cell = &weights->router_weight_bf16;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_MOE_ROUTER_BIAS:
		binding->pointer_cell = (const void **)&weights->router_score_bias_f32;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_MOE_SHARED_GATE:
		binding->view = &weights->shared_gate;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_MOE_SHARED_UP:
		binding->view = &weights->shared_up;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_MOE_SHARED_DOWN:
		binding->view = &weights->shared_down;
		return(0);
	default:
		return(-1);
	}
}

static int32_t SparkK3ModuleBindingForTensor(SparkK3ModuleState *state, uint32_t tensor_kind, uint32_t layer_index, SparkK3ModuleTensorBinding *binding)
{
	binding->view = 0;
	binding->pointer_cell = 0;
	if ( SparkK3ModuleBindingGlobal(state,tensor_kind,binding) == 0 )
		return(0);
	if ( layer_index >= SPARK_K3_MODEL_LAYER_COUNT )
		return(-1);
	if ( SparkK3ModuleBindingSite(state,tensor_kind,layer_index,binding) == 0 )
		return(0);
	if ( SparkK3ModuleBindingKda(state,tensor_kind,layer_index,binding) == 0 )
		return(0);
	if ( SparkK3ModuleBindingMla(state,tensor_kind,layer_index,binding) == 0 )
		return(0);
	return(SparkK3ModuleBindingMoe(state,tensor_kind,layer_index,binding));
}

static uint32_t SparkK3ModuleTensorIsExpertConcatenation(uint32_t tensor_kind)
{
	return(tensor_kind == SPARK_K3_STAGEPACK_TENSOR_MOE_EXPERT_GATE || tensor_kind == SPARK_K3_STAGEPACK_TENSOR_MOE_EXPERT_UP || tensor_kind == SPARK_K3_STAGEPACK_TENSOR_MOE_EXPERT_DOWN);
}

/*
 * An expert tensor is one allocation holding every expert's matrix back to
 * back, so a route only needs a base pointer and a stride. The kernels index
 * expert e at payload + e*payload_stride, which is why the stride is derived
 * from the per-expert shape rather than stored in the pack.
 */
static SparkStatus SparkK3ModuleBindExpertConcatenation(SparkK3ModuleState *state, const SparkK3StagePackEntry *entry, void *payload, void *scale)
{
	SparkK3MoeLayerWeights *weights = &state->moe_weights[entry->layer_index];
	uint32_t expert_rows = entry->rows / SPARK_K3_MODEL_MOE_EXPERT_COUNT;
	uint64_t payload_stride = SparkK3StagePackPayloadBytes(entry->weight_format,expert_rows,entry->columns);
	uint64_t scale_stride = SparkK3StagePackScaleBytes(entry->weight_format,expert_rows,entry->columns);
	weights->expert_count = SPARK_K3_MODEL_MOE_EXPERT_COUNT;
	weights->intermediate_dimension = SPARK_K3_MODEL_MOE_INTERMEDIATE_DIMENSION;
	weights->weight_format = entry->weight_format;
	if ( entry->tensor_kind == SPARK_K3_STAGEPACK_TENSOR_MOE_EXPERT_GATE )
	{
		weights->expert_gate_payload = payload;
		weights->expert_gate_scale_e8m0 = (const uint8_t *)scale;
		weights->expert_gate_payload_stride_bytes = payload_stride;
		weights->expert_gate_scale_stride_bytes = scale_stride;
		return(SPARK_STATUS_OK);
	}
	if ( entry->tensor_kind == SPARK_K3_STAGEPACK_TENSOR_MOE_EXPERT_UP )
	{
		weights->expert_up_payload = payload;
		weights->expert_up_scale_e8m0 = (const uint8_t *)scale;
		weights->expert_up_payload_stride_bytes = payload_stride;
		weights->expert_up_scale_stride_bytes = scale_stride;
		return(SPARK_STATUS_OK);
	}
	weights->expert_down_payload = payload;
	weights->expert_down_scale_e8m0 = (const uint8_t *)scale;
	weights->expert_down_payload_stride_bytes = payload_stride;
	weights->expert_down_scale_stride_bytes = scale_stride;
	return(SPARK_STATUS_OK);
}

/*
 * Validate one directory entry against the shape table before a byte of it is
 * trusted: the kind must exist, the layer must be in range and per-layer or
 * global as declared, the format must be the tensor's natural format or MXFP4
 * where quantization is allowed, the extents must match exactly, and the
 * payload and scale byte counts must be exactly what that shape implies.
 */
static SparkStatus SparkK3ModuleValidateEntry(const SparkK3StagePackEntry *entry, uint64_t file_bytes)
{
	SparkK3StagePackTensorShape shape;
	uint32_t is_global = (entry->layer_index == SPARK_K3_STAGEPACK_GLOBAL_LAYER);
	uint32_t layer_index = is_global ? 0u : entry->layer_index;
	uint64_t payload_bytes,scale_bytes;
	if ( SparkK3StagePackResolvedShape(entry->tensor_kind,layer_index,&shape) < 0 )
	{
		fprintf(stderr,"k3_stage pack_entry_unknown kind=%u layer=%u\n",entry->tensor_kind,entry->layer_index);
		return(SPARK_STATUS_SCHEMA_ERROR);
	}
	if ( (shape.per_layer != 0u) == (is_global != 0u) )
	{
		fprintf(stderr,"k3_stage pack_entry_layer_scope kind=%u layer=%u per_layer=%u\n",entry->tensor_kind,entry->layer_index,shape.per_layer);
		return(SPARK_STATUS_SCHEMA_ERROR);
	}
	if ( is_global == 0u && entry->layer_index >= SPARK_K3_MODEL_LAYER_COUNT )
		return(SPARK_STATUS_SCHEMA_ERROR);
	if ( entry->rows != shape.rows || entry->columns != shape.columns )
	{
		fprintf(stderr,"k3_stage pack_entry_shape kind=%u layer=%u rows=%u expected=%u columns=%u expected=%u\n",entry->tensor_kind,entry->layer_index,entry->rows,shape.rows,entry->columns,shape.columns);
		return(SPARK_STATUS_SCHEMA_ERROR);
	}
	if ( entry->weight_format != shape.natural_format && !(shape.quantizable != 0u && entry->weight_format == SPARK_K3_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_MXFP4_E2M1) )
	{
		fprintf(stderr,"k3_stage pack_entry_format kind=%u layer=%u format=%u natural=%u quantizable=%u\n",entry->tensor_kind,entry->layer_index,entry->weight_format,shape.natural_format,shape.quantizable);
		return(SPARK_STATUS_SCHEMA_ERROR);
	}
	if ( entry->weight_format == SPARK_K3_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_MXFP4_E2M1 && entry->scale_group_size != SPARK_K3_MODEL_MXFP4_GROUP_SIZE )
		return(SPARK_STATUS_SCHEMA_ERROR);
	if ( entry->weight_format == SPARK_K3_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_MXFP4_E2M1 && (entry->columns % SPARK_K3_MODEL_MXFP4_GROUP_SIZE) != 0u )
		return(SPARK_STATUS_SCHEMA_ERROR);
	payload_bytes = SparkK3StagePackPayloadBytes(entry->weight_format,entry->rows,entry->columns);
	scale_bytes = SparkK3StagePackScaleBytes(entry->weight_format,entry->rows,entry->columns);
	if ( entry->payload_bytes != payload_bytes || entry->scale_bytes != scale_bytes )
	{
		fprintf(stderr,"k3_stage pack_entry_bytes kind=%u layer=%u payload=%llu expected=%llu scale=%llu expected=%llu\n",entry->tensor_kind,entry->layer_index,(unsigned long long)entry->payload_bytes,(unsigned long long)payload_bytes,(unsigned long long)entry->scale_bytes,(unsigned long long)scale_bytes);
		return(SPARK_STATUS_SCHEMA_ERROR);
	}
	if ( entry->payload_bytes > file_bytes || entry->payload_offset > file_bytes - entry->payload_bytes )
		return(SPARK_STATUS_SCHEMA_ERROR);
	if ( scale_bytes != 0u && (entry->scale_bytes > file_bytes || entry->scale_offset > file_bytes - entry->scale_bytes) )
		return(SPARK_STATUS_SCHEMA_ERROR);
	return(SPARK_STATUS_OK);
}

static void SparkK3ModuleFillLinearView(SparkK3Mxfp4LinearView *view, const SparkK3StagePackEntry *entry, void *payload, void *scale)
{
	view->abi_version = SPARK_K3_RESIDENT_DECODE_STAGE_MXFP4_LINEAR_VIEW_ABI_VERSION;
	view->weight_format = entry->weight_format;
	view->input_dimension = entry->columns;
	view->output_dimension = entry->rows;
	view->weight_payload = payload;
	view->weight_scale_e8m0 = (const uint8_t *)scale;
	view->weight_payload_bytes = entry->payload_bytes;
	view->weight_scale_bytes = entry->scale_bytes;
}

static SparkStatus SparkK3ModuleLoadTensor(SparkK3ModuleState *state, FILE *file, const SparkK3StagePackEntry *entry, uint64_t file_bytes)
{
	SparkK3ModuleTensorBinding binding;
	SparkStatus status;
	void *payload = 0;
	void *scale = 0;
	status = SparkK3ModuleValidateEntry(entry,file_bytes);
	if ( status != SPARK_STATUS_OK )
		return(status);
	status = SparkK3ModuleLoadDeviceRegion(state,file,entry->payload_offset,entry->payload_bytes,&payload);
	if ( status == SPARK_STATUS_OK && entry->scale_bytes != 0u )
		status = SparkK3ModuleLoadDeviceRegion(state,file,entry->scale_offset,entry->scale_bytes,&scale);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( SparkK3ModuleTensorIsExpertConcatenation(entry->tensor_kind) != 0u )
		return(SparkK3ModuleBindExpertConcatenation(state,entry,payload,scale));
	if ( SparkK3ModuleBindingForTensor(state,entry->tensor_kind,entry->layer_index,&binding) < 0 )
		return(SPARK_STATUS_SCHEMA_ERROR);
	if ( binding.view != 0 )
		SparkK3ModuleFillLinearView(binding.view,entry,payload,scale);
	else
		*binding.pointer_cell = payload;
	return(SPARK_STATUS_OK);
}

/*
 * Version 1 executes the whole stack on one node. A pack carrying a slice is
 * rejected rather than half-served: the hidden-state transport that a
 * pipelined K3 needs must also carry the AttnRes block array, which is a
 * protocol change tracked in DIFFERENCES.md, not something to fake here.
 */
static SparkStatus SparkK3ModuleValidateSliceIsWholeStack(const SparkK3StagePackHeader *header)
{
	if ( header->first_layer_index != 0u || header->layer_count != SPARK_K3_MODEL_LAYER_COUNT )
	{
		fprintf(stderr,"k3_stage pack_slice_unsupported first_layer=%u layers=%u expected_first=0 expected_layers=%u\n",header->first_layer_index,header->layer_count,SPARK_K3_MODEL_LAYER_COUNT);
		return(SPARK_STATUS_SCHEMA_ERROR);
	}
	return(SPARK_STATUS_OK);
}

/*
 * Load the pack: header first, then the geometry comparison, then every
 * directory entry. A geometry mismatch names the offending field and fails
 * the load; the driver never adapts to the pack.
 */
static SparkStatus SparkK3ModuleReadPackHeader(FILE *file, const char *path, SparkK3StagePackHeader *header)
{
	SparkK3StagePackHeader expected;
	SparkStatus status;
	int32_t mismatch;
	status = SparkK3ModulePackRead(file,0u,header,sizeof(*header));
	if ( status != SPARK_STATUS_OK )
		return(status);
	SparkK3StagePackExpectedGeometry(&expected,header->first_layer_index,header->layer_count,header->tensor_count);
	mismatch = SparkK3StagePackCompareGeometry(header,&expected);
	if ( mismatch != 0 )
	{
		fprintf(stderr,"k3_stage pack_geometry_mismatch field=%s path=%s\n",SparkK3StagePackGeometryFieldName(mismatch),path);
		return(SPARK_STATUS_SCHEMA_ERROR);
	}
	status = SparkK3ModuleValidateSliceIsWholeStack(header);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( ((uint64_t)header->tensor_count * sizeof(SparkK3StagePackEntry)) > header->file_bytes || header->directory_offset > header->file_bytes - ((uint64_t)header->tensor_count * sizeof(SparkK3StagePackEntry)) )
	{
		fprintf(stderr,"k3_stage pack_directory_bounds offset=%llu tensors=%u file=%llu\n",(unsigned long long)header->directory_offset,header->tensor_count,(unsigned long long)header->file_bytes);
		return(SPARK_STATUS_SCHEMA_ERROR);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkK3ModuleLoadPack(SparkK3ModuleState *state, const char *path)
{
	SparkK3StagePackHeader header;
	SparkK3StagePackEntry *directory;
	SparkStatus status;
	FILE *file;
	uint32_t index;
	file = fopen(path,"rb");
	if ( file == 0 )
	{
		fprintf(stderr,"k3_stage pack_open_failed path=%s\n",path);
		return(SPARK_STATUS_NOT_FOUND);
	}
	status = SparkK3ModuleReadPackHeader(file,path,&header);
	if ( status != SPARK_STATUS_OK )
	{
		fclose(file);
		return(status);
	}
	directory = (SparkK3StagePackEntry *)malloc((size_t)header.tensor_count * sizeof(SparkK3StagePackEntry));
	if ( directory == 0 )
	{
		fclose(file);
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	status = SparkK3ModulePackRead(file,header.directory_offset,directory,(uint64_t)header.tensor_count * sizeof(SparkK3StagePackEntry));
	for (index = 0; index < header.tensor_count && status == SPARK_STATUS_OK; index++)
		status = SparkK3ModuleLoadTensor(state,file,&directory[index],header.file_bytes);
	free(directory);
	fclose(file);
	if ( status != SPARK_STATUS_OK )
		return(status);
	fprintf(stderr,"k3_stage pack_loaded path=%s tensors=%u device_bytes=%llu\n",path,header.tensor_count,(unsigned long long)state->device_bytes_resident);
	return(SPARK_STATUS_OK);
}

/*
 * Every tensor the layer stack will dereference must have arrived. A pack
 * that is missing a tensor fails here rather than at the first launch that
 * reads a null pointer, and rather than running with a silently absent term.
 */
static SparkStatus SparkK3ModuleValidateLayerWeights(const SparkK3ModuleState *state, uint32_t layer_index)
{
	const SparkK3KdaLayerWeights *kda = &state->kda_weights[layer_index];
	const SparkK3MlaLayerWeights *mla = &state->mla_weights[layer_index];
	const SparkK3MoeLayerWeights *moe = &state->moe_weights[layer_index];
	uint32_t site_base = layer_index * SPARK_K3_RESIDENT_DECODE_STAGE_ATTNRES_SITES_PER_LAYER;
	if ( state->attention_norm_by_layer[layer_index] == 0 || state->mlp_norm_by_layer[layer_index] == 0 )
		return(SPARK_STATUS_SCHEMA_ERROR);
	if ( state->attnres_sites[site_base].pseudo_query_bf16 == 0 || state->attnres_sites[site_base].key_norm_weight_bf16 == 0 )
		return(SPARK_STATUS_SCHEMA_ERROR);
	if ( state->attnres_sites[site_base + 1u].pseudo_query_bf16 == 0 || state->attnres_sites[site_base + 1u].key_norm_weight_bf16 == 0 )
		return(SPARK_STATUS_SCHEMA_ERROR);
	if ( SPARK_K3_MODEL_LAYER_IS_KDA(layer_index) != 0u )
	{
		if ( kda->query.weight_payload == 0 || kda->key.weight_payload == 0 || kda->value.weight_payload == 0 || kda->output.weight_payload == 0 )
			return(SPARK_STATUS_SCHEMA_ERROR);
		if ( kda->decay_low.weight_payload == 0 || kda->decay_high.weight_payload == 0 || kda->beta.weight_payload == 0 )
			return(SPARK_STATUS_SCHEMA_ERROR);
		if ( kda->output_gate_low.weight_payload == 0 || kda->output_gate_high.weight_payload == 0 || kda->head_norm_weight_bf16 == 0 )
			return(SPARK_STATUS_SCHEMA_ERROR);
	}
	else
	{
		if ( mla->query_a.weight_payload == 0 || mla->query_b.weight_payload == 0 || mla->kv_a.weight_payload == 0 || mla->kv_b.weight_payload == 0 )
			return(SPARK_STATUS_SCHEMA_ERROR);
		if ( mla->output.weight_payload == 0 || mla->head_gate.weight_payload == 0 || mla->query_a_norm_weight_bf16 == 0 || mla->kv_a_norm_weight_bf16 == 0 )
			return(SPARK_STATUS_SCHEMA_ERROR);
	}
	if ( moe->shared_gate.weight_payload == 0 || moe->shared_up.weight_payload == 0 || moe->shared_down.weight_payload == 0 )
		return(SPARK_STATUS_SCHEMA_ERROR);
	if ( layer_index < SPARK_K3_MODEL_FIRST_ROUTED_LAYER )
		return(SPARK_STATUS_OK);
	if ( moe->router_weight_bf16 == 0 || moe->router_score_bias_f32 == 0 )
		return(SPARK_STATUS_SCHEMA_ERROR);
	if ( moe->expert_gate_payload == 0 || moe->expert_up_payload == 0 || moe->expert_down_payload == 0 )
		return(SPARK_STATUS_SCHEMA_ERROR);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkK3ModuleValidateResidentWeights(const SparkK3ModuleState *state)
{
	SparkStatus status;
	uint32_t layer_index;
	if ( state->node_context.token_embedding_bf16 == 0 || state->node_context.final_norm_weight_bf16 == 0 )
		return(SPARK_STATUS_SCHEMA_ERROR);
	if ( state->node_context.restricted_lm_head_weight_bf16 == 0 || state->node_context.restricted_token_ids == 0 )
		return(SPARK_STATUS_SCHEMA_ERROR);
	if ( state->attnres_final_site.pseudo_query_bf16 == 0 || state->attnres_final_site.key_norm_weight_bf16 == 0 )
		return(SPARK_STATUS_SCHEMA_ERROR);
	for (layer_index = 0; layer_index < SPARK_K3_MODEL_LAYER_COUNT; layer_index++)
	{
		status = SparkK3ModuleValidateLayerWeights(state,layer_index);
		if ( status != SPARK_STATUS_OK )
		{
			fprintf(stderr,"k3_stage weights_incomplete layer=%u kind=%s\n",layer_index,SPARK_K3_MODEL_LAYER_IS_KDA(layer_index) != 0u ? "kda" : "mla");
			return(status);
		}
	}
	return(SPARK_STATUS_OK);
}

// Attention alternates KDA and gated MLA on a fixed period; each kind numbers
// its own layers so the state pool and the latent cache stay dense.
static void SparkK3ModuleAssignLayerOrdinals(SparkK3ModuleState *state)
{
	uint32_t layer_index;
	state->kda_layer_count = 0u;
	state->mla_layer_count = 0u;
	for (layer_index = 0; layer_index < SPARK_K3_MODEL_LAYER_COUNT; layer_index++)
	{
		if ( SPARK_K3_MODEL_LAYER_IS_KDA(layer_index) != 0u )
		{
			state->kda_ordinal_by_layer[layer_index] = state->kda_layer_count;
			state->kda_layer_count++;
		}
		else
		{
			state->mla_ordinal_by_layer[layer_index] = state->mla_layer_count;
			state->mla_layer_count++;
		}
	}
}

/*
 * The KDA state is the whole history: one dk by dv fp32 matrix per head, per
 * layer, per lane, carried across dispatches. state_valid_by_lane is the cold
 * flag the decode kernel reads to zero a lane's matrix on first touch instead
 * of paying a memset over the pool.
 */
static SparkStatus SparkK3ModuleAllocateKdaStatePool(SparkK3ModuleState *state)
{
	SparkK3KdaStatePool *pool = &state->node_context.kda_state_pool;
	SparkStatus status;
	uint64_t layer_stride = SPARK_K3_MODEL_KDA_STATE_ELEMENTS_PER_LAYER;
	uint64_t lane_stride = layer_stride * (uint64_t)state->kda_layer_count;
	void *pointer = 0;
	pool->abi_version = SPARK_K3_RESIDENT_DECODE_STAGE_KDA_STATE_POOL_ABI_VERSION;
	pool->lane_capacity = state->lane_capacity;
	pool->kda_layer_count = state->kda_layer_count;
	pool->lane_stride_elements = lane_stride;
	pool->layer_stride_elements = layer_stride;
	status = SparkK3ModuleDeviceAllocate(state,lane_stride * (uint64_t)state->lane_capacity * sizeof(float),&pointer);
	if ( status != SPARK_STATUS_OK )
		return(status);
	pool->state_f32 = (float *)pointer;
	status = SparkK3ModuleDeviceAllocateZeroed(state,(uint64_t)state->row_capacity * sizeof(uint32_t),&pointer);
	if ( status != SPARK_STATUS_OK )
		return(status);
	pool->state_cold_by_row = (uint32_t *)pointer;
	fprintf(stderr,"k3_stage kda_state_pool lanes=%u layers=%u bytes=%llu\n",state->lane_capacity,state->kda_layer_count,(unsigned long long)(lane_stride * (uint64_t)state->lane_capacity * sizeof(float)));
	return(SPARK_STATUS_OK);
}

/*
 * The module owns the latent cache and hands each lane a contiguous run of
 * physical blocks, plus one scratch block that absorbs the writes of padded
 * prefill rows. A serving layer that wants shared prefixes supplies its own
 * table through the frame context instead.
 */
static SparkStatus SparkK3ModuleAllocateMlaCache(SparkK3ModuleState *state)
{
	SparkK3MlaBlockTableView *table = &state->owned_block_table;
	SparkStatus status;
	uint64_t cache_elements;
	uint32_t lane,block;
	void *pointer = 0;
	state->blocks_per_lane = (state->max_context_tokens + SPARK_K3_RESIDENT_DECODE_STAGE_MLA_BLOCK_TOKENS - 1u) / SPARK_K3_RESIDENT_DECODE_STAGE_MLA_BLOCK_TOKENS;
	state->node_context.mla_cache_block_count = (state->lane_capacity * state->blocks_per_lane) + 1u;
	state->scratch_block_index = state->node_context.mla_cache_block_count - 1u;
	cache_elements = (uint64_t)state->mla_layer_count * state->node_context.mla_cache_block_count * SPARK_K3_RESIDENT_DECODE_STAGE_MLA_BLOCK_TOKENS * SPARK_K3_MODEL_MLA_CACHE_TOKEN_ELEMENTS;
	status = SparkK3ModuleDeviceAllocate(state,cache_elements * SPARK_K3_MODEL_BF16_ELEMENT_BYTES,&pointer);
	if ( status != SPARK_STATUS_OK )
		return(status);
	state->node_context.mla_cache_bf16 = pointer;
	state->host_block_indices = (uint32_t *)malloc((size_t)state->lane_capacity * state->blocks_per_lane * sizeof(uint32_t));
	state->host_lane_block_counts = (uint32_t *)malloc((size_t)state->lane_capacity * sizeof(uint32_t));
	if ( state->host_block_indices == 0 || state->host_lane_block_counts == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	for (lane = 0; lane < state->lane_capacity; lane++)
	{
		for (block = 0; block < state->blocks_per_lane; block++)
			state->host_block_indices[(lane * state->blocks_per_lane) + block] = (lane * state->blocks_per_lane) + block;
		state->host_lane_block_counts[lane] = state->blocks_per_lane;
	}
	table->abi_version = SPARK_K3_RESIDENT_DECODE_STAGE_MLA_BLOCK_TABLE_ABI_VERSION;
	table->descriptor_bytes = (uint32_t)sizeof(SparkK3MlaBlockTableView);
	table->block_token_count = SPARK_K3_RESIDENT_DECODE_STAGE_MLA_BLOCK_TOKENS;
	table->lane_count = state->lane_capacity;
	table->lane_stride = state->blocks_per_lane;
	table->lane_capacity = state->lane_capacity;
	table->host_physical_block_indices = state->host_block_indices;
	table->host_lane_physical_block_counts = state->host_lane_block_counts;
	status = SparkK3ModuleDeviceAllocate(state,(uint64_t)state->lane_capacity * state->blocks_per_lane * sizeof(uint32_t),&pointer);
	if ( status != SPARK_STATUS_OK )
		return(status);
	table->physical_block_indices = (const uint32_t *)pointer;
	status = SparkK3ModuleCudaStatus(cudaMemcpy(pointer,state->host_block_indices,(size_t)state->lane_capacity * state->blocks_per_lane * sizeof(uint32_t),cudaMemcpyHostToDevice),"block_table_h2d");
	if ( status != SPARK_STATUS_OK )
		return(status);
	status = SparkK3ModuleDeviceAllocate(state,(uint64_t)state->lane_capacity * sizeof(uint32_t),&pointer);
	if ( status != SPARK_STATUS_OK )
		return(status);
	table->lane_physical_block_counts = (const uint32_t *)pointer;
	status = SparkK3ModuleCudaStatus(cudaMemcpy(pointer,state->host_lane_block_counts,(size_t)state->lane_capacity * sizeof(uint32_t),cudaMemcpyHostToDevice),"block_counts_h2d");
	if ( status != SPARK_STATUS_OK )
		return(status);
	fprintf(stderr,"k3_stage mla_cache layers=%u blocks=%u tokens_per_lane=%u bytes=%llu\n",state->mla_layer_count,state->node_context.mla_cache_block_count,state->blocks_per_lane * SPARK_K3_RESIDENT_DECODE_STAGE_MLA_BLOCK_TOKENS,(unsigned long long)(cache_elements * SPARK_K3_MODEL_BF16_ELEMENT_BYTES));
	return(SPARK_STATUS_OK);
}

/*
 * Per-slot activation buffers. Every buffer is sized for row_capacity rows,
 * which is also the representation stride the launchers compute from
 * max_active_sequence_count, so the two can never disagree.
 */
static SparkStatus SparkK3ModuleAllocateSlotBuffers(SparkK3ModuleState *state, SparkK3PipelineSlot *slot)
{
	uint64_t rows = state->row_capacity;
	SparkStatus status;
	void *pointer = 0;
	status = SparkK3ModuleDeviceAllocateZeroed(state,rows * sizeof(uint32_t),&pointer);
	if ( status == SPARK_STATUS_OK )
		slot->input_token_ids = (const uint32_t *)pointer;
	if ( status == SPARK_STATUS_OK )
		status = SparkK3ModuleDeviceAllocateZeroed(state,rows * sizeof(uint32_t),&pointer);
	if ( status == SPARK_STATUS_OK )
		slot->output_token_ids = (uint32_t *)pointer;
	if ( status == SPARK_STATUS_OK )
		status = SparkK3ModuleDeviceAllocateZeroed(state,rows * sizeof(uint32_t),&pointer);
	if ( status == SPARK_STATUS_OK )
		slot->slot_mapping = (const uint32_t *)pointer;
	if ( status == SPARK_STATUS_OK )
		status = SparkK3ModuleDeviceAllocateZeroed(state,rows * sizeof(uint32_t),&pointer);
	if ( status == SPARK_STATUS_OK )
		slot->lane_indices = (const uint32_t *)pointer;
	if ( status == SPARK_STATUS_OK )
		status = SparkK3ModuleDeviceAllocateZeroed(state,rows * sizeof(uint32_t),&pointer);
	if ( status == SPARK_STATUS_OK )
		slot->context_lengths = (const uint32_t *)pointer;
	if ( status == SPARK_STATUS_OK )
		status = SparkK3ModuleDeviceAllocateZeroed(state,(uint64_t)SPARK_K3_MODEL_ATTNRES_MAX_REPRESENTATIONS * rows * SPARK_K3_MODEL_HIDDEN_BF16_BYTES,&pointer);
	if ( status == SPARK_STATUS_OK )
		slot->attnres_representations_bf16 = pointer;
	if ( status == SPARK_STATUS_OK )
		status = SparkK3ModuleDeviceAllocateZeroed(state,rows * SPARK_K3_MODEL_HIDDEN_BF16_BYTES,&pointer);
	if ( status == SPARK_STATUS_OK )
		slot->mixed_hidden_bf16 = pointer;
	if ( status == SPARK_STATUS_OK )
		status = SparkK3ModuleDeviceAllocateZeroed(state,rows * SPARK_K3_MODEL_HIDDEN_BF16_BYTES,&pointer);
	if ( status == SPARK_STATUS_OK )
		slot->normalized_hidden_bf16 = pointer;
	if ( status == SPARK_STATUS_OK )
		status = SparkK3ModuleDeviceAllocateZeroed(state,rows * SPARK_K3_MODEL_HIDDEN_BF16_BYTES,&pointer);
	if ( status == SPARK_STATUS_OK )
		slot->attention_output_hidden_bf16 = pointer;
	if ( status == SPARK_STATUS_OK )
		status = SparkK3ModuleDeviceAllocateZeroed(state,rows * SPARK_K3_MODEL_HIDDEN_BF16_BYTES,&pointer);
	if ( status == SPARK_STATUS_OK )
		slot->moe_output_hidden_bf16 = pointer;
	return(status);
}

static SparkStatus SparkK3ModuleAllocateSlotKdaBuffers(SparkK3ModuleState *state, SparkK3PipelineSlot *slot)
{
	uint64_t rows = state->row_capacity;
	SparkStatus status;
	void *pointer = 0;
	status = SparkK3ModuleDeviceAllocateZeroed(state,rows * SPARK_K3_MODEL_KDA_QK_DIMENSION * SPARK_K3_MODEL_BF16_ELEMENT_BYTES,&pointer);
	if ( status == SPARK_STATUS_OK )
		slot->kda_query_bf16 = pointer;
	if ( status == SPARK_STATUS_OK )
		status = SparkK3ModuleDeviceAllocateZeroed(state,rows * SPARK_K3_MODEL_KDA_QK_DIMENSION * SPARK_K3_MODEL_BF16_ELEMENT_BYTES,&pointer);
	if ( status == SPARK_STATUS_OK )
		slot->kda_key_bf16 = pointer;
	if ( status == SPARK_STATUS_OK )
		status = SparkK3ModuleDeviceAllocateZeroed(state,rows * SPARK_K3_MODEL_KDA_VALUE_DIMENSION * SPARK_K3_MODEL_BF16_ELEMENT_BYTES,&pointer);
	if ( status == SPARK_STATUS_OK )
		slot->kda_value_bf16 = pointer;
	if ( status == SPARK_STATUS_OK )
		status = SparkK3ModuleDeviceAllocateZeroed(state,rows * SPARK_K3_MODEL_KDA_QK_DIMENSION * SPARK_K3_MODEL_BF16_ELEMENT_BYTES,&pointer);
	if ( status == SPARK_STATUS_OK )
		slot->kda_log_decay_bf16 = pointer;
	if ( status == SPARK_STATUS_OK )
		status = SparkK3ModuleDeviceAllocateZeroed(state,rows * SPARK_K3_MODEL_KDA_HEAD_COUNT * SPARK_K3_MODEL_BF16_ELEMENT_BYTES,&pointer);
	if ( status == SPARK_STATUS_OK )
		slot->kda_beta_bf16 = pointer;
	if ( status == SPARK_STATUS_OK )
		status = SparkK3ModuleDeviceAllocateZeroed(state,rows * SPARK_K3_MODEL_KDA_LOW_RANK_DIMENSION * SPARK_K3_MODEL_BF16_ELEMENT_BYTES,&pointer);
	if ( status == SPARK_STATUS_OK )
		slot->kda_low_rank_bf16 = pointer;
	if ( status == SPARK_STATUS_OK )
		status = SparkK3ModuleDeviceAllocateZeroed(state,rows * SPARK_K3_MODEL_KDA_VALUE_DIMENSION * SPARK_K3_MODEL_BF16_ELEMENT_BYTES,&pointer);
	if ( status == SPARK_STATUS_OK )
		slot->kda_core_output_bf16 = pointer;
	if ( status == SPARK_STATUS_OK )
		status = SparkK3ModuleDeviceAllocateZeroed(state,rows * SPARK_K3_MODEL_KDA_VALUE_DIMENSION * SPARK_K3_MODEL_BF16_ELEMENT_BYTES,&pointer);
	if ( status == SPARK_STATUS_OK )
		slot->kda_gate_bf16 = pointer;
	return(status);
}

static SparkStatus SparkK3ModuleAllocateSlotMlaBuffers(SparkK3ModuleState *state, SparkK3PipelineSlot *slot)
{
	uint64_t rows = state->row_capacity;
	uint64_t latent_elements = (uint64_t)SPARK_K3_MODEL_MLA_HEAD_COUNT * SPARK_K3_MODEL_MLA_LATENT_DIMENSION;
	SparkStatus status;
	void *pointer = 0;
	status = SparkK3ModuleDeviceAllocateZeroed(state,rows * SPARK_K3_MODEL_MLA_QUERY_A_DIMENSION * SPARK_K3_MODEL_BF16_ELEMENT_BYTES,&pointer);
	if ( status == SPARK_STATUS_OK )
		slot->mla_query_a_bf16 = pointer;
	if ( status == SPARK_STATUS_OK )
		status = SparkK3ModuleDeviceAllocateZeroed(state,rows * SPARK_K3_MODEL_MLA_QUERY_B_DIMENSION * SPARK_K3_MODEL_BF16_ELEMENT_BYTES,&pointer);
	if ( status == SPARK_STATUS_OK )
		slot->mla_query_b_bf16 = pointer;
	if ( status == SPARK_STATUS_OK )
		status = SparkK3ModuleDeviceAllocateZeroed(state,rows * latent_elements * SPARK_K3_MODEL_BF16_ELEMENT_BYTES,&pointer);
	if ( status == SPARK_STATUS_OK )
		slot->mla_query_latent_bf16 = pointer;
	if ( status == SPARK_STATUS_OK )
		status = SparkK3ModuleDeviceAllocateZeroed(state,rows * SPARK_K3_MODEL_MLA_KV_A_DIMENSION * SPARK_K3_MODEL_BF16_ELEMENT_BYTES,&pointer);
	if ( status == SPARK_STATUS_OK )
		slot->mla_kv_a_bf16 = pointer;
	if ( status == SPARK_STATUS_OK )
		status = SparkK3ModuleDeviceAllocateZeroed(state,rows * latent_elements * SPARK_K3_MODEL_BF16_ELEMENT_BYTES,&pointer);
	if ( status == SPARK_STATUS_OK )
		slot->mla_attention_latent_bf16 = pointer;
	if ( status == SPARK_STATUS_OK )
		status = SparkK3ModuleDeviceAllocateZeroed(state,rows * SPARK_K3_MODEL_MLA_ATTENTION_PROJECTION_DIMENSION * SPARK_K3_MODEL_BF16_ELEMENT_BYTES,&pointer);
	if ( status == SPARK_STATUS_OK )
		slot->mla_head_output_bf16 = pointer;
	if ( status == SPARK_STATUS_OK )
		status = SparkK3ModuleDeviceAllocateZeroed(state,rows * SPARK_K3_MODEL_MLA_HEAD_COUNT * SPARK_K3_MODEL_BF16_ELEMENT_BYTES,&pointer);
	if ( status == SPARK_STATUS_OK )
		slot->mla_head_gate_bf16 = pointer;
	return(status);
}

static SparkStatus SparkK3ModuleAllocateSlotMoeBuffers(SparkK3ModuleState *state, SparkK3PipelineSlot *slot)
{
	uint64_t rows = state->row_capacity;
	SparkStatus status;
	void *pointer = 0;
	status = SparkK3ModuleDeviceAllocateZeroed(state,rows * SPARK_K3_MODEL_MOE_TOP_K * sizeof(uint32_t),&pointer);
	if ( status == SPARK_STATUS_OK )
		slot->moe_topk_expert_ids = (uint32_t *)pointer;
	if ( status == SPARK_STATUS_OK )
		status = SparkK3ModuleDeviceAllocateZeroed(state,rows * SPARK_K3_MODEL_MOE_TOP_K * sizeof(float),&pointer);
	if ( status == SPARK_STATUS_OK )
		slot->moe_topk_weights_f32 = (float *)pointer;
	if ( status == SPARK_STATUS_OK )
		status = SparkK3ModuleDeviceAllocateZeroed(state,rows * SPARK_K3_MODEL_MOE_EXPERT_COUNT * SPARK_K3_MODEL_BF16_ELEMENT_BYTES,&pointer);
	if ( status == SPARK_STATUS_OK )
		slot->moe_gate_bf16 = pointer;
	if ( status == SPARK_STATUS_OK )
		status = SparkK3ModuleDeviceAllocateZeroed(state,rows * SPARK_K3_MODULE_MAX_MOE_INTERMEDIATE_ROW_ELEMENTS * SPARK_K3_MODEL_BF16_ELEMENT_BYTES,&pointer);
	if ( status == SPARK_STATUS_OK )
		slot->moe_intermediate_bf16 = pointer;
	if ( status == SPARK_STATUS_OK )
		status = SparkK3ModuleDeviceAllocateZeroed(state,rows * SPARK_K3_MODEL_RESTRICTED_VOCAB_COUNT * sizeof(float),&pointer);
	if ( status == SPARK_STATUS_OK )
		slot->restricted_logits_f32 = (float *)pointer;
	return(status);
}

static SparkStatus SparkK3ModuleAllocatePipelineSlots(SparkK3ModuleState *state)
{
	SparkStatus status = SPARK_STATUS_OK;
	cudaStream_t stream;
	uint32_t index;
	for (index = 0; index < state->node_context.pipeline_slot_count && status == SPARK_STATUS_OK; index++)
	{
		status = SparkK3ModuleCudaStatus(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"cudaStreamCreate");
		if ( status != SPARK_STATUS_OK )
			return(status);
		state->pipeline_slots[index].cuda_stream = stream;
		status = SparkK3ModuleAllocateSlotBuffers(state,&state->pipeline_slots[index]);
		if ( status == SPARK_STATUS_OK )
			status = SparkK3ModuleAllocateSlotKdaBuffers(state,&state->pipeline_slots[index]);
		if ( status == SPARK_STATUS_OK )
			status = SparkK3ModuleAllocateSlotMlaBuffers(state,&state->pipeline_slots[index]);
		if ( status == SPARK_STATUS_OK )
			status = SparkK3ModuleAllocateSlotMoeBuffers(state,&state->pipeline_slots[index]);
	}
	return(status);
}

static SparkStatus SparkK3ModuleAllocateSlotStaging(SparkK3ModuleState *state, SparkK3ModuleSlotStaging *staging)
{
	void *pointer = 0;
	SparkStatus status;
	staging->row_token_ids = (uint32_t *)malloc((size_t)state->row_capacity * sizeof(uint32_t));
	staging->row_slot_mapping = (uint32_t *)malloc((size_t)state->row_capacity * sizeof(uint32_t));
	staging->row_lane_indices = (uint32_t *)malloc((size_t)state->row_capacity * sizeof(uint32_t));
	staging->row_context_lengths = (uint32_t *)malloc((size_t)state->row_capacity * sizeof(uint32_t));
	staging->row_cold_flags = (uint32_t *)malloc((size_t)state->row_capacity * sizeof(uint32_t));
	staging->row_output_token_ids = (uint32_t *)malloc((size_t)state->row_capacity * sizeof(uint32_t));
	if ( staging->row_token_ids == 0 || staging->row_slot_mapping == 0 || staging->row_lane_indices == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( staging->row_context_lengths == 0 || staging->row_cold_flags == 0 || staging->row_output_token_ids == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	status = SparkK3ModuleDeviceAllocateZeroed(state,sizeof(int32_t),&pointer);
	if ( status != SPARK_STATUS_OK )
		return(status);
	staging->device_sequence_token_count = (int32_t *)pointer;
	atomic_init(&staging->busy,0u);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkK3ModuleAllocateHostStaging(SparkK3ModuleState *state)
{
	SparkStatus status = SPARK_STATUS_OK;
	uint32_t index;
	for (index = 0; index < state->node_context.pipeline_slot_count && status == SPARK_STATUS_OK; index++)
		status = SparkK3ModuleAllocateSlotStaging(state,&state->slot_staging[index]);
	return(status);
}

static void SparkK3ModuleConfigureNodeContext(SparkK3ModuleState *state)
{
	SparkK3ResidentDecodeStageNodeContext *node = &state->node_context;
	node->abi_version = SPARK_K3_RESIDENT_DECODE_STAGE_NODE_CONTEXT_ABI_VERSION;
	node->max_active_sequence_count = state->row_capacity;
	node->max_prefill_tokens = SPARK_K3_MODEL_KDA_CHUNK_TOKENS;
	node->first_layer_index = 0u;
	node->layer_count = SPARK_K3_MODEL_LAYER_COUNT;
	node->owns_embedding = 1u;
	node->owns_final_head = 1u;
	node->rms_norm_epsilon = SPARK_K3_MODEL_RMS_NORM_EPSILON;
	node->moe_routed_scaling_factor = SPARK_K3_MODEL_MOE_ROUTED_SCALING_FACTOR;
	node->moe_norm_topk_prob = SPARK_K3_MODEL_MOE_NORM_TOPK_PROB;
	node->enable_cuda_graph_replay = 0u;
	node->attnres_sites_by_layer = state->attnres_sites;
	node->attnres_final_site = &state->attnres_final_site;
	node->attention_norm_weights_by_layer_bf16 = state->attention_norm_by_layer;
	node->mlp_norm_weights_by_layer_bf16 = state->mlp_norm_by_layer;
	node->kda_weights_by_layer = state->kda_weights;
	node->mla_weights_by_layer = state->mla_weights;
	node->moe_weights_by_layer = state->moe_weights;
	node->restricted_vocab_count = SPARK_K3_MODEL_RESTRICTED_VOCAB_COUNT;
	node->pipeline_slots = state->pipeline_slots;
}

static SparkStatus SparkK3ModuleReadConfiguration(SparkK3ModuleState *state, const char **pack_path)
{
	SparkStatus status;
	uint32_t slots;
	status = SparkK3ModuleEnvironmentText("K3_STAGE_PACK",pack_path);
	if ( status == SPARK_STATUS_OK )
		status = SparkK3ModuleEnvironmentUnsigned("K3_MAX_LANES",1u,SPARK_K3_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,&state->lane_capacity);
	if ( status == SPARK_STATUS_OK )
		status = SparkK3ModuleEnvironmentUnsigned("K3_MAX_CONTEXT_TOKENS",SPARK_K3_RESIDENT_DECODE_STAGE_MLA_BLOCK_TOKENS,SPARK_K3_MODEL_MAXIMUM_CONTEXT_TOKENS,&state->max_context_tokens);
	if ( status == SPARK_STATUS_OK )
		status = SparkK3ModuleEnvironmentUnsigned("K3_PIPELINE_SLOTS",1u,SPARK_K3_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT,&slots);
	if ( status != SPARK_STATUS_OK )
		return(status);
	state->node_context.pipeline_slot_count = slots;
	// A prefill dispatch pads its batch to a full KDA chunk, so the row
	// capacity that sizes every buffer is the larger of the lane count and
	// the chunk width. It is also the representation stride the launchers
	// derive from max_active_sequence_count.
	state->row_capacity = state->lane_capacity > SPARK_K3_MODEL_KDA_CHUNK_TOKENS ? state->lane_capacity : SPARK_K3_MODEL_KDA_CHUNK_TOKENS;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkK3ResidentDecodeStageInitialize(const SparkFirmwareModuleConfiguration *configuration, const SparkFirmwareModuleHostServices *host_services, void **module_state)
{
	SparkK3ModuleState *state;
	const char *pack_path = 0;
	SparkStatus status;
	if ( configuration == 0 || host_services == 0 || module_state == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( configuration->abi_version != SPARK_FIRMWARE_MODULE_ABI_VERSION )
	{
		fprintf(stderr,"k3_stage abi_mismatch module=%u host=%u\n",SPARK_FIRMWARE_MODULE_ABI_VERSION,configuration->abi_version);
		return(SPARK_STATUS_ABI_MISMATCH);
	}
	state = (SparkK3ModuleState *)calloc(1u,sizeof(SparkK3ModuleState));
	if ( state == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	state->host_services = *host_services;
	state->program_id = configuration->operation_index;
	status = SparkK3ModuleReadConfiguration(state,&pack_path);
	if ( status == SPARK_STATUS_OK )
	{
		SparkK3ModuleAssignLayerOrdinals(state);
		SparkK3ModuleConfigureNodeContext(state);
		status = SparkK3ModuleLoadPack(state,pack_path);
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkK3ModuleValidateResidentWeights(state);
	if ( status == SPARK_STATUS_OK )
		status = SparkK3ModuleAllocateKdaStatePool(state);
	if ( status == SPARK_STATUS_OK )
		status = SparkK3ModuleAllocateMlaCache(state);
	if ( status == SPARK_STATUS_OK )
		status = SparkK3ModuleAllocatePipelineSlots(state);
	if ( status == SPARK_STATUS_OK )
		status = SparkK3ModuleAllocateHostStaging(state);
	if ( status != SPARK_STATUS_OK )
	{
		SparkK3ResidentDecodeStageDestroy(state);
		return(status);
	}
	fprintf(stderr,"k3_stage initialized lanes=%u rows=%u context=%u slots=%u device_bytes=%llu\n",state->lane_capacity,state->row_capacity,state->max_context_tokens,state->node_context.pipeline_slot_count,(unsigned long long)state->device_bytes_resident);
	*module_state = state;
	return(SPARK_STATUS_OK);
}

/*
 * Row metadata for one dispatch. A decode row is one lane's next token, so
 * the row's cache slot follows that lane's own position. A prefill dispatch
 * puts many consecutive tokens of one lane in flight, and pads the batch to a
 * full KDA chunk: padded rows carry token id zero, park their latent writes
 * in the scratch block and attend only to themselves, so they compute
 * finite garbage that nothing reads instead of poisoning the cache.
 */
static uint32_t SparkK3ModuleCacheSlotForPosition(const SparkK3ModuleState *state, uint32_t lane, uint32_t position)
{
	uint32_t logical_block = position / SPARK_K3_RESIDENT_DECODE_STAGE_MLA_BLOCK_TOKENS;
	uint32_t physical_block = state->host_block_indices[(lane * state->blocks_per_lane) + logical_block];
	return((physical_block * SPARK_K3_RESIDENT_DECODE_STAGE_MLA_BLOCK_TOKENS) + (position % SPARK_K3_RESIDENT_DECODE_STAGE_MLA_BLOCK_TOKENS));
}

/*
 * Wire token ids feed the embedding gather directly, so an id outside the
 * vocabulary is an out-of-bounds device read; it is rejected here, on the
 * host, before anything is uploaded.
 */
static SparkStatus SparkK3ModuleValidateWireTokens(const uint32_t *token_ids, uint32_t token_count)
{
	uint32_t row;
	for (row = 0; row < token_count; row++)
		if ( token_ids[row] >= SPARK_K3_MODEL_OUTPUT_VOCAB_COUNT )
		{
			fprintf(stderr,"k3_stage token_out_of_vocab row=%u token=%u vocab=%u\n",row,token_ids[row],SPARK_K3_MODEL_OUTPUT_VOCAB_COUNT);
			return(SPARK_STATUS_INVALID_ARGUMENT);
		}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkK3ModuleFillRowMetadata(SparkK3ModuleState *state, SparkK3ModuleSlotStaging *staging, const SparkModelDriverFrame *frame, uint32_t lane, uint32_t token_count, uint32_t padded_rows)
{
	uint32_t base_position = (uint32_t)frame->sequence_position;
	uint32_t row,position;
	if ( (uint64_t)base_position + token_count > state->max_context_tokens )
	{
		fprintf(stderr,"k3_stage context_exhausted lane=%u position=%u tokens=%u capacity=%u\n",lane,base_position,token_count,state->max_context_tokens);
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	for (row = 0; row < padded_rows; row++)
	{
		position = base_position + row;
		staging->row_lane_indices[row] = lane;
		if ( row < token_count )
		{
			staging->row_slot_mapping[row] = SparkK3ModuleCacheSlotForPosition(state,lane,position);
			staging->row_context_lengths[row] = position + 1u;
		}
		else
		{
			staging->row_token_ids[row] = 0u;
			staging->row_slot_mapping[row] = (state->scratch_block_index * SPARK_K3_RESIDENT_DECODE_STAGE_MLA_BLOCK_TOKENS) + (row % SPARK_K3_RESIDENT_DECODE_STAGE_MLA_BLOCK_TOKENS);
			staging->row_context_lengths[row] = 1u;
		}
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkK3ModuleUploadRowMetadata(SparkK3ModuleSlotStaging *staging, const SparkK3PipelineSlot *slot, uint32_t padded_rows, uint32_t token_count, cudaStream_t stream)
{
	uint64_t bytes = (uint64_t)padded_rows * sizeof(uint32_t);
	SparkStatus status;
	staging->sequence_token_count = (int32_t)token_count;
	status = SparkK3ModuleCudaStatus(cudaMemcpyAsync((void *)slot->input_token_ids,staging->row_token_ids,(size_t)bytes,cudaMemcpyHostToDevice,stream),"token_ids_h2d");
	if ( status == SPARK_STATUS_OK )
		status = SparkK3ModuleCudaStatus(cudaMemcpyAsync((void *)slot->slot_mapping,staging->row_slot_mapping,(size_t)bytes,cudaMemcpyHostToDevice,stream),"slot_mapping_h2d");
	if ( status == SPARK_STATUS_OK )
		status = SparkK3ModuleCudaStatus(cudaMemcpyAsync((void *)slot->lane_indices,staging->row_lane_indices,(size_t)bytes,cudaMemcpyHostToDevice,stream),"lane_indices_h2d");
	if ( status == SPARK_STATUS_OK )
		status = SparkK3ModuleCudaStatus(cudaMemcpyAsync((void *)slot->context_lengths,staging->row_context_lengths,(size_t)bytes,cudaMemcpyHostToDevice,stream),"context_lengths_h2d");
	if ( status == SPARK_STATUS_OK )
		status = SparkK3ModuleCudaStatus(cudaMemcpyAsync(staging->device_sequence_token_count,&staging->sequence_token_count,sizeof(int32_t),cudaMemcpyHostToDevice,stream),"token_counts_h2d");
	return(status);
}

/*
 * A sequence at position zero has no history: its KDA state is stale from a
 * previous tenant of the lane and must be treated as zero on first touch. The
 * flag is per row because both KDA kernels read it per row.
 */
static SparkStatus SparkK3ModuleUpdateLaneState(SparkK3ModuleState *state, SparkK3ModuleSlotStaging *staging, const SparkModelDriverFrame *frame, uint32_t padded_rows, cudaStream_t stream)
{
	SparkK3KdaStatePool *pool = &state->node_context.kda_state_pool;
	uint32_t cold = (frame->sequence_position == 0u) ? 1u : 0u;
	uint32_t row;
	for (row = 0; row < padded_rows; row++)
		staging->row_cold_flags[row] = cold;
	return(SparkK3ModuleCudaStatus(cudaMemcpyAsync(pool->state_cold_by_row,staging->row_cold_flags,(size_t)padded_rows * sizeof(uint32_t),cudaMemcpyHostToDevice,stream),"cold_flags_h2d"));
}

/*
 * Every rowwise kernel in the layer stack runs at the padded width, so a
 * prefill's padded rows compute deterministic finite garbage end to end
 * instead of dragging stale device memory through the mixes; the true token
 * count matters only to the chunk kernel's masking (via the staged device
 * count) and to the host wire I/O in Execute.
 */
static SparkStatus SparkK3ModuleRunAttention(SparkK3ModuleState *state, const SparkK3PipelineSlot *slot, SparkK3ModuleSlotStaging *staging, const SparkK3MlaBlockTableView *block_table, uint32_t layer_index, uint32_t row_count, uint32_t is_prefill, uint32_t carry_state, cudaStream_t stream)
{
	const SparkK3ResidentDecodeStageNodeContext *node = &state->node_context;
	SparkStatus status;
	if ( SPARK_K3_MODEL_LAYER_IS_KDA(layer_index) == 0u )
	{
		if ( is_prefill != 0u )
			return(SparkK3LaunchMlaPrefill(node,slot,&state->mla_weights[layer_index],block_table,state->mla_ordinal_by_layer[layer_index],row_count,stream));
		return(SparkK3LaunchMlaDecode(node,slot,&state->mla_weights[layer_index],block_table,state->mla_ordinal_by_layer[layer_index],row_count,stream));
	}
	status = SparkK3LaunchKdaMaterialize(node,slot,&state->kda_weights[layer_index],row_count,stream);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( is_prefill != 0u )
		status = SparkK3LaunchKdaChunk(node,slot,state->kda_ordinal_by_layer[layer_index],1u,SPARK_K3_MODEL_KDA_CHUNK_TOKENS,staging->device_sequence_token_count,carry_state,1u,stream);
	else
		status = SparkK3LaunchKdaDecodeStep(node,slot,state->kda_ordinal_by_layer[layer_index],row_count,stream);
	if ( status != SPARK_STATUS_OK )
		return(status);
	return(SparkK3LaunchKdaFinish(node,slot,&state->kda_weights[layer_index],row_count,stream));
}

static SparkStatus SparkK3ModuleRunMlp(SparkK3ModuleState *state, const SparkK3PipelineSlot *slot, uint32_t layer_index, uint32_t row_count, cudaStream_t stream)
{
	const SparkK3ResidentDecodeStageNodeContext *node = &state->node_context;
	SparkStatus status;
	if ( layer_index < SPARK_K3_MODEL_FIRST_ROUTED_LAYER )
		return(SparkK3LaunchDenseMlp(node,slot,&state->moe_weights[layer_index],row_count,stream));
	status = SparkK3LaunchMoeRoute(node,slot,&state->moe_weights[layer_index],row_count,stream);
	if ( status != SPARK_STATUS_OK )
		return(status);
	return(SparkK3LaunchMoeExperts(node,slot,&state->moe_weights[layer_index],row_count,stream));
}

/*
 * One layer, in the order the published AttnRes forward() prescribes: mix the
 * completed blocks with the running partial, check the block boundary, run
 * attention on the normalized mixture, accumulate, mix again for the mlp
 * site, run the mlp, accumulate. Only the attention site can open a block.
 */
static SparkStatus SparkK3ModuleRunLayer(SparkK3ModuleState *state, const SparkK3PipelineSlot *slot, SparkK3ModuleSlotStaging *staging, const SparkK3MlaBlockTableView *block_table, uint32_t layer_index, uint32_t row_count, uint32_t is_prefill, uint32_t carry_state, cudaStream_t stream)
{
	const SparkK3ResidentDecodeStageNodeContext *node = &state->node_context;
	uint32_t site_base = layer_index * SPARK_K3_RESIDENT_DECODE_STAGE_ATTNRES_SITES_PER_LAYER;
	uint32_t completed = SPARK_K3_MODEL_ATTNRES_COMPLETED_BLOCKS_BEFORE_LAYER(layer_index);
	uint32_t opens = SPARK_K3_MODEL_ATTNRES_LAYER_OPENS_BLOCK(layer_index);
	SparkStatus status;
	status = SparkK3LaunchAttnResMix(node,slot,&state->attnres_sites[site_base + SPARK_K3_RESIDENT_DECODE_STAGE_ATTNRES_ATTENTION_SITE],completed + 1u,row_count,stream);
	if ( status == SPARK_STATUS_OK )
		status = SparkK3LaunchRmsNorm(slot->mixed_hidden_bf16,state->attention_norm_by_layer[layer_index],slot->normalized_hidden_bf16,row_count,SPARK_K3_MODEL_HIDDEN_DIMENSION,node->rms_norm_epsilon,stream);
	if ( status == SPARK_STATUS_OK )
		status = SparkK3ModuleRunAttention(state,slot,staging,block_table,layer_index,row_count,is_prefill,carry_state,stream);
	if ( status == SPARK_STATUS_OK )
		status = SparkK3LaunchAttnResAccumulate(node,slot,slot->attention_output_hidden_bf16,opens,completed,row_count,stream);
	if ( status == SPARK_STATUS_OK )
		status = SparkK3LaunchAttnResMix(node,slot,&state->attnres_sites[site_base + SPARK_K3_RESIDENT_DECODE_STAGE_ATTNRES_MLP_SITE],completed + opens + 1u,row_count,stream);
	if ( status == SPARK_STATUS_OK )
		status = SparkK3LaunchRmsNorm(slot->mixed_hidden_bf16,state->mlp_norm_by_layer[layer_index],slot->normalized_hidden_bf16,row_count,SPARK_K3_MODEL_HIDDEN_DIMENSION,node->rms_norm_epsilon,stream);
	if ( status == SPARK_STATUS_OK )
		status = SparkK3ModuleRunMlp(state,slot,layer_index,row_count,stream);
	if ( status == SPARK_STATUS_OK )
		status = SparkK3LaunchAttnResAccumulate(node,slot,slot->moe_output_hidden_bf16,0u,completed + opens,row_count,stream);
	return(status);
}

static SparkStatus SparkK3ModuleRunStage(SparkK3ModuleState *state, const SparkK3PipelineSlot *slot, SparkK3ModuleSlotStaging *staging, const SparkK3MlaBlockTableView *block_table, uint32_t row_count, uint32_t is_prefill, uint32_t carry_state, cudaStream_t stream)
{
	const SparkK3ResidentDecodeStageNodeContext *node = &state->node_context;
	uint32_t final_completed = SPARK_K3_MODEL_ATTNRES_COMPLETED_BLOCKS_BEFORE_LAYER(SPARK_K3_MODEL_LAYER_COUNT - 1u) + SPARK_K3_MODEL_ATTNRES_LAYER_OPENS_BLOCK(SPARK_K3_MODEL_LAYER_COUNT - 1u);
	SparkStatus status;
	uint32_t layer_index;
	status = SparkK3LaunchEmbeddingGather(node,slot,row_count,stream);
	for (layer_index = 0; layer_index < SPARK_K3_MODEL_LAYER_COUNT && status == SPARK_STATUS_OK; layer_index++)
		status = SparkK3ModuleRunLayer(state,slot,staging,block_table,layer_index,row_count,is_prefill,carry_state,stream);
	if ( status == SPARK_STATUS_OK )
		status = SparkK3LaunchAttnResMix(node,slot,&state->attnres_final_site,final_completed + 1u,row_count,stream);
	if ( status == SPARK_STATUS_OK )
		status = SparkK3LaunchRestrictedLogits(node,slot,row_count,stream);
	return(status);
}

/*
 * Frame contract for version 1. One frame is one sequence's dispatch:
 *   driver_dispatch_slot  the lane, which owns the KDA state and cache blocks
 *   buffers[0]            host, read, new_token_count token ids
 *   buffers[1]            host, write, new_token_count sampled token ids
 * A decode frame carries exactly one token; a prefill frame carries up to one
 * KDA chunk and is padded to the chunk width inside this module. Batching
 * several sequences into one dispatch needs per-row lane arrays on the wire
 * and a completion that can name more than one sequence, so it is refused
 * here rather than approximated. See DIFFERENCES.md.
 */
static SparkStatus SparkK3ModuleValidateFrameShape(const SparkK3ModuleState *state, const SparkModelDriverFrame *frame, uint32_t *is_prefill, uint32_t *rows)
{
	uint32_t prefill = (frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL) != 0u ? 1u : 0u;
	if ( frame->program_id == 0u || frame->buffer_count != 2u || frame->buffers == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( (frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_DRIVER_DISPATCH_SLOT_VALID) == 0u || frame->driver_dispatch_slot >= state->lane_capacity )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( frame->active_slot_count == 0u || frame->active_slot_count > state->lane_capacity )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( frame->sequence_position > (uint64_t)state->max_context_tokens )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( frame->new_token_count == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( prefill != 0u && frame->new_token_count > SPARK_K3_MODEL_KDA_CHUNK_TOKENS )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( prefill == 0u && frame->new_token_count != 1u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( (frame->buffers[0].flags & SPARK_MODEL_DRIVER_BUFFER_FLAG_READ) == 0u || frame->buffers[0].address == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( (frame->buffers[1].flags & SPARK_MODEL_DRIVER_BUFFER_FLAG_WRITE) == 0u || frame->buffers[1].address == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( frame->buffers[0].bytes < (uint64_t)frame->new_token_count * sizeof(uint32_t) || frame->buffers[1].bytes < (uint64_t)frame->new_token_count * sizeof(uint32_t) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	*is_prefill = prefill;
	*rows = frame->new_token_count;
	return(SPARK_STATUS_OK);
}

/*
 * A frame context is optional and, when present, may only override the block
 * table. Version 1 has no pipeline transport, so a frame asking for one is
 * rejected instead of being served without it.
 */
static SparkStatus SparkK3ModuleResolveBlockTable(SparkK3ModuleState *state, const SparkModelDriverFrame *frame, const SparkK3MlaBlockTableView **block_table)
{
	const SparkK3ResidentDecodeStageFrameContext *context;
	*block_table = &state->owned_block_table;
	if ( frame->user_context == 0 )
		return(SPARK_STATUS_OK);
	context = (const SparkK3ResidentDecodeStageFrameContext *)frame->user_context;
	if ( context->abi_version != SPARK_K3_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION || context->descriptor_bytes != (uint32_t)sizeof(SparkK3ResidentDecodeStageFrameContext) )
		return(SPARK_STATUS_ABI_MISMATCH);
	if ( (context->flags & (SPARK_K3_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_INPUT_TRANSPORT | SPARK_K3_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_OUTPUT_TRANSPORT)) != 0u )
	{
		fprintf(stderr,"k3_stage transport_unsupported flags=0x%08x\n",context->flags);
		return(SPARK_STATUS_INVALID_ARGUMENT);
	}
	if ( (context->flags & SPARK_K3_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_MLA_BLOCK_TABLE) == 0u )
		return(SPARK_STATUS_OK);
	if ( context->mla_block_table == 0 || context->mla_block_table->abi_version != SPARK_K3_RESIDENT_DECODE_STAGE_MLA_BLOCK_TABLE_ABI_VERSION )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( context->mla_block_table->block_token_count != SPARK_K3_RESIDENT_DECODE_STAGE_MLA_BLOCK_TOKENS || context->mla_block_table->lane_count > state->lane_capacity )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( context->mla_block_table->physical_block_indices == 0 || context->mla_block_table->lane_physical_block_counts == 0 || context->mla_block_table->lane_stride == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( context->mla_block_table->host_physical_block_indices == 0 || context->mla_block_table->host_lane_physical_block_counts == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	*block_table = context->mla_block_table;
	return(SPARK_STATUS_OK);
}

/*
 * The attend kernel walks the lane's block list up to the frame's last
 * context position; a table whose lane does not own that many blocks would
 * send the kernel past its list, so the coverage is proven on the host from
 * the table's own mirror before anything launches.
 */
static SparkStatus SparkK3ModuleValidateBlockCoverage(const SparkK3MlaBlockTableView *block_table, const SparkModelDriverFrame *frame, uint32_t lane, uint32_t token_count)
{
	uint32_t blocks_needed = (uint32_t)(((frame->sequence_position + token_count) + SPARK_K3_RESIDENT_DECODE_STAGE_MLA_BLOCK_TOKENS - 1u) / SPARK_K3_RESIDENT_DECODE_STAGE_MLA_BLOCK_TOKENS);
	if ( lane >= block_table->lane_count || blocks_needed > block_table->host_lane_physical_block_counts[lane] || blocks_needed > block_table->lane_stride )
	{
		fprintf(stderr,"k3_stage block_table_short lane=%u needed=%u owned=%u stride=%u\n",lane,blocks_needed,lane < block_table->lane_count ? block_table->host_lane_physical_block_counts[lane] : 0u,block_table->lane_stride);
		return(SPARK_STATUS_INVALID_ARGUMENT);
	}
	return(SPARK_STATUS_OK);
}

static void SparkK3ModuleFillCompletion(const SparkK3ModuleSlotStaging *staging, const SparkModelDriverFrame *frame, uint32_t rows, SparkModelDriverCompletion *completion)
{
	memset(completion,0,sizeof(*completion));
	completion->request_id = frame->request_id;
	completion->sequence_id = frame->sequence_id;
	completion->sequence_position = frame->sequence_position + rows;
	completion->program_id = frame->program_id;
	completion->driver_dispatch_slot = frame->driver_dispatch_slot;
	completion->accepted_token_count = rows;
	completion->token_count = 1u;
	completion->token_ids[0] = staging->row_output_token_ids[rows - 1u];
	completion->status = SPARK_STATUS_OK;
	completion->residency = frame->residency;
	completion->device_memcpy_bytes = (uint64_t)rows * sizeof(uint32_t) * 2u;
	completion->host_staging_bytes = (uint64_t)rows * sizeof(uint32_t);
}

// Claim a free pipeline slot by compare-and-swap, scanning once around the
// ring from an atomic hint. Every slot busy means the node is saturated and
// the frame bounces with BUSY rather than queueing.
static int32_t SparkK3ModuleClaimSlot(SparkK3ModuleState *state, uint32_t *slot_index)
{
	uint32_t count = state->node_context.pipeline_slot_count;
	uint32_t hint = atomic_fetch_add_explicit(&state->next_pipeline_slot,1u,memory_order_relaxed);
	uint32_t probe,candidate;
	unsigned int expected;
	for (probe = 0; probe < count; probe++)
	{
		candidate = (hint + probe) % count;
		expected = 0u;
		if ( atomic_compare_exchange_strong_explicit(&state->slot_staging[candidate].busy,&expected,1u,memory_order_acquire,memory_order_relaxed) )
		{
			*slot_index = candidate;
			return(0);
		}
	}
	return(-1);
}

static SparkStatus SparkK3ModuleExecuteOnSlot(SparkK3ModuleState *state, const SparkK3PipelineSlot *slot, SparkK3ModuleSlotStaging *staging, SparkModelDriverFrame *frame, const SparkK3MlaBlockTableView *block_table, uint32_t is_prefill, uint32_t rows)
{
	SparkModelDriverCompletion completion;
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	SparkStatus status;
	uint32_t lane = frame->driver_dispatch_slot;
	uint32_t carry_state = frame->sequence_position != 0u ? 1u : 0u;
	uint32_t padded_rows = is_prefill != 0u ? SPARK_K3_MODEL_KDA_CHUNK_TOKENS : rows;
	memcpy(staging->row_token_ids,frame->buffers[0].address,(size_t)rows * sizeof(uint32_t));
	status = SparkK3ModuleFillRowMetadata(state,staging,frame,lane,rows,padded_rows);
	if ( status == SPARK_STATUS_OK )
		status = SparkK3ModuleUploadRowMetadata(staging,slot,padded_rows,rows,stream);
	if ( status == SPARK_STATUS_OK )
		status = SparkK3ModuleUpdateLaneState(state,staging,frame,padded_rows,stream);
	if ( status == SPARK_STATUS_OK )
		status = SparkK3ModuleRunStage(state,slot,staging,block_table,padded_rows,is_prefill,carry_state,stream);
	if ( status == SPARK_STATUS_OK )
		status = SparkK3ModuleCudaStatus(cudaMemcpyAsync(staging->row_output_token_ids,slot->output_token_ids,(size_t)rows * sizeof(uint32_t),cudaMemcpyDeviceToHost,stream),"output_ids_d2h");
	if ( status == SPARK_STATUS_OK )
		status = SparkK3ModuleCudaStatus(cudaStreamSynchronize(stream),"cudaStreamSynchronize");
	if ( status != SPARK_STATUS_OK )
		return(status);
	memcpy(frame->buffers[1].address,staging->row_output_token_ids,(size_t)rows * sizeof(uint32_t));
	if ( frame->completion_function != 0 )
	{
		SparkK3ModuleFillCompletion(staging,frame,rows,&completion);
		frame->completion_function(frame->completion_context,&completion);
	}
	return(SPARK_STATUS_OK);
}

SparkStatus SparkK3ResidentDecodeStageExecute(void *module_state, SparkModelDriverFrame *frame)
{
	SparkK3ModuleState *state = (SparkK3ModuleState *)module_state;
	const SparkK3MlaBlockTableView *block_table;
	SparkStatus status;
	uint32_t is_prefill,rows,slot_index;
	if ( state == 0 || frame == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkK3ModuleValidateFrameShape(state,frame,&is_prefill,&rows);
	if ( status == SPARK_STATUS_OK )
		status = SparkK3ModuleResolveBlockTable(state,frame,&block_table);
	if ( status == SPARK_STATUS_OK )
		status = SparkK3ModuleValidateBlockCoverage(block_table,frame,frame->driver_dispatch_slot,rows);
	if ( status == SPARK_STATUS_OK )
		status = SparkK3ModuleValidateWireTokens((const uint32_t *)frame->buffers[0].address,rows);
	if ( status != SPARK_STATUS_OK )
	{
		atomic_fetch_add_explicit(&state->rejected_count,1u,memory_order_relaxed);
		fprintf(stderr,"k3_stage execute_reject status=%d flags=0x%08x tokens=%u position=%llu\n",(int32_t)status,frame->flags,frame->new_token_count,(unsigned long long)frame->sequence_position);
		return(status);
	}
	if ( SparkK3ModuleClaimSlot(state,&slot_index) < 0 )
	{
		atomic_fetch_add_explicit(&state->rejected_count,1u,memory_order_relaxed);
		return(SPARK_STATUS_BUSY);
	}
	atomic_fetch_add_explicit(&state->submitted_count,1u,memory_order_relaxed);
	status = SparkK3ModuleExecuteOnSlot(state,&state->pipeline_slots[slot_index],&state->slot_staging[slot_index],frame,block_table,is_prefill,rows);
	atomic_fetch_add_explicit(status == SPARK_STATUS_OK ? &state->completed_count : &state->failed_count,1u,memory_order_relaxed);
	atomic_store_explicit(&state->slot_staging[slot_index].busy,0u,memory_order_release);
	return(status);
}

SparkStatus SparkK3ResidentDecodeStageAdmit(void *module_state, const SparkModelDriverAdmissionRequest *request, SparkModelDriverAdmissionDecision *decision)
{
	SparkK3ModuleState *state = (SparkK3ModuleState *)module_state;
	uint32_t prefill;
	if ( state == 0 || request == 0 || decision == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(decision,0,sizeof(*decision));
	decision->descriptor_bytes = (uint32_t)sizeof(SparkModelDriverAdmissionDecision);
	decision->available_dispatch_slot_count = state->lane_capacity;
	decision->estimated_service_time_ns = state->node_context.estimated_service_time_ns;
	prefill = (request->frame_flags & SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL) != 0u ? 1u : 0u;
	if ( request->active_slot_count == 0u || request->active_slot_count > state->lane_capacity )
		decision->rejection_reason = SPARK_STATUS_CAPACITY_EXCEEDED;
	else if ( request->new_token_count == 0u || (prefill != 0u && request->new_token_count > SPARK_K3_MODEL_KDA_CHUNK_TOKENS) || (prefill == 0u && request->new_token_count != 1u) )
		decision->rejection_reason = SPARK_STATUS_INVALID_ARGUMENT;
	else if ( request->sequence_position + request->new_token_count > (uint64_t)state->max_context_tokens )
		decision->rejection_reason = SPARK_STATUS_CAPACITY_EXCEEDED;
	if ( decision->rejection_reason != SPARK_STATUS_OK )
	{
		atomic_fetch_add_explicit(&state->rejected_count,1u,memory_order_relaxed);
		return(SPARK_STATUS_OK);
	}
	decision->accepted = 1u;
	decision->driver_dispatch_slot = (uint32_t)(request->sequence_id % (uint64_t)state->lane_capacity);
	decision->residency_match_score = 1u;
	decision->host_staging_bytes = (uint64_t)request->new_token_count * sizeof(uint32_t);
	decision->device_memcpy_bytes = (uint64_t)request->new_token_count * sizeof(uint32_t) * 2u;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkK3ResidentDecodeStageSnapshot(void *module_state, uint32_t program_id, SparkModelDriverRuntimeSnapshot *snapshot)
{
	SparkK3ModuleState *state = (SparkK3ModuleState *)module_state;
	if ( state == 0 || snapshot == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(snapshot,0,sizeof(*snapshot));
	snapshot->descriptor_bytes = (uint32_t)sizeof(SparkModelDriverRuntimeSnapshot);
	snapshot->program_id = program_id;
	snapshot->available_dispatch_slot_count = state->lane_capacity;
	snapshot->submitted_count = atomic_load_explicit(&state->submitted_count,memory_order_relaxed);
	snapshot->completed_count = atomic_load_explicit(&state->completed_count,memory_order_relaxed);
	snapshot->rejected_count = atomic_load_explicit(&state->rejected_count,memory_order_relaxed) + atomic_load_explicit(&state->failed_count,memory_order_relaxed);
	snapshot->kv_token_capacity = (uint64_t)state->lane_capacity * state->blocks_per_lane * SPARK_K3_RESIDENT_DECODE_STAGE_MLA_BLOCK_TOKENS;
	snapshot->active_submission_count = (uint32_t)(snapshot->submitted_count - snapshot->completed_count - atomic_load_explicit(&state->failed_count,memory_order_relaxed));
	return(SPARK_STATUS_OK);
}

void SparkK3ResidentDecodeStageDestroy(void *module_state)
{
	SparkK3ModuleState *state = (SparkK3ModuleState *)module_state;
	uint32_t index;
	if ( state == 0 )
		return;
	for (index = 0; index < state->node_context.pipeline_slot_count; index++)
		if ( state->pipeline_slots[index].cuda_stream != 0 )
			cudaStreamDestroy((cudaStream_t)state->pipeline_slots[index].cuda_stream);
	for (index = 0; index < state->device_allocation_count; index++)
		cudaFree(state->device_allocations[index]);
	for (index = 0; index < state->node_context.pipeline_slot_count; index++)
	{
		free(state->slot_staging[index].row_token_ids);
		free(state->slot_staging[index].row_slot_mapping);
		free(state->slot_staging[index].row_lane_indices);
		free(state->slot_staging[index].row_context_lengths);
		free(state->slot_staging[index].row_cold_flags);
		free(state->slot_staging[index].row_output_token_ids);
	}
	free(state->host_block_indices);
	free(state->host_lane_block_counts);
	free(state);
}
