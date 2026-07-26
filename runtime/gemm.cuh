#pragma once

// The four CUDA calls a grouped GEMM needs, and the dispatch over tile height.
//
// runtime/launch.h decides everything arithmetic - tile height, shared bytes,
// swizzle span, grid width - and is checkable on a host. This is what turns a
// plan into a launch, and it is deliberately thin: if logic appears here that
// is not a CUDA call or a compile-time dispatch, it belongs next door where it
// can be tested.
//
// TILE HEIGHT IS A SWITCH, NOT A VALUE. It sizes shared memory and the
// accumulator array, both compile-time, so a runtime tile is not expressible.
// A height outside the table is a caller error rather than a case to
// approximate - approximating it would mean silently splitting groups and
// doubling the weight stream.

#include "kernels/gemm.cuh"
#include "runtime/launch.h"
#include "runtime/tensor_map.h"
#include <cuda_runtime.h>

// Dynamic shared above 48 KB requires an explicit opt-in before the first
// launch, once per kernel. Doing it per launch would be a driver call on the hot
// path; skipping it fails the launch every time rather than corrupting anything,
// which is the right failure but a total one.
template<class Format, uint32_t TILE_M, uint32_t TILE_N, uint32_t TILE_K, uint32_t STAGES, uint32_t WARPS>
static cudaError_t LmGemmOptIn(uint32_t shared_bytes)
{
	static bool granted = false;
	cudaError_t status;
	if ( granted )
		return(cudaSuccess);
	status = cudaFuncSetAttribute(
		(const void *)LmGemmKernel<Format,TILE_M,TILE_N,TILE_K,STAGES,WARPS>,
		cudaFuncAttributeMaxDynamicSharedMemorySize,
		(int)shared_bytes);
	if ( status == cudaSuccess )
		granted = true;
	return(status);
}

template<class Format, uint32_t TILE_M, uint32_t TILE_N, uint32_t TILE_K, uint32_t STAGES, uint32_t WARPS>
static cudaError_t LmGemmLaunchTile(const LmGemmArguments &args, const LmTileSource &a, const LmTileSource &b, bool grouped, const LmLaunchPlan &plan, cudaStream_t stream)
{
	constexpr uint32_t shared = LmGemmSharedBytes<Format,TILE_M,TILE_N,TILE_K,STAGES>();
	cudaError_t status;
	static_assert(shared <= LM_SMEM_SM_TOTAL,"tile exceeds the shared memory an SM has");
	status = LmGemmOptIn<Format,TILE_M,TILE_N,TILE_K,STAGES,WARPS>(shared);
	if ( status != cudaSuccess )
		return(status);
	LmGemmKernel<Format,TILE_M,TILE_N,TILE_K,STAGES,WARPS>
		<<<plan.grid_blocks,plan.block_threads,shared,stream>>>(args,a,b,grouped);
	return(cudaPeekAtLastError());
}

// Encode both descriptors. Weights are expert-major so one rank-3 map covers
// every expert and the third coordinate selects; activations are a rank-2 packed
// slab. Both go through LmTensorMapPlanBuild, which halves every K extent for a
// sub-byte format so no call site has to remember to.
template<class Format>
static int32_t LmGemmEncodeMaps(CUtensorMap *activation, CUtensorMap *weight, const void *activation_bytes, const void *weight_bytes, uint32_t packed_rows, uint32_t input_dimension, uint32_t output_dimension, uint32_t group_count, uint32_t tile_m, uint32_t tile_n, uint32_t tile_k)
{
	LmTensorMapRequest request;
	int32_t status;
	memset(&request,0,sizeof(request));
	request.global_address = activation_bytes;
	request.rows = packed_rows;
	request.columns = input_dimension;
	request.groups = 1u;
	request.box_rows = tile_m;
	request.box_columns = tile_k;
	request.element_bits = Format::kStoredBits;
	status = LmTensorMapPrepare(activation,&request);
	if ( status != LM_TM_ENCODE_OK )
		return(status);
	memset(&request,0,sizeof(request));
	request.global_address = weight_bytes;
	request.rows = output_dimension;
	request.columns = input_dimension;
	request.groups = group_count;
	request.box_rows = tile_n;
	request.box_columns = tile_k;
	request.element_bits = Format::kStoredBits;
	return(LmTensorMapPrepare(weight,&request));
}

// Plan, encode, dispatch. The one entry point a model's driver calls.
//
// The tile switch lists exactly the instantiations a model's unity.cu compiles.
// Adding a height means adding it in both places, and omitting it here is a
// runtime error while omitting it there is a link error - both loud.
template<class Format, uint32_t TILE_N, uint32_t TILE_K, uint32_t STAGES, uint32_t WARPS>
static int32_t LmGemmLaunch(LmGemmArguments *args, const void *activation_bytes, const void *weight_bytes, uint32_t packed_rows, uint32_t tokens, uint32_t top_k, uint32_t group_count, uint32_t input_dimension, uint32_t output_dimension, uint32_t multiprocessors, bool grouped, cudaStream_t stream)
{
	LmLaunchShape shape;
	LmLaunchPlan plan;
	LmTileSource source_a,source_b;
	CUtensorMap activation_map,weight_map;
	int32_t status;
	memset(&shape,0,sizeof(shape));
	shape.tokens = tokens;
	shape.top_k = top_k;
	shape.expert_count = group_count;
	shape.input_dimension = input_dimension;
	shape.output_dimension = output_dimension;
	shape.stored_bits = Format::kStoredBits;
	shape.tile_n = TILE_N;
	shape.tile_k = TILE_K;
	shape.stages = STAGES;
	status = LmLaunchPlanBuild(&shape,multiprocessors,&plan);
	if ( status != LM_LAUNCH_OK )
		return(status);
	status = LmGemmEncodeMaps<Format>(&activation_map,&weight_map,activation_bytes,weight_bytes,
		packed_rows,input_dimension,output_dimension,group_count,plan.tile_m,TILE_N,TILE_K);
	if ( status != LM_TM_ENCODE_OK )
		return(LM_LAUNCH_ERR_MAP);
	args->tensor_map_a = &activation_map;
	args->tensor_map_b = &weight_map;
	args->group_count = group_count;
	args->input_dimension = input_dimension;
	args->output_dimension = output_dimension;
	memset(&source_a,0,sizeof(source_a));
	memset(&source_b,0,sizeof(source_b));
	source_a.tensor_map = &activation_map;
	source_a.rows = plan.tile_m;
	source_a.depth = TILE_K;
	source_a.element_bits = Format::kStoredBits;
	source_b.tensor_map = &weight_map;
	source_b.rows = TILE_N;
	source_b.depth = TILE_K;
	source_b.element_bits = Format::kStoredBits;
	switch ( plan.tile_m )
	{
		case 16u:
			return(LmGemmLaunchTile<Format,16u,TILE_N,TILE_K,STAGES,WARPS>(*args,source_a,source_b,grouped,plan,stream) == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
		case 32u:
			return(LmGemmLaunchTile<Format,32u,TILE_N,TILE_K,STAGES,WARPS>(*args,source_a,source_b,grouped,plan,stream) == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
		case 64u:
			return(LmGemmLaunchTile<Format,64u,TILE_N,TILE_K,STAGES,WARPS>(*args,source_a,source_b,grouped,plan,stream) == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
		default:
			return(LM_LAUNCH_ERR_TILE);
	}
}
