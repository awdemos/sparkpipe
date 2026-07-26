// The cuTensorMapEncodeTiled call for the first-party grouped GEMM.
//
// spark_lm_tensor_map.h computes the geometry and is testable with no CUDA
// runtime; this file is the thin driver call over it. The split is deliberate:
// the arithmetic is where the silent errors live (NVFP4 K-extents halving, the
// swizzle span, the rank-1 stride convention) and it should be checkable on any
// host. What is left here is a translation of an already-validated plan.
//
// cuTensorMapEncodeTiled is a DRIVER API entry point. It needs cuda.h and a
// current context, not just the runtime API. Modules that only ever called
// cudaMemcpy have not needed the driver API before, so the link line gains
// -lcuda alongside -lcudart.

#include "sparkpipe/spark_lm_tensor_map.h"

#include <cuda.h>
#include <stdint.h>
#include <string.h>

#define SPARK_LM_TENSOR_MAP_ENCODE_OK 0
#define SPARK_LM_TENSOR_MAP_ENCODE_ERR_PLAN (-21)
#define SPARK_LM_TENSOR_MAP_ENCODE_ERR_NULL (-22)
#define SPARK_LM_TENSOR_MAP_ENCODE_ERR_DRIVER (-23)
#define SPARK_LM_TENSOR_MAP_ENCODE_ERR_SWIZZLE (-24)

// The descriptor swizzle must be the one the kernel's chunk xor implements.
// spark_lm_tensor_map.h fixes the span at 128 bytes and the kernel
// static_asserts its chunk count against the shared contract header, so this
// only has to reject a plan that arrived with a different span rather than
// choose one.
static int32_t spark_lm_tensor_map_swizzle_enum(uint32_t swizzle_bytes, CUtensorMapSwizzle *out)
{
	if ( out == 0 )
		return(SPARK_LM_TENSOR_MAP_ENCODE_ERR_NULL);
	if ( swizzle_bytes == 128u )
	{
		*out = CU_TENSOR_MAP_SWIZZLE_128B;
		return(SPARK_LM_TENSOR_MAP_ENCODE_OK);
	}
	if ( swizzle_bytes == 64u )
	{
		*out = CU_TENSOR_MAP_SWIZZLE_64B;
		return(SPARK_LM_TENSOR_MAP_ENCODE_OK);
	}
	if ( swizzle_bytes == 32u )
	{
		*out = CU_TENSOR_MAP_SWIZZLE_32B;
		return(SPARK_LM_TENSOR_MAP_ENCODE_OK);
	}
	return(SPARK_LM_TENSOR_MAP_ENCODE_ERR_SWIZZLE);
}

// Encode one descriptor from a built plan.
//
// Both FP8 and NVFP4 are described as CU_TENSOR_MAP_DATA_TYPE_UINT8. There is
// no 4-bit data type, so an NVFP4 tensor is a byte tensor of half the K extent
// and the plan has already halved every K-axis figure. Passing element counts
// here instead of the plan's byte counts is the mistake this signature is
// shaped to prevent: it takes a plan, never raw dimensions.
//
// globalStrides carries rank-1 entries and excludes the innermost axis, so
// plan->global_stride_bytes[0] is the row pitch and [1] is the expert stride.
// The driver reads exactly rank-1 of them.
static int32_t spark_lm_tensor_map_encode(CUtensorMap *tensor_map, const spark_lm_tensor_map_plan_t *plan, void *global_address)
{
	CUtensorMapSwizzle swizzle;
	CUresult driver_status;
	int32_t status;
	if ( tensor_map == 0 || plan == 0 || global_address == 0 )
		return(SPARK_LM_TENSOR_MAP_ENCODE_ERR_NULL);
	if ( plan->rank < 2u || plan->rank > SPARK_LM_TENSOR_MAP_MAX_RANK )
		return(SPARK_LM_TENSOR_MAP_ENCODE_ERR_PLAN);
	status = spark_lm_tensor_map_swizzle_enum(plan->swizzle_bytes,&swizzle);
	if ( status != SPARK_LM_TENSOR_MAP_ENCODE_OK )
		return(status);
	memset(tensor_map,0,sizeof(*tensor_map));
	driver_status = cuTensorMapEncodeTiled(
		tensor_map,
		CU_TENSOR_MAP_DATA_TYPE_UINT8,
		plan->rank,
		global_address,
		plan->global_dimension,
		plan->global_stride_bytes,
		plan->box_dimension,
		plan->element_stride,
		CU_TENSOR_MAP_INTERLEAVE_NONE,
		swizzle,
		CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
		// A grouped GEMM's last M tile is ragged whenever a group's row count is
		// not a multiple of TILE_M, which at decode is almost always. Zero-fill
		// is what makes that safe without a branch or an epilogue kernel: the
		// padded rows contribute zero to the accumulator.
		CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
	if ( driver_status != CUDA_SUCCESS )
		return(SPARK_LM_TENSOR_MAP_ENCODE_ERR_DRIVER);
	return(SPARK_LM_TENSOR_MAP_ENCODE_OK);
}

// Build and encode in one step, which is how callers should use it - there is
// no legitimate reason to encode a plan that was not just built and checked.
static int32_t spark_lm_tensor_map_prepare(CUtensorMap *tensor_map, const spark_lm_tensor_map_request_t *request)
{
	spark_lm_tensor_map_plan_t plan;
	int32_t status;
	status = spark_lm_tensor_map_plan_build(request,&plan);
	if ( status != SPARK_LM_TENSOR_MAP_OK )
		return(status);
	return(spark_lm_tensor_map_encode(tensor_map,&plan,(void *)request->global_address));
}
