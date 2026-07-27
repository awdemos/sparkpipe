// TMA tensor-map geometry for the first-party grouped GEMM.
//
// No first-party code in this tree has ever called cuTensorMapEncodeTiled.
// CUTLASS built its descriptors internally, so the existing seams pass raw
// pointers and the descriptor concept never surfaced. It surfaces now, because
// spark_lm_group_gemm.cuh stages tiles with cp.async.bulk.tensor, which is
// addressed by tensor coordinates against a descriptor rather than by a pointer.
//
// This header computes the geometry. The driver call itself lives in the host
// launcher; splitting them is what lets the arithmetic be tested with no CUDA
// runtime, which is the part that is easy to get silently wrong.
//
// THE ONE AGREEMENT THAT MATTERS. The descriptor encodes a swizzle mode and the
// kernel applies a matching xor when computing fragment addresses. The span is
// a function of the row pitch, not a constant - see LmSwizzleSpanFor in
// kernels/mma.cuh, which this must agree with. If they
// disagree the kernel reads real data from the wrong place and produces
// plausible wrong numbers with no error anywhere. CU_TENSOR_MAP_SWIZZLE_128B
// and an 8-chunk xor over 16-byte chunks are the same transform;
// LM_TM_SWIZZLE_CHUNKS and SPARK_LM_GROUP_GEMM_SWIZZLE_CHUNKS
// must stay equal and tests/test_tensor_map_geometry.c asserts it.
//
// NVFP4 IS DESCRIBED AS BYTES. There is no 4-bit CUtensorMapDataType, so a
// 4-bit tensor of K columns is described as UINT8 with K/2 columns and every
// K-axis extent is halved. Getting this wrong yields a descriptor that encodes
// cleanly and transfers half or twice the intended data.

#ifndef LM_TENSOR_MAP_H
#define LM_TENSOR_MAP_H

#include <stdint.h>

#define LM_TM_SWIZZLE_CHUNKS 8u
#define LM_TM_CHUNK_BYTES 16u
#define LM_TM_SWIZZLE_BYTES 128u
#define LM_TM_ALIGNMENT 16u
#define LM_TM_MAX_RANK 3u
#define LM_TM_BITS_FP8 8u
#define LM_TM_BITS_NVFP4 4u

// Unique negative codes so a rejection names its own site.
#define LM_TM_OK 0
#define LM_TM_ERR_NULL (-1)
#define LM_TM_ERR_BITS (-2)
#define LM_TM_ERR_RANK (-3)
#define LM_TM_ERR_ODD_COLUMNS (-4)
#define LM_TM_ERR_BOX_ODD (-5)
#define LM_TM_ERR_ROW_SWIZZLE (-6)
#define LM_TM_ERR_BOX_ALIGN (-7)
#define LM_TM_ERR_BOX_EXCEEDS (-8)
#define LM_TM_ERR_ADDRESS_ALIGN (-9)

typedef struct LmTensorMaprequest
{
	const void *global_address;
	uint64_t rows,columns,groups;
	uint32_t box_rows,box_columns,element_bits;
}
LmTensorMapRequest;

typedef struct LmTensorMapplan
{
	uint32_t rank;
	uint64_t global_dimension[LM_TM_MAX_RANK];
	uint64_t global_stride_bytes[LM_TM_MAX_RANK];
	uint32_t box_dimension[LM_TM_MAX_RANK];
	uint32_t element_stride[LM_TM_MAX_RANK];
	uint32_t swizzle_bytes;
	uint64_t row_bytes,box_bytes;
}
LmTensorMapPlan;

// Element count to bytes. Its own function because it is the conversion NVFP4
// makes easy to get wrong, and it should exist exactly once.
static uint64_t LmTensorMapBytes(uint64_t elements, uint32_t element_bits)
{
	return((elements * (uint64_t)element_bits) / 8u);
}

// Build the descriptor geometry. Every rejection below is a case that would
// otherwise encode cleanly and move the wrong bytes at runtime.
static int32_t LmTensorMapPlanBuild(const LmTensorMapRequest *request, LmTensorMapPlan *plan)
{
	uint64_t column_bytes;
	uint32_t box_column_bytes,index;
	if ( request == 0 || plan == 0 || request->global_address == 0 )
		return(LM_TM_ERR_NULL);
	if ( request->element_bits != LM_TM_BITS_FP8 && request->element_bits != LM_TM_BITS_NVFP4 )
		return(LM_TM_ERR_BITS);
	if ( request->groups == 0 || request->rows == 0 || request->columns == 0 )
		return(LM_TM_ERR_RANK);
	// A 4-bit tensor is described as bytes, so an odd column count has no byte
	// representation and would silently round.
	if ( ((request->columns * (uint64_t)request->element_bits) % 8u) != 0u )
		return(LM_TM_ERR_ODD_COLUMNS);
	if ( ((request->box_columns * request->element_bits) % 8u) != 0u )
		return(LM_TM_ERR_BOX_ODD);
	column_bytes = LmTensorMapBytes(request->columns,request->element_bits);
	box_column_bytes = (uint32_t)LmTensorMapBytes(request->box_columns,request->element_bits);
	// 128B swizzle permutes 16-byte chunks within a 128-byte span, so both the
	// row pitch and the box width must be whole multiples of that span or the
	// permutation straddles a boundary the hardware does not cross.
	if ( (column_bytes % LM_TM_SWIZZLE_BYTES) != 0u )
		return(LM_TM_ERR_ROW_SWIZZLE);
	if ( (box_column_bytes % LM_TM_SWIZZLE_BYTES) != 0u )
		return(LM_TM_ERR_BOX_ALIGN);
	if ( (uint64_t)request->box_rows > request->rows || (uint64_t)box_column_bytes > column_bytes )
		return(LM_TM_ERR_BOX_EXCEEDS);
	if ( (((uintptr_t)request->global_address) % LM_TM_ALIGNMENT) != 0u )
		return(LM_TM_ERR_ADDRESS_ALIGN);
	// Innermost dimension first, as the driver expects. Dimension 0 is the
	// contiguous byte axis, 1 is rows, 2 selects the expert.
	plan->rank = request->groups > 1u ? 3u : 2u;
	plan->global_dimension[0] = column_bytes;
	plan->global_dimension[1] = request->rows;
	plan->global_dimension[2] = request->groups;
	// globalStrides has rank-1 meaningful entries and excludes the innermost
	// axis, which is implicitly one element. Slot 0 holds the row pitch.
	plan->global_stride_bytes[0] = column_bytes;
	plan->global_stride_bytes[1] = column_bytes * request->rows;
	plan->global_stride_bytes[2] = 0u;
	plan->box_dimension[0] = box_column_bytes;
	plan->box_dimension[1] = request->box_rows;
	plan->box_dimension[2] = 1u;
	for (index = 0u; index < LM_TM_MAX_RANK; ++index)
		plan->element_stride[index] = 1u;
	plan->swizzle_bytes = LM_TM_SWIZZLE_BYTES;
	plan->row_bytes = column_bytes;
	plan->box_bytes = (uint64_t)box_column_bytes * (uint64_t)request->box_rows;
	return(LM_TM_OK);
}

#endif
