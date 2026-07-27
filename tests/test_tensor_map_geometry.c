// Descriptor geometry for the first-party grouped GEMM, checked without CUDA.
//
// The failure this guards against: a descriptor that encodes cleanly and moves
// the wrong bytes. Every case below is one that cuTensorMapEncodeTiled would
// accept.
#include "inference/kernels/tensor_map.cuh"

#include <stdio.h>
#include <string.h>

static int32_t failures = 0;

static void expect(int32_t condition, const char *label)
{
	if ( condition == 0 )
	{
		printf("  FAIL %s\n",label);
		failures++;
		return;
	}
	printf("  ok   %s\n",label);
}

static int32_t build(uint64_t rows, uint64_t columns, uint64_t groups, uint32_t box_rows, uint32_t box_columns, uint32_t bits, LmTensorMapPlan *plan)
{
	LmTensorMapRequest request;
	static uint8_t aligned_storage[256] __attribute__((aligned(128)));
	memset(&request,0,sizeof(request));
	request.global_address = aligned_storage;
	request.rows = rows;
	request.columns = columns;
	request.groups = groups;
	request.box_rows = box_rows;
	request.box_columns = box_columns;
	request.element_bits = bits;
	return(LmTensorMapPlanBuild(&request,plan));
}

int32_t main(void)
{
	LmTensorMapPlan plan;
	int32_t status;
	printf("TMA descriptor geometry\n");
	printf("\nswizzle agreement between descriptor and kernel\n");
	expect(LM_TM_SWIZZLE_CHUNKS == 8u,
		"descriptor swizzle chunk count equals the kernel's");
	expect(LM_TM_SWIZZLE_CHUNKS * LM_TM_CHUNK_BYTES
		== LM_TM_SWIZZLE_BYTES,
		"8 chunks x 16 bytes equals the 128-byte swizzle span");
	printf("\nFP8 activation tile, M rows x K=6144, box 16x128\n");
	status = build(128u,6144u,1u,16u,128u,LM_TM_BITS_FP8,&plan);
	expect(status == LM_TM_OK,"builds");
	expect(plan.rank == 2u,"rank 2 for a single group");
	expect(plan.row_bytes == 6144u,"FP8 row is K bytes");
	expect(plan.box_dimension[0] == 128u,"box inner extent is 128 bytes");
	expect(plan.box_bytes == 2048u,"16x128 FP8 box is 2048 bytes");
	// TILE_K for NVFP4 must be 256 ELEMENTS, not 128. At 4 bits a 128-element
	// K tile is 64 bytes wide, which is narrower than the 128-byte swizzle span
	// and cannot be permuted at that granularity. This constraint does not exist
	// for FP8, where 128 elements are already 128 bytes, and it is invisible
	// until a descriptor is actually constructed.
	printf("\nNVFP4 weight tensor, 256 experts x N=4096 x K=6144\n");
	expect(build(4096u,6144u,256u,128u,128u,LM_TM_BITS_NVFP4,&plan)
		== LM_TM_ERR_BOX_ALIGN,
		"TILE_K=128 elements is 64 bytes at NVFP4 and is correctly rejected");
	status = build(4096u,6144u,256u,128u,256u,LM_TM_BITS_NVFP4,&plan);
	expect(status == LM_TM_OK,"TILE_K=256 elements builds");
	expect(plan.rank == 3u,"rank 3 so one descriptor covers every expert");
	expect(plan.row_bytes == 3072u,"NVFP4 row is K/2 bytes, not K");
	expect(plan.box_dimension[0] == 128u,"NVFP4 box inner extent is 128 bytes");
	expect(plan.box_bytes == 16384u,"128x256 NVFP4 box is 16384 bytes");
	expect(plan.global_stride_bytes[1] == 3072u * 4096u,"expert stride spans a whole weight matrix");
	printf("\nrejections that would otherwise encode cleanly\n");
	expect(build(128u,6144u,1u,16u,128u,6u,&plan) == LM_TM_ERR_BITS,
		"6-bit element width rejected");
	expect(build(128u,6145u,1u,16u,128u,LM_TM_BITS_NVFP4,&plan)
		== LM_TM_ERR_ODD_COLUMNS,
		"odd NVFP4 column count has no byte representation");
	expect(build(128u,320u,1u,16u,128u,LM_TM_BITS_FP8,&plan)
		== LM_TM_ERR_ROW_SWIZZLE,
		"row pitch not a multiple of 128 bytes rejected");
	expect(build(8u,6144u,1u,16u,128u,LM_TM_BITS_FP8,&plan)
		== LM_TM_ERR_BOX_EXCEEDS,
		"box taller than the tensor rejected");
	printf("\n%s (%d failing checks)\n",failures == 0 ? "PASS" : "FAIL",failures);
	return(failures == 0 ? 0 : 1);
}
