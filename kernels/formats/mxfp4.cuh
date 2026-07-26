#pragma once

// MXFP4: E2M1 data, UE8M0 scale, one scale per 32 elements.
//
// Identical to NVFP4 except the scale type and group. Written as its own file
// rather than a flag on LmNvfp4 because the group size changes the K tile the
// GEMM must use, and a format whose geometry differs is a different format.

#include "kernels/formats/nvfp4.cuh"

struct LmMxfp4 : LmNvfp4
{
	static constexpr uint32_t kScaleGroup = LM_MMA4_MXFP4_GROUP;

	static __device__ __forceinline__ void Mma(float acc[4], const uint32_t a[4], const uint32_t b[2], uint32_t scale_a, uint32_t scale_b)
	{
		LmMmaMxfp4(acc,a,b,scale_a,scale_b);
	}
};
