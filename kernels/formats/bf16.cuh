#pragma once

// BF16: no quantisation, no scales.
//
// Kept because prefill and the unquantised families use it, and because a path
// that avoids quantisation entirely is the cheapest way to isolate a numerics
// bug to the quantiser rather than the GEMM.

#include "kernels/mma.cuh"

struct LmBf16Format
{
	static constexpr uint32_t kBits = 16u;
	static constexpr uint32_t kMmaM = LM_MMA16_M;
	static constexpr uint32_t kMmaN = LM_MMA16_N;
	static constexpr uint32_t kMmaK = LM_MMA16_K;
	static constexpr bool kScaleInMma = false;
	static constexpr uint32_t kScaleGroup = 0u;   // no scales at all

	static __device__ __forceinline__ void Mma(float acc[4], const uint32_t a[4], const uint32_t b[2], uint32_t, uint32_t)
	{
		LmMmaBf16(acc,a,b);
	}
};
