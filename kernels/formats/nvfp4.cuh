#pragma once

// NVFP4: E2M1 data, UE4M3 scale, one scale per 16 elements.
//
// The scale is consumed BY the mma rather than applied after it, which removes
// the per-K-tile partial accumulator and the whole rescale pass that FP8 needs.
// That difference is real and is why kScaleInMma exists; everything else about
// the GEMM is identical and is written once.
//
// CUTLASS emits scale_vec::4X with ue8m0 for its VS=16 path, guarded by
// CUTE_ARCH_MXF4NVF4_4X_UE8M0_MMA_ENABLED. ptxas rejects that combination for
// sm_121a - it is an sm_100 capability - so the collective's emission cannot be
// transplanted here. tests/test_ptx_capability_gate.py keeps a negative probe on
// it so a toolkit change fails the build rather than changing behaviour.

#include "kernels/mma.cuh"

struct LmNvfp4
{
	typedef float Accumulator;
	static constexpr uint32_t kBits = 4u;
	static constexpr uint32_t kMmaM = LM_MMA4_M;
	static constexpr uint32_t kMmaN = LM_MMA4_N;
	static constexpr uint32_t kMmaK = LM_MMA4_K;
	static constexpr bool kScaleInMma = true;
	static constexpr uint32_t kScaleGroup = LM_MMA4_NVFP4_GROUP;
	static constexpr float kMax = LM_E2M1_MAX;

	static __device__ __forceinline__ uint32_t OperandARow(uint32_t lane, uint32_t reg)
	{
		return(LmMma4OperandARow(lane,reg));
	}
	static __device__ __forceinline__ uint32_t OperandAByte(uint32_t lane, uint32_t reg)
	{
		return(LmMma4OperandAByte(lane,reg));
	}
	static __device__ __forceinline__ uint32_t OperandBRow(uint32_t lane)
	{
		return(LmMma4OperandBRow(lane));
	}
	static __device__ __forceinline__ uint32_t OperandBByte(uint32_t lane, uint32_t reg)
	{
		return(LmMma4OperandBByte(lane,reg));
	}
	static __device__ __forceinline__ void Mma(float acc[4], const uint32_t a[4], const uint32_t b[2], uint32_t scale_a, uint32_t scale_b)
	{
		LmMmaNvfp4(acc,a,b,scale_a,scale_b);
	}
	static __device__ __forceinline__ uint8_t EncodePair(float low, float high, float inverse_scale)
	{
		return(LmFloatPairToE2m1(low * inverse_scale,high * inverse_scale));
	}
};
