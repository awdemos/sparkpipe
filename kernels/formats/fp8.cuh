#pragma once

// FP8 E4M3 as a format trait.
//
// A format is a type, not a directory of kernels. Everything that varies with
// the weight format is declared here - the mma to issue, its K depth, how scales
// are laid out and consumed, how many elements pack into a byte - and every
// kernel takes the trait as a template parameter. That is what keeps one GEMM
// body in this tree instead of one per format.
//
// The test of the factoring: adding INT8 should be a file in this directory and
// nothing else.

#include "kernels/mma.cuh"

struct LmFp8
{
	typedef float Accumulator;
	static constexpr uint32_t kBits = 8u;
	// Stored width equals compute width: this format is native, so what TMA
	// moves is what the mma register holds and staging is a copy.
	static constexpr uint32_t kStoredBits = 8u;
	static constexpr uint32_t kMmaM = LM_MMA8_M;
	static constexpr uint32_t kMmaN = LM_MMA8_N;
	static constexpr uint32_t kMmaK = LM_MMA8_K;
	// Scales are FP32, one per 128x128 weight block, applied once per K tile
	// OUTSIDE the mma. That is why this format needs a per-K-tile partial
	// accumulator and the 4-bit formats do not.
	static constexpr bool kScaleInMma = false;
	static constexpr uint32_t kScaleGroup = 128u;
	static constexpr float kMax = LM_E4M3_MAX;

	static __device__ __forceinline__ uint32_t OperandARow(uint32_t lane, uint32_t reg)
	{
		return(LmMma8OperandARow(lane,reg));
	}
	static __device__ __forceinline__ uint32_t OperandAByte(uint32_t lane, uint32_t reg)
	{
		return(LmMma8OperandAByte(lane,reg));
	}
	static __device__ __forceinline__ uint32_t OperandBRow(uint32_t lane)
	{
		return(LmMma8OperandBRow(lane));
	}
	static __device__ __forceinline__ uint32_t OperandBByte(uint32_t lane, uint32_t reg)
	{
		return(LmMma8OperandBByte(lane,reg));
	}
	static __device__ __forceinline__ void Mma(float acc[4], const uint32_t a[4], const uint32_t b[2], uint32_t, uint32_t)
	{
		LmMmaE4m3(acc,a,b);
	}
	static __device__ __forceinline__ uint8_t Encode(float value, float inverse_scale)
	{
		return(LmFloatToE4m3(value * inverse_scale));
	}
};
