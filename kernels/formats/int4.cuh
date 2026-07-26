#pragma once

// INT4. Native s4 mma at m16n8k64, S32 accumulator.
//
// Shares the 4-bit operand layout with NVFP4, so a K tile must be 256 elements
// to be a whole swizzle span in bytes; kernels/tile.cuh asserts it.

#include "kernels/mma.cuh"

struct LmInt4
{
	typedef int32_t Accumulator;
	static constexpr uint32_t kBits = 4u;
	// Stored width equals compute width: this format is native, so what TMA
	// moves is what the mma register holds and staging is a copy.
	static constexpr uint32_t kStoredBits = 4u;
	static constexpr uint32_t kMmaM = LM_MMA4_M;
	static constexpr uint32_t kMmaN = LM_MMA4_N;
	static constexpr uint32_t kMmaK = LM_MMA4_K;
	static constexpr bool kScaleInMma = false;
	static constexpr uint32_t kScaleGroup = 128u;
	static constexpr float kMax = 7.0f;

	static __device__ __forceinline__ uint32_t OperandARow(uint32_t lane, uint32_t reg) { return(LmMma4OperandARow(lane,reg)); }
	static __device__ __forceinline__ uint32_t OperandAByte(uint32_t lane, uint32_t reg) { return(LmMma4OperandAByte(lane,reg)); }
	static __device__ __forceinline__ uint32_t OperandBRow(uint32_t lane) { return(LmMma4OperandBRow(lane)); }
	static __device__ __forceinline__ uint32_t OperandBByte(uint32_t lane, uint32_t reg) { return(LmMma4OperandBByte(lane,reg)); }
	static __device__ __forceinline__ void Mma(int32_t acc[4], const uint32_t a[4], const uint32_t b[2], uint32_t, uint32_t)
	{
		LmMmaS4(acc,a,b);
	}
};
