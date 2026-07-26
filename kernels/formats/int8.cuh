#pragma once

// INT8. Native s8 mma, S32 accumulator.
//
// The accumulator type is why this is not a variant of LmFp8. An integer mma
// sums exact products into S32 and the epilogue converts through a scale; a
// float mma's accumulator already holds the value. Sharing an epilogue between
// them gives output wrong by whatever the scale was, silently.
//
// Operand layout is SM80_16x8x32_S32S8S8S32_TN, which is the same layout the FP8
// path uses and which tests/test_mma_fragment_mapping.c already verifies. That
// is not a coincidence to rely on - it is asserted there by name.

#include "kernels/mma.cuh"

struct LmInt8
{
	typedef int32_t Accumulator;
	static constexpr uint32_t kBits = 8u;
	static constexpr uint32_t kMmaM = LM_MMA8_M;
	static constexpr uint32_t kMmaN = LM_MMA8_N;
	static constexpr uint32_t kMmaK = LM_MMA8_K;
	static constexpr bool kScaleInMma = false;
	static constexpr uint32_t kScaleGroup = 128u;
	static constexpr float kMax = 127.0f;

	static __device__ __forceinline__ uint32_t OperandARow(uint32_t lane, uint32_t reg) { return(LmMma8OperandARow(lane,reg)); }
	static __device__ __forceinline__ uint32_t OperandAByte(uint32_t lane, uint32_t reg) { return(LmMma8OperandAByte(lane,reg)); }
	static __device__ __forceinline__ uint32_t OperandBRow(uint32_t lane) { return(LmMma8OperandBRow(lane)); }
	static __device__ __forceinline__ uint32_t OperandBByte(uint32_t lane, uint32_t reg) { return(LmMma8OperandBByte(lane,reg)); }
	static __device__ __forceinline__ void Mma(int32_t acc[4], const uint32_t a[4], const uint32_t b[2], uint32_t, uint32_t)
	{
		LmMmaS8(acc,a,b);
	}
};
