#pragma once

// Six bits. E3M2 and E2M3, native through kind::f8f6f4.
//
// This is what "6-bit" means on this target. There is no s6 mma and no s7 mma of
// any shape, verified by tests/test_ptx_capability_gate.py rather than assumed -
// a 7-bit scheme would have to unpack to 8 in the kernel, which spends exactly
// the storage saving it was chosen for.
//
// E3M2 has three exponent bits and two mantissa: wider range, coarser steps.
// E2M3 is the reverse. Weights with outliers want E3M2; weights already
// normalised into a narrow band want E2M3. Both are here because that choice
// belongs to whoever quantised the checkpoint, not to the kernel.
//
// TILING IS AWKWARD AND THE COMPILER SAYS SO. Six bits is 0.75 bytes per
// element, so a K tile is a whole 128-byte swizzle span only when TILE_K is
// divisible by 512 - against 128 for 8-bit and 256 for 4-bit. The first
// workable depth is 512 elements, 384 bytes, three spans:
//
//     TILE_K   bytes   span   TILE_M=16   TILE_M=64
//        128      96     no      27,680      36,896
//        256     192     no      55,328      73,760
//        512     384    YES     110,624     147,488  over 128 KB
//
// So fp6 runs at TILE_M=16 and nowhere else on this target, which caps the
// batch it is useful at. That is a property of the format, not a bug, and the
// static_assert in kernels/gemm.cuh reports it at the instantiation rather than
// letting a wrong tile reach the ring. It caught this exact case during the
// first attempt to instantiate fp6 at TILE_K=128.
//
// Operands occupy a byte each in registers regardless; only the stored form is
// narrower, which is why kBits is 6 and the tile arithmetic follows it.

#include "kernels/mma.cuh"

struct LmE3m2
{
	typedef float Accumulator;
	static constexpr uint32_t kBits = 6u;
	static constexpr uint32_t kMmaM = LM_MMA8_M;
	static constexpr uint32_t kMmaN = LM_MMA8_N;
	static constexpr uint32_t kMmaK = LM_MMA8_K;
	static constexpr bool kScaleInMma = false;
	static constexpr uint32_t kScaleGroup = 128u;
	static constexpr float kMax = 28.0f;

	static __device__ __forceinline__ uint32_t OperandARow(uint32_t lane, uint32_t reg) { return(LmMma8OperandARow(lane,reg)); }
	static __device__ __forceinline__ uint32_t OperandAByte(uint32_t lane, uint32_t reg) { return(LmMma8OperandAByte(lane,reg)); }
	static __device__ __forceinline__ uint32_t OperandBRow(uint32_t lane) { return(LmMma8OperandBRow(lane)); }
	static __device__ __forceinline__ uint32_t OperandBByte(uint32_t lane, uint32_t reg) { return(LmMma8OperandBByte(lane,reg)); }
	static __device__ __forceinline__ void Mma(float acc[4], const uint32_t a[4], const uint32_t b[2], uint32_t, uint32_t)
	{
		LmMmaE3m2(acc,a,b);
	}
};

struct LmE2m3 : LmE3m2
{
	static constexpr float kMax = 7.5f;

	static __device__ __forceinline__ void Mma(float acc[4], const uint32_t a[4], const uint32_t b[2], uint32_t, uint32_t)
	{
		LmMmaE2m3(acc,a,b);
	}
};
