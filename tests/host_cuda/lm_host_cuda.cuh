#pragma once

// Run this tree's CUDA kernels on a CPU, single-threaded.
//
// We are developing without hardware, so review and emulation are the whole of
// the verification budget. Emulation is the stronger half: a review says the
// arithmetic looks right, a run says it produced these numbers and the
// reference produced those.
//
// WHY ONE THREAD PER BLOCK WORKS, which is what makes this ten lines rather
// than a fibre scheduler. Every kernel in inference/kernels writes its work
// loops as
//
//     for (index = threadIdx.x; index < N; index += THREADS)
//
// so at THREADS == 1 a single thread covers the whole range in order.
// __syncthreads() is then trivially satisfied, and shared memory is just an
// array. Tree reductions written as
//
//     for (stride = THREADS / 2u; stride > 0u; stride >>= 1)
//
// start at stride 0 and do not execute, leaving shared[0] holding the one
// thread's accumulation - which is already the total. The kernels are correct
// at THREADS == 1 by construction, not by accident, because that loop shape is
// the house style.
//
// WHAT THIS CANNOT CATCH, stated so nobody reads a pass as more than it is:
// races between threads, warp-level assumptions, shared-memory bank conflicts,
// anything about occupancy or launch overhead, and any bug that only appears
// when THREADS > 1. It catches arithmetic, indexing, and argument order - which
// is where every defect this tree has found so far actually lived.

#include <math.h>
#include "cuda_fp16.h"
#include <string.h>
#include <stdint.h>

#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __launch_bounds__(...)
#define __restrict__

struct LmHostDim3
{
	unsigned x, y, z;
	LmHostDim3(unsigned a = 1u, unsigned b = 1u, unsigned c = 1u) : x(a), y(b), z(c) {}
};

typedef LmHostDim3 dim3;

// The grid position the harness is currently emulating. A kernel reads these
// exactly as it would on a device; the harness sets them and calls the kernel
// body directly, because <<< >>> is not C++ and no host compiler will parse it.
extern LmHostDim3 blockIdx;
extern LmHostDim3 threadIdx;
extern LmHostDim3 blockDim;
extern LmHostDim3 gridDim;

// Plain locals, not statics: with one thread per block each invocation wants
// its own storage, and a static would leak the previous block's values into the
// next one. Dynamic shared memory - extern __shared__ T name[] - is backed by
// the fixed buffers below, which is why __shared__ must expand to nothing
// rather than to a storage class an extern declaration cannot carry.
#define __shared__
#define LM_HOST_SHARED_BYTES 65536u
#define __syncthreads() ((void)0)
#define __syncwarp(...) ((void)0)

// The fast-math intrinsics, mapped to the accurate ones. A device __expf is an
// approximation and this is not, so results differ in the last bits - which is
// the right trade for a correctness harness and the wrong one for a bit-exactness
// claim. No test here asserts bit-exactness.
#define __expf(x) expf(x)
#define __logf(x) logf(x)
#define __powf(x, y) powf((x), (y))
#define __cosf(x) cosf(x)
#define __sinf(x) sinf(x)
#define __tanf(x) tanf(x)
#define __fdividef(x, y) ((x) / (y))

static inline float rsqrtf(float value) { return 1.0f / sqrtf(value); }

// The bit-reinterpret intrinsics and the small vector types dtype.cuh uses.
static inline float __uint_as_float(unsigned bits)
{
	float out; memcpy(&out, &bits, sizeof(out)); return out;
}
static inline unsigned __float_as_uint(float value)
{
	unsigned bits; memcpy(&bits, &value, sizeof(bits)); return bits;
}
struct float2 { float x, y; };
struct float4 { float x, y, z, w; };
struct uint2 { unsigned x, y; };
struct uint4 { unsigned x, y, z, w; };
static inline float2 make_float2(float a, float b) { float2 v; v.x = a; v.y = b; return v; }
static inline __half __ushort_as_half(unsigned short bits) { __half h; h.raw = bits; return h; }
static inline unsigned short __half_as_ushort(__half h) { return h.raw; }
static inline float __shfl_down_sync(unsigned, float value, unsigned, int = 32) { return value; }
static inline float __shfl_sync(unsigned, float value, int, int = 32) { return value; }
static inline unsigned __ballot_sync(unsigned, int) { return 0u; }
static inline void __threadfence_block(void) {}

// Atomics are ordinary reads and writes with one thread. That is exactly the
// sequential schedule a correct kernel must also be valid under, so a race this
// cannot see is a race the harness is honest about not seeing - see the shim's
// header note.
static inline unsigned atomicAdd(unsigned *address, unsigned value)
{
	unsigned old = *address; *address = old + value; return old;
}
static inline int atomicAdd(int *address, int value)
{
	int old = *address; *address = old + value; return old;
}
static inline float atomicAdd(float *address, float value)
{
	float old = *address; *address = old + value; return old;
}
static inline unsigned atomicMax(unsigned *address, unsigned value)
{
	unsigned old = *address; if (value > old) *address = value; return old;
}

typedef int cudaStream_t;
typedef int cudaError_t;
#define cudaSuccess 0
static inline cudaError_t cudaPeekAtLastError(void) { return cudaSuccess; }
static inline cudaError_t cudaGetLastError(void) { return cudaSuccess; }

// Emulate one launch. The kernel is invoked once per block with threadIdx.x
// fixed at zero, in block order, which is the sequential schedule a correct
// kernel must also be valid under.
#define LM_HOST_LAUNCH(grid, kernel_call)                                     \
	do {                                                                      \
		dim3 lm_grid = (grid);                                                \
		blockDim = dim3(1u, 1u, 1u);                                          \
		gridDim = lm_grid;                                                    \
		threadIdx = dim3(0u, 0u, 0u);                                         \
		for (unsigned lm_z = 0u; lm_z < lm_grid.z; ++lm_z)                    \
			for (unsigned lm_y = 0u; lm_y < lm_grid.y; ++lm_y)                \
				for (unsigned lm_x = 0u; lm_x < lm_grid.x; ++lm_x)            \
				{                                                             \
					blockIdx = dim3(lm_x, lm_y, lm_z);                        \
					kernel_call;                                              \
				}                                                             \
	} while (0)
