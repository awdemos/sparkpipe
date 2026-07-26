// GLM 5.2. The whole model, one translation unit.
//
// A unity build is not a compile-time trick here, it is the link surface. The
// old tree reached its kernels through extern "C" seams, a dlopen plugin, a
// generated kernel table and an AOT object pack; every one of those is a way for
// a build to succeed and a link to fail, or to succeed and dispatch to the wrong
// thing. One TU per model has none of them.
//
// This file does three things and should never do a fourth:
//   1. name the geometries this model uses
//   2. instantiate the kernels for the buckets it supports
//   3. expose the entry points runtime/ calls
//
// If something cannot be expressed as a parameter to kernels/, it goes in its
// own file in this directory with a comment saying why. That should be rare.

#include "kernels/gemm.cuh"
#include "kernels/formats/fp8.cuh"
#include "kernels/formats/int7.cuh"
#include "kernels/formats/int6.cuh"
#include "kernels/formats/nvfp4.cuh"
#include "kernels/formats/bf16.cuh"
#include "kernels/kv.cuh"
#include "llms/glm5_2/config.h"

// -- geometries --------------------------------------------------------------

using Glm52Kv = LmKvLatent<GLM52_KV_BITS, GLM52_LATENT, GLM52_ROPE_DIM, GLM52_KV_PAGE_SLOTS>;

static_assert(Glm52Kv::kSlotBytes == GLM52_KV_SLOT_BYTES,
	"config.h and kernels/kv.cuh disagree about the slot size");

// -- GEMM instantiations -----------------------------------------------------
//
// One per tile height the bucket selector can produce. Rows per expert is
// batch/32 at 256 experts and top-8, so B1024 needs a 64-row tile; a 16-row tile
// there would split every expert and double the weight stream, which is 96
// percent of all traffic on this path.
//
// TILE_K differs by format because the swizzle span is in BYTES: NVFP4 at 4 bits
// needs 256 elements where FP8 needs 128. kernels/tile.cuh asserts it rather
// than trusting this file to remember.

#define GLM52_TILE_N 128u
#define GLM52_STAGES 2u
#define GLM52_WARPS 8u

// Weight formats this model can be served with. All use DYNAMIC shared memory,
// so none is limited by the 48 KB static cap; the launcher requests
// LmGemmSharedBytes<...>() through cudaFuncAttributeMaxDynamicSharedMemorySize
// and passes it at launch. Each is a decoder into the same
// BF16 fragment, so these six instantiations differ only in how many bits cross
// the bus and how a code becomes a number - not in what the kernel does.
//
// TILE_K is in ELEMENTS and must make a whole swizzle span of the STORED width,
// which kernels/tile.cuh asserts.
//
// __grid_constant__ is repeated here because nvcc requires the annotation on an
// explicit instantiation to match the primary template. Omitting it is an error,
// not a silent difference, which is the good outcome.

template __global__ void LmGemmKernel<LmFp8,   16u, GLM52_TILE_N, 128u, GLM52_STAGES, GLM52_WARPS>(__grid_constant__ const LmGemmArguments, LmTileSource, LmTileSource, bool);
template __global__ void LmGemmKernel<LmInt6,  16u, GLM52_TILE_N, 128u, GLM52_STAGES, GLM52_WARPS>(__grid_constant__ const LmGemmArguments, LmTileSource, LmTileSource, bool);
template __global__ void LmGemmKernel<LmNvfp4, 16u, GLM52_TILE_N, 128u, GLM52_STAGES, GLM52_WARPS>(__grid_constant__ const LmGemmArguments, LmTileSource, LmTileSource, bool);
template __global__ void LmGemmKernel<LmFp8,   64u, GLM52_TILE_N, 128u, GLM52_STAGES, GLM52_WARPS>(__grid_constant__ const LmGemmArguments, LmTileSource, LmTileSource, bool);
template __global__ void LmGemmKernel<LmInt7,  16u, GLM52_TILE_N, 256u, GLM52_STAGES, GLM52_WARPS>(__grid_constant__ const LmGemmArguments, LmTileSource, LmTileSource, bool);
template __global__ void LmGemmKernel<LmBf16Format, 16u, GLM52_TILE_N, 128u, GLM52_STAGES, GLM52_WARPS>(__grid_constant__ const LmGemmArguments, LmTileSource, LmTileSource, bool);
