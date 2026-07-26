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

#include "runtime/gemm.cuh"
#include "kernels/norm.cuh"
#include "kernels/attn.cuh"
#include "kernels/topk.cuh"
#include "kernels/speculate.cuh"
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













// Every (format, tile height) the dispatch in runtime/gemm.cuh can select.
//
// The switch there and this list must agree. Omitting a height here is a LINK
// error and omitting it there is a runtime error - both loud, neither silent,
// which is the property worth having when the alternative is a kernel that
// launches with the wrong shared-memory size.
//
// TILE_K differs by stored width because the swizzle span is in BYTES: 128
// elements at 8 bits, 256 at 7. kernels/layout.cuh asserts it.

template __global__ void LmGemmKernel<LmFp8, 16u, GLM52_TILE_N, 128u, GLM52_STAGES, GLM52_WARPS>(__grid_constant__ const LmGemmArguments, LmTileSource, LmTileSource, bool);
template __global__ void LmGemmKernel<LmFp8, 32u, GLM52_TILE_N, 128u, GLM52_STAGES, GLM52_WARPS>(__grid_constant__ const LmGemmArguments, LmTileSource, LmTileSource, bool);
template __global__ void LmGemmKernel<LmFp8, 64u, GLM52_TILE_N, 128u, GLM52_STAGES, GLM52_WARPS>(__grid_constant__ const LmGemmArguments, LmTileSource, LmTileSource, bool);
template __global__ void LmGemmKernel<LmInt7, 16u, GLM52_TILE_N, 256u, GLM52_STAGES, GLM52_WARPS>(__grid_constant__ const LmGemmArguments, LmTileSource, LmTileSource, bool);
template __global__ void LmGemmKernel<LmInt7, 32u, GLM52_TILE_N, 256u, GLM52_STAGES, GLM52_WARPS>(__grid_constant__ const LmGemmArguments, LmTileSource, LmTileSource, bool);
template __global__ void LmGemmKernel<LmInt7, 64u, GLM52_TILE_N, 256u, GLM52_STAGES, GLM52_WARPS>(__grid_constant__ const LmGemmArguments, LmTileSource, LmTileSource, bool);
template __global__ void LmGemmKernel<LmInt6, 16u, GLM52_TILE_N, 128u, GLM52_STAGES, GLM52_WARPS>(__grid_constant__ const LmGemmArguments, LmTileSource, LmTileSource, bool);
template __global__ void LmGemmKernel<LmInt6, 32u, GLM52_TILE_N, 128u, GLM52_STAGES, GLM52_WARPS>(__grid_constant__ const LmGemmArguments, LmTileSource, LmTileSource, bool);
template __global__ void LmGemmKernel<LmInt6, 64u, GLM52_TILE_N, 128u, GLM52_STAGES, GLM52_WARPS>(__grid_constant__ const LmGemmArguments, LmTileSource, LmTileSource, bool);
template __global__ void LmGemmKernel<LmNvfp4, 16u, GLM52_TILE_N, 128u, GLM52_STAGES, GLM52_WARPS>(__grid_constant__ const LmGemmArguments, LmTileSource, LmTileSource, bool);
template __global__ void LmGemmKernel<LmNvfp4, 32u, GLM52_TILE_N, 128u, GLM52_STAGES, GLM52_WARPS>(__grid_constant__ const LmGemmArguments, LmTileSource, LmTileSource, bool);
template __global__ void LmGemmKernel<LmNvfp4, 64u, GLM52_TILE_N, 128u, GLM52_STAGES, GLM52_WARPS>(__grid_constant__ const LmGemmArguments, LmTileSource, LmTileSource, bool);

#define GLM52_NORM_THREADS 256u

// Norm, activation and quantise. Generic kernels with this model's dimensions
// passed at the call, not baked in: the old decode stage's RmsNormKernel was 62
// lines carrying seven SPARK_GLM52_MODEL_* references, none of which changed
// what it computed.
template __global__ void LmFusedResidualRmsNormKernel<GLM52_NORM_THREADS>(const uint16_t *, const uint16_t *, const uint16_t *, uint16_t *, uint16_t *, uint32_t, float);
template __global__ void LmSiluMulKernel<GLM52_NORM_THREADS>(const uint16_t *, uint16_t *, uint32_t, bool);
template __global__ void LmQuantiseRowsKernel<LmFp8, GLM52_NORM_THREADS>(const uint16_t *, const uint32_t *, uint8_t *, uint8_t *, uint32_t, uint32_t);
template __global__ void LmQuantiseRowsKernel<LmInt7, GLM52_NORM_THREADS>(const uint16_t *, const uint32_t *, uint8_t *, uint8_t *, uint32_t, uint32_t);
template __global__ void LmQuantiseRowsKernel<LmInt6, GLM52_NORM_THREADS>(const uint16_t *, const uint32_t *, uint8_t *, uint8_t *, uint32_t, uint32_t);
template __global__ void LmQuantiseRowsKernel<LmNvfp4, GLM52_NORM_THREADS>(const uint16_t *, const uint32_t *, uint8_t *, uint8_t *, uint32_t, uint32_t);

// Attention. Generic over the cache geometry, so the same three kernels serve
// any model whose cache is a paged pool of opaque slots - which after
// kernels/kv.cuh is all of them.
//
// The sparse path is the dense path with a selected-position array. Passing null
// makes it dense; the old tree had them as separate kernels and the difference
// was which positions the loop visited.
template __global__ void LmRopeKernel<GLM52_NORM_THREADS>(uint16_t *, const uint32_t *, uint32_t, uint32_t, uint32_t, float);
template __global__ void LmAttentionDecodeKernel<Glm52Kv, GLM52_NORM_THREADS, GLM52_LATENT, GLM52_ROPE_DIM>(const uint16_t *, const uint16_t *, LmKvView, const uint32_t *, const uint32_t *, const uint32_t *, uint32_t, uint32_t, float, uint16_t *);
template __global__ void LmSparseScoreKernel<Glm52Kv, GLM52_NORM_THREADS, GLM52_DSA_INDEX_DIM>(const uint16_t *, LmKvView, const uint32_t *, const uint32_t *, uint32_t, float *);

// Top-k and speculation. The router picks 8 of 256 with the small path; DSA
// picks its positions with the radix path. Both were separate kernels before and
// the algorithm was the same.
template __global__ void LmTopkSmallKernel<GLM52_NORM_THREADS, GLM52_TOP_K>(const float *, uint32_t, uint32_t *, float *, float);
template __global__ void LmTopkHistogramKernel<GLM52_NORM_THREADS>(const float *, uint32_t, uint32_t, uint32_t *);
template __global__ void LmTopkGatherKernel<GLM52_NORM_THREADS>(const float *, uint32_t, uint32_t, const uint32_t *, uint32_t *, uint32_t *);
template __global__ void LmSpeculativeVerifyGreedyKernel<GLM52_NORM_THREADS>(const uint32_t *, const uint32_t *, uint32_t, uint32_t *, uint32_t *, uint32_t *);
template __global__ void LmSpeculativeVerifySampledKernel<GLM52_NORM_THREADS>(const uint32_t *, const float *, const float *, const float *, uint32_t, uint32_t, uint32_t *, uint32_t *, uint32_t *);

// -- entry points ------------------------------------------------------------
//
// What runtime/ calls. One per weight format; the tile height is chosen inside
// from the token count, so a caller never picks one.
//
// extern "C" because the scheduler is C. This is the whole ABI surface of the
// model - four functions and a plan struct - against the old tree's extern "C"
// seams, dlopen plugin, generated kernel table and AOT object pack.

extern "C" int32_t Glm52GemmFp8(LmGemmArguments *args, const void *a, const void *b,
	uint32_t packed_rows, uint32_t tokens, uint32_t groups,
	uint32_t k, uint32_t n, uint32_t sms, bool grouped, cudaStream_t stream)
{
	return(LmGemmLaunch<LmFp8,GLM52_TILE_N,128u,GLM52_STAGES,GLM52_WARPS>(
		args,a,b,packed_rows,tokens,GLM52_TOP_K,groups,k,n,sms,grouped,stream));
}

extern "C" int32_t Glm52GemmInt7(LmGemmArguments *args, const void *a, const void *b,
	uint32_t packed_rows, uint32_t tokens, uint32_t groups,
	uint32_t k, uint32_t n, uint32_t sms, bool grouped, cudaStream_t stream)
{
	return(LmGemmLaunch<LmInt7,GLM52_TILE_N,256u,GLM52_STAGES,GLM52_WARPS>(
		args,a,b,packed_rows,tokens,GLM52_TOP_K,groups,k,n,sms,grouped,stream));
}

extern "C" int32_t Glm52GemmInt6(LmGemmArguments *args, const void *a, const void *b,
	uint32_t packed_rows, uint32_t tokens, uint32_t groups,
	uint32_t k, uint32_t n, uint32_t sms, bool grouped, cudaStream_t stream)
{
	return(LmGemmLaunch<LmInt6,GLM52_TILE_N,128u,GLM52_STAGES,GLM52_WARPS>(
		args,a,b,packed_rows,tokens,GLM52_TOP_K,groups,k,n,sms,grouped,stream));
}

extern "C" int32_t Glm52GemmNvfp4(LmGemmArguments *args, const void *a, const void *b,
	uint32_t packed_rows, uint32_t tokens, uint32_t groups,
	uint32_t k, uint32_t n, uint32_t sms, bool grouped, cudaStream_t stream)
{
	return(LmGemmLaunch<LmNvfp4,GLM52_TILE_N,128u,GLM52_STAGES,GLM52_WARPS>(
		args,a,b,packed_rows,tokens,GLM52_TOP_K,groups,k,n,sms,grouped,stream));
}
