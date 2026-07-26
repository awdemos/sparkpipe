// DeepSeek V4. The whole model, one translation unit.
//
// One KV head at 512 dims is a latent in all but name, so this uses the same
// LmKvLatent geometry GLM 5.2 does. The sparse index is the same mechanism at a
// quarter the top-k, and the sliding window rides the same selected-position
// array. Nothing here is a new kernel.
#include "runtime/gemm.cuh"
#include "kernels/norm.cuh"
#include "kernels/attn.cuh"
#include "kernels/topk.cuh"
#include "kernels/formats/fp8.cuh"
#include "kernels/formats/int7.cuh"
#include "llms/deepseek_v4/config.h"

using Dsv4Kv = LmKvLatent<DSV4_KV_BITS, DSV4_HEAD_DIM, DSV4_ROPE_DIM, DSV4_KV_PAGE_SLOTS>;
static_assert(Dsv4Kv::kSlotBytes == 1152u, "512 + 64 latent elements at bf16");

#define DSV4_TILE_N 128u
#define DSV4_STAGES 2u
#define DSV4_WARPS 8u
#define DSV4_THREADS 256u

template __global__ void LmGemmKernel<LmFp8, 16u, DSV4_TILE_N, 128u, DSV4_STAGES, DSV4_WARPS>(__grid_constant__ const LmGemmArguments, LmTileSource, LmTileSource, bool);
template __global__ void LmGemmKernel<LmFp8, 32u, DSV4_TILE_N, 128u, DSV4_STAGES, DSV4_WARPS>(__grid_constant__ const LmGemmArguments, LmTileSource, LmTileSource, bool);
template __global__ void LmGemmKernel<LmFp8, 64u, DSV4_TILE_N, 128u, DSV4_STAGES, DSV4_WARPS>(__grid_constant__ const LmGemmArguments, LmTileSource, LmTileSource, bool);
template __global__ void LmFusedResidualRmsNormKernel<DSV4_THREADS>(const uint16_t *, const uint16_t *, const uint16_t *, uint16_t *, uint16_t *, uint32_t, float);
template __global__ void LmSiluMulKernel<DSV4_THREADS>(const uint16_t *, uint16_t *, uint32_t, bool);
template __global__ void LmQuantiseRowsKernel<LmFp8, DSV4_THREADS>(const uint16_t *, const uint32_t *, uint8_t *, uint8_t *, uint32_t, uint32_t);
template __global__ void LmRopeKernel<DSV4_THREADS>(uint16_t *, const uint32_t *, uint32_t, uint32_t, uint32_t, float);
template __global__ void LmAttentionDecodeKernel<Dsv4Kv, DSV4_THREADS, DSV4_HEAD_DIM, DSV4_ROPE_DIM>(const uint16_t *, const uint16_t *, LmKvView, const uint32_t *, const uint32_t *, const uint32_t *, uint32_t, uint32_t, float, uint16_t *);
template __global__ void LmSparseScoreKernel<Dsv4Kv, DSV4_THREADS, DSV4_INDEX_DIM>(const uint16_t *, LmKvView, const uint32_t *, const uint32_t *, uint32_t, float *);
template __global__ void LmTopkSmallKernel<DSV4_THREADS, DSV4_TOP_K>(const float *, uint32_t, uint32_t *, float *, float);

extern "C" int32_t Dsv4GemmFp8(LmGemmArguments *a, const void *x, const void *w,
	uint32_t rows, uint32_t tokens, uint32_t groups, uint32_t k, uint32_t n,
	uint32_t sms, bool grouped, cudaStream_t s)
{
	return(LmGemmLaunch<LmFp8,DSV4_TILE_N,128u,DSV4_STAGES,DSV4_WARPS>(a,x,w,rows,tokens,DSV4_TOP_K,groups,k,n,sms,grouped,s));
}
