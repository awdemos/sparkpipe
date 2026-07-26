// Qwen 3.6. The whole model, one translation unit.
//
// Gated DeltaNet on 48 of 64 layers, full attention on 16. The same two-pool
// shape K3 has, reached from a different vendor - which is the argument that
// this is an architecture class and not one company's choice, and the reason
// kernels/kv.cuh parameterises growth rather than special-casing it.
//
// The GDN state carries a short causal convolution window alongside the
// delta-rule matrix. Both are per-sequence, neither grows, so both live in one
// non-growing slot and the pool does not need to know which is which.
#include "runtime/gemm.cuh"
#include "kernels/norm.cuh"
#include "kernels/attn.cuh"
#include "kernels/topk.cuh"
#include "kernels/formats/fp8.cuh"
#include "kernels/formats/int7.cuh"
#include "llms/qwen_3_6/config.h"

using Qwen36FullKv = LmKvHeads<QWEN36_KV_BITS, 8u, 128u, QWEN36_KV_PAGE_SLOTS>;
using Qwen36GdnState = LmKvState<QWEN36_GDN_STATE_BYTES>;

static_assert(Qwen36GdnState::kGrows == false, "GDN state is fixed per sequence");
static_assert(QWEN36_LAYER_IS_LINEAR(0) && !QWEN36_LAYER_IS_LINEAR(3),
	"period 4, full attention in phase 3");

#define QWEN36_TILE_N 128u
#define QWEN36_STAGES 2u
#define QWEN36_WARPS 8u
#define QWEN36_THREADS 256u

template __global__ void LmGemmKernel<LmFp8, 16u, QWEN36_TILE_N, 128u, QWEN36_STAGES, QWEN36_WARPS>(__grid_constant__ const LmGemmArguments, LmTileSource, LmTileSource, bool);
template __global__ void LmGemmKernel<LmFp8, 32u, QWEN36_TILE_N, 128u, QWEN36_STAGES, QWEN36_WARPS>(__grid_constant__ const LmGemmArguments, LmTileSource, LmTileSource, bool);
template __global__ void LmGemmKernel<LmFp8, 64u, QWEN36_TILE_N, 128u, QWEN36_STAGES, QWEN36_WARPS>(__grid_constant__ const LmGemmArguments, LmTileSource, LmTileSource, bool);
template __global__ void LmFusedResidualRmsNormKernel<QWEN36_THREADS>(const uint16_t *, const uint16_t *, const uint16_t *, uint16_t *, uint16_t *, uint32_t, float);
template __global__ void LmSiluMulKernel<QWEN36_THREADS>(const uint16_t *, uint16_t *, uint32_t, bool);
template __global__ void LmQuantiseRowsKernel<LmFp8, QWEN36_THREADS>(const uint16_t *, const uint32_t *, uint8_t *, uint8_t *, uint32_t, uint32_t);
template __global__ void LmRopeKernel<QWEN36_THREADS>(uint16_t *, const uint32_t *, uint32_t, uint32_t, uint32_t, float);
template __global__ void LmAttentionDecodeKernel<Qwen36FullKv, QWEN36_THREADS, 128u, 64u>(const uint16_t *, const uint16_t *, LmKvView, const uint32_t *, const uint32_t *, const uint32_t *, uint32_t, uint32_t, float, uint16_t *);

extern "C" int32_t Qwen36GemmFp8(LmGemmArguments *a, const void *x, const void *w,
	uint32_t rows, uint32_t tokens, uint32_t groups, uint32_t k, uint32_t n,
	uint32_t sms, bool grouped, cudaStream_t s)
{
	return(LmGemmLaunch<LmFp8,QWEN36_TILE_N,128u,QWEN36_STAGES,QWEN36_WARPS>(a,x,w,rows,tokens,8u,groups,k,n,sms,grouped,s));
}
