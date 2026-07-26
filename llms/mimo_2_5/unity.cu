// MiMo 2.5. The whole model, one translation unit.
//
// The point of this file is what it does NOT contain. No kernels: every one
// comes from kernels/, instantiated here with this model's shapes. The old
// tree's mimo25 decode stage was 2,868 lines of wiring around a 10,385-line
// shared library it could not share with glm52, which had its own 27,268.
//
// Two attention geometries because the branches differ in KV head count, and
// that sizes the slot. Everything else - the GEMM, norm, quantise, RoPE, top-k,
// speculation - is the same instantiation glm5_2 uses, differing only in
// arguments.

#include "runtime/gemm.cuh"
#include "kernels/norm.cuh"
#include "kernels/attn.cuh"
#include "kernels/topk.cuh"
#include "kernels/formats/bf16.cuh"
#include "kernels/formats/fp8.cuh"
#include "kernels/formats/int7.cuh"
#include "llms/mimo_2_5/config.h"

// -- geometries ----------------------------------------------------------------
//
// No latent compression, so LmKvHeads rather than LmKvLatent. Same allocator,
// same page table, same eviction - only the slot size differs, which is exactly
// the parameterisation kernels/kv.cuh exists for.

using Mimo25FullKv = LmKvHeads<MIMO25_KV_BITS, MIMO25_FULL_KV_HEADS, MIMO25_HEAD_DIM, MIMO25_KV_PAGE_SLOTS>;
using Mimo25SwaKv  = LmKvHeads<MIMO25_KV_BITS, MIMO25_SWA_KV_HEADS, MIMO25_HEAD_DIM, MIMO25_KV_PAGE_SLOTS>;

static_assert(Mimo25FullKv::kSlotBytes == 3072u, "4 heads x 192 dim x (k+v) x bf16");
static_assert(Mimo25SwaKv::kSlotBytes == 6144u, "8 heads, twice the slot");

#define MIMO25_TILE_N 128u
#define MIMO25_STAGES 2u
#define MIMO25_WARPS 8u
#define MIMO25_THREADS 256u

// -- kernels -------------------------------------------------------------------
//
// 256 experts at top-8 is the same routing shape as GLM 5.2, so the tile
// selector produces the same heights and these are the same instantiations.

template __global__ void LmGemmKernel<LmFp8,  16u, MIMO25_TILE_N, 128u, MIMO25_STAGES, MIMO25_WARPS>(__grid_constant__ const LmGemmArguments, LmTileSource, LmTileSource, bool);
template __global__ void LmGemmKernel<LmFp8,  32u, MIMO25_TILE_N, 128u, MIMO25_STAGES, MIMO25_WARPS>(__grid_constant__ const LmGemmArguments, LmTileSource, LmTileSource, bool);
template __global__ void LmGemmKernel<LmFp8,  64u, MIMO25_TILE_N, 128u, MIMO25_STAGES, MIMO25_WARPS>(__grid_constant__ const LmGemmArguments, LmTileSource, LmTileSource, bool);
template __global__ void LmGemmKernel<LmInt7, 16u, MIMO25_TILE_N, 256u, MIMO25_STAGES, MIMO25_WARPS>(__grid_constant__ const LmGemmArguments, LmTileSource, LmTileSource, bool);
template __global__ void LmGemmKernel<LmInt7, 32u, MIMO25_TILE_N, 256u, MIMO25_STAGES, MIMO25_WARPS>(__grid_constant__ const LmGemmArguments, LmTileSource, LmTileSource, bool);
template __global__ void LmGemmKernel<LmInt7, 64u, MIMO25_TILE_N, 256u, MIMO25_STAGES, MIMO25_WARPS>(__grid_constant__ const LmGemmArguments, LmTileSource, LmTileSource, bool);

template __global__ void LmFusedResidualRmsNormKernel<MIMO25_THREADS>(const uint16_t *, const uint16_t *, const uint16_t *, uint16_t *, uint16_t *, uint32_t, float);
template __global__ void LmSiluMulKernel<MIMO25_THREADS>(const uint16_t *, uint16_t *, uint32_t, bool);
template __global__ void LmQuantiseRowsKernel<LmFp8, MIMO25_THREADS>(const uint16_t *, const uint32_t *, uint8_t *, uint8_t *, uint32_t, uint32_t);
template __global__ void LmQuantiseRowsKernel<LmInt7, MIMO25_THREADS>(const uint16_t *, const uint32_t *, uint8_t *, uint8_t *, uint32_t, uint32_t);

// One RoPE kernel serves both branches; the theta is an argument, which is the
// whole difference between them and was a second kernel in the old tree.
template __global__ void LmRopeKernel<MIMO25_THREADS>(uint16_t *, const uint32_t *, uint32_t, uint32_t, uint32_t, float);

// Attention instantiated per geometry, because the slot size is compile-time.
// The sliding window is the selected-position array, not a different kernel.
template __global__ void LmAttentionDecodeKernel<Mimo25FullKv, MIMO25_THREADS, MIMO25_HEAD_DIM, MIMO25_ROPE_DIM>(const uint16_t *, const uint16_t *, LmKvView, const uint32_t *, const uint32_t *, const uint32_t *, uint32_t, uint32_t, float, uint16_t *);
template __global__ void LmAttentionDecodeKernel<Mimo25SwaKv, MIMO25_THREADS, MIMO25_HEAD_DIM, MIMO25_ROPE_DIM>(const uint16_t *, const uint16_t *, LmKvView, const uint32_t *, const uint32_t *, const uint32_t *, uint32_t, uint32_t, float, uint16_t *);

template __global__ void LmTopkSmallKernel<MIMO25_THREADS, MIMO25_TOP_K>(const float *, uint32_t, uint32_t *, float *, float);

// -- entry points --------------------------------------------------------------

extern "C" int32_t Mimo25GemmFp8(LmGemmArguments *args, const void *a, const void *b,
	uint32_t packed_rows, uint32_t tokens, uint32_t groups,
	uint32_t k, uint32_t n, uint32_t sms, bool grouped, cudaStream_t stream)
{
	return(LmGemmLaunch<LmFp8,MIMO25_TILE_N,128u,MIMO25_STAGES,MIMO25_WARPS>(
		args,a,b,packed_rows,tokens,MIMO25_TOP_K,groups,k,n,sms,grouped,stream));
}

extern "C" int32_t Mimo25GemmInt7(LmGemmArguments *args, const void *a, const void *b,
	uint32_t packed_rows, uint32_t tokens, uint32_t groups,
	uint32_t k, uint32_t n, uint32_t sms, bool grouped, cudaStream_t stream)
{
	return(LmGemmLaunch<LmInt7,MIMO25_TILE_N,256u,MIMO25_STAGES,MIMO25_WARPS>(
		args,a,b,packed_rows,tokens,MIMO25_TOP_K,groups,k,n,sms,grouped,stream));
}
