// DeepSeek V4. The whole model, one translation unit.
//
// One KV head at 512 dims is a latent in all but name, so this uses the same
// LmKvLatent geometry GLM 5.2 does. The sparse index is the same mechanism at a
// quarter the top-k, and the sliding window rides the same selected-position
// array. Nothing here is a new kernel.
#include "runtime/gemm.cuh"
#include "inference/kernels/norm.cuh"
#include "inference/kernels/attn.cuh"
#include "inference/kernels/topk.cuh"
#include "inference/kernels/formats/fp8.cuh"
#include "inference/kernels/formats/int7.cuh"
#include "inference/kernels/formats/bf16.cuh"
#include "inference/llms/deepseek_v4/config.h"
#include "inference/llms/deepseek_v4/layer.cuh"

static_assert(Dsv4Kv::kSlotBytes == 1152u, "512 + 64 latent elements at bf16");

#define DSV4_TILE_N 128u
#define DSV4_STAGES 2u
#define DSV4_WARPS 8u
#define DSV4_THREADS 256u

template __global__ void LmGemmKernel<LmFp8, 16u, DSV4_TILE_N, 128u, DSV4_STAGES, DSV4_WARPS>(__grid_constant__ const LmGemmArguments, LmTileSource, LmTileSource, bool);
template __global__ void LmGemmKernel<LmFp8, 32u, DSV4_TILE_N, 128u, DSV4_STAGES, DSV4_WARPS>(__grid_constant__ const LmGemmArguments, LmTileSource, LmTileSource, bool);
template __global__ void LmGemmKernel<LmFp8, 64u, DSV4_TILE_N, 128u, DSV4_STAGES, DSV4_WARPS>(__grid_constant__ const LmGemmArguments, LmTileSource, LmTileSource, bool);
template __global__ void LmFusedResidualRmsNormKernel<DSV4_THREADS,uint16_t>(const uint16_t *, const uint16_t *, const uint16_t *, uint16_t *, uint16_t *, uint32_t, uint32_t, float);
template __global__ void LmSiluMulKernel<DSV4_THREADS>(const uint16_t *, uint16_t *, uint32_t, bool);
template __global__ void LmQuantiseRowsKernel<LmFp8, DSV4_THREADS>(const uint16_t *, const uint32_t *, uint8_t *, uint8_t *, uint32_t, uint32_t);
template __global__ void LmRopeKernel<DSV4_THREADS,LM_ROPE_INTERLEAVED>(uint16_t *, const uint32_t *, uint32_t, uint32_t, uint32_t, float);
template __global__ void LmAttentionDecodeKernel<Dsv4Kv, DSV4_THREADS, DSV4_HEAD_DIM, DSV4_ROPE_DIM>(const uint16_t *, const uint16_t *, LmKvView, const uint32_t *, const uint32_t *, const uint32_t *, uint32_t, uint32_t, float, uint16_t *, const uint32_t *);
template __global__ void LmSparseScoreKernel<Dsv4Kv, DSV4_THREADS, DSV4_INDEX_DIM>(const uint16_t *, LmKvView, const uint32_t *, const uint32_t *, uint32_t, float *);
template __global__ void LmRopeYarnKernel<DSV4_THREADS,LM_ROPE_INTERLEAVED>(uint16_t *, const uint32_t *, uint32_t, uint32_t, uint32_t, float, float, float, float, float);
template __global__ void LmKvStoreKernel<Dsv4Kv, DSV4_THREADS>(LmKvView, const uint16_t *, const uint32_t *, const uint32_t *, uint32_t, uint32_t);
template __global__ void LmAddRowsKernel<DSV4_THREADS>(const uint16_t *, const uint16_t *, uint16_t *, uint32_t, uint32_t);
template __global__ void LmTopkHistogramKernel<DSV4_THREADS>(const float *, uint32_t, uint32_t, uint32_t *);
template __global__ void LmTopkGatherKernel<DSV4_THREADS>(const float *, uint32_t, uint32_t, const uint32_t *, uint32_t *, uint32_t *);
template __global__ void LmMoeFinalizeKernel<DSV4_THREADS>(const uint16_t *, const uint32_t *, const float *, uint16_t *, uint32_t, uint32_t, uint32_t);
template __global__ void LmHeadCandidateKernel<DSV4_THREADS, 1024u>(const uint16_t *, const uint16_t *, const uint32_t *, float *, uint32_t *, uint32_t, uint32_t, uint32_t);
template __global__ void LmHeadCommitKernel<DSV4_THREADS>(const float *, const uint32_t *, uint32_t, uint32_t *, float *, uint32_t);
template __global__ void LmTopkSmallKernel<DSV4_THREADS, DSV4_TOP_K>(const float *, uint32_t, uint32_t *, float *, const float *, const uint16_t *);

extern "C" int32_t Dsv4GemmFp8(LmGemmArguments *a, const void *x, const void *w,
	uint32_t rows, uint32_t tokens, uint32_t groups, uint32_t k, uint32_t n,
	uint32_t sms, bool grouped, cudaStream_t s)
{
	return(LmGemmLaunch<LmFp8,DSV4_TILE_N,128u,DSV4_STAGES,DSV4_WARPS>(a,x,w,rows,tokens,DSV4_TOP_K,groups,k,n,sms,grouped,s));
}

extern "C" int32_t Dsv4LayerAttentionFp8(const Dsv4LayerBuffers *b, uint32_t rows, uint32_t context, uint32_t sms, cudaStream_t stream)
{
	return(Dsv4LayerAttention<LmFp8>(b,rows,context,sms,stream));
}

extern "C" int32_t Dsv4LayerMoeFp8(const Dsv4LayerBuffers *b, uint32_t rows, uint32_t packed_rows, uint32_t sms, cudaStream_t stream)
{
	return(Dsv4LayerMoe<LmFp8>(b,rows,packed_rows,sms,stream));
}

extern "C" int32_t Dsv4HeadFullVocab(const Dsv4LayerBuffers *b, const void *norm_weight, const void *head_weight, uint32_t rows, cudaStream_t stream)
{
	return(Dsv4Head(b,norm_weight,head_weight,0,DSV4_VOCAB,rows,stream));
}

extern "C" int32_t Dsv4HeadRestricted(const Dsv4LayerBuffers *b, const void *norm_weight, const void *head_weight, const uint32_t *token_ids, uint32_t count, uint32_t rows, cudaStream_t stream)
{
	return(Dsv4Head(b,norm_weight,head_weight,token_ids,count,rows,stream));
}

