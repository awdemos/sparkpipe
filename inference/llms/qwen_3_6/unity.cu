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
#include "inference/kernels/norm.cuh"
#include "inference/kernels/attn.cuh"
#include "inference/kernels/topk.cuh"
#include "inference/kernels/linear_attn.cuh"
#include "inference/kernels/head.cuh"
#include "inference/kernels/formats/fp8.cuh"
#include "inference/kernels/formats/int7.cuh"
#include "inference/llms/qwen_3_6/layer.cuh"


static_assert(Qwen36GdnState::kGrows == false, "GDN state is fixed per sequence");
static_assert(QWEN36_NOPE_DIM + QWEN36_ROPE_DIM == QWEN36_HEAD_DIM,
	"the decode kernel splits a head into nope and rope; they must be the head");
static_assert(QWEN36_LAYER_IS_LINEAR(0) && !QWEN36_LAYER_IS_LINEAR(3),
	"period 4, full attention in phase 3");

#define QWEN36_TILE_N 128u
#define QWEN36_STAGES 2u
#define QWEN36_WARPS 8u
#define QWEN36_THREADS 256u

template __global__ void LmGemmKernel<LmFp8, 16u, QWEN36_TILE_N, 128u, QWEN36_STAGES, QWEN36_WARPS>(__grid_constant__ const LmGemmArguments, LmTileSource, LmTileSource, bool);
template __global__ void LmGemmKernel<LmFp8, 32u, QWEN36_TILE_N, 128u, QWEN36_STAGES, QWEN36_WARPS>(__grid_constant__ const LmGemmArguments, LmTileSource, LmTileSource, bool);
template __global__ void LmGemmKernel<LmFp8, 64u, QWEN36_TILE_N, 128u, QWEN36_STAGES, QWEN36_WARPS>(__grid_constant__ const LmGemmArguments, LmTileSource, LmTileSource, bool);
template __global__ void LmFusedResidualRmsNormKernel<QWEN36_THREADS>(const uint16_t *, const uint16_t *, const uint16_t *, uint16_t *, uint16_t *, uint32_t, uint32_t, float);
template __global__ void LmSiluMulKernel<QWEN36_THREADS>(const uint16_t *, uint16_t *, uint32_t, bool);
template __global__ void LmQuantiseRowsKernel<LmFp8, QWEN36_THREADS>(const uint16_t *, const uint32_t *, uint8_t *, uint8_t *, uint32_t, uint32_t);
template __global__ void LmRopePerHeadKernel<QWEN36_THREADS>(uint16_t *, const uint32_t *, uint32_t, uint32_t, uint32_t, float);
template __global__ void LmSplitQkvKernel<QWEN36_THREADS>(const uint16_t *, LmQkvLayout, uint16_t *, uint16_t *, uint16_t *, uint32_t, float);
// The linear layers. 48 of 64, with a fixed state instead of a growing cache.
//
// The state and the convolution window share one non-growing slot, which is why
// QWEN36_GDN_STATE_BYTES is their sum and kernels/kv.cuh sizes the pool from it.
template __global__ void LmDeltaRuleKernel<QWEN36_THREADS, QWEN36_GDN_KEY_DIM, QWEN36_GDN_VALUE_DIM>(uint8_t *, uint32_t, const uint32_t *, const uint32_t *, const uint32_t *, const uint16_t *, const uint16_t *, const uint16_t *, const float *, const float *, uint16_t *, uint32_t, uint32_t, uint32_t, uint32_t);
template __global__ void LmCausalConvKernel<QWEN36_THREADS, QWEN36_GDN_CONV_KERNEL, LM_CONV_SWISH>(uint16_t *, const uint32_t *, const uint32_t *, const uint32_t *, const uint16_t *, const uint16_t *, uint16_t *, uint32_t, uint32_t, uint32_t);
template __global__ void LmKvStoreKernel<Qwen36FullKv, QWEN36_THREADS>(LmKvView, const uint16_t *, const uint32_t *, const uint32_t *, uint32_t, uint32_t);
template __global__ void LmHeadCandidateKernel<QWEN36_THREADS, 1024u>(const uint16_t *, const uint16_t *, const uint32_t *, float *, uint32_t *, uint32_t, uint32_t, uint32_t);
template __global__ void LmHeadCommitKernel<QWEN36_THREADS>(const float *, const uint32_t *, uint32_t, uint32_t *, float *, uint32_t);
template __global__ void LmMoeFinalizeKernel<QWEN36_THREADS>(const uint16_t *, const uint32_t *, const float *, uint16_t *, uint32_t, uint32_t, uint32_t);

template __global__ void LmAttentionDecodeKernel<Qwen36FullKv, QWEN36_THREADS, QWEN36_NOPE_DIM, QWEN36_ROPE_DIM>(const uint16_t *, const uint16_t *, LmKvView, const uint32_t *, const uint32_t *, const uint32_t *, uint32_t, uint32_t, float, uint16_t *, const uint32_t *);

extern "C" int32_t Qwen36GemmFp8(LmGemmArguments *a, const void *x, const void *w,
	uint32_t rows, uint32_t tokens, uint32_t groups, uint32_t k, uint32_t n,
	uint32_t sms, bool grouped, cudaStream_t s)
{
	return(LmGemmLaunch<LmFp8,QWEN36_TILE_N,128u,QWEN36_STAGES,QWEN36_WARPS>(a,x,w,rows,tokens,8u,groups,k,n,sms,grouped,s));
}

// -- entry points ---------------------------------------------------------------
//
// Two layer kinds, chosen by the host from the layer index through
// QWEN36_LAYER_IS_LINEAR. Separate entry points rather than a flag: the state
// pool and the KV pool are different geometries, and that belongs in the type.

extern "C" int32_t Qwen36LayerLinearFp8(const Qwen36LayerBuffers *b, uint32_t rows, uint32_t sms, cudaStream_t s)
{
	return(Qwen36LayerLinear<LmFp8>(b,rows,sms,s));
}

extern "C" int32_t Qwen36LayerAttentionFp8(const Qwen36LayerBuffers *b, uint32_t rows, uint32_t context, uint32_t sms, cudaStream_t s)
{
	return(Qwen36LayerAttention<LmFp8,Qwen36FullKv>(b,rows,context,sms,s));
}

extern "C" int32_t Qwen36LayerDenseMlpFp8(const Qwen36LayerBuffers *b, uint32_t rows, uint32_t sms, cudaStream_t s)
{
	return(Qwen36LayerDenseMlp<LmFp8>(b,rows,sms,s));
}

extern "C" int32_t Qwen36HeadFullVocab(const Qwen36LayerBuffers *b, const void *norm_weight, const void *head_weight, uint32_t rows, cudaStream_t s)
{
	return(Qwen36Head(b,norm_weight,head_weight,0,QWEN36_VOCAB,rows,s));
}

extern "C" int32_t Qwen36HeadRestricted(const Qwen36LayerBuffers *b, const void *norm_weight, const void *head_weight, const uint32_t *token_ids, uint32_t count, uint32_t rows, cudaStream_t s)
{
	return(Qwen36Head(b,norm_weight,head_weight,token_ids,count,rows,s));
}

