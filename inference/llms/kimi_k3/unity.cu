// Kimi K3. The whole model, one translation unit.
//
// TWO POOLS. Three of every four layers carry a recurrent Kimi Delta Attention
// state; the fourth attends over a KV cache. kernels/kv.cuh claimed a recurrent
// state is a pool that does not grow, and this is the first model that tests it:
// LmKvState and LmKvLatent are the same allocator, the same page table and the
// same slot accessor, differing in kGrows and a page count of one.
//
// The scheduler asks both the same questions in the same units, which is the
// property that matters - a model whose layers page differently would otherwise
// need two schedulers.
#include "runtime/gemm.cuh"
#include "inference/kernels/norm.cuh"
#include "inference/kernels/attn.cuh"
#include "inference/kernels/topk.cuh"
#include "inference/kernels/linear_attn.cuh"
#include "inference/kernels/head.cuh"
#include "inference/kernels/formats/fp8.cuh"
#include "inference/kernels/formats/int7.cuh"
#include "inference/llms/kimi_k3/config.h"

// THE MLA LATENT IS kv_lora_rank, NOT THE KDA HEAD DIM. This was
// LmKvLatent<..., K3_KDA_KEY_DIM, 64u, ...> - 128 elements, the width of a KDA
// head, standing in for a 512-element MLA latent. Two unrelated dimensions that
// both happen to be head-shaped, so nothing looked wrong and the pool came out
// four times too small.
//
// Same defect qwen_3_6 had one commit earlier, found the same way: the constant
// the geometry needed was not in config.h, so a nearby one was used.
using K3GlobalKv = LmKvLatent<K3_KV_BITS, K3_KV_LORA_RANK, K3_QK_UNROTATED_DIM, K3_KV_PAGE_SLOTS>;
using K3LinearState = LmKvState<K3_KDA_STATE_BYTES>;

static_assert(K3LinearState::kGrows == false, "a delta-rule state does not grow with context");
static_assert(K3LinearState::kPageSlots == 1u, "one slot per sequence");
static_assert(K3GlobalKv::kGrows == true, "the global layers still page");

#define K3_TILE_N 128u
#define K3_STAGES 2u
#define K3_WARPS 8u
#define K3_THREADS 256u

// 896 experts at top-16 gives 16 rows per expert at B1024, half GLM 5.2's, so
// the tile selector lands lower and the 64-row instantiation is never chosen.
template __global__ void LmGemmKernel<LmFp8, 16u, K3_TILE_N, 128u, K3_STAGES, K3_WARPS>(__grid_constant__ const LmGemmArguments, LmTileSource, LmTileSource, bool);
template __global__ void LmGemmKernel<LmFp8, 32u, K3_TILE_N, 128u, K3_STAGES, K3_WARPS>(__grid_constant__ const LmGemmArguments, LmTileSource, LmTileSource, bool);
template __global__ void LmGemmKernel<LmInt7, 16u, K3_TILE_N, 256u, K3_STAGES, K3_WARPS>(__grid_constant__ const LmGemmArguments, LmTileSource, LmTileSource, bool);
template __global__ void LmGemmKernel<LmInt7, 32u, K3_TILE_N, 256u, K3_STAGES, K3_WARPS>(__grid_constant__ const LmGemmArguments, LmTileSource, LmTileSource, bool);
template __global__ void LmFusedResidualRmsNormKernel<K3_THREADS>(const uint16_t *, const uint16_t *, const uint16_t *, uint16_t *, uint16_t *, uint32_t, float);
template __global__ void LmSiluMulKernel<K3_THREADS>(const uint16_t *, uint16_t *, uint32_t, bool);
template __global__ void LmQuantiseRowsKernel<LmInt7, K3_THREADS>(const uint16_t *, const uint32_t *, uint8_t *, uint8_t *, uint32_t, uint32_t);
// No rope instantiation. K3 is NoPE - the reference sets rotary_emb to None and
// carries the qk_rope slice through unrotated. This line built a rope kernel for
// a model that never rotates, which cost nothing at runtime and would have told
// whoever wrote layer.cuh that calling it was expected.
// KDA on three of every four layers, gated MLA on the fourth. The same delta
// rule Qwen 3.6 uses, at K3's widths - which is the argument that this is an
// architecture class and not one vendor's design.
template __global__ void LmDeltaRuleDecodeKernel<K3_THREADS, K3_KDA_KEY_DIM, K3_KDA_VALUE_DIM>(uint8_t *, const uint32_t *, const uint16_t *, const uint16_t *, const uint16_t *, const float *, const float *, uint16_t *, uint32_t, uint32_t, uint32_t);
// The rest of the KDA path, none of which unity built: the short convolution
// with its Swish, the L2 normalisation of q and k, the bounded decay mapping,
// the output gate, and SiTU on every MLP.
template __global__ void LmCausalConvDecodeKernel<K3_THREADS, K3_KDA_CONV_KERNEL, LM_CONV_SWISH>(uint16_t *, const uint32_t *, const uint16_t *, const uint16_t *, uint16_t *, uint32_t, uint32_t);
template __global__ void LmL2NormalisePerHeadKernel<K3_THREADS, K3_KDA_KEY_DIM>(uint16_t *, uint32_t, uint32_t, float);
template __global__ void LmBoundedDecayKernel<K3_THREADS, K3_KDA_KEY_DIM>(const uint16_t *, const float *, const float *, float *, uint32_t, float, uint32_t);
template __global__ void LmOutputGateKernel<K3_THREADS>(uint16_t *, const uint16_t *, uint32_t);
template __global__ void LmSituMulKernel<K3_THREADS>(const uint16_t *, uint16_t *, uint32_t, float, float);
template __global__ void LmKvStoreKernel<K3GlobalKv, K3_THREADS>(LmKvView, const uint16_t *, const uint32_t *, const uint32_t *, uint32_t, uint32_t);
template __global__ void LmHeadCandidateKernel<K3_THREADS, 1024u>(const uint16_t *, const uint16_t *, const uint32_t *, float *, uint32_t *, uint32_t, uint32_t, uint32_t);
template __global__ void LmHeadCommitKernel<K3_THREADS>(const float *, const uint32_t *, uint32_t, uint32_t *, float *, uint32_t);
template __global__ void LmMoeFinalizeKernel<K3_THREADS>(const uint16_t *, const uint32_t *, const float *, uint16_t *, uint32_t, uint32_t, uint32_t);

template __global__ void LmAttentionDecodeKernel<K3GlobalKv, K3_THREADS, K3_KV_LORA_RANK, K3_QK_UNROTATED_DIM>(const uint16_t *, const uint16_t *, LmKvView, const uint32_t *, const uint32_t *, const uint32_t *, uint32_t, uint32_t, float, uint16_t *, const uint32_t *);
template __global__ void LmTopkSmallKernel<K3_THREADS, K3_TOP_K>(const float *, uint32_t, uint32_t *, float *, float);

extern "C" int32_t K3GemmInt7(LmGemmArguments *a, const void *x, const void *w,
	uint32_t rows, uint32_t tokens, uint32_t groups, uint32_t k, uint32_t n,
	uint32_t sms, bool grouped, cudaStream_t s)
{
	return(LmGemmLaunch<LmInt7,K3_TILE_N,256u,K3_STAGES,K3_WARPS>(a,x,w,rows,tokens,K3_TOP_K,groups,k,n,sms,grouped,s));
}
