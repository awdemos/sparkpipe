#pragma once
// Kimi K3. Shapes and constants only.
//
// WEIGHTS ARE OUT AS OF 2026-07-27, AND THIS FILE HAS NOT SEEN THEM.
// moonshotai/Kimi-K3 is gated (401); the public repository is
// moonshotai/Kimi-K3-MXFP4. Nothing below was read from its config.json - the
// corrections in this pass come from the vLLM K3 preview and Moonshot's launch
// material, which are implementers' accounts rather than the config itself.
//
// Corrected: K3_LAYERS was a guess of 72 from K2 lineage. vLLM's preview says
// 93. Still second-hand, but an implementer counting layers beats an inference
// from a previous model.
//
// Confirmed: 3 of every 4 attention layers are KDA, and the expert pool is
// top-16 of 896. Both were already right.
//
// FOUR THINGS THE RELEASE ADDS THAT ARE NOT MODELLED HERE:
//
//   AttnRes. Layers retrieve representations from earlier layer BLOCKS rather
//   than reading one uniformly accumulated residual stream. This is the same
//   class of change as DeepSeek V4's hyper-connections - a shape change, not a
//   kernel - and the two should be designed together. See
//   docs/MODEL_SUPPORT.md item 7.
//
//   The KDA output gate and post-normalisation. vLLM's fused decode kernel
//   performs "the short convolution, KDA state update, output gate, and
//   normalization" as one. kernels/linear_attn.cuh has the first two. Qwen 3.6
//   needs an attention output gate too, so this is one kernel serving both.
//
//   MXFP4 weights with MXFP8 activations, from quantisation-aware training.
//   GLM52_MXFP4_GROUP is currently exempted from the coverage gate as "a
//   supported format with no checkpoint using it". There is now a checkpoint
//   using it.
//
//   Stable LatentMoE, which is not plain top-k routing over 896 experts, and
//   native vision, which is out of scope for a text decode path.
//
// MOST OF THESE ARE GUESSES AND SAY SO. The old tree marked them and the marking
// is carried forward deliberately: a constant inferred from a lineage is not the
// same kind of fact as one read from a published config, and losing that
// distinction is how an inferred number becomes a load-bearing assumption.
//
// The architecture is the interesting part. Three of every four layers use Kimi
// Delta Attention - a recurrent state, no growing cache - and the fourth uses
// gated MLA over a KV cache. So this model needs BOTH pools and a per-layer
// dispatch, which is the first real test of kernels/kv.cuh's claim that a
// recurrent state is just a pool that does not grow.
#include <stdint.h>
#include "inference/kernels/layer_kind.cuh"

#define K3_HIDDEN 7168u                        /* GUESS (K2 lineage) */
#define K3_LAYERS 93u                          /* vLLM K3 preview: "93-layer network" */
#define K3_FIRST_ROUTED_LAYER 1u               /* GUESS */
#define K3_VOCAB 163840u                       /* GUESS (K2 tokenizer family) */
#define K3_RMS_EPSILON 1e-05f                  /* GUESS (K2 value) */

#define K3_EXPERTS 896u                        /* DISCLOSED */
#define K3_TOP_K 16u                           /* DISCLOSED */
#define K3_SHARED_EXPERTS 1u                   /* GUESS */
#define K3_EXPERT_INTERMEDIATE 2048u           /* GUESS (K2 value) */
#define K3_DENSE_INTERMEDIATE 18432u           /* GUESS (K2 value) */
#define K3_ROUTED_SCALE 2.5f                   /* GUESS (K2 value) */

// 3:1 linear to full. The period and phase decide which pool a layer uses.
#define K3_ATTENTION_PERIOD 4u                 /* GUESS (Kimi Linear 3:1) */
#define K3_GLOBAL_PHASE 3u                     /* GUESS (MLA last in the period) */
#define K3_LAYER_IS_LINEAR(layer) (((layer) % K3_ATTENTION_PERIOD) != K3_GLOBAL_PHASE)

#define K3_KDA_HEADS 64u                       /* GUESS */
#define K3_KDA_KEY_DIM 128u                    /* GUESS (Kimi Linear head dim) */
#define K3_KV_BITS 16u
#define K3_KV_PAGE_SLOTS 64u

// A recurrent state is a fixed allocation per sequence: heads * key_dim *
// value_dim would be the full outer product, but the delta rule keeps a
// key-by-value matrix per head. That is the whole resident cost, and it does not
// grow with context - which is the property the scheduler needs to know about
// and the only one it needs.
#define K3_KDA_STATE_BYTES (K3_KDA_HEADS * K3_KDA_KEY_DIM * 2u)

// 896 experts at top-16 is 16 rows per expert at B1024, half GLM 5.2's - more
// experts touched, fewer rows each, so the tile selector lands lower.
static inline uint32_t K3RowsPerExpert(uint32_t tokens)
{
	return(((tokens * K3_TOP_K) + K3_EXPERTS - 1u) / K3_EXPERTS);
}

// -- layer kinds --
// 3 x Kimi Delta Attention -> 1 x gated MLA. Provisional with the rest of this
// file until the release lands.
#define K3_LAYER_KIND(layer) \
	(K3_LAYER_IS_LINEAR(layer) ? LM_LAYER_RECURRENT : LM_LAYER_LATENT)
