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

// Read from moonshotai/Kimi-K3 config.json, 2026-07-27. Everything below is
// DISCLOSED unless marked otherwise, and the guesses this replaced are recorded
// where they were wrong, because which ones failed is the useful information.
//
//   right:  hidden 7168, vocab 163840, rms eps 1e-5, experts 896, top-k 16,
//           first dense layer count 1, KDA head dim 128, the 3:1 period
//   WRONG:  shared experts 1 -> 2
//           expert intermediate 2048 -> 3072
//           dense intermediate 18432 -> 33792
//           routed scale 2.5 -> 1.0
//           KDA heads 64 -> 96
//
// Five of thirteen. Every wrong one was a K2-lineage inference, and every one
// of those would have produced a model that built, ran, and was wrong.

#define K3_HIDDEN 7168u
#define K3_LAYERS 93u
#define K3_VOCAB 163840u
#define K3_RMS_EPSILON 1e-05f
#define K3_MAX_CONTEXT 1048576u

// MoE. Stable LatentMoE: the routed experts run at their own hidden width and
// are projected, which is what routed_expert_hidden_size is and why it is not
// K3_HIDDEN. Router is sigmoid with noaux_tc grouping, renormalised.
#define K3_FIRST_ROUTED_LAYER 1u
#define K3_EXPERTS 896u
#define K3_TOP_K 16u
#define K3_SHARED_EXPERTS 2u
#define K3_EXPERT_INTERMEDIATE 3072u
#define K3_ROUTED_EXPERT_HIDDEN 3584u
#define K3_DENSE_INTERMEDIATE 33792u
#define K3_ROUTED_SCALE 1.0f

// MLA, on the full-attention layers. mla_use_nope and mla_use_output_gate are
// both true: the nope half carries no rotation, and the attention output is
// gated - the same gate qwen_3_6 needs, so one kernel serves both.
#define K3_MLA_HEADS 96u
#define K3_KV_LORA_RANK 512u
#define K3_Q_LORA_RANK 1536u
#define K3_QK_NOPE_DIM 128u
#define K3_QK_ROPE_DIM 64u
#define K3_V_HEAD_DIM 128u
#define K3_MLA_USE_NOPE 1u
#define K3_MLA_OUTPUT_GATE 1u

// KDA, on the other three in four. 96 heads at 128, a 4-wide causal convolution,
// a full-rank gate floored at -5.
#define K3_KDA_HEADS 96u
#define K3_KDA_KEY_DIM 128u
#define K3_KDA_VALUE_DIM 128u
#define K3_KDA_CONV_KERNEL 4u
#define K3_KDA_GATE_LOWER_BOUND -5.0f
#define K3_KDA_FULL_RANK_GATE 1u

// AttnRes: layers retrieve from earlier layer BLOCKS, twelve layers to a block.
// Not modelled - see docs/MODEL_SUPPORT.md item 7, which deepseek_v4's mHC
// shares.
#define K3_ATTNRES_BLOCK_SIZE 12u

// SiTU - "Sigmoid Tanh Unit" - replaces SwiGLU on every layer. THE FORMULA IS
// NOT PUBLISHED anywhere I can reach. Moonshot's tech blog names it and says it
// improves "activation control"; the GGUF conversion effort records it as a new
// activation and does not implement it either. The name and the two betas are
// all that is public.
//
// I am not writing this kernel from the name. "Sigmoid Tanh Unit" with betas at
// 4.0 and 25.0 admits several readings - sigmoid(beta*x)*tanh(x), x*sigmoid of a
// tanh, a two-branch gate with a beta each - and they are different functions
// that all produce fluent text. This is the one gap on the list where a
// plausible guess is worse than an empty space, because nothing downstream
// would contradict it.
//
// UNBLOCKED BY: modeling_kimi_linear.py from the released repository, which
// contains the implementation. The config.json alone cannot settle it.
#define K3_SITU_BETA 4.0f
#define K3_SITU_LINEAR_BETA 25.0f

// MXFP4 at group 32, and the ignore list matters: attention, shared experts,
// the dense MLP, lm_head and the vision tower are NOT quantised. Only the
// routed experts are 4-bit.
#define K3_MXFP4_GROUP 32u

// num_nextn_predict_layers is 0. This model has no MTP head, unlike glm5_2.
#define K3_MTP_LAYERS 0u

#define K3_KV_BITS 16u
#define K3_KV_PAGE_SLOTS 64u

// The delta rule carries a key-by-value outer product per head, so the state is
// heads * key_dim * value_dim, not heads * key_dim. The old expression here was
// the latter and understated the pool by 128x: 3 MiB per sequence in bf16, and
// it does not grow with context, which is the whole point of the mechanism.
#define K3_KDA_STATE_BYTES \
	(K3_KDA_HEADS * K3_KDA_KEY_DIM * K3_KDA_VALUE_DIM * (K3_KV_BITS / 8u))

// 1-INDEXED IN THE CONFIG, 0-INDEXED HERE. full_attn_layers is
// {4,8,...,92} plus 93; subtract one and that is {3,7,...,91} plus 92. So the
// period-4 rule holds and THE LAST LAYER IS AN EXCEPTION - 92 % 4 is 0, which
// the formula alone would call KDA. 24 full and 69 KDA, which is what the two
// lists in the config contain.
#define K3_LAYER_IS_LINEAR(layer) \
	((((layer) % 4u) != 3u) && ((layer) != (K3_LAYERS - 1u)))

#define K3_LAYER_KIND(layer) \
	(K3_LAYER_IS_LINEAR(layer) ? LM_LAYER_RECURRENT : LM_LAYER_LATENT)
