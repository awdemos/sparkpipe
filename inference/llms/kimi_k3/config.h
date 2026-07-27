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

// -- what the modelling file says KDA and the MoE actually are ------------------
//
// LATENT MoE. The router runs on the FULL hidden, not the latent, which is the
// part a shape-only reading gets backwards:
//
//     topk, weights = gate(hidden)                 router at 7168
//     latent        = down_proj(hidden)            7168 -> 3584
//     y             = experts(latent, topk)        experts at 3584, inter 3072
//     y             = rms_norm(y)                  latent_moe_use_norm is true
//     y             = up_proj(y)                   3584 -> 7168
//     out           = y + shared_experts(hidden)   shared run on the ORIGINAL
//
// The shared experts are NOT in the latent space and take the pre-projection
// hidden at intermediate moe_intermediate * num_shared = 3072 * 2 = 6144.
//
// KDA DOES NOT FIT LmDeltaRuleDecodeKernel AS IT STANDS. The projections:
//
//     q_proj, k_proj, v_proj   hidden -> heads * head_dim = 12288, each with its
//                              OWN short convolution, each with a SiLU
//     g (forget)               f_b_proj(f_a_proj(hidden)): 7168 -> 128 -> 12288,
//                              so PER HEAD PER CHANNEL
//     beta (write)             b_proj(hidden): 7168 -> 96, per head scalar,
//                              sigmoid applied inside the kernel
//     output gate              g_proj(hidden) -> 12288, full rank, per channel
//     o_norm                   gated RMS norm with a sigmoid, at head_dim
//     A_log                    per head, log-uniform over [1,16]
//     dt_bias                  per channel
//     lower bound              -5.0 clamp on the gate
//
// linear_attn.cuh reads forget_gate[(row * key_heads) + head] - ONE SCALAR PER
// HEAD. KDA's forget gate is head_dim wide per head. The kernel cannot express
// this model's decay, and widening that argument is the first piece of K3 work
// that changes a shared kernel rather than adding one.
//
// Three separate convolutions with SiLU, not one; LmCausalConvDecodeKernel has
// no activation. The final decay arithmetic combining g, A_log and dt_bias lives
// in fla's fused_recurrent_kda and is NOT in the released modelling file, so it
// is the one piece still unread.

// KDA, on the other three in four. 96 heads at 128, a 4-wide causal convolution,
// a full-rank gate floored at -5.
#define K3_KDA_HEADS 96u
#define K3_KDA_KEY_DIM 128u
#define K3_KDA_VALUE_DIM 128u
#define K3_KDA_CONV_KERNEL 4u
#define K3_KDA_GATE_LOWER_BOUND -5.0f
#define K3_KDA_FULL_RANK_GATE 1u

// AttnRes, now read from the modelling file. It is an ATTENTION over residuals,
// not a weighted sum with learned scalars:
//
//     v       = [saved block residuals ..., current prefix sum]
//     k       = v * rsqrt(mean(v^2) + eps)          RMS-normalise each candidate
//     scores  = sum(k * (norm.weight * proj.weight))
//     out     = softmax(scores) @ v
//
// Applied TWICE per layer - once before attention with self_attention_res_proj,
// once before the MLP with mlp_res_proj. A new block residual is appended every
// attn_res_block_size layers, so 93 layers at block size 12 accumulate 8 blocks
// and the candidate set reaches 9.
//
// THIS IS THE SHAPE CHANGE, AND THE NUMBER IS 9x. Every buffer in this tree
// carries hidden_bf16 as one tensor. Under AttnRes a token carries up to nine,
// and the ring moves hidden state between ranks, so the stage payload per row
// goes from 14 KiB to 126 KiB at hidden 7168. That is a transport and pool
// question before it is a kernel question. deepseek_v4's hyper-connections are
// the same class at n_hc=4 - see docs/MODEL_SUPPORT.md item 7.
#define K3_ATTNRES_BLOCK_SIZE 12u

// SiTU, from the released modeling_kimi_linear.py. IMPLEMENTED as
// LmSituMulKernel, checked numerically by tests/test_situ_activation.py:
//
//     situ_a = beta * tanh(gate / beta) * sigmoid(gate)
//     up     = linear_beta * tanh(up / linear_beta)
//     out    = situ_a * up
//
// It is SiLU-mul with both arms soft-clamped to their own beta - gate to 4,
// linear to 25 - which is what "activation control" meant. Gate is the FIRST
// half of the fused projection.
//
// The two betas are not interchangeable: swapping them clamps the gate at 25
// and the linear arm at 4, which runs and is wrong, so the gate checks the
// saturation points rather than only the values near zero.
//
// (Historical note, kept because the reasoning still applies to AttnRes below:
// before the modelling file arrived this was the one gap where a plausible
// guess was worse than an empty space. THE FORMULA WAS
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
// NOT PUBLISHED and the name admitted several readings. It came from the
// modelling file, not from reasoning about the name.)
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
