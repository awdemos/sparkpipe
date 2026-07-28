#pragma once
// DeepSeek V4 Flash. Shapes and constants only.
//
// AUDITED against deepseek-ai/DeepSeek-V4-Flash, 2026-07-27. The shapes below
// match. What does NOT match is the amount of the architecture that is
// implemented, and the list is here rather than in a tracker because the next
// person to read this file is the one who needs it.
//
// THERE ARE TWO TARGETS UNDER THIS NAME, and they differ where it is least
// visible. Flash is 43 layers at hidden 4096 and its first two layers are pure
// sliding window. Pro is 61 layers at hidden 7168 and its first two layers are
// HIGH-COMPRESSION, not sliding window. Everything after is CSA and HCA
// interleaved in both. A layer-kind table written for one is wrong for the
// other in exactly the two positions nobody checks. This config is Flash.
//
// FIVE GAPS BETWEEN THIS CONFIG AND A MODEL THAT DECODES CORRECTLY:
//
// 1. THE LAYER-KIND TABLE IS NOT HERE AT ALL. The reference carries
//    compress_ratios, 44 entries reading [0,0,4,128,4,128,...,4,0]: ratio 0 is
//    sliding-window attention, 4 is compressed sparse attention with the
//    indexer, 128 is high-compression. unity.cu exports ONE
//    Dsv4LayerAttentionFp8. Three kinds, one entry point, and nothing selects.
//
// 2. DUAL ROPE IS DECLARED AND UNREACHABLE. DSV4_ROPE_THETA and
//    DSV4_COMPRESS_ROPE_THETA are both below and correct - 10000 for the
//    ratio-0 layers, 160000 with YaRN for the rest - but one entry point
//    cannot choose between them, so the second is dead.
//
// 3. ROPE PAIRING IS WRONG FOR THIS CHECKPOINT. LmRopePerHeadKernel pairs
//    index with index + rope_dim/2, the half-split convention. The reference
//    encodes INTERLEAVED pairs, view_as_complex style, pairing 2i with 2i+1.
//    These are different rotations. Half-split is right for GLM 5.2, which is
//    presumably why it is what the kernel does; DSV4 needs either a pairing
//    mode on the kernel or a permutation at pack time.
//
// 4. HASH ROUTING ON THE FIRST LAYERS IS MISSING. num_hash_layers defaults to
//    3: those blocks route through a hash gate with a token-id to expert-id
//    table, not through the router. The MoE path here has no such branch.
//
// 5. HYPER-CONNECTIONS ARE MISSING ENTIRELY. Every block carries hc_mult=4
//    streams of the hidden state across the stage boundary, mixed by a
//    Sinkhorn-normalised doubly-stochastic matrix. Nothing in the buffers
//    carries four streams, so this is not a kernel gap but a shape gap - the
//    residual is a different object than the one the layer assumes.
//
// The reference bringup was validated at 0.998 final-logits cosine against the
// official implementation. That is the bar, and none of it is reachable until
// at least 1, 2 and 3 exist.
//
// Closest to GLM 5.2 of the five: one KV head at 512 dims is a latent in all but
// name, and the sparse index is the same mechanism at a different top-k. What
// differs is a sliding window on top of the sparse selection, and two rope
// thetas - one for the compressed path, one for the rest.
#include <stdint.h>
#include "inference/kernels/layer_kind.cuh"

#define DSV4_HIDDEN 4096u                      /* CONFIG hidden_size */
#define DSV4_LAYERS 43u                        /* CONFIG num_hidden_layers */
#define DSV4_VOCAB 129280u                     /* CONFIG vocab_size */
#define DSV4_RMS_EPSILON 1e-6f                 /* CONFIG rms_norm_eps */

#define DSV4_ATTN_HEADS 64u                    /* CONFIG num_attention_heads */
#define DSV4_KV_HEADS 1u                       /* CONFIG num_key_value_heads */
#define DSV4_HEAD_DIM 512u                     /* CONFIG head_dim */
#define DSV4_ROPE_DIM 64u                      /* CONFIG qk_rope_head_dim */
#define DSV4_ROPE_THETA 10000.0f               /* CONFIG rope_theta */
#define DSV4_COMPRESS_ROPE_THETA 160000.0f     /* CONFIG compress_rope_theta */
#define DSV4_SLIDING_WINDOW 128u               /* CONFIG sliding_window */

// Sparse selection, same mechanism as GLM 5.2's at a quarter the top-k. That it
// is the same mechanism is the point: kernels/attn.cuh's LmSparseScoreKernel
// serves both, and the difference is two arguments.
#define DSV4_INDEX_HEADS 64u                   /* CONFIG index_n_heads */
#define DSV4_INDEX_DIM 128u                    /* CONFIG index_head_dim */
#define DSV4_INDEX_TOP_K 512u                  /* CONFIG index_topk */

// -- MoE -----------------------------------------------------------------------
//
// SIX experts per token, not eight. I wrote 8 here first by carrying GLM 5.2's
// value across, which is the exact mistake this file exists to prevent - a
// constant that looks like another model's and is not. Rows per expert is
// tokens*6/256, so the tile selector lands lower than GLM 5.2's at every batch.
#define DSV4_EXPERTS 256u                      /* CONFIG n_routed_experts */
#define DSV4_TOP_K 6u                          /* CONFIG num_experts_per_tok */
#define DSV4_EXPERT_INTERMEDIATE 2048u         /* CONFIG moe_intermediate_size */
#define DSV4_ROUTED_SCALE 1.5f                 /* CONFIG routed_scaling_factor */

// A SHARED EXPERT, which GLM 5.2 does not have. Every token passes through it in
// addition to its six routed experts, and its output is added rather than
// weighted - it is not part of the top-k and has no gate. Omitting it drops a
// dense contribution from every token, which degrades quality uniformly instead
// of visibly.
#define DSV4_SHARED_EXPERTS 1u                 /* CONFIG n_shared_experts */
#define DSV4_SHARED_INTERMEDIATE (DSV4_EXPERT_INTERMEDIATE * DSV4_SHARED_EXPERTS)

// The low-rank query path: hidden -> 1024 -> norm -> heads. Half GLM 5.2's rank
// on a model with two thirds its hidden size.
#define DSV4_QUERY_LORA_RANK 1024u             /* CONFIG q_lora_rank */

// YaRN rope scaling, which neither GLM 5.2 nor MiMo 2.5 uses. Positions beyond
// the original training length are interpolated rather than extrapolated, with
// the interpolation applied per frequency band. A model trained to 65,536 and
// scaled by 16 serves a million tokens of context, and applying plain rope to it
// instead degrades smoothly with distance rather than failing - which is why it
// is worth a constant and a comment rather than being assumed away.
#define DSV4_YARN_FACTOR 16u                   /* CONFIG rope_scaling.factor */
#define DSV4_YARN_ORIGINAL_POSITIONS 65536u    /* CONFIG rope_scaling.original */

// The KV cache is quantised at block 64 on the nope dimensions. Independent of
// the weight format, as kernels/kv.cuh allows.
#define DSV4_KV_QUANT_BLOCK 64u
#define DSV4_KV_BITS 16u
#define DSV4_KV_PAGE_SLOTS 64u

// -- layer kinds --
// Flash: the first two layers are pure sliding window, then CSA and HCA
// interleaved. Layer 2 is CSA (compression 4) and layer 3 is HCA (128), which
// matches the published compress_ratios [0,0,4,128,4,128,...].
//
// UNVERIFIED: the published table ends '...,4,0' and has 44 entries for 43
// layers. Either the last layer is sliding window or the table carries a
// trailing entry for the MTP head. This macro says COMPRESSED for layer 42 and
// that is a guess. PRO DIFFERS HERE TOO: its first two layers are HCA, not
// window, so this macro is Flash's and a Pro config needs its own.
#define DSV4_LAYER_KIND(layer) \
	((layer) < 2u ? LM_LAYER_WINDOW \
		: (((layer) % 2u) == 0u ? LM_LAYER_SPARSE : LM_LAYER_COMPRESSED))
