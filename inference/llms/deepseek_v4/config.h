#pragma once
// DeepSeek V4. Shapes and constants only.
//
// Closest to GLM 5.2 of the five: one KV head at 512 dims is a latent in all but
// name, and the sparse index is the same mechanism at a different top-k. What
// differs is a sliding window on top of the sparse selection, and two rope
// thetas - one for the compressed path, one for the rest.
#include <stdint.h>

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
