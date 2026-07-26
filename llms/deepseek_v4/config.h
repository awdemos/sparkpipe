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

#define DSV4_EXPERTS 256u
#define DSV4_TOP_K 8u
#define DSV4_KV_BITS 16u
#define DSV4_KV_PAGE_SLOTS 64u
