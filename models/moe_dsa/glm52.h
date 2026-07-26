#pragma once

// GLM 5.2. Shapes and constants only.
//
// A model in this tree is a header. The driver lives beside it in
// models/moe_dsa/driver.cu and is shared with every other model of the same
// architecture; adding GLM 5.3 means adding a header, not a tree. The old
// layout had glm52 spread across five directories and 109 files, none of which
// named itself a driver.
//
// Values marked CONFIG come from the published model config. Nothing here is
// derived from a measurement, and nothing here is a guess - a guessed constant
// belongs in the file of the model it was guessed for, marked, not here.

#include <stdint.h>

#define GLM52_HIDDEN 6144u                     /* CONFIG hidden_size */
#define GLM52_LAYERS 78u                       /* CONFIG num_hidden_layers */
#define GLM52_FIRST_ROUTED_LAYER 3u            /* CONFIG first_k_dense_replace */
#define GLM52_VOCAB 154880u                    /* CONFIG vocab_size */
#define GLM52_RMS_EPSILON 1e-05f               /* CONFIG rms_norm_eps */
#define GLM52_ROPE_THETA 8000000.0f            /* CONFIG rope_theta */

// -- attention: MLA with sparse selection ------------------------------------
//
// The KV cache stores a shared latent row per (sequence, slot) rather than
// per-head key and value rows. That is the whole reason this model is bandwidth
// tractable at decode: the cache read per slot is 1152 bytes instead of 57 KB,
// at the cost of small per-head GEMMs to reconstruct - a large byte reduction
// for negligible added compute, which is the right trade on a bandwidth-bound
// machine.

#define GLM52_ATTN_HEADS 64u                   /* CONFIG num_attention_heads */
#define GLM52_LATENT 512u                      /* CONFIG kv_lora_rank */
#define GLM52_ROPE_DIM 64u                     /* CONFIG qk_rope_head_dim */
#define GLM52_QK_NOPE_DIM 192u                 /* CONFIG qk_nope_head_dim */
#define GLM52_VALUE_DIM 256u                   /* CONFIG v_head_dim */
#define GLM52_QUERY_A_DIM 2048u                /* CONFIG q_lora_rank */

// Sparse selection. The index pass scores all slots with a cheap low-rank head
// and keeps the top GLM52_DSA_SELECTED; the full attention then reads only
// those. Index state is shared across a group of layers, so the score is
// computed once per group rather than once per layer.
#define GLM52_DSA_SELECTED 2048u               /* CONFIG index_topk */
#define GLM52_DSA_INDEX_HEADS 32u              /* CONFIG index_n_heads */
#define GLM52_DSA_INDEX_DIM 128u               /* CONFIG index_head_dim */
#define GLM52_DSA_SHARE_GROUP_LAYERS 4u        /* CONFIG index layer sharing period */
#define GLM52_DSA_INDEX_EPSILON 1e-06f

// -- MoE ---------------------------------------------------------------------
//
// 256 experts at top-8 means rows per expert at decode is batch/32. That number
// drives the GEMM tile height and is the reason TILE_M is selected per token
// bucket rather than fixed: at B1024 it is 32 rows, and a 16-row tile would
// split every expert and double the weight stream.

#define GLM52_EXPERTS 256u                     /* CONFIG n_routed_experts */
#define GLM52_TOP_K 8u                         /* CONFIG num_experts_per_tok */
#define GLM52_EXPERT_INTERMEDIATE 2048u        /* CONFIG moe_intermediate_size */
#define GLM52_DENSE_INTERMEDIATE 12288u        /* CONFIG intermediate_size */
#define GLM52_ROUTED_SCALE 2.5f                /* CONFIG routed_scaling_factor */
#define GLM52_W1_COMPONENTS 2u                 /* gate and up, emitted together */

// -- quantisation ------------------------------------------------------------
//
// Group sizes are format properties, not model choices, but they are stated
// here because the GEMM's K tile must be a whole swizzle span in BYTES and that
// depends on the element width. NVFP4 at 4 bits needs a 256-element K tile
// where FP8 needs 128; kernels/tile.cuh asserts it.

#define GLM52_FP8_SCALE_BLOCK 128u
#define GLM52_NVFP4_GROUP 16u
#define GLM52_MXFP4_GROUP 32u

// -- speculative decode ------------------------------------------------------

#define GLM52_MTP_DRAFT_TOKENS 6u
#define GLM52_MTP_LAYER_INDEX GLM52_LAYERS

// -- derived -----------------------------------------------------------------
//
// Written once here rather than recomputed at call sites. A dimension derived
// twice is a dimension that can disagree with itself.

#define GLM52_ROUTED_LAYERS (GLM52_LAYERS - GLM52_FIRST_ROUTED_LAYER)
#define GLM52_GATE_UP_DIM (GLM52_EXPERT_INTERMEDIATE * GLM52_W1_COMPONENTS)
#define GLM52_LATENT_ROW (GLM52_LATENT + GLM52_ROPE_DIM)
#define GLM52_WEIGHT_LAYERS (GLM52_LAYERS + 1u)

// Rows one expert receives on average at a given batch. The GEMM tile height is
// selected from this, so it is defined where the constants are rather than in
// the launcher.
static inline uint32_t Glm52RowsPerExpert(uint32_t tokens)
{
	return(((tokens * GLM52_TOP_K) + GLM52_EXPERTS - 1u) / GLM52_EXPERTS);
}
