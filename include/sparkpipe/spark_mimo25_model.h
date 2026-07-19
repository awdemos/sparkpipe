#ifndef SPARKPIPE_SPARK_MIMO25_MODEL_H
#define SPARKPIPE_SPARK_MIMO25_MODEL_H

#include <stdint.h>

/*
 * MiMo V2.5 (310B, 15B active) geometry, pinned against the checkpoint
 * config (huggingface.co/XiaomiMiMo/MiMo-V2.5 config.json, fetched
 * 2026-07-19, model_type mimo_v2). Every constant names its source field.
 * The Pro variant (1.02T) follows the dsv4 mechanism: a second header with
 * the same macro names under the same include guard.
 *
 * Architecture: 48 layers over hidden 4096. The hybrid attention map is an
 * explicit config array, embedded below: nine GLOBAL layers at 0, 5, then
 * every sixth, thirty-nine sliding-window layers (window 128, chunked at
 * 128, learned sink bias on the SWA branch only). Both branches run 64
 * query heads x 192, value scale 0.707, fused qkv projection layout
 * (declared by the config), partial RoPE on the first 64 dims (factor
 * 0.334, theta 1e7) - but the branches differ in key/value heads: GLOBAL
 * has 4, SWA has 8, so the per-token cache cost differs by branch and the
 * pack format carries per-branch projection shapes. MoE on every layer
 * except the dense layer 0 (dense intermediate 16384): 256 routed experts,
 * top-8, sigmoid scoring with the noaux_tc balancer, expert intermediate
 * 2048, no shared expert, FP8 e4m3 native storage. Vocabulary 152576,
 * untied. Epsilon 1e-5. Natively multimodal: the vision and audio towers
 * are out of scope and rejected at pack build, the qwen36 precedent.
 *
 * REFERENCE-PIN (against modeling_mimo_v2.py, which the repo ships, and
 * the safetensors index): the MTP layers (the launch report says three,
 * the config carries no num_nextn field - the mtp.* tensor set is the
 * ground truth), the sink-bias application form, the value-scale
 * placement, and the fused qkv split order.
 */

#define SPARK_MIMO25_MODEL_HIDDEN_DIMENSION 4096u               /* CONFIG hidden_size */
#define SPARK_MIMO25_MODEL_LAYER_COUNT 48u                      /* CONFIG num_hidden_layers */
#define SPARK_MIMO25_MODEL_VOCAB_COUNT 152576u                  /* CONFIG vocab_size */
#define SPARK_MIMO25_MODEL_MAX_POSITIONS 1048576u               /* CONFIG max_position_embeddings */
#define SPARK_MIMO25_MODEL_RMS_NORM_EPSILON 1e-5f               /* CONFIG layernorm_epsilon */
#define SPARK_MIMO25_MODEL_ATTN_QUERY_HEAD_COUNT 64u            /* CONFIG num_attention_heads */
#define SPARK_MIMO25_MODEL_ATTN_HEAD_DIMENSION 192u             /* CONFIG head_dim */
#define SPARK_MIMO25_MODEL_GLOBAL_KV_HEAD_COUNT 4u              /* CONFIG num_key_value_heads */
#define SPARK_MIMO25_MODEL_SWA_KV_HEAD_COUNT 8u                 /* CONFIG swa_num_key_value_heads */
#define SPARK_MIMO25_MODEL_ATTN_ROPE_DIMENSION 64u              /* CONFIG partial_rotary_factor 0.334 x 192 */
#define SPARK_MIMO25_MODEL_ATTN_ROPE_THETA 10000000.0f          /* CONFIG rope_theta */
#define SPARK_MIMO25_MODEL_ATTN_VALUE_SCALE 0.707f              /* CONFIG attention_value_scale */
#define SPARK_MIMO25_MODEL_SLIDING_WINDOW_TOKENS 128u           /* CONFIG sliding_window */
#define SPARK_MIMO25_MODEL_ATTENTION_CHUNK_TOKENS 128u          /* CONFIG attention_chunk_size */
#define SPARK_MIMO25_MODEL_DENSE_INTERMEDIATE_DIMENSION 16384u  /* CONFIG intermediate_size */
#define SPARK_MIMO25_MODEL_EXPERT_INTERMEDIATE_DIMENSION 2048u  /* CONFIG moe_intermediate_size */
#define SPARK_MIMO25_MODEL_ROUTED_EXPERT_COUNT 256u             /* CONFIG n_routed_experts */
#define SPARK_MIMO25_MODEL_EXPERTS_PER_TOKEN 8u                 /* CONFIG num_experts_per_tok */
#define SPARK_MIMO25_MODEL_DENSE_LAYER_COUNT 1u                 /* CONFIG moe_layer_freq: zero only at layer 0 */
#define SPARK_MIMO25_MODEL_BF16_ELEMENT_BYTES 2u

#define SPARK_MIMO25_MODEL_ATTN_QUERY_DIMENSION (SPARK_MIMO25_MODEL_ATTN_QUERY_HEAD_COUNT * SPARK_MIMO25_MODEL_ATTN_HEAD_DIMENSION)
#define SPARK_MIMO25_MODEL_GLOBAL_KV_DIMENSION (SPARK_MIMO25_MODEL_GLOBAL_KV_HEAD_COUNT * SPARK_MIMO25_MODEL_ATTN_HEAD_DIMENSION)
#define SPARK_MIMO25_MODEL_SWA_KV_DIMENSION (SPARK_MIMO25_MODEL_SWA_KV_HEAD_COUNT * SPARK_MIMO25_MODEL_ATTN_HEAD_DIMENSION)
#define SPARK_MIMO25_MODEL_LAYER_IS_MOE(layer_index) ((layer_index) != 0u)

#define SPARK_MIMO25_MODEL_LAYER_KIND_GLOBAL 0u
#define SPARK_MIMO25_MODEL_LAYER_KIND_SWA 1u

// Transcribed from CONFIG hybrid_layer_pattern: nine global layers at 0, 5,
// then every sixth through 47; thirty-nine sliding-window layers.
static const uint8_t SPARK_MIMO25_MODEL_LAYER_KIND[SPARK_MIMO25_MODEL_LAYER_COUNT] =
{
	0,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0,
	1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0
};

static inline uint32_t SparkMimo25ModelLayerKind(uint32_t layer_index)
{
	return(layer_index < SPARK_MIMO25_MODEL_LAYER_COUNT ? (uint32_t)SPARK_MIMO25_MODEL_LAYER_KIND[layer_index] : SPARK_MIMO25_MODEL_LAYER_KIND_GLOBAL);
}

#endif
