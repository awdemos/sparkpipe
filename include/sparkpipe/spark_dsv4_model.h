#ifndef SPARKPIPE_SPARK_DSV4_MODEL_H
#define SPARKPIPE_SPARK_DSV4_MODEL_H

#include <stdint.h>

/*
 * DeepSeek V4 Flash geometry, pinned against the checkpoint config
 * (huggingface.co/deepseek-ai/DeepSeek-V4-Flash config.json, fetched
 * 2026-07-19, transformers 4.57.1, model_type deepseek_v4). Every constant
 * below carries a CONFIG marker naming its source field. The Pro variant is
 * the same module with a second geometry header, the same trick as the
 * shared attention/MoE machinery it will reuse.
 *
 * Architecture in one paragraph: 43 layers over hidden 4096. The per-layer
 * attention kind comes from the config's compress_ratios array, embedded
 * below as the layer-kind table: the first two layers are sliding-window
 * attention (window 128), then compressed sparse attention (compress ratio
 * 4 with a 64-head x 128 indexer selecting top-512) alternates with
 * high-compression attention (ratio 128) through layer 42. All attention is
 * 64 query heads x head dim 512 over ONE key/value head, with a 1024-rank
 * query compression, rope on the first 64 dims (theta 10000 yarn-scaled x16
 * from 64K to 1M; the compressed stream has its own theta 160000), and the
 * output projection grouped 8 ways. MoE everywhere the reference says so:
 * 256 routed experts (fp4 native) top-6 with sqrtsoftplus scoring and the
 * noaux_tc balancer, one shared expert, expert intermediate 2048, swiglu
 * clamp 10, routed scale 1.5, and the first three layers hash-routed. The
 * residual stream is hyper-connected: hc_mult 4 streams cross every stage
 * boundary (Sinkhorn iterations are the training-side mixer). One MTP
 * layer. Vocabulary 129280, untied. FP8 e4m3 with ue8m0 scales outside the
 * experts.
 *
 * REFERENCE-PIN (settled against the repo's inference reference model.py
 * before any CUDA, one line each in the coming shape table): the MTP
 * layer's attention kind (the map's 44th entry is 0), the dense-vs-MoE
 * status of the SWA layers and any dense intermediate width (the config
 * carries neither intermediate_size nor first_k_dense_replace), the CSA
 * compressed-cache token layout, the hash router's exact form for layers
 * 0..2, and the o_groups output composition.
 */

#define SPARK_DSV4_MODEL_HIDDEN_DIMENSION 4096u                 /* CONFIG hidden_size */
#define SPARK_DSV4_MODEL_LAYER_COUNT 43u                        /* CONFIG num_hidden_layers */
#define SPARK_DSV4_MODEL_VOCAB_COUNT 129280u                    /* CONFIG vocab_size */
#define SPARK_DSV4_MODEL_MAX_POSITIONS 1048576u                 /* CONFIG max_position_embeddings */
#define SPARK_DSV4_MODEL_RMS_NORM_EPSILON 1e-6f                 /* CONFIG rms_norm_eps */
#define SPARK_DSV4_MODEL_ATTN_QUERY_HEAD_COUNT 64u              /* CONFIG num_attention_heads */
#define SPARK_DSV4_MODEL_ATTN_KV_HEAD_COUNT 1u                  /* CONFIG num_key_value_heads */
#define SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION 512u               /* CONFIG head_dim */
#define SPARK_DSV4_MODEL_ATTN_ROPE_DIMENSION 64u                /* CONFIG qk_rope_head_dim */
#define SPARK_DSV4_MODEL_ATTN_ROPE_THETA 10000.0f               /* CONFIG rope_theta */
#define SPARK_DSV4_MODEL_ATTN_YARN_FACTOR 16u                   /* CONFIG rope_scaling.factor */
#define SPARK_DSV4_MODEL_ATTN_YARN_ORIGINAL_POSITIONS 65536u    /* CONFIG rope_scaling.original */
#define SPARK_DSV4_MODEL_COMPRESS_ROPE_THETA 160000.0f          /* CONFIG compress_rope_theta */
#define SPARK_DSV4_MODEL_QUERY_LORA_RANK 1024u                  /* CONFIG q_lora_rank */
#define SPARK_DSV4_MODEL_OUTPUT_LORA_RANK 1024u                 /* CONFIG o_lora_rank */
#define SPARK_DSV4_MODEL_OUTPUT_GROUP_COUNT 8u                  /* CONFIG o_groups */
#define SPARK_DSV4_MODEL_SLIDING_WINDOW_TOKENS 128u             /* CONFIG sliding_window */
#define SPARK_DSV4_MODEL_CSA_COMPRESS_RATIO 4u                  /* CONFIG compress_ratios */
#define SPARK_DSV4_MODEL_HCA_COMPRESS_RATIO 128u                /* CONFIG compress_ratios */
#define SPARK_DSV4_MODEL_INDEX_HEAD_COUNT 64u                   /* CONFIG index_n_heads */
#define SPARK_DSV4_MODEL_INDEX_HEAD_DIMENSION 128u              /* CONFIG index_head_dim */
#define SPARK_DSV4_MODEL_INDEX_TOP_K 512u                       /* CONFIG index_topk */
#define SPARK_DSV4_MODEL_ROUTED_EXPERT_COUNT 256u               /* CONFIG n_routed_experts */
#define SPARK_DSV4_MODEL_SHARED_EXPERT_COUNT 1u                 /* CONFIG n_shared_experts */
#define SPARK_DSV4_MODEL_EXPERTS_PER_TOKEN 6u                   /* CONFIG num_experts_per_tok */
#define SPARK_DSV4_MODEL_EXPERT_INTERMEDIATE_DIMENSION 2048u    /* CONFIG moe_intermediate_size */
#define SPARK_DSV4_MODEL_HASH_ROUTED_LAYER_COUNT 3u             /* CONFIG num_hash_layers */
#define SPARK_DSV4_MODEL_ROUTED_SCALING_FACTOR 1.5f             /* CONFIG routed_scaling_factor */
#define SPARK_DSV4_MODEL_SWIGLU_LIMIT 10.0f                     /* CONFIG swiglu_limit */
#define SPARK_DSV4_MODEL_HC_STREAM_COUNT 4u                     /* CONFIG hc_mult */
#define SPARK_DSV4_MODEL_MTP_LAYER_COUNT 1u                     /* CONFIG num_nextn_predict_layers */
#define SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES 2u

#define SPARK_DSV4_MODEL_ATTN_QUERY_DIMENSION (SPARK_DSV4_MODEL_ATTN_QUERY_HEAD_COUNT * SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION)
#define SPARK_DSV4_MODEL_OUTPUT_GROUP_DIMENSION (SPARK_DSV4_MODEL_ATTN_QUERY_DIMENSION / SPARK_DSV4_MODEL_OUTPUT_GROUP_COUNT)
#define SPARK_DSV4_MODEL_INDEX_DIMENSION (SPARK_DSV4_MODEL_INDEX_HEAD_COUNT * SPARK_DSV4_MODEL_INDEX_HEAD_DIMENSION)
#define SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS (SPARK_DSV4_MODEL_HC_STREAM_COUNT * SPARK_DSV4_MODEL_HIDDEN_DIMENSION)

#define SPARK_DSV4_MODEL_LAYER_KIND_SWA 0u
#define SPARK_DSV4_MODEL_LAYER_KIND_CSA 1u
#define SPARK_DSV4_MODEL_LAYER_KIND_HCA 2u

// The per-layer attention map, transcribed from CONFIG compress_ratios
// (0 -> SWA, 4 -> CSA, 128 -> HCA); the array's 44th entry belongs to the
// MTP layer and is a REFERENCE-PIN. Static per translation unit; the two
// consumers are the pack format and the module.
static const uint8_t SPARK_DSV4_MODEL_LAYER_KIND[SPARK_DSV4_MODEL_LAYER_COUNT] =
{
	0,0,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,
	1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1
};

static inline uint32_t SparkDsv4ModelLayerKind(uint32_t layer_index)
{
	return(layer_index < SPARK_DSV4_MODEL_LAYER_COUNT ? (uint32_t)SPARK_DSV4_MODEL_LAYER_KIND[layer_index] : SPARK_DSV4_MODEL_LAYER_KIND_SWA);
}

#endif
