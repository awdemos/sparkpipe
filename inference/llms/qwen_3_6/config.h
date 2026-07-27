#pragma once
// Qwen 3.6. Shapes and constants only.
//
// Gated DeltaNet on 48 of 64 layers, full attention on the other 16, in a fixed
// period. Same shape as K3 - a recurrent state for most layers and a KV cache
// for a few - reached from a different direction, which is the argument that
// this is an architecture class rather than one vendor's choice.
#include <stdint.h>

#define QWEN36_HIDDEN 5120u                    /* CONFIG hidden_size */
#define QWEN36_LAYERS 64u                      /* CONFIG num_hidden_layers */
#define QWEN36_VOCAB 248320u                   /* CONFIG vocab_size, padded */
#define QWEN36_RMS_EPSILON 1e-06f              /* CONFIG rms_norm_eps */
#define QWEN36_FFN_INTERMEDIATE 17408u         /* CONFIG intermediate_size */

#define QWEN36_ATTENTION_PERIOD 4u             /* CONFIG full_attention_interval */
#define QWEN36_FULL_PHASE 3u                   /* CONFIG layer_types order */
#define QWEN36_LAYER_IS_LINEAR(layer) (((layer) % QWEN36_ATTENTION_PERIOD) != QWEN36_FULL_PHASE)

#define QWEN36_GDN_KEY_HEADS 16u               /* CONFIG linear_num_key_heads */
#define QWEN36_GDN_VALUE_HEADS 48u             /* CONFIG linear_num_value_heads */
#define QWEN36_GDN_KEY_DIM 128u                /* CONFIG linear_key_head_dim */
#define QWEN36_GDN_VALUE_DIM 128u              /* CONFIG linear_value_head_dim */
#define QWEN36_GDN_CONV_KERNEL 4u              /* CONFIG linear_conv_kernel_dim */

// The GDN state plus the short causal convolution window it carries. Both are
// per-sequence and neither grows, so both live in one non-growing pool.
#define QWEN36_GDN_STATE_BYTES \
	((QWEN36_GDN_KEY_HEADS * QWEN36_GDN_KEY_DIM * QWEN36_GDN_VALUE_DIM * 2u) \
	 + (QWEN36_GDN_KEY_HEADS * QWEN36_GDN_KEY_DIM * QWEN36_GDN_CONV_KERNEL * 2u))

#define QWEN36_MTP_LAYERS 1u                   /* CONFIG mtp_num_hidden_layers */
#define QWEN36_KV_BITS 16u
#define QWEN36_KV_PAGE_SLOTS 64u
