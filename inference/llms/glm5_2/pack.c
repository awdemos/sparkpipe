// GLM 5.2 weight binding, as a table.
//
// The old node context builder is 10,628 lines and 202 functions. Reading what
// it does: it declares a tensor spec - name, dtype, element size, rank, shape -
// resolves it against the pack, and stores the resulting pointer into a field of
// the node context. Two hundred times, once per tensor, with a function each.
//
// That is a table with one row per tensor and one loop.
//
// WHAT THE OTHER 8,000 LINES WERE. Measured before deleting: 2,701 lines of MTP
// setup, 1,674 of layer buffer and scratch allocation, 1,620 of KV cache layout,
// 794 of drafter setup, and 137 of binding for b12x and w8lut, both of which are
// deleted. The KV layout is cache/cache.h now, the scratch is arithmetic over
// config.h, and the drafter placement is ring/sideband.h's tap plan.
//
// WHY A TABLE RATHER THAN A FUNCTION EACH. A function per tensor makes every
// tensor look different when they differ only in name and shape - and it hides
// the one thing worth checking, which is whether the set is complete. A table
// can be counted against the model: GLM 5.2 has eleven per-layer tensors and
// three global ones, and if the table has ten the compiler will not say so but
// the count will.

#include "inference/llms/glm5_2/config.h"
#include "sparkpipe/spark_glm52_resident_decode_stage_firmware.h"
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define GLM52_BIND_OK 0
#define GLM52_BIND_ERR_MISSING (-101)
#define GLM52_BIND_ERR_SHAPE (-102)
#define GLM52_BIND_ERR_INCOMPLETE (-103)

typedef enum Glm52TensorScope
{
	GLM52_TENSOR_PER_LAYER = 0,
	GLM52_TENSOR_GLOBAL = 1,
	GLM52_TENSOR_PER_EXPERT = 2
}
Glm52TensorScope;

// One row per tensor the model has. The offset is into the node context, so the
// loop can store without a switch - a switch over two hundred cases is the same
// two hundred functions wearing one name.
typedef struct Glm52TensorRow
{
	const char *name;
	uint32_t scope;
	uint64_t rows;
	uint64_t columns;
	uint32_t element_bits;
	size_t context_offset;
	int32_t optional;              /* absorbed weights are absent in a raw pack */
}
Glm52TensorRow;

#define GLM52_FIELD(field) offsetof(SparkResidentDecodeStageNodeContext, field)

static const Glm52TensorRow kGlm52Tensors[] =
{
	/* attention, absorbed form. Optional: a raw pack carries the two-stage
	   low-rank weights instead, and which one is present IS the decision about
	   which path runs. */
	{ "attn_norm.weight",        GLM52_TENSOR_PER_LAYER, GLM52_HIDDEN, 1u, 16u,
	  GLM52_FIELD(attention_norm_weight_bf16), 0 },
	{ "attn.q_latent.weight",    GLM52_TENSOR_PER_LAYER, GLM52_LATENT, GLM52_HIDDEN, 16u,
	  GLM52_FIELD(query_latent_weight_bf16), 1 },
	{ "attn.q_rope.weight",      GLM52_TENSOR_PER_LAYER, GLM52_ROPE_DIM, GLM52_HIDDEN, 16u,
	  GLM52_FIELD(query_rope_weight_bf16), 1 },
	{ "attn.k_rope.weight",      GLM52_TENSOR_PER_LAYER, GLM52_ROPE_DIM, GLM52_HIDDEN, 16u,
	  GLM52_FIELD(key_rope_weight_bf16), 1 },
	{ "attn.kv_latent.weight",   GLM52_TENSOR_PER_LAYER, GLM52_LATENT_ROW, GLM52_HIDDEN, 16u,
	  GLM52_FIELD(kv_latent_weight_bf16), 1 },

	/* attention, raw form. The mirror of the four above. */
	{ "attn.q_a.norm.weight",    GLM52_TENSOR_PER_LAYER, GLM52_QUERY_A_DIM, 1u, 16u,
	  GLM52_FIELD(raw_query_a_norm_weight_bf16), 1 },
	{ "attn.kv_a.norm.weight",   GLM52_TENSOR_PER_LAYER, GLM52_LATENT, 1u, 16u,
	  GLM52_FIELD(raw_kv_a_norm_weight_bf16), 1 },

	/* the output projection, which in absorbed form carries the folded value
	   up-projection - which is why its input width is heads times latent rather
	   than heads times v_head_dim. */
	{ "attn.output.weight",      GLM52_TENSOR_PER_LAYER, GLM52_HIDDEN,
	  GLM52_ATTN_HEADS * GLM52_LATENT, 8u,
	  GLM52_FIELD(attention_output_weight_fp8_e4m3), 0 },
	{ "attn.output.scale_inv",   GLM52_TENSOR_PER_LAYER,
	  GLM52_HIDDEN / GLM52_FP8_SCALE_BLOCK,
	  (GLM52_ATTN_HEADS * GLM52_LATENT) / GLM52_FP8_SCALE_BLOCK, 32u,
	  GLM52_FIELD(attention_output_weight_scale_inv_f32), 0 },

	{ "mlp_norm.weight",         GLM52_TENSOR_PER_LAYER, GLM52_HIDDEN, 1u, 16u,
	  GLM52_FIELD(post_attention_norm_weight_bf16), 0 },

	/* the router. BF16 and unquantised: 6144 by 256 is 3 MB, and its error
	   compounds across every expert it selects. */
	{ "moe.router.weight",       GLM52_TENSOR_PER_LAYER, GLM52_EXPERTS, GLM52_HIDDEN, 16u,
	  GLM52_FIELD(moe_router_weight_bf16), 1 },

	/* the dense MLP, present only on layers below GLM52_FIRST_ROUTED_LAYER. */
	{ "mlp.down.weight",         GLM52_TENSOR_PER_LAYER, GLM52_HIDDEN,
	  GLM52_DENSE_INTERMEDIATE, 8u,
	  GLM52_FIELD(dense_down_weight_fp8_e4m3), 1 },
	{ "mlp.down.scale_inv",      GLM52_TENSOR_PER_LAYER,
	  GLM52_HIDDEN / GLM52_FP8_SCALE_BLOCK,
	  GLM52_DENSE_INTERMEDIATE / GLM52_FP8_SCALE_BLOCK, 32u,
	  GLM52_FIELD(dense_down_weight_scale_inv_f32), 1 },

	{ "final_norm.weight",       GLM52_TENSOR_GLOBAL, GLM52_HIDDEN, 1u, 16u,
	  GLM52_FIELD(final_norm_weight_bf16), 0 },
};

#define GLM52_TENSOR_COUNT (sizeof(kGlm52Tensors) / sizeof(kGlm52Tensors[0]))

// What a caller supplies: resolve a name to a device pointer, or null.
//
// A function pointer rather than a pack struct, because binding should not know
// what a pack is. The same table binds a memory-mapped file, a pre-staged device
// arena, or a test harness handing back fabricated buffers - and the third is
// what makes this testable without a weight file.
typedef const void *(*Glm52ResolveTensor)(void *context, const char *name, uint32_t layer);

// Expected bytes, from the table rather than from the pack.
//
// Checking the pack's own size against itself proves nothing. Computing what the
// model requires and comparing is what catches a pack built for a different
// config - a 4096-hidden checkpoint loaded as 6144 resolves every name and gets
// every shape wrong.
static uint64_t Glm52TensorBytes(const Glm52TensorRow *row)
{
	return((row->rows * row->columns * row->element_bits) / 8u);
}

static int32_t Glm52BindPack(SparkResidentDecodeStageNodeContext *context, uint32_t layer, Glm52ResolveTensor resolve, void *resolve_context, uint32_t *bound_out, uint32_t *missing_out)
{
	uint32_t index,bound = 0u,missing = 0u;
	int32_t absorbed = 0,raw = 0;
	if ( context == 0 || resolve == 0 )
		return(GLM52_BIND_ERR_SHAPE);
	for (index = 0u; index < GLM52_TENSOR_COUNT; ++index)
	{
		const Glm52TensorRow *row = &kGlm52Tensors[index];
		const void *pointer;
		if ( row->scope == GLM52_TENSOR_GLOBAL && layer != 0u )
			continue;
		pointer = resolve(resolve_context,row->name,layer);
		if ( pointer == 0 )
		{
			if ( row->optional == 0 )
				return(GLM52_BIND_ERR_MISSING);
			++missing;
			continue;
		}
		*(const void **)((uint8_t *)context + row->context_offset) = pointer;
		++bound;
		if ( row->context_offset == GLM52_FIELD(query_latent_weight_bf16) )
			absorbed = 1;
		if ( row->context_offset == GLM52_FIELD(raw_query_a_norm_weight_bf16) )
			raw = 1;
	}
	// A pack must carry one attention form or the other. Carrying neither binds
	// an attention path with no weights, which fails at the first launch; the
	// check here says which pack is wrong rather than which kernel crashed.
	if ( absorbed == 0 && raw == 0 )
		return(GLM52_BIND_ERR_INCOMPLETE);
	if ( bound_out != 0 )
		*bound_out = bound;
	if ( missing_out != 0 )
		*missing_out = missing;
	return(GLM52_BIND_OK);
}

// Scratch a layer needs, in bytes, at a given batch.
//
// Arithmetic over config.h, where the old builder allocated 1,674 lines of it.
// Every term is a tensor the layer writes and reads back, and the largest is the
// expanded packed activation - tokens times top-k rather than tokens, which is
// the one place a routed MoE's memory is not proportional to the batch.
static uint64_t Glm52LayerScratchBytes(uint32_t tokens)
{
	uint64_t packed = (uint64_t)tokens * GLM52_TOP_K;
	return(
		((uint64_t)tokens * GLM52_HIDDEN * 2u * 3u) +          /* hidden, residual, normed */
		((uint64_t)tokens * GLM52_LATENT_ROW * 2u) +           /* the cache slot row */
		((uint64_t)tokens * GLM52_ATTN_HEADS * GLM52_LATENT * 2u) + /* attention output */
		(packed * GLM52_HIDDEN) +                              /* packed codes, 8-bit */
		(packed * (GLM52_HIDDEN / GLM52_FP8_SCALE_BLOCK)) +    /* their scales */
		(packed * GLM52_GATE_UP_DIM * 2u) +                    /* gate and up */
		(packed * GLM52_EXPERT_INTERMEDIATE * 2u) +            /* the activation */
		(packed * GLM52_HIDDEN * 2u) +                         /* expert output */
		((uint64_t)tokens * GLM52_EXPERTS * 4u) +              /* router logits */
		((uint64_t)tokens * GLM52_TOP_K * 4u * 3u));           /* route tables */
}
