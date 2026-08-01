#ifndef SPARKPIPE_SPARK_GLM52_BATCH_TUNING_H
#define SPARKPIPE_SPARK_GLM52_BATCH_TUNING_H

// THE BATCH-VARIANT TUNING HEADER, glm52 resident decode stage.
//
// One source tree, N compiled modules. make emits
// libglm52_resident_decode_stage_b8.a / _b64.a / _b256.a / _b1024.a from the
// same translation units with -DSPARK_BATCH_BUCKET=<n>, and this header is
// the ONLY thing that differs between them. A per-bucket fork of the layer
// behind an #if would be two sources wearing one name - exactly what the
// variant system exists to prevent, so the tuning constants live here or
// nowhere.
//
// A bucket is a CAPACITY CEILING plus tuned geometry, not a fixed batch: the
// b8 module serves 1-8 rows, b64 serves 9-64, and a runtime batch below the
// ceiling needs no recompile. What a smaller ceiling buys is compile-time
// truth the optimizer can act on - the grouped-GEMM tile height below, and
// the pool sizes the consumers of this header scale by the bucket.
//
// THE SET IS {8, 64, 256, 1024}. B8 is the chat batch. B1024 is the maximum
// the stage planner knows (SPARK_STAGE_PLAN_MAX_BATCH_BUCKET) and WINS any
// B64-vs-B1024 tradeoff: the intermediate ceilings exist because their pool
// footprint differs materially from both neighbours, and 1024 is the last
// variant a footprint cut ever drops.

#include <stdint.h>

#include "sparkpipe/spark_glm52_model.h"

#ifndef SPARK_BATCH_BUCKET
// The unflagged build IS the b1024 module: one spelling for the default, so
// every existing consumer of the firmware header sees no change.
#define SPARK_BATCH_BUCKET 1024u
#endif

#if SPARK_BATCH_BUCKET != 8u && SPARK_BATCH_BUCKET != 64u && \
	SPARK_BATCH_BUCKET != 256u && SPARK_BATCH_BUCKET != 1024u
#error SPARK_BATCH_BUCKET must name a built variant bucket: 8, 64, 256, 1024
#endif

// THE CANONICAL MODULE IDENTITY. The batch bucket is the only module-ID field
// that varies per variant, and the prefix and suffix are written once so a
// rename cannot drift the four IDs apart. Each variant publishes under its
// own ID, which is what keeps SPEC.md's content-addressed artifact contract
// untouched: four identities, four immutable records, each validated once,
// resolved by the same identity-key mechanism as any other module.
#define SPARK_GLM52_BATCH_VARIANT_MODULE_ID_PREFIX \
	"spark.glm52.resident_decode_stage.bf16.h6144.h64.d512.r64.k2048"
#define SPARK_GLM52_BATCH_VARIANT_MODULE_ID_SUFFIX \
	"rv256.mtp6.v1"
#define SPARK_GLM52_BATCH_VARIANT_MODULE_ID_B8 \
	SPARK_GLM52_BATCH_VARIANT_MODULE_ID_PREFIX ".b8." \
	SPARK_GLM52_BATCH_VARIANT_MODULE_ID_SUFFIX
#define SPARK_GLM52_BATCH_VARIANT_MODULE_ID_B64 \
	SPARK_GLM52_BATCH_VARIANT_MODULE_ID_PREFIX ".b64." \
	SPARK_GLM52_BATCH_VARIANT_MODULE_ID_SUFFIX
#define SPARK_GLM52_BATCH_VARIANT_MODULE_ID_B256 \
	SPARK_GLM52_BATCH_VARIANT_MODULE_ID_PREFIX ".b256." \
	SPARK_GLM52_BATCH_VARIANT_MODULE_ID_SUFFIX
#define SPARK_GLM52_BATCH_VARIANT_MODULE_ID_B1024 \
	SPARK_GLM52_BATCH_VARIANT_MODULE_ID_PREFIX ".b1024." \
	SPARK_GLM52_BATCH_VARIANT_MODULE_ID_SUFFIX

#if SPARK_BATCH_BUCKET == 8u
#define SPARK_GLM52_BATCH_TUNING_MODULE_ID SPARK_GLM52_BATCH_VARIANT_MODULE_ID_B8
#elif SPARK_BATCH_BUCKET == 64u
#define SPARK_GLM52_BATCH_TUNING_MODULE_ID SPARK_GLM52_BATCH_VARIANT_MODULE_ID_B64
#elif SPARK_BATCH_BUCKET == 256u
#define SPARK_GLM52_BATCH_TUNING_MODULE_ID SPARK_GLM52_BATCH_VARIANT_MODULE_ID_B256
#else
#define SPARK_GLM52_BATCH_TUNING_MODULE_ID SPARK_GLM52_BATCH_VARIANT_MODULE_ID_B1024
#endif

// THE GROUPED TILE HEIGHT AT THE BUCKET CEILING. This is LmLaunchGroupedTileM
// (runtime/launch.h) with tokens fixed at the bucket: the mean group holds
// bucket*top_k/experts rows, the busiest group is priced at twice the mean,
// and the tile rounds UP through 16/32/64 - a tile shorter than the group
// splits it and doubles the weight stream, which is 96 percent of decode
// traffic, while padded mma rows are free. Derived rather than tabulated, so
// a top_k or expert-count change reprices every variant at once. For glm52
// (top-8 of 256) the ceilings land at 16/16/16/64: only b1024's busiest
// group outgrows the shortest tile.
#define SPARK_GLM52_BATCH_TUNING_GROUPED_PEAK_ROWS \
	((((SPARK_BATCH_BUCKET) * SPARK_GLM52_MODEL_MOE_TOP_K + \
	SPARK_GLM52_MODEL_MOE_EXPERT_COUNT - 1u) / \
	SPARK_GLM52_MODEL_MOE_EXPERT_COUNT) * 2u)
#define SPARK_GLM52_BATCH_TUNING_GROUPED_TILE_M \
	(SPARK_GLM52_BATCH_TUNING_GROUPED_PEAK_ROWS <= 16u ? 16u : \
	SPARK_GLM52_BATCH_TUNING_GROUPED_PEAK_ROWS <= 32u ? 32u : 64u)

// RUNTIME VARIANT SELECTION. The stage loader resolves the smallest built
// bucket >= the requested maximum active-sequence count - the ceiling rule
// the bucket semantics above promise. A request above b1024 gets 0, which the
// caller must treat as no-variant: there is nothing larger to fall back to,
// and silently serving it under b1024 would oversubscribe the pools the
// ceiling sizes. The stage-plan bucket ladder (16/32/128/512) does NOT apply
// here - those buckets size plans, these four name compiled modules.
static inline uint32_t SparkGlm52BatchVariantBucketCeiling(
	uint32_t max_active_sequence_count)
{
	if (max_active_sequence_count == 0u ||
		max_active_sequence_count > 1024u)
		return(0u);
	if (max_active_sequence_count <= 8u)
		return(8u);
	if (max_active_sequence_count <= 64u)
		return(64u);
	if (max_active_sequence_count <= 256u)
		return(256u);
	return(1024u);
}

static inline const char *SparkGlm52BatchVariantModuleId(
	uint32_t batch_bucket)
{
	switch (batch_bucket)
	{
	case 8u:
		return(SPARK_GLM52_BATCH_VARIANT_MODULE_ID_B8);
	case 64u:
		return(SPARK_GLM52_BATCH_VARIANT_MODULE_ID_B64);
	case 256u:
		return(SPARK_GLM52_BATCH_VARIANT_MODULE_ID_B256);
	case 1024u:
		return(SPARK_GLM52_BATCH_VARIANT_MODULE_ID_B1024);
	default:
		return(0);
	}
}

#endif
