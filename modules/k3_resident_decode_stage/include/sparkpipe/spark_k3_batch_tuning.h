#ifndef SPARKPIPE_SPARK_K3_BATCH_TUNING_H
#define SPARKPIPE_SPARK_K3_BATCH_TUNING_H

// THE BATCH-VARIANT TUNING HEADER, k3 resident decode stage.
//
// Same contract as the glm52 variant header: one source tree, N compiled
// modules, -DSPARK_BATCH_BUCKET=<n> the ONLY difference between them, a
// bucket a capacity ceiling rather than a fixed batch. Read the why in
// modules/glm52_resident_decode_stage/include/sparkpipe/spark_glm52_batch_tuning.h
// first; this file carries only what is k3's.
//
// k3's module ID has no batch field today, so the variant convention INSERTS
// .b<n>. ahead of the version suffix. That makes a variant-published k3
// module a new module identity, not a rename of the old one: the unbucketed
// ID stays valid for the unbucketed archive until the k3 module Makefile and
// the contract generator adopt the bucketed form. Those two are the open
// dependency - this header is the interface they adopt against.
//
// THE SET IS {8, 64, 256, 1024}, the same ladders as glm52: B8 is chat,
// B1024 wins any B64-vs-B1024 tradeoff, and the intermediate ceilings exist
// only where their pool footprint differs materially.

#include <stdint.h>

#include "inference/llms/kimi_k3/config.h"

#ifndef SPARK_BATCH_BUCKET
// The unflagged build IS the b1024 module.
#define SPARK_BATCH_BUCKET 1024u
#endif

#if SPARK_BATCH_BUCKET != 8u && SPARK_BATCH_BUCKET != 64u && \
	SPARK_BATCH_BUCKET != 256u && SPARK_BATCH_BUCKET != 1024u
#error SPARK_BATCH_BUCKET must name a built variant bucket: 8, 64, 256, 1024
#endif

// THE CANONICAL MODULE IDENTITY. Prefix and suffix written once; the four
// variant IDs are compositions, so a rename cannot drift them apart. Each
// variant publishes under its own ID and keeps SPEC.md's content-addressed
// artifact contract intact.
#define SPARK_K3_BATCH_VARIANT_MODULE_ID_PREFIX \
	"spark.k3.resident_decode_stage.mxfp4_routed_bf16_rest.h7168.l93.kda69.mla24"
#define SPARK_K3_BATCH_VARIANT_MODULE_ID_SUFFIX \
	"v2"
#define SPARK_K3_BATCH_VARIANT_MODULE_ID_B8 \
	SPARK_K3_BATCH_VARIANT_MODULE_ID_PREFIX ".b8." \
	SPARK_K3_BATCH_VARIANT_MODULE_ID_SUFFIX
#define SPARK_K3_BATCH_VARIANT_MODULE_ID_B64 \
	SPARK_K3_BATCH_VARIANT_MODULE_ID_PREFIX ".b64." \
	SPARK_K3_BATCH_VARIANT_MODULE_ID_SUFFIX
#define SPARK_K3_BATCH_VARIANT_MODULE_ID_B256 \
	SPARK_K3_BATCH_VARIANT_MODULE_ID_PREFIX ".b256." \
	SPARK_K3_BATCH_VARIANT_MODULE_ID_SUFFIX
#define SPARK_K3_BATCH_VARIANT_MODULE_ID_B1024 \
	SPARK_K3_BATCH_VARIANT_MODULE_ID_PREFIX ".b1024." \
	SPARK_K3_BATCH_VARIANT_MODULE_ID_SUFFIX

#if SPARK_BATCH_BUCKET == 8u
#define SPARK_K3_BATCH_TUNING_MODULE_ID SPARK_K3_BATCH_VARIANT_MODULE_ID_B8
#elif SPARK_BATCH_BUCKET == 64u
#define SPARK_K3_BATCH_TUNING_MODULE_ID SPARK_K3_BATCH_VARIANT_MODULE_ID_B64
#elif SPARK_BATCH_BUCKET == 256u
#define SPARK_K3_BATCH_TUNING_MODULE_ID SPARK_K3_BATCH_VARIANT_MODULE_ID_B256
#else
#define SPARK_K3_BATCH_TUNING_MODULE_ID SPARK_K3_BATCH_VARIANT_MODULE_ID_B1024
#endif

// THE GROUPED TILE HEIGHT AT THE BUCKET CEILING, LmLaunchGroupedTileM
// (runtime/launch.h) with tokens fixed at the bucket. Derived, not tabulated.
// For k3 (top-16 of 896) the ceilings land at 16/16/16/64, same shape as
// glm52: only b1024's busiest group outgrows the shortest tile.
#define SPARK_K3_BATCH_TUNING_GROUPED_PEAK_ROWS \
	((((SPARK_BATCH_BUCKET) * K3_TOP_K + K3_EXPERTS - 1u) / \
	K3_EXPERTS) * 2u)
#define SPARK_K3_BATCH_TUNING_GROUPED_TILE_M \
	(SPARK_K3_BATCH_TUNING_GROUPED_PEAK_ROWS <= 16u ? 16u : \
	SPARK_K3_BATCH_TUNING_GROUPED_PEAK_ROWS <= 32u ? 32u : 64u)

// RUNTIME VARIANT SELECTION: smallest built bucket >= the requested maximum
// active-sequence count, 0 above b1024. Same contract as glm52's; duplicated
// per family because the module IDs it selects between are model content.
static inline uint32_t SparkK3BatchVariantBucketCeiling(
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

static inline const char *SparkK3BatchVariantModuleId(
	uint32_t batch_bucket)
{
	switch (batch_bucket)
	{
	case 8u:
		return(SPARK_K3_BATCH_VARIANT_MODULE_ID_B8);
	case 64u:
		return(SPARK_K3_BATCH_VARIANT_MODULE_ID_B64);
	case 256u:
		return(SPARK_K3_BATCH_VARIANT_MODULE_ID_B256);
	case 1024u:
		return(SPARK_K3_BATCH_VARIANT_MODULE_ID_B1024);
	default:
		return(0);
	}
}

#endif
