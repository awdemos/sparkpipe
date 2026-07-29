#ifndef SPARKPIPE_SPARK_GLM52_KV_CACHE_H
#define SPARKPIPE_SPARK_GLM52_KV_CACHE_H

#include "sparkpipe/spark_kv_cache.h"
#include "sparkpipe/spark_glm52_model.h"

// What remains after A3: the glm NUMBERS. The paged arena, the block views,
// the prefetch lanes, the JIT stage budgets and both MLA-compressed layouts
// are include/sparkpipe/spark_kv_cache.h now - the machinery already took
// its geometry through the configuration, and the header's glm content was
// three constants and a name.

#define SPARK_GLM52_KV_BLOCK_TOKENS 64u
#define SPARK_GLM52_KV_CONTEXT_TOKENS \
    SPARK_GLM52_MODEL_MAXIMUM_CONTEXT_TOKENS
#define SPARK_GLM52_KV_POOL_TOKENS SPARK_GLM52_MODEL_KV_POOL_TOKENS


// The divisibility of glm's context and pool by glm's block size is a claim
// about GLM'S NUMBERS, so the guard lives with them, not in the machinery.
#if (SPARK_GLM52_KV_CONTEXT_TOKENS % SPARK_GLM52_KV_BLOCK_TOKENS) != 0u || \
    (SPARK_GLM52_KV_POOL_TOKENS % SPARK_GLM52_KV_BLOCK_TOKENS) != 0u
#error kv context and pool token counts must be multiples of the kv block token count
#endif

#endif
