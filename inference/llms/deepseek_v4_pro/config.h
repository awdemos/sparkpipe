#pragma once

#include "sparkpipe/spark_dsv4_pro_model.h"
#include "inference/kernels/layer_kind.cuh"

#define DSV4_PRO_HIDDEN SPARK_DSV4_PRO_HIDDEN_DIMENSION
#define DSV4_PRO_LAYERS SPARK_DSV4_PRO_LAYER_COUNT
#define DSV4_PRO_VOCAB SPARK_DSV4_PRO_VOCAB_COUNT
#define DSV4_PRO_ATTN_HEADS SPARK_DSV4_PRO_ATTENTION_HEAD_COUNT
#define DSV4_PRO_KV_HEADS SPARK_DSV4_PRO_KV_HEAD_COUNT
#define DSV4_PRO_HEAD_DIM SPARK_DSV4_PRO_HEAD_DIMENSION
#define DSV4_PRO_INDEX_TOP_K SPARK_DSV4_PRO_INDEX_TOP_K
#define DSV4_PRO_EXPERTS SPARK_DSV4_PRO_ROUTED_EXPERT_COUNT
#define DSV4_PRO_TOP_K SPARK_DSV4_PRO_EXPERTS_PER_TOKEN
#define DSV4_PRO_EXPERT_INTERMEDIATE SPARK_DSV4_PRO_EXPERT_INTERMEDIATE_DIMENSION
#define DSV4_PRO_HYPER_CONNECTION_STREAMS SPARK_DSV4_PRO_HYPER_CONNECTION_STREAM_COUNT


static inline uint16_t Dsv4ProCompressionRatio(uint32_t layer_index)
{
    return SparkDsv4ProBackboneCompressionRatio(layer_index);
}

#define DSV4_PRO_LAYER_KIND(layer) \
    (Dsv4ProCompressionRatio((uint32_t)(layer)) == 0u \
        ? LM_LAYER_WINDOW \
        : (Dsv4ProCompressionRatio((uint32_t)(layer)) == 4u \
            ? LM_LAYER_SPARSE \
            : LM_LAYER_COMPRESSED))
