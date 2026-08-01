#include "inference/llms/glm5_2/layer.cuh"
#include "sparkpipe/spark_glm52_resident_decode_stage_firmware.h"

static int32_t Glm52BindAbsorbed(
    const SparkResidentDecodeStageNodeContext *node,
    LmAbsorbedWeights *weights)
{
    if (node == 0 || weights == 0 ||
        node->query_latent_weight_bf16 == 0 ||
        node->query_rope_weight_bf16 == 0 ||
        node->key_rope_weight_bf16 == 0 ||
        node->kv_latent_weight_bf16 == 0)
    {
        return LM_LAUNCH_ERR_SHAPE;
    }

    memset(weights, 0, sizeof(*weights));
    weights->query_latent_weight = node->query_latent_weight_bf16;
    weights->query_rope_weight = node->query_rope_weight_bf16;
    weights->key_rope_weight = node->key_rope_weight_bf16;
    weights->kv_latent_weight = node->kv_latent_weight_bf16;
    weights->input_dimension = GLM52_HIDDEN;
    weights->query_latent_dimension = GLM52_ATTN_HEADS * GLM52_LATENT;
    weights->query_rope_dimension = GLM52_ATTN_HEADS * GLM52_ROPE_DIM;
    weights->key_rope_dimension = GLM52_ROPE_DIM;
    weights->kv_latent_dimension = GLM52_LATENT;
    return LM_LAUNCH_OK;
}

static int32_t Glm52BindLayer(
    const SparkResidentDecodeStageNodeContext *node,
    uint32_t layer_index,
    const Glm52LayerBuffers *base_buffers,
    Glm52LayerBuffers *layer_buffers)
{
    const SparkGlm52ResidentDecodeStageFp8MoePlan *fp8_plan;
    int32_t status;

    if (node == 0 || base_buffers == 0 || layer_buffers == 0 ||
        layer_index >= GLM52_LAYERS || node->layer_index != layer_index ||
        node->attention_norm_weight_bf16 == 0 ||
        node->post_attention_norm_weight_bf16 == 0 ||
        node->attention_output_weight_bf16 == 0 || node->qk_scale <= 0.0f)
    {
        return LM_LAUNCH_ERR_SHAPE;
    }

    *layer_buffers = *base_buffers;
    layer_buffers->attn_norm_weight = node->attention_norm_weight_bf16;
    layer_buffers->mlp_norm_weight = node->post_attention_norm_weight_bf16;
    layer_buffers->output_weight = node->attention_output_weight_bf16;
    layer_buffers->qk_scale = node->qk_scale;
    layer_buffers->use_absorbed = true;
    status = Glm52BindAbsorbed(node, &layer_buffers->absorbed);
    if (status != LM_LAUNCH_OK)
    {
        return status;
    }

    if (layer_index < GLM52_FIRST_ROUTED_LAYER)
    {
        if (node->dense_gate_weight_bf16 == 0 ||
            node->dense_up_weight_bf16 == 0 ||
            node->dense_down_weight_bf16 == 0)
        {
            return LM_LAUNCH_ERR_SHAPE;
        }
        layer_buffers->dense_gate_weight = node->dense_gate_weight_bf16;
        layer_buffers->dense_up_weight = node->dense_up_weight_bf16;
        layer_buffers->dense_down_weight = node->dense_down_weight_bf16;
        // One GEMM can serve gate and up only when the pack laid the up rows
        // immediately behind the gate rows. That is a pointer comparison at
        // bind time, never an assumption about the pack: anything else takes
        // the layer's two-launch path and loses nothing but the fusion.
        layer_buffers->dense_gate_up_fused =
            (const uint8_t *)node->dense_up_weight_bf16 ==
                (const uint8_t *)node->dense_gate_weight_bf16 +
                    ((uint64_t)GLM52_DENSE_INTERMEDIATE * GLM52_HIDDEN * 2u)
            ? 1u
            : 0u;
        layer_buffers->router_weight = 0;
        layer_buffers->expert_w1_weight = 0;
        layer_buffers->expert_w1_scale = 0;
        layer_buffers->expert_w2_weight = 0;
        layer_buffers->expert_w2_scale = 0;
        return LM_LAUNCH_OK;
    }

    if (node->moe_router_weight_bf16 == 0 ||
        !SparkGlm52ResidentDecodeStageFp8MoePlanIsUsable(node))
    {
        return LM_LAUNCH_ERR_SHAPE;
    }
    fp8_plan = (const SparkGlm52ResidentDecodeStageFp8MoePlan *)
        node->fp8_moe_plan;
    layer_buffers->router_weight = node->moe_router_weight_bf16;
    layer_buffers->expert_w1_weight = fp8_plan->w1_weight_fp8_e4m3;
    layer_buffers->expert_w1_scale = fp8_plan->w1_scale_inv_f32;
    layer_buffers->expert_w2_weight = fp8_plan->w2_weight_fp8_e4m3;
    layer_buffers->expert_w2_scale = fp8_plan->w2_scale_inv_f32;
    layer_buffers->dense_gate_weight = 0;
    layer_buffers->dense_up_weight = 0;
    layer_buffers->dense_down_weight = 0;
    layer_buffers->dense_gate_up_fused = 0u;
    return LM_LAUNCH_OK;
}

static int32_t Glm52LaunchSlice(
    const SparkResidentDecodeStageNodeContext *const *node_contexts,
    const Glm52LayerBuffers *base_buffers,
    uint32_t first_layer,
    uint32_t layer_count,
    uint32_t rows,
    uint32_t packed_rows,
    uint32_t context,
    uint32_t multiprocessors,
    cudaStream_t stream)
{
    Glm52LayerBuffers layer_buffers;
    uint32_t offset;
    uint32_t layer_index;
    int32_t status;

    if (node_contexts == 0 || base_buffers == 0 || layer_count == 0u ||
        first_layer >= GLM52_LAYERS ||
        layer_count > GLM52_LAYERS - first_layer || rows == 0u ||
        multiprocessors == 0u)
    {
        return LM_LAUNCH_ERR_SHAPE;
    }

    for (offset = 0u; offset < layer_count; ++offset)
    {
        layer_index = first_layer + offset;
        status = Glm52BindLayer(
            node_contexts[offset],
            layer_index,
            base_buffers,
            &layer_buffers);
        if (status != LM_LAUNCH_OK)
        {
            return status;
        }
        status = Glm52LayerAttention(
            &layer_buffers,
            rows,
            context,
            layer_index % GLM52_DSA_SHARE_GROUP_LAYERS,
            multiprocessors,
            stream);
        if (status != LM_LAUNCH_OK)
        {
            return status;
        }
        if (layer_index < GLM52_FIRST_ROUTED_LAYER)
        {
            status = Glm52LayerDenseMlp(
                &layer_buffers,
                rows,
                multiprocessors,
                stream);
        }
        else
        {
            status = Glm52LayerMoe(
                &layer_buffers,
                rows,
                packed_rows,
                multiprocessors,
                stream);
        }
        if (status != LM_LAUNCH_OK)
        {
            return status;
        }
    }
    return LM_LAUNCH_OK;
}

extern "C" int32_t Glm52StageSlicePrefill(
    const SparkResidentDecodeStageNodeContext *const *node_contexts,
    void *layer_buffers,
    uint32_t first_layer,
    uint32_t layer_count,
    uint32_t rows,
    uint32_t packed_rows,
    uint32_t context,
    uint32_t multiprocessors,
    const uint32_t *row_positions,
    void *stream)
{
    Glm52LayerBuffers prefill_buffers;

    if (layer_buffers == 0 || row_positions == 0)
    {
        return LM_LAUNCH_ERR_SHAPE;
    }
    prefill_buffers = *(const Glm52LayerBuffers *)layer_buffers;
    prefill_buffers.row_positions = row_positions;
    return Glm52LaunchSlice(
        node_contexts,
        &prefill_buffers,
        first_layer,
        layer_count,
        rows,
        packed_rows,
        context,
        multiprocessors,
        (cudaStream_t)(uintptr_t)stream);
}

extern "C" int32_t Glm52StageSlice(
    const SparkResidentDecodeStageNodeContext *const *node_contexts,
    void *layer_buffers,
    uint32_t first_layer,
    uint32_t layer_count,
    uint32_t rows,
    uint32_t packed_rows,
    uint32_t context,
    uint32_t multiprocessors,
    void *stream)
{
    if (layer_buffers == 0)
    {
        return LM_LAUNCH_ERR_SHAPE;
    }
    return Glm52LaunchSlice(
        node_contexts,
        (const Glm52LayerBuffers *)layer_buffers,
        first_layer,
        layer_count,
        rows,
        packed_rows,
        context,
        multiprocessors,
        (cudaStream_t)(uintptr_t)stream);
}
