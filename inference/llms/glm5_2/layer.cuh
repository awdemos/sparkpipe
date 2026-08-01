#pragma once

#include "runtime/gemm.cuh"
#include "inference/kernels/norm.cuh"
#include "inference/kernels/attn.cuh"
#include "inference/kernels/topk.cuh"
#include "inference/kernels/route.cuh"
#include "inference/kernels/project.cuh"
#include "inference/kernels/head.cuh"
#include "inference/kernels/formats/bf16.cuh"
#include "inference/kernels/formats/fp8.cuh"
#include "inference/llms/glm5_2/config.h"

using Glm52Kv = LmKvLatent<
    GLM52_KV_BITS,
    GLM52_LATENT,
    GLM52_ROPE_DIM,
    GLM52_KV_PAGE_SLOTS>;

#define GLM52_LAYER_THREADS 256u
#define GLM52_LAYER_TILE_N 128u
#define GLM52_LAYER_STAGES 2u
#define GLM52_LAYER_WARPS 8u
#define GLM52_HEAD_TILE 1024u

static_assert(
    GLM52_HIDDEN % LmBf16Format::kTileK == 0u,
    "GLM 5.2 hidden projections must cover every BF16 K tile");
static_assert(
    GLM52_QUERY_A_DIM % LmBf16Format::kTileK == 0u,
    "GLM 5.2 low-rank query projections must cover every BF16 K tile");
static_assert(
    GLM52_DSA_QUERY_DIM % LmBf16Format::kTileK == 0u,
    "GLM 5.2 DSA index queries must cover every BF16 K tile");
static_assert(
    (GLM52_ATTN_HEADS * GLM52_LATENT) % LmBf16Format::kTileK == 0u,
    "GLM 5.2 absorbed attention output must cover every BF16 K tile");
static_assert(
    GLM52_DENSE_INTERMEDIATE % LmBf16Format::kTileK == 0u,
    "GLM 5.2 dense FFN down projection must cover every BF16 K tile");
static_assert(
    GLM52_EXPERT_INTERMEDIATE % LmBf16Format::kTileK == 0u,
    "GLM 5.2 expert down projection must cover every BF16 K tile");
static_assert(
    GLM52_HIDDEN % LmFp8::kScaleGroup == 0u &&
        GLM52_EXPERT_INTERMEDIATE % LmFp8::kScaleGroup == 0u,
    "GLM 5.2 FP8 expert weights require complete scale groups");

struct Glm52LayerBuffers
{
    const uint32_t *dense_row_offset;
    uint32_t *dense_tile_prefix;

    const void *attn_norm_weight;
    LmAbsorbedWeights absorbed;
    bool use_absorbed;
    float qk_scale;
    const void *output_weight;
    const void *mlp_norm_weight;
    const void *router_weight;
    const void *dense_gate_weight;
    const void *dense_up_weight;
    const void *dense_down_weight;
    const void *expert_w1_weight;
    const void *expert_w1_scale;
    const void *expert_w2_weight;
    const void *expert_w2_scale;

    uint16_t *hidden_bf16;
    uint16_t *residual_bf16;
    uint16_t *normed_bf16;
    LmAbsorbedOutputs projected;
    uint16_t *kv_slot_bf16;
    uint16_t *attention_latent_bf16;
    uint16_t *attention_out_bf16;
    uint16_t *gate_up_bf16;
    uint16_t *intermediate_bf16;
    uint16_t *expert_out_bf16;
    float *router_logits;
    uint32_t *route_expert;
    float *route_weight;
    uint32_t *route_source_token;
    uint32_t *route_packed_row;
    float *head_candidate_score;
    uint32_t *head_candidate_token;
    uint32_t *output_token;
    float *output_score;
    uint32_t *group_row_offset;
    uint32_t *group_tile_prefix_w1;
    uint32_t *group_tile_prefix_w2;

    LmKvView cache;
    const uint32_t *sequence_of_row;
    const uint32_t *context_length;
    const uint32_t *positions;
    const uint32_t *row_positions;
};

static int32_t Glm52LaunchBf16Linear(
    const uint16_t *activation_bf16,
    const void *weight_bf16,
    uint16_t *output_bf16,
    const uint32_t *row_offset,
    uint32_t *tile_prefix,
    uint32_t rows,
    uint32_t input_dimension,
    uint32_t output_dimension,
    uint32_t output_row_stride,
    uint32_t output_column_offset,
    uint32_t multiprocessors,
    cudaStream_t stream)
{
    LmGemmArguments gemm;

    if (activation_bf16 == 0 || weight_bf16 == 0 || output_bf16 == 0 ||
        row_offset == 0 || tile_prefix == 0 || rows == 0u ||
        input_dimension == 0u || output_dimension == 0u ||
        multiprocessors == 0u)
    {
        return LM_LAUNCH_ERR_SHAPE;
    }

    memset(&gemm, 0, sizeof(gemm));
    gemm.scale_a = LmScaleTensorNone();
    gemm.scale_b = LmScaleTensorNone();
    gemm.group_row_offset = row_offset;
    gemm.group_tile_prefix = tile_prefix;
    gemm.output_bf16 = output_bf16;
    gemm.output_row_stride = output_row_stride;
    gemm.output_column_offset = output_column_offset;
    return LmGemmLaunch<
        LmBf16Format,
        GLM52_LAYER_TILE_N,
        LmBf16Format::kTileK,
        GLM52_LAYER_STAGES,
        GLM52_LAYER_WARPS>(
            &gemm,
            activation_bf16,
            weight_bf16,
            rows,
            rows,
            1u,
            1u,
            input_dimension,
            output_dimension,
            multiprocessors,
            false,
            stream);
}

static int32_t Glm52LayerAttention(
    const Glm52LayerBuffers *buffers,
    uint32_t rows,
    uint32_t context,
    uint32_t layer_in_group,
    uint32_t multiprocessors,
    cudaStream_t stream)
{
    int32_t status;

    (void)layer_in_group;
    if (buffers == 0 || rows == 0u || context == 0u ||
        context > GLM52_DSA_SELECTED || !buffers->use_absorbed ||
        buffers->qk_scale <= 0.0f || buffers->hidden_bf16 == 0 ||
        buffers->residual_bf16 == 0 || buffers->normed_bf16 == 0 ||
        buffers->attn_norm_weight == 0 || buffers->kv_slot_bf16 == 0 ||
        buffers->attention_latent_bf16 == 0 ||
        buffers->attention_out_bf16 == 0 || buffers->output_weight == 0 ||
        buffers->sequence_of_row == 0 || buffers->context_length == 0 ||
        buffers->positions == 0)
    {
        return LM_LAUNCH_ERR_SHAPE;
    }

    LM_LAUNCH(
        (LmFusedResidualRmsNormKernel<GLM52_LAYER_THREADS, uint16_t>),
        rows,
        GLM52_LAYER_THREADS,
        (GLM52_HIDDEN + 8u) * sizeof(float),
        stream,
        buffers->hidden_bf16,
        buffers->residual_bf16,
        (const uint16_t *)buffers->attn_norm_weight,
        buffers->residual_bf16,
        buffers->normed_bf16,
        GLM52_HIDDEN,
        GLM52_HIDDEN,
        GLM52_RMS_EPSILON);

    status = LmAbsorbedProject<LmBf16Format>(
        &buffers->absorbed,
        &buffers->projected,
        buffers->normed_bf16,
        0,
        0,
        buffers->dense_row_offset,
        buffers->dense_tile_prefix,
        rows,
        multiprocessors,
        stream);
    if (status != LM_LAUNCH_OK)
    {
        return status;
    }

    LM_LAUNCH(
        (LmRopePerHeadKernel<GLM52_LAYER_THREADS>),
        dim3(rows, GLM52_ATTN_HEADS),
        GLM52_LAYER_THREADS,
        0,
        stream,
        buffers->projected.query_rope_bf16,
        buffers->positions,
        GLM52_ATTN_HEADS,
        GLM52_ROPE_DIM,
        GLM52_ROPE_DIM,
        GLM52_ROPE_THETA);
    LM_LAUNCH(
        (LmRopeKernel<GLM52_LAYER_THREADS>),
        rows,
        GLM52_LAYER_THREADS,
        0,
        stream,
        buffers->projected.key_rope_bf16,
        buffers->positions,
        GLM52_ROPE_DIM,
        0u,
        GLM52_ROPE_DIM,
        GLM52_ROPE_THETA);
    LM_LAUNCH(
        (LmJoinRowsKernel<GLM52_LAYER_THREADS>),
        rows,
        GLM52_LAYER_THREADS,
        0,
        stream,
        buffers->projected.kv_latent_bf16,
        GLM52_LATENT,
        buffers->projected.key_rope_bf16,
        GLM52_ROPE_DIM,
        buffers->kv_slot_bf16,
        rows);
    LM_LAUNCH(
        (LmKvStoreKernel<Glm52Kv, GLM52_LAYER_THREADS>),
        rows,
        GLM52_LAYER_THREADS,
        0,
        stream,
        buffers->cache,
        buffers->kv_slot_bf16,
        buffers->sequence_of_row,
        buffers->positions,
        rows,
        GLM52_LATENT_ROW);
    LM_LAUNCH(
        (LmLatentAttentionDecodeKernel<
            Glm52Kv,
            GLM52_LAYER_THREADS,
            GLM52_LATENT,
            GLM52_ROPE_DIM>),
        dim3(rows, GLM52_ATTN_HEADS),
        GLM52_LAYER_THREADS,
        0,
        stream,
        buffers->projected.query_latent_bf16,
        buffers->projected.query_rope_bf16,
        buffers->cache,
        buffers->sequence_of_row,
        buffers->context_length,
        0,
        0u,
        GLM52_ATTN_HEADS,
        buffers->qk_scale,
        buffers->attention_latent_bf16,
        buffers->row_positions);

    return Glm52LaunchBf16Linear(
        buffers->attention_latent_bf16,
        buffers->output_weight,
        buffers->attention_out_bf16,
        buffers->dense_row_offset,
        buffers->dense_tile_prefix,
        rows,
        GLM52_ATTN_HEADS * GLM52_LATENT,
        GLM52_HIDDEN,
        GLM52_HIDDEN,
        0u,
        multiprocessors,
        stream);
}

static int32_t Glm52LayerDenseMlp(
    const Glm52LayerBuffers *buffers,
    uint32_t rows,
    uint32_t multiprocessors,
    cudaStream_t stream)
{
    int32_t status;

    if (buffers == 0 || rows == 0u || buffers->attention_out_bf16 == 0 ||
        buffers->residual_bf16 == 0 || buffers->mlp_norm_weight == 0 ||
        buffers->normed_bf16 == 0 || buffers->dense_gate_weight == 0 ||
        buffers->dense_up_weight == 0 || buffers->dense_down_weight == 0 ||
        buffers->gate_up_bf16 == 0 || buffers->intermediate_bf16 == 0 ||
        buffers->hidden_bf16 == 0)
    {
        return LM_LAUNCH_ERR_SHAPE;
    }

    LM_LAUNCH(
        (LmFusedResidualRmsNormKernel<GLM52_LAYER_THREADS, uint16_t>),
        rows,
        GLM52_LAYER_THREADS,
        (GLM52_HIDDEN + 8u) * sizeof(float),
        stream,
        buffers->attention_out_bf16,
        buffers->residual_bf16,
        (const uint16_t *)buffers->mlp_norm_weight,
        buffers->residual_bf16,
        buffers->normed_bf16,
        GLM52_HIDDEN,
        GLM52_HIDDEN,
        GLM52_RMS_EPSILON);

    status = Glm52LaunchBf16Linear(
        buffers->normed_bf16,
        buffers->dense_gate_weight,
        buffers->gate_up_bf16,
        buffers->dense_row_offset,
        buffers->dense_tile_prefix,
        rows,
        GLM52_HIDDEN,
        GLM52_DENSE_INTERMEDIATE,
        GLM52_DENSE_INTERMEDIATE * 2u,
        0u,
        multiprocessors,
        stream);
    if (status != LM_LAUNCH_OK)
    {
        return status;
    }
    status = Glm52LaunchBf16Linear(
        buffers->normed_bf16,
        buffers->dense_up_weight,
        buffers->gate_up_bf16,
        buffers->dense_row_offset,
        buffers->dense_tile_prefix,
        rows,
        GLM52_HIDDEN,
        GLM52_DENSE_INTERMEDIATE,
        GLM52_DENSE_INTERMEDIATE * 2u,
        GLM52_DENSE_INTERMEDIATE,
        multiprocessors,
        stream);
    if (status != LM_LAUNCH_OK)
    {
        return status;
    }

    LM_LAUNCH(
        (LmSiluMulKernel<GLM52_LAYER_THREADS>),
        rows,
        GLM52_LAYER_THREADS,
        0,
        stream,
        buffers->gate_up_bf16,
        buffers->intermediate_bf16,
        GLM52_DENSE_INTERMEDIATE,
        true);

    return Glm52LaunchBf16Linear(
        buffers->intermediate_bf16,
        buffers->dense_down_weight,
        buffers->hidden_bf16,
        buffers->dense_row_offset,
        buffers->dense_tile_prefix,
        rows,
        GLM52_DENSE_INTERMEDIATE,
        GLM52_HIDDEN,
        GLM52_HIDDEN,
        0u,
        multiprocessors,
        stream);
}

static int32_t Glm52LayerMoe(
    const Glm52LayerBuffers *buffers,
    uint32_t rows,
    uint32_t packed_rows,
    uint32_t multiprocessors,
    cudaStream_t stream)
{
    LmGemmArguments gemm;
    int32_t status;

    if (buffers == 0 || rows == 0u ||
        packed_rows != rows * GLM52_TOP_K ||
        buffers->attention_out_bf16 == 0 || buffers->residual_bf16 == 0 ||
        buffers->mlp_norm_weight == 0 || buffers->normed_bf16 == 0 ||
        buffers->router_weight == 0 || buffers->router_logits == 0 ||
        buffers->route_expert == 0 || buffers->route_weight == 0 ||
        buffers->route_source_token == 0 || buffers->route_packed_row == 0 ||
        buffers->group_row_offset == 0 ||
        buffers->group_tile_prefix_w1 == 0 ||
        buffers->group_tile_prefix_w2 == 0 ||
        buffers->expert_w1_weight == 0 || buffers->expert_w1_scale == 0 ||
        buffers->expert_w2_weight == 0 || buffers->expert_w2_scale == 0 ||
        buffers->expert_out_bf16 == 0 || buffers->gate_up_bf16 == 0 ||
        buffers->intermediate_bf16 == 0 || buffers->hidden_bf16 == 0)
    {
        return LM_LAUNCH_ERR_SHAPE;
    }

    LM_LAUNCH(
        (LmFusedResidualRmsNormKernel<GLM52_LAYER_THREADS, uint16_t>),
        rows,
        GLM52_LAYER_THREADS,
        (GLM52_HIDDEN + 8u) * sizeof(float),
        stream,
        buffers->attention_out_bf16,
        buffers->residual_bf16,
        (const uint16_t *)buffers->mlp_norm_weight,
        buffers->residual_bf16,
        buffers->normed_bf16,
        GLM52_HIDDEN,
        GLM52_HIDDEN,
        GLM52_RMS_EPSILON);

    memset(&gemm, 0, sizeof(gemm));
    gemm.scale_a = LmScaleTensorNone();
    gemm.scale_b = LmScaleTensorNone();
    gemm.group_row_offset = buffers->dense_row_offset;
    gemm.group_tile_prefix = buffers->dense_tile_prefix;
    gemm.output_f32 = buffers->router_logits;
    status = LmGemmLaunch<
        LmBf16Format,
        GLM52_LAYER_TILE_N,
        LmBf16Format::kTileK,
        GLM52_LAYER_STAGES,
        GLM52_LAYER_WARPS>(
            &gemm,
            buffers->normed_bf16,
            buffers->router_weight,
            rows,
            rows,
            1u,
            1u,
            GLM52_HIDDEN,
            GLM52_EXPERTS,
            multiprocessors,
            false,
            stream);
    if (status != LM_LAUNCH_OK)
    {
        return status;
    }

    LM_LAUNCH(
        (LmTopkSmallKernel<
            GLM52_LAYER_THREADS,
            GLM52_TOP_K,
            true,
            1u,
            1u,
            LM_TOPK_SCORE_SIGMOID>),
        rows,
        GLM52_LAYER_THREADS,
        2u * LM_TOPK_SMALL_LIMIT * sizeof(uint32_t),
        stream,
        buffers->router_logits,
        GLM52_EXPERTS,
        buffers->route_expert,
        buffers->route_weight,
        0,
        0,
        GLM52_ROUTED_SCALE);
    status = LmRouteBuild<GLM52_LAYER_THREADS, GLM52_EXPERTS>(
        buffers->route_expert,
        rows,
        packed_rows,
        GLM52_TOP_K,
        buffers->group_row_offset,
        buffers->route_packed_row,
        buffers->route_source_token,
        GLM52_GATE_UP_DIM,
        GLM52_HIDDEN,
        GLM52_LAYER_TILE_N,
        buffers->group_tile_prefix_w1,
        buffers->group_tile_prefix_w2,
        stream);
    if (status != LM_LAUNCH_OK)
    {
        return status;
    }

    LM_LAUNCH(
        (LmGatherRowsKernel<GLM52_LAYER_THREADS>),
        dim3(
            (GLM52_HIDDEN + GLM52_LAYER_THREADS - 1u) /
                GLM52_LAYER_THREADS,
            packed_rows),
        GLM52_LAYER_THREADS,
        0,
        stream,
        buffers->normed_bf16,
        buffers->route_source_token,
        buffers->expert_out_bf16,
        packed_rows,
        GLM52_HIDDEN);

    memset(&gemm, 0, sizeof(gemm));
    gemm.scale_a = LmScaleTensorNone();
    gemm.scale_b = LmScaleTensorBlockF32(
        buffers->expert_w1_scale,
        GLM52_EXPERTS,
        GLM52_GATE_UP_DIM,
        GLM52_HIDDEN,
        GLM52_FP8_SCALE_BLOCK,
        GLM52_FP8_SCALE_BLOCK);
    gemm.prefix_built = 1u;
    gemm.group_row_offset = buffers->group_row_offset;
    gemm.group_tile_prefix = buffers->group_tile_prefix_w1;
    gemm.output_bf16 = buffers->gate_up_bf16;
    status = LmGemmWeightOnlyLaunch<
        LmFp8,
        GLM52_LAYER_TILE_N,
        GLM52_LAYER_STAGES,
        GLM52_LAYER_WARPS>(
            &gemm,
            buffers->expert_out_bf16,
            buffers->expert_w1_weight,
            packed_rows,
            rows,
            GLM52_TOP_K,
            GLM52_EXPERTS,
            GLM52_HIDDEN,
            GLM52_GATE_UP_DIM,
            multiprocessors,
            true,
            stream);
    if (status != LM_LAUNCH_OK)
    {
        return status;
    }

    LM_LAUNCH(
        (LmSiluMulKernel<GLM52_LAYER_THREADS>),
        packed_rows,
        GLM52_LAYER_THREADS,
        0,
        stream,
        buffers->gate_up_bf16,
        buffers->intermediate_bf16,
        GLM52_EXPERT_INTERMEDIATE,
        false);

    memset(&gemm, 0, sizeof(gemm));
    gemm.scale_a = LmScaleTensorNone();
    gemm.scale_b = LmScaleTensorBlockF32(
        buffers->expert_w2_scale,
        GLM52_EXPERTS,
        GLM52_HIDDEN,
        GLM52_EXPERT_INTERMEDIATE,
        GLM52_FP8_SCALE_BLOCK,
        GLM52_FP8_SCALE_BLOCK);
    gemm.prefix_built = 1u;
    gemm.group_row_offset = buffers->group_row_offset;
    gemm.group_tile_prefix = buffers->group_tile_prefix_w2;
    gemm.output_bf16 = buffers->expert_out_bf16;
    status = LmGemmWeightOnlyLaunch<
        LmFp8,
        GLM52_LAYER_TILE_N,
        GLM52_LAYER_STAGES,
        GLM52_LAYER_WARPS>(
            &gemm,
            buffers->intermediate_bf16,
            buffers->expert_w2_weight,
            packed_rows,
            rows,
            GLM52_TOP_K,
            GLM52_EXPERTS,
            GLM52_EXPERT_INTERMEDIATE,
            GLM52_HIDDEN,
            multiprocessors,
            true,
            stream);
    if (status != LM_LAUNCH_OK)
    {
        return status;
    }

    LM_LAUNCH(
        (LmMoeFinalizeKernel<GLM52_LAYER_THREADS>),
        dim3(
            (GLM52_HIDDEN + GLM52_LAYER_THREADS - 1u) /
                GLM52_LAYER_THREADS,
            rows),
        GLM52_LAYER_THREADS,
        0,
        stream,
        buffers->expert_out_bf16,
        buffers->route_packed_row,
        buffers->route_weight,
        buffers->hidden_bf16,
        rows,
        GLM52_TOP_K,
        GLM52_HIDDEN);
    return cudaPeekAtLastError() == cudaSuccess
        ? LM_LAUNCH_OK
        : LM_LAUNCH_ERR_LAUNCH;
}

static int32_t Glm52Head(
    const Glm52LayerBuffers *buffers,
    const void *head_norm_weight,
    const void *head_weight,
    const uint32_t *token_ids,
    uint32_t vocabulary,
    uint32_t rows,
    cudaStream_t stream)
{
    uint32_t tiles;

    if (buffers == 0 || head_norm_weight == 0 || head_weight == 0 ||
        rows == 0u || vocabulary == 0u || buffers->hidden_bf16 == 0 ||
        buffers->normed_bf16 == 0 || buffers->head_candidate_score == 0 ||
        buffers->head_candidate_token == 0 || buffers->output_token == 0 ||
        buffers->output_score == 0)
    {
        return LM_LAUNCH_ERR_SHAPE;
    }

    tiles = (vocabulary + GLM52_HEAD_TILE - 1u) / GLM52_HEAD_TILE;
    LM_LAUNCH(
        (LmFusedResidualRmsNormKernel<GLM52_LAYER_THREADS, uint16_t>),
        rows,
        GLM52_LAYER_THREADS,
        (GLM52_HIDDEN + 8u) * sizeof(float),
        stream,
        buffers->hidden_bf16,
        0,
        (const uint16_t *)head_norm_weight,
        0,
        buffers->normed_bf16,
        GLM52_HIDDEN,
        GLM52_HIDDEN,
        GLM52_RMS_EPSILON);
    LM_LAUNCH(
        (LmHeadCandidateKernel<GLM52_LAYER_THREADS, GLM52_HEAD_TILE>),
        dim3(tiles, rows),
        GLM52_LAYER_THREADS,
        0,
        stream,
        buffers->normed_bf16,
        (const uint16_t *)head_weight,
        token_ids,
        buffers->head_candidate_score,
        buffers->head_candidate_token,
        rows,
        GLM52_HIDDEN,
        vocabulary);
    LM_LAUNCH(
        (LmHeadCommitKernel<GLM52_LAYER_THREADS>),
        rows,
        GLM52_LAYER_THREADS,
        0,
        stream,
        buffers->head_candidate_score,
        buffers->head_candidate_token,
        tiles,
        buffers->output_token,
        buffers->output_score,
        rows);
    return cudaPeekAtLastError() == cudaSuccess
        ? LM_LAUNCH_OK
        : LM_LAUNCH_ERR_LAUNCH;
}
