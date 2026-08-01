// DeepSeek V4 Pro compile surface.
//
// This translation unit binds the checkpoint precision recipe and exact Pro
// dimensions to the shared CUDA kernel library. The shipping stage runner is
// intentionally not exported here until HCA, CSA, hyper-connections, hash MoE,
// and the Pro stage-pack contract are connected end to end.

#include "runtime/gemm.cuh"
#include "inference/kernels/formats/fp8.cuh"
#include "inference/kernels/formats/mxfp4.cuh"
#include "inference/llms/deepseek_v4_pro/config.h"

#define DSV4_PRO_TILE_N 128u
#define DSV4_PRO_TILE_K 128u
#define DSV4_PRO_STAGES 2u
#define DSV4_PRO_WARPS 8u

static_assert(
    (DSV4_PRO_HIDDEN % DSV4_PRO_TILE_K) == 0u,
    "DeepSeek V4 Pro hidden dimension must contain complete GEMM K tiles");
static_assert(
    (DSV4_PRO_EXPERT_INTERMEDIATE % DSV4_PRO_TILE_K) == 0u,
    "DeepSeek V4 Pro expert dimension must contain complete GEMM K tiles");
static_assert(
    SPARK_DSV4_PRO_FP8_WEIGHT_BLOCK_COLUMNS == DSV4_PRO_TILE_K,
    "DeepSeek V4 Pro FP8 block width must match the retained GEMM tile");
static_assert(
    SPARK_DSV4_PRO_EXPERT_WEIGHTS_ARE_CHECKPOINT_FP4 == 1u,
    "DeepSeek V4 Pro routed experts must retain checkpoint FP4 weights");
static_assert(
    SPARK_DSV4_PRO_NON_EXPERT_WEIGHTS_ARE_FP8_E4M3 == 1u,
    "DeepSeek V4 Pro non-expert projections must retain FP8 E4M3 weights");

template __global__ void LmGemmKernel<
    LmFp8,
    LmFp8,
    16u,
    DSV4_PRO_TILE_N,
    DSV4_PRO_TILE_K,
    DSV4_PRO_STAGES,
    DSV4_PRO_WARPS>(
        __grid_constant__ const LmGemmArguments,
        __grid_constant__ const CUtensorMap,
        __grid_constant__ const CUtensorMap,
        LmTileGeometry,
        LmTileGeometry,
        bool);

template __global__ void LmGemmKernel<
    LmFp8,
    LmFp8,
    32u,
    DSV4_PRO_TILE_N,
    DSV4_PRO_TILE_K,
    DSV4_PRO_STAGES,
    DSV4_PRO_WARPS>(
        __grid_constant__ const LmGemmArguments,
        __grid_constant__ const CUtensorMap,
        __grid_constant__ const CUtensorMap,
        LmTileGeometry,
        LmTileGeometry,
        bool);

template __global__ void LmGemmKernel<
    LmFp8,
    LmFp8,
    64u,
    DSV4_PRO_TILE_N,
    DSV4_PRO_TILE_K,
    DSV4_PRO_STAGES,
    DSV4_PRO_WARPS>(
        __grid_constant__ const LmGemmArguments,
        __grid_constant__ const CUtensorMap,
        __grid_constant__ const CUtensorMap,
        LmTileGeometry,
        LmTileGeometry,
        bool);

template __global__ void LmGemmKernel<
    LmFp8,
    LmMxfp4,
    16u,
    DSV4_PRO_TILE_N,
    DSV4_PRO_TILE_K,
    DSV4_PRO_STAGES,
    DSV4_PRO_WARPS>(
        __grid_constant__ const LmGemmArguments,
        __grid_constant__ const CUtensorMap,
        __grid_constant__ const CUtensorMap,
        LmTileGeometry,
        LmTileGeometry,
        bool);

template __global__ void LmGemmKernel<
    LmFp8,
    LmMxfp4,
    32u,
    DSV4_PRO_TILE_N,
    DSV4_PRO_TILE_K,
    DSV4_PRO_STAGES,
    DSV4_PRO_WARPS>(
        __grid_constant__ const LmGemmArguments,
        __grid_constant__ const CUtensorMap,
        __grid_constant__ const CUtensorMap,
        LmTileGeometry,
        LmTileGeometry,
        bool);

template __global__ void LmGemmKernel<
    LmFp8,
    LmMxfp4,
    64u,
    DSV4_PRO_TILE_N,
    DSV4_PRO_TILE_K,
    DSV4_PRO_STAGES,
    DSV4_PRO_WARPS>(
        __grid_constant__ const LmGemmArguments,
        __grid_constant__ const CUtensorMap,
        __grid_constant__ const CUtensorMap,
        LmTileGeometry,
        LmTileGeometry,
        bool);
