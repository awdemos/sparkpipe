#include "sparkpipe/spark_glm52_resident_decode_stage_required_cuda.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SPARK_GLM52_FP8_GATE_THREADS 256u

typedef struct SparkGlm52Fp8GateState
{
    uint32_t active_sequence_count;
    uint32_t logical_output_dimension;
    uint32_t storage_output_dimension;
    uint32_t input_dimension;
    uint32_t iteration_count;
    uint32_t graph_mode;
    uint16_t *input_bf16;
    uint16_t *output_bf16;
    uint16_t *storage_output_bf16;
    uint16_t *host_output_bf16;
    uint8_t *weight_fp8;
    float *weight_scale;
    void *linear_workspace;
    void *backend_workspace;
    uint64_t input_elements;
    uint64_t logical_output_elements;
    uint64_t storage_output_elements;
    uint64_t weight_elements;
    uint64_t weight_scale_elements;
    uint64_t linear_workspace_bytes;
    uint64_t backend_workspace_bytes;
    cudaStream_t stream;
    cudaEvent_t start_event;
    cudaEvent_t stop_event;
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    SparkGlm52ResidentDecodeStageQuantizedLinearView view;
    SparkGlm52ResidentDecodeStageLinearPlan plan;
    SparkGlm52Sm121RequiredDecodeStageBuiltinFp8ScaledGemmState backend_state;
    SparkGlm52Sm121RequiredDecodeStageFp8ScaledGemmBackend backend;
} SparkGlm52Fp8GateState;

static __global__ void SparkGlm52Fp8GateFillInput(
    uint16_t *input_bf16,
    uint32_t row_count,
    uint32_t column_count)
{
    uint64_t index;
    uint64_t element_count;
    float value;
    index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    element_count = (uint64_t)row_count * column_count;
    if (index >= element_count)
        return;
    value = (float)((int32_t)(index % 17u) - 8) * 0.0625f;
    input_bf16[index] = __bfloat16_as_ushort(__float2bfloat16(value));
}

static __global__ void SparkGlm52Fp8GateFillIdentityWeight(
    uint8_t *weight_fp8,
    uint32_t input_dimension,
    uint32_t logical_output_dimension,
    uint32_t storage_output_dimension)
{
    uint64_t index;
    uint64_t element_count;
    uint32_t input_index;
    uint32_t output_index;
    float value;
    index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    element_count = (uint64_t)input_dimension * storage_output_dimension;
    if (index >= element_count)
        return;
    input_index = (uint32_t)(index % input_dimension);
    output_index = (uint32_t)(index / input_dimension);
    value = output_index < logical_output_dimension && input_index == output_index
        ? 1.0f
        : 0.0f;
    weight_fp8[index] = __nv_cvt_float_to_fp8(
        value,
        __NV_SATFINITE,
        __NV_E4M3);
}

static __global__ void SparkGlm52Fp8GateFillOnes(
    float *values,
    uint64_t value_count)
{
    uint64_t index;
    index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < value_count)
        values[index] = 1.0f;
}

static uint64_t SparkGlm52Fp8GateDivideRoundUp(
    uint64_t value,
    uint64_t divisor)
{
    return divisor == 0u ? 0u : (value + divisor - 1u) / divisor;
}

static uint32_t SparkGlm52Fp8GateAlignOutputDimension(
    uint32_t output_dimension)
{
    uint32_t alignment;
    alignment =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALED_GEMM_OUTPUT_ALIGNMENT;
    if (output_dimension == 0u || output_dimension > UINT32_MAX - alignment + 1u)
        return 0u;
    return ((output_dimension + alignment - 1u) / alignment) * alignment;
}

static uint32_t SparkGlm52Fp8GateParseU32(
    const char *text,
    uint32_t default_value)
{
    char *end;
    unsigned long value;
    if (text == 0)
        return default_value;
    end = 0;
    value = strtoul(text,&end,10);
    if (end == text || *end != '\0' || value == 0u || value > UINT32_MAX)
        return 0u;
    return (uint32_t)value;
}

static void SparkGlm52Fp8GateDestroy(SparkGlm52Fp8GateState *state)
{
    if (state->graph_exec != 0)
        cudaGraphExecDestroy(state->graph_exec);
    if (state->graph != 0)
        cudaGraphDestroy(state->graph);
    if (state->backend_workspace != 0)
        cudaFree(state->backend_workspace);
    if (state->linear_workspace != 0)
        cudaFree(state->linear_workspace);
    if (state->weight_scale != 0)
        cudaFree(state->weight_scale);
    if (state->weight_fp8 != 0)
        cudaFree(state->weight_fp8);
    if (state->storage_output_bf16 != 0)
        cudaFree(state->storage_output_bf16);
    if (state->output_bf16 != 0)
        cudaFree(state->output_bf16);
    if (state->input_bf16 != 0)
        cudaFree(state->input_bf16);
    if (state->stop_event != 0)
        cudaEventDestroy(state->stop_event);
    if (state->start_event != 0)
        cudaEventDestroy(state->start_event);
    if (state->stream != 0)
        cudaStreamDestroy(state->stream);
    free(state->host_output_bf16);
    memset(state,0,sizeof(*state));
}

static int32_t SparkGlm52Fp8GateCalculateStorage(
    SparkGlm52Fp8GateState *state)
{
    state->storage_output_dimension = SparkGlm52Fp8GateAlignOutputDimension(
        state->logical_output_dimension);
    if (state->storage_output_dimension == 0u)
        return -1;
    state->input_elements =
        (uint64_t)state->active_sequence_count * state->input_dimension;
    state->logical_output_elements =
        (uint64_t)state->active_sequence_count * state->logical_output_dimension;
    state->storage_output_elements =
        (uint64_t)state->active_sequence_count * state->storage_output_dimension;
    state->weight_elements =
        (uint64_t)state->input_dimension * state->storage_output_dimension;
    state->weight_scale_elements = SparkGlm52Fp8GateDivideRoundUp(
        state->input_dimension,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_BLOCK) *
        SparkGlm52Fp8GateDivideRoundUp(
            state->storage_output_dimension,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_BLOCK);
    state->linear_workspace_bytes =
        SparkGlm52Sm121RequiredDecodeStageCalculateFp8E4m3ActivationLinearBackendWorkspaceBytes(
            state->active_sequence_count,
            state->input_dimension,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_BLOCK,
            0u);
    state->backend_workspace_bytes =
        SparkGlm52Sm121RequiredDecodeStageCalculateBuiltinFp8ScaledGemmWorkspaceBytes();
    return state->linear_workspace_bytes == 0u ||
        state->backend_workspace_bytes == 0u ? -2 : 0;
}

static int32_t SparkGlm52Fp8GateAllocate(SparkGlm52Fp8GateState *state)
{
    state->host_output_bf16 = (uint16_t *)malloc(
        (size_t)(state->logical_output_elements * sizeof(uint16_t)));
    if (state->host_output_bf16 == 0 ||
        cudaStreamCreate(&state->stream) != cudaSuccess ||
        cudaEventCreate(&state->start_event) != cudaSuccess ||
        cudaEventCreate(&state->stop_event) != cudaSuccess ||
        cudaMalloc((void **)&state->input_bf16,
            (size_t)(state->input_elements * sizeof(uint16_t))) != cudaSuccess ||
        cudaMalloc((void **)&state->output_bf16,
            (size_t)(state->logical_output_elements * sizeof(uint16_t))) != cudaSuccess ||
        cudaMalloc((void **)&state->weight_fp8,
            (size_t)state->weight_elements) != cudaSuccess ||
        cudaMalloc((void **)&state->weight_scale,
            (size_t)(state->weight_scale_elements * sizeof(float))) != cudaSuccess ||
        cudaMalloc(&state->linear_workspace,
            (size_t)state->linear_workspace_bytes) != cudaSuccess ||
        cudaMalloc(&state->backend_workspace,
            (size_t)state->backend_workspace_bytes) != cudaSuccess)
        return -1;
    if (state->storage_output_dimension != state->logical_output_dimension &&
        cudaMalloc((void **)&state->storage_output_bf16,
            (size_t)(state->storage_output_elements * sizeof(uint16_t))) != cudaSuccess)
        return -2;
    return 0;
}

static int32_t SparkGlm52Fp8GateFill(SparkGlm52Fp8GateState *state)
{
    uint32_t block_count;
    block_count = (uint32_t)SparkGlm52Fp8GateDivideRoundUp(
        state->input_elements,
        SPARK_GLM52_FP8_GATE_THREADS);
    SparkGlm52Fp8GateFillInput<<<
        block_count,SPARK_GLM52_FP8_GATE_THREADS,0,state->stream>>>(
        state->input_bf16,
        state->active_sequence_count,
        state->input_dimension);
    block_count = (uint32_t)SparkGlm52Fp8GateDivideRoundUp(
        state->weight_elements,
        SPARK_GLM52_FP8_GATE_THREADS);
    SparkGlm52Fp8GateFillIdentityWeight<<<
        block_count,SPARK_GLM52_FP8_GATE_THREADS,0,state->stream>>>(
        state->weight_fp8,
        state->input_dimension,
        state->logical_output_dimension,
        state->storage_output_dimension);
    block_count = (uint32_t)SparkGlm52Fp8GateDivideRoundUp(
        state->weight_scale_elements,
        SPARK_GLM52_FP8_GATE_THREADS);
    SparkGlm52Fp8GateFillOnes<<<
        block_count,SPARK_GLM52_FP8_GATE_THREADS,0,state->stream>>>(
        state->weight_scale,
        state->weight_scale_elements);
    if (cudaMemsetAsync(
            state->output_bf16,
            0,
            (size_t)(state->logical_output_elements * sizeof(uint16_t)),
            state->stream) != cudaSuccess ||
        cudaStreamSynchronize(state->stream) != cudaSuccess)
        return -1;
    return 0;
}

static int32_t SparkGlm52Fp8GateBuildPlan(SparkGlm52Fp8GateState *state)
{
    SparkStatus status;
    memset(&state->view,0,sizeof(state->view));
    state->view.abi_version =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_QUANTIZED_LINEAR_VIEW_ABI_VERSION;
    state->view.weight_format =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_FP8_E4M3;
    state->view.input_dimension = state->input_dimension;
    state->view.output_dimension = state->logical_output_dimension;
    state->view.storage_output_dimension = state->storage_output_dimension;
    state->view.scale_block_size =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_BLOCK;
    state->view.weight_payload = state->weight_fp8;
    state->view.weight_scale = state->weight_scale;
    state->view.weight_payload_bytes = state->weight_elements;
    state->view.weight_scale_bytes =
        state->weight_scale_elements * (uint64_t)sizeof(float);
    state->view.output_workspace = state->storage_output_bf16;
    state->view.output_workspace_bytes = state->storage_output_bf16 == 0
        ? 0u
        : state->storage_output_elements * (uint64_t)sizeof(uint16_t);
    memset(&state->plan,0,sizeof(state->plan));
    state->plan.abi_version =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_ABI_VERSION;
    state->plan.plan_kind =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_TENSOR_CORE_FP8_E4M3_ROW_MAJOR;
    state->plan.input_dimension = state->input_dimension;
    state->plan.output_dimension = state->logical_output_dimension;
    state->plan.maximum_active_sequence_count = state->active_sequence_count;
    state->plan.workspace = state->linear_workspace;
    state->plan.workspace_bytes = state->linear_workspace_bytes;
    state->plan.custom_state = &state->view;
    status = SparkGlm52Sm121RequiredDecodeStageInitializeBuiltinFp8ScaledGemmBackend(
        &state->backend_state,
        state->backend_workspace,
        state->backend_workspace_bytes,
        &state->backend);
    if (status != SPARK_STATUS_OK)
        return -1;
    status = SparkGlm52Sm121RequiredDecodeStageBindFp8E4m3LinearScaledGemmBackend(
        &state->plan,
        &state->backend);
    return status == SPARK_STATUS_OK ? 0 : -2;
}

static SparkStatus SparkGlm52Fp8GateLaunch(SparkGlm52Fp8GateState *state)
{
    return SparkGlm52Sm121RequiredDecodeStageLaunchBlackwellQuantizedTensorCoreLinearPlan(
        &state->plan,
        state->input_bf16,
        state->weight_fp8,
        state->output_bf16,
        state->active_sequence_count,
        (void *)state->stream);
}

static int32_t SparkGlm52Fp8GateCapture(SparkGlm52Fp8GateState *state)
{
    SparkStatus status;
    if (state->graph_mode == 0u)
        return 0;
    if (cudaStreamBeginCapture(
            state->stream,
            cudaStreamCaptureModeThreadLocal) != cudaSuccess)
        return -1;
    status = SparkGlm52Fp8GateLaunch(state);
    if (status != SPARK_STATUS_OK ||
        cudaStreamEndCapture(state->stream,&state->graph) != cudaSuccess ||
        cudaGraphInstantiate(&state->graph_exec,state->graph,0,0,0) != cudaSuccess)
        return -2;
    return 0;
}

static int32_t SparkGlm52Fp8GateLaunchOne(SparkGlm52Fp8GateState *state)
{
    if (state->graph_mode != 0u)
        return cudaGraphLaunch(state->graph_exec,state->stream) == cudaSuccess
            ? 0
            : -1;
    return SparkGlm52Fp8GateLaunch(state) == SPARK_STATUS_OK ? 0 : -2;
}

static int32_t SparkGlm52Fp8GateCheckOutput(
    const SparkGlm52Fp8GateState *state,
    float *maximum_error_out)
{
    uint64_t index;
    uint32_t input_index;
    uint32_t output_index;
    float actual;
    float expected;
    float error;
    float maximum_error;
    maximum_error = 0.0f;
    for (index=0u; index<state->logical_output_elements; ++index)
    {
        output_index = (uint32_t)(index % state->logical_output_dimension);
        input_index = (uint32_t)(((index / state->logical_output_dimension) *
            state->input_dimension) + output_index);
        expected = output_index < state->input_dimension
            ? (float)((int32_t)(input_index % 17u) - 8) * 0.0625f
            : 0.0f;
        actual = __bfloat162float(__ushort_as_bfloat16(
            state->host_output_bf16[index]));
        error = fabsf(actual - expected);
        if (error > maximum_error)
            maximum_error = error;
    }
    *maximum_error_out = maximum_error;
    return maximum_error <= 0.0625f ? 0 : -1;
}

static int32_t SparkGlm52Fp8GateMeasure(
    SparkGlm52Fp8GateState *state,
    float *average_ms_out,
    float *maximum_error_out)
{
    uint32_t iteration;
    float elapsed_ms;
    for (iteration=0u; iteration<2u; ++iteration)
        if (SparkGlm52Fp8GateLaunchOne(state) != 0)
            return -1;
    if (cudaEventRecord(state->start_event,state->stream) != cudaSuccess)
        return -2;
    for (iteration=0u; iteration<state->iteration_count; ++iteration)
        if (SparkGlm52Fp8GateLaunchOne(state) != 0)
            return -3;
    if (cudaEventRecord(state->stop_event,state->stream) != cudaSuccess ||
        cudaEventSynchronize(state->stop_event) != cudaSuccess ||
        cudaEventElapsedTime(&elapsed_ms,state->start_event,state->stop_event) != cudaSuccess ||
        cudaMemcpy(state->host_output_bf16,state->output_bf16,
            (size_t)(state->logical_output_elements * sizeof(uint16_t)),
            cudaMemcpyDeviceToHost) != cudaSuccess)
        return -4;
    *average_ms_out = elapsed_ms / (float)state->iteration_count;
    return SparkGlm52Fp8GateCheckOutput(state,maximum_error_out);
}

static int32_t SparkGlm52Fp8GateRun(SparkGlm52Fp8GateState *state)
{
    float average_ms;
    float maximum_error;
    int32_t result;
    result = SparkGlm52Fp8GateCalculateStorage(state);
    if (result == 0)
        result = SparkGlm52Fp8GateAllocate(state);
    if (result == 0)
        result = SparkGlm52Fp8GateFill(state);
    if (result == 0)
        result = SparkGlm52Fp8GateBuildPlan(state);
    if (result == 0)
        result = SparkGlm52Fp8GateCapture(state);
    if (result == 0)
        result = SparkGlm52Fp8GateMeasure(state,&average_ms,&maximum_error);
    if (result == 0)
        printf(
            "fp8_scaled_gemm_gate status=pass active=%u logical_output=%u storage_output=%u input=%u iterations=%u graph=%u average_ms=%.6f maximum_error=%.6f\n",
            state->active_sequence_count,
            state->logical_output_dimension,
            state->storage_output_dimension,
            state->input_dimension,
            state->iteration_count,
            state->graph_mode,
            average_ms,
            maximum_error);
    else
        fprintf(stderr,"fp8_scaled_gemm_gate status=fail step=%d cuda=%s\n",
            result,cudaGetErrorString(cudaPeekAtLastError()));
    SparkGlm52Fp8GateDestroy(state);
    return result;
}

int main(int argc,char **argv)
{
    SparkGlm52Fp8GateState state;
    memset(&state,0,sizeof(state));
    state.active_sequence_count = SparkGlm52Fp8GateParseU32(
        argc > 1 ? argv[1] : 0,1u);
    state.logical_output_dimension = SparkGlm52Fp8GateParseU32(
        argc > 2 ? argv[2] : 0,16384u);
    state.input_dimension = SparkGlm52Fp8GateParseU32(
        argc > 3 ? argv[3] : 0,2048u);
    state.iteration_count = SparkGlm52Fp8GateParseU32(
        argc > 4 ? argv[4] : 0,10u);
    state.graph_mode = argc > 5 ? SparkGlm52Fp8GateParseU32(argv[5],0u) : 0u;
    if (state.active_sequence_count == 0u ||
        state.logical_output_dimension == 0u ||
        state.input_dimension == 0u ||
        state.iteration_count == 0u || state.graph_mode > 1u)
        return 2;
    return SparkGlm52Fp8GateRun(&state) == 0 ? 0 : 1;
}
