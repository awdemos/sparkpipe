#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "sparkpipe/spark_glm52_dspark_draft_backend.h"
#include "sparkpipe/spark_json.h"

#define SPARK_GLM52_DSPARK_BACKEND_REDUCE_THREADS 256u
#define SPARK_GLM52_DSPARK_BACKEND_WINDOW_TOKENS SPARK_GLM52_DSPARK_BLOCK_SIZE
#define SPARK_GLM52_DSPARK_BACKEND_ARGMAX_WORD_COUNT 2u

typedef enum SparkGlm52DsparkWeightRole
{
    SPARK_GLM52_DSPARK_WEIGHT_EMBED_TOKENS = 0,
    SPARK_GLM52_DSPARK_WEIGHT_FUSION_FC,
    SPARK_GLM52_DSPARK_WEIGHT_FINAL_NORM,
    SPARK_GLM52_DSPARK_WEIGHT_LM_HEAD,
    SPARK_GLM52_DSPARK_WEIGHT_MARKOV_HIDDEN,
    SPARK_GLM52_DSPARK_WEIGHT_MARKOV_VOCAB,
    SPARK_GLM52_DSPARK_WEIGHT_CONFIDENCE,
    SPARK_GLM52_DSPARK_WEIGHT_CONFIDENCE_BIAS,
    SPARK_GLM52_DSPARK_WEIGHT_LAYER_BASE
} SparkGlm52DsparkWeightRole;

typedef enum SparkGlm52DsparkLayerWeight
{
    SPARK_GLM52_DSPARK_LAYER_WEIGHT_INPUT_NORM = 0,
    SPARK_GLM52_DSPARK_LAYER_WEIGHT_Q,
    SPARK_GLM52_DSPARK_LAYER_WEIGHT_K,
    SPARK_GLM52_DSPARK_LAYER_WEIGHT_V,
    SPARK_GLM52_DSPARK_LAYER_WEIGHT_Q_NORM,
    SPARK_GLM52_DSPARK_LAYER_WEIGHT_K_NORM,
    SPARK_GLM52_DSPARK_LAYER_WEIGHT_O,
    SPARK_GLM52_DSPARK_LAYER_WEIGHT_POST_NORM,
    SPARK_GLM52_DSPARK_LAYER_WEIGHT_GATE,
    SPARK_GLM52_DSPARK_LAYER_WEIGHT_UP,
    SPARK_GLM52_DSPARK_LAYER_WEIGHT_DOWN,
    SPARK_GLM52_DSPARK_LAYER_WEIGHT_COUNT
} SparkGlm52DsparkLayerWeight;

typedef struct SparkGlm52DsparkTensorSpec
{
    const char *name;
    uint32_t role;
    uint32_t rows;
    uint32_t columns;
    uint32_t optional;
} SparkGlm52DsparkTensorSpec;

static float SparkGlm52DsparkAssumedQwen3RopeTheta(void)
{
    return 1000000.0f;
}

static float SparkGlm52DsparkAssumedQwen3RmsNormEpsilon(void)
{
    return 1e-6f;
}

static uint32_t SparkGlm52DsparkLayerWeightIndex(
    uint32_t layer_index,
    uint32_t layer_weight)
{
    return SPARK_GLM52_DSPARK_WEIGHT_LAYER_BASE +
        (layer_index * (uint32_t)SPARK_GLM52_DSPARK_LAYER_WEIGHT_COUNT) +
        layer_weight;
}

static SparkStatus SparkGlm52DsparkCudaStatus(cudaError_t status)
{
    return status == cudaSuccess ? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
}

static __global__ void SparkGlm52DsparkComposeStepInputKernel(
    const uint16_t *__restrict__ embed_weight_bf16,
    const uint16_t *__restrict__ residual_bf16,
    uint16_t *__restrict__ hidden_bf16,
    uint32_t token_id,
    uint32_t hidden_dimension)
{
    uint32_t element_index;
    float embedding_value;
    float residual_value;

    element_index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (element_index >= hidden_dimension)
    {
        return;
    }
    embedding_value = __bfloat162float(
        ((const __nv_bfloat16 *)embed_weight_bf16)[
            ((uint64_t)token_id * (uint64_t)hidden_dimension) + element_index]);
    residual_value = __bfloat162float(
        ((const __nv_bfloat16 *)residual_bf16)[element_index]);
    ((__nv_bfloat16 *)hidden_bf16)[element_index] =
        __float2bfloat16(embedding_value + residual_value);
}

static __global__ void SparkGlm52DsparkRmsNormKernel(
    const uint16_t *__restrict__ input_bf16,
    const uint16_t *__restrict__ norm_weight_bf16,
    uint16_t *__restrict__ output_bf16,
    uint32_t dimension,
    float epsilon)
{
    __shared__ float shared_partials[SPARK_GLM52_DSPARK_BACKEND_REDUCE_THREADS];
    __shared__ float shared_inverse_norm;
    uint32_t element_index;
    uint32_t stride_index;
    float sum_squares;
    float value;

    sum_squares = 0.0f;
    for (element_index = threadIdx.x;
         element_index < dimension;
         element_index += blockDim.x)
    {
        value = __bfloat162float(
            ((const __nv_bfloat16 *)input_bf16)[element_index]);
        sum_squares += value * value;
    }
    shared_partials[threadIdx.x] = sum_squares;
    __syncthreads();
    for (stride_index = blockDim.x / 2u; stride_index > 0u; stride_index /= 2u)
    {
        if (threadIdx.x < stride_index)
            shared_partials[threadIdx.x] += shared_partials[threadIdx.x + stride_index];
        __syncthreads();
    }
    if (threadIdx.x == 0u)
        shared_inverse_norm = rsqrtf((shared_partials[0] / (float)dimension) + epsilon);
    __syncthreads();
    for (element_index = threadIdx.x;
         element_index < dimension;
         element_index += blockDim.x)
    {
        value = __bfloat162float(
            ((const __nv_bfloat16 *)input_bf16)[element_index]);
        ((__nv_bfloat16 *)output_bf16)[element_index] = __float2bfloat16(
            value * shared_inverse_norm *
            __bfloat162float(
                ((const __nv_bfloat16 *)norm_weight_bf16)[element_index]));
    }
}

static __global__ void SparkGlm52DsparkLinearKernel(
    const uint16_t *__restrict__ weight_bf16,
    const uint16_t *__restrict__ input_bf16,
    uint16_t *__restrict__ output_bf16,
    uint32_t row_count,
    uint32_t column_count,
    uint32_t accumulate_output)
{
    __shared__ float shared_partials[SPARK_GLM52_DSPARK_BACKEND_REDUCE_THREADS];
    uint32_t row_index;
    uint32_t column_index;
    uint32_t stride_index;
    uint64_t row_offset;
    float partial_sum;
    float output_value;

    row_index = blockIdx.x;
    if (row_index >= row_count)
    {
        return;
    }
    row_offset = (uint64_t)row_index * (uint64_t)column_count;
    partial_sum = 0.0f;
    for (column_index = threadIdx.x;
         column_index < column_count;
         column_index += blockDim.x)
    {
        partial_sum += __bfloat162float(
            ((const __nv_bfloat16 *)weight_bf16)[row_offset + column_index]) *
            __bfloat162float(((const __nv_bfloat16 *)input_bf16)[column_index]);
    }
    shared_partials[threadIdx.x] = partial_sum;
    __syncthreads();
    for (stride_index = blockDim.x / 2u; stride_index > 0u; stride_index /= 2u)
    {
        if (threadIdx.x < stride_index)
            shared_partials[threadIdx.x] += shared_partials[threadIdx.x + stride_index];
        __syncthreads();
    }
    if (threadIdx.x != 0u)
    {
        return;
    }
    output_value = shared_partials[0];
    if (accumulate_output != 0u)
        output_value += __bfloat162float(((const __nv_bfloat16 *)output_bf16)[row_index]);
    ((__nv_bfloat16 *)output_bf16)[row_index] = __float2bfloat16(output_value);
}

static __global__ void SparkGlm52DsparkHeadNormRopeAppendKernel(
    const uint16_t *__restrict__ input_bf16,
    const uint16_t *__restrict__ head_norm_weight_bf16,
    uint16_t *__restrict__ output_bf16,
    uint32_t head_dimension,
    uint32_t position,
    float rope_theta,
    float epsilon)
{
    __shared__ float shared_head_values[SPARK_GLM52_DSPARK_DRAFT_HEAD_DIMENSION];
    __shared__ float shared_sum_squares;
    uint32_t head_index;
    uint32_t pair_index;
    uint32_t rotate_index;
    uint64_t head_offset;
    float value;
    float paired_value;
    float inverse_norm;
    float frequency;
    float angle;

    head_index = blockIdx.x;
    head_offset = (uint64_t)head_index * (uint64_t)head_dimension;
    value = __bfloat162float(
        ((const __nv_bfloat16 *)input_bf16)[head_offset + threadIdx.x]);
    shared_head_values[threadIdx.x] = value * value;
    __syncthreads();
    if (threadIdx.x == 0u)
    {
        float sum;
        uint32_t element_index;

        sum = 0.0f;
        for (element_index = 0u; element_index < head_dimension; ++element_index)
        {
            sum += shared_head_values[element_index];
        }
        shared_sum_squares = sum;
    }
    __syncthreads();
    inverse_norm = rsqrtf((shared_sum_squares / (float)head_dimension) + epsilon);
    value = __bfloat162float(
        ((const __nv_bfloat16 *)input_bf16)[head_offset + threadIdx.x]) *
        inverse_norm *
        __bfloat162float(((const __nv_bfloat16 *)head_norm_weight_bf16)[threadIdx.x]);
    shared_head_values[threadIdx.x] = value;
    __syncthreads();
    pair_index = threadIdx.x % (head_dimension / 2u);
    rotate_index = threadIdx.x < (head_dimension / 2u)
        ? threadIdx.x + (head_dimension / 2u)
        : threadIdx.x - (head_dimension / 2u);
    paired_value = shared_head_values[rotate_index];
    frequency = powf(rope_theta, -2.0f * (float)pair_index / (float)head_dimension);
    angle = (float)position * frequency;
    value = threadIdx.x < (head_dimension / 2u)
        ? (shared_head_values[threadIdx.x] * cosf(angle)) - (paired_value * sinf(angle))
        : (shared_head_values[threadIdx.x] * cosf(angle)) + (paired_value * sinf(angle));
    ((__nv_bfloat16 *)output_bf16)[head_offset + threadIdx.x] = __float2bfloat16(value);
}

static __global__ void SparkGlm52DsparkCopyVectorKernel(
    const uint16_t *__restrict__ input_bf16,
    uint16_t *__restrict__ output_bf16,
    uint32_t element_count)
{
    uint32_t element_index;

    element_index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (element_index >= element_count)
    {
        return;
    }
    output_bf16[element_index] = input_bf16[element_index];
}

static __global__ void SparkGlm52DsparkWindowAttentionKernel(
    const uint16_t *__restrict__ query_bf16,
    const uint16_t *__restrict__ window_key_bf16,
    const uint16_t *__restrict__ window_value_bf16,
    uint16_t *__restrict__ output_bf16,
    uint32_t key_count,
    uint32_t head_dimension,
    uint32_t attention_dimension)
{
    __shared__ float shared_scores[SPARK_GLM52_DSPARK_BACKEND_WINDOW_TOKENS];
    __shared__ float shared_probabilities[SPARK_GLM52_DSPARK_BACKEND_WINDOW_TOKENS];
    uint32_t head_index;
    uint32_t key_index;
    uint32_t element_index;
    uint64_t head_offset;
    float score;
    float maximum_score;
    float denominator;
    float accumulated;

    head_index = blockIdx.x;
    head_offset = (uint64_t)head_index * (uint64_t)head_dimension;
    if (threadIdx.x < key_count)
    {
        score = 0.0f;
        for (element_index = 0u; element_index < head_dimension; ++element_index)
        {
            score += __bfloat162float(
                ((const __nv_bfloat16 *)query_bf16)[head_offset + element_index]) *
                __bfloat162float(((const __nv_bfloat16 *)window_key_bf16)[
                    ((uint64_t)threadIdx.x * (uint64_t)attention_dimension) +
                    head_offset + element_index]);
        }
        shared_scores[threadIdx.x] = score * rsqrtf((float)head_dimension);
    }
    __syncthreads();
    if (threadIdx.x == 0u)
    {
        maximum_score = shared_scores[0];
        for (key_index = 1u; key_index < key_count; ++key_index)
        {
            if (shared_scores[key_index] > maximum_score)
                maximum_score = shared_scores[key_index];
        }
        denominator = 0.0f;
        for (key_index = 0u; key_index < key_count; ++key_index)
        {
            shared_probabilities[key_index] =
                expf(shared_scores[key_index] - maximum_score);
            denominator += shared_probabilities[key_index];
        }
        for (key_index = 0u; key_index < key_count; ++key_index)
        {
            shared_probabilities[key_index] /= denominator;
        }
    }
    __syncthreads();
    accumulated = 0.0f;
    for (key_index = 0u; key_index < key_count; ++key_index)
    {
        accumulated += shared_probabilities[key_index] *
            __bfloat162float(((const __nv_bfloat16 *)window_value_bf16)[
                ((uint64_t)key_index * (uint64_t)attention_dimension) +
                head_offset + threadIdx.x]);
    }
    ((__nv_bfloat16 *)output_bf16)[head_offset + threadIdx.x] =
        __float2bfloat16(accumulated);
}

static __global__ void SparkGlm52DsparkSwigluKernel(
    const uint16_t *__restrict__ gate_bf16,
    const uint16_t *__restrict__ up_bf16,
    uint16_t *__restrict__ output_bf16,
    uint32_t element_count)
{
    uint32_t element_index;
    float gate_value;
    float up_value;

    element_index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (element_index >= element_count)
    {
        return;
    }
    gate_value = __bfloat162float(((const __nv_bfloat16 *)gate_bf16)[element_index]);
    up_value = __bfloat162float(((const __nv_bfloat16 *)up_bf16)[element_index]);
    ((__nv_bfloat16 *)output_bf16)[element_index] = __float2bfloat16(
        (gate_value / (1.0f + expf(-gate_value))) * up_value);
}

static __global__ void SparkGlm52DsparkRestrictedArgmaxKernel(
    const uint16_t *__restrict__ hidden_bf16,
    const uint16_t *__restrict__ lm_head_bf16,
    const uint16_t *__restrict__ markov_vocab_bf16,
    const float *__restrict__ markov_rank_f32,
    const uint32_t *__restrict__ restricted_token_ids,
    uint32_t *__restrict__ argmax_token_out,
    float *__restrict__ argmax_logit_out,
    uint32_t restricted_count,
    uint32_t hidden_dimension,
    uint32_t markov_rank,
    uint32_t markov_vocab_transposed,
    uint32_t vocab_size)
{
    __shared__ float shared_logits[SPARK_GLM52_DSPARK_BACKEND_REDUCE_THREADS];
    __shared__ uint32_t shared_tokens[SPARK_GLM52_DSPARK_BACKEND_REDUCE_THREADS];
    uint32_t row_index;
    uint32_t element_index;
    uint32_t stride_index;
    uint32_t token_id;
    uint64_t row_offset;
    float logit;
    float best_logit;
    uint32_t best_token;

    best_logit = -3.402823466e+38f;
    best_token = 0u;
    for (row_index = threadIdx.x;
         row_index < restricted_count;
         row_index += blockDim.x)
    {
        token_id = restricted_token_ids != 0
            ? restricted_token_ids[row_index]
            : row_index;
        row_offset = (uint64_t)token_id * (uint64_t)hidden_dimension;
        logit = 0.0f;
        for (element_index = 0u; element_index < hidden_dimension; ++element_index)
        {
            logit += __bfloat162float(
                ((const __nv_bfloat16 *)lm_head_bf16)[row_offset + element_index]) *
                __bfloat162float(((const __nv_bfloat16 *)hidden_bf16)[element_index]);
        }
        if (markov_vocab_bf16 != 0 && markov_rank_f32 != 0)
        {
            for (element_index = 0u; element_index < markov_rank; ++element_index)
            {
                uint64_t markov_offset;

                markov_offset = markov_vocab_transposed != 0u
                    ? ((uint64_t)element_index * (uint64_t)vocab_size) + token_id
                    : ((uint64_t)token_id * (uint64_t)markov_rank) + element_index;
                logit += __bfloat162float(
                    ((const __nv_bfloat16 *)markov_vocab_bf16)[markov_offset]) *
                    markov_rank_f32[element_index];
            }
        }
        if (logit > best_logit)
        {
            best_logit = logit;
            best_token = token_id;
        }
    }
    shared_logits[threadIdx.x] = best_logit;
    shared_tokens[threadIdx.x] = best_token;
    __syncthreads();
    for (stride_index = blockDim.x / 2u; stride_index > 0u; stride_index /= 2u)
    {
        if (threadIdx.x < stride_index &&
            (shared_logits[threadIdx.x + stride_index] > shared_logits[threadIdx.x] ||
             (shared_logits[threadIdx.x + stride_index] == shared_logits[threadIdx.x] &&
              shared_tokens[threadIdx.x + stride_index] < shared_tokens[threadIdx.x])))
        {
            shared_logits[threadIdx.x] = shared_logits[threadIdx.x + stride_index];
            shared_tokens[threadIdx.x] = shared_tokens[threadIdx.x + stride_index];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0u)
    {
        *argmax_token_out = shared_tokens[0];
        *argmax_logit_out = shared_logits[0];
    }
}

static __global__ void SparkGlm52DsparkMarkovRankKernel(
    const uint16_t *__restrict__ markov_hidden_bf16,
    const uint16_t *__restrict__ hidden_bf16,
    float *__restrict__ markov_rank_f32,
    uint32_t markov_rank,
    uint32_t hidden_dimension)
{
    uint32_t rank_index;
    uint32_t element_index;
    uint64_t row_offset;
    float value;

    rank_index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (rank_index >= markov_rank)
    {
        return;
    }
    row_offset = (uint64_t)rank_index * (uint64_t)hidden_dimension;
    value = 0.0f;
    for (element_index = 0u; element_index < hidden_dimension; ++element_index)
    {
        value += __bfloat162float(
            ((const __nv_bfloat16 *)markov_hidden_bf16)[row_offset + element_index]) *
            __bfloat162float(((const __nv_bfloat16 *)hidden_bf16)[element_index]);
    }
    markov_rank_f32[rank_index] = value;
}

static __global__ void SparkGlm52DsparkAssumedSigmoidConfidenceKernel(
    const uint16_t *__restrict__ confidence_weight_bf16,
    const uint16_t *__restrict__ confidence_bias_bf16,
    const uint16_t *__restrict__ hidden_bf16,
    float *__restrict__ confidence_out,
    uint32_t hidden_dimension)
{
    __shared__ float shared_partials[SPARK_GLM52_DSPARK_BACKEND_REDUCE_THREADS];
    uint32_t element_index;
    uint32_t stride_index;
    float partial_sum;

    partial_sum = 0.0f;
    for (element_index = threadIdx.x;
         element_index < hidden_dimension;
         element_index += blockDim.x)
    {
        partial_sum += __bfloat162float(
            ((const __nv_bfloat16 *)confidence_weight_bf16)[element_index]) *
            __bfloat162float(((const __nv_bfloat16 *)hidden_bf16)[element_index]);
    }
    shared_partials[threadIdx.x] = partial_sum;
    __syncthreads();
    for (stride_index = blockDim.x / 2u; stride_index > 0u; stride_index /= 2u)
    {
        if (threadIdx.x < stride_index)
            shared_partials[threadIdx.x] += shared_partials[threadIdx.x + stride_index];
        __syncthreads();
    }
    if (threadIdx.x != 0u)
    {
        return;
    }
    partial_sum = shared_partials[0];
    if (confidence_bias_bf16 != 0)
        partial_sum += __bfloat162float(((const __nv_bfloat16 *)confidence_bias_bf16)[0]);
    *confidence_out = 1.0f / (1.0f + expf(-partial_sum));
}

typedef struct SparkGlm52DsparkSafetensorsFile
{
    int file_descriptor;
    uint64_t file_bytes;
    uint64_t data_offset;
    const uint8_t *mapped_bytes;
    SparkJsonDocument header;
} SparkGlm52DsparkSafetensorsFile;

static SparkStatus SparkGlm52DsparkSafetensorsOpen(
    const char *path,
    SparkGlm52DsparkSafetensorsFile *file)
{
    struct stat file_stat;
    uint64_t header_bytes;
    SparkStatus status;

    memset(file, 0, sizeof(*file));
    file->file_descriptor = open(path, O_RDONLY);
    if (file->file_descriptor < 0)
    {
        return SPARK_STATUS_NOT_FOUND;
    }
    if (fstat(file->file_descriptor, &file_stat) != 0 ||
        (uint64_t)file_stat.st_size < 8u)
    {
        close(file->file_descriptor);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    file->file_bytes = (uint64_t)file_stat.st_size;
    file->mapped_bytes = (const uint8_t *)mmap(
        0, file->file_bytes, PROT_READ, MAP_PRIVATE, file->file_descriptor, 0);
    if (file->mapped_bytes == MAP_FAILED)
    {
        close(file->file_descriptor);
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    memcpy(&header_bytes, file->mapped_bytes, sizeof(header_bytes));
    if (header_bytes == 0u || (8u + header_bytes) > file->file_bytes)
    {
        munmap((void *)file->mapped_bytes, file->file_bytes);
        close(file->file_descriptor);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    file->data_offset = 8u + header_bytes;
    status = SparkJsonParseText(
        (const char *)(file->mapped_bytes + 8u),
        (size_t)header_bytes,
        &file->header);
    if (status != SPARK_STATUS_OK)
    {
        munmap((void *)file->mapped_bytes, file->file_bytes);
        close(file->file_descriptor);
        return status;
    }
    return SPARK_STATUS_OK;
}

static void SparkGlm52DsparkSafetensorsClose(
    SparkGlm52DsparkSafetensorsFile *file)
{
    SparkJsonDocumentDestroy(&file->header);
    if (file->mapped_bytes != 0 && file->mapped_bytes != MAP_FAILED)
        munmap((void *)file->mapped_bytes, file->file_bytes);
    if (file->file_descriptor >= 0)
        close(file->file_descriptor);
    memset(file, 0, sizeof(*file));
}

static void SparkGlm52DsparkSafetensorsPrintInventory(
    const SparkGlm52DsparkSafetensorsFile *file)
{
    int32_t root_token;
    uint32_t token_index;
    const SparkJsonToken *token;

    root_token = SparkJsonGetRootToken(&file->header);
    for (token_index = 0u; token_index < file->header.token_count; ++token_index)
    {
        token = &file->header.tokens[token_index];
        if (token->parent != root_token ||
            token->type != SPARK_JSON_TOKEN_STRING)
        {
            continue;
        }
        fprintf(
            stderr,
            "dspark safetensors tensor: %.*s\n",
            (int)(token->end - token->start),
            file->header.text + token->start);
    }
}

static SparkStatus SparkGlm52DsparkSafetensorsFindTensor(
    const SparkGlm52DsparkSafetensorsFile *file,
    const char *tensor_name,
    uint32_t expected_rows,
    uint32_t expected_columns,
    uint64_t *byte_offset_out,
    uint64_t *byte_count_out)
{
    char prefixed_name[192];
    int32_t root_token;
    int32_t tensor_token;
    int32_t dtype_token;
    int32_t shape_token;
    int32_t offsets_token;
    uint64_t begin_offset;
    uint64_t end_offset;
    uint32_t shape_count;
    uint32_t shape_rows;
    uint32_t shape_columns;

    root_token = SparkJsonGetRootToken(&file->header);
    tensor_token = SparkJsonFindObjectMember(&file->header, root_token, tensor_name);
    if (tensor_token < 0)
    {
        snprintf(prefixed_name, sizeof(prefixed_name), "model.%s", tensor_name);
        tensor_token = SparkJsonFindObjectMember(
            &file->header, root_token, prefixed_name);
    }
    if (tensor_token < 0)
    {
        return SPARK_STATUS_NOT_FOUND;
    }
    dtype_token = SparkJsonFindObjectMember(&file->header, tensor_token, "dtype");
    shape_token = SparkJsonFindObjectMember(&file->header, tensor_token, "shape");
    offsets_token = SparkJsonFindObjectMember(
        &file->header, tensor_token, "data_offsets");
    if (dtype_token < 0 || shape_token < 0 || offsets_token < 0 ||
        !SparkJsonStringEquals(&file->header, dtype_token, "BF16"))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    shape_count = SparkJsonGetArrayElementCount(&file->header, shape_token);
    shape_rows = 0u;
    shape_columns = 1u;
    if (shape_count == 0u || shape_count > 2u ||
        SparkJsonGetUInt32(
            &file->header,
            SparkJsonGetArrayElement(&file->header, shape_token, 0u),
            &shape_rows) != SPARK_STATUS_OK ||
        (shape_count == 2u &&
         SparkJsonGetUInt32(
             &file->header,
             SparkJsonGetArrayElement(&file->header, shape_token, 1u),
             &shape_columns) != SPARK_STATUS_OK) ||
        shape_rows != expected_rows || shape_columns != expected_columns)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkJsonGetUInt64(
            &file->header,
            SparkJsonGetArrayElement(&file->header, offsets_token, 0u),
            &begin_offset) != SPARK_STATUS_OK ||
        SparkJsonGetUInt64(
            &file->header,
            SparkJsonGetArrayElement(&file->header, offsets_token, 1u),
            &end_offset) != SPARK_STATUS_OK ||
        end_offset <= begin_offset ||
		(end_offset - begin_offset) !=
			((uint64_t)expected_rows * (uint64_t)expected_columns *
			 sizeof(uint16_t)) ||
        (file->data_offset + end_offset) > file->file_bytes)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *byte_offset_out = file->data_offset + begin_offset;
    *byte_count_out = end_offset - begin_offset;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52DsparkUploadTensor(
    SparkGlm52DsparkDraftBackend *backend,
    const SparkGlm52DsparkSafetensorsFile *file,
    const SparkGlm52DsparkTensorSpec *spec,
    uint32_t weight_index)
{
    uint64_t byte_offset;
    uint64_t byte_count;
    SparkStatus status;

    status = SparkGlm52DsparkSafetensorsFindTensor(
        file,
        spec->name,
        spec->rows,
        spec->columns,
        &byte_offset,
        &byte_count);
    if (status != SPARK_STATUS_OK)
    {
        if (spec->optional != 0u && status == SPARK_STATUS_NOT_FOUND)
        {
            backend->device_weights[weight_index] = 0;
            return SPARK_STATUS_OK;
        }
        fprintf(
            stderr,
            "dspark tensor lookup failed: name=%s rows=%u columns=%u status=%d\n",
            spec->name,
            spec->rows,
            spec->columns,
            (int)status);
        SparkGlm52DsparkSafetensorsPrintInventory(file);
        return status;
    }
    status = SparkGlm52DsparkCudaStatus(
        cudaMalloc(&backend->device_weights[weight_index], (size_t)byte_count));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    return SparkGlm52DsparkCudaStatus(
        cudaMemcpy(
            backend->device_weights[weight_index],
            file->mapped_bytes + byte_offset,
            (size_t)byte_count,
            cudaMemcpyHostToDevice));
}

static SparkStatus SparkGlm52DsparkLoadFixedTensors(
    SparkGlm52DsparkDraftBackend *backend,
    const SparkGlm52DsparkSafetensorsFile *file,
    uint32_t *markov_vocab_transposed_out)
{
    static const SparkGlm52DsparkTensorSpec fixed_specs[] =
    {
        {"embed_tokens.weight", SPARK_GLM52_DSPARK_WEIGHT_EMBED_TOKENS,
            SPARK_GLM52_DSPARK_FULL_VOCAB_SIZE,
            SPARK_GLM52_DSPARK_HIDDEN_DIMENSION, 0u},
        {"fc.weight", SPARK_GLM52_DSPARK_WEIGHT_FUSION_FC,
            SPARK_GLM52_DSPARK_HIDDEN_DIMENSION,
            SPARK_GLM52_DSPARK_DRAFT_BACKEND_FUSED_INPUT_DIMENSION, 0u},
        {"norm.weight", SPARK_GLM52_DSPARK_WEIGHT_FINAL_NORM,
            SPARK_GLM52_DSPARK_HIDDEN_DIMENSION, 1u, 0u},
        {"lm_head.weight", SPARK_GLM52_DSPARK_WEIGHT_LM_HEAD,
            SPARK_GLM52_DSPARK_FULL_VOCAB_SIZE,
            SPARK_GLM52_DSPARK_HIDDEN_DIMENSION, 0u},
        {"markov_head.hidden_proj.weight", SPARK_GLM52_DSPARK_WEIGHT_MARKOV_HIDDEN,
            SPARK_GLM52_DSPARK_MARKOV_RANK,
            SPARK_GLM52_DSPARK_HIDDEN_DIMENSION, 0u},
        {"confidence_head.weight", SPARK_GLM52_DSPARK_WEIGHT_CONFIDENCE,
            1u, SPARK_GLM52_DSPARK_HIDDEN_DIMENSION, 0u},
        {"confidence_head.bias", SPARK_GLM52_DSPARK_WEIGHT_CONFIDENCE_BIAS,
            1u, 1u, 1u}
    };
    uint32_t spec_index;
    SparkStatus status;

    for (spec_index = 0u;
         spec_index < (uint32_t)(sizeof(fixed_specs) / sizeof(fixed_specs[0]));
         ++spec_index)
    {
        status = SparkGlm52DsparkUploadTensor(
            backend,
            file,
            &fixed_specs[spec_index],
            fixed_specs[spec_index].role);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }
    {
        SparkGlm52DsparkTensorSpec vocab_spec;

        vocab_spec.name = "markov_head.vocab_proj.weight";
        vocab_spec.role = SPARK_GLM52_DSPARK_WEIGHT_MARKOV_VOCAB;
        vocab_spec.rows = SPARK_GLM52_DSPARK_FULL_VOCAB_SIZE;
        vocab_spec.columns = SPARK_GLM52_DSPARK_MARKOV_RANK;
        vocab_spec.optional = 1u;
        *markov_vocab_transposed_out = 0u;
        status = SparkGlm52DsparkUploadTensor(
            backend, file, &vocab_spec, vocab_spec.role);
        if (status == SPARK_STATUS_OK &&
            backend->device_weights[SPARK_GLM52_DSPARK_WEIGHT_MARKOV_VOCAB] == 0)
        {
            vocab_spec.rows = SPARK_GLM52_DSPARK_MARKOV_RANK;
            vocab_spec.columns = SPARK_GLM52_DSPARK_FULL_VOCAB_SIZE;
            vocab_spec.optional = 0u;
            *markov_vocab_transposed_out = 1u;
            status = SparkGlm52DsparkUploadTensor(
                backend, file, &vocab_spec, vocab_spec.role);
        }
        return status;
    }
}

static SparkStatus SparkGlm52DsparkLoadLayerTensors(
    SparkGlm52DsparkDraftBackend *backend,
    const SparkGlm52DsparkSafetensorsFile *file)
{
    static const struct { const char *suffix; uint32_t layer_weight;
        uint32_t rows; uint32_t columns; } layer_specs[] =
    {
        {"input_layernorm.weight", SPARK_GLM52_DSPARK_LAYER_WEIGHT_INPUT_NORM,
            SPARK_GLM52_DSPARK_HIDDEN_DIMENSION, 1u},
        {"self_attn.q_proj.weight", SPARK_GLM52_DSPARK_LAYER_WEIGHT_Q,
            SPARK_GLM52_DSPARK_DRAFT_BACKEND_ATTENTION_DIMENSION,
            SPARK_GLM52_DSPARK_HIDDEN_DIMENSION},
        {"self_attn.k_proj.weight", SPARK_GLM52_DSPARK_LAYER_WEIGHT_K,
            SPARK_GLM52_DSPARK_DRAFT_BACKEND_ATTENTION_DIMENSION,
            SPARK_GLM52_DSPARK_HIDDEN_DIMENSION},
        {"self_attn.v_proj.weight", SPARK_GLM52_DSPARK_LAYER_WEIGHT_V,
            SPARK_GLM52_DSPARK_DRAFT_BACKEND_ATTENTION_DIMENSION,
            SPARK_GLM52_DSPARK_HIDDEN_DIMENSION},
        {"self_attn.q_norm.weight", SPARK_GLM52_DSPARK_LAYER_WEIGHT_Q_NORM,
            SPARK_GLM52_DSPARK_DRAFT_HEAD_DIMENSION, 1u},
        {"self_attn.k_norm.weight", SPARK_GLM52_DSPARK_LAYER_WEIGHT_K_NORM,
            SPARK_GLM52_DSPARK_DRAFT_HEAD_DIMENSION, 1u},
        {"self_attn.o_proj.weight", SPARK_GLM52_DSPARK_LAYER_WEIGHT_O,
            SPARK_GLM52_DSPARK_HIDDEN_DIMENSION,
            SPARK_GLM52_DSPARK_DRAFT_BACKEND_ATTENTION_DIMENSION},
        {"post_attention_layernorm.weight", SPARK_GLM52_DSPARK_LAYER_WEIGHT_POST_NORM,
            SPARK_GLM52_DSPARK_HIDDEN_DIMENSION, 1u},
        {"mlp.gate_proj.weight", SPARK_GLM52_DSPARK_LAYER_WEIGHT_GATE,
            SPARK_GLM52_DSPARK_DRAFT_INTERMEDIATE_DIMENSION,
            SPARK_GLM52_DSPARK_HIDDEN_DIMENSION},
        {"mlp.up_proj.weight", SPARK_GLM52_DSPARK_LAYER_WEIGHT_UP,
            SPARK_GLM52_DSPARK_DRAFT_INTERMEDIATE_DIMENSION,
            SPARK_GLM52_DSPARK_HIDDEN_DIMENSION},
        {"mlp.down_proj.weight", SPARK_GLM52_DSPARK_LAYER_WEIGHT_DOWN,
            SPARK_GLM52_DSPARK_HIDDEN_DIMENSION,
            SPARK_GLM52_DSPARK_DRAFT_INTERMEDIATE_DIMENSION}
    };
    SparkGlm52DsparkTensorSpec spec;
    char tensor_name[160];
    uint32_t layer_index;
    uint32_t spec_index;
    SparkStatus status;

    for (layer_index = 0u;
         layer_index < SPARK_GLM52_DSPARK_DRAFT_LAYER_COUNT;
         ++layer_index)
    {
        for (spec_index = 0u;
             spec_index < (uint32_t)(sizeof(layer_specs) / sizeof(layer_specs[0]));
             ++spec_index)
        {
            snprintf(
                tensor_name,
                sizeof(tensor_name),
                "layers.%u.%s",
                layer_index,
                layer_specs[spec_index].suffix);
            spec.name = tensor_name;
            spec.role = SparkGlm52DsparkLayerWeightIndex(
                layer_index, layer_specs[spec_index].layer_weight);
            spec.rows = layer_specs[spec_index].rows;
            spec.columns = layer_specs[spec_index].columns;
            spec.optional = 0u;
            status = SparkGlm52DsparkUploadTensor(backend, file, &spec, spec.role);
            if (status != SPARK_STATUS_OK)
            {
                return status;
            }
        }
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52DsparkAllocateWorkspaces(
    SparkGlm52DsparkDraftBackend *backend)
{
    uint32_t vocab_capacity;
    SparkStatus status;

    vocab_capacity = backend->restricted_vocabulary_count != 0u
        ? backend->restricted_vocabulary_count
        : SPARK_GLM52_DSPARK_FULL_VOCAB_SIZE;
    status = SparkGlm52DsparkCudaStatus(cudaMalloc(
		(void **)&backend->device_tap_arena_bf16,
		(size_t)backend->maximum_lane_count *
			SPARK_GLM52_DSPARK_DRAFT_BACKEND_FUSED_INPUT_DIMENSION *
			sizeof(uint16_t)));
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkCudaStatus(cudaMalloc(
            (void **)&backend->device_step_hidden_bf16,
			SPARK_GLM52_DSPARK_HIDDEN_DIMENSION * sizeof(uint16_t)));
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkCudaStatus(cudaMalloc(
            (void **)&backend->device_step_normed_bf16,
			SPARK_GLM52_DSPARK_HIDDEN_DIMENSION * sizeof(uint16_t)));
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkCudaStatus(cudaMalloc(
            (void **)&backend->device_step_attention_bf16,
			SPARK_GLM52_DSPARK_DRAFT_BACKEND_ATTENTION_DIMENSION * sizeof(uint16_t)));
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkCudaStatus(cudaMalloc(
            (void **)&backend->device_step_query_bf16,
			SPARK_GLM52_DSPARK_DRAFT_BACKEND_ATTENTION_DIMENSION * sizeof(uint16_t)));
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkCudaStatus(cudaMalloc(
            (void **)&backend->device_step_key_bf16,
			SPARK_GLM52_DSPARK_DRAFT_BACKEND_ATTENTION_DIMENSION * sizeof(uint16_t)));
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkCudaStatus(cudaMalloc(
            (void **)&backend->device_step_value_bf16,
			SPARK_GLM52_DSPARK_DRAFT_BACKEND_ATTENTION_DIMENSION * sizeof(uint16_t)));
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkCudaStatus(cudaMalloc(
            (void **)&backend->device_step_gate_bf16,
			SPARK_GLM52_DSPARK_DRAFT_INTERMEDIATE_DIMENSION * sizeof(uint16_t)));
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkCudaStatus(cudaMalloc(
            (void **)&backend->device_step_up_bf16,
			SPARK_GLM52_DSPARK_DRAFT_INTERMEDIATE_DIMENSION * sizeof(uint16_t)));
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkCudaStatus(cudaMalloc(
            (void **)&backend->device_step_mlp_bf16,
			SPARK_GLM52_DSPARK_DRAFT_INTERMEDIATE_DIMENSION * sizeof(uint16_t)));
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkCudaStatus(cudaMalloc(
            (void **)&backend->device_step_fused_bf16,
			SPARK_GLM52_DSPARK_HIDDEN_DIMENSION * sizeof(uint16_t)));
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkCudaStatus(cudaMalloc(
            (void **)&backend->device_window_key_bf16,
			(size_t)SPARK_GLM52_DSPARK_DRAFT_LAYER_COUNT *
				SPARK_GLM52_DSPARK_BACKEND_WINDOW_TOKENS *
				SPARK_GLM52_DSPARK_DRAFT_BACKEND_ATTENTION_DIMENSION *
				sizeof(uint16_t)));
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkCudaStatus(cudaMalloc(
            (void **)&backend->device_window_value_bf16,
			(size_t)SPARK_GLM52_DSPARK_DRAFT_LAYER_COUNT *
				SPARK_GLM52_DSPARK_BACKEND_WINDOW_TOKENS *
				SPARK_GLM52_DSPARK_DRAFT_BACKEND_ATTENTION_DIMENSION *
				sizeof(uint16_t)));
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkCudaStatus(cudaMalloc(
            (void **)&backend->device_step_logits_f32,
            (size_t)vocab_capacity * sizeof(float)));
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkCudaStatus(cudaMalloc(
            (void **)&backend->device_step_markov_f32,
            SPARK_GLM52_DSPARK_MARKOV_RANK * sizeof(float)));
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkCudaStatus(cudaMalloc(
			(void **)&backend->device_step_argmax_u32,
			SPARK_GLM52_DSPARK_BACKEND_ARGMAX_WORD_COUNT * sizeof(uint32_t)));
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkCudaStatus(cudaMalloc(
            (void **)&backend->device_step_confidence_f32, sizeof(float)));
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkCudaStatus(cudaMallocHost(
			(void **)&backend->host_step_argmax_u32,
			SPARK_GLM52_DSPARK_BACKEND_ARGMAX_WORD_COUNT * sizeof(uint32_t)));
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkCudaStatus(cudaMallocHost(
            (void **)&backend->host_step_confidence_f32, sizeof(float)));
    return status;
}

static SparkStatus SparkGlm52DsparkUploadRestrictedTokenIds(
    SparkGlm52DsparkDraftBackend *backend,
    const SparkGlm52DsparkDraftBackendConfiguration *configuration)
{
    SparkStatus status;

    if (configuration->restricted_vocabulary_count == 0u)
    {
        backend->device_restricted_token_ids = 0;
        return SPARK_STATUS_OK;
    }
    if (configuration->restricted_token_ids == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkGlm52DsparkCudaStatus(cudaMalloc(
        (void **)&backend->device_restricted_token_ids,
        (size_t)configuration->restricted_vocabulary_count * sizeof(uint32_t)));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    return SparkGlm52DsparkCudaStatus(cudaMemcpy(
        backend->device_restricted_token_ids,
        configuration->restricted_token_ids,
        (size_t)configuration->restricted_vocabulary_count * sizeof(uint32_t),
        cudaMemcpyHostToDevice));
}

static void SparkGlm52DsparkFillModelContract(
    SparkGlm52DsparkModelContract *contract)
{
    memset(contract, 0, sizeof(*contract));
    contract->abi_version = SPARK_GLM52_DSPARK_ABI_VERSION;
    contract->descriptor_bytes = SPARK_GLM52_DSPARK_MODEL_CONTRACT_DESCRIPTOR_BYTES;
    contract->verifier_quantization_mode =
        SPARK_GLM52_DSPARK_VERIFIER_QUANTIZATION_FP8_E4M3_8BIT;
    contract->draft_dtype = SPARK_GLM52_DSPARK_DRAFT_DTYPE_BF16;
    contract->draft_layer_count = SPARK_GLM52_DSPARK_DRAFT_LAYER_COUNT;
    contract->block_size = SPARK_GLM52_DSPARK_BLOCK_SIZE;
    contract->hidden_dimension = SPARK_GLM52_DSPARK_HIDDEN_DIMENSION;
    contract->intermediate_dimension =
        SPARK_GLM52_DSPARK_DRAFT_INTERMEDIATE_DIMENSION;
    contract->attention_head_count = SPARK_GLM52_DSPARK_DRAFT_ATTENTION_HEAD_COUNT;
    contract->kv_head_count = SPARK_GLM52_DSPARK_DRAFT_KV_HEAD_COUNT;
    contract->head_dimension = SPARK_GLM52_DSPARK_DRAFT_HEAD_DIMENSION;
    contract->vocab_size = SPARK_GLM52_DSPARK_FULL_VOCAB_SIZE;
    contract->draft_vocab_size = SPARK_GLM52_DSPARK_FULL_VOCAB_SIZE;
    contract->markov_rank = SPARK_GLM52_DSPARK_MARKOV_RANK;
    contract->max_anchors = SPARK_GLM52_DSPARK_MAX_ANCHORS;
    contract->maximum_speculative_token_count =
        SPARK_GLM52_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT;
    contract->verifier_accept_k = 1u;
    contract->aux_layer_count = SPARK_GLM52_DSPARK_AUX_LAYER_COUNT;
    contract->enable_confidence_head = 1u;
    contract->confidence_head_with_markov = 1u;
    contract->aux_layer_ids[0] = 8u;
    contract->aux_layer_ids[1] = 23u;
    contract->aux_layer_ids[2] = 39u;
    contract->aux_layer_ids[3] = 55u;
    contract->aux_layer_ids[4] = 70u;
}

SparkStatus SparkGlm52DsparkDraftBackendInitialize(
    SparkGlm52DsparkDraftBackend *backend,
    const SparkGlm52DsparkDraftBackendConfiguration *configuration)
{
    SparkGlm52DsparkSafetensorsFile safetensors;
    uint32_t markov_vocab_transposed;
    SparkStatus status;

    if (backend == 0 || configuration == 0 ||
        configuration->abi_version != SPARK_GLM52_DSPARK_DRAFT_BACKEND_ABI_VERSION ||
        configuration->descriptor_bytes !=
            SPARK_GLM52_DSPARK_DRAFT_BACKEND_CONFIGURATION_DESCRIPTOR_BYTES ||
        configuration->safetensors_path == 0 ||
        configuration->maximum_lane_count == 0u ||
        configuration->maximum_lane_count >
            SPARK_GLM52_DSPARK_DRAFT_BACKEND_MAX_LANE_COUNT)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    memset(backend, 0, sizeof(*backend));
    backend->abi_version = SPARK_GLM52_DSPARK_DRAFT_BACKEND_ABI_VERSION;
    backend->descriptor_bytes = SPARK_GLM52_DSPARK_DRAFT_BACKEND_DESCRIPTOR_BYTES;
    backend->maximum_lane_count = configuration->maximum_lane_count;
    backend->restricted_vocabulary_count =
        configuration->restricted_vocabulary_count;
    backend->weight_count = SPARK_GLM52_DSPARK_DRAFT_BACKEND_WEIGHT_COUNT;
	backend->tap_arena_lane_stride_bytes =
		(uint64_t)SPARK_GLM52_DSPARK_DRAFT_BACKEND_FUSED_INPUT_DIMENSION *
		sizeof(uint16_t);
    SparkGlm52DsparkFillModelContract(&backend->contract);
    if (configuration->cuda_stream != 0)
    {
        backend->cuda_stream = configuration->cuda_stream;
    }
    else
    {
        status = SparkGlm52DsparkCudaStatus(
            cudaStreamCreate((cudaStream_t *)&backend->cuda_stream));
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        backend->owns_cuda_stream = 1u;
    }
    status = SparkGlm52DsparkSafetensorsOpen(
        configuration->safetensors_path, &safetensors);
    if (status != SPARK_STATUS_OK)
    {
        SparkGlm52DsparkDraftBackendTeardown(backend);
        return status;
    }
    status = SparkGlm52DsparkLoadFixedTensors(
        backend, &safetensors, &markov_vocab_transposed);
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkLoadLayerTensors(backend, &safetensors);
    SparkGlm52DsparkSafetensorsClose(&safetensors);
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkUploadRestrictedTokenIds(backend, configuration);
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkAllocateWorkspaces(backend);
    if (status != SPARK_STATUS_OK)
    {
        SparkGlm52DsparkDraftBackendTeardown(backend);
        return status;
    }
    backend->markov_vocab_transposed = markov_vocab_transposed;
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52DsparkDraftBackendTeardown(
    SparkGlm52DsparkDraftBackend *backend)
{
    uint32_t weight_index;

    if (backend == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    for (weight_index = 0u;
         weight_index < SPARK_GLM52_DSPARK_DRAFT_BACKEND_WEIGHT_COUNT;
         ++weight_index)
    {
        if (backend->device_weights[weight_index] != 0)
            cudaFree(backend->device_weights[weight_index]);
    }
    if (backend->device_restricted_token_ids != 0)
        cudaFree(backend->device_restricted_token_ids);
    if (backend->device_tap_arena_bf16 != 0)
        cudaFree(backend->device_tap_arena_bf16);
    if (backend->device_step_hidden_bf16 != 0)
        cudaFree(backend->device_step_hidden_bf16);
    if (backend->device_step_normed_bf16 != 0)
        cudaFree(backend->device_step_normed_bf16);
    if (backend->device_step_attention_bf16 != 0)
        cudaFree(backend->device_step_attention_bf16);
    if (backend->device_step_query_bf16 != 0)
        cudaFree(backend->device_step_query_bf16);
    if (backend->device_step_key_bf16 != 0)
        cudaFree(backend->device_step_key_bf16);
    if (backend->device_step_value_bf16 != 0)
        cudaFree(backend->device_step_value_bf16);
    if (backend->device_step_gate_bf16 != 0)
        cudaFree(backend->device_step_gate_bf16);
    if (backend->device_step_up_bf16 != 0)
        cudaFree(backend->device_step_up_bf16);
    if (backend->device_step_mlp_bf16 != 0)
        cudaFree(backend->device_step_mlp_bf16);
    if (backend->device_step_fused_bf16 != 0)
        cudaFree(backend->device_step_fused_bf16);
    if (backend->device_window_key_bf16 != 0)
        cudaFree(backend->device_window_key_bf16);
    if (backend->device_window_value_bf16 != 0)
        cudaFree(backend->device_window_value_bf16);
    if (backend->device_step_logits_f32 != 0)
        cudaFree(backend->device_step_logits_f32);
    if (backend->device_step_markov_f32 != 0)
        cudaFree(backend->device_step_markov_f32);
    if (backend->device_step_argmax_u32 != 0)
        cudaFree(backend->device_step_argmax_u32);
    if (backend->device_step_confidence_f32 != 0)
        cudaFree(backend->device_step_confidence_f32);
    if (backend->host_step_argmax_u32 != 0)
        cudaFreeHost(backend->host_step_argmax_u32);
    if (backend->host_step_confidence_f32 != 0)
        cudaFreeHost(backend->host_step_confidence_f32);
    if (backend->owns_cuda_stream != 0u && backend->cuda_stream != 0)
        cudaStreamDestroy((cudaStream_t)backend->cuda_stream);
    memset(backend, 0, sizeof(*backend));
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52DsparkDraftBackendModelContract(
    const SparkGlm52DsparkDraftBackend *backend,
    SparkGlm52DsparkModelContract *contract_out)
{
    if (backend == 0 || contract_out == 0 ||
        backend->abi_version != SPARK_GLM52_DSPARK_DRAFT_BACKEND_ABI_VERSION)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *contract_out = backend->contract;
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52DsparkDraftBackendTapOutputPointers(
    SparkGlm52DsparkDraftBackend *backend,
    uint32_t lane_index,
    void *tap_output_bf16[SPARK_GLM52_DSPARK_AUX_LAYER_COUNT],
    uint64_t *lane_stride_bytes_out)
{
    uint32_t tap_index;
    uint8_t *lane_base;

    if (backend == 0 || tap_output_bf16 == 0 || lane_stride_bytes_out == 0 ||
        backend->abi_version != SPARK_GLM52_DSPARK_DRAFT_BACKEND_ABI_VERSION ||
        lane_index >= backend->maximum_lane_count ||
        backend->device_tap_arena_bf16 == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    lane_base = (uint8_t *)backend->device_tap_arena_bf16 +
        ((uint64_t)lane_index * backend->tap_arena_lane_stride_bytes);
    for (tap_index = 0u;
         tap_index < SPARK_GLM52_DSPARK_AUX_LAYER_COUNT;
         ++tap_index)
    {
		tap_output_bf16[tap_index] = lane_base +
			((uint64_t)tap_index * SPARK_GLM52_DSPARK_HIDDEN_DIMENSION *
			 sizeof(uint16_t));
    }
    *lane_stride_bytes_out = backend->tap_arena_lane_stride_bytes;
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52DsparkDraftBackendStageLane(
    SparkGlm52DsparkDraftBackend *backend,
    uint32_t lane_index,
    uint64_t sequence_id,
    uint64_t sequence_position,
    uint32_t last_token_id,
    uint64_t tap_generation)
{
    SparkGlm52DsparkDraftBackendLaneState *lane_state;

    if (backend == 0 ||
        backend->abi_version != SPARK_GLM52_DSPARK_DRAFT_BACKEND_ABI_VERSION ||
        lane_index >= backend->maximum_lane_count ||
        last_token_id >= SPARK_GLM52_DSPARK_FULL_VOCAB_SIZE)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    lane_state = &backend->lane_states[lane_index];
    lane_state->sequence_id = sequence_id;
    lane_state->sequence_position = sequence_position;
    lane_state->last_token_id = last_token_id;
    lane_state->tap_generation = tap_generation;
    lane_state->staged = 1u;
    return SPARK_STATUS_OK;
}

static void SparkGlm52DsparkLaunchLinear(
    SparkGlm52DsparkDraftBackend *backend,
    uint32_t weight_index,
    const uint16_t *input_bf16,
    uint16_t *output_bf16,
    uint32_t row_count,
    uint32_t column_count,
    uint32_t accumulate_output)
{
    SparkGlm52DsparkLinearKernel<<<
        row_count,
        SPARK_GLM52_DSPARK_BACKEND_REDUCE_THREADS,
        0u,
        (cudaStream_t)backend->cuda_stream>>>(
        (const uint16_t *)backend->device_weights[weight_index],
        input_bf16,
        output_bf16,
        row_count,
        column_count,
        accumulate_output);
}

static void SparkGlm52DsparkLayerForward(
    SparkGlm52DsparkDraftBackend *backend,
    uint32_t layer_index,
    uint32_t position,
    uint32_t window_token_index)
{
    cudaStream_t stream;
    uint16_t *window_key_slot;
    uint16_t *window_value_slot;
    uint16_t *window_key_layer_base;
    uint16_t *window_value_layer_base;
    uint64_t layer_window_offset;

    stream = (cudaStream_t)backend->cuda_stream;
    layer_window_offset = (uint64_t)layer_index *
        SPARK_GLM52_DSPARK_BACKEND_WINDOW_TOKENS *
        SPARK_GLM52_DSPARK_DRAFT_BACKEND_ATTENTION_DIMENSION;
    window_key_layer_base = backend->device_window_key_bf16 + layer_window_offset;
    window_value_layer_base =
        backend->device_window_value_bf16 + layer_window_offset;
    window_key_slot = window_key_layer_base + ((uint64_t)window_token_index *
        SPARK_GLM52_DSPARK_DRAFT_BACKEND_ATTENTION_DIMENSION);
    window_value_slot = window_value_layer_base + ((uint64_t)window_token_index *
        SPARK_GLM52_DSPARK_DRAFT_BACKEND_ATTENTION_DIMENSION);
    SparkGlm52DsparkRmsNormKernel<<<1u,
        SPARK_GLM52_DSPARK_BACKEND_REDUCE_THREADS, 0u, stream>>>(
        backend->device_step_hidden_bf16,
        (const uint16_t *)backend->device_weights[
            SparkGlm52DsparkLayerWeightIndex(
                layer_index, SPARK_GLM52_DSPARK_LAYER_WEIGHT_INPUT_NORM)],
        backend->device_step_normed_bf16,
        SPARK_GLM52_DSPARK_HIDDEN_DIMENSION,
        SparkGlm52DsparkAssumedQwen3RmsNormEpsilon());
    SparkGlm52DsparkLaunchLinear(backend,
        SparkGlm52DsparkLayerWeightIndex(
            layer_index, SPARK_GLM52_DSPARK_LAYER_WEIGHT_Q),
        backend->device_step_normed_bf16, backend->device_step_query_bf16,
        SPARK_GLM52_DSPARK_DRAFT_BACKEND_ATTENTION_DIMENSION,
        SPARK_GLM52_DSPARK_HIDDEN_DIMENSION, 0u);
    SparkGlm52DsparkLaunchLinear(backend,
        SparkGlm52DsparkLayerWeightIndex(
            layer_index, SPARK_GLM52_DSPARK_LAYER_WEIGHT_K),
        backend->device_step_normed_bf16, backend->device_step_key_bf16,
        SPARK_GLM52_DSPARK_DRAFT_BACKEND_ATTENTION_DIMENSION,
        SPARK_GLM52_DSPARK_HIDDEN_DIMENSION, 0u);
    SparkGlm52DsparkLaunchLinear(backend,
        SparkGlm52DsparkLayerWeightIndex(
            layer_index, SPARK_GLM52_DSPARK_LAYER_WEIGHT_V),
        backend->device_step_normed_bf16, backend->device_step_value_bf16,
        SPARK_GLM52_DSPARK_DRAFT_BACKEND_ATTENTION_DIMENSION,
        SPARK_GLM52_DSPARK_HIDDEN_DIMENSION, 0u);
    SparkGlm52DsparkHeadNormRopeAppendKernel<<<
        SPARK_GLM52_DSPARK_DRAFT_ATTENTION_HEAD_COUNT,
        SPARK_GLM52_DSPARK_DRAFT_HEAD_DIMENSION, 0u, stream>>>(
        backend->device_step_query_bf16,
        (const uint16_t *)backend->device_weights[
            SparkGlm52DsparkLayerWeightIndex(
                layer_index, SPARK_GLM52_DSPARK_LAYER_WEIGHT_Q_NORM)],
        backend->device_step_query_bf16,
        SPARK_GLM52_DSPARK_DRAFT_HEAD_DIMENSION,
        position,
        SparkGlm52DsparkAssumedQwen3RopeTheta(),
        SparkGlm52DsparkAssumedQwen3RmsNormEpsilon());
    SparkGlm52DsparkHeadNormRopeAppendKernel<<<
        SPARK_GLM52_DSPARK_DRAFT_ATTENTION_HEAD_COUNT,
        SPARK_GLM52_DSPARK_DRAFT_HEAD_DIMENSION, 0u, stream>>>(
        backend->device_step_key_bf16,
        (const uint16_t *)backend->device_weights[
            SparkGlm52DsparkLayerWeightIndex(
                layer_index, SPARK_GLM52_DSPARK_LAYER_WEIGHT_K_NORM)],
        window_key_slot,
        SPARK_GLM52_DSPARK_DRAFT_HEAD_DIMENSION,
        position,
        SparkGlm52DsparkAssumedQwen3RopeTheta(),
        SparkGlm52DsparkAssumedQwen3RmsNormEpsilon());
    SparkGlm52DsparkCopyVectorKernel<<<
        (SPARK_GLM52_DSPARK_DRAFT_BACKEND_ATTENTION_DIMENSION +
         SPARK_GLM52_DSPARK_BACKEND_REDUCE_THREADS - 1u) /
            SPARK_GLM52_DSPARK_BACKEND_REDUCE_THREADS,
        SPARK_GLM52_DSPARK_BACKEND_REDUCE_THREADS, 0u, stream>>>(
        backend->device_step_value_bf16,
        window_value_slot,
        SPARK_GLM52_DSPARK_DRAFT_BACKEND_ATTENTION_DIMENSION);
    SparkGlm52DsparkWindowAttentionKernel<<<
        SPARK_GLM52_DSPARK_DRAFT_ATTENTION_HEAD_COUNT,
        SPARK_GLM52_DSPARK_DRAFT_HEAD_DIMENSION, 0u, stream>>>(
        backend->device_step_query_bf16,
        window_key_layer_base,
        window_value_layer_base,
        backend->device_step_attention_bf16,
        window_token_index + 1u,
        SPARK_GLM52_DSPARK_DRAFT_HEAD_DIMENSION,
        SPARK_GLM52_DSPARK_DRAFT_BACKEND_ATTENTION_DIMENSION);
    SparkGlm52DsparkLaunchLinear(backend,
        SparkGlm52DsparkLayerWeightIndex(
            layer_index, SPARK_GLM52_DSPARK_LAYER_WEIGHT_O),
        backend->device_step_attention_bf16, backend->device_step_hidden_bf16,
        SPARK_GLM52_DSPARK_HIDDEN_DIMENSION,
        SPARK_GLM52_DSPARK_DRAFT_BACKEND_ATTENTION_DIMENSION, 1u);
    SparkGlm52DsparkRmsNormKernel<<<1u,
        SPARK_GLM52_DSPARK_BACKEND_REDUCE_THREADS, 0u, stream>>>(
        backend->device_step_hidden_bf16,
        (const uint16_t *)backend->device_weights[
            SparkGlm52DsparkLayerWeightIndex(
                layer_index, SPARK_GLM52_DSPARK_LAYER_WEIGHT_POST_NORM)],
        backend->device_step_normed_bf16,
        SPARK_GLM52_DSPARK_HIDDEN_DIMENSION,
        SparkGlm52DsparkAssumedQwen3RmsNormEpsilon());
    SparkGlm52DsparkLaunchLinear(backend,
        SparkGlm52DsparkLayerWeightIndex(
            layer_index, SPARK_GLM52_DSPARK_LAYER_WEIGHT_GATE),
        backend->device_step_normed_bf16, backend->device_step_gate_bf16,
        SPARK_GLM52_DSPARK_DRAFT_INTERMEDIATE_DIMENSION,
        SPARK_GLM52_DSPARK_HIDDEN_DIMENSION, 0u);
    SparkGlm52DsparkLaunchLinear(backend,
        SparkGlm52DsparkLayerWeightIndex(
            layer_index, SPARK_GLM52_DSPARK_LAYER_WEIGHT_UP),
        backend->device_step_normed_bf16, backend->device_step_up_bf16,
        SPARK_GLM52_DSPARK_DRAFT_INTERMEDIATE_DIMENSION,
        SPARK_GLM52_DSPARK_HIDDEN_DIMENSION, 0u);
    SparkGlm52DsparkSwigluKernel<<<
        (SPARK_GLM52_DSPARK_DRAFT_INTERMEDIATE_DIMENSION +
         SPARK_GLM52_DSPARK_BACKEND_REDUCE_THREADS - 1u) /
            SPARK_GLM52_DSPARK_BACKEND_REDUCE_THREADS,
        SPARK_GLM52_DSPARK_BACKEND_REDUCE_THREADS, 0u, stream>>>(
        backend->device_step_gate_bf16,
        backend->device_step_up_bf16,
        backend->device_step_mlp_bf16,
        SPARK_GLM52_DSPARK_DRAFT_INTERMEDIATE_DIMENSION);
    SparkGlm52DsparkLaunchLinear(backend,
        SparkGlm52DsparkLayerWeightIndex(
            layer_index, SPARK_GLM52_DSPARK_LAYER_WEIGHT_DOWN),
        backend->device_step_mlp_bf16, backend->device_step_hidden_bf16,
        SPARK_GLM52_DSPARK_HIDDEN_DIMENSION,
        SPARK_GLM52_DSPARK_DRAFT_INTERMEDIATE_DIMENSION, 1u);
}

static SparkStatus SparkGlm52DsparkDraftOneToken(
    SparkGlm52DsparkDraftBackend *backend,
    uint32_t position,
    uint32_t window_token_index,
    uint32_t input_token_id,
    uint32_t first_step,
    uint32_t *token_out,
    float *confidence_out)
{
    cudaStream_t stream;
    const uint16_t *residual_bf16;
    const uint16_t *markov_vocab_bf16;
    const float *markov_rank_f32;
    uint32_t layer_index;
    uint32_t argmax_row_count;
    SparkStatus status;

    stream = (cudaStream_t)backend->cuda_stream;
    residual_bf16 = first_step != 0u
        ? backend->device_step_fused_bf16
        : backend->device_step_hidden_bf16;
    SparkGlm52DsparkComposeStepInputKernel<<<
        (SPARK_GLM52_DSPARK_HIDDEN_DIMENSION +
         SPARK_GLM52_DSPARK_BACKEND_REDUCE_THREADS - 1u) /
            SPARK_GLM52_DSPARK_BACKEND_REDUCE_THREADS,
        SPARK_GLM52_DSPARK_BACKEND_REDUCE_THREADS, 0u, stream>>>(
        (const uint16_t *)backend->device_weights[
            SPARK_GLM52_DSPARK_WEIGHT_EMBED_TOKENS],
        residual_bf16,
        backend->device_step_hidden_bf16,
        input_token_id,
        SPARK_GLM52_DSPARK_HIDDEN_DIMENSION);
    for (layer_index = 0u;
         layer_index < SPARK_GLM52_DSPARK_DRAFT_LAYER_COUNT;
         ++layer_index)
    {
        SparkGlm52DsparkLayerForward(
            backend, layer_index, position, window_token_index);
    }
    SparkGlm52DsparkRmsNormKernel<<<1u,
        SPARK_GLM52_DSPARK_BACKEND_REDUCE_THREADS, 0u, stream>>>(
        backend->device_step_hidden_bf16,
        (const uint16_t *)backend->device_weights[
            SPARK_GLM52_DSPARK_WEIGHT_FINAL_NORM],
        backend->device_step_normed_bf16,
        SPARK_GLM52_DSPARK_HIDDEN_DIMENSION,
        SparkGlm52DsparkAssumedQwen3RmsNormEpsilon());
    markov_vocab_bf16 = (const uint16_t *)backend->device_weights[
        SPARK_GLM52_DSPARK_WEIGHT_MARKOV_VOCAB];
    markov_rank_f32 = 0;
    if (markov_vocab_bf16 != 0 &&
        backend->device_weights[SPARK_GLM52_DSPARK_WEIGHT_MARKOV_HIDDEN] != 0)
    {
        SparkGlm52DsparkMarkovRankKernel<<<1u,
            SPARK_GLM52_DSPARK_MARKOV_RANK, 0u, stream>>>(
            (const uint16_t *)backend->device_weights[
                SPARK_GLM52_DSPARK_WEIGHT_MARKOV_HIDDEN],
            backend->device_step_normed_bf16,
            backend->device_step_markov_f32,
            SPARK_GLM52_DSPARK_MARKOV_RANK,
            SPARK_GLM52_DSPARK_HIDDEN_DIMENSION);
        markov_rank_f32 = backend->device_step_markov_f32;
    }
    argmax_row_count = backend->restricted_vocabulary_count != 0u
        ? backend->restricted_vocabulary_count
        : SPARK_GLM52_DSPARK_FULL_VOCAB_SIZE;
    SparkGlm52DsparkRestrictedArgmaxKernel<<<1u,
        SPARK_GLM52_DSPARK_BACKEND_REDUCE_THREADS, 0u, stream>>>(
        backend->device_step_normed_bf16,
        (const uint16_t *)backend->device_weights[SPARK_GLM52_DSPARK_WEIGHT_LM_HEAD],
        markov_vocab_bf16,
        markov_rank_f32,
        backend->device_restricted_token_ids,
        backend->device_step_argmax_u32,
        backend->device_step_logits_f32,
        argmax_row_count,
        SPARK_GLM52_DSPARK_HIDDEN_DIMENSION,
        SPARK_GLM52_DSPARK_MARKOV_RANK,
        backend->markov_vocab_transposed,
        SPARK_GLM52_DSPARK_FULL_VOCAB_SIZE);
    SparkGlm52DsparkAssumedSigmoidConfidenceKernel<<<1u,
        SPARK_GLM52_DSPARK_BACKEND_REDUCE_THREADS, 0u, stream>>>(
        (const uint16_t *)backend->device_weights[
            SPARK_GLM52_DSPARK_WEIGHT_CONFIDENCE],
        (const uint16_t *)backend->device_weights[
            SPARK_GLM52_DSPARK_WEIGHT_CONFIDENCE_BIAS],
        backend->device_step_normed_bf16,
        backend->device_step_confidence_f32,
        SPARK_GLM52_DSPARK_HIDDEN_DIMENSION);
    status = SparkGlm52DsparkCudaStatus(cudaMemcpyAsync(
        backend->host_step_argmax_u32,
        backend->device_step_argmax_u32,
        sizeof(uint32_t),
        cudaMemcpyDeviceToHost,
        stream));
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkCudaStatus(cudaMemcpyAsync(
            backend->host_step_confidence_f32,
            backend->device_step_confidence_f32,
            sizeof(float),
            cudaMemcpyDeviceToHost,
            stream));
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkCudaStatus(cudaStreamSynchronize(stream));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    *token_out = backend->host_step_argmax_u32[0];
    *confidence_out = backend->host_step_confidence_f32[0];
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52DsparkDraftBackendDraft(
    void *context,
    const SparkGlm52DsparkDraftRequest *request,
    SparkGlm52DsparkDraftResult *result)
{
    SparkGlm52DsparkDraftBackend *backend;
    SparkGlm52DsparkDraftBackendLaneState *lane_state;
    uint32_t requested_token_count;
    uint32_t token_index;
    uint32_t token_id;
    uint32_t confidence_milli;
    float confidence;
    SparkStatus status;

    backend = (SparkGlm52DsparkDraftBackend *)context;
    if (backend == 0 || request == 0 || result == 0 ||
        backend->abi_version != SPARK_GLM52_DSPARK_DRAFT_BACKEND_ABI_VERSION ||
        request->abi_version != SPARK_GLM52_DSPARK_ABI_VERSION ||
        request->descriptor_bytes !=
            SPARK_GLM52_DSPARK_DRAFT_REQUEST_DESCRIPTOR_BYTES ||
        request->requested_token_count == 0u ||
        request->requested_token_count >
            SPARK_GLM52_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT ||
        request->active_sequence_index >= backend->maximum_lane_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    lane_state = &backend->lane_states[request->active_sequence_index];
    if (lane_state->staged == 0u ||
        lane_state->sequence_id != request->sequence_id ||
        lane_state->tap_generation != request->tap_generation)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    SparkGlm52DsparkLaunchLinear(
        backend,
        SPARK_GLM52_DSPARK_WEIGHT_FUSION_FC,
        backend->device_tap_arena_bf16 +
            (((uint64_t)request->active_sequence_index *
              backend->tap_arena_lane_stride_bytes) / 2u),
        backend->device_step_fused_bf16,
        SPARK_GLM52_DSPARK_HIDDEN_DIMENSION,
        SPARK_GLM52_DSPARK_DRAFT_BACKEND_FUSED_INPUT_DIMENSION,
        0u);
    memset(result, 0, sizeof(*result));
    result->abi_version = SPARK_GLM52_DSPARK_ABI_VERSION;
    result->descriptor_bytes = SPARK_GLM52_DSPARK_DRAFT_RESULT_DESCRIPTOR_BYTES;
    requested_token_count = request->requested_token_count;
    token_id = lane_state->last_token_id;
    for (token_index = 0u; token_index < requested_token_count; ++token_index)
    {
        status = SparkGlm52DsparkDraftOneToken(
            backend,
            (uint32_t)(lane_state->sequence_position + token_index),
            token_index,
            token_id,
            token_index == 0u ? 1u : 0u,
            &token_id,
            &confidence);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        confidence_milli = (uint32_t)(confidence *
            (float)SPARK_GLM52_DSPARK_CONFIDENCE_MILLI_ONE);
        if (confidence_milli > SPARK_GLM52_DSPARK_CONFIDENCE_MILLI_ONE)
            confidence_milli = SPARK_GLM52_DSPARK_CONFIDENCE_MILLI_ONE;
        result->token_ids[token_index] = token_id;
        result->confidence_milli[token_index] = confidence_milli;
    }
    result->token_count = requested_token_count;
    return SPARK_STATUS_OK;
}
