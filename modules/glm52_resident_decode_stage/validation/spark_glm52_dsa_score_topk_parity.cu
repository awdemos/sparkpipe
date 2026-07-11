#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "sparkpipe/spark_glm52_resident_decode_stage_required_cuda.h"

static uint32_t SparkValidationReadU32Env(const char *name,uint32_t default_value)
{
    const char *text;
    char *end;
    unsigned long value;
    text = getenv(name);
    if (text == 0 || text[0] == '\0')
        return default_value;
    end = 0;
    value = strtoul(text,&end,10);
    if (end == text || *end != '\0' || value > UINT32_MAX)
        return default_value;
    return (uint32_t)value;
}

static uint16_t SparkValidationFloatToBf16(float value)
{
    uint32_t bits;
    uint32_t rounding_bias;
    memcpy(&bits,&value,sizeof(bits));
    rounding_bias = 0x00007fffu + ((bits >> 16u) & 1u);
    return (uint16_t)((bits + rounding_bias) >> 16u);
}

static void SparkValidationBuildInputs(
    std::vector<uint16_t> *query_heads,
    std::vector<uint16_t> *key_cache,
    std::vector<uint16_t> *head_weights,
    std::vector<uint32_t> *block_table,
    std::vector<uint32_t> *context_lengths,
    std::vector<uint32_t> *first_block_offsets,
    uint32_t active_sequence_count,
    uint32_t candidate_count,
    uint32_t block_count)
{
    uint32_t sequence_index;
    uint32_t token_index;
    uint32_t bit_index;
    query_heads->assign(
        (uint64_t)active_sequence_count *
        SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_INDEX_QUERY_DIMENSION,
        0u);
    key_cache->assign(
        (uint64_t)candidate_count *
        SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_INDEX_KEY_DIMENSION,
        0u);
    head_weights->assign(
        (uint64_t)active_sequence_count *
        SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_INDEX_WEIGHT_DIMENSION,
        0u);
    block_table->resize((uint64_t)active_sequence_count * block_count);
    context_lengths->resize(active_sequence_count);
    first_block_offsets->assign(active_sequence_count,0u);
    for (token_index = 0u; token_index < candidate_count; ++token_index)
        for (bit_index = 0u; bit_index < 20u; ++bit_index)
            if ((token_index & (1u << bit_index)) != 0u)
                (*key_cache)[
                    (uint64_t)token_index *
                    SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_INDEX_KEY_DIMENSION +
                    bit_index] = SparkValidationFloatToBf16(1.0f);
    for (sequence_index = 0u; sequence_index < active_sequence_count;
         ++sequence_index)
    {
        uint64_t query_offset;
        query_offset =
            (uint64_t)sequence_index *
            SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_INDEX_QUERY_DIMENSION;
        for (bit_index = 0u; bit_index < 20u; ++bit_index)
            (*query_heads)[query_offset + bit_index] =
                SparkValidationFloatToBf16((float)(1u << bit_index));
        (*head_weights)[
            (uint64_t)sequence_index *
            SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_INDEX_WEIGHT_DIMENSION] =
            SparkValidationFloatToBf16(1.0f);
        (*context_lengths)[sequence_index] =
            candidate_count > sequence_index % 31u
            ? candidate_count - (sequence_index % 31u)
            : candidate_count;
        for (uint32_t block_index = 0u; block_index < block_count; ++block_index)
            (*block_table)[
                (uint64_t)sequence_index * block_count + block_index] =
                block_index;
    }
}

static int SparkValidationCheckOutput(
    const std::vector<uint32_t> &output,
    const std::vector<uint32_t> &context_lengths,
    uint32_t active_sequence_count,
    uint32_t selected_token_count)
{
    uint32_t sequence_index;
    for (sequence_index = 0u; sequence_index < active_sequence_count;
         ++sequence_index)
    {
        std::vector<uint8_t> observed(context_lengths[sequence_index],0u);
        uint32_t effective_count;
        uint32_t selected_index;
        effective_count = context_lengths[sequence_index] < selected_token_count
            ? context_lengths[sequence_index]
            : selected_token_count;
        for (selected_index = 0u; selected_index < effective_count;
             ++selected_index)
        {
            uint32_t token_index;
            token_index = output[
                (uint64_t)sequence_index *
                SPARK_GLM52_RESIDENT_DECODE_STAGE_SELECTED_TOKEN_COUNT +
                selected_index];
            if (token_index >= context_lengths[sequence_index] ||
                observed[token_index] != 0u)
                return 2;
            observed[token_index] = 1u;
        }
        for (selected_index = context_lengths[sequence_index] - effective_count;
             selected_index < context_lengths[sequence_index];
             ++selected_index)
            if (observed[selected_index] == 0u)
                return 2;
    }
    return 0;
}

int main(void)
{
    std::vector<uint16_t> query_heads,key_cache,head_weights;
    std::vector<uint32_t> block_table,context_lengths,first_block_offsets,output;
    void *device_query,*device_key,*device_weights;
    uint32_t *device_blocks,*device_contexts,*device_offsets,*device_output;
    float *device_scores;
    cudaStream_t stream;
    cudaEvent_t start_event,stop_event;
    uint32_t candidate_count,active_sequence_count,row_capacity,block_count;
    uint32_t warmup_count,measure_count,iteration;
    float elapsed_ms;
    SparkStatus status;
    int result;
    candidate_count = SparkValidationReadU32Env(
        "GLM52_DSA_SCORE_CANDIDATE_COUNT",
        SPARK_GLM52_MODEL_MAXIMUM_CONTEXT_TOKENS);
    active_sequence_count = SparkValidationReadU32Env(
        "GLM52_DSA_SCORE_ACTIVE_SEQUENCE_COUNT",1u);
    row_capacity = SparkValidationReadU32Env(
        "GLM52_DSA_SCORE_ROW_CAPACITY",
        SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_SCORE_TILE_ROWS);
    warmup_count = SparkValidationReadU32Env(
        "GLM52_DSA_SCORE_WARMUP_COUNT",0u);
    measure_count = SparkValidationReadU32Env(
        "GLM52_DSA_SCORE_MEASURE_COUNT",1u);
    if (candidate_count == 0u || candidate_count >
            SPARK_GLM52_MODEL_MAXIMUM_CONTEXT_TOKENS ||
        active_sequence_count == 0u || row_capacity == 0u ||
        measure_count == 0u)
        return 2;
    block_count =
        (candidate_count + SPARK_GLM52_RESIDENT_DECODE_STAGE_BLOCK_TOKENS - 1u) /
        SPARK_GLM52_RESIDENT_DECODE_STAGE_BLOCK_TOKENS;
    SparkValidationBuildInputs(
        &query_heads,&key_cache,&head_weights,&block_table,&context_lengths,
        &first_block_offsets,active_sequence_count,candidate_count,block_count);
    output.assign(
        (uint64_t)active_sequence_count *
        SPARK_GLM52_RESIDENT_DECODE_STAGE_SELECTED_TOKEN_COUNT,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_INVALID_TOKEN_ID);
    device_query = device_key = device_weights = 0;
    device_blocks = device_contexts = device_offsets = device_output = 0;
    device_scores = 0;
    stream = 0;
    start_event = stop_event = 0;
    elapsed_ms = 0.0f;
    if (cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking) != cudaSuccess ||
        cudaEventCreate(&start_event) != cudaSuccess ||
        cudaEventCreate(&stop_event) != cudaSuccess ||
        cudaMalloc(&device_query,query_heads.size() * sizeof(uint16_t)) != cudaSuccess ||
        cudaMalloc(&device_key,key_cache.size() * sizeof(uint16_t)) != cudaSuccess ||
        cudaMalloc(&device_weights,head_weights.size() * sizeof(uint16_t)) != cudaSuccess ||
        cudaMalloc((void **)&device_blocks,block_table.size() * sizeof(uint32_t)) != cudaSuccess ||
        cudaMalloc((void **)&device_contexts,context_lengths.size() * sizeof(uint32_t)) != cudaSuccess ||
        cudaMalloc((void **)&device_offsets,first_block_offsets.size() * sizeof(uint32_t)) != cudaSuccess ||
        cudaMalloc((void **)&device_output,output.size() * sizeof(uint32_t)) != cudaSuccess ||
        cudaMalloc((void **)&device_scores,(uint64_t)row_capacity * candidate_count * sizeof(float)) != cudaSuccess ||
        cudaMemcpyAsync(device_query,query_heads.data(),query_heads.size() * sizeof(uint16_t),cudaMemcpyHostToDevice,stream) != cudaSuccess ||
        cudaMemcpyAsync(device_key,key_cache.data(),key_cache.size() * sizeof(uint16_t),cudaMemcpyHostToDevice,stream) != cudaSuccess ||
        cudaMemcpyAsync(device_weights,head_weights.data(),head_weights.size() * sizeof(uint16_t),cudaMemcpyHostToDevice,stream) != cudaSuccess ||
        cudaMemcpyAsync(device_blocks,block_table.data(),block_table.size() * sizeof(uint32_t),cudaMemcpyHostToDevice,stream) != cudaSuccess ||
        cudaMemcpyAsync(device_contexts,context_lengths.data(),context_lengths.size() * sizeof(uint32_t),cudaMemcpyHostToDevice,stream) != cudaSuccess ||
        cudaMemcpyAsync(device_offsets,first_block_offsets.data(),first_block_offsets.size() * sizeof(uint32_t),cudaMemcpyHostToDevice,stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess)
        return 2;
    for (iteration = 0u; iteration < warmup_count; ++iteration)
    {
        status = SparkGlm52Sm121RequiredDecodeStageLaunchDsaIndexShareScoreTopk(
            device_query,device_key,device_weights,device_blocks,device_contexts,
            device_offsets,device_scores,device_output,active_sequence_count,
            candidate_count,row_capacity,SPARK_GLM52_RESIDENT_DECODE_STAGE_BLOCK_TOKENS,
            block_count,block_count,candidate_count,1.0f,stream);
        if (status != SPARK_STATUS_OK)
            return 2;
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess ||
        cudaEventRecord(start_event,stream) != cudaSuccess)
        return 2;
    for (iteration = 0u; iteration < measure_count; ++iteration)
    {
        status = SparkGlm52Sm121RequiredDecodeStageLaunchDsaIndexShareScoreTopk(
            device_query,device_key,device_weights,device_blocks,device_contexts,
            device_offsets,device_scores,device_output,active_sequence_count,
            candidate_count,row_capacity,SPARK_GLM52_RESIDENT_DECODE_STAGE_BLOCK_TOKENS,
            block_count,block_count,candidate_count,1.0f,stream);
        if (status != SPARK_STATUS_OK)
            return 2;
    }
    if (
        cudaEventRecord(stop_event,stream) != cudaSuccess ||
        cudaMemcpyAsync(output.data(),device_output,output.size() * sizeof(uint32_t),cudaMemcpyDeviceToHost,stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess ||
        cudaEventElapsedTime(&elapsed_ms,start_event,stop_event) != cudaSuccess)
        return 2;
    elapsed_ms /= (float)measure_count;
    result = SparkValidationCheckOutput(
        output,context_lengths,active_sequence_count,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_SELECTED_TOKEN_COUNT);
    if (result != 0)
    {
        fprintf(stderr,"dsa_score_topk_parity failed\n");
        return result;
    }
    printf(
        "dsa_score_topk_parity_pass active_sequences=%u candidate_count=%u row_capacity=%u warmups=%u measurements=%u elapsed_ms=%.3f\n",
        active_sequence_count,candidate_count,row_capacity,warmup_count,
        measure_count,(double)elapsed_ms);
    return 0;
}
