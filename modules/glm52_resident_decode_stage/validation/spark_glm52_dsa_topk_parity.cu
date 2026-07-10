#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <queue>
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

static uint32_t SparkValidationOrderedFloatKey(float value)
{
    uint32_t bits;

    memcpy(&bits,&value,sizeof(bits));
    if ((bits & 0x80000000u) != 0u)
        return ~bits;
    return bits ^ 0x80000000u;
}

static uint64_t SparkValidationDsaSelectionKey(
    float score,
    uint32_t token_index,
    uint32_t context_length)
{
    uint64_t ordered_score;
    uint64_t tie_breaker;

    if (token_index >= context_length || score != score || score <= -FLT_MAX)
        return 0ull;
    ordered_score = (uint64_t)SparkValidationOrderedFloatKey(score);
    tie_breaker = (uint64_t)(UINT32_MAX - token_index);
    return (ordered_score << 32u) | tie_breaker;
}

static uint32_t SparkValidationDsaTokenFromKey(uint64_t key)
{
    return UINT32_MAX - (uint32_t)key;
}

static uint32_t SparkValidationHashU32(uint32_t value)
{
    value ^= value >> 16u;
    value *= 2246822519u;
    value ^= value >> 13u;
    value *= 3266489917u;
    value ^= value >> 16u;
    return value;
}

static float SparkValidationDsaScore(uint32_t sequence_index,uint32_t token_index,uint32_t context_length)
{
    uint32_t hash;
    float score;

    if (token_index >= context_length)
        return 33554432.0f;
    hash = SparkValidationHashU32(
        token_index * 747796405u + sequence_index * 2891336453u);
    score = (float)(hash & 0x000fffffu) * (1.0f / 1024.0f);
    score += (float)((int32_t)((hash >> 20u) & 255u) - 128) * 0.0001f;
    if ((token_index % 4093u) == 0u)
        return std::numeric_limits<float>::quiet_NaN();
    if ((token_index % 6197u) == 0u)
        return -FLT_MAX;
    if ((token_index % 257u) == 0u)
        score = (float)((uint32_t)score & 8191u);
    return score;
}

static void SparkValidationBuildScores(
    std::vector<float> *scores,
    std::vector<uint32_t> *context_lengths,
    uint32_t active_sequence_count,
    uint32_t candidate_count)
{
    uint32_t sequence_index;
    uint32_t token_index;

    scores->resize((uint64_t)active_sequence_count * (uint64_t)candidate_count);
    context_lengths->resize(active_sequence_count);
    for (sequence_index = 0u; sequence_index < active_sequence_count; ++sequence_index)
    {
        uint32_t context_length;

        context_length = candidate_count;
        if (sequence_index != 0u && candidate_count > (sequence_index * 137u))
            context_length = candidate_count - (sequence_index * 137u);
        (*context_lengths)[sequence_index] = context_length;
        for (token_index = 0u; token_index < candidate_count; ++token_index)
        {
            (*scores)[
                ((uint64_t)sequence_index * (uint64_t)candidate_count) +
                (uint64_t)token_index] =
                SparkValidationDsaScore(sequence_index,token_index,context_length);
        }
    }
}

static void SparkValidationBuildReferenceMask(
    const std::vector<float> &scores,
    const std::vector<uint32_t> &context_lengths,
    std::vector<uint8_t> *reference_mask,
    uint32_t sequence_index,
    uint32_t candidate_count,
    uint32_t selected_token_count)
{
    std::priority_queue<uint64_t,std::vector<uint64_t>,std::greater<uint64_t> > heap;
    uint64_t score_row_offset;
    uint32_t token_index;

    reference_mask->assign(candidate_count,0u);
    score_row_offset = (uint64_t)sequence_index * (uint64_t)candidate_count;
    for (token_index = 0u; token_index < context_lengths[sequence_index]; ++token_index)
    {
        uint64_t key;

        key = SparkValidationDsaSelectionKey(
            scores[score_row_offset + (uint64_t)token_index],
            token_index,
            context_lengths[sequence_index]);
        if (key == 0ull)
            continue;
        if (heap.size() < selected_token_count)
        {
            heap.push(key);
        }
        else if (key > heap.top())
        {
            heap.pop();
            heap.push(key);
        }
    }
    while (!heap.empty())
    {
        (*reference_mask)[SparkValidationDsaTokenFromKey(heap.top())] = 1u;
        heap.pop();
    }
}

static int SparkValidationCheckOutput(
    const std::vector<uint32_t> &output,
    const std::vector<float> &scores,
    const std::vector<uint32_t> &context_lengths,
    uint32_t active_sequence_count,
    uint32_t candidate_count,
    uint32_t selected_token_count)
{
    std::vector<uint8_t> reference_mask;
    std::vector<uint8_t> observed_mask;
    uint32_t sequence_index;

    observed_mask.resize(candidate_count);
    for (sequence_index = 0u; sequence_index < active_sequence_count; ++sequence_index)
    {
        uint32_t selected_index;
        uint32_t observed_count;
        uint32_t reference_count;
        uint32_t mismatch_count;

        SparkValidationBuildReferenceMask(
            scores,
            context_lengths,
            &reference_mask,
            sequence_index,
            candidate_count,
            selected_token_count);
        std::fill(observed_mask.begin(),observed_mask.end(),0u);
        observed_count = 0u;
        for (selected_index = 0u;
             selected_index < selected_token_count;
             ++selected_index)
        {
            uint32_t token_index;

            token_index = output[
                ((uint64_t)sequence_index *
                 (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_SELECTED_TOKEN_COUNT) +
                (uint64_t)selected_index];
            if (token_index == SPARK_GLM52_RESIDENT_DECODE_STAGE_INVALID_TOKEN_ID)
                continue;
            if (token_index >= candidate_count)
            {
                fprintf(stderr,"dsa_topk_parity invalid token sequence=%u token=%u\n",sequence_index,token_index);
                return 2;
            }
            if (observed_mask[token_index] != 0u)
            {
                fprintf(stderr,"dsa_topk_parity duplicate token sequence=%u token=%u\n",sequence_index,token_index);
                return 2;
            }
            observed_mask[token_index] = 1u;
            observed_count += 1u;
        }
        reference_count = 0u;
        mismatch_count = 0u;
        for (selected_index = 0u; selected_index < candidate_count; ++selected_index)
        {
            reference_count += reference_mask[selected_index] != 0u ? 1u : 0u;
            if (reference_mask[selected_index] != observed_mask[selected_index])
            {
                if (mismatch_count < 8u)
                {
                    fprintf(
                        stderr,
                        "dsa_topk_parity mismatch sequence=%u token=%u reference=%u observed=%u score=%.8g\n",
                        sequence_index,
                        selected_index,
                        reference_mask[selected_index],
                        observed_mask[selected_index],
                        (double)scores[
                            ((uint64_t)sequence_index *
                             (uint64_t)candidate_count) +
                            (uint64_t)selected_index]);
                }
                mismatch_count += 1u;
            }
        }
        if (mismatch_count != 0u || observed_count != reference_count)
        {
            fprintf(
                stderr,
                "dsa_topk_parity failed sequence=%u observed=%u reference=%u mismatches=%u\n",
                sequence_index,
                observed_count,
                reference_count,
                mismatch_count);
            return 2;
        }
    }
    return 0;
}

int main(void)
{
    std::vector<float> scores;
    std::vector<uint32_t> context_lengths;
    std::vector<uint32_t> output;
    float *device_scores;
    uint32_t *device_context_lengths;
    uint32_t *device_output;
    cudaStream_t stream;
    cudaEvent_t start_event;
    cudaEvent_t stop_event;
    float elapsed_ms;
    SparkStatus status;
    uint32_t candidate_count;
    uint32_t selected_token_count;
    uint32_t active_sequence_count;
    int result;

    device_scores = 0;
    device_context_lengths = 0;
    device_output = 0;
    stream = 0;
    start_event = 0;
    stop_event = 0;
    elapsed_ms = 0.0f;
    candidate_count = SparkValidationReadU32Env(
        "GLM52_DSA_TOPK_CANDIDATE_COUNT",
        SPARK_GLM52_MODEL_MAXIMUM_CONTEXT_TOKENS);
    selected_token_count = SparkValidationReadU32Env(
        "GLM52_DSA_TOPK_SELECTED_TOKEN_COUNT",
        SPARK_GLM52_RESIDENT_DECODE_STAGE_SELECTED_TOKEN_COUNT);
    active_sequence_count = SparkValidationReadU32Env("GLM52_DSA_TOPK_ACTIVE_SEQUENCE_COUNT",2u);
    if (candidate_count == 0u || selected_token_count == 0u ||
        selected_token_count > SPARK_GLM52_RESIDENT_DECODE_STAGE_SELECTED_TOKEN_COUNT ||
        active_sequence_count == 0u)
    {
        fprintf(stderr,"invalid DSA top-k parity shape\n");
        return 2;
    }
    SparkValidationBuildScores(
        &scores,
        &context_lengths,
        active_sequence_count,
        candidate_count);
    output.assign(
        (uint64_t)active_sequence_count *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_SELECTED_TOKEN_COUNT,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_INVALID_TOKEN_ID);
    if (cudaSetDevice(0) != cudaSuccess ||
        cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking) != cudaSuccess ||
        cudaEventCreate(&start_event) != cudaSuccess ||
        cudaEventCreate(&stop_event) != cudaSuccess ||
        cudaMalloc((void **)&device_scores,scores.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **)&device_context_lengths,context_lengths.size() * sizeof(uint32_t)) != cudaSuccess ||
        cudaMalloc((void **)&device_output,output.size() * sizeof(uint32_t)) != cudaSuccess ||
        cudaMemcpyAsync(device_scores,scores.data(),scores.size() * sizeof(float),cudaMemcpyHostToDevice,stream) != cudaSuccess ||
        cudaMemcpyAsync(device_context_lengths,context_lengths.data(),context_lengths.size() * sizeof(uint32_t),cudaMemcpyHostToDevice,stream) != cudaSuccess ||
        cudaMemcpyAsync(device_output,output.data(),output.size() * sizeof(uint32_t),cudaMemcpyHostToDevice,stream) != cudaSuccess ||
        cudaEventRecord(start_event,stream) != cudaSuccess)
    {
        fprintf(stderr,"dsa_topk_parity cuda setup failed\n");
        return 2;
    }
    status = SparkGlm52Sm121RequiredDecodeStageLaunchDsaIndexShareSelectTopkFromScores(
        device_scores,
        device_context_lengths,
        device_output,
        active_sequence_count,
        candidate_count,
        selected_token_count,
        stream);
    if (status != SPARK_STATUS_OK)
    {
        fprintf(stderr,"dsa_topk_parity launch failed status=%d\n",(int)status);
        return 2;
    }
    if (cudaEventRecord(stop_event,stream) != cudaSuccess ||
        cudaMemcpyAsync(output.data(),device_output,output.size() * sizeof(uint32_t),cudaMemcpyDeviceToHost,stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess ||
        cudaEventElapsedTime(&elapsed_ms,start_event,stop_event) != cudaSuccess)
    {
        fprintf(stderr,"dsa_topk_parity cuda completion failed\n");
        return 2;
    }
    result = SparkValidationCheckOutput(
        output,
        scores,
        context_lengths,
        active_sequence_count,
        candidate_count,
        selected_token_count);
    if (result != 0)
        return result;
    printf(
        "dsa_topk_parity_pass active_sequences=%u candidate_count=%u selected_token_count=%u elapsed_ms=%.3f context0=%u context_last=%u\n",
        active_sequence_count,
        candidate_count,
        selected_token_count,
        (double)elapsed_ms,
        context_lengths[0],
        context_lengths[active_sequence_count - 1u]);
    cudaFree(device_scores);
    cudaFree(device_context_lengths);
    cudaFree(device_output);
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    cudaStreamDestroy(stream);
    return 0;
}
