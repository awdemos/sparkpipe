#ifndef GLM5_2_API_H
#define GLM5_2_API_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct LmGemmArguments;

int32_t Glm52GemmBf16(
    struct LmGemmArguments *arguments,
    const void *activation_bf16,
    const void *weight_bf16,
    uint32_t packed_rows,
    uint32_t tokens,
    uint32_t group_count,
    uint32_t input_dimension,
    uint32_t output_dimension,
    uint32_t multiprocessors,
    bool grouped,
    void *stream);

int32_t Glm52GemmFp8ExpertWeightBf16Activation(
    struct LmGemmArguments *arguments,
    const void *activation_bf16,
    const void *weight_fp8_e4m3,
    uint32_t packed_rows,
    uint32_t tokens,
    uint32_t group_count,
    uint32_t input_dimension,
    uint32_t output_dimension,
    uint32_t multiprocessors,
    bool grouped,
    void *stream);

#ifdef __cplusplus
}
#endif

#endif
