#ifndef GLM5_2_API_H
#define GLM5_2_API_H

// The model's entire ABI. Four functions and one struct.
//
// unity.cu defines them; anything that calls a GEMM includes this. That is the
// whole surface, against the old tree's extern "C" seams plus a dlopen plugin
// plus a generated kernel table plus an AOT object pack - four mechanisms for a
// build to succeed and a link to fail.
//
// The tile height is chosen inside from the token count, so a caller never picks
// one and cannot pick a wrong one.

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct LmGemmArguments;

int32_t Glm52GemmFp8(struct LmGemmArguments *args, const void *a, const void *b,
	uint32_t packed_rows, uint32_t tokens, uint32_t groups,
	uint32_t k, uint32_t n, uint32_t sms, bool grouped, void *stream);

int32_t Glm52GemmInt7(struct LmGemmArguments *args, const void *a, const void *b,
	uint32_t packed_rows, uint32_t tokens, uint32_t groups,
	uint32_t k, uint32_t n, uint32_t sms, bool grouped, void *stream);

int32_t Glm52GemmInt6(struct LmGemmArguments *args, const void *a, const void *b,
	uint32_t packed_rows, uint32_t tokens, uint32_t groups,
	uint32_t k, uint32_t n, uint32_t sms, bool grouped, void *stream);

int32_t Glm52GemmNvfp4(struct LmGemmArguments *args, const void *a, const void *b,
	uint32_t packed_rows, uint32_t tokens, uint32_t groups,
	uint32_t k, uint32_t n, uint32_t sms, bool grouped, void *stream);

#ifdef __cplusplus
}
#endif

#endif
