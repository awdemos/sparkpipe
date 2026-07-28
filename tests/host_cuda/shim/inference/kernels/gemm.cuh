#pragma once
// The tile-level GEMM, replaced. inference/kernels/gemm.cuh holds the mma loop
// and the LmGemmArguments the layer fills; runtime/gemm.cuh holds the launch
// that consumes them. Shimming only the launch left two definitions of the
// argument struct, which is how this file came to exist.
//
// The recorder's LmGemmArguments lives here so there is exactly one, and the
// launch shim includes this rather than declaring its own.
#include <stdint.h>

struct LmGemmArguments
{
	const float *scale_a;
	const float *scale_b;
	const uint32_t *group_row_offset;
	const uint32_t *group_tile_prefix;
	uint16_t *output_bf16;
};
