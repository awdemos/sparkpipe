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
	const uint8_t *scale_b_e8m0;
	uint32_t scale_groups;
	uint32_t prefix_built;
	const uint32_t *group_row_offset;
	uint32_t *group_tile_prefix;
	uint16_t *output_bf16;
	float *output_f32;
	uint16_t *accumulate_bf16;
};
