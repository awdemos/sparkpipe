#pragma once
// A recording stand-in for runtime/gemm.cuh, reached by putting
// tests/host_cuda/shim ahead of the tree on the include path.
//
// The real file is tensor-core code that will not build for a host, and its
// arithmetic is already covered - tests/test_kernel_algorithms.py checks the
// tile mapping and the host harnesses check every kernel around it. What was
// never covered is which buffer a layer hands it and which it fills, and that
// is where an external audit found three P0s that every per-kernel test passed
// straight through.
//
// So this does not multiply. It records the call and writes a value derived
// from the call index, which makes an overwrite visible: if a later GEMM lands
// on a buffer an earlier one filled and nothing read it between, the earlier
// value is simply gone and the check downstream sees the wrong index.
#include <stdint.h>
#include <vector>

#include "inference/kernels/gemm.cuh"

struct LmRecordedGemm
{
	const void *activation;
	const void *weight;
	const void *output;
	uint32_t input_dimension;
	uint32_t output_dimension;
	uint32_t packed_rows;
	bool grouped;
};

extern std::vector<LmRecordedGemm> lm_recorded_gemms;

#include "runtime/launch.h"

template<class Format, uint32_t TILE_N, uint32_t TILE_K, uint32_t STAGES, uint32_t WARPS>
static int32_t LmGemmLaunch(LmGemmArguments *args, const void *activation_bytes, const void *weight_bytes, uint32_t packed_rows, uint32_t tokens, uint32_t top_k, uint32_t group_count, uint32_t input_dimension, uint32_t output_dimension, uint32_t multiprocessors, bool grouped, cudaStream_t stream)
{
	LmRecordedGemm record;
	uint32_t row, element;
	record.activation = activation_bytes;
	record.weight = weight_bytes;
	record.output = args->output_bf16;
	record.input_dimension = input_dimension;
	record.output_dimension = output_dimension;
	record.packed_rows = packed_rows;
	record.grouped = grouped;
	lm_recorded_gemms.push_back(record);
	for (row = 0u; row < packed_rows; ++row)
		for (element = 0u; element < output_dimension; ++element)
			args->output_bf16[((uint64_t)row * output_dimension) + element] =
				LmFloatToBf16(0.125f * (float)lm_recorded_gemms.size());
	(void)tokens; (void)top_k; (void)group_count;
	(void)multiprocessors; (void)stream;
	return LM_LAUNCH_OK;
}

// The weight-only launch records identically: the layer checks are about which
// buffers flow where, and quantisation is exactly what this shim does not do.
template<class FormatB, uint32_t TILE_N, uint32_t TILE_K, uint32_t STAGES, uint32_t WARPS>
static int32_t LmGemmWeightOnlyLaunch(LmGemmArguments *args, const void *activation_bf16, const void *weight_bytes, uint32_t packed_rows, uint32_t tokens, uint32_t top_k, uint32_t group_count, uint32_t input_dimension, uint32_t output_dimension, uint32_t multiprocessors, bool grouped, cudaStream_t stream)
{
	return(LmGemmLaunch<FormatB,TILE_N,TILE_K,STAGES,WARPS>(args,activation_bf16,
		weight_bytes,packed_rows,tokens,top_k,group_count,input_dimension,
		output_dimension,multiprocessors,grouped,stream));
}
