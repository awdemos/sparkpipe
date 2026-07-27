// The six entry points inference/stage/dispatch.cu still names, over the
// first-party path.
//
// dispatch.cu is the CUDA backend: completion callbacks, final-token copies,
// stream management. It calls six functions the deleted decode stage used to
// provide - Initialize, Quiesce, Launch, LaunchBulkPrefill, LaunchStageSlice and
// LaunchStageSliceBulkPrefill - and rewriting it to call Glm52StageSlice
// directly would mean editing a file whose job is unrelated to which kernels
// run.
//
// So these are the six, thin. Initialize and Quiesce have nothing to do: the
// first-party path holds no lazily-built state, because the kernels are
// instantiated at compile time and the workspace is bound by the host. That the
// old ones did work is a property of a design that built plans at first use.
//
// This file is a seam and should shrink to nothing when dispatch.cu is rewritten
// to call the slice directly. It says so here so that nobody mistakes it for
// architecture.

#include "sparkpipe/spark_glm52_resident_decode_stage_required_cuda.h"
#include <stdint.h>

extern "C" int32_t Glm52StageSlice(const void *node_context, void *layer_buffers,
	uint32_t first_layer, uint32_t layer_count, uint32_t rows, uint32_t packed_rows,
	uint32_t context, uint32_t multiprocessors, void *stream);

extern "C" int32_t Glm52StageSlicePrefill(const void *node_context, void *layer_buffers,
	uint32_t first_layer, uint32_t layer_count, uint32_t rows, uint32_t packed_rows,
	uint32_t context, uint32_t multiprocessors, const uint32_t *row_positions, void *stream);

// Nothing to build. The kernels exist because they were instantiated in
// llms/glm5_2/unity.cu, and the workspace is the host's.
SparkStatus SparkGlm52Sm121RequiredDecodeStageInitialize(void)
{
	return SPARK_STATUS_OK;
}

// Nothing to drain. The stream is the caller's and it synchronises it.
SparkStatus SparkGlm52Sm121RequiredDecodeStageQuiesce(void)
{
	return SPARK_STATUS_OK;
}
