// Bind weights to the layer and run a rank's slice of Kimi K3.
//
// This file is now the format choice and the C ABI, nothing else. The slice
// loop, the weight table and the per-layer state binding live in slice.cuh so
// a host harness can execute them with a recorder format - this translation
// unit includes the 4-bit format headers, whose inline PTX assembles nowhere on
// a CPU, and that include is exactly what kept the loop out of every gate.
//
// Weights arrive as an explicit per-layer table for the same reason as the
// other drivers: no K3 pack format exists, and inventing one from the layer's
// requirements is the mistake glm5_2/bind.cu records its author stopping to
// avoid. What a packer produces is its business; this file needs the pointers.

#include "inference/kernels/formats/int7.cuh"
#include "inference/kernels/formats/mxfp4.cuh"
#include "inference/llms/kimi_k3/slice.cuh"

extern "C" int32_t K3StageSlice(const void *layer_weights, const void *slice_state, void *layer_buffers, uint32_t first_layer, uint32_t layer_count, uint32_t rows, uint32_t packed_rows, uint32_t context, uint32_t multiprocessors, void *stream)
{
	// THE FORMAT FOLLOWS THE CHECKPOINT'S RECIPE, NOT A GLOBAL CHOICE.
	//
	// K3's quantization_config quantises the routed experts to MXFP4 group 32
	// and its ignore list excludes attention, latent projections, shared
	// experts, routers and lm_head - and the report says the quantisation-aware
	// training ran from SFT onward, so the routed experts were trained INTO
	// that grid and nothing else was. Requantising attention to INT7 is
	// off-recipe in the same way that storing derived factors at MXFP4 would
	// be. The grid is not the protection; the training into the grid is.
	return(K3LaunchSlice<LmMxfp4,K3GlobalKv>(
		(const K3LayerWeights *)layer_weights,
		(const K3SliceState *)slice_state,
		(K3LayerBuffers *)layer_buffers,
		first_layer,layer_count,rows,packed_rows,context,multiprocessors,
		(cudaStream_t)stream));
}
