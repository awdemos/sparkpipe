// Run a whole DeepSeek V4 layer - attention half and MoE half - on a CPU and
// check where its data went.
//
// Every per-kernel harness passes because every kernel is individually
// correct; the defects a driver carries are dataflow, and no per-kernel test
// can see dataflow. This is the DSv4 instance of tests/host_cuda/
// k3_layer_host.cu's argument, and the questions are the 2026-08-01 audit's:
//
//   the KV latent GEMM must read the low-rank path's quantised input - the
//     second quantise of the normed rows was deleted, so the activation
//     pointer IS the query scratch or the dedup regressed;
//   the expert GEMMs must be grouped and see rows * top_k rows;
//   the router must write f32 logits and read the full hidden;
//   the routed result must survive into hidden AND the shared expert must
//     add to it: hidden == expert_out x sum(route weights) + shared_out,
//     checked against the weights the router actually emitted rather than a
//     constant, because the host shim runs one thread per block and the
//     topk kernel's renormalise block reduction reads shared slots that
//     only threads 1..255 would have written - a device-correct pattern
//     the one-thread schedule cannot reproduce.
//
// The GEMM is a recorder reached through the include path; every other
// kernel is the one that ships.

#include "tests/host_cuda/lm_host_cuda.cuh"

#include <stdio.h>
#include <vector>

LmHostDim3 blockIdx, threadIdx, blockDim, gridDim;

uint32_t lm_topk_shared[LM_HOST_SHARED_BYTES / sizeof(uint32_t)];
float lm_norm_shared[LM_HOST_SHARED_BYTES / sizeof(float)];
float state_s[LM_HOST_SHARED_BYTES / sizeof(float)];
float lm_fused_shared[LM_HOST_SHARED_BYTES / sizeof(float)];
float lm_quant_shared[LM_HOST_SHARED_BYTES / sizeof(float)];

#include "inference/kernels/dtype.cuh"
#include "inference/kernels/tile.cuh"
#include "inference/kernels/mma.cuh"
#undef LM_WARP_LANES
#define LM_WARP_LANES LM_HOST_WARP_LANES

#include "runtime/gemm.cuh"
std::vector<LmRecordedGemm> lm_recorded_gemms;

// kv.cuh guards its store kernel on __CUDACC__ - the only such guard in the
// kernel tree. Declaring the macro for the whole file turns on PTX asm in
// dtype.cuh, so it is scoped to this one include and withdrawn: kv.cuh has a
// pragma once, so layer.cuh's own include of it is then a no-op.
#define __CUDACC__ 1
#include "inference/kernels/kv.cuh"
#undef __CUDACC__

// RECORDER FORMATS, NOT REAL ONES. The recorder does not quantise - it logs
// the call and writes an index - so the only things it touches on a Format
// are kScaleGroup and kTileK, and the real quantise kernels touch Encode.
// Two of them because the MoE static_asserts pin both groups: 128 for the
// dynamic activations, 32 for the checkpoint FP4 expert weights. Including
// the real mxfp4.cuh Encode path is not an option - it is unconditional PTX
// and assembles nowhere on a host.
struct LmHostActFormat
{
	static constexpr uint32_t kScaleGroup = 128u;
	static constexpr uint32_t kTileK = 128u;
	static constexpr uint32_t kStoredBits = 8u;
	static constexpr float kMax = 6.0f;
	static __device__ __forceinline__ uint8_t Encode(float value)
	{
		float clamped = value > kMax ? kMax : (value < -kMax ? -kMax : value);
		return((uint8_t)(((int)clamped) & 255));
	}
};
struct LmHostExpertFormat
{
	static constexpr uint32_t kScaleGroup = 32u;
	static constexpr uint32_t kTileK = 128u;
	static constexpr uint32_t kStoredBits = 4u;
	static constexpr float kMax = 6.0f;
	static __device__ __forceinline__ uint8_t Encode(float value)
	{
		float clamped = value > kMax ? kMax : (value < -kMax ? -kMax : value);
		return((uint8_t)(((int)clamped) & 15));
	}
};

#include "inference/llms/deepseek_v4/layer.cuh"

// Two tokens, six routes each, four positions of context - small enough to
// run, wide enough that the route expansion is visible: packed row 7 must
// read token 1, and a null map would have it read row 7 of a two-row buffer.
#define ROWS 2u
#define ROUTES (ROWS * DSV4_TOP_K)
#define CONTEXT 4u
// The attention kernel reads the query at a per-head stride of LATENT + ROPE;
// that is the layout this buffer sizes, and the rope-span comment in
// layer.cuh is the audit trail for why nothing in config.h names it.
#define QUERY_ROW (DSV4_HEAD_DIM + DSV4_ROPE_DIM)
#define QUERY_WIDTH (DSV4_ATTN_HEADS * QUERY_ROW)

static const uint8_t dummy_weight[16] = {0};

static uint16_t norm_weight[DSV4_HIDDEN], lowrank_norm_weight[DSV4_QUERY_LORA_RANK];
static uint16_t hidden[ROWS * DSV4_HIDDEN], residual[ROWS * DSV4_HIDDEN];
static uint16_t normed[ROWS * DSV4_HIDDEN];
static uint16_t query_bf16[ROWS * QUERY_WIDTH];
static uint16_t kv_slot[ROWS * (DSV4_HEAD_DIM + DSV4_ROPE_DIM)];
static uint16_t attention_latent[ROWS * DSV4_ATTN_HEADS * DSV4_HEAD_DIM];
static uint16_t attention_out[ROWS * DSV4_HIDDEN];
static uint8_t packed_activation[ROWS * DSV4_ATTN_HEADS * DSV4_HEAD_DIM];
static uint8_t packed_scale[ROWS * (DSV4_ATTN_HEADS * DSV4_HEAD_DIM / 128u)];
static uint8_t lowrank_input_codes[ROWS * DSV4_HIDDEN];
static uint8_t lowrank_input_scales[ROWS * (DSV4_HIDDEN / 128u)];
static uint16_t lowrank_compressed[ROWS * DSV4_QUERY_LORA_RANK];
static uint8_t lowrank_compressed_codes[ROWS * DSV4_QUERY_LORA_RANK];
static uint8_t lowrank_compressed_scales[ROWS * (DSV4_QUERY_LORA_RANK / 128u)];
static uint16_t gate_up[ROUTES * DSV4_EXPERT_INTERMEDIATE * 2u];
static uint16_t intermediate[ROUTES * DSV4_EXPERT_INTERMEDIATE];
static uint16_t expert_out[ROUTES * DSV4_HIDDEN];
static uint16_t shared_out[ROWS * DSV4_HIDDEN];
static float router_logits[ROWS * DSV4_EXPERTS];
static float route_weight[ROUTES];
static uint32_t route_expert[ROUTES], route_packed[ROUTES], route_source[ROUTES];
static uint32_t group_offsets[DSV4_EXPERTS + 1u], group_tiles_w1[DSV4_EXPERTS + 1u];
static uint32_t group_tiles_w2[DSV4_EXPERTS + 1u];
static uint32_t dense_offsets[2], dense_tiles[2];
static uint8_t kv_pool[2u * 73728u];
static uint32_t kv_pages[2];
static uint32_t sequence_of_row[ROWS], context_length[ROWS], positions[ROWS];

int main(void)
{
	static Dsv4LayerBuffers b;
	uint32_t index;
	int32_t status;
	memset(&b, 0, sizeof(b));
	for (index = 0u; index < DSV4_HIDDEN; ++index)
		norm_weight[index] = LmFloatToBf16(1.0f);
	for (index = 0u; index < DSV4_QUERY_LORA_RANK; ++index)
		lowrank_norm_weight[index] = LmFloatToBf16(1.0f);
	for (index = 0u; index < ROWS * DSV4_HIDDEN; ++index)
	{
		residual[index] = LmFloatToBf16(0.01f * (float)(index % 13));
		attention_out[index] = LmFloatToBf16(0.01f * (float)(index % 7));
	}
	b.attn_norm_weight = norm_weight;
	b.mlp_norm_weight = norm_weight;
	b.query.down_weight = dummy_weight;
	b.query.down_scale = dummy_weight;
	b.query.norm_weight = lowrank_norm_weight;
	b.query.up_weight = dummy_weight;
	b.query.up_scale = dummy_weight;
	b.query.input_dimension = DSV4_HIDDEN;
	b.query.rank = DSV4_QUERY_LORA_RANK;
	b.query.output_dimension = QUERY_WIDTH;
	b.query.norm_epsilon = DSV4_RMS_EPSILON;
	b.query_scratch.input_codes = lowrank_input_codes;
	b.query_scratch.input_scales = lowrank_input_scales;
	b.query_scratch.compressed_bf16 = lowrank_compressed;
	b.query_scratch.compressed_codes = lowrank_compressed_codes;
	b.query_scratch.compressed_scales = lowrank_compressed_scales;
	b.query_scratch.dense_row_offset = dense_offsets;
	b.query_scratch.dense_tile_prefix = dense_tiles;
	b.kv_latent_weight = dummy_weight;
	b.kv_latent_scale = dummy_weight;
	b.output_weight = dummy_weight;
	b.output_scale = dummy_weight;
	b.router_weight = dummy_weight;
	b.expert_w1_weight = dummy_weight;
	b.expert_w1_scale = dummy_weight;
	b.expert_w2_weight = dummy_weight;
	b.expert_w2_scale = dummy_weight;
	b.shared_w1_weight = dummy_weight;
	b.shared_w1_scale = dummy_weight;
	b.shared_w2_weight = dummy_weight;
	b.shared_w2_scale = dummy_weight;
	b.hidden_bf16 = hidden;
	b.residual_bf16 = residual;
	b.normed_bf16 = normed;
	b.query_bf16 = query_bf16;
	b.kv_slot_bf16 = kv_slot;
	b.attention_latent_bf16 = attention_latent;
	b.attention_out_bf16 = attention_out;
	b.packed_activation = packed_activation;
	b.packed_scale = packed_scale;
	b.gate_up_bf16 = gate_up;
	b.intermediate_bf16 = intermediate;
	b.expert_out_bf16 = expert_out;
	b.shared_out_bf16 = shared_out;
	b.router_logits = router_logits;
	b.route_expert = route_expert;
	b.route_weight = route_weight;
	b.route_source_token = route_source;
	b.route_packed_row = route_packed;
	b.dense_row_offset = dense_offsets;
	b.dense_tile_prefix = dense_tiles;
	b.group_row_offset = group_offsets;
	b.group_tile_prefix_w1 = group_tiles_w1;
	b.group_tile_prefix_w2 = group_tiles_w2;
	kv_pages[0] = 0u;
	kv_pages[1] = 1u;
	b.cache.pool = kv_pool;
	b.cache.page_table = kv_pages;
	b.cache.page_table_stride = 1u;
	b.cache.sequence_count = 2u;
	sequence_of_row[0] = 0u;
	sequence_of_row[1] = 1u;
	context_length[0] = CONTEXT;
	context_length[1] = CONTEXT;
	positions[0] = CONTEXT - 1u;
	positions[1] = CONTEXT - 1u;
	b.sequence_of_row = sequence_of_row;
	b.context_length = context_length;
	b.positions = positions;
	dense_offsets[0] = 0u;
	dense_offsets[1] = ROWS;

	// Bisect the fault: report before each half of the layer runs.
	printf("start\n"); fflush(stdout);
	status = Dsv4LayerAttention<LmHostActFormat>(&b, ROWS, CONTEXT, 1u, 0);
	printf("attention %d\n", (int)status); fflush(stdout);
	status = Dsv4LayerMoe<LmHostActFormat, LmHostExpertFormat>(
		&b, ROWS, ROUTES, 1u, 0);
	printf("moe %d\n", (int)status); fflush(stdout);
	printf("survived\n"); fflush(stdout);

	printf("gemms %u\n", (unsigned)lm_recorded_gemms.size());
	for (size_t i = 0u; i < lm_recorded_gemms.size(); ++i)
	{
		const LmRecordedGemm &g = lm_recorded_gemms[i];
		const char *name = g.output == (void *)lowrank_compressed ? "lowrank_compressed"
			: g.output == (void *)query_bf16 ? "query"
			: g.output == (void *)kv_slot ? "kv_slot"
			: g.output == (void *)attention_out ? "attention_out"
			: g.output == (void *)router_logits ? "router_logits"
			: g.output == (void *)gate_up ? "gate_up"
			: g.output == (void *)shared_out ? "shared_out"
			: g.output == (void *)expert_out ? "expert_out" : "other";
		printf("gemm %zu dest %s in %u out %u rows %u grouped %d\n",
			i, name, g.input_dimension, g.output_dimension,
			g.packed_rows, g.grouped ? 1 : 0);
	}
	// The KV latent GEMM is recorded third; its activation pointer is the
	// whole dedup. The recorder cannot see a re-quantise - it would only see
	// the pointer change back to packed_activation.
	printf("kv activation is query scratch %d\n",
		lm_recorded_gemms.size() > 2u &&
		lm_recorded_gemms[2].activation == (const void *)lowrank_input_codes &&
		lm_recorded_gemms[2].activation_scale.data ==
			(const void *)lowrank_input_scales ? 1 : 0);
	printf("hidden[0] %.6f\n", (double)LmBf16ToFloat(hidden[0]));
	printf("hidden[1] %.6f\n", (double)LmBf16ToFloat(hidden[DSV4_HIDDEN]));
	printf("shared_out[0] %.6f\n", (double)LmBf16ToFloat(shared_out[0]));
	// expert_out is the recorder's constant everywhere, so the expected hidden
	// is shared + expert x the sum of token 0's route weights. Comparing
	// against the emitted weights - not 6 x 0.25 - keeps the dataflow check
	// exact under the one-thread host schedule (see the header).
	{
		uint32_t route;
		double weight_sum = 0.0;
		for (route = 0u; route < DSV4_TOP_K; ++route)
			weight_sum += (double)route_weight[route];
		printf("route weight sum %.6f\n", weight_sum);
		printf("hidden expected %.6f\n",
			(double)LmBf16ToFloat(shared_out[0]) +
			(double)LmBf16ToFloat(expert_out[0]) * weight_sum);
	}
	return(0);
}
