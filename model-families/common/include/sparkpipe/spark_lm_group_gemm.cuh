#pragma once

// Multi-stage, TMA-staged, blockwise-scaled grouped FP8 GEMM.
//
// SHAPE, READ FROM CUTLASS RATHER THAN ASSUMED. From
// third_party/flashinfer/3rdparty/cutlass/include/cutlass/gemm/collective/
// sm120_mma_array_tma_blockwise_scaling.hpp:
//   MainloopPipeline = PipelineTmaAsync<Stages>, Stages >= 2
//   GmemTiledCopyA/B static_asserted to SM90_TMA_LOAD  -> TMA, not cp.async
//   "MMA atom must source both A and B operands from rmem" -> mma.sync, not wgmma
//   NumProducerThreadEvents = 33, one producer thread per CTA
//   static_assert TileShape K == ScaleGranularityK -> TILE_K is the scale block
// and from group_gemm_fp8_groupwise_sm120.cuh: ClusterShape 1x1x1, so no
// cluster machinery is required.
//
// FRAGMENT MAPPING IS VERIFIED, NOT DERIVED. The (lane, register, byte) ->
// (m, k) formulas below are transcribed from MMA_Traits<
// SM89_16x8x32_F32E4M3E4M3F32_TN> and SM80_16x8_Row, and
// tests/test_mma_fragment_mapping.c evaluates those CuTe layouts and checks
// every one of the 512 A, 256 B and 128 C elements against these formulas,
// including a bijection check and a negative control. That test runs on any
// host with a C compiler and gates the build. It is the reason this file is not
// in the same position as spark_lm_fp8_tile.cuh, whose header records that a
// hand-derived mapping is the one error that assembles cleanly and computes
// silently wrong numbers.
//
// WHERE THIS DIVERGES FROM CUTLASS, DELIBERATELY. CUTLASS fixes MmaTileShape at
// 128x128x128, a prefill shape. At decode the routed-MoE group height is
// batch * top_k / experts, which for GLM 5.2 at B128 is about 4 rows, so a
// 128-row M tile is 97 percent padding. The padding costs FLOP, which is free
// on a path measured near its bandwidth roof, but it also costs 16 KB per stage
// that the weight stream could spend on depth. TILE_M is a template parameter
// and tools/spark_lm_autotune.c sweeps it per token bucket.
//
// PIPELINE. barrier_full is the only mbarrier: the CTA-wide __syncthreads that
// every stage already needs (all warps read the same staged tile) also
// establishes that the stage consumed STAGES-1 iterations ago is free, so a
// separate empty barrier would be redundant synchronisation. The producer
// issues the tile STAGES-1 ahead before waiting on the current one, which keeps
// a transfer in flight across the wait.
//
// VALIDATION STATE. Every PTX form assembles against sm_121a
// (tests/test_ptx_capability_gate.py, 41 probes). The fragment mapping is
// verified (tests/test_mma_fragment_mapping.c). NOT YET VERIFIED, and requiring
// the ring: end-to-end numerics against SparkLmGroupGemmReference, the TMA
// tensor-map encoding agreeing with the swizzle assumed here, and whether the
// autotuner's candidate ordering survives measurement.

#include "sparkpipe/lm/lm_tma.cuh"
#include "sparkpipe/spark_lm_group_gemm_swizzle_contract.h"
#include <cuda_fp8.h>
#include <stdint.h>

#define SPARK_LM_GROUP_GEMM_MMA_M 16u
#define SPARK_LM_GROUP_GEMM_MMA_N 8u
#define SPARK_LM_GROUP_GEMM_MMA_K 32u
#define SPARK_LM_GROUP_GEMM_WARP_LANES 32u
#define SPARK_LM_GROUP_GEMM_SWIZZLE_CHUNKS 8u
#define SPARK_LM_GROUP_GEMM_CHUNK_BYTES 16u

// The descriptor and the kernel must apply the same swizzle or the kernel reads
// real data from the wrong place with no error anywhere.
static_assert(SPARK_LM_GROUP_GEMM_SWIZZLE_CHUNKS == SPARK_LM_GROUP_GEMM_SWIZZLE_CHUNKS_CONTRACT,
	"kernel swizzle chunk count must match the tensor-map descriptor contract");

// Tensor maps are CUtensorMap objects encoded host-side by cuTensorMapEncodeTiled
// with CU_TENSOR_MAP_SWIZZLE_128B, passed by value in __grid_constant__ storage.
// The swizzle they encode and SparkLmGroupGemmSwizzleChunk below must agree;
// that agreement is established in one place, the host launcher.
typedef struct SparkLmGroupGemmArguments
{
	const void *tensor_map_a;
	const void *tensor_map_b;
	const float *scale_a;
	const float *scale_b;
	const uint32_t *group_row_offset;
	const uint32_t *group_tile_prefix;
	void *output_bf16;
	uint32_t group_count;
	uint32_t input_dimension;
	uint32_t output_dimension;
	// Grid-sizing hint only. The kernel bounds its loop on
	// group_tile_prefix[group_count], which is the count the route actually
	// produced. An over-estimate here costs an idle block, never a phantom tile.
	uint32_t total_tiles;
}
SparkLmGroupGemmArguments;

// 128-byte swizzle: within a row, the 16-byte chunk at index c lives at
// c ^ (row % 8). Without it, ldmatrix.x4 puts 16 of 32 lanes on one bank; with
// it, 4. tests/test_mma_fragment_mapping.c reports both counts.
static __device__ __forceinline__ uint32_t SparkLmGroupGemmSwizzleChunk(uint32_t chunk, uint32_t row)
{
	return(chunk ^ (row % SPARK_LM_GROUP_GEMM_SWIZZLE_CHUNKS));
}

static __device__ __forceinline__ void SparkLmGroupGemmMma(float accumulator[4], const uint32_t a[4], const uint32_t b[2])
{
	asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
		: "+f"(accumulator[0]), "+f"(accumulator[1]), "+f"(accumulator[2]), "+f"(accumulator[3])
		: "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Composing Copy_Traits<SM75_U32x4_LDSM_N>'s SrcLayout (32,128):(128,1) with its
// DstLayout (32,(32,4)):(32,(1,1024)) puts register r of thread t at source
// bytes [t*4 + r*128, +4), which is the chunk supplied by lane t/4 + 8r. For
// that to feed the verified A layout, lane L must supply row (L%8) + 8*((L/8)%2)
// at chunk offset L/16.
static __device__ __forceinline__ uint32_t SparkLmGroupGemmLdmatrixRow(uint32_t lane)
{
	return((lane % 8u) + (8u * ((lane / 8u) % 2u)));
}

static __device__ __forceinline__ void SparkLmGroupGemmLoadFragmentA(uint32_t fragment[4], const uint8_t *tile, uint32_t tile_k, uint32_t row_base, uint32_t k_chunk_base, uint32_t lane)
{
	uint32_t row,chunk;
	row = row_base + SparkLmGroupGemmLdmatrixRow(lane);
	chunk = SparkLmGroupGemmSwizzleChunk(k_chunk_base + (lane / 16u),row);
	asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
		: "=r"(fragment[0]), "=r"(fragment[1]), "=r"(fragment[2]), "=r"(fragment[3])
		: "r"(LmTmaSharedAddress(tile + (row * tile_k) + (chunk * SPARK_LM_GROUP_GEMM_CHUNK_BYTES))));
}

// B needs 8 rows, not 16, so x2 with lanes 0-15 supplying rows and 16-31 the
// second k chunk. Neuron-major storage puts k contiguous, so no .trans.
static __device__ __forceinline__ void SparkLmGroupGemmLoadFragmentB(uint32_t fragment[2], const uint8_t *tile, uint32_t tile_k, uint32_t neuron_base, uint32_t k_chunk_base, uint32_t lane)
{
	uint32_t row,chunk;
	row = neuron_base + (lane % 8u);
	chunk = SparkLmGroupGemmSwizzleChunk(k_chunk_base + ((lane / 8u) % 2u),row);
	asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
		: "=r"(fragment[0]), "=r"(fragment[1])
		: "r"(LmTmaSharedAddress(tile + (row * tile_k) + (chunk * SPARK_LM_GROUP_GEMM_CHUNK_BYTES))));
}

// Verified accumulator mapping: SM80_16x8_Row gives entry e of lane l the
// element at row l/4 + 8*(e/2), column 2*(l%4) + (e%2).
static __device__ __forceinline__ uint32_t SparkLmGroupGemmAccumulatorRow(uint32_t lane, uint32_t entry)
{
	return((lane / 4u) + (8u * (entry / 2u)));
}

static __device__ __forceinline__ uint32_t SparkLmGroupGemmAccumulatorColumn(uint32_t lane, uint32_t entry)
{
	return((2u * (lane % 4u)) + (entry % 2u));
}

// Tile index -> group over group_count+1 prefix entries. O(log G); a scan over
// tiles would be O(G) per tile, which is the shape this codebase forbids.
static __device__ __forceinline__ uint32_t SparkLmGroupGemmFindGroup(const uint32_t *group_tile_prefix, uint32_t group_count, uint32_t tile_index)
{
	uint32_t low = 0u,high = group_count,middle;
	while ( low + 1u < high )
	{
		middle = (low + high) >> 1u;
		if ( group_tile_prefix[middle] <= tile_index )
			low = middle;
		else
			high = middle;
	}
	return(low);
}

template<uint32_t STAGES>
static __device__ void SparkLmGroupGemmInitialiseBarriers(uint64_t *barrier_full, uint32_t arrive_count)
{
	uint32_t stage;
	if ( threadIdx.x == 0u )
		for (stage = 0u; stage < STAGES; ++stage)
			LmMbarrierInit(&barrier_full[stage],arrive_count);
	LmMbarrierInitFence();
	__syncthreads();
}

template<uint32_t STAGES>
static __device__ void SparkLmGroupGemmReleaseBarriers(uint64_t *barrier_full)
{
	uint32_t stage;
	__syncthreads();
	if ( threadIdx.x == 0u )
		for (stage = 0u; stage < STAGES; ++stage)
			LmMbarrierInvalidate(&barrier_full[stage]);
}

static __device__ __forceinline__ void SparkLmGroupGemmZeroAccumulators(float (*accumulator)[4], uint32_t count)
{
	uint32_t index,entry;
	for (index = 0u; index < count; ++index)
		for (entry = 0u; entry < 4u; ++entry)
			accumulator[index][entry] = 0.0f;
}

// One TMA pair into one stage, issued by one elected thread. The expected
// transaction byte count must equal the sum of both boxes or the barrier never
// flips, so it is derived here rather than passed in.
template<uint32_t TILE_M, uint32_t TILE_N, uint32_t TILE_K>
static __device__ __forceinline__ void SparkLmGroupGemmProduceStage(const SparkLmGroupGemmArguments &arguments, void *stage_a, void *stage_b, uint64_t *barrier, uint32_t row_base, uint32_t neuron_base, uint32_t k_base, uint32_t group_index)
{
	if ( LmTmaElectOne() == false )
		return;
	LmMbarrierArriveExpect(barrier,
		LmTmaBoxBytes(TILE_M,TILE_K,1u) + LmTmaBoxBytes(TILE_N,TILE_K,1u));
	LmTmaLoad2d(stage_a,arguments.tensor_map_a,barrier,(int32_t)k_base,(int32_t)row_base);
	LmTmaLoad3d(stage_b,arguments.tensor_map_b,barrier,(int32_t)k_base,(int32_t)neuron_base,(int32_t)group_index);
}

// Accumulate one staged K tile into the per-tile partial. The block scale is
// applied outside this loop because ScaleGranularityK equals TILE_K, which is a
// static_assert in the CUTLASS collective and the reason the scale can live
// outside the inner loop at all.
template<uint32_t TILE_M, uint32_t TILE_N, uint32_t TILE_K, uint32_t WARPS>
static __device__ void SparkLmGroupGemmConsumeStage(float (*partial)[4], const uint8_t *stage_a, const uint8_t *stage_b, uint32_t warp_index, uint32_t lane_index)
{
	const uint32_t m_fragments = TILE_M / SPARK_LM_GROUP_GEMM_MMA_M;
	const uint32_t n_fragments = TILE_N / WARPS / SPARK_LM_GROUP_GEMM_MMA_N;
	const uint32_t chunks_per_mma_k = SPARK_LM_GROUP_GEMM_MMA_K / SPARK_LM_GROUP_GEMM_CHUNK_BYTES;
	uint32_t k_step,m_index,n_index,neuron_local,chunk_base;
	uint32_t fragment_a[4],fragment_b[2];
	for (k_step = 0u; k_step < TILE_K / SPARK_LM_GROUP_GEMM_MMA_K; ++k_step)
	{
		chunk_base = k_step * chunks_per_mma_k;
		for (m_index = 0u; m_index < m_fragments; ++m_index)
		{
			SparkLmGroupGemmLoadFragmentA(fragment_a,stage_a,TILE_K,
				m_index * SPARK_LM_GROUP_GEMM_MMA_M,chunk_base,lane_index);
			for (n_index = 0u; n_index < n_fragments; ++n_index)
			{
				neuron_local = (warp_index * (TILE_N / WARPS)) + (n_index * SPARK_LM_GROUP_GEMM_MMA_N);
				SparkLmGroupGemmLoadFragmentB(fragment_b,stage_b,TILE_K,neuron_local,chunk_base,lane_index);
				SparkLmGroupGemmMma(partial[(m_index * n_fragments) + n_index],fragment_a,fragment_b);
			}
		}
	}
}

// Fold one K tile's blockwise scales into the running total and clear the
// partial. Scales are read straight from global rather than staged in shared:
// each thread touches at most two distinct A rows per accumulator pair and one
// B value per fragment, so staging them would cost a barrier interaction to
// save loads that are L2-resident.
//
// ScaleGranularityN is 128 and TILE_N is a multiple of it, so a tile can span
// several N scale blocks - indexing scale_b by group alone would silently read
// the wrong block for every fragment past the first 128 columns. The block is
// therefore derived from the fragment's own column base.
template<uint32_t TILE_M, uint32_t TILE_N, uint32_t WARPS>
static __device__ void SparkLmGroupGemmApplyBlockScale(float (*total)[4], float (*partial)[4], const SparkLmGroupGemmArguments &arguments, uint32_t count, uint32_t row_base, uint32_t neuron_base, uint32_t k_block, uint32_t group_index, uint32_t warp_index, uint32_t lane_index)
{
	const uint32_t n_fragments = TILE_N / WARPS / SPARK_LM_GROUP_GEMM_MMA_N;
	const uint32_t k_blocks = arguments.input_dimension / 128u;
	const uint32_t n_blocks = arguments.output_dimension / 128u;
	uint32_t index,entry,row,column,m_index,n_index,n_block;
	float scale_b_value,combined;
	for (index = 0u; index < count; ++index)
	{
		m_index = index / n_fragments;
		n_index = index % n_fragments;
		column = neuron_base + (warp_index * (TILE_N / WARPS)) + (n_index * SPARK_LM_GROUP_GEMM_MMA_N);
		n_block = column / 128u;
		scale_b_value = __ldg(arguments.scale_b
			+ (((uint64_t)group_index * n_blocks) + n_block) * k_blocks + k_block);
		for (entry = 0u; entry < 4u; ++entry)
		{
			row = row_base + (m_index * SPARK_LM_GROUP_GEMM_MMA_M)
				+ SparkLmGroupGemmAccumulatorRow(lane_index,entry);
			combined = __ldg(arguments.scale_a + ((uint64_t)row * k_blocks) + k_block) * scale_b_value;
			total[index][entry] = fmaf(partial[index][entry],combined,total[index][entry]);
			partial[index][entry] = 0.0f;
		}
	}
}

// Round-to-nearest-even on the way to bf16. A plain 16-bit truncation biases
// every value toward zero by up to one ulp, which accumulates across 78 layers
// and is invisible in any single-layer comparison.
static __device__ __forceinline__ uint16_t SparkLmGroupGemmFloatToBf16(float value)
{
	uint32_t bits = __float_as_uint(value);
	return((uint16_t)((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u));
}

template<uint32_t TILE_M, uint32_t TILE_N, uint32_t WARPS>
static __device__ void SparkLmGroupGemmStoreAccumulators(const SparkLmGroupGemmArguments &arguments, float (*total)[4], uint32_t count, uint32_t row_base, uint32_t row_limit, uint32_t neuron_base, uint32_t warp_index, uint32_t lane_index)
{
	const uint32_t n_fragments = TILE_N / WARPS / SPARK_LM_GROUP_GEMM_MMA_N;
	uint16_t *output = (uint16_t *)arguments.output_bf16;
	uint32_t index,entry,m_index,n_index,row,column;
	for (index = 0u; index < count; ++index)
	{
		m_index = index / n_fragments;
		n_index = index % n_fragments;
		for (entry = 0u; entry < 4u; ++entry)
		{
			row = row_base + (m_index * SPARK_LM_GROUP_GEMM_MMA_M)
				+ SparkLmGroupGemmAccumulatorRow(lane_index,entry);
			column = neuron_base + (warp_index * (TILE_N / WARPS))
				+ (n_index * SPARK_LM_GROUP_GEMM_MMA_N)
				+ SparkLmGroupGemmAccumulatorColumn(lane_index,entry);
			if ( row >= row_limit || column >= arguments.output_dimension )
				continue;
			output[((uint64_t)row * arguments.output_dimension) + column] =
				SparkLmGroupGemmFloatToBf16(total[index][entry]);
		}
	}
}

// Persistent grouped GEMM. One CTA walks tiles strided by the grid, so the grid
// is sized to the machine rather than to the problem and a short group never
// leaves an SM idle behind a long one.
template<uint32_t TILE_M, uint32_t TILE_N, uint32_t TILE_K, uint32_t STAGES, uint32_t WARPS>
static __global__ __launch_bounds__(WARPS * SPARK_LM_GROUP_GEMM_WARP_LANES, 1)
void SparkLmGroupGemmFp8Kernel(__grid_constant__ const SparkLmGroupGemmArguments arguments)
{
	__shared__ __align__(LM_TMA_ALIGNMENT_BYTES) uint8_t stage_a[STAGES][TILE_M * TILE_K];
	__shared__ __align__(LM_TMA_ALIGNMENT_BYTES) uint8_t stage_b[STAGES][TILE_N * TILE_K];
	__shared__ __align__(8) uint64_t barrier_full[STAGES];
	const uint32_t count = (TILE_M / SPARK_LM_GROUP_GEMM_MMA_M) * (TILE_N / WARPS / SPARK_LM_GROUP_GEMM_MMA_N);
	const uint32_t neuron_tiles = (arguments.output_dimension + TILE_N - 1u) / TILE_N;
	const uint32_t k_tiles = arguments.input_dimension / TILE_K;
	float total[(TILE_M / SPARK_LM_GROUP_GEMM_MMA_M) * (TILE_N / WARPS / SPARK_LM_GROUP_GEMM_MMA_N)][4];
	float partial[(TILE_M / SPARK_LM_GROUP_GEMM_MMA_M) * (TILE_N / WARPS / SPARK_LM_GROUP_GEMM_MMA_N)][4];
	uint32_t warp_index = threadIdx.x / SPARK_LM_GROUP_GEMM_WARP_LANES;
	uint32_t lane_index = threadIdx.x % SPARK_LM_GROUP_GEMM_WARP_LANES;
	uint32_t tile_index,group_index,tile_in_group,row_base,row_limit,neuron_base,k_tile,stage,ahead,total_tiles;
	SparkLmGroupGemmInitialiseBarriers<STAGES>(barrier_full,WARPS * SPARK_LM_GROUP_GEMM_WARP_LANES);
	// The tile total is whatever the route actually produced, read from the end
	// of the prefix the route-build wrote. A host-side estimate cannot know it:
	// it is a function of per-expert row counts, which only exist on device.
	// Bounding on an estimate computed from AVERAGE rows per expert launches
	// tiles for groups that received none, and each phantom tile still streams
	// a full weight tile through the K loop - at B1, where a handful of experts
	// are touched out of 256, that is tens of times the necessary weight traffic.
	total_tiles = arguments.group_tile_prefix[arguments.group_count];
	for (tile_index = blockIdx.x; tile_index < total_tiles; tile_index += gridDim.x)
	{
		group_index = SparkLmGroupGemmFindGroup(arguments.group_tile_prefix,arguments.group_count,tile_index);
		tile_in_group = tile_index - arguments.group_tile_prefix[group_index];
		row_base = arguments.group_row_offset[group_index] + ((tile_in_group / neuron_tiles) * TILE_M);
		row_limit = arguments.group_row_offset[group_index + 1u];
		neuron_base = (tile_in_group % neuron_tiles) * TILE_N;
		SparkLmGroupGemmZeroAccumulators(total,count);
		SparkLmGroupGemmZeroAccumulators(partial,count);
		for (stage = 0u; stage + 1u < STAGES && stage < k_tiles; ++stage)
			SparkLmGroupGemmProduceStage<TILE_M,TILE_N,TILE_K>(arguments,stage_a[stage],stage_b[stage],&barrier_full[stage],row_base,neuron_base,stage * TILE_K,group_index);
		for (k_tile = 0u; k_tile < k_tiles; ++k_tile)
		{
			stage = k_tile % STAGES;
			ahead = k_tile + STAGES - 1u;
			if ( ahead < k_tiles )
				SparkLmGroupGemmProduceStage<TILE_M,TILE_N,TILE_K>(arguments,stage_a[ahead % STAGES],stage_b[ahead % STAGES],&barrier_full[ahead % STAGES],row_base,neuron_base,ahead * TILE_K,group_index);
			LmMbarrierWait(&barrier_full[stage],(k_tile / STAGES) & 1u);
			SparkLmGroupGemmConsumeStage<TILE_M,TILE_N,TILE_K,WARPS>(partial,stage_a[stage],stage_b[stage],warp_index,lane_index);
			SparkLmGroupGemmApplyBlockScale<TILE_M,TILE_N,WARPS>(total,partial,arguments,count,row_base,neuron_base,k_tile,group_index,warp_index,lane_index);
			__syncthreads();
		}
		SparkLmGroupGemmStoreAccumulators<TILE_M,TILE_N,WARPS>(arguments,total,count,row_base,row_limit,neuron_base,warp_index,lane_index);
		__syncthreads();
	}
	SparkLmGroupGemmReleaseBarriers<STAGES>(barrier_full);
}

// -- NVFP4 path -------------------------------------------------------------
//
// The routed MoE runs NVFP4 in the shipped recipe
// (examples/model_descriptions/glm52_resident_decode_stage_firmware.json sets
// resident_layer_progression to LAYER_ROUTED_NVFP4_TOPK), and that path
// currently fails closed unless the MLP execution mode is FlashInfer B12x. This
// is the first-party kernel that removes that dependency.
//
// Atom: SM120::BLOCKSCALED::SM120_16x8x64_TN_VS, verified in
// tests/test_mma_fragment_mapping.c:
//   ALayout ((4,8),(8,2,2)) : ((128,1),(16,8,512)) -> (M16,K64)
//   BLayout ((4,8),(8,2))   : ((64,1),(8,256))     -> (N8,K64)
//   CLayout SM80_16x8_Row, unchanged from the FP8 path
// Four A registers and two B registers, same counts as m16n8k32 - the operands
// are half the width and twice the depth, so the register footprint is
// identical and only the element-to-lane mapping changes.

#define SPARK_LM_GROUP_GEMM_NVFP4_MMA_K 64u
#define SPARK_LM_GROUP_GEMM_NVFP4_GROUP 16u
// TILE_K for NVFP4 is in ELEMENTS and must be 256, not the 128 the FP8 path
// uses. At 4 bits a 128-element K tile is 64 bytes wide, narrower than the
// 128-byte swizzle span, so it cannot be permuted at that granularity and
// cuTensorMapEncodeTiled would describe a box the kernel cannot address
// consistently. tests/test_tensor_map_geometry.c rejects the 128 case
// explicitly. The constraint does not exist for FP8, where 128 elements are
// already 128 bytes, which is exactly why it is easy to miss.
#define SPARK_LM_GROUP_GEMM_NVFP4_TILE_K 256u
#define SPARK_LM_GROUP_GEMM_DEFAULT_TILE_M 16u
#define SPARK_LM_GROUP_GEMM_DEFAULT_TILE_N 128u

// scale_vec::4X with ue4m3 scales is one scale per 16 elements, which matches
// SPARK_GLM52_MODEL_NVFP4_GROUP_SIZE. The scale operands are a b32 register
// each plus immediate byte and thread selectors; the selectors are zero for the
// canonical layout the SFA/SFB CuTe layouts describe.
static __device__ __forceinline__ void SparkLmGroupGemmMmaNvfp4(float accumulator[4], const uint32_t a[4], const uint32_t b[2], uint32_t scale_a, uint32_t scale_b)
{
	asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0, 0}, %11, {0, 0};\n"
		: "+f"(accumulator[0]), "+f"(accumulator[1]), "+f"(accumulator[2]), "+f"(accumulator[3])
		: "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
		  "r"(scale_a), "r"(scale_b));
}

// A row is K/2 bytes at 4 bits per element. Register r covers nibbles
// [8*(lane%4) + 32*(r/2), +8) of row lane/4 + 8*(r%2), which is 4 contiguous
// bytes - one aligned 32-bit shared load, no ldmatrix and no transpose.
static __device__ __forceinline__ void SparkLmGroupGemmLoadFragmentANvfp4(uint32_t fragment[4], const uint8_t *tile, uint32_t tile_k_bytes, uint32_t row_base, uint32_t k_byte_base, uint32_t lane)
{
	uint32_t reg,row,byte_offset,chunk;
	for (reg = 0u; reg < 4u; ++reg)
	{
		row = row_base + (lane / 4u) + (8u * (reg % 2u));
		byte_offset = k_byte_base + (4u * (lane % 4u)) + (16u * (reg / 2u));
		chunk = SparkLmGroupGemmSwizzleChunk(byte_offset / SPARK_LM_GROUP_GEMM_CHUNK_BYTES,row);
		fragment[reg] = *(const uint32_t *)(tile + (row * tile_k_bytes)
			+ (chunk * SPARK_LM_GROUP_GEMM_CHUNK_BYTES)
			+ (byte_offset % SPARK_LM_GROUP_GEMM_CHUNK_BYTES));
	}
}

static __device__ __forceinline__ void SparkLmGroupGemmLoadFragmentBNvfp4(uint32_t fragment[2], const uint8_t *tile, uint32_t tile_k_bytes, uint32_t neuron_base, uint32_t k_byte_base, uint32_t lane)
{
	uint32_t reg,row,byte_offset,chunk;
	for (reg = 0u; reg < 2u; ++reg)
	{
		row = neuron_base + (lane / 4u);
		byte_offset = k_byte_base + (4u * (lane % 4u)) + (16u * reg);
		chunk = SparkLmGroupGemmSwizzleChunk(byte_offset / SPARK_LM_GROUP_GEMM_CHUNK_BYTES,row);
		fragment[reg] = *(const uint32_t *)(tile + (row * tile_k_bytes)
			+ (chunk * SPARK_LM_GROUP_GEMM_CHUNK_BYTES)
			+ (byte_offset % SPARK_LM_GROUP_GEMM_CHUNK_BYTES));
	}
}

// One staged K tile at NVFP4. The block scale is consumed by the MMA itself
// rather than folded afterwards, so there is no separate rescale pass and no
// per-K-tile partial accumulator - the whole ApplyBlockScale step the FP8 path
// needs disappears here.
template<uint32_t TILE_M, uint32_t TILE_N, uint32_t TILE_K, uint32_t WARPS>
static __device__ void SparkLmGroupGemmConsumeStageNvfp4(float (*total)[4], const uint8_t *stage_a, const uint8_t *stage_b, const uint32_t *scale_a_tile, const uint32_t *scale_b_tile, uint32_t warp_index, uint32_t lane_index)
{
	const uint32_t m_fragments = TILE_M / SPARK_LM_GROUP_GEMM_MMA_M;
	const uint32_t n_fragments = TILE_N / WARPS / SPARK_LM_GROUP_GEMM_MMA_N;
	const uint32_t tile_k_bytes = TILE_K / 2u;
	uint32_t k_step,m_index,n_index,neuron_local,k_byte_base;
	uint32_t fragment_a[4],fragment_b[2];
	for (k_step = 0u; k_step < TILE_K / SPARK_LM_GROUP_GEMM_NVFP4_MMA_K; ++k_step)
	{
		k_byte_base = k_step * (SPARK_LM_GROUP_GEMM_NVFP4_MMA_K / 2u);
		for (m_index = 0u; m_index < m_fragments; ++m_index)
		{
			SparkLmGroupGemmLoadFragmentANvfp4(fragment_a,stage_a,tile_k_bytes,
				m_index * SPARK_LM_GROUP_GEMM_MMA_M,k_byte_base,lane_index);
			for (n_index = 0u; n_index < n_fragments; ++n_index)
			{
				neuron_local = (warp_index * (TILE_N / WARPS)) + (n_index * SPARK_LM_GROUP_GEMM_MMA_N);
				SparkLmGroupGemmLoadFragmentBNvfp4(fragment_b,stage_b,tile_k_bytes,
					neuron_local,k_byte_base,lane_index);
				SparkLmGroupGemmMmaNvfp4(total[(m_index * n_fragments) + n_index],fragment_a,fragment_b,
					scale_a_tile[(m_index * (TILE_K / SPARK_LM_GROUP_GEMM_NVFP4_MMA_K)) + k_step],
					scale_b_tile[(n_index * (TILE_K / SPARK_LM_GROUP_GEMM_NVFP4_MMA_K)) + k_step]);
			}
		}
	}
}


// NVFP4 grouped GEMM entry point. Same pipeline, same tile scheduler and same
// TMA staging as the FP8 kernel; the differences are the tile body and the
// absence of a rescale pass, because the block-scaled MMA consumes the UE4M3
// scales directly and there is no per-K-tile partial to fold.
template<uint32_t TILE_M, uint32_t TILE_N, uint32_t TILE_K, uint32_t STAGES, uint32_t WARPS>
static __global__ __launch_bounds__(WARPS * SPARK_LM_GROUP_GEMM_WARP_LANES, 1)
void SparkLmGroupGemmNvfp4Kernel(__grid_constant__ const SparkLmGroupGemmArguments arguments)
{
	static_assert(TILE_K == SPARK_LM_GROUP_GEMM_NVFP4_TILE_K,
		"NVFP4 TILE_K must be 256 elements; at 4 bits a 128-element tile is 64 bytes "
		"and is narrower than the 128-byte swizzle span");
	__shared__ __align__(LM_TMA_ALIGNMENT_BYTES) uint8_t stage_a[STAGES][TILE_M * TILE_K / 2u];
	__shared__ __align__(LM_TMA_ALIGNMENT_BYTES) uint8_t stage_b[STAGES][TILE_N * TILE_K / 2u];
	__shared__ __align__(8) uint64_t barrier_full[STAGES];
	const uint32_t count = (TILE_M / SPARK_LM_GROUP_GEMM_MMA_M) * (TILE_N / WARPS / SPARK_LM_GROUP_GEMM_MMA_N);
	const uint32_t neuron_tiles = (arguments.output_dimension + TILE_N - 1u) / TILE_N;
	const uint32_t k_tiles = arguments.input_dimension / TILE_K;
	const uint32_t scales_per_row = TILE_K / SPARK_LM_GROUP_GEMM_NVFP4_GROUP;
	float total[(TILE_M / SPARK_LM_GROUP_GEMM_MMA_M) * (TILE_N / WARPS / SPARK_LM_GROUP_GEMM_MMA_N)][4];
	uint32_t warp_index = threadIdx.x / SPARK_LM_GROUP_GEMM_WARP_LANES;
	uint32_t lane_index = threadIdx.x % SPARK_LM_GROUP_GEMM_WARP_LANES;
	uint32_t tile_index,group_index,tile_in_group,row_base,row_limit,neuron_base,k_tile,stage,ahead,total_tiles;
	SparkLmGroupGemmInitialiseBarriers<STAGES>(barrier_full,WARPS * SPARK_LM_GROUP_GEMM_WARP_LANES);
	// The tile total is whatever the route actually produced, read from the end
	// of the prefix the route-build wrote. A host-side estimate cannot know it:
	// it is a function of per-expert row counts, which only exist on device.
	// Bounding on an estimate computed from AVERAGE rows per expert launches
	// tiles for groups that received none, and each phantom tile still streams
	// a full weight tile through the K loop - at B1, where a handful of experts
	// are touched out of 256, that is tens of times the necessary weight traffic.
	total_tiles = arguments.group_tile_prefix[arguments.group_count];
	for (tile_index = blockIdx.x; tile_index < total_tiles; tile_index += gridDim.x)
	{
		group_index = SparkLmGroupGemmFindGroup(arguments.group_tile_prefix,arguments.group_count,tile_index);
		tile_in_group = tile_index - arguments.group_tile_prefix[group_index];
		row_base = arguments.group_row_offset[group_index] + ((tile_in_group / neuron_tiles) * TILE_M);
		row_limit = arguments.group_row_offset[group_index + 1u];
		neuron_base = (tile_in_group % neuron_tiles) * TILE_N;
		SparkLmGroupGemmZeroAccumulators(total,count);
		for (stage = 0u; stage + 1u < STAGES && stage < k_tiles; ++stage)
			SparkLmGroupGemmProduceStage<TILE_M,TILE_N,TILE_K / 2u>(arguments,stage_a[stage],stage_b[stage],&barrier_full[stage],row_base,neuron_base,stage * (TILE_K / 2u),group_index);
		for (k_tile = 0u; k_tile < k_tiles; ++k_tile)
		{
			stage = k_tile % STAGES;
			ahead = k_tile + STAGES - 1u;
			if ( ahead < k_tiles )
				SparkLmGroupGemmProduceStage<TILE_M,TILE_N,TILE_K / 2u>(arguments,stage_a[ahead % STAGES],stage_b[ahead % STAGES],&barrier_full[ahead % STAGES],row_base,neuron_base,ahead * (TILE_K / 2u),group_index);
			LmMbarrierWait(&barrier_full[stage],(k_tile / STAGES) & 1u);
			SparkLmGroupGemmConsumeStageNvfp4<TILE_M,TILE_N,TILE_K,WARPS>(total,stage_a[stage],stage_b[stage],
				(const uint32_t *)(arguments.scale_a) + (k_tile * scales_per_row),
				(const uint32_t *)(arguments.scale_b) + (k_tile * scales_per_row),
				warp_index,lane_index);
			__syncthreads();
		}
		SparkLmGroupGemmStoreAccumulators<TILE_M,TILE_N,WARPS>(arguments,total,count,row_base,row_limit,neuron_base,warp_index,lane_index);
		__syncthreads();
	}
	SparkLmGroupGemmReleaseBarriers<STAGES>(barrier_full);
}
