#pragma once

// The GEMM. One body, every model family, every weight format.
//
// A dense linear is a grouped GEMM with one group. That is not a convenience -
// it is why there is one implementation here where the old tree had seven: a
// CUTLASS dense call, a CUTLASS grouped call, the B12x generated kernels, a
// reference, two BF16 tile policies behind a macro, and an FP8 tile excluded
// from every build. Four of the seven never executed.
//
// WHAT VARIES, AND WHAT DOES NOT. The mma instruction and its K depth vary with
// the weight format. Everything else - the staged pipeline, the tile scheduler,
// the swizzled operand loads, the epilogue - does not, and is written once. The
// two tile bodies below differ only in which mma they issue and how they source
// scales; that difference is real and the rest of the duplication was not.
//
// TILE HEIGHT. Rows per group at decode is batch * top_k / experts, so it grows
// with the batch while the tile height is compile-time. Each M tile re-reads its
// group's weight tile, so the moment rows exceed TILE_M every group splits and
// the weight stream doubles - and the weight stream is 96 percent of all traffic
// on this path. TILE_M is therefore a template parameter selected per token
// bucket, and rounding is one-directional: up wastes mma throughput on padded
// rows, which is free on a bandwidth-bound path; down costs bandwidth, which is
// not.
//
// SCALES. FP8 carries per-128-block FP32 scales applied once per K tile outside
// the mma. NVFP4 and MXFP4 carry per-group UE4M3 or UE8M0 scales consumed by the
// mma itself, so the whole rescale pass and its per-K-tile partial accumulator
// disappear on those paths.

#include "kernels/tile.cuh"
#include <stdint.h>

// Grid-independent arguments. Descriptors are CUtensorMap values encoded host
// side and passed by value in __grid_constant__ storage; the swizzle they carry
// must match LmSwizzleChunk, which the descriptor builder asserts.
struct LmGemmArguments
{
	const void *tensor_map_a;
	const void *tensor_map_b;
	const void *scale_a;
	const void *scale_b;
	const uint32_t *group_row_offset;
	const uint32_t *group_tile_prefix;
	void *output_bf16;
	uint32_t group_count;
	uint32_t input_dimension;
	uint32_t output_dimension;
};

// One operand register is four contiguous bytes in both the 8-bit and 4-bit
// atoms, so every operand load is a single aligned 32-bit shared read through
// the swizzle. This is the only place tile memory is addressed.
static __device__ __forceinline__ uint32_t LmGemmLoadOperand(const uint8_t *tile, uint32_t row, uint32_t byte_in_row, uint32_t row_pitch_bytes)
{
	return(*(const uint32_t *)(tile + LmSwizzledOffset(row,byte_in_row,row_pitch_bytes)));
}

// -- 8-bit tile body ---------------------------------------------------------
//
// Accumulates one staged K tile into a per-tile partial. The block scale is
// applied outside this loop because the scale granularity equals the tile depth,
// which is what lets it live outside the inner loop at all.
template<uint32_t TILE_M, uint32_t TILE_N, uint32_t TILE_K, uint32_t WARPS>
static __device__ void LmGemmConsume8(float (*partial)[4], const uint8_t *stage_a, const uint8_t *stage_b, uint32_t warp, uint32_t lane)
{
	const uint32_t m_frags = TILE_M / LM_MMA8_M;
	const uint32_t n_frags = TILE_N / WARPS / LM_MMA8_N;
	const uint32_t pitch = TILE_K;
	uint32_t step,mi,ni,neuron,k_byte,reg;
	uint32_t a[4],b[2];
	for (step = 0u; step < TILE_K / LM_MMA8_K; ++step)
	{
		k_byte = step * LM_MMA8_K;
		for (mi = 0u; mi < m_frags; ++mi)
		{
			for (reg = 0u; reg < 4u; ++reg)
				a[reg] = LmGemmLoadOperand(stage_a,
					(mi * LM_MMA8_M) + LmMma8OperandARow(lane,reg),
					k_byte + LmMma8OperandAByte(lane,reg),pitch);
			for (ni = 0u; ni < n_frags; ++ni)
			{
				neuron = (warp * (TILE_N / WARPS)) + (ni * LM_MMA8_N);
				for (reg = 0u; reg < 2u; ++reg)
					b[reg] = LmGemmLoadOperand(stage_b,
						neuron + LmMma8OperandBRow(lane),
						k_byte + LmMma8OperandBByte(lane,reg),pitch);
				LmMmaE4m3(partial[(mi * n_frags) + ni],a,b);
			}
		}
	}
}

// Fold one K tile's blockwise scales into the running total and clear the
// partial. Scales are read from global rather than staged: each thread touches
// two A rows per accumulator pair and one B value per fragment, so staging them
// would cost a barrier interaction to save loads that are L2 resident.
template<uint32_t TILE_M, uint32_t TILE_N, uint32_t WARPS>
static __device__ void LmGemmApplyScale8(float (*total)[4], float (*partial)[4], const LmGemmArguments &args, uint32_t count, uint32_t row_base, uint32_t neuron_base, uint32_t k_block, uint32_t group, uint32_t warp, uint32_t lane)
{
	const uint32_t n_frags = TILE_N / WARPS / LM_MMA8_N;
	const uint32_t k_blocks = args.input_dimension / 128u;
	const uint32_t n_blocks = args.output_dimension / 128u;
	const float *scale_a = (const float *)args.scale_a;
	const float *scale_b = (const float *)args.scale_b;
	uint32_t i,e,row,column,mi,ni,n_block;
	float b_value,combined;
	for (i = 0u; i < count; ++i)
	{
		mi = i / n_frags;
		ni = i % n_frags;
		column = neuron_base + (warp * (TILE_N / WARPS)) + (ni * LM_MMA8_N);
		// Scale granularity in N is 128 and a tile may span several blocks, so
		// the block is derived from the fragment's own column. Indexing by group
		// alone reads the wrong block for every fragment past the first 128.
		n_block = column / 128u;
		b_value = __ldg(scale_b + (((uint64_t)group * n_blocks) + n_block) * k_blocks + k_block);
		for (e = 0u; e < 4u; ++e)
		{
			row = row_base + (mi * LM_MMA8_M) + LmMmaAccumulatorRow(lane,e);
			combined = __ldg(scale_a + ((uint64_t)row * k_blocks) + k_block) * b_value;
			total[i][e] = fmaf(partial[i][e],combined,total[i][e]);
			partial[i][e] = 0.0f;
		}
	}
}

// -- 4-bit tile body ---------------------------------------------------------
//
// No partial accumulator and no rescale pass: the block-scaled mma consumes the
// scales directly. Byte offsets halve because two elements share a byte.
template<uint32_t TILE_M, uint32_t TILE_N, uint32_t TILE_K, uint32_t WARPS, bool NVFP4>
static __device__ void LmGemmConsume4(float (*total)[4], const uint8_t *stage_a, const uint8_t *stage_b, const uint32_t *scale_a_tile, const uint32_t *scale_b_tile, uint32_t warp, uint32_t lane)
{
	const uint32_t m_frags = TILE_M / LM_MMA4_M;
	const uint32_t n_frags = TILE_N / WARPS / LM_MMA4_N;
	const uint32_t pitch = TILE_K / 2u;
	const uint32_t steps = TILE_K / LM_MMA4_K;
	uint32_t step,mi,ni,neuron,k_byte,reg;
	uint32_t a[4],b[2];
	for (step = 0u; step < steps; ++step)
	{
		k_byte = step * (LM_MMA4_K / 2u);
		for (mi = 0u; mi < m_frags; ++mi)
		{
			for (reg = 0u; reg < 4u; ++reg)
				a[reg] = LmGemmLoadOperand(stage_a,
					(mi * LM_MMA4_M) + LmMma4OperandARow(lane,reg),
					k_byte + LmMma4OperandAByte(lane,reg),pitch);
			for (ni = 0u; ni < n_frags; ++ni)
			{
				neuron = (warp * (TILE_N / WARPS)) + (ni * LM_MMA4_N);
				for (reg = 0u; reg < 2u; ++reg)
					b[reg] = LmGemmLoadOperand(stage_b,
						neuron + LmMma4OperandBRow(lane),
						k_byte + LmMma4OperandBByte(lane,reg),pitch);
				if constexpr ( NVFP4 )
					LmMmaNvfp4(total[(mi * n_frags) + ni],a,b,
						scale_a_tile[(mi * steps) + step],scale_b_tile[(ni * steps) + step]);
				else
					LmMmaMxfp4(total[(mi * n_frags) + ni],a,b,
						scale_a_tile[(mi * steps) + step],scale_b_tile[(ni * steps) + step]);
			}
		}
	}
}

// -- epilogue ----------------------------------------------------------------
//
// Rows past a group's own count are dropped rather than written. The tile
// height is chosen to cover the busiest group, so most groups have a ragged tail
// and this is the steady state, not an edge case.
template<uint32_t TILE_M, uint32_t TILE_N, uint32_t MMA_N, uint32_t WARPS>
static __device__ void LmGemmStore(const LmGemmArguments &args, float (*total)[4], uint32_t count, uint32_t row_base, uint32_t row_limit, uint32_t neuron_base, uint32_t warp, uint32_t lane)
{
	const uint32_t n_frags = TILE_N / WARPS / MMA_N;
	uint16_t *output = (uint16_t *)args.output_bf16;
	uint32_t i,e,mi,ni,row,column;
	for (i = 0u; i < count; ++i)
	{
		mi = i / n_frags;
		ni = i % n_frags;
		for (e = 0u; e < 4u; ++e)
		{
			row = row_base + (mi * LM_MMA8_M) + LmMmaAccumulatorRow(lane,e);
			column = neuron_base + (warp * (TILE_N / WARPS)) + (ni * MMA_N)
				+ LmMmaAccumulatorColumn(lane,e);
			if ( row >= row_limit || column >= args.output_dimension )
				continue;
			output[((uint64_t)row * args.output_dimension) + column] =
				LmFloatToBf16(total[i][e]);
		}
	}
}

static __device__ __forceinline__ void LmGemmZero(float (*acc)[4], uint32_t count)
{
	uint32_t i,e;
	for (i = 0u; i < count; ++i)
		for (e = 0u; e < 4u; ++e)
			acc[i][e] = 0.0f;
}

// -- kernels -----------------------------------------------------------------
//
// The persistent grid walks tiles strided by gridDim, so it is sized to the
// machine rather than the problem and a short group never leaves an SM idle
// behind a long one. The tile total comes from the device-side prefix, never a
// host estimate: an estimate from average rows launches tiles for groups that
// received none, and each still streams a full weight tile before its stores
// are rejected.

template<uint32_t TILE_M, uint32_t TILE_N, uint32_t TILE_K, uint32_t STAGES, uint32_t WARPS>
static __global__ __launch_bounds__(WARPS * LM_WARP_LANES, 1)
void LmGemmFp8Kernel(__grid_constant__ const LmGemmArguments args, LmTileSource source_a, LmTileSource source_b, bool grouped)
{
	static_assert(LmTileKIsSwizzleable(TILE_K,8u),"FP8 TILE_K must be a whole swizzle span");
	static_assert(LmPipelineSharedBytes(TILE_M,TILE_N,TILE_K,STAGES,8u) <= 131072u,"tile does not fit shared memory");
	static_assert(TILE_M % LM_MMA8_M == 0u && TILE_N % (WARPS * LM_MMA8_N) == 0u,"tile is not a whole number of mma fragments");
	__shared__ __align__(LM_TMA_ALIGNMENT_BYTES) uint8_t stage_a[STAGES][LmTileBytes(TILE_M,TILE_K,8u)];
	__shared__ __align__(LM_TMA_ALIGNMENT_BYTES) uint8_t stage_b[STAGES][LmTileBytes(TILE_N,TILE_K,8u)];
	__shared__ __align__(8) uint64_t barrier[STAGES];
	const uint32_t count = (TILE_M / LM_MMA8_M) * (TILE_N / WARPS / LM_MMA8_N);
	const uint32_t neuron_tiles = (args.output_dimension + TILE_N - 1u) / TILE_N;
	const uint32_t k_tiles = args.input_dimension / TILE_K;
	float total[(TILE_M / LM_MMA8_M) * (TILE_N / WARPS / LM_MMA8_N)][4];
	float partial[(TILE_M / LM_MMA8_M) * (TILE_N / WARPS / LM_MMA8_N)][4];
	uint32_t warp = threadIdx.x / LM_WARP_LANES,lane = threadIdx.x % LM_WARP_LANES;
	uint32_t tile,group,in_group,row_base,row_limit,neuron_base,k,stage,ahead,total_tiles;
	LmPipelineInitialise<STAGES>(barrier,WARPS * LM_WARP_LANES);
	total_tiles = LmTotalTiles(args.group_tile_prefix,args.group_count);
	for (tile = blockIdx.x; tile < total_tiles; tile += gridDim.x)
	{
		group = LmGroupOfTile(args.group_tile_prefix,args.group_count,tile);
		in_group = tile - args.group_tile_prefix[group];
		row_base = args.group_row_offset[group] + ((in_group / neuron_tiles) * TILE_M);
		row_limit = args.group_row_offset[group + 1u];
		neuron_base = (in_group % neuron_tiles) * TILE_N;
		LmGemmZero(total,count);
		LmGemmZero(partial,count);
		for (stage = 0u; stage + 1u < STAGES && stage < k_tiles; ++stage)
			LmPipelineProduce(&source_a,&source_b,stage_a[stage],stage_b[stage],&barrier[stage],row_base,neuron_base,stage * TILE_K,group,grouped);
		for (k = 0u; k < k_tiles; ++k)
		{
			stage = LmPipelineStage(k,STAGES);
			ahead = LmPipelineAhead(k,STAGES);
			if ( ahead < k_tiles )
				LmPipelineProduce(&source_a,&source_b,stage_a[ahead % STAGES],stage_b[ahead % STAGES],&barrier[ahead % STAGES],row_base,neuron_base,ahead * TILE_K,group,grouped);
			LmMbarrierWait(&barrier[stage],LmPipelinePhase(k,STAGES));
			LmGemmConsume8<TILE_M,TILE_N,TILE_K,WARPS>(partial,stage_a[stage],stage_b[stage],warp,lane);
			LmGemmApplyScale8<TILE_M,TILE_N,WARPS>(total,partial,args,count,row_base,neuron_base,k,group,warp,lane);
			__syncthreads();
		}
		LmGemmStore<TILE_M,TILE_N,LM_MMA8_N,WARPS>(args,total,count,row_base,row_limit,neuron_base,warp,lane);
		__syncthreads();
	}
	LmPipelineRelease<STAGES>(barrier);
}

template<uint32_t TILE_M, uint32_t TILE_N, uint32_t TILE_K, uint32_t STAGES, uint32_t WARPS, bool NVFP4>
static __global__ __launch_bounds__(WARPS * LM_WARP_LANES, 1)
void LmGemmFp4Kernel(__grid_constant__ const LmGemmArguments args, LmTileSource source_a, LmTileSource source_b, bool grouped)
{
	static_assert(LmTileKIsSwizzleable(TILE_K,4u),"FP4 TILE_K must be a whole swizzle span; 128 elements is only 64 bytes");
	static_assert(LmPipelineSharedBytes(TILE_M,TILE_N,TILE_K,STAGES,4u) <= 131072u,"tile does not fit shared memory");
	static_assert(TILE_M % LM_MMA4_M == 0u && TILE_N % (WARPS * LM_MMA4_N) == 0u,"tile is not a whole number of mma fragments");
	__shared__ __align__(LM_TMA_ALIGNMENT_BYTES) uint8_t stage_a[STAGES][LmTileBytes(TILE_M,TILE_K,4u)];
	__shared__ __align__(LM_TMA_ALIGNMENT_BYTES) uint8_t stage_b[STAGES][LmTileBytes(TILE_N,TILE_K,4u)];
	__shared__ __align__(8) uint64_t barrier[STAGES];
	const uint32_t count = (TILE_M / LM_MMA4_M) * (TILE_N / WARPS / LM_MMA4_N);
	const uint32_t neuron_tiles = (args.output_dimension + TILE_N - 1u) / TILE_N;
	const uint32_t k_tiles = args.input_dimension / TILE_K;
	const uint32_t scales_per_tile = TILE_K / (NVFP4 ? LM_MMA4_NVFP4_GROUP : LM_MMA4_MXFP4_GROUP);
	float total[(TILE_M / LM_MMA4_M) * (TILE_N / WARPS / LM_MMA4_N)][4];
	uint32_t warp = threadIdx.x / LM_WARP_LANES,lane = threadIdx.x % LM_WARP_LANES;
	uint32_t tile,group,in_group,row_base,row_limit,neuron_base,k,stage,ahead,total_tiles;
	LmPipelineInitialise<STAGES>(barrier,WARPS * LM_WARP_LANES);
	total_tiles = LmTotalTiles(args.group_tile_prefix,args.group_count);
	for (tile = blockIdx.x; tile < total_tiles; tile += gridDim.x)
	{
		group = LmGroupOfTile(args.group_tile_prefix,args.group_count,tile);
		in_group = tile - args.group_tile_prefix[group];
		row_base = args.group_row_offset[group] + ((in_group / neuron_tiles) * TILE_M);
		row_limit = args.group_row_offset[group + 1u];
		neuron_base = (in_group % neuron_tiles) * TILE_N;
		LmGemmZero(total,count);
		for (stage = 0u; stage + 1u < STAGES && stage < k_tiles; ++stage)
			LmPipelineProduce(&source_a,&source_b,stage_a[stage],stage_b[stage],&barrier[stage],row_base,neuron_base,stage * (TILE_K / 2u),group,grouped);
		for (k = 0u; k < k_tiles; ++k)
		{
			stage = LmPipelineStage(k,STAGES);
			ahead = LmPipelineAhead(k,STAGES);
			if ( ahead < k_tiles )
				LmPipelineProduce(&source_a,&source_b,stage_a[ahead % STAGES],stage_b[ahead % STAGES],&barrier[ahead % STAGES],row_base,neuron_base,ahead * (TILE_K / 2u),group,grouped);
			LmMbarrierWait(&barrier[stage],LmPipelinePhase(k,STAGES));
			LmGemmConsume4<TILE_M,TILE_N,TILE_K,WARPS,NVFP4>(total,stage_a[stage],stage_b[stage],
				(const uint32_t *)args.scale_a + (k * scales_per_tile),
				(const uint32_t *)args.scale_b + (k * scales_per_tile),warp,lane);
			__syncthreads();
		}
		LmGemmStore<TILE_M,TILE_N,LM_MMA4_N,WARPS>(args,total,count,row_base,row_limit,neuron_base,warp,lane);
		__syncthreads();
	}
	LmPipelineRelease<STAGES>(barrier);
}
