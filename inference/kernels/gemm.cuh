#pragma once

// The GEMM. One body, one mma, every format, every model family.
//
// A dense linear is a grouped GEMM with one group, and every weight format is a
// decoder into the same BF16 fragment. Those two facts are why this file is one
// kernel where the old tree had seven - a CUTLASS dense call, a CUTLASS grouped
// call, generated B12x kernels, a reference, two BF16 tile policies behind a
// macro, and an FP8 tile excluded from every build. Four of the seven never
// executed.
//
// WHY EVERYTHING DECODES TO BF16. GB10's BF16 ridge is 573 FLOP/byte. Decode
// arithmetic intensity is 8 at B128 and 64 at B1024, so the kernel needs between
// 1.4 and 11 percent of BF16 peak. Unpacking INT7 costs about 2 percent of the
// CUDA cores. Compute is free by roughly sixty times, and the only number that
// matters is bytes crossing the bus.
//
// So a format's job is to be narrow in memory and to hand back a BF16 register.
// Nothing else varies. There is no integer accumulator, no block-scaled mma
// variant, no rescale pass, and no second code path - all of which existed here
// two commits ago and all of which were buying arithmetic on a machine with
// arithmetic to spare.
//
// WHAT A FORMAT MUST PROVIDE. kStoredBits, the four fragment coordinate helpers,
// and Fragment(). That is the whole interface, and it is why an entropy coder
// drops in as a replacement for one function rather than as a second kernel.
//
// TILE HEIGHT. Rows per group at decode is batch * top_k / experts, so it grows
// with the batch while the tile height is compile-time. Each M tile re-reads its
// group's weight tile, so the moment rows exceed TILE_M every group splits and
// the weight stream doubles. TILE_M is a template parameter selected per token
// bucket, and rounding is one-directional: up wastes mma throughput on padded
// rows, which is free here; down costs bandwidth, which is not.

#include "inference/kernels/formats/bf16.cuh"
#include "inference/kernels/tile.cuh"
#include <stdint.h>

// Bytes the launcher must request and pass. Constexpr so a host can compute it
// without a device query and so it cannot drift from what the kernel carves.
template<class Format, uint32_t TILE_M, uint32_t TILE_N, uint32_t TILE_K, uint32_t STAGES>
static __host__ __device__ constexpr uint32_t LmGemmSharedBytes(void)
{
	return((STAGES * LmTileBytes(TILE_M,TILE_K,Format::kStoredBits))
		+ (STAGES * LmTileBytes(TILE_N,TILE_K,Format::kStoredBits))
		+ (STAGES * 8u));
}

struct LmGemmArguments
{
	const void *tensor_map_a;
	const void *tensor_map_b;
	const float *scale_a;
	const float *scale_b;
	// The weight-only path: E8M0 codes as shipped, [group][neuron][k_group]
	// row-major with scale_groups codes per neuron. Pre-expanding to FP32 costs
	// 17.6 percent of expert bandwidth for a value exp2f decodes in one
	// instruction; the checkpoint's bytes stream as the checkpoint packed them.
	const uint8_t *scale_b_e8m0;
	uint32_t scale_groups;
	const uint32_t *group_row_offset;
	const uint32_t *group_tile_prefix;
	void *output_bf16;
	uint32_t group_count;
	uint32_t input_dimension;
	uint32_t output_dimension;
};

// Accumulate one staged K tile.
//
// The scale is fetched once per fragment rather than once per element: a
// fragment spans two adjacent k, and the scale group is at least 16, so both
// halves always share it. That is asserted rather than assumed.
//
// Formats that dequantise for free hand back raw BF16 bit patterns holding
// code + bias, and the correction is applied here with the multiply that had to
// happen anyway - (v - bias) * scale becomes one fma against a precomputed
// -bias*scale. Formats already in a real numeric form skip it. Both branches are
// on a compile-time constant, so only one exists in any instantiation.
template<class Format, uint32_t TILE_M, uint32_t TILE_N, uint32_t TILE_K, uint32_t WARPS>
static __device__ void LmGemmConsume(float (*total)[4], const uint8_t *stage_a, const uint8_t *stage_b, const float *scale_a, const float *scale_b, uint32_t warp, uint32_t lane)
{
	static_assert(Format::kScaleGroup == 0u || Format::kScaleGroup >= 2u,
		"a scale group below 2 would split a fragment's two adjacent k values");
	const uint32_t m_frags = TILE_M / Format::kMmaM;
	const uint32_t n_frags = TILE_N / WARPS / Format::kMmaN;
	const uint32_t pitch = LmTileBytes(1u,TILE_K,Format::kStoredBits);
	const uint32_t steps = TILE_K / Format::kMmaK;
	uint32_t step,mi,ni,neuron,k_base,reg;
	uint32_t a[4],b[2];
	for (step = 0u; step < steps; ++step)
	{
		k_base = step * Format::kMmaK;
		for (mi = 0u; mi < m_frags; ++mi)
		{
			for (reg = 0u; reg < 4u; ++reg)
				a[reg] = Format::Fragment(stage_a,
					(mi * Format::kMmaM) + Format::OperandARow(lane,reg),
					k_base + Format::OperandAK(lane,reg),pitch,
					scale_a[(mi * Format::kMmaM) + Format::OperandARow(lane,reg)]);
			for (ni = 0u; ni < n_frags; ++ni)
			{
				neuron = (warp * (TILE_N / WARPS)) + (ni * Format::kMmaN);
				for (reg = 0u; reg < 2u; ++reg)
					b[reg] = Format::Fragment(stage_b,
						neuron + Format::OperandBRow(lane),
						k_base + Format::OperandBK(lane,reg),pitch,
						scale_b[neuron + Format::OperandBRow(lane)]);
				LmMmaBf16(total[(mi * n_frags) + ni],a,b);
			}
		}
	}
}

// Rows past a group's own count are dropped rather than written. The tile height
// covers the busiest group, so most groups have a ragged tail and this is the
// steady state, not an edge case.
template<class Format, uint32_t TILE_M, uint32_t TILE_N, uint32_t WARPS>
static __device__ void LmGemmStore(const LmGemmArguments &args, float (*total)[4], uint32_t count, uint32_t row_base, uint32_t row_limit, uint32_t neuron_base, uint32_t warp, uint32_t lane)
{
	const uint32_t n_frags = TILE_N / WARPS / Format::kMmaN;
	uint16_t *output = (uint16_t *)args.output_bf16;
	uint32_t i,e,mi,ni,row,column;
	for (i = 0u; i < count; ++i)
	{
		mi = i / n_frags;
		ni = i % n_frags;
		for (e = 0u; e < 4u; ++e)
		{
			row = row_base + (mi * Format::kMmaM) + LmMmaAccumulatorRow(lane,e);
			column = neuron_base + (warp * (TILE_N / WARPS)) + (ni * Format::kMmaN)
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

// The persistent grid walks tiles strided by gridDim, so it is sized to the
// machine rather than the problem and a short group never leaves an SM idle
// behind a long one. The tile total comes from the device-side prefix, never a
// host estimate: an estimate from average rows launches tiles for groups that
// received none, and each still streams a full weight tile before its stores
// are rejected.
//
// Not static: a model's unity.cu names this in an explicit instantiation, and
// explicit instantiation of an internal symbol is ill-formed.
template<class Format, uint32_t TILE_M, uint32_t TILE_N, uint32_t TILE_K, uint32_t STAGES, uint32_t WARPS>
__global__ __launch_bounds__(WARPS * LM_WARP_LANES, 1)
void LmGemmKernel(__grid_constant__ const LmGemmArguments args, LmTileSource source_a, LmTileSource source_b, bool grouped)
{
	static_assert(LmTileKIsSwizzleable(TILE_K,Format::kStoredBits),"no swizzle span divides this row pitch; see the table in kernels/tile.cuh");
	// Dynamic rather than static shared memory. ptxas caps a static __shared__
	// declaration at 48 KB - "uses too much shared data (0xc000 max)" - which
	// excludes BF16 and INT7 at any useful tile. Dynamic shared has no compile
	// time cap; the launcher opts in with
	// cudaFuncSetAttribute(cudaFuncAttributeMaxDynamicSharedMemorySize) and
	// passes the size at launch. LmGemmSharedBytes below is what it must pass,
	// and the assert here is against the SM total rather than the static limit.
	static_assert(LmPipelineSharedBytes(TILE_M,TILE_N,TILE_K,STAGES,Format::kStoredBits) <= LM_SMEM_SM_TOTAL,"tile exceeds the shared memory an SM has");
	static_assert(TILE_M % Format::kMmaM == 0u && TILE_N % (WARPS * Format::kMmaN) == 0u,"tile is not a whole number of mma fragments");
	static_assert(TILE_K % Format::kMmaK == 0u,"tile depth is not a whole number of mma steps");
	extern __shared__ __align__(LM_TMA_ALIGNMENT_BYTES) uint8_t lm_shared[];
	const uint32_t a_bytes = LmTileBytes(TILE_M,TILE_K,Format::kStoredBits);
	const uint32_t b_bytes = LmTileBytes(TILE_N,TILE_K,Format::kStoredBits);
	uint8_t (*stage_a)[LmTileBytes(TILE_M,TILE_K,Format::kStoredBits)] =
		(uint8_t (*)[LmTileBytes(TILE_M,TILE_K,Format::kStoredBits)])lm_shared;
	uint8_t (*stage_b)[LmTileBytes(TILE_N,TILE_K,Format::kStoredBits)] =
		(uint8_t (*)[LmTileBytes(TILE_N,TILE_K,Format::kStoredBits)])(lm_shared + (STAGES * a_bytes));
	uint64_t *barrier = (uint64_t *)(lm_shared + (STAGES * (a_bytes + b_bytes)));
	const uint32_t count = (TILE_M / Format::kMmaM) * (TILE_N / WARPS / Format::kMmaN);
	const uint32_t neuron_tiles = (args.output_dimension + TILE_N - 1u) / TILE_N;
	const uint32_t k_tiles = args.input_dimension / TILE_K;
	float total[(TILE_M / Format::kMmaM) * (TILE_N / WARPS / Format::kMmaN)][4];
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
			LmPipelineProduce(&source_a,&source_b,stage_a[stage],stage_b[stage],&barrier[stage],row_base,neuron_base,stage,group,grouped);
		for (k = 0u; k < k_tiles; ++k)
		{
			stage = LmPipelineStage(k,STAGES);
			ahead = LmPipelineAhead(k,STAGES);
			if ( ahead < k_tiles )
				LmPipelineProduce(&source_a,&source_b,stage_a[ahead % STAGES],stage_b[ahead % STAGES],&barrier[ahead % STAGES],row_base,neuron_base,ahead,group,grouped);
			LmMbarrierWait(&barrier[stage],LmPipelinePhase(k,STAGES));
			LmGemmConsume<Format,TILE_M,TILE_N,TILE_K,WARPS>(total,stage_a[stage],stage_b[stage],
				args.scale_a + (k * (TILE_K / (Format::kScaleGroup ? Format::kScaleGroup : TILE_K))),
				args.scale_b + (k * (TILE_K / (Format::kScaleGroup ? Format::kScaleGroup : TILE_K))),
				warp,lane);
			__syncthreads();
		}
		LmGemmStore<Format,TILE_M,TILE_N,WARPS>(args,total,count,row_base,row_limit,neuron_base,warp,lane);
		__syncthreads();
	}
	LmPipelineRelease<STAGES>(barrier);
}

// -- weight-only: BF16 activations against a quantised weight stream ---------
//
// The recipe Kimi K3 ships: input_activations null, weights MXFP4 group 32.
// The activation operand is BF16 as produced, unquantised and unscaled; the
// weight operand decodes through its Format with an E8M0 code fetched from the
// scale plane as shipped. One kernel body again, because the only asymmetry a
// weight-only GEMM has is width and scale - the mma, the pipeline and the
// store are the symmetric kernel's, and a second copy of those is how the old
// tree came to hold seven GEMMs.

template<class FormatB, uint32_t TILE_M, uint32_t TILE_N, uint32_t TILE_K, uint32_t STAGES>
static __host__ __device__ constexpr uint32_t LmGemmWeightOnlySharedBytes(void)
{
	return((STAGES * LmTileBytes(TILE_M,TILE_K,LmBf16Format::kStoredBits))
		+ (STAGES * LmTileBytes(TILE_N,TILE_K,FormatB::kStoredBits))
		+ (STAGES * 8u));
}

// The scale fetch happens here, per fragment, from the E8M0 plane: the two
// adjacent k of a fragment always share a group (group >= 2, asserted), and
// the plane pointer arrives already offset to this tile's expert, neuron base
// and K tile, so the index is local rows by local groups.
template<class FormatB, uint32_t TILE_M, uint32_t TILE_N, uint32_t TILE_K, uint32_t WARPS>
static __device__ void LmGemmWeightOnlyConsume(float (*total)[4], const uint8_t *stage_a, const uint8_t *stage_b, const uint8_t *scale_tile, uint32_t scale_groups, uint32_t warp, uint32_t lane)
{
	static_assert(FormatB::kScaleGroup >= 2u,
		"a weight-only format carries a scale plane; a group below 2 splits a fragment");
	static_assert(FormatB::kMmaM == LmBf16Format::kMmaM
		&& FormatB::kMmaN == LmBf16Format::kMmaN
		&& FormatB::kMmaK == LmBf16Format::kMmaK,
		"both operands must agree on the mma the fragments feed");
	const uint32_t m_frags = TILE_M / LmBf16Format::kMmaM;
	const uint32_t n_frags = TILE_N / WARPS / LmBf16Format::kMmaN;
	const uint32_t pitch_a = LmTileBytes(1u,TILE_K,LmBf16Format::kStoredBits);
	const uint32_t pitch_b = LmTileBytes(1u,TILE_K,FormatB::kStoredBits);
	const uint32_t steps = TILE_K / LmBf16Format::kMmaK;
	uint32_t step,mi,ni,neuron,k_base,k_local,reg;
	uint32_t a[4],b[2];
	for (step = 0u; step < steps; ++step)
	{
		k_base = step * LmBf16Format::kMmaK;
		for (mi = 0u; mi < m_frags; ++mi)
		{
			for (reg = 0u; reg < 4u; ++reg)
				a[reg] = LmBf16Format::Fragment(stage_a,
					(mi * LmBf16Format::kMmaM) + LmBf16Format::OperandARow(lane,reg),
					k_base + LmBf16Format::OperandAK(lane,reg),pitch_a,0.0f);
			for (ni = 0u; ni < n_frags; ++ni)
			{
				neuron = (warp * (TILE_N / WARPS)) + (ni * FormatB::kMmaN);
				for (reg = 0u; reg < 2u; ++reg)
				{
					k_local = k_base + FormatB::OperandBK(lane,reg);
					b[reg] = FormatB::Fragment(stage_b,
						neuron + FormatB::OperandBRow(lane),k_local,pitch_b,
						FormatB::ScaleDecode(scale_tile[
							(((uint64_t)neuron + FormatB::OperandBRow(lane))
								* scale_groups)
							+ (k_local / FormatB::kScaleGroup)]));
				}
				LmMmaBf16(total[(mi * n_frags) + ni],a,b);
			}
		}
	}
}

template<class FormatB, uint32_t TILE_M, uint32_t TILE_N, uint32_t TILE_K, uint32_t STAGES, uint32_t WARPS>
__global__ __launch_bounds__(WARPS * LM_WARP_LANES, 1)
void LmGemmWeightOnlyKernel(__grid_constant__ const LmGemmArguments args, LmTileSource source_a, LmTileSource source_b, bool grouped)
{
	static_assert(LmTileKIsSwizzleable(TILE_K,LmBf16Format::kStoredBits),"the BF16 activation pitch must be a whole swizzle span");
	static_assert(LmTileKIsSwizzleable(TILE_K,FormatB::kStoredBits),"the weight pitch must be a whole swizzle span");
	static_assert(TILE_M % LmBf16Format::kMmaM == 0u && TILE_N % (WARPS * FormatB::kMmaN) == 0u,"tile is not a whole number of mma fragments");
	static_assert(TILE_K % LmBf16Format::kMmaK == 0u,"tile depth is not a whole number of mma steps");
	extern __shared__ __align__(LM_TMA_ALIGNMENT_BYTES) uint8_t lm_shared[];
	const uint32_t a_bytes = LmTileBytes(TILE_M,TILE_K,LmBf16Format::kStoredBits);
	const uint32_t b_bytes = LmTileBytes(TILE_N,TILE_K,FormatB::kStoredBits);
	uint8_t (*stage_a)[LmTileBytes(TILE_M,TILE_K,LmBf16Format::kStoredBits)] =
		(uint8_t (*)[LmTileBytes(TILE_M,TILE_K,LmBf16Format::kStoredBits)])lm_shared;
	uint8_t (*stage_b)[LmTileBytes(TILE_N,TILE_K,FormatB::kStoredBits)] =
		(uint8_t (*)[LmTileBytes(TILE_N,TILE_K,FormatB::kStoredBits)])(lm_shared + (STAGES * a_bytes));
	uint64_t *barrier = (uint64_t *)(lm_shared + (STAGES * (a_bytes + b_bytes)));
	const uint32_t count = (TILE_M / LmBf16Format::kMmaM) * (TILE_N / WARPS / LmBf16Format::kMmaN);
	const uint32_t neuron_tiles = (args.output_dimension + TILE_N - 1u) / TILE_N;
	const uint32_t k_tiles = args.input_dimension / TILE_K;
	const uint32_t tile_groups = TILE_K / FormatB::kScaleGroup;
	float total[(TILE_M / LmBf16Format::kMmaM) * (TILE_N / WARPS / LmBf16Format::kMmaN)][4];
	uint32_t warp = threadIdx.x / LM_WARP_LANES,lane = threadIdx.x % LM_WARP_LANES;
	uint32_t tile,group,in_group,row_base,row_limit,neuron_base,k,stage,ahead,total_tiles;
	const uint8_t *scale_plane;
	LmPipelineInitialise<STAGES>(barrier,WARPS * LM_WARP_LANES);
	total_tiles = LmTotalTiles(args.group_tile_prefix,args.group_count);
	for (tile = blockIdx.x; tile < total_tiles; tile += gridDim.x)
	{
		group = LmGroupOfTile(args.group_tile_prefix,args.group_count,tile);
		in_group = tile - args.group_tile_prefix[group];
		row_base = args.group_row_offset[group] + ((in_group / neuron_tiles) * TILE_M);
		row_limit = args.group_row_offset[group + 1u];
		neuron_base = (in_group % neuron_tiles) * TILE_N;
		// This tile's slice of the E8M0 plane: expert, then neuron base, in a
		// [group][neuron][k_group] layout whose row stride is args.scale_groups.
		scale_plane = args.scale_b_e8m0
			+ ((((uint64_t)group * args.output_dimension) + neuron_base)
				* args.scale_groups);
		LmGemmZero(total,count);
		for (stage = 0u; stage + 1u < STAGES && stage < k_tiles; ++stage)
			LmPipelineProduce(&source_a,&source_b,stage_a[stage],stage_b[stage],&barrier[stage],row_base,neuron_base,stage,group,grouped);
		for (k = 0u; k < k_tiles; ++k)
		{
			stage = LmPipelineStage(k,STAGES);
			ahead = LmPipelineAhead(k,STAGES);
			if ( ahead < k_tiles )
				LmPipelineProduce(&source_a,&source_b,stage_a[ahead % STAGES],stage_b[ahead % STAGES],&barrier[ahead % STAGES],row_base,neuron_base,ahead,group,grouped);
			LmMbarrierWait(&barrier[stage],LmPipelinePhase(k,STAGES));
			LmGemmWeightOnlyConsume<FormatB,TILE_M,TILE_N,TILE_K,WARPS>(total,
				stage_a[stage],stage_b[stage],
				scale_plane + ((uint64_t)k * tile_groups),args.scale_groups,
				warp,lane);
			__syncthreads();
		}
		LmGemmStore<LmBf16Format,TILE_M,TILE_N,WARPS>(args,total,count,row_base,row_limit,neuron_base,warp,lane);
		__syncthreads();
	}
	LmPipelineRelease<STAGES>(barrier);
}
