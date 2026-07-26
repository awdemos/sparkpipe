// Workspace layout for the first-party NVFP4 routed-MoE pipeline.
//
// Seven stages run between a normalised hidden state and a routed output:
// router top-k, packed-route build, activation quantise, w1 GEMM, SiLU-mul
// requantise, w2 GEMM, finalise. Each needs scratch, the sizes are all
// functions of (tokens, top_k, experts, hidden, intermediate), and getting one
// extent wrong overlaps two buffers - which corrupts data with no allocation
// error anywhere.
//
// The arithmetic lives here, separate from any CUDA, so it can be checked on a
// host. tests/test_group_gemm_workspace.c asserts every region is aligned, that
// regions do not overlap, and that the total is what the sum of the parts says.
//
// NVFP4 EXTENTS. Payloads are 4-bit, so element counts halve into bytes; scales
// are UE4M3, one byte per NVFP4_GROUP_SIZE elements. Both conversions are the
// kind that look right at a glance and are wrong by a factor of two.

#ifndef SPARK_LM_GROUP_GEMM_WORKSPACE_H
#define SPARK_LM_GROUP_GEMM_WORKSPACE_H

#include <stdint.h>

#define SPARK_LM_WORKSPACE_ALIGNMENT 256u
#define SPARK_LM_WORKSPACE_NVFP4_GROUP 16u
#define SPARK_LM_WORKSPACE_REGION_COUNT 9u
#define SPARK_LM_WORKSPACE_NVFP4_TILE_K 256u
#define SPARK_LM_TENSOR_MAP_BITS_NVFP4_LOCAL 4u
// 128 KB of L1/shared per SM on GB10, per GB10_CUDA_COST_MODEL_CALIBRATION.md.
#define SPARK_LM_WORKSPACE_SHARED_LIMIT 131072u

#define SPARK_LM_WORKSPACE_REGION_PACKED_HIDDEN 0u
#define SPARK_LM_WORKSPACE_REGION_PACKED_HIDDEN_SCALE 1u
#define SPARK_LM_WORKSPACE_REGION_ROUTE_ROWS 2u
#define SPARK_LM_WORKSPACE_REGION_ROUTE_INDPTR 3u
#define SPARK_LM_WORKSPACE_REGION_GROUP_TILE_PREFIX 4u
#define SPARK_LM_WORKSPACE_REGION_GATE_UP_BF16 5u
#define SPARK_LM_WORKSPACE_REGION_INTERMEDIATE 6u
#define SPARK_LM_WORKSPACE_REGION_INTERMEDIATE_SCALE 7u
#define SPARK_LM_WORKSPACE_REGION_ROUTE_OUTPUT_BF16 8u

#define SPARK_LM_WORKSPACE_OK 0
#define SPARK_LM_WORKSPACE_ERR_NULL (-31)
#define SPARK_LM_WORKSPACE_ERR_SHAPE (-32)
#define SPARK_LM_WORKSPACE_ERR_GROUP (-33)
#define SPARK_LM_WORKSPACE_ERR_OVERFLOW (-34)
#define SPARK_LM_WORKSPACE_ERR_SHARED (-35)

typedef struct spark_lm_workspace_shape
{
	uint32_t tokens,top_k,expert_count,hidden_dimension,intermediate_dimension,tile_m,tile_n;
}
spark_lm_workspace_shape_t;

typedef struct spark_lm_workspace_layout
{
	uint64_t offset[SPARK_LM_WORKSPACE_REGION_COUNT];
	uint64_t bytes[SPARK_LM_WORKSPACE_REGION_COUNT];
	uint64_t total_bytes;
	uint64_t packed_rows;
	uint64_t total_tiles;
	uint64_t shared_bytes;
	uint32_t tile_m;
	uint32_t stages;
}
spark_lm_workspace_layout_t;

static uint64_t spark_lm_workspace_align_up(uint64_t value)
{
	return((value + (uint64_t)SPARK_LM_WORKSPACE_ALIGNMENT - 1u)
		& ~((uint64_t)SPARK_LM_WORKSPACE_ALIGNMENT - 1u));
}

// Rows in the packed layout: one per (token, route). Every downstream extent is
// a multiple of this, so it exists once.
static uint64_t spark_lm_workspace_packed_rows(const spark_lm_workspace_shape_t *shape)
{
	return((uint64_t)shape->tokens * (uint64_t)shape->top_k);
}

// Choose TILE_M for a token bucket.
//
// Rows per expert is tokens*top_k/experts, so it grows with the batch while
// TILE_M is a compile-time tile height. A fixed TILE_M=16 is exact through the
// point where rows reach 16 and then splits every expert into two M tiles - and
// because each M tile re-reads the expert's weight tile, that split doubles the
// weight stream, which is 96 percent of all traffic on this path. At GLM 5.2's
// 256 experts and top-8 that lands at B1024: rows/expert 32, two tiles, 2x the
// bytes. Selecting TILE_M per bucket is what keeps it at one.
//
// Rounding UP to the next tile height wastes MMA throughput on padded rows,
// which is free on a bandwidth-bound path. Rounding down would cost bandwidth,
// which is not. The asymmetry is why this only ever grows the tile.
static uint32_t spark_lm_workspace_select_tile_m(uint64_t rows_per_expert)
{
	if ( rows_per_expert <= 16u )
		return(16u);
	if ( rows_per_expert <= 32u )
		return(32u);
	if ( rows_per_expert <= 64u )
		return(64u);
	return(128u);
}

// Shared memory one CTA needs. NVFP4 payloads are 4-bit, so both tile buffers
// halve; the barriers are two 8-byte mbarriers per stage.
static uint64_t spark_lm_workspace_shared_bytes(uint32_t tile_m, uint32_t tile_n, uint32_t tile_k, uint32_t stages, uint32_t element_bits)
{
	uint64_t per_stage;
	per_stage = ((uint64_t)tile_m + (uint64_t)tile_n) * (uint64_t)tile_k
		* (uint64_t)element_bits / 8u;
	return((uint64_t)stages * (per_stage + 16u));
}

// Deepest pipeline that fits, given the tile height already chosen.
//
// This is not a tuning knob dressed up as a constraint: at TILE_M=128 with the
// four stages that were hardcoded, an NVFP4 tile needs 131,136 bytes against a
// 131,072 limit and the launch fails outright. Selecting stages jointly with the
// tile height also turns a constraint into a gain at the small end - a 16-row
// tile leaves room for six stages where a 128-row tile leaves room for three,
// and the weight stream is what the extra depth covers.
//
// Depth beyond what Little's Law needs buys nothing. At 218 GB/s effective and
// an estimated 400-600 ns latency the requirement is about 2.7 KB in flight per
// SM, and even two stages of an 18 KB tile clear that by an order of magnitude.
// The cap therefore exists to stop shared memory being spent for no reason, not
// because more would help.
static uint32_t spark_lm_workspace_select_stages(uint32_t tile_m, uint32_t tile_n, uint32_t tile_k, uint32_t element_bits, uint64_t shared_limit)
{
	static const uint32_t candidates[] = { 6u, 4u, 3u, 2u };
	uint32_t index;
	for (index = 0u; index < 4u; ++index)
		if ( spark_lm_workspace_shared_bytes(tile_m,tile_n,tile_k,candidates[index],element_bits) <= shared_limit )
			return(candidates[index]);
	return(0u);
}

// Rows the busiest expert is expected to hold. Routing is not uniform, so the
// mean understates the tile height a heavily loaded expert needs; the max-loaded
// group is what sets step time under a grouped launch. A 2x headroom factor is a
// heuristic and is the one number here that wants a measured route distribution
// behind it - the route-log collection already planned is what would supply it.
static uint64_t spark_lm_workspace_peak_rows_per_expert(const spark_lm_workspace_shape_t *shape)
{
	uint64_t mean_rows;
	mean_rows = (spark_lm_workspace_packed_rows(shape) + (uint64_t)shape->expert_count - 1u)
		/ (uint64_t)shape->expert_count;
	return(mean_rows * 2u);
}

// Grid tiles for the grouped GEMM: each expert contributes ceil(rows/TILE_M)
// M tiles times the N tile count. At decode most experts hold fewer rows than
// TILE_M, so this is dominated by the expert count, not the token count.
static uint64_t spark_lm_workspace_total_tiles_for_tile_m(const spark_lm_workspace_shape_t *shape, uint32_t tile_m, uint32_t output_dimension)
{
	uint64_t rows_per_expert,m_tiles,n_tiles;
	rows_per_expert = (spark_lm_workspace_packed_rows(shape) + shape->expert_count - 1u)
		/ (uint64_t)shape->expert_count;
	m_tiles = (rows_per_expert + (uint64_t)tile_m - 1u) / (uint64_t)tile_m;
	if ( m_tiles == 0u )
		m_tiles = 1u;
	n_tiles = ((uint64_t)output_dimension + (uint64_t)shape->tile_n - 1u) / (uint64_t)shape->tile_n;
	return((uint64_t)shape->expert_count * m_tiles * n_tiles);
}

static int32_t spark_lm_workspace_layout_build(const spark_lm_workspace_shape_t *shape, spark_lm_workspace_layout_t *layout)
{
	uint64_t rows,hidden_bytes,hidden_scales,intermediate_bytes,intermediate_scales,cursor;
	uint32_t region,effective;
	if ( shape == 0 || layout == 0 )
		return(SPARK_LM_WORKSPACE_ERR_NULL);
	if ( shape->tokens == 0 || shape->top_k == 0 || shape->expert_count == 0
		|| shape->hidden_dimension == 0 || shape->intermediate_dimension == 0
		|| shape->tile_n == 0 )
		return(SPARK_LM_WORKSPACE_ERR_SHAPE);
	// A 4-bit payload needs an even element count, and a UE4M3 scale needs the
	// group to divide the row, or the last group is partial and the GEMM reads
	// a scale that was never written.
	if ( (shape->hidden_dimension % SPARK_LM_WORKSPACE_NVFP4_GROUP) != 0u
		|| (shape->intermediate_dimension % SPARK_LM_WORKSPACE_NVFP4_GROUP) != 0u )
		return(SPARK_LM_WORKSPACE_ERR_GROUP);
	// tile_m == 0 means "choose for this bucket", which is what the launcher
	// passes. An explicit value is honoured so a sweep can pin it.
	effective = shape->tile_m != 0u
		? shape->tile_m
		: spark_lm_workspace_select_tile_m(spark_lm_workspace_peak_rows_per_expert(shape));
	layout->tile_m = effective;
	layout->stages = spark_lm_workspace_select_stages(effective,shape->tile_n,
		SPARK_LM_WORKSPACE_NVFP4_TILE_K,SPARK_LM_TENSOR_MAP_BITS_NVFP4_LOCAL,
		SPARK_LM_WORKSPACE_SHARED_LIMIT);
	if ( layout->stages == 0u )
		return(SPARK_LM_WORKSPACE_ERR_SHARED);
	layout->shared_bytes = spark_lm_workspace_shared_bytes(effective,shape->tile_n,
		SPARK_LM_WORKSPACE_NVFP4_TILE_K,layout->stages,
		SPARK_LM_TENSOR_MAP_BITS_NVFP4_LOCAL);
	rows = spark_lm_workspace_packed_rows(shape);
	if ( rows == 0u || rows > 0xffffffffu )
		return(SPARK_LM_WORKSPACE_ERR_OVERFLOW);
	hidden_bytes = rows * ((uint64_t)shape->hidden_dimension / 2u);
	hidden_scales = rows * ((uint64_t)shape->hidden_dimension / SPARK_LM_WORKSPACE_NVFP4_GROUP);
	intermediate_bytes = rows * ((uint64_t)shape->intermediate_dimension / 2u);
	intermediate_scales = rows * ((uint64_t)shape->intermediate_dimension / SPARK_LM_WORKSPACE_NVFP4_GROUP);
	layout->bytes[SPARK_LM_WORKSPACE_REGION_PACKED_HIDDEN] = hidden_bytes;
	layout->bytes[SPARK_LM_WORKSPACE_REGION_PACKED_HIDDEN_SCALE] = hidden_scales;
	layout->bytes[SPARK_LM_WORKSPACE_REGION_ROUTE_ROWS] = rows * sizeof(uint32_t);
	layout->bytes[SPARK_LM_WORKSPACE_REGION_ROUTE_INDPTR] = ((uint64_t)shape->expert_count + 1u) * sizeof(uint32_t);
	layout->bytes[SPARK_LM_WORKSPACE_REGION_GROUP_TILE_PREFIX] = ((uint64_t)shape->expert_count + 1u) * sizeof(uint32_t);
	// w1 emits gate and up together, bf16, hence the factor of two on each.
	layout->bytes[SPARK_LM_WORKSPACE_REGION_GATE_UP_BF16] = rows * (uint64_t)shape->intermediate_dimension * 2u * 2u;
	layout->bytes[SPARK_LM_WORKSPACE_REGION_INTERMEDIATE] = intermediate_bytes;
	layout->bytes[SPARK_LM_WORKSPACE_REGION_INTERMEDIATE_SCALE] = intermediate_scales;
	layout->bytes[SPARK_LM_WORKSPACE_REGION_ROUTE_OUTPUT_BF16] = rows * (uint64_t)shape->hidden_dimension * 2u;
	cursor = 0u;
	for (region = 0u; region < SPARK_LM_WORKSPACE_REGION_COUNT; ++region)
	{
		layout->offset[region] = cursor;
		cursor = spark_lm_workspace_align_up(cursor + layout->bytes[region]);
	}
	layout->total_bytes = cursor;
	layout->packed_rows = rows;
	layout->total_tiles = spark_lm_workspace_total_tiles_for_tile_m(shape,effective,
		shape->intermediate_dimension * 2u);
	return(SPARK_LM_WORKSPACE_OK);
}

#endif
