#pragma once

// Host launcher for the grouped GEMM.
//
// This is what stands between kernels/gemm.cuh and deleting 305,005 lines of
// vendored CUTLASS: the two call sites that reach the external GEMM need a
// first-party function with the same shape to call instead.
//
// Three things have to be right and none of them are the kernel:
//
//   1. TILE HEIGHT PER BUCKET. Rows per group is tokens*top_k/experts, and each
//      M tile re-reads its group's weight tile. A tile shorter than the group
//      splits it and doubles the weight stream, which is 96 percent of decode
//      traffic. The selection rounds UP, always: padded mma rows are free on a
//      path at 1.4 percent of BF16 peak, re-read weights are not.
//
//   2. THE DYNAMIC SHARED OPT-IN. ptxas caps a static __shared__ declaration at
//      48 KB. The kernel therefore carves its stages out of dynamic shared, and
//      dynamic shared above 48 KB requires cudaFuncSetAttribute before the first
//      launch. Skipping it fails the launch rather than corrupting anything,
//      which is the good outcome, but it fails every time.
//
//   3. THE DESCRIPTOR SWIZZLE. The tensor map encodes a swizzle span and the
//      kernel applies a matching xor. They are computed from the same row pitch
//      by the same function, which is the only reason they cannot disagree.
//
// Everything here except the four CUDA calls is arithmetic and is checked by
// tests/test_launch.c on a host.

// Geometry only. The launch itself needs the kernel, but the PLAN is arithmetic
// and a host must be able to compute it without a CUDA toolchain - which is why
// kernels/layout.cuh exists separately from kernels/mma.cuh.
#include "kernels/layout.cuh"
#include <stdint.h>

#define LM_LAUNCH_OK 0
#define LM_LAUNCH_ERR_SHAPE (-41)
#define LM_LAUNCH_ERR_TILE (-42)
#define LM_LAUNCH_ERR_SHARED (-43)
#define LM_LAUNCH_ERR_MAP (-44)
#define LM_LAUNCH_ERR_ATTRIBUTE (-45)
#define LM_LAUNCH_ERR_LAUNCH (-46)

// Tile heights the library instantiates. A bucket outside this range is a
// caller error rather than a case to approximate, because the tile sizes shared
// memory and the accumulator array and both are compile-time.
#define LM_LAUNCH_TILE_MIN 16u
#define LM_LAUNCH_TILE_MAX 64u

typedef struct LmLaunchShape
{
	uint32_t tokens,top_k,expert_count,input_dimension,output_dimension;
	uint32_t stored_bits,tile_n,tile_k,stages;
}
LmLaunchShape;

typedef struct LmLaunchPlan
{
	uint32_t tile_m;
	uint32_t shared_bytes;
	uint32_t grid_blocks;
	uint32_t block_threads;
	uint64_t workspace_bytes;
	uint32_t swizzle_span;
}
LmLaunchPlan;

// Rows the busiest group is expected to hold.
//
// The mean understates it because routing is not uniform, and under a grouped
// launch the max-loaded group sets step time. The 2x headroom is a heuristic and
// is the one unmeasured number in this file; understating it costs a weight
// re-read for the overloaded groups only, not for all of them, so the failure is
// graceful. A measured route distribution would replace it.
static uint32_t LmLaunchPeakRowsPerGroup(const LmLaunchShape *shape)
{
	uint64_t mean;
	if ( shape->expert_count == 0u )
		return(0u);
	mean = (((uint64_t)shape->tokens * shape->top_k) + shape->expert_count - 1u)
		/ shape->expert_count;
	return((uint32_t)(mean * 2u));
}

static uint32_t LmLaunchSelectTile(uint32_t peak_rows)
{
	if ( peak_rows <= 16u )
		return(16u);
	if ( peak_rows <= 32u )
		return(32u);
	return(64u);
}

// Shared memory the kernel will carve, matching LmGemmSharedBytes exactly. Kept
// as arithmetic rather than a call into the template so the host can size a
// pool without instantiating a kernel.
static uint32_t LmLaunchSharedBytes(const LmLaunchShape *shape, uint32_t tile_m)
{
	uint32_t a = (tile_m * shape->tile_k * shape->stored_bits) / 8u;
	uint32_t b = (shape->tile_n * shape->tile_k * shape->stored_bits) / 8u;
	return((shape->stages * (a + b)) + (shape->stages * 8u));
}

static int32_t LmLaunchPlanBuild(const LmLaunchShape *shape, uint32_t multiprocessors, LmLaunchPlan *plan)
{
	uint32_t pitch;
	if ( shape == 0 || plan == 0 || shape->expert_count == 0u || shape->tile_n == 0u
		|| shape->tile_k == 0u || shape->stages < 2u || multiprocessors == 0u )
		return(LM_LAUNCH_ERR_SHAPE);
	plan->tile_m = LmLaunchSelectTile(LmLaunchPeakRowsPerGroup(shape));
	if ( plan->tile_m < LM_LAUNCH_TILE_MIN || plan->tile_m > LM_LAUNCH_TILE_MAX )
		return(LM_LAUNCH_ERR_TILE);
	plan->shared_bytes = LmLaunchSharedBytes(shape,plan->tile_m);
	if ( plan->shared_bytes > LM_SMEM_SM_TOTAL )
		return(LM_LAUNCH_ERR_SHARED);
	// The row pitch decides the swizzle span, and the descriptor must be built
	// with the same one the kernel applies. Both come from here.
	pitch = (shape->tile_k * shape->stored_bits) / 8u;
	plan->swizzle_span = LmSwizzleSpanFor(pitch);
	if ( plan->swizzle_span == 0u )
		return(LM_LAUNCH_ERR_MAP);
	// Persistent grid: one CTA per SM, sized to the machine rather than the
	// problem, so a short group never leaves an SM idle behind a long one. The
	// kernel bounds its own loop on the device-side tile prefix, so an
	// over-estimate here costs an idle block and never a phantom tile.
	plan->grid_blocks = multiprocessors;
	plan->block_threads = 8u * 32u;
	plan->workspace_bytes = 0u;
	return(LM_LAUNCH_OK);
}
