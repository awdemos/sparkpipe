#pragma once

// The route build: from the router's per-token expert choices to the packed,
// expert-major order the grouped GEMM streams.
//
// Nothing in the tree produced these arrays - every harness filled them by
// hand, which is how a driver came to be impossible: the top-k output lives on
// the device, and a host cannot pack what it cannot see without a sync on the
// hot path. This is the producer, and it is one kernel because the whole job
// is a counting sort over 896 buckets: count, prefix, scatter, with the
// counters in shared memory and the prefix serial - 896 additions is not a
// problem that needs a scan.
//
// THE ORDER WITHIN AN EXPERT IS WHATEVER THE ATOMICS SAY. The GEMM does not
// care - every packed row carries its source token - and demanding a stable
// order would buy determinism the finalize never reads.

#include "inference/kernels/mma.cuh"

// route_expert:      [routes]  the router's choice per (token, k)
// group_row_offset:  [experts + 1]  exclusive prefix of per-expert row counts
// route_packed_row:  [routes]  where (token, k) landed in the packed order
// route_source_token:[routes]  which token a packed row came from
// The two tile tables ride along: the layer knows the tile heights its two
// expert GEMMs will select (the planner's choice is a pure function of the
// worst-case rows, host-computable), so the same thread that writes the row
// offsets prices both tables and the launcher makes no prefix launch at all.
// Null tables skip the pricing, which is any caller that is not a MoE layer.
template<uint32_t THREADS, uint32_t EXPERTS>
__global__ __launch_bounds__(THREADS, 1)
void LmRouteBuildKernel(const uint32_t *__restrict__ route_expert, uint32_t routes, uint32_t top_k, uint32_t *__restrict__ group_row_offset, uint32_t *__restrict__ route_packed_row, uint32_t *__restrict__ route_source_token, uint32_t tile_m, uint32_t neuron_tiles_up, uint32_t *__restrict__ tile_prefix_up, uint32_t neuron_tiles_down, uint32_t *__restrict__ tile_prefix_down)
{
	__shared__ uint32_t count[EXPERTS];
	uint32_t index,expert,packed;
	for (index = threadIdx.x; index < EXPERTS; index += THREADS)
		count[index] = 0u;
	__syncthreads();
	for (index = threadIdx.x; index < routes; index += THREADS)
		atomicAdd(&count[route_expert[index]],1u);
	__syncthreads();
	// Serial exclusive prefix, one thread. After this, count[] holds each
	// expert's running cursor - the same array serves both jobs.
	if ( threadIdx.x == 0u )
	{
		uint32_t total = 0u,held;
		for (index = 0u; index < EXPERTS; ++index)
		{
			held = count[index];
			group_row_offset[index] = total;
			count[index] = total;
			total += held;
		}
		group_row_offset[EXPERTS] = total;
		if ( tile_prefix_up != 0 )
		{
			uint32_t up = 0u,down = 0u,rows,row_tiles;
			for (index = 0u; index < EXPERTS; ++index)
			{
				rows = group_row_offset[index + 1u] - group_row_offset[index];
				row_tiles = (rows + tile_m - 1u) / tile_m;
				tile_prefix_up[index] = up;
				tile_prefix_down[index] = down;
				up += row_tiles * neuron_tiles_up;
				down += row_tiles * neuron_tiles_down;
			}
			tile_prefix_up[EXPERTS] = up;
			tile_prefix_down[EXPERTS] = down;
		}
	}
	__syncthreads();
	for (index = threadIdx.x; index < routes; index += THREADS)
	{
		expert = route_expert[index];
		packed = atomicAdd(&count[expert],1u);
		route_packed_row[index] = packed;
		route_source_token[packed] = index / top_k;
	}
}
