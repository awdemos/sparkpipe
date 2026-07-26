// Per-token-bucket kernel selection for the grouped GEMM.
//
// The B12x AOT path compiles one kernel per token bucket and picks a backend per
// bucket from a build-time benchmark; docs/GLM52_SM121_B12X_AOT_RUNTIME.md
// records "selected backend per token bucket, timing record" in its manifest and
// fails closed on a bucket it was not generated for. That per-shape
// specialisation, not any single kernel, is what a replacement has to preserve.
// This tool is that mechanism for a first-party kernel.
//
// Two phases. Phase one enumerates the configuration space and prunes it against
// shared memory and register limits - arithmetic, no GPU, runs anywhere, and is
// what this file can be tested on. Phase two consumes measured timings and emits
// the selection table and manifest; the measurement itself must happen on the
// ring and is not simulated here. A config that survives phase one is a
// candidate, never a winner: the cost model orders the search, measurement
// decides it.
//
// Nothing here invents a timing. Absent a measurement file the tool emits the
// candidate set and exits non-zero rather than emitting a table.

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define AUTOTUNE_MAX_CONFIGS 256
#define AUTOTUNE_MAX_BUCKETS 32
#define AUTOTUNE_SMEM_BYTES_PER_SM 131072u
#define AUTOTUNE_REGISTERS_PER_THREAD 255u
#define AUTOTUNE_WARP_LANES 32u
#define AUTOTUNE_MMA_M 16u
#define AUTOTUNE_MMA_N 8u
#define AUTOTUNE_MMA_K 32u
#define AUTOTUNE_SCALE_BLOCK_K 128u
#define AUTOTUNE_EFFECTIVE_BANDWIDTH_GBPS 218.0
#define AUTOTUNE_STREAMING_MULTIPROCESSORS 48u

typedef struct autotune_config
{
	uint32_t tile_m,tile_n,tile_k,stages,warps;
	uint32_t smem_bytes,registers,accumulators;
	double predicted_ms;
}
autotune_config_t;

typedef struct autotune_problem
{
	uint32_t tokens,top_k,expert_count,input_dimension,output_dimension,weight_bits;
}
autotune_problem_t;

// Rows a single expert receives on average. This is the number that decides
// whether a tall M tile is padding: at 256 experts and top-8, B128 gives 4.
static uint32_t autotune_rows_per_group(const autotune_problem_t *problem)
{
	uint32_t routed_rows,groups_touched;
	routed_rows = problem->tokens * problem->top_k;
	groups_touched = routed_rows < problem->expert_count ? routed_rows : problem->expert_count;
	if ( groups_touched == 0 )
		return(1);
	return((routed_rows + groups_touched - 1) / groups_touched);
}

static uint32_t autotune_smem_bytes(const autotune_config_t *config)
{
	uint32_t per_stage;
	// A tile, B tile, per-row A scales, one B scale (ScaleGranularityN == TILE_N)
	per_stage = (config->tile_m * config->tile_k) + (config->tile_n * config->tile_k)
		+ (config->tile_m * 4u) + 4u;
	// two mbarriers per stage, eight bytes each
	return((config->stages * per_stage) + (config->stages * 16u));
}

// Accumulators are held twice - a running fp32 total and a per-K-tile partial
// that the block scale folds in - plus operand fragments and addressing. The
// estimate is deliberately conservative; ptxas is the authority and the
// generated table records its actual count.
static uint32_t autotune_registers(const autotune_config_t *config, uint32_t *accumulators_out)
{
	uint32_t m_fragments,n_fragments,accumulators;
	m_fragments = config->tile_m / AUTOTUNE_MMA_M;
	n_fragments = config->tile_n / config->warps / AUTOTUNE_MMA_N;
	accumulators = m_fragments * n_fragments;
	*accumulators_out = accumulators;
	return((accumulators * 4u * 2u) + 6u + 24u);
}

static int32_t autotune_config_is_shaped(const autotune_config_t *config)
{
	if ( config->tile_k != AUTOTUNE_SCALE_BLOCK_K )
		return(-1);
	if ( (config->tile_m % AUTOTUNE_MMA_M) != 0 )
		return(-2);
	if ( (config->tile_n % (config->warps * AUTOTUNE_MMA_N)) != 0 )
		return(-3);
	if ( (config->tile_k % AUTOTUNE_MMA_K) != 0 )
		return(-4);
	if ( config->stages < 2 )
		return(-5);
	if ( config->warps < 1 || config->warps > 32 )
		return(-6);
	return(0);
}

static int32_t autotune_config_fits(autotune_config_t *config)
{
	int32_t shaped;
	shaped = autotune_config_is_shaped(config);
	if ( shaped != 0 )
		return(shaped);
	config->smem_bytes = autotune_smem_bytes(config);
	config->registers = autotune_registers(config,&config->accumulators);
	if ( config->smem_bytes > AUTOTUNE_SMEM_BYTES_PER_SM )
		return(-7);
	if ( config->registers > AUTOTUNE_REGISTERS_PER_THREAD )
		return(-8);
	return(0);
}

// Bytes this configuration moves for the whole grouped problem. Weights dominate
// and are read once per (group, neuron tile) regardless of tile_m, so a taller M
// tile does not reduce weight traffic - it only reduces the activation re-read,
// which is the smaller term. That asymmetry is why the cost model is worth
// having before any measurement.
static double autotune_predicted_bytes(const autotune_config_t *config, const autotune_problem_t *problem)
{
	double weight_bytes,activation_bytes,groups,m_tiles,rows;
	groups = (double)(problem->tokens * problem->top_k < problem->expert_count
		? problem->tokens * problem->top_k : problem->expert_count);
	rows = (double)autotune_rows_per_group(problem);
	m_tiles = (rows + (double)config->tile_m - 1.0) / (double)config->tile_m;
	if ( m_tiles < 1.0 )
		m_tiles = 1.0;
	weight_bytes = groups * (double)problem->input_dimension * (double)problem->output_dimension
		* ((double)problem->weight_bits / 8.0);
	activation_bytes = groups * m_tiles * (double)config->tile_m * (double)problem->input_dimension
		* ((double)problem->output_dimension / (double)config->tile_n);
	return(weight_bytes + activation_bytes);
}

static double autotune_predicted_ms(const autotune_config_t *config, const autotune_problem_t *problem)
{
	double bytes;
	bytes = autotune_predicted_bytes(config,problem);
	return((bytes / (AUTOTUNE_EFFECTIVE_BANDWIDTH_GBPS * 1.0e9)) * 1000.0);
}

static int32_t autotune_enumerate(autotune_config_t *configs, int32_t capacity, const autotune_problem_t *problem)
{
	static const uint32_t tile_m_set[] = { 16u, 32u, 64u, 128u };
	static const uint32_t tile_n_set[] = { 64u, 128u, 256u };
	static const uint32_t stage_set[] = { 2u, 3u, 4u, 5u, 6u, 8u };
	static const uint32_t warp_set[] = { 4u, 8u, 16u };
	autotune_config_t candidate;
	int32_t count = 0,m,n,s,w;
	for (m = 0; m < 4; ++m)
		for (n = 0; n < 3; ++n)
			for (s = 0; s < 6; ++s)
				for (w = 0; w < 3; ++w)
				{
					memset(&candidate,0,sizeof(candidate));
					candidate.tile_m = tile_m_set[m];
					candidate.tile_n = tile_n_set[n];
					candidate.tile_k = AUTOTUNE_SCALE_BLOCK_K;
					candidate.stages = stage_set[s];
					candidate.warps = warp_set[w];
					if ( autotune_config_fits(&candidate) != 0 )
						continue;
					if ( count >= capacity )
						return(count);
					candidate.predicted_ms = autotune_predicted_ms(&candidate,problem);
					configs[count++] = candidate;
				}
	return(count);
}

static void autotune_sort_by_prediction(autotune_config_t *configs, int32_t count)
{
	autotune_config_t swap;
	int32_t i,j;
	// count is bounded by AUTOTUNE_MAX_CONFIGS and this runs once per bucket at
	// build time; an insertion sort is the right shape for a list this small.
	for (i = 1; i < count; ++i)
	{
		swap = configs[i];
		j = i - 1;
		while ( j >= 0 && configs[j].predicted_ms > swap.predicted_ms )
		{
			configs[j + 1] = configs[j];
			j = j - 1;
		}
		configs[j + 1] = swap;
	}
}

static void autotune_report_bucket(const autotune_problem_t *problem, const autotune_config_t *configs, int32_t count, int32_t show)
{
	int32_t i;
	printf("bucket tokens=%-5u rows/group=%-3u candidates=%-3d\n",
		problem->tokens,autotune_rows_per_group(problem),count);
	for (i = 0; i < count && i < show; ++i)
		printf("    m=%-4u n=%-4u k=%-4u stages=%-2u warps=%-3u smem=%6u B  regs~%-4u acc=%-3u  pred=%8.3f ms\n",
			configs[i].tile_m,configs[i].tile_n,configs[i].tile_k,configs[i].stages,
			configs[i].warps,configs[i].smem_bytes,configs[i].registers,
			configs[i].accumulators,configs[i].predicted_ms);
}

int32_t main(int32_t argc, char **argv)
{
	static const uint32_t bucket_set[] = { 1u, 2u, 4u, 8u, 16u, 32u, 64u, 128u, 256u, 512u, 1024u };
	autotune_config_t configs[AUTOTUNE_MAX_CONFIGS];
	autotune_problem_t problem;
	int32_t bucket,count,total_candidates = 0;
	(void)argc;
	(void)argv;
	printf("GLM 5.2 routed MoE w1, NVFP4 weights, 256 experts top-8, K=6144 N=4096\n");
	printf("smem limit %u B, register limit %u/thread, effective BW %.0f GB/s\n\n",
		AUTOTUNE_SMEM_BYTES_PER_SM,AUTOTUNE_REGISTERS_PER_THREAD,AUTOTUNE_EFFECTIVE_BANDWIDTH_GBPS);
	for (bucket = 0; bucket < 11; ++bucket)
	{
		memset(&problem,0,sizeof(problem));
		problem.tokens = bucket_set[bucket];
		problem.top_k = 8u;
		problem.expert_count = 256u;
		problem.input_dimension = 6144u;
		problem.output_dimension = 4096u;
		problem.weight_bits = 4u;
		count = autotune_enumerate(configs,AUTOTUNE_MAX_CONFIGS,&problem);
		autotune_sort_by_prediction(configs,count);
		autotune_report_bucket(&problem,configs,count,3);
		total_candidates += count;
	}
	printf("\n%d candidate configurations across %d buckets survived resource pruning.\n",
		total_candidates,11);
	printf("NO TIMING DATA. Selection requires measurement on the ring; this tool\n");
	printf("emits candidates only and does not guess a winner.\n");
	return(1);
}
