// sparkpipe_glm52_batchplane_sim — integrated host simulation of the expert
// queue batch plane: real SparkGlm52ExpertQueue firings, real
// SparkGlm52JitKvPool fragment tiering, driven by longmem-shaped traffic
// (sequences of B8 shared-prefix exchanges) over a virtual bandwidth clock
// calibrated to the measured single-row decode anchor.
//
// One rank is simulated (6 MoE layers x 256 experts); all 13 ranks carry the
// identical load in steady state, so system committed throughput equals rank
// throughput and wave transit is 13x the rank pass. Time advances only by
// resource consumption: expert firings and attention reads spend the batch
// plane's bandwidth share; the JIT pool spends the NVMe budget on its own
// clock. Shared-prefix attention charges the shared portion of the DSA
// selection once per sequence wave instead of once per lane.
//
// Usage: sparkpipe_glm52_batchplane_sim sequences waves [share_milli] [rt_milli] [fire_rows] [quant4]
#include "sparkpipe/spark_glm52_expert_queue.h"
#include "sparkpipe/spark_glm52_jit_kv_pool.h"

#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BSIM_MAX_SEQUENCES 4096u
#define BSIM_LANES 8u
#define BSIM_TOPK 8u
#define BSIM_LAYERS 6u
#define BSIM_RANKS 13u
#define BSIM_EXPERTS 256u
#define BSIM_BW_EFF 174.0e9
#define BSIM_EXPERT_BYTES_8B 37750000.0
#define BSIM_EXPERT_BYTES_FP4 19600000.0
#define BSIM_LATENT_BYTES_PER_TOKEN 1152.0
#define BSIM_INDEX_BYTES_PER_TOKEN 32.0
#define BSIM_CONTEXT_TOKENS 8192u
#define BSIM_SELECT_TOKENS 2048u
#define BSIM_DELTA_FRAGMENTS 3u
#define BSIM_FRAGMENT_TOKENS 64u
#define BSIM_COMMIT_PER_WAVE 5.67
#define BSIM_KV_POOL_BYTES 29.0e9
#define BSIM_NVME_BPS 6.0e9

typedef struct
{
	uint32_t layer_index;
	uint32_t pending_participations;
	uint32_t wave_number;
	uint32_t fragment_base;
	uint64_t wave_start_ns;
	uint64_t transit_accum_ns;
	uint32_t transit_samples;
} bsim_sequence_t;

static SparkGlm52ExpertQueue bsim_queue;
static SparkGlm52JitKvPool bsim_pool;
static bsim_sequence_t bsim_sequences[BSIM_MAX_SEQUENCES];

static uint64_t bsim_route_hash(uint64_t a,uint64_t b,uint64_t c,uint64_t d)
{
	uint64_t h = (a * 0x9E3779B97F4A7C15u) ^ (b * 0xC2B2AE3D27D4EB4Fu) ^ (c * 0x165667B19E3779F9u) ^ (d * 0x27D4EB2F165667C5u);
	h ^= (h >> 29); h *= 0xBF58476D1CE4E5B9u; h ^= (h >> 32);
	return(h);
}

static void bsim_enqueue_layer(uint32_t sequence_index,uint64_t now_ns)
{
	bsim_sequence_t *sequence = &bsim_sequences[sequence_index];
	uint32_t lane_index,pick_index,expert_index;
	uint64_t row_id;
	sequence->pending_participations = (BSIM_LANES * BSIM_TOPK);
	for (lane_index=0u; lane_index<BSIM_LANES; lane_index++)
	{
		row_id = (((uint64_t)sequence_index << 8) | lane_index);
		for (pick_index=0u; pick_index<BSIM_TOPK; pick_index++)
		{
			expert_index = (uint32_t)(bsim_route_hash(sequence_index,sequence->wave_number,(uint64_t)sequence->layer_index << 8 | lane_index,pick_index) % BSIM_EXPERTS);
			if ( SparkGlm52ExpertQueueEnqueueRow(&bsim_queue,sequence->layer_index,expert_index,row_id,now_ns) != SPARK_STATUS_OK )
			{
				fprintf(stderr,"enqueue overflow seq=%u\n",sequence_index);
				exit(1);
			}
		}
	}
}

int main(int argc,char **argv)
{
	SparkGlm52ExpertQueueConfiguration queue_configuration;
	SparkGlm52JitKvPoolConfiguration pool_configuration;
	SparkGlm52ExpertQueueFiring firing;
	uint64_t fragment_bytes = (uint64_t)(BSIM_FRAGMENT_TOKENS * BSIM_LATENT_BYTES_PER_TOKEN * BSIM_LAYERS);
	uint32_t sequence_count = (argc > 1 ? (uint32_t)strtoul(argv[1],0,10) : 500u);
	uint32_t wave_target = (argc > 2 ? (uint32_t)strtoul(argv[2],0,10) : 12u);
	uint32_t share_milli = (argc > 3 ? (uint32_t)strtoul(argv[3],0,10) : 900u);
	uint32_t realtime_milli = (argc > 4 ? (uint32_t)strtoul(argv[4],0,10) : 250u);
	uint32_t fire_rows = (argc > 5 ? (uint32_t)strtoul(argv[5],0,10) : 512u);
	uint32_t quant4 = (argc > 6 ? (uint32_t)strtoul(argv[6],0,10) : 1u);
	double expert_bytes = (quant4 != 0u ? BSIM_EXPERT_BYTES_FP4 : BSIM_EXPERT_BYTES_8B);
	double bw = (BSIM_BW_EFF * (1000.0 - (double)realtime_milli) / 1000.0);
	double shared_fraction = ((double)share_milli / 1000.0);
	uint64_t now_ns = 0u,total_committed_waves = 0u,attention_bytes_total = 0u,expert_bytes_total = 0u;
	uint32_t sequence_index,fragments_per_sequence = (BSIM_CONTEXT_TOKENS / BSIM_FRAGMENT_TOKENS);
	if ( sequence_count * fragments_per_sequence > SPARK_GLM52_JIT_KV_POOL_MAX_FRAGMENTS )
		fragments_per_sequence = (SPARK_GLM52_JIT_KV_POOL_MAX_FRAGMENTS / sequence_count);
	uint32_t select_fragments = (BSIM_SELECT_TOKENS / BSIM_FRAGMENT_TOKENS);
	uint32_t require_ids[BSIM_SELECT_TOKENS / BSIM_FRAGMENT_TOKENS];
	if ( sequence_count == 0u || sequence_count > BSIM_MAX_SEQUENCES )
		return(1);
	memset(&queue_configuration,0,sizeof(queue_configuration));
	queue_configuration.abi_version = SPARK_GLM52_EXPERT_QUEUE_ABI_VERSION;
	queue_configuration.layer_count = BSIM_LAYERS;
	queue_configuration.expert_count = BSIM_EXPERTS;
	queue_configuration.firing_threshold_rows = fire_rows;
	queue_configuration.firing_deadline_ns = 50000000u;
	if ( SparkGlm52ExpertQueueInitialize(&bsim_queue,&queue_configuration) != SPARK_STATUS_OK )
		return(1);
	memset(&pool_configuration,0,sizeof(pool_configuration));
	pool_configuration.abi_version = SPARK_GLM52_JIT_KV_POOL_ABI_VERSION;
	pool_configuration.fragment_capacity = (sequence_count * fragments_per_sequence > SPARK_GLM52_JIT_KV_POOL_MAX_FRAGMENTS ? SPARK_GLM52_JIT_KV_POOL_MAX_FRAGMENTS : sequence_count * fragments_per_sequence);
	pool_configuration.dram_fragment_capacity = (uint32_t)(BSIM_KV_POOL_BYTES / (double)fragment_bytes);
	if ( pool_configuration.dram_fragment_capacity > pool_configuration.fragment_capacity )
		pool_configuration.dram_fragment_capacity = pool_configuration.fragment_capacity;
	pool_configuration.fragment_bytes = fragment_bytes;
	pool_configuration.nvme_bytes_per_second = (uint64_t)BSIM_NVME_BPS;
	if ( SparkGlm52JitKvPoolInitialize(&bsim_pool,&pool_configuration) != SPARK_STATUS_OK )
		return(1);
	memset(bsim_sequences,0,sizeof(bsim_sequences));
	for (sequence_index=0u; sequence_index<sequence_count; sequence_index++)
	{
		bsim_sequence_t *sequence = &bsim_sequences[sequence_index];
		uint32_t fragment_index,admitted_dram = 0u;
		sequence->fragment_base = (sequence_index * fragments_per_sequence);
		for (fragment_index=0u; fragment_index<fragments_per_sequence && sequence->fragment_base + fragment_index < pool_configuration.fragment_capacity; fragment_index++)
		{
			uint32_t want_dram = (fragment_index < select_fragments && bsim_pool.dram_resident_count < pool_configuration.dram_fragment_capacity) ? 1u : 0u;
			if ( SparkGlm52JitKvPoolAdmitFragment(&bsim_pool,sequence->fragment_base + fragment_index,sequence_index,fragment_index,want_dram != 0u ? SPARK_GLM52_JIT_KV_FRAGMENT_STATE_DRAM : SPARK_GLM52_JIT_KV_FRAGMENT_STATE_NVME) != SPARK_STATUS_OK )
				return(1);
			admitted_dram += want_dram;
		}
		sequence->wave_start_ns = now_ns;
		bsim_enqueue_layer(sequence_index,now_ns);
	}
	while (total_committed_waves < (uint64_t)sequence_count * wave_target)
	{
		SparkStatus status = SparkGlm52ExpertQueueNextFiring(&bsim_queue,now_ns,&firing);
		if ( status == SPARK_STATUS_NOT_FOUND )
		{
			now_ns += 1000000u;
			SparkGlm52JitKvPoolTick(&bsim_pool,now_ns);
			continue;
		}
		expert_bytes_total += (uint64_t)expert_bytes;
		now_ns += (uint64_t)(expert_bytes / bw * 1.0e9);
		for (uint32_t fired_index=0u; fired_index<firing.row_count; fired_index++)
		{
			uint32_t fired_sequence = (uint32_t)(firing.row_ids[fired_index] >> 8);
			bsim_sequence_t *sequence = &bsim_sequences[fired_sequence];
			sequence->pending_participations -= 1u;
			if ( sequence->pending_participations != 0u )
				continue;
			{
				double shared_bytes = ((double)BSIM_SELECT_TOKENS * BSIM_LATENT_BYTES_PER_TOKEN * shared_fraction);
				double lane_bytes = ((double)BSIM_SELECT_TOKENS * BSIM_LATENT_BYTES_PER_TOKEN * (1.0 - shared_fraction) * (double)BSIM_LANES);
				double index_bytes = ((double)BSIM_CONTEXT_TOKENS * BSIM_INDEX_BYTES_PER_TOKEN);
				double attention_bytes = (shared_bytes + lane_bytes + index_bytes);
				attention_bytes_total += (uint64_t)attention_bytes;
				now_ns += (uint64_t)(attention_bytes / bw * 1.0e9);
			}
			sequence->layer_index += 1u;
			if ( sequence->layer_index < BSIM_LAYERS )
			{
				bsim_enqueue_layer(fired_sequence,now_ns);
				continue;
			}
			sequence->layer_index = 0u;
			sequence->transit_accum_ns += ((now_ns - sequence->wave_start_ns) * BSIM_RANKS);
			sequence->transit_samples += 1u;
			sequence->wave_number += 1u;
			total_committed_waves += 1u;
			sequence->wave_start_ns = now_ns;
			{
				uint32_t require_index,rotate = (sequence->wave_number * BSIM_DELTA_FRAGMENTS);
				for (require_index=0u; require_index<select_fragments; require_index++)
					require_ids[require_index] = sequence->fragment_base + ((rotate + require_index) % fragments_per_sequence);
				SparkGlm52JitKvPoolRequireByEta(&bsim_pool,now_ns,require_ids,select_fragments,now_ns + (uint64_t)(now_ns > 0u ? 20000000000u : 20000000000u));
			}
			bsim_enqueue_layer(fired_sequence,now_ns);
		}
		SparkGlm52JitKvPoolTick(&bsim_pool,now_ns);
	}
	{
		double elapsed_s = ((double)now_ns / 1.0e9);
		double committed = ((double)total_committed_waves * BSIM_COMMIT_PER_WAVE);
		double transit_s = 0.0;
		uint64_t transit_total = 0u;
		uint32_t transit_count = 0u;
		for (sequence_index=0u; sequence_index<sequence_count; sequence_index++)
		{
			transit_total += bsim_sequences[sequence_index].transit_accum_ns;
			transit_count += bsim_sequences[sequence_index].transit_samples;
		}
		transit_s = (transit_count != 0u ? (double)transit_total / (double)transit_count / 1.0e9 : 0.0);
		printf("batchplane_sim seqs=%u waves=%" PRIu64 " commit_tok_per_s=%.0f per_req=%.3f transit_s=%.1f avg_rows_per_firing=%.1f expert_GBps=%.1f attn_GBps=%.1f jit_hit=%" PRIu64 " jit_miss=%" PRIu64 " jit_late=%" PRIu64 " stagein=%" PRIu64 "\n",
			sequence_count,total_committed_waves,committed / elapsed_s,committed / elapsed_s / ((double)sequence_count * BSIM_LANES),transit_s,
			(double)bsim_queue.fired_row_count / (double)(bsim_queue.firing_count != 0u ? bsim_queue.firing_count : 1u),
			(double)expert_bytes_total / elapsed_s / 1.0e9,(double)attention_bytes_total / elapsed_s / 1.0e9,
			bsim_pool.hit_count,bsim_pool.miss_count,bsim_pool.late_count,bsim_pool.stage_in_count);
	}
	return(0);
}
